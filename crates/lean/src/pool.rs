//! Copied verbatim from `t0-web/crates/t0-fast/src/pool.rs` (generic over
//! any model, no t0-specific content).
//!
//! Buffer + bind-group pool, keyed by a stable per-call-site string (e.g.
//! `"layer3.qkv"`). Grow-only: a buffer is only (re)created when a request
//! needs more bytes than the cached one already has, so a forward pass
//! repeated at a fixed `(v, p)` shape settles to zero new
//! `wgpu::Buffer`/`BindGroup` allocations after the first call. Uniform
//! buffers are updated in place via `queue.write_buffer` instead of being
//! recreated every call.
//!
//! Safety of buffer reuse *within* one forward call: every call site gets
//! its own key, so no two live dispatches in the same encoder ever share a
//! buffer. Safety *across* forward calls: this pool assumes forward calls
//! are sequential and each one's GPU work (including its final readback)
//! has completed before the next one starts -- true for this crate's own
//! `forecast`/`forecast_async` (readback is always awaited) and for
//! `t0-cli bench`'s call loop. A pipelined/concurrent caller would need a
//! different pool per in-flight call.
//!
//! Bind-group cache invalidation is coarse: any buffer (re)allocation bumps
//! a global generation counter, and every cached bind group is stamped
//! with the generation it was built against. This means the *first* call
//! at a new shape pays for rebuilding every bind group touched so far (not
//! just the ones whose buffers actually grew), but every call after that
//! at the same shape reuses all of them.
//!
//! **Bug found in lean's slice 2 (2026-09-27), fixed by `reset()` below:**
//! this generation scheme only knows about buffers the pool itself
//! allocated (via `data`/`upload_*`/`uniform`). lean's `KvCache` buffers are
//! allocated directly on the `Engine` (`engine.buf_empty`, not
//! `pool.data`), so a brand-new `KvCache` per independent generation call
//! (as `lean-cli`'s fixture harness does — one call per prompt, sharing one
//! `GpuModel`/`Pool`) does NOT bump the pool's generation. If no
//! pool-owned buffer happens to regrow between two such calls (e.g. a
//! shorter prompt run after a longer one, so no prefill buffer needs to
//! grow), decode's cached bind groups — which do bind directly to
//! `&cache.k[i]`/`&cache.v[i]` — stay "valid" by generation number while
//! silently still pointing at the *previous* call's now-dropped `KvCache`
//! buffers. Symptom: prefill and the very first decode token are correct,
//! then decode silently reads/writes the wrong KV cache and diverges hard
//! from the second decode token on. Caught by `lean-cli`'s three-case
//! fixture (short seq=36, long seq=86, non_english seq=54 — the *smaller*
//! seq after a *larger* one is exactly the no-regrowth case that hid it).
//! Fix: any caller starting an independent generation (a new `KvCache`)
//! must call `pool.reset()` first.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;


pub struct Pool {
    device: wgpu::Device,
    queue: wgpu::Queue,
    generation: Cell<u64>,
    buffers: RefCell<HashMap<String, wgpu::Buffer>>,
    /// Last bytes `uniform()` wrote for a given key, so a call site whose
    /// value is unchanged from the previous call (every `linear()` dims
    /// uniform at decode: same `m`/`k`/`n`/`blocks_per_row`/`n_offset` every
    /// step, since decode's shape never changes step to step - only
    /// position-dependent uniforms like RoPE's or attention's `kv_len` do)
    /// can skip the `queue.write_buffer` call entirely. Pure CPU-side
    /// per-step overhead reduction, no numerical change: the GPU-side bytes
    /// are identical either way.
    uniform_cache: RefCell<HashMap<String, Vec<u8>>>,
    bind_groups: RefCell<HashMap<String, (wgpu::BindGroup, u64)>>,
    alloc_count: Cell<u64>,
}

impl Pool {
    /// Drops every cached buffer and bind group and bumps the generation
    /// counter. Call this before starting an independent generation run
    /// (a new `KvCache`) against a `GpuModel` whose `Pool` may have served
    /// a previous, now-stale `KvCache` — see this module's bug note above.
    /// Cheap: the next forward call simply repays the "first call at a new
    /// shape" allocation cost this module's doc comment already describes.
    pub fn reset(&self) {
        self.buffers.borrow_mut().clear();
        self.uniform_cache.borrow_mut().clear();
        self.bind_groups.borrow_mut().clear();
        self.generation.set(self.generation.get() + 1);
    }
}

impl Pool {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Pool {
            device,
            queue,
            generation: Cell::new(0),
            buffers: RefCell::new(HashMap::new()),
            uniform_cache: RefCell::new(HashMap::new()),
            bind_groups: RefCell::new(HashMap::new()),
            alloc_count: Cell::new(0),
        }
    }

    pub fn reset_alloc_count(&self) {
        self.alloc_count.set(0);
    }

    pub fn alloc_count(&self) -> u64 {
        self.alloc_count.get()
    }

    /// Sum of every pooled buffer's current size -- the per-forward
    /// working set (activations, masks, RoPE tables, uniforms), not the
    /// model's persistent weights. Meaningful after at least one forward
    /// call (the pool starts empty and grows to its steady-state shape on
    /// the first call at a given `(v, p)`).
    pub fn resident_bytes(&self) -> u64 {
        self.buffers.borrow().values().map(|b| b.size()).sum()
    }

    /// Debug-only: the `n` largest pooled buffers by size, as
    /// `(key, bytes)`. Added for this session's memory investigation
    /// (docs/runs/2026-09-28-lean-decode-breakdown.md) - not used by any
    /// non-debug caller.
    pub fn debug_top_buffers(&self, n: usize) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self.buffers.borrow().iter().map(|(k, b)| (k.clone(), b.size())).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    fn bump_generation(&self) {
        self.generation.set(self.generation.get() + 1);
        self.alloc_count.set(self.alloc_count.get() + 1);
    }

    /// Get-or-grow a `STORAGE|COPY_SRC|COPY_DST` buffer for `key`, at least
    /// `len_f32` f32s. Content is untouched on reuse (kernels write it).
    pub fn data(&self, key: &str, len_f32: usize) -> wgpu::Buffer {
        let need = (len_f32.max(1) * 4) as u64;
        let mut bufs = self.buffers.borrow_mut();
        if let Some(b) = bufs.get(key) {
            if b.size() >= need {
                return b.clone();
            }
        }
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(key),
            size: need,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        bufs.insert(key.to_string(), b.clone());
        drop(bufs);
        self.bump_generation();
        b
    }

    /// Get-or-grow a `STORAGE` buffer for `key` and upload `data` into it
    /// via `queue.write_buffer` (safe to reuse across calls: see module
    /// doc's sequential-calls assumption).
    pub fn upload_f32(&self, key: &str, data: &[f32]) -> wgpu::Buffer {
        let b = self.data(key, data.len());
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    pub fn upload_u32(&self, key: &str, data: &[u32]) -> wgpu::Buffer {
        let need = (data.len().max(1) * 4) as u64;
        let mut bufs = self.buffers.borrow_mut();
        let reuse = matches!(bufs.get(key), Some(b) if b.size() >= need);
        let b = if reuse {
            bufs.get(key).unwrap().clone()
        } else {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(key),
                size: need,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            bufs.insert(key.to_string(), b.clone());
            drop(bufs);
            self.bump_generation();
            b
        };
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    /// Get-or-create a `UNIFORM` buffer for `key` sized to `T`, writing
    /// `value` into it every call (uniform buffers never need to grow --
    /// `T` is fixed per call site).
    pub fn uniform<T: bytemuck::Pod>(&self, key: &str, value: T) -> wgpu::Buffer {
        let bytes = bytemuck::bytes_of(&value);
        let mut bufs = self.buffers.borrow_mut();
        if let Some(b) = bufs.get(key) {
            let b = b.clone();
            drop(bufs);
            let mut cache = self.uniform_cache.borrow_mut();
            if let Some(prev) = cache.get(key) {
                debug_assert_eq!(
                    prev.len(),
                    bytes.len(),
                    "pool key {key:?} was previously written with a different byte length ({} vs {}) - \
                     one key must map to one T/shape for its whole lifetime, see this module's doc comment",
                    prev.len(),
                    bytes.len()
                );
            }
            let unchanged = matches!(cache.get(key), Some(prev) if prev.as_slice() == bytes);
            if !unchanged {
                self.queue.write_buffer(&b, 0, bytes);
                cache.insert(key.to_string(), bytes.to_vec());
            }
            return b;
        }
        // Plain create + `write_buffer`, never mapped-at-creation: see
        // `Engine::buf_upload`'s doc comment.
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(key),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b, 0, bytes);
        bufs.insert(key.to_string(), b.clone());
        drop(bufs);
        self.uniform_cache.borrow_mut().insert(key.to_string(), bytes.to_vec());
        self.bump_generation();
        b
    }

    /// Cached bind group for `key`: reused as long as the pool's
    /// generation hasn't advanced since it was built (see module doc).
    pub fn bind_group(&self, key: &str, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        let gen = self.generation.get();
        if let Some((bg, g)) = self.bind_groups.borrow().get(key) {
            if *g == gen {
                return bg.clone();
            }
        }
        let layout = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &layout,
            entries,
        });
        self.bind_groups.borrow_mut().insert(key.to_string(), (bg.clone(), gen));
        bg
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
