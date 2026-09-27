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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use wgpu::util::DeviceExt;

pub struct Pool {
    device: wgpu::Device,
    queue: wgpu::Queue,
    generation: Cell<u64>,
    buffers: RefCell<HashMap<String, wgpu::Buffer>>,
    bind_groups: RefCell<HashMap<String, (wgpu::BindGroup, u64)>>,
    alloc_count: Cell<u64>,
}

impl Pool {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Pool {
            device,
            queue,
            generation: Cell::new(0),
            buffers: RefCell::new(HashMap::new()),
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
        let mut bufs = self.buffers.borrow_mut();
        if let Some(b) = bufs.get(key) {
            let b = b.clone();
            drop(bufs);
            self.queue.write_buffer(&b, 0, bytemuck::bytes_of(&value));
            return b;
        }
        let b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(key),
            contents: bytemuck::bytes_of(&value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        bufs.insert(key.to_string(), b.clone());
        drop(bufs);
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
