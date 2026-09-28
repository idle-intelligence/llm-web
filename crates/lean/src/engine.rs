//! Owns the wgpu device/queue and compute pipelines. Ported from
//! `t0-web/crates/t0-fast/src/engine.rs` almost verbatim (device init,
//! `make_pipeline`, `buf_*` helpers, `dispatch`, the single `read_buffer`
//! async readback): swapped in this crate's own kernel set (Qwen2 decoder
//! ops instead of t0's forecaster ops). Every pipeline uses an
//! auto-derived (`layout: None`) bind group layout, group 0, bindings in
//! declaration order matching each `.wgsl` file.

use std::borrow::Cow;
use std::cell::Cell;
use wgpu::util::DeviceExt;

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Total `dispatch()` calls since the last `reset_dispatch_count()` -
    /// same op-count metric on native and wasm (same Rust forward-pass
    /// code), used to separate "more work" from "slower per-op overhead"
    /// when comparing the two - see docs/runs/2026-09-28-lean-perf.md.
    dispatch_count: Cell<u64>,
    pub embed_gather_q4: wgpu::ComputePipeline,
    /// Q8_0 counterpart of `embed_gather_q4` (`shaders/embed_gather_q8.wgsl`)
    /// - Qwen3's official GGUFs ship no Q4_0 quant, only Q8_0 (qwen3 survey).
    pub embed_gather_q8: wgpu::ComputePipeline,
    /// Q6_K counterpart of `embed_gather_q4`/`embed_gather_q8`
    /// (`shaders/embed_gather_q6k.wgsl`) - Qwen2.5-3B-Instruct's official
    /// "q4_0" GGUF carries `token_embd.weight` at Q6_K residency (see
    /// `gguf.rs::GgmlDtype`'s doc comment).
    pub embed_gather_q6k: wgpu::ComputePipeline,
    pub rmsnorm: wgpu::ComputePipeline,
    pub rope: wgpu::ComputePipeline,
    pub linear: wgpu::ComputePipeline,
    pub linear_q4: wgpu::ComputePipeline,
    pub linear_q8: wgpu::ComputePipeline,
    /// Coalesced Q8_0 matvec (decode, M=1): same structure as
    /// `linear_q4_decode`, adapted to Q8_0's 32-int8 blocks (8 u32 words/
    /// block vs Q4_0's 4). Used whenever a decode-time linear's weight is
    /// Q8_0-resident - not just the lm head (Qwen3's official GGUFs ship
    /// every tensor as Q8_0, no Q4_0 quant at all - see qwen3 survey).
    pub linear_q8_decode: wgpu::ComputePipeline,
    /// Naive Q6_K matmul/matvec (`shaders/linear_q6k.wgsl`) - see that
    /// file's doc comment on why no tiled/decode-specialized variant exists
    /// yet.
    pub linear_q6k: wgpu::ComputePipeline,
    /// Tiled Q4_0 matmul (prefill, M>1): ported from llm-wasm's
    /// shader_q4_tiled.wgsl. Only faster than `linear_q4` once weight reuse
    /// across rows outweighs the tile/barrier overhead: see the shader's
    /// doc comment for llm-wasm's own measured regression at small M.
    pub linear_q4_tiled: wgpu::ComputePipeline,
    /// See `linear_q4_tiled_rb.wgsl`'s header: register-blocked 32x32/TK=16
    /// alternative to `linear_q4_tiled`, size-gated in `model.rs::linear`.
    pub linear_q4_tiled_rb: wgpu::ComputePipeline,
    /// See `linear_q8_tiled_rb.wgsl`'s header: same scheme for Q8_0.
    pub linear_q8_tiled_rb: wgpu::ComputePipeline,
    /// Coalesced Q4_0 matvec (decode, M=1): ported from llm-wasm's
    /// shader_q4_matvec_coalesced.wgsl. The fast decode kernel; it uses no
    /// cooperative-group extension, so it runs on any WebGPU adapter.
    pub linear_q4_decode: wgpu::ComputePipeline,
    pub attn_prefill: wgpu::ComputePipeline,
    /// `shaders/attn_decode.wgsl` compiled with its `HEAD_DIM` override
    /// constant set to 64 (Qwen2.5). One thread owns one output dim, so
    /// workgroup width must equal head_dim exactly - see the shader's doc
    /// comment. `model.rs::attn_decode` picks between this and
    /// `attn_decode_128` by `cfg.head_dim`.
    pub attn_decode: wgpu::ComputePipeline,
    /// Same shader as `attn_decode`, `HEAD_DIM` overridden to 128 (Qwen3).
    pub attn_decode_128: wgpu::ComputePipeline,
    /// Split-K ("flash-decoding") pass 1: `shaders/attn_decode_split.wgsl`,
    /// `HEAD_DIM` = 64. Used instead of `attn_decode` once `kv_len` is long
    /// enough that splitting one head's KV range across multiple workgroups
    /// is worth the second pass - see `model.rs::attn_decode`'s
    /// `num_splits` gate.
    pub attn_decode_split: wgpu::ComputePipeline,
    /// Same shader as `attn_decode_split`, `HEAD_DIM` overridden to 128.
    pub attn_decode_split_128: wgpu::ComputePipeline,
    /// Split-K pass 2: `shaders/attn_decode_reduce.wgsl`, `HEAD_DIM` = 64.
    pub attn_decode_reduce: wgpu::ComputePipeline,
    /// Same shader as `attn_decode_reduce`, `HEAD_DIM` overridden to 128.
    pub attn_decode_reduce_128: wgpu::ComputePipeline,
    pub add_inplace: wgpu::ComputePipeline,
    pub silu_mul: wgpu::ComputePipeline,
    /// Single-workgroup argmax over the logits vector -> one u32 output.
    /// Lets decode read back 4 bytes instead of the full `vocab_size * 4`
    /// bytes (~608KB for this model's 151936-vocab lm head) per step - see
    /// `model.rs::forward_decode_step`'s `argmax_readback` parameter.
    pub argmax: wgpu::ComputePipeline,
    /// Per-step constrained-decoding mask (`shaders/mask_logits.wgsl`):
    /// zeroes-out (sets to -inf) any vocab id whose bit is unset in a
    /// caller-supplied bitset, in place on the logits buffer, before
    /// argmax/readback - see `model.rs::mask_logits_gpu`.
    pub mask_logits: wgpu::ComputePipeline,
    /// Per-row absolute-position RoPE (`shaders/rope_positions.wgsl`) - the
    /// caller-supplied-position-ids mechanism (llm-life's per-cell RoPE
    /// restart), see `model.rs::rope_positions`.
    pub rope_positions: wgpu::ComputePipeline,
    /// Pluggable-mask GQA attention over a resident-prefix KV cache
    /// (`shaders/attn_chunk_masked.wgsl`) - see `model.rs::attn_chunk_masked`.
    pub attn_chunk_masked: wgpu::ComputePipeline,
    /// Gathers + dequantizes selected rows of a Q8_0 weight (the sliced
    /// lm-head mechanism, `shaders/gather_dequant_q8_rows.wgsl`) - see
    /// `model.rs::gather_dequant_head_rows`.
    pub gather_dequant_q8_rows: wgpu::ComputePipeline,
}

fn make_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    make_pipeline_with_constants(device, label, src, &[])
}

/// Like `make_pipeline`, but sets WGSL `override` constants
/// (`shaders/attn_decode.wgsl`'s `HEAD_DIM`) at pipeline-creation time via
/// `PipelineCompilationOptions`, so one shader source compiles to a
/// different fixed `@workgroup_size` per model.
fn make_pipeline_with_constants(device: &wgpu::Device, label: &str, src: &str, constants: &[(&str, f64)]) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
        cache: None,
    })
}

impl Engine {
    pub async fn new_async() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
        // wgpu::Limits::default() caps max_storage_buffer_binding_size at
        // 128MB: too small for this model's lm_head/embedding Q8_0/Q4_0
        // buffers in one binding (e.g. output.weight's Q8_0 qs buffer is
        // ~130MB). Request the adapter's own limits instead (native Metal
        // supports far more); a wasm build will need to cap this at
        // whatever WebGPU's downlevel limits actually allow and split
        // large tensors across bindings if it doesn't.
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lean"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu device: {e}"))?;

        Ok(Engine {
            embed_gather_q4: make_pipeline(&device, "embed_gather_q4", include_str!("shaders/embed_gather_q4.wgsl")),
            embed_gather_q8: make_pipeline(&device, "embed_gather_q8", include_str!("shaders/embed_gather_q8.wgsl")),
            embed_gather_q6k: make_pipeline(&device, "embed_gather_q6k", include_str!("shaders/embed_gather_q6k.wgsl")),
            rmsnorm: make_pipeline(&device, "rmsnorm", include_str!("shaders/rmsnorm.wgsl")),
            rope: make_pipeline(&device, "rope", include_str!("shaders/rope_neox.wgsl")),
            linear: make_pipeline(&device, "linear", include_str!("shaders/linear.wgsl")),
            linear_q4: make_pipeline(&device, "linear_q4", include_str!("shaders/linear_q4.wgsl")),
            linear_q8: make_pipeline(&device, "linear_q8", include_str!("shaders/linear_q8.wgsl")),
            linear_q8_decode: make_pipeline(&device, "linear_q8_decode", include_str!("shaders/linear_q8_decode.wgsl")),
            linear_q6k: make_pipeline(&device, "linear_q6k", include_str!("shaders/linear_q6k.wgsl")),
            linear_q4_tiled: make_pipeline(&device, "linear_q4_tiled", include_str!("shaders/linear_q4_tiled.wgsl")),
            linear_q4_tiled_rb: make_pipeline(&device, "linear_q4_tiled_rb", include_str!("shaders/linear_q4_tiled_rb.wgsl")),
            linear_q8_tiled_rb: make_pipeline(&device, "linear_q8_tiled_rb", include_str!("shaders/linear_q8_tiled_rb.wgsl")),
            linear_q4_decode: make_pipeline(&device, "linear_q4_decode", include_str!("shaders/linear_q4_decode.wgsl")),
            attn_prefill: make_pipeline(&device, "attn_prefill", include_str!("shaders/attn_prefill.wgsl")),
            attn_decode: make_pipeline_with_constants(&device, "attn_decode", include_str!("shaders/attn_decode.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_128: make_pipeline_with_constants(&device, "attn_decode_128", include_str!("shaders/attn_decode.wgsl"), &[("HEAD_DIM", 128.0)]),
            attn_decode_split: make_pipeline_with_constants(&device, "attn_decode_split", include_str!("shaders/attn_decode_split.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_split_128: make_pipeline_with_constants(&device, "attn_decode_split_128", include_str!("shaders/attn_decode_split.wgsl"), &[("HEAD_DIM", 128.0)]),
            attn_decode_reduce: make_pipeline_with_constants(&device, "attn_decode_reduce", include_str!("shaders/attn_decode_reduce.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_reduce_128: make_pipeline_with_constants(&device, "attn_decode_reduce_128", include_str!("shaders/attn_decode_reduce.wgsl"), &[("HEAD_DIM", 128.0)]),
            add_inplace: make_pipeline(&device, "add_inplace", include_str!("shaders/add_inplace.wgsl")),
            silu_mul: make_pipeline(&device, "silu_mul", include_str!("shaders/silu_mul.wgsl")),
            argmax: make_pipeline(&device, "argmax", include_str!("shaders/argmax.wgsl")),
            mask_logits: make_pipeline(&device, "mask_logits", include_str!("shaders/mask_logits.wgsl")),
            rope_positions: make_pipeline(&device, "rope_positions", include_str!("shaders/rope_positions.wgsl")),
            attn_chunk_masked: make_pipeline(&device, "attn_chunk_masked", include_str!("shaders/attn_chunk_masked.wgsl")),
            gather_dequant_q8_rows: make_pipeline(&device, "gather_dequant_q8_rows", include_str!("shaders/gather_dequant_q8_rows.wgsl")),
            device,
            queue,
            dispatch_count: Cell::new(0),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Browser adapters commonly cap a single storage-buffer binding well
    /// below what `max_buffer_size` allows (WebGPU's downlevel default is
    /// 128 MiB; native Metal reports far more). `quant.rs::load_matmul_weight_gguf`
    /// splits any weight whose per-row byte size times its row count would
    /// exceed this into row-aligned chunks bound separately, so a single
    /// tensor's residency is never assumed to fit in one binding.
    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size as u64
    }

    pub fn buf_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn buf_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn buf_empty(&self, len_f32: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len_f32.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn buf_uniform<T: bytemuck::Pod>(&self, data: T, label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn bind_group(&self, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries })
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, wgs: (u32, u32, u32), label: &str) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);
        self.dispatch_count.set(self.dispatch_count.get() + 1);
    }

    /// Submits `encoder`'s recorded work, replaces it with a fresh encoder
    /// of the same label, and (native only) blocks until that submission
    /// completes.
    ///
    /// Root cause, found by bisecting with a temporary
    /// `LEAN_DEBUG_MAX_LAYERS` cap on the model's layer loop: Qwen3-0.6B's
    /// 28-layer prefill, run as one encoder/submit per model.rs's
    /// `forward_prefill`/`decode_layers`/`forward_chunk_spec` (as every
    /// model's forward pass always was), reliably produced a native `device
    /// poll failed: Timeout` even on a 15-token prompt - but 24, 26 and 27
    /// real layers all finished correctly in ~1.5s, only the real 28th
    /// tipped it over: not a smooth compute-time trend, a threshold. It
    /// turned out not to be about command-buffer size or a bad kernel at
    /// all (the full 28-layer computation is numerically correct - it
    /// matches `reference/fixture_qwen3.json` exactly once it completes):
    /// splitting into one submit-per-layer with *no* intervening wait (just
    /// `queue.submit`, or even a non-blocking `PollType::Poll` maintain
    /// tick) hung exactly as badly. Only an actual blocking
    /// `device.poll(PollType::Wait)` between layers fixed it. Best
    /// explanation: every submission's bookkeeping (temp resources,
    /// buffer-map callbacks, retired-submission tracking) stays pending
    /// until something actually blocks on it, so without a real wait
    /// in between, ~500 dispatches' worth of unretired submissions still
    /// pile up for the final `read_buffer` call to wait on in one shot -
    /// and `device.poll(PollType::Wait)`'s wait has a hardcoded 60s
    /// timeout (`wgpu_core::device::CLEANUP_WAIT_MS`), which that pile-up
    /// reliably exceeded. Waiting once per layer keeps each wait's
    /// backlog bounded to one layer's dispatches regardless of model
    /// depth, at the cost of a CPU/GPU round-trip per layer (measured:
    /// `fixture_parity_qwen3`'s full 4-case suite, prefill and greedy
    /// decode included, at 106.97s - see docs/runs/2026-09-28-lean-qwen3.md;
    /// this is a native-only correctness fix, not a perf target - the
    /// production path is WASM/WebGPU).
    ///
    /// WASM-gated out entirely: the `wgpu/webgpu` backend used there talks
    /// to the browser's own WebGPU implementation directly, not
    /// wgpu-core's native Metal/Vulkan/DX12 path, so `CLEANUP_WAIT_MS`
    /// doesn't apply there, and this crate's WASM code must never call a
    /// blocking poll at all (see this crate's `read_buffer` doc comment on
    /// `into_data_async`-style async-only readback).
    pub fn flush_encoder(&self, encoder: &mut wgpu::CommandEncoder, label: &str) {
        let fresh = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let old = std::mem::replace(encoder, fresh);
        self.queue.submit(Some(old.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    pub fn reset_dispatch_count(&self) {
        self.dispatch_count.set(0);
    }

    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.get()
    }

    /// One copy-to-staging + map + read. NEVER call this crate's equivalent
    /// of `.into_data()` synchronously in WASM (deadlocks the browser) :
    /// this is the only readback path, always awaited.
    pub async fn read_buffer(&self, buf: &wgpu::Buffer, len_f32: usize) -> Vec<f32> {
        let size = (len_f32 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Same shape as `read_buffer` but for a single `u32` (the argmax
    /// kernel's output) - a 4-byte readback instead of a `vocab_size * 4`
    /// one for every decode step.
    pub async fn read_u32(&self, buf: &wgpu::Buffer) -> u32 {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging_u32"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback_u32") });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let result: u32 = bytemuck::cast_slice(&data)[0];
        drop(data);
        staging.unmap();
        result
    }
}
