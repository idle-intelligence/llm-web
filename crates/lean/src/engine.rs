//! Owns the wgpu device/queue and compute pipelines. Ported from
//! `t0-web/crates/t0-fast/src/engine.rs` almost verbatim (device init,
//! `make_pipeline`, `buf_*` helpers, `dispatch`, the single `read_buffer`
//! async readback): swapped in this crate's own kernel set (Qwen2 decoder
//! ops instead of t0's forecaster ops). Every pipeline uses an
//! auto-derived (`layout: None`) bind group layout, group 0, bindings in
//! declaration order matching each `.wgsl` file.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};

/// Opt-in per-dispatch GPU timing, default off. Set `LEAN_PROFILE_KERNELS=1`
/// (native only - wasm's `std::env::var` always errs, which is the desired
/// "never on in the browser" behavior) to request
/// `Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES` at
/// device creation; silently stays off if the adapter doesn't report both
/// (checked once here, never inferred from a device/vendor name). See
/// docs/runs/2026-09-29-lean-vs-llamacpp-profile.md
/// for the kernel-time table this path was built to produce. Zero cost when
/// disabled: `dispatch()`'s profiling branch is one `bool` check, and
/// `profile_labels` stays an empty `Vec`.
const PROFILE_ENV_VAR: &str = "LEAN_PROFILE_KERNELS";
/// Max recorded dispatches/step (2 timestamps each) before profiling starts
/// silently dropping further writes - comfortably above every model in this
/// project's fixture set (largest measured: 508 dispatches/step,
/// docs/runs/2026-09-28-lean-decode-breakdown.md).
const PROFILE_QUERY_CAPACITY: u32 = 4096;

/// Pass-level timestamp slots for the diagnostics path (two per compute
/// pass). WebGPU caps a query set at 4096 queries; a split prefill records
/// about 11 passes per layer, so this covers the deepest model here.
const DIAG_QUERY_CAPACITY: u32 = 4096;

#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Wall clock in milliseconds for the diagnostics timings: `performance.now()`
/// in the browser (window or worker), a process-relative `Instant` natively.
pub fn now_ms() -> f64 {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        performance_now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1e3
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "web")))]
    {
        0.0
    }
}

/// GPU time of one forward call from pass-level timestamps: `span_ms` is
/// first pass begin to last pass end (includes the encoder-level KV copies
/// and any gaps between passes), `pass_sum_ms` the sum of in-pass time, and
/// `segments` the in-pass time summed per pass label, in first-seen order,
/// with the number of passes carrying that label.
#[derive(Clone, Debug, Default)]
pub struct DiagGpu {
    pub span_ms: f64,
    pub pass_sum_ms: f64,
    pub segments: Vec<(String, f64, u32)>,
}

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Total `dispatch()` calls since the last `reset_dispatch_count()` -
    /// same op-count metric on native and wasm (same Rust forward-pass
    /// code), used to separate "more work" from "slower per-op overhead"
    /// when comparing the two - see docs/runs/2026-09-28-lean-perf.md.
    dispatch_count: Cell<u64>,
    /// `Some` only when `LEAN_PROFILE_KERNELS=1` and the adapter granted both
    /// timestamp features - see the module-level doc comment above.
    query_set: Option<wgpu::QuerySet>,
    timestamp_period: f32,
    /// One label per recorded dispatch, in order - index `i` owns query
    /// slots `2*i`/`2*i+1`. Cleared by `reset_profile()` at the start of
    /// each profiled step.
    profile_labels: RefCell<Vec<String>>,
    /// Diagnostics (off unless a caller turns them on - see `set_diag`):
    /// a pass-level timestamp query set, created only when the device was
    /// requested with `want_pass_timestamps` and the adapter has
    /// `TIMESTAMP_QUERY` (feature detection only).
    diag_query_set: Option<wgpu::QuerySet>,
    diag_timestamps: Cell<bool>,
    diag_split: Cell<bool>,
    diag_labels: RefCell<Vec<String>>,
    /// CPU-side split of the last forward call: recording + submit, then
    /// waiting for the readback.
    pub diag_encode_ms: Cell<f64>,
    pub diag_wait_ms: Cell<f64>,
    pub diag_last_gpu: RefCell<Option<DiagGpu>>,
    /// `(request_adapter + request_device ms, all pipeline creation calls ms)`.
    pub diag_init_ms: (f64, f64),
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
    /// Coalesced F32 matvec (decode, M=1): same 128-thread,
    /// rows-per-workgroup structure as `linear_q4_decode`/`linear_q8_decode`,
    /// adapted to a plain (undequantized) f32 weight row - see
    /// `shaders/linear_f32_decode.wgsl`'s header for why a F32-resident
    /// tensor shows up at all in an otherwise-quantized GGUF.
    pub linear_f32_decode: wgpu::ComputePipeline,
    pub linear_q4: wgpu::ComputePipeline,
    pub linear_q8: wgpu::ComputePipeline,
    /// Coalesced Q8_0 matvec (decode, M=1): same structure as
    /// `linear_q4_decode`, adapted to Q8_0's 32-int8 blocks (8 u32 words/
    /// block vs Q4_0's 4). Used whenever a decode-time linear's weight is
    /// Q8_0-resident - not just the lm head (Qwen3's official GGUFs ship
    /// every tensor as Q8_0, no Q4_0 quant at all - see qwen3 survey).
    pub linear_q8_decode: wgpu::ComputePipeline,
    /// Naive Q6_K matmul/matvec (`shaders/linear_q6k.wgsl`) - used for
    /// prefill (M>1); decode (M=1) uses `linear_q6k_decode` below.
    pub linear_q6k: wgpu::ComputePipeline,
    /// Coalesced Q6_K matvec (decode, M=1): same 128-thread,
    /// rows-per-workgroup structure as `linear_q4_decode`/`linear_q8_decode`,
    /// see `shaders/linear_q6k_decode.wgsl`'s header for the per-block split.
    pub linear_q6k_decode: wgpu::ComputePipeline,
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
    /// `shaders/add_rmsnorm.wgsl`: fuses a residual `add_inplace` with the
    /// rmsnorm that always immediately follows it in the decoder layer's
    /// decode path (see that shader's header). Used only by decode's
    /// `add1`/`add2` sites in `model.rs` (prefill's own add+norm pairs are
    /// unchanged, already amortized across many rows per dispatch).
    pub add_rmsnorm: wgpu::ComputePipeline,
    /// `shaders/silu_mul_fused.wgsl`: SwiGLU over one `[rows, 2*hidden]`
    /// fused gate/up matmul output (see `model.rs::gguf_matmul_concat2`'s
    /// doc comment) - replaces a two-buffer `silu_mul` now that gate/up
    /// share one matmul dispatch.
    pub silu_mul_fused: wgpu::ComputePipeline,
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
        Self::new_async_with(false).await
    }

    /// `want_pass_timestamps` requests `Features::TIMESTAMP_QUERY` when the
    /// adapter reports it, for the diagnostics path's pass-level GPU timing.
    /// Every other device setting is the same as `new_async`.
    pub async fn new_async_with(want_pass_timestamps: bool) -> anyhow::Result<Self> {
        let t_start = now_ms();
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
        let want_profiling = std::env::var(PROFILE_ENV_VAR).as_deref() == Ok("1");
        let profile_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let grant_profiling = want_profiling && adapter.features().contains(profile_features);
        let mut required_features = if grant_profiling { profile_features } else { wgpu::Features::empty() };
        let grant_pass_timestamps = want_pass_timestamps && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if grant_pass_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lean"),
                required_features,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu device: {e}"))?;

        let query_set = grant_profiling.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("lean_profile_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: PROFILE_QUERY_CAPACITY,
            })
        });
        let diag_query_set = grant_pass_timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("lean_diag_pass_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: DIAG_QUERY_CAPACITY,
            })
        });
        let timestamp_period = queue.get_timestamp_period();
        let t_device = now_ms();

        let mut engine = Engine {
            query_set,
            timestamp_period,
            profile_labels: RefCell::new(Vec::new()),
            diag_query_set,
            diag_timestamps: Cell::new(false),
            diag_split: Cell::new(false),
            diag_labels: RefCell::new(Vec::new()),
            diag_encode_ms: Cell::new(0.0),
            diag_wait_ms: Cell::new(0.0),
            diag_last_gpu: RefCell::new(None),
            diag_init_ms: (0.0, 0.0),
            embed_gather_q4: make_pipeline(&device, "embed_gather_q4", include_str!("shaders/embed_gather_q4.wgsl")),
            embed_gather_q8: make_pipeline(&device, "embed_gather_q8", include_str!("shaders/embed_gather_q8.wgsl")),
            embed_gather_q6k: make_pipeline(&device, "embed_gather_q6k", include_str!("shaders/embed_gather_q6k.wgsl")),
            rmsnorm: make_pipeline(&device, "rmsnorm", include_str!("shaders/rmsnorm.wgsl")),
            rope: make_pipeline(&device, "rope", include_str!("shaders/rope_neox.wgsl")),
            linear: make_pipeline(&device, "linear", include_str!("shaders/linear.wgsl")),
            linear_f32_decode: make_pipeline(&device, "linear_f32_decode", include_str!("shaders/linear_f32_decode.wgsl")),
            linear_q4: make_pipeline(&device, "linear_q4", include_str!("shaders/linear_q4.wgsl")),
            linear_q8: make_pipeline(&device, "linear_q8", include_str!("shaders/linear_q8.wgsl")),
            linear_q8_decode: make_pipeline(&device, "linear_q8_decode", include_str!("shaders/linear_q8_decode.wgsl")),
            linear_q6k: make_pipeline(&device, "linear_q6k", include_str!("shaders/linear_q6k.wgsl")),
            linear_q6k_decode: make_pipeline(&device, "linear_q6k_decode", include_str!("shaders/linear_q6k_decode.wgsl")),
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
            add_rmsnorm: make_pipeline(&device, "add_rmsnorm", include_str!("shaders/add_rmsnorm.wgsl")),
            silu_mul_fused: make_pipeline(&device, "silu_mul_fused", include_str!("shaders/silu_mul_fused.wgsl")),
            argmax: make_pipeline(&device, "argmax", include_str!("shaders/argmax.wgsl")),
            mask_logits: make_pipeline(&device, "mask_logits", include_str!("shaders/mask_logits.wgsl")),
            rope_positions: make_pipeline(&device, "rope_positions", include_str!("shaders/rope_positions.wgsl")),
            attn_chunk_masked: make_pipeline(&device, "attn_chunk_masked", include_str!("shaders/attn_chunk_masked.wgsl")),
            gather_dequant_q8_rows: make_pipeline(&device, "gather_dequant_q8_rows", include_str!("shaders/gather_dequant_q8_rows.wgsl")),
            device,
            queue,
            dispatch_count: Cell::new(0),
        };
        engine.diag_init_ms = (t_device - t_start, now_ms() - t_device);
        Ok(engine)
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
        self.buf_upload(label, bytemuck::cast_slice(data), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC)
    }

    pub fn buf_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        self.buf_upload(label, bytemuck::cast_slice(data), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)
    }

    /// Creates a buffer and fills it with `bytes` through `queue.write_buffer`.
    /// Every upload in this crate goes through here (weights, biases, rope
    /// tables, uniforms), never through `create_buffer_init`'s
    /// mapped-at-creation path: in Chromium on an RTX 3080 (Vulkan), a few
    /// of the ~560 small weight buffers of Qwen2.5-0.5B created that way per
    /// load held different bytes on the device than the ones written into
    /// the mapping (different buffers on every page load, found by reading
    /// every buffer back after load: docs/runs/2026-10-02-webgpu-nondeterminism.md),
    /// which made greedy output vary from load to load. `write_buffer` is the
    /// plain WebGPU upload every backend supports, so this is the path on
    /// every device, not a per-device choice.
    pub fn buf_upload(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(4), usage, mapped_at_creation: false });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buf, 0, bytes);
        }
        buf
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
        self.buf_upload(label, bytemuck::bytes_of(&data), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
    }

    pub fn bind_group(&self, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries })
    }

    /// Records one dispatch into an already-open `pass`. Callers batch many
    /// dispatches into one `wgpu::ComputePass` (see model.rs's orchestrator
    /// functions) rather than opening a pass per dispatch: on native Metal,
    /// ending and beginning a `MTLComputeCommandEncoder` per dispatch is a
    /// real, measured cost (profiled with `sample` on a 3B-model decode
    /// step: ~69% of wall time inside `wgpu_core`'s device-wait path at 532
    /// single-dispatch passes/step - docs/runs/2026-09-28-lean-decode-breakdown.md).
    /// A `label` is still accepted for call-site readability but no longer
    /// used to name a pass (the caller's `begin_compute_pass` call names the
    /// batch instead).
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, wgs: (u32, u32, u32), _label: &str) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if let Some(qs) = &self.query_set {
            let mut labels = self.profile_labels.borrow_mut();
            let i = labels.len() as u32;
            if i * 2 + 1 < PROFILE_QUERY_CAPACITY {
                pass.write_timestamp(qs, i * 2);
                pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);
                pass.write_timestamp(qs, i * 2 + 1);
                labels.push(_label.to_string());
                self.dispatch_count.set(self.dispatch_count.get() + 1);
                return;
            }
        }
        pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);
        self.dispatch_count.set(self.dispatch_count.get() + 1);
    }

    pub fn profiling_enabled(&self) -> bool {
        self.query_set.is_some()
    }

    /// Clears the previous step's labels. Call once per profiled step,
    /// before its dispatches.
    pub fn reset_profile(&self) {
        self.profile_labels.borrow_mut().clear();
    }

    /// Resolves this step's recorded timestamp pairs into a fresh buffer -
    /// must be called on the same `encoder` the profiled dispatches were
    /// recorded into, before that encoder is finished/submitted (query
    /// resolution is an encoder-timeline command). Returns `None` when
    /// profiling is off or nothing was recorded this step.
    pub fn resolve_profile(&self, encoder: &mut wgpu::CommandEncoder) -> Option<(wgpu::Buffer, u32)> {
        let qs = self.query_set.as_ref()?;
        let count = self.profile_labels.borrow().len() as u32;
        if count == 0 {
            return None;
        }
        let resolve_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile_resolve"),
            size: u64::from(count) * 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(qs, 0..count * 2, &resolve_buf, 0);
        Some((resolve_buf, count))
    }

    /// Reads back `resolve_profile`'s buffer and returns `(label, gpu_ns)`
    /// per dispatch, in dispatch order - the caller aggregates by label (see
    /// `lean_cli.rs`'s `print_profile_table`). Async readback only, per this
    /// crate's own rule (see `read_buffer`'s doc comment); the native
    /// blocking `device.poll` below matches every other native-only readback
    /// path in this file.
    pub async fn read_profile(&self, resolve_buf: &wgpu::Buffer, count: u32) -> Vec<(String, f64)> {
        let size = u64::from(count) * 2 * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("profile_readback") });
        encoder.copy_buffer_to_buffer(resolve_buf, 0, &staging, 0, size);
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
        let raw: &[u64] = bytemuck::cast_slice(&data);
        let labels = self.profile_labels.borrow();
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let ticks = raw[i * 2 + 1].saturating_sub(raw[i * 2]);
            let ns = ticks as f64 * f64::from(self.timestamp_period);
            out.push((labels[i].clone(), ns));
        }
        drop(data);
        staging.unmap();
        out
    }

    pub fn begin_pass<'e>(&self, encoder: &'e mut wgpu::CommandEncoder, label: &str) -> wgpu::ComputePass<'e> {
        let timestamp_writes = self.diag_pass_writes(label);
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes })
    }

    /// Diagnostics switches. `timestamps` records a begin/end timestamp on
    /// every compute pass (no-op without `TIMESTAMP_QUERY`); `split` makes
    /// the forward pass close and reopen its compute pass at each op group
    /// so each group gets its own timestamps (see `model.rs`'s `seg!`).
    /// Both off by default: the forward pass's command stream is then the
    /// same as without diagnostics.
    pub fn set_diag(&self, timestamps: bool, split: bool) {
        self.diag_timestamps.set(timestamps && self.diag_query_set.is_some());
        self.diag_split.set(split);
    }

    pub fn diag_split(&self) -> bool {
        self.diag_split.get()
    }

    pub fn has_pass_timestamps(&self) -> bool {
        self.diag_query_set.is_some()
    }

    fn diag_pass_writes(&self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.diag_timestamps.get() {
            return None;
        }
        let qs = self.diag_query_set.as_ref()?;
        let mut labels = self.diag_labels.borrow_mut();
        let i = labels.len() as u32;
        if i * 2 + 1 >= DIAG_QUERY_CAPACITY {
            return None;
        }
        labels.push(label.to_string());
        Some(wgpu::ComputePassTimestampWrites { query_set: qs, beginning_of_pass_write_index: Some(i * 2), end_of_pass_write_index: Some(i * 2 + 1) })
    }

    /// Start of a forward call: clears the previous call's pass labels.
    pub fn diag_begin(&self) {
        self.diag_labels.borrow_mut().clear();
    }

    /// Resolves every pass timestamp recorded since `diag_begin` (including
    /// ones in encoders already submitted - query slots persist) into a
    /// buffer, on the forward call's last encoder before it is finished.
    pub fn diag_resolve(&self, encoder: &mut wgpu::CommandEncoder) -> Option<(wgpu::Buffer, u32)> {
        if !self.diag_timestamps.get() {
            return None;
        }
        let qs = self.diag_query_set.as_ref()?;
        let count = self.diag_labels.borrow().len() as u32;
        if count == 0 {
            return None;
        }
        let size = u64::from(count) * 2 * 8;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diag_resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(qs, 0..count * 2, &resolve, 0);
        Some((resolve, count))
    }

    /// Reads `diag_resolve`'s buffer back (async) and stores the per-label
    /// GPU times in `diag_last_gpu`.
    pub async fn diag_collect(&self, resolve: &wgpu::Buffer, count: u32) {
        let size = u64::from(count) * 2 * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diag_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diag_readback") });
        encoder.copy_buffer_to_buffer(resolve, 0, &staging, 0, size);
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
        let raw: &[u64] = bytemuck::cast_slice(&data);
        let period = f64::from(self.timestamp_period);
        let labels = self.diag_labels.borrow();
        let mut out = DiagGpu::default();
        let (mut first, mut last) = (u64::MAX, 0u64);
        for i in 0..count as usize {
            let (b, e) = (raw[i * 2], raw[i * 2 + 1]);
            first = first.min(b);
            last = last.max(e);
            let ms = e.saturating_sub(b) as f64 * period / 1e6;
            out.pass_sum_ms += ms;
            match out.segments.iter_mut().find(|s| s.0 == labels[i]) {
                Some(s) => {
                    s.1 += ms;
                    s.2 += 1;
                }
                None => out.segments.push((labels[i].clone(), ms, 1)),
            }
        }
        out.span_ms = last.saturating_sub(first) as f64 * period / 1e6;
        drop(data);
        staging.unmap();
        *self.diag_last_gpu.borrow_mut() = Some(out);
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
