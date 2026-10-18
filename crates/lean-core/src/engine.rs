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

/// Largest single `write_buffer` and the pending bytes that trigger a flush
/// submit; see `Engine::buf_upload`.
const UPLOAD_PIECE: usize = 8 << 20;

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
/// with the number of passes carrying that label. A pass whose begin or
/// end slot reads back as 0 (or ends before it begins) was never written
/// (an empty pass can be skipped by the browser's timestamp writes): it is
/// counted in `unwritten` and left out of every sum; a segment with no
/// written pass, and the span when no slot was written, are `None`.
#[derive(Clone, Debug, Default)]
pub struct DiagGpu {
    pub span_ms: Option<f64>,
    pub pass_sum_ms: f64,
    pub segments: Vec<(String, Option<f64>, u32)>,
    pub unwritten: u32,
}

pub struct Engine {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
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
    /// Debug-only op taps (off unless `set_debug_taps(true)`): every
    /// `model.rs` `tap!` site copies its op's output into a fresh buffer
    /// here, so `debug_collect` can checksum each op of one forward call
    /// and two runs can be compared op by op. `None` costs one borrow per
    /// tap site and changes nothing in the command stream.
    debug_taps: RefCell<Option<Vec<(String, wgpu::Buffer, u64)>>>,
    /// Debug-only upload check (off unless `set_debug_uploads(true)`):
    /// every `buf_f32`/`buf_u32` upload is recorded with the FNV-1a hash of
    /// the bytes handed to the GPU, so `debug_verify_uploads` can read each
    /// buffer back and say whether the device holds what was uploaded.
    debug_uploads: RefCell<Option<Vec<(String, wgpu::Buffer, u64)>>>,
    /// Bytes handed to `queue.write_buffer` since the last submit (see `buf_upload`).
    pending_upload_bytes: Cell<usize>,
    /// Adapter name/backend, features and the limits kernel selection or
    /// buffer chunking depends on, as seen at device creation (JSON).
    pub adapter_report: String,
    pub(crate) embed_gather_q4: wgpu::ComputePipeline,
    /// Q8_0 counterpart of `embed_gather_q4` (`shaders/embed_gather_q8.wgsl`)
    /// - Qwen3's official GGUFs ship no Q4_0 quant, only Q8_0 (qwen3 survey).
    pub(crate) embed_gather_q8: wgpu::ComputePipeline,
    /// Q6_K counterpart of `embed_gather_q4`/`embed_gather_q8`
    /// (`shaders/embed_gather_q6k.wgsl`) - Qwen2.5-3B-Instruct's official
    /// "q4_0" GGUF carries `token_embd.weight` at Q6_K residency (see
    /// `gguf.rs::GgmlDtype`'s doc comment).
    pub(crate) embed_gather_q6k: wgpu::ComputePipeline,
    pub(crate) rmsnorm: wgpu::ComputePipeline,
    pub(crate) rope: wgpu::ComputePipeline,
    pub(crate) linear: wgpu::ComputePipeline,
    /// Coalesced F32 matvec (decode, M=1): same 128-thread,
    /// vec4 loads and word-wide unpacking,
    /// adapted to a plain (undequantized) f32 weight row - see
    /// `shaders/linear_f32_decode.wgsl`'s header for why a F32-resident
    /// tensor shows up at all in an otherwise-quantized GGUF.
    pub(crate) linear_f32_decode: wgpu::ComputePipeline,
    pub(crate) linear_q4: wgpu::ComputePipeline,
    pub(crate) linear_q8: wgpu::ComputePipeline,
    /// Coalesced Q8_0 matvec (decode, M=1): same structure as
    /// `linear_q4_decode`, adapted to Q8_0's 32-int8 blocks (8 u32 words/
    /// block vs Q4_0's 4). Used whenever a decode-time linear's weight is
    /// Q8_0-resident - not just the lm head (Qwen3's official GGUFs ship
    /// every tensor as Q8_0, no Q4_0 quant at all - see qwen3 survey).
    pub(crate) linear_q8_decode: wgpu::ComputePipeline,
    /// Naive Q6_K matmul/matvec (`shaders/linear_q6k.wgsl`) - used for
    /// prefill (M>1); decode (M=1) uses `linear_q6k_decode` below.
    pub(crate) linear_q6k: wgpu::ComputePipeline,
    /// Q6_K matvec (decode, M=1): 128 threads, 16 rows per workgroup, vec4
    /// loads and word-wide unpacking; see `shaders/linear_q6k_decode.wgsl`'s
    /// header.
    pub(crate) linear_q6k_decode: wgpu::ComputePipeline,
    /// See `linear_q4_tiled_rb.wgsl`'s header: 64x64 register-blocked tiled
    /// Q4_0 matmul, prefill at `SMALL_M_MAX_ROWS` rows and above.
    pub(crate) linear_q4_tiled_rb: wgpu::ComputePipeline,
    /// `linear_f32_tiled.wgsl`: the same tile over F32 weights (prefill).
    pub(crate) linear_f32_tiled: wgpu::ComputePipeline,
    /// `linear_q4_tiled_rb.wgsl` with its `Q8` override: Q8_0 prefill.
    pub(crate) linear_q8_tiled_rb: wgpu::ComputePipeline,
    /// `linear_q4_tiled_f16.wgsl` (and its `Q8` override): the prefill tile
    /// with f16 shared tiles and f16 partial sums. `Some` only when the
    /// device was granted `shader-f16`, which is requested whenever the
    /// adapter offers it; the F32 pipelines above are used otherwise.
    pub(crate) linear_q4_tiled_f16: Option<wgpu::ComputePipeline>,
    pub(crate) linear_q8_tiled_f16: Option<wgpu::ComputePipeline>,
    /// Coalesced Q4_0 matvec (decode, M=1): ported from llm-wasm's
    /// shader_q4_matvec_coalesced.wgsl. The fast decode kernel; it uses no
    /// cooperative-group extension, so it runs on any WebGPU adapter.
    pub(crate) linear_q4_decode: wgpu::ComputePipeline,
    /// `linear_q4_decode.wgsl` with `GATE_UP`: the fused gate/up matvec
    /// with silu(gate) * up in its epilogue (decode MLP).
    pub(crate) linear_q4_decode_swiglu: wgpu::ComputePipeline,
    /// 4-rows-per-lane decode matvecs for wide outputs
    /// (`linear_q4_decode_r4.wgsl`, `linear_q8_decode_r4.wgsl`; see
    /// `DECODE_R4_MIN_ROWS` in model.rs), and their `GATE_UP` variants.
    pub(crate) linear_q4_decode_r4: wgpu::ComputePipeline,
    pub(crate) linear_q8_decode_r4: wgpu::ComputePipeline,
    pub(crate) linear_q4_decode_swiglu_r4: wgpu::ComputePipeline,
    pub(crate) linear_q8_decode_swiglu_r4: wgpu::ComputePipeline,
    /// Small-M Q4_0 matmul (short prefill): see `linear_q4_small_m.wgsl`'s
    /// header and `SMALL_M_MAX_ROWS` in model.rs.
    pub(crate) linear_q4_small_m: wgpu::ComputePipeline,
    /// `linear_q4_small_m.wgsl` with its `Q8` override: Q8_0 short prefill.
    pub(crate) linear_q8_small_m: wgpu::ComputePipeline,
    /// Q4_1 builds (`Q41` override) of the Q4_0 matmul shaders: naive,
    /// decode (one row per lane and four rows per lane), short and long prefill.
    pub(crate) linear_q41: wgpu::ComputePipeline,
    pub(crate) linear_q41_decode: wgpu::ComputePipeline,
    pub(crate) linear_q41_decode_r4: wgpu::ComputePipeline,
    pub(crate) linear_q41_small_m: wgpu::ComputePipeline,
    pub(crate) linear_q41_tiled_rb: wgpu::ComputePipeline,
    /// Q5_0 and Q4_K builds of the Q4_0-family kernels, compiled on first use.
    pub(crate) q50: Q4Family,
    pub(crate) k4: Q4Family,
    /// F16 and BF16 weights kept at two bytes per value: naive, decode
    /// matvec and the prefill tile (`linear_half*.wgsl`).
    pub(crate) half_f16: HalfKernels,
    pub(crate) half_bf16: HalfKernels,
    /// `linear_q6k_tiled.wgsl`: Q6_K prefill tile.
    pub(crate) linear_q6k_tiled: LazyPipeline,
    pub(crate) embed_gather_q5: LazyPipeline,
    pub(crate) embed_gather_q4k: LazyPipeline,
    pub(crate) embed_gather_f16: LazyPipeline,
    pub(crate) embed_gather_bf16: LazyPipeline,
    pub(crate) attn_prefill: wgpu::ComputePipeline,
    /// `attn_prefill_tiled.wgsl` (head_dim 64) and its HEAD_DIM = 128 build.
    pub(crate) attn_prefill_tiled: wgpu::ComputePipeline,
    pub(crate) attn_prefill_tiled_128: wgpu::ComputePipeline,
    /// The same kernels reading K/V from the cache (suffix prefill).
    pub(crate) attn_prefill_cache: wgpu::ComputePipeline,
    pub(crate) attn_prefill_cache_128: wgpu::ComputePipeline,
    /// `shaders/attn_decode.wgsl` compiled with its `HEAD_DIM` override
    /// constant set to 64 (Qwen2.5). One thread owns one output dim, so
    /// workgroup width must equal head_dim exactly - see the shader's doc
    /// comment. `model.rs::attn_decode` picks between this and
    /// `attn_decode_128` by `cfg.head_dim`.
    pub(crate) attn_decode: wgpu::ComputePipeline,
    /// Same shader as `attn_decode`, `HEAD_DIM` overridden to 128 (Qwen3).
    pub(crate) attn_decode_128: wgpu::ComputePipeline,
    /// Split-K ("flash-decoding") pass 1: `shaders/attn_decode_split.wgsl`,
    /// `HEAD_DIM` = 64. Used instead of `attn_decode` once `kv_len` is long
    /// enough that splitting one head's KV range across multiple workgroups
    /// is worth the second pass - see `model.rs::attn_decode`'s
    /// `num_splits` gate.
    pub(crate) attn_decode_split: wgpu::ComputePipeline,
    /// Same shader as `attn_decode_split`, `HEAD_DIM` overridden to 128.
    pub(crate) attn_decode_split_128: wgpu::ComputePipeline,
    /// Split-K pass 2: `shaders/attn_decode_reduce.wgsl`, `HEAD_DIM` = 64.
    pub(crate) attn_decode_reduce: wgpu::ComputePipeline,
    /// Same shader as `attn_decode_reduce`, `HEAD_DIM` overridden to 128.
    pub(crate) attn_decode_reduce_128: wgpu::ComputePipeline,
    pub(crate) add_inplace: wgpu::ComputePipeline,
    pub(crate) kv_scatter: wgpu::ComputePipeline,
    /// Decode's RoPE + K/V cache write in one dispatch (`rope_kv_decode.wgsl`).
    pub(crate) rope_kv_decode: wgpu::ComputePipeline,
    /// `shaders/add_rmsnorm.wgsl`: fuses a residual `add_inplace` with the
    /// rmsnorm that always immediately follows it in the decoder layer's
    /// decode path (see that shader's header). Used only by decode's
    /// `add1`/`add2` sites in `model.rs` (prefill's own add+norm pairs are
    /// unchanged, already amortized across many rows per dispatch).
    pub(crate) add_rmsnorm: wgpu::ComputePipeline,
    /// `shaders/silu_mul_fused.wgsl`: SwiGLU over one `[rows, 2*hidden]`
    /// fused gate/up matmul output (see `model.rs::gguf_matmul_concat2`'s
    /// doc comment) - replaces a two-buffer `silu_mul` now that gate/up
    /// share one matmul dispatch.
    pub(crate) silu_mul_fused: wgpu::ComputePipeline,
    /// Single-workgroup argmax over the logits vector -> one u32 output.
    /// Lets decode read back 4 bytes instead of the full `vocab_size * 4`
    /// bytes (~608KB for this model's 151936-vocab lm head) per step - see
    /// `model.rs::forward_decode_step`'s `argmax_readback` parameter.
    pub(crate) argmax: wgpu::ComputePipeline,
    /// Per-step constrained-decoding mask (`shaders/mask_logits.wgsl`):
    /// zeroes-out (sets to -inf) any vocab id whose bit is unset in a
    /// caller-supplied bitset, in place on the logits buffer, before
    /// argmax/readback - see `model.rs::mask_logits_gpu`.
    pub(crate) mask_logits: wgpu::ComputePipeline,
    /// GPU top-k for sampling: one stage of the 1024-way bitonic selection
    /// (`shaders/topk_stage.wgsl`) - see `model.rs::topk_gpu`.
    pub(crate) topk_stage: wgpu::ComputePipeline,
    /// GPU-resident sampling (`resident_sampling.rs`): the repetition
    /// penalty over the history counts, and the draw from the top-k.
    pub(crate) sample_penalty: wgpu::ComputePipeline,
    pub(crate) sample_final: wgpu::ComputePipeline,
    /// Per-row absolute-position RoPE (`shaders/rope_positions.wgsl`) - the
    /// caller-supplied-position-ids mechanism (llm-life's per-cell RoPE
    /// restart), see `model.rs::rope_positions`.
    pub(crate) rope_positions: wgpu::ComputePipeline,
    /// Pluggable-mask GQA attention over a resident-prefix KV cache
    /// (`shaders/attn_chunk_masked.wgsl`) - see `model.rs::attn_chunk_masked`.
    pub(crate) attn_chunk_masked: wgpu::ComputePipeline,
    /// Gathers + dequantizes selected rows of a Q8_0 weight (the sliced
    /// lm-head mechanism, `shaders/gather_dequant_q8_rows.wgsl`) - see
    /// `model.rs::gather_dequant_head_rows`.
    pub(crate) gather_dequant_q8_rows: wgpu::ComputePipeline,
}

/// Set to `0` to leave `shader-f16` ungranted even when the adapter has it.
const F16_ENV_VAR: &str = "LEAN_F16";

/// Where the Q4_0-family shaders (`linear_q4*.wgsl`) take their block-metadata
/// declarations: `compose_aux` replaces this comment with `aux_with.wgsl`
/// (a seventh binding, `aux`, for the Q5_0 and Q4_K builds) or `aux_none.wgsl`
/// (no binding, so the Q4_0, Q4_1 and Q8_0 builds keep the six-binding layout
/// they always had), then `aux_common.wgsl`.
const AUX_MARKER: &str = "//@AUX";
/// Where a shader that declares its own `aux` and `scales` bindings
/// (`embed_gather_q4k.wgsl`) takes the shared block-metadata helpers.
const K4_DM_MARKER: &str = "//@K4_DM";

fn compose_aux(src: &str, with_binding: bool) -> String {
    let decl = if with_binding { include_str!("shaders/aux_with.wgsl") } else { include_str!("shaders/aux_none.wgsl") };
    src.replace(AUX_MARKER, &format!("{decl}\n{}", include_str!("shaders/aux_common.wgsl")))
}

/// A pipeline compiled on first use: formats a model does not contain never
/// pay for their shaders at engine start.
pub(crate) struct LazyPipeline {
    cell: std::cell::OnceCell<wgpu::ComputePipeline>,
    label: &'static str,
    src: &'static str,
    constants: &'static [(&'static str, f64)],
    with_aux: bool,
}

impl LazyPipeline {
    pub(crate) const fn new(label: &'static str, src: &'static str, constants: &'static [(&'static str, f64)], with_aux: bool) -> Self {
        Self { cell: std::cell::OnceCell::new(), label, src, constants, with_aux }
    }

    fn get(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.cell.get_or_init(|| {
            let mut src = if self.with_aux { Cow::Owned(compose_aux(self.src, true)) } else { Cow::Borrowed(self.src) };
            if src.contains(K4_DM_MARKER) {
                src = Cow::Owned(src.replace(K4_DM_MARKER, include_str!("shaders/aux_common.wgsl")));
            }
            make_pipeline_with_constants(device, self.label, &src, self.constants)
        })
    }
}

/// The Q4_0-family kernels built for one block format that needs the `aux`
/// binding (Q5_0 or Q4_K): naive reference, decode matvec (one and four rows
/// per lane), their fused gate/up builds, short and long prefill, and the f16
/// prefill tile when the device grants `shader-f16`.
pub(crate) struct Q4Family {
    pub(crate) naive: LazyPipeline,
    pub(crate) decode: LazyPipeline,
    pub(crate) decode_r4: LazyPipeline,
    pub(crate) swiglu: LazyPipeline,
    pub(crate) swiglu_r4: LazyPipeline,
    pub(crate) small_m: LazyPipeline,
    pub(crate) tiled_rb: LazyPipeline,
    pub(crate) tiled_f16: Option<LazyPipeline>,
}

/// Kernels over F16 or BF16 weights, compiled on first use.
pub(crate) struct HalfKernels {
    pub(crate) naive: LazyPipeline,
    pub(crate) decode: LazyPipeline,
    pub(crate) tiled: LazyPipeline,
}

impl HalfKernels {
    fn new(bf16: bool) -> Self {
        const F16: &[(&str, f64)] = &[];
        const BF16: &[(&str, f64)] = &[("BF16", 1.0)];
        let c = if bf16 { BF16 } else { F16 };
        let tag = |f16: &'static str, bf16_label: &'static str| if bf16 { bf16_label } else { f16 };
        Self {
            naive: LazyPipeline::new(tag("linear_f16", "linear_bf16"), include_str!("shaders/linear_half.wgsl"), c, false),
            decode: LazyPipeline::new(tag("linear_f16_decode", "linear_bf16_decode"), include_str!("shaders/linear_half_decode.wgsl"), c, false),
            tiled: LazyPipeline::new(tag("linear_f16_tiled", "linear_bf16_tiled"), include_str!("shaders/linear_half_tiled.wgsl"), c, false),
        }
    }
}

const EMBED_GATHER_Q4K: &str = include_str!("shaders/embed_gather_q4k.wgsl");

const SRC_Q4_NAIVE: &str = include_str!("shaders/linear_q4.wgsl");
const SRC_Q4_DECODE: &str = include_str!("shaders/linear_q4_decode.wgsl");
const SRC_Q4_DECODE_R4: &str = include_str!("shaders/linear_q4_decode_r4.wgsl");
const SRC_Q4_SMALL_M: &str = include_str!("shaders/linear_q4_small_m.wgsl");
const SRC_Q4_TILED_RB: &str = include_str!("shaders/linear_q4_tiled_rb.wgsl");
const SRC_Q4_TILED_F16: &str = include_str!("shaders/linear_q4_tiled_f16.wgsl");

impl Q4Family {
    fn q50(grant_f16: bool) -> Self {
        const C: &[(&str, f64)] = &[("Q50", 1.0)];
        const C_GATE_UP: &[(&str, f64)] = &[("Q50", 1.0), ("GATE_UP", 1.0)];
        Self {
            naive: LazyPipeline::new("linear_q50", SRC_Q4_NAIVE, C, true),
            decode: LazyPipeline::new("linear_q50_decode", SRC_Q4_DECODE, C, true),
            decode_r4: LazyPipeline::new("linear_q50_decode_r4", SRC_Q4_DECODE_R4, C, true),
            swiglu: LazyPipeline::new("linear_q50_decode_swiglu", SRC_Q4_DECODE, C_GATE_UP, true),
            swiglu_r4: LazyPipeline::new("linear_q50_decode_swiglu_r4", SRC_Q4_DECODE_R4, C_GATE_UP, true),
            small_m: LazyPipeline::new("linear_q50_small_m", SRC_Q4_SMALL_M, C, true),
            tiled_rb: LazyPipeline::new("linear_q50_tiled_rb", SRC_Q4_TILED_RB, C, true),
            tiled_f16: grant_f16.then(|| LazyPipeline::new("linear_q50_tiled_f16", SRC_Q4_TILED_F16, C, true)),
        }
    }

    fn k4(grant_f16: bool) -> Self {
        const C: &[(&str, f64)] = &[("K4", 1.0)];
        const C_GATE_UP: &[(&str, f64)] = &[("K4", 1.0), ("GATE_UP", 1.0)];
        Self {
            naive: LazyPipeline::new("linear_k4", SRC_Q4_NAIVE, C, true),
            decode: LazyPipeline::new("linear_k4_decode", SRC_Q4_DECODE, C, true),
            decode_r4: LazyPipeline::new("linear_k4_decode_r4", SRC_Q4_DECODE_R4, C, true),
            swiglu: LazyPipeline::new("linear_k4_decode_swiglu", SRC_Q4_DECODE, C_GATE_UP, true),
            swiglu_r4: LazyPipeline::new("linear_k4_decode_swiglu_r4", SRC_Q4_DECODE_R4, C_GATE_UP, true),
            small_m: LazyPipeline::new("linear_k4_small_m", SRC_Q4_SMALL_M, C, true),
            tiled_rb: LazyPipeline::new("linear_k4_tiled_rb", SRC_Q4_TILED_RB, C, true),
            tiled_f16: grant_f16.then(|| LazyPipeline::new("linear_k4_tiled_f16", SRC_Q4_TILED_F16, C, true)),
        }
    }
}

fn make_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    make_pipeline_with_constants(device, label, src, &[])
}

/// Writes build constants into a shader's source: each `(name, value)`
/// replaces the initializer of the shader's `const NAME: bool = ...;` or
/// `const NAME: u32 = ...;` declaration, so one source compiles to a
/// different fixed kernel per model (`HEAD_DIM`, which also sets
/// `@workgroup_size`) or per block format (`Q8`, `Q41`, `GATE_UP`, ...).
/// The values are plain WGSL `const`s rather than pipeline-overridable
/// `override` constants, so no pipeline depends on the browser evaluating
/// override expressions at pipeline creation.
pub(crate) fn specialize<'a>(src: &'a str, constants: &[(&str, f64)]) -> Cow<'a, str> {
    if constants.is_empty() {
        return Cow::Borrowed(src);
    }
    let mut out = src.to_owned();
    for &(name, value) in constants {
        let head = format!("const {name}: ");
        let start = out
            .match_indices(&head)
            .map(|(i, _)| i)
            .find(|&i| i == 0 || out.as_bytes()[i - 1] == b'\n')
            .unwrap_or_else(|| panic!("shader has no `const {name}` declaration"));
        let ty_start = start + head.len();
        let eq = ty_start + out[ty_start..].find(" = ").expect("const declaration without an initializer");
        let end = eq + out[eq..].find(';').expect("const declaration without `;`");
        let literal = match &out[ty_start..eq] {
            "bool" => (value != 0.0).to_string(),
            "u32" => format!("{}u", value as u32),
            ty => panic!("build constant {name} has unsupported type {ty}"),
        };
        out.replace_range(eq + 3..end, &literal);
    }
    Cow::Owned(out)
}

/// Like `make_pipeline`, with build constants written into the source
/// (`specialize`) before the module is created.
fn make_pipeline_with_constants(device: &wgpu::Device, label: &str, src: &str, constants: &[(&str, f64)]) -> wgpu::ComputePipeline {
    let src: Cow<str> = if src.contains(AUX_MARKER) { Cow::Owned(compose_aux(src, false)) } else { Cow::Borrowed(src) };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(specialize(&src, constants)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn adapter_report(adapter: &wgpu::Adapter, device: &wgpu::Device) -> String {
    let info = adapter.get_info();
    let l = device.limits();
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "{{\"name\":\"{}\",\"vendor\":{},\"device\":{},\"backend\":\"{:?}\",\"driver\":\"{}\",\"driverInfo\":\"{}\",\"adapterFeatures\":\"{}\",\"deviceFeatures\":\"{}\",\
\"maxStorageBufferBindingSize\":{},\"maxBufferSize\":{},\"maxStorageBuffersPerShaderStage\":{},\"maxComputeWorkgroupStorageSize\":{},\
\"maxComputeInvocationsPerWorkgroup\":{},\"maxComputeWorkgroupSizeX\":{},\"minStorageBufferOffsetAlignment\":{},\"minUniformBufferOffsetAlignment\":{},\
\"minSubgroupSize\":{},\"maxSubgroupSize\":{}}}",
        esc(&info.name),
        info.vendor,
        info.device,
        info.backend,
        esc(&info.driver),
        esc(&info.driver_info),
        esc(&format!("{:?}", adapter.features())),
        esc(&format!("{:?}", device.features())),
        l.max_storage_buffer_binding_size,
        l.max_buffer_size,
        l.max_storage_buffers_per_shader_stage,
        l.max_compute_workgroup_storage_size,
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroup_size_x,
        l.min_storage_buffer_offset_alignment,
        l.min_uniform_buffer_offset_alignment,
        l.min_subgroup_size,
        l.max_subgroup_size,
    )
}

/// Every engine owns its device, so dropping the engine destroys the device
/// and with it every buffer, texture and bind group made from it. Without
/// this, a browser frees GPU memory only when the JS wrapper objects are
/// garbage collected (Firefox does so late), and a page that loads several
/// engines in a row runs the GPU out of memory.
impl Drop for Engine {
    fn drop(&mut self) {
        self.device.destroy();
    }
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
        // The tiled prefill kernels use a 16x8 workgroup and up to 16 KiB of
        // workgroup storage: the WebGPU spec's guaranteed minimums, so every
        // conforming adapter passes this.
        anyhow::ensure!(
            adapter_limits.max_compute_invocations_per_workgroup >= 256 && adapter_limits.max_compute_workgroup_storage_size >= 16384,
            "adapter below the WebGPU minimum compute limits (invocations {}, workgroup storage {})",
            adapter_limits.max_compute_invocations_per_workgroup,
            adapter_limits.max_compute_workgroup_storage_size
        );
        let want_profiling = std::env::var(PROFILE_ENV_VAR).as_deref() == Ok("1");
        let profile_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let grant_profiling = want_profiling && adapter.features().contains(profile_features);
        let mut required_features = if grant_profiling { profile_features } else { wgpu::Features::empty() };
        let grant_pass_timestamps = want_pass_timestamps && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if grant_pass_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // shader-f16 is requested whenever the adapter has it. LEAN_F16=0
        // keeps the F32 prefill kernels on such an adapter (native A/B runs).
        let grant_f16 = adapter.features().contains(wgpu::Features::SHADER_F16) && std::env::var(F16_ENV_VAR).as_deref() != Ok("0");
        if grant_f16 {
            required_features |= wgpu::Features::SHADER_F16;
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
        let adapter_report = adapter_report(&adapter, &device);

        // Every pipeline below is checked before the engine is handed out:
        // an implementation that rejects one of them (a shader it cannot
        // compile, a pipeline it cannot create) makes this call fail, so the
        // backend ladder moves on to the CPU instead of running a forward
        // pass with invalid pipelines.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::Internal);
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
            debug_taps: RefCell::new(None),
            debug_uploads: RefCell::new(None),
            pending_upload_bytes: Cell::new(0),
            adapter_report,
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
            linear_q4_tiled_rb: make_pipeline(&device, "linear_q4_tiled_rb", include_str!("shaders/linear_q4_tiled_rb.wgsl")),
            linear_f32_tiled: make_pipeline(&device, "linear_f32_tiled", include_str!("shaders/linear_f32_tiled.wgsl")),
            linear_q8_tiled_rb: make_pipeline_with_constants(&device, "linear_q8_tiled_rb", include_str!("shaders/linear_q4_tiled_rb.wgsl"), &[("Q8", 1.0)]),
            linear_q4_tiled_f16: grant_f16.then(|| make_pipeline(&device, "linear_q4_tiled_f16", include_str!("shaders/linear_q4_tiled_f16.wgsl"))),
            linear_q8_tiled_f16: grant_f16.then(|| make_pipeline_with_constants(&device, "linear_q8_tiled_f16", include_str!("shaders/linear_q4_tiled_f16.wgsl"), &[("Q8", 1.0)])),
            linear_q4_decode: make_pipeline(&device, "linear_q4_decode", include_str!("shaders/linear_q4_decode.wgsl")),
            linear_q4_decode_swiglu: make_pipeline_with_constants(&device, "linear_q4_decode_swiglu", include_str!("shaders/linear_q4_decode.wgsl"), &[("GATE_UP", 1.0)]),
            linear_q4_decode_r4: make_pipeline(&device, "linear_q4_decode_r4", include_str!("shaders/linear_q4_decode_r4.wgsl")),
            linear_q8_decode_r4: make_pipeline(&device, "linear_q8_decode_r4", include_str!("shaders/linear_q8_decode_r4.wgsl")),
            linear_q4_decode_swiglu_r4: make_pipeline_with_constants(&device, "linear_q4_decode_swiglu_r4", include_str!("shaders/linear_q4_decode_r4.wgsl"), &[("GATE_UP", 1.0)]),
            linear_q8_decode_swiglu_r4: make_pipeline_with_constants(&device, "linear_q8_decode_swiglu_r4", include_str!("shaders/linear_q8_decode_r4.wgsl"), &[("GATE_UP", 1.0)]),
            linear_q4_small_m: make_pipeline(&device, "linear_q4_small_m", include_str!("shaders/linear_q4_small_m.wgsl")),
            linear_q8_small_m: make_pipeline_with_constants(&device, "linear_q8_small_m", include_str!("shaders/linear_q4_small_m.wgsl"), &[("Q8", 1.0)]),
            linear_q41: make_pipeline_with_constants(&device, "linear_q41", include_str!("shaders/linear_q4.wgsl"), &[("Q41", 1.0)]),
            linear_q41_decode: make_pipeline_with_constants(&device, "linear_q41_decode", include_str!("shaders/linear_q4_decode.wgsl"), &[("Q41", 1.0)]),
            linear_q41_decode_r4: make_pipeline_with_constants(&device, "linear_q41_decode_r4", include_str!("shaders/linear_q4_decode_r4.wgsl"), &[("Q41", 1.0)]),
            linear_q41_small_m: make_pipeline_with_constants(&device, "linear_q41_small_m", include_str!("shaders/linear_q4_small_m.wgsl"), &[("Q41", 1.0)]),
            linear_q41_tiled_rb: make_pipeline_with_constants(&device, "linear_q41_tiled_rb", include_str!("shaders/linear_q4_tiled_rb.wgsl"), &[("Q41", 1.0)]),
            half_f16: HalfKernels::new(false),
            half_bf16: HalfKernels::new(true),
            linear_q6k_tiled: LazyPipeline::new("linear_q6k_tiled", include_str!("shaders/linear_q6k_tiled.wgsl"), &[], false),
            embed_gather_q5: LazyPipeline::new("embed_gather_q5", include_str!("shaders/embed_gather_q5.wgsl"), &[], false),
            embed_gather_q4k: LazyPipeline::new("embed_gather_q4k", EMBED_GATHER_Q4K, &[], false),
            embed_gather_f16: LazyPipeline::new("embed_gather_f16", include_str!("shaders/embed_gather_half.wgsl"), &[], false),
            embed_gather_bf16: LazyPipeline::new("embed_gather_bf16", include_str!("shaders/embed_gather_half.wgsl"), &[("BF16", 1.0)], false),
            q50: Q4Family::q50(grant_f16),
            k4: Q4Family::k4(grant_f16),
            attn_prefill: make_pipeline(&device, "attn_prefill", include_str!("shaders/attn_prefill.wgsl")),
            attn_prefill_tiled: make_pipeline(&device, "attn_prefill_tiled", include_str!("shaders/attn_prefill_tiled.wgsl")),
            attn_prefill_tiled_128: make_pipeline_with_constants(&device, "attn_prefill_tiled_128", include_str!("shaders/attn_prefill_tiled.wgsl"), &[("HEAD_DIM", 128.0)]),
            attn_prefill_cache: make_pipeline_with_constants(&device, "attn_prefill_cache", include_str!("shaders/attn_prefill_tiled.wgsl"), &[("CACHE", 1.0)]),
            attn_prefill_cache_128: make_pipeline_with_constants(&device, "attn_prefill_cache_128", include_str!("shaders/attn_prefill_tiled.wgsl"), &[("HEAD_DIM", 128.0), ("CACHE", 1.0)]),
            attn_decode: make_pipeline_with_constants(&device, "attn_decode", include_str!("shaders/attn_decode.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_128: make_pipeline_with_constants(&device, "attn_decode_128", include_str!("shaders/attn_decode.wgsl"), &[("HEAD_DIM", 128.0)]),
            attn_decode_split: make_pipeline_with_constants(&device, "attn_decode_split", include_str!("shaders/attn_decode_split.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_split_128: make_pipeline_with_constants(&device, "attn_decode_split_128", include_str!("shaders/attn_decode_split.wgsl"), &[("HEAD_DIM", 128.0)]),
            attn_decode_reduce: make_pipeline_with_constants(&device, "attn_decode_reduce", include_str!("shaders/attn_decode_reduce.wgsl"), &[("HEAD_DIM", 64.0)]),
            attn_decode_reduce_128: make_pipeline_with_constants(&device, "attn_decode_reduce_128", include_str!("shaders/attn_decode_reduce.wgsl"), &[("HEAD_DIM", 128.0)]),
            add_inplace: make_pipeline(&device, "add_inplace", include_str!("shaders/add_inplace.wgsl")),
            kv_scatter: make_pipeline(&device, "kv_scatter", include_str!("shaders/kv_scatter.wgsl")),
            rope_kv_decode: make_pipeline(&device, "rope_kv_decode", include_str!("shaders/rope_kv_decode.wgsl")),
            add_rmsnorm: make_pipeline(&device, "add_rmsnorm", include_str!("shaders/add_rmsnorm.wgsl")),
            silu_mul_fused: make_pipeline(&device, "silu_mul_fused", include_str!("shaders/silu_mul_fused.wgsl")),
            argmax: make_pipeline(&device, "argmax", include_str!("shaders/argmax.wgsl")),
            mask_logits: make_pipeline(&device, "mask_logits", include_str!("shaders/mask_logits.wgsl")),
            topk_stage: make_pipeline(&device, "topk_stage", include_str!("shaders/topk_stage.wgsl")),
            sample_penalty: make_pipeline(&device, "sample_penalty", include_str!("shaders/sample_penalty.wgsl")),
            sample_final: make_pipeline(&device, "sample_final", include_str!("shaders/sample_final.wgsl")),
            rope_positions: make_pipeline(&device, "rope_positions", include_str!("shaders/rope_positions.wgsl")),
            attn_chunk_masked: make_pipeline(&device, "attn_chunk_masked", include_str!("shaders/attn_chunk_masked.wgsl")),
            gather_dequant_q8_rows: make_pipeline(&device, "gather_dequant_q8_rows", include_str!("shaders/gather_dequant_q8_rows.wgsl")),
            device,
            queue,
            dispatch_count: Cell::new(0),
        };
        let internal = engine.device.pop_error_scope().await;
        let validation = engine.device.pop_error_scope().await;
        if let Some(e) = validation.or(internal) {
            anyhow::bail!("WebGPU pipeline creation failed, not using this device: {e}");
        }
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
        let buf = self.buf_upload(label, bytemuck::cast_slice(data), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        self.debug_record_upload(label, &buf, bytemuck::cast_slice(data));
        buf
    }

    pub fn buf_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if self.debug_uploads.borrow().is_some() {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        let buf = self.buf_upload(label, bytemuck::cast_slice(data), usage);
        self.debug_record_upload(label, &buf, bytemuck::cast_slice(data));
        buf
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
    ///
    /// A `write_buffer` stays staged until the next submit. A browser can
    /// hold only so much staged data (Firefox on Linux loses the device with
    /// "Not enough memory left" once about 100 MB are staged, which a ~1 GB model
    /// reaches), so uploads go in pieces of at most `UPLOAD_PIECE` bytes and
    /// an empty submit flushes them once `UPLOAD_PIECE` bytes are pending.
    /// The same on every device.
    pub(crate) fn buf_upload(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(4), usage, mapped_at_creation: false });
        for (i, piece) in bytes.chunks(UPLOAD_PIECE).enumerate() {
            self.queue.write_buffer(&buf, (i * UPLOAD_PIECE) as u64, piece);
            let pending = self.pending_upload_bytes.get() + piece.len();
            if pending >= UPLOAD_PIECE {
                self.queue.submit([]);
                self.pending_upload_bytes.set(0);
            } else {
                self.pending_upload_bytes.set(pending);
            }
        }
        buf
    }

    pub fn set_debug_uploads(&self, on: bool) {
        *self.debug_uploads.borrow_mut() = on.then(Vec::new);
    }

    fn debug_record_upload(&self, label: &str, buf: &wgpu::Buffer, bytes: &[u8]) {
        if let Some(list) = self.debug_uploads.borrow_mut().as_mut() {
            list.push((label.to_string(), buf.clone(), fnv1a(bytes)));
        }
    }

    /// Reads back every buffer recorded since `set_debug_uploads(true)` and
    /// returns JSON: how many were checked, and for each one whose device
    /// bytes differ from what was uploaded, its label, size, and the first
    /// and last differing byte offsets.
    pub async fn debug_verify_uploads(&self) -> String {
        let list: Vec<(String, wgpu::Buffer, u64)> = self.debug_uploads.borrow().clone().unwrap_or_default();
        let mut bad = Vec::new();
        let mut bytes_checked = 0u64;
        for (label, buf, want) in &list {
            let size = buf.size();
            bytes_checked += size;
            let got = self.read_buffer(buf, (size / 4) as usize).await;
            let got: &[u8] = bytemuck::cast_slice(&got);
            if fnv1a(got) != *want {
                bad.push(format!("{{\"label\":\"{label}\",\"bytes\":{size}}}"));
            }
        }
        format!("{{\"checked\":{},\"bytes\":{bytes_checked},\"mismatched\":[{}]}}", list.len(), bad.join(","))
    }

    pub fn buf_empty(&self, len_f32: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len_f32.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }


    pub(crate) fn bind_group(&self, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
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
    /// The compiled pipeline of a lazily created kernel.
    pub(crate) fn pipe<'a>(&'a self, p: &'a LazyPipeline) -> &'a wgpu::ComputePipeline {
        p.get(&self.device)
    }

    pub(crate) fn dispatch(&self, pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, wgs: (u32, u32, u32), _label: &str) {
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
    pub(crate) fn reset_profile(&self) {
        self.profile_labels.borrow_mut().clear();
    }

    /// Resolves this step's recorded timestamp pairs into a fresh buffer -
    /// must be called on the same `encoder` the profiled dispatches were
    /// recorded into, before that encoder is finished/submitted (query
    /// resolution is an encoder-timeline command). Returns `None` when
    /// profiling is off or nothing was recorded this step.
    pub(crate) fn resolve_profile(&self, encoder: &mut wgpu::CommandEncoder) -> Option<(wgpu::Buffer, u32)> {
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
    pub(crate) async fn read_profile(&self, resolve_buf: &wgpu::Buffer, count: u32) -> Vec<(String, f64)> {
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
        self.map_done(rx).await;
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

    pub(crate) fn begin_pass<'e>(&self, encoder: &'e mut wgpu::CommandEncoder, label: &str) -> wgpu::ComputePass<'e> {
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

    pub(crate) fn diag_split(&self) -> bool {
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
    pub(crate) fn diag_begin(&self) {
        self.diag_labels.borrow_mut().clear();
    }

    /// Resolves every pass timestamp recorded since `diag_begin` (including
    /// ones in encoders already submitted - query slots persist) into a
    /// buffer, on the forward call's last encoder before it is finished.
    pub(crate) fn diag_resolve(&self, encoder: &mut wgpu::CommandEncoder) -> Option<(wgpu::Buffer, u32)> {
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
    pub(crate) async fn diag_collect(&self, resolve: &wgpu::Buffer, count: u32) {
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
        self.map_done(rx).await;
        let data = slice.get_mapped_range();
        let raw: &[u64] = bytemuck::cast_slice(&data);
        let period = f64::from(self.timestamp_period);
        let labels = self.diag_labels.borrow();
        let mut out = DiagGpu::default();
        let (mut first, mut last) = (u64::MAX, 0u64);
        for i in 0..count as usize {
            let (b, e) = (raw[i * 2], raw[i * 2 + 1]);
            let ms = if b == 0 || e < b {
                out.unwritten += 1;
                None
            } else {
                first = first.min(b);
                last = last.max(e);
                let ms = (e - b) as f64 * period / 1e6;
                out.pass_sum_ms += ms;
                Some(ms)
            };
            match out.segments.iter_mut().find(|s| s.0 == labels[i]) {
                Some(s) => {
                    s.1 = match (s.1, ms) {
                        (Some(a), Some(b)) => Some(a + b),
                        (a, b) => a.or(b),
                    };
                    s.2 += 1;
                }
                None => out.segments.push((labels[i].clone(), ms, 1)),
            }
        }
        out.span_ms = (first <= last).then(|| (last - first) as f64 * period / 1e6);
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
    pub(crate) fn flush_encoder(&self, encoder: &mut wgpu::CommandEncoder, label: &str) {
        let fresh = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let old = std::mem::replace(encoder, fresh);
        self.queue.submit(Some(old.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    pub fn set_debug_taps(&self, on: bool) {
        *self.debug_taps.borrow_mut() = on.then(Vec::new);
    }

    pub(crate) fn debug_taps_on(&self) -> bool {
        self.debug_taps.borrow().is_some()
    }

    /// Records a copy of `size` bytes of `src` at `offset` into a fresh
    /// buffer, labelled, on `encoder` (outside any compute pass).
    pub(crate) fn debug_tap(&self, encoder: &mut wgpu::CommandEncoder, label: &str, src: &wgpu::Buffer, offset: u64, size: u64) {
        let mut taps = self.debug_taps.borrow_mut();
        let Some(taps) = taps.as_mut() else { return };
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_tap"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, offset, &dst, 0, size);
        taps.push((label.to_string(), dst, size));
    }

    /// Reads every tap recorded since `set_debug_taps(true)` (or the last
    /// collect) and returns one JSON object per tap, in recording order:
    /// `op`, `n` (f32 count), `hash` (FNV-1a 64 of the raw bits), `sum`,
    /// `maxAbs`, and the first four values (every value when `n <= 8192`).
    pub async fn debug_collect(&self) -> String {
        let taps = self.debug_taps.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default();
        let mut rows = Vec::with_capacity(taps.len());
        for (label, buf, size) in taps {
            let v = self.read_buffer(&buf, (size / 4) as usize).await;
            let h = fnv1a(bytemuck::cast_slice(&v));
            let (mut sum, mut max_abs) = (0f64, 0f32);
            for x in &v {
                sum += f64::from(*x);
                max_abs = max_abs.max(x.abs());
            }
            // Small taps (q/k/v of a short prompt) carry every value, so
            // two runs can be diffed element by element.
            let head: Vec<String> = v.iter().take(if v.len() <= 8192 { v.len() } else { 4 }).map(|x| format!("{x:e}")).collect();
            rows.push(format!(
                "{{\"op\":\"{label}\",\"n\":{},\"hash\":\"{h:016x}\",\"sum\":{sum:e},\"maxAbs\":{max_abs:e},\"head\":[{}]}}",
                v.len(),
                head.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(",")
            ));
        }
        format!("[{}]", rows.join(","))
    }

    pub fn reset_dispatch_count(&self) {
        self.dispatch_count.set(0);
    }

    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.get()
    }

    /// Waits until a `map_async` request resolved. Natively that is one
    /// blocking device poll. In the browser nothing here drives the
    /// resolution, and Firefox only resolves a map when a queue submission
    /// wakes its poll: left alone, a map takes about 100 ms to resolve there
    /// (against well under a millisecond in Chromium), so every token read
    /// back cost 100 ms. While the map is pending, an empty submit goes out
    /// every timer tick (about 4 ms), which gets it resolved within one tick
    /// of the GPU finishing; the wait ends the moment the map resolves, so
    /// browsers that resolve on their own lose nothing.
    async fn map_done(&self, rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>) {
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        #[cfg(all(target_arch = "wasm32", feature = "web"))]
        let mut rx = rx;
        #[cfg(all(target_arch = "wasm32", feature = "web"))]
        let res = loop {
            use std::future::Future;
            use std::pin::Pin;
            use std::task::Poll;
            let mut tick = timer::Tick::new();
            let ready = std::future::poll_fn(|cx| {
                if let Poll::Ready(v) = Pin::new(&mut rx).poll(cx) {
                    return Poll::Ready(Some(v));
                }
                if Pin::new(&mut tick).poll(cx).is_ready() {
                    return Poll::Ready(None);
                }
                Poll::Pending
            })
            .await;
            match ready {
                Some(v) => break v,
                None => {
                    self.queue.submit([]);
                }
            }
        };
        #[cfg(not(all(target_arch = "wasm32", feature = "web")))]
        let res = rx.await;
        res.expect("map_async channel dropped").expect("buffer map failed");
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
        self.map_done(rx).await;
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
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback_u32") });
        let staging = self.stage_u32(&mut encoder, buf);
        self.queue.submit(Some(encoder.finish()));
        self.map_u32(&staging).await
    }

    /// Records the copy of `buf`'s first `u32` into a fresh mappable buffer
    /// on `encoder`, so a forward pass can submit its work and its readback
    /// copy together (one submit per decode step instead of two); read the
    /// value with `map_u32` after submitting.
    pub(crate) fn stage_u32(&self, encoder: &mut wgpu::CommandEncoder, buf: &wgpu::Buffer) -> wgpu::Buffer {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging_u32"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, 4);
        staging
    }

    pub(crate) async fn map_u32(&self, staging: &wgpu::Buffer) -> u32 {
        let rx = self.start_map_u32(staging);
        self.finish_map_u32(staging, rx).await
    }

    /// Requests the map of a `stage_u32` buffer right away, without waiting
    /// for it. A pipeline that keeps several steps in flight asks for every
    /// step's map when it submits the step, so the browser has all of them
    /// outstanding at once and can resolve them together, instead of
    /// starting each map only after the previous one resolved.
    pub(crate) fn start_map_u32(&self, staging: &wgpu::Buffer) -> futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = futures_channel::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        rx
    }

    pub(crate) async fn finish_map_u32(&self, staging: &wgpu::Buffer, rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>) -> u32 {
        self.map_done(rx).await;
        let slice = staging.slice(..);
        let data = slice.get_mapped_range();
        let result: u32 = bytemuck::cast_slice(&data)[0];
        drop(data);
        staging.unmap();
        result
    }

    /// Reads every `u32` of a mapped block buffer made by `start_map_u32`.
    pub(crate) async fn finish_map_u32s(&self, staging: &wgpu::Buffer, rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>) -> Vec<u32> {
        self.map_done(rx).await;
        let slice = staging.slice(..);
        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

/// A timer tick for `Engine::map_done`: resolves after one `setTimeout(0)`.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
mod timer {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = setTimeout)]
        fn set_timeout(handler: JsValue, ms: i32) -> i32;
    }

    #[derive(Default)]
    struct State {
        fired: bool,
        waker: Option<Waker>,
    }

    pub(super) struct Tick {
        state: Rc<RefCell<State>>,
        started: bool,
    }

    impl Tick {
        pub(super) fn new() -> Self {
            Tick { state: Rc::default(), started: false }
        }
    }

    impl Future for Tick {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.state.borrow().fired {
                return Poll::Ready(());
            }
            self.state.borrow_mut().waker = Some(cx.waker().clone());
            if !self.started {
                self.started = true;
                let state = self.state.clone();
                let fire = Closure::once_into_js(move || {
                    let mut s = state.borrow_mut();
                    s.fired = true;
                    if let Some(w) = s.waker.take() {
                        w.wake();
                    }
                });
                set_timeout(fire, 0);
            }
            Poll::Pending
        }
    }
}
