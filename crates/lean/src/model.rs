//! Qwen2.5-0.5B-Instruct GPU-resident forward pass: GGUF weights straight
//! onto the GPU (two-phase loading - `GgufReader`'s file handle is dropped
//! at the end of `GpuModel::load`, never held alongside the GPU-resident
//! copies), one dispatch per op recorded into a single `wgpu::CommandEncoder`
//! per prefill/decode call, one readback per call (`Engine::read_buffer`).
//! Structure (Engine/Pool split, `linear`/`rmsnorm`/`add_inplace`/`silu_mul`
//! helper-fn shape) ported from `t0-web/crates/t0-fast/src/model.rs`; the
//! attention/RoPE/embedding-gather kernels are new (t0 has no GQA, no
//! causal mask, no token embedding table - see the shaders' own doc
//! comments for exactly what's ported vs new).
//!
//! Reference for every op's numerics: `crates/lean/reference/gen_fixture.py`
//! (HF transformers' own Qwen2 forward, `AutoModelForCausalLM` loading the
//! same GGUF via `gguf_file=`) - see `modeling_qwen2.py` in that venv's
//! transformers install for `rotate_half`, GQA `repeat_kv`, RMSNorm, SwiGLU.
//!
//! Tensor names follow llama.cpp's GGUF convention
//! (`blk.N.attn_{q,k,v,output}`, `ffn_{gate,up,down}`). This GGUF carries a
//! separate `output.weight` (Q8_0) distinct from `token_embd.weight`
//! (Q4_0) despite `tie_word_embeddings: true` in `config.json` - verified
//! against the file directly (not assumed), so the lm head uses
//! `output.weight` and needs no tied-embedding sharing logic.
//!
//! KV cache layout (load-bearing, chosen here): per layer, two buffers
//! `[n_kv_heads, max_ctx, head_dim]`, head-major and contiguous per head -
//! matches `shaders/attn_decode.wgsl`'s indexing. Not a ring buffer: `kv_len`
//! only grows, capped by `max_ctx` (the CLI's fixed context, not the
//! model's `qwen2.context_length`). Writes into it are GPU-resident, no CPU
//! readback, recorded in the same encoder as the dispatch that produced the
//! source K/V: plain `copy_buffer_to_buffer` calls for a decode step's one
//! row, one `kv_scatter` dispatch per buffer for multi-row prefill/chunk
//! writes - so a freshly-computed decode step's own K/V is visible to that
//! same step's causal attention (`kv_len` passed to `attn_decode` already
//! includes the current position).

use anyhow::{Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::BufReader;
use wgpu::BindGroupEntry;

use crate::config::{config_from_gguf, Architecture, Qwen2Config};
use crate::engine::Engine;
use crate::gguf::GgufReader;
use crate::pool::Pool;
use crate::quant::{load_embedding_table_gguf, load_matmul_weight_gguf, EmbeddingTable, MatMulWeight};
use crate::gguf::GgmlDtype;

/// Below this many rows (and above 1), Q4_0 and Q8_0 prefill use
/// `linear_q4_small_m.wgsl`: it reads and dequantises each weight word once
/// per group of 8 query rows, where the tiled kernels pay for a whole
/// padded row tile. Measured on a mobile GPU (Adreno 6xx), the 32x32 tiled
/// kernel ran a 36-token prompt's MLP matmuls about 5x slower than 36
/// decode-style matvecs would have; see docs/runs/2026-10-02-lean-mobile.md
/// for the M2 sweep this value comes from. A row-count rule, the same on
/// every device: below 64 rows the 64-row tile of `linear_q4_tiled_rb.wgsl`
/// is mostly padding, at and above it that kernel reuses each dequantised
/// weight stage across 64 rows instead of 8.
const SMALL_M_MAX_ROWS: u32 = 64;

/// WebGPU's `max_compute_workgroups_per_dimension`: 65535 on every backend
/// (spec-mandated minimum-and-typical value, not a per-device tuned
/// number - `wgpu::Limits::default()` and this project's adapters alike
/// report exactly this for the dimension). A single-dimension dispatch of
/// a wide 1-D elementwise kernel (`add_inplace`/`silu_mul_fused`/the
/// embedding gathers) can exceed this at long-context prefill on models
/// with a wide `intermediate_size` - found on Qwen2.5-3B-Instruct
/// (`intermediate_size` 11008): `silu_mul_fused`'s dispatch,
/// `(rows*intermediate_size).div_ceil(256)`, is 65,532 groups at
/// `rows`=1524 and 65,575 at `rows`=1525 - the actual cause of a native
/// wgpu panic ("Encoder is invalid" at `CommandEncoder::finish`, no
/// specific validation message surfaced) that this constant's use in
/// `grid1d` below fixes, by folding into a second grid dimension instead
/// of ever exceeding it in one.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// `(x, y, stride_x)` dispatch shape for a 1-D elementwise kernel over
/// `total` elements at `@workgroup_size(256)`, safe against
/// `MAX_WORKGROUPS_PER_DIM` regardless of how large `total` gets: `x` is
/// capped at the limit and any remainder folds into `y`. `stride_x = x *
/// 256` is passed through the kernel's own Dims uniform (see
/// `AddDims`/`SiluDims`/`GatherDims`'s `stride_x` field) so its shader can
/// recover the flat element index as `gid.y * stride_x + gid.x` instead of
/// just `gid.x` - chosen from `total` alone (a shape property), identical
/// on every device and every call at the same shape, never autotuned.
fn grid1d(total: u32) -> (u32, u32, u32) {
    let groups = total.div_ceil(256).max(1);
    let x = groups.min(MAX_WORKGROUPS_PER_DIM);
    let y = groups.div_ceil(x);
    (x, y, x * 256)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearQDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    /// Row-chunk offset/total for weights `quant.rs::chunk_rows` split
    /// across multiple bindings (see its doc comment). A single-chunk
    /// weight passes `n_offset: 0, n_total: n`.
    n_offset: u32,
    n_total: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AddDims {
    len: u32,
    /// See `grid1d`'s doc comment: `x * 256` for the dispatch that used
    /// this Dims value, so the shader can recover the flat index from a
    /// 2-D grid instead of assuming `gid.x` alone spans `len`.
    stride_x: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsDims {
    rows: u32,
    dim: u32,
    eps: f32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeKvDims {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    max_ctx: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeDims {
    rows: u32,
    heads: u32,
    head_dim: u32,
    pos_base: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluDims {
    rows: u32,
    hidden: u32,
    /// See `AddDims::stride_x`'s doc comment.
    stride_x: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnPrefillDims {
    seq: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeDims {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    max_ctx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeSplitDims {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    max_ctx: u32,
    scale: f32,
    chunk: u32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeReduceDims {
    n_heads: u32,
    head_dim: u32,
    num_splits: u32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherDims {
    rows: u32,
    hidden: u32,
    blocks_per_row: u32,
    /// See `AddDims::stride_x`'s doc comment.
    stride_x: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxDims {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskDims {
    n: u32,
    offset: u32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopePosDims {
    rows: u32,
    heads: u32,
    head_dim: u32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnChunkDims {
    t: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_total: u32,
    max_ctx: u32,
    scale: f32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherRowsDims {
    n_selected: u32,
    k: u32,
    blocks_per_row: u32,
    out_row_offset: u32,
}

/// Caller-supplied positions and attention topology for `forward_chunk_spec`
///: the mechanism behind llm-life variant A/B's per-cell RoPE restart and
/// block-diagonal/sparse masks (this crate's consumer survey, gap #5).
/// `ForwardSpec::default()` (no positions, no mask) means "continue the
/// resident cache causally": the same behavior `forward_prefill_suffix`
/// already gives, expressed generically.
#[derive(Default, Clone)]
pub struct ForwardSpec {
    /// One absolute position per token, same length as the chunk. `None`
    /// means "continue counting from the cache's current `kv_len`" (plain
    /// causal continuation).
    pub positions: Option<Vec<u32>>,
    /// Packed bitset (`build_mask_bitset`-style, 32 bits/word), row-major
    /// `[t, prefix_len + t]`: bit `(i, j)` set means query row `i` may
    /// attend key `j`. `None` means "causal over the resident prefix plus
    /// this chunk" (`build_prefix_causal_bits`).
    pub allowed_bits: Option<Vec<u32>>,
}

impl ForwardSpec {
    pub fn with_positions(mut self, positions: Vec<u32>) -> Self {
        self.positions = Some(positions);
        self
    }

    /// `allowed[i * cols + j]`: a plain `bool` mask, `rows` query positions
    /// by `cols = prefix_len + rows` keys: packed into this crate's bitset
    /// format. The convenience form for a caller building a mask as
    /// `Vec<bool>` (llm-life's `Chunk::allowed`, for instance) rather than
    /// bit-packing it themselves.
    pub fn with_allowed(mut self, allowed: &[bool], rows: usize, cols: usize) -> Self {
        self.allowed_bits = Some(pack_bool_mask(allowed, rows, cols));
        self
    }

    pub fn with_allowed_bits(mut self, bits: Vec<u32>) -> Self {
        self.allowed_bits = Some(bits);
        self
    }
}

/// Packs a row-major `[rows, cols]` bool mask into this crate's bitset
/// format (32 entries per `u32`, bit `idx % 32` of word `idx / 32`, `idx =
/// row * cols + col`): shared by `ForwardSpec::with_allowed` and any
/// caller building the same shape directly.
pub fn pack_bool_mask(allowed: &[bool], rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(allowed.len(), rows * cols, "pack_bool_mask: allowed.len() must be rows*cols");
    let mut bits = vec![0u32; (rows * cols).div_ceil(32)];
    for (idx, &a) in allowed.iter().enumerate() {
        if a {
            bits[idx / 32] |= 1u32 << (idx % 32);
        }
    }
    bits
}

/// The default mask `forward_chunk_spec` builds when `ForwardSpec::allowed_bits`
/// is `None`: query row `i` (0-based within the chunk) may attend every
/// resident-prefix key plus its own chunk keys `[0, i]`: i.e. plain causal
/// continuation of a `prefix_len`-long resident cache, the same topology
/// `forward_prefill_suffix` already gives without an explicit mask.
pub fn build_prefix_causal_bits(prefix_len: u32, t: u32) -> Vec<u32> {
    let cols = prefix_len + t;
    let total = (t as usize) * (cols as usize);
    let mut bits = vec![0u32; total.div_ceil(32)];
    for i in 0..t {
        let allowed_upto = prefix_len + i + 1; // this row may attend [0, allowed_upto)
        for j in 0..allowed_upto {
            let idx = (i * cols + j) as usize;
            bits[idx / 32] |= 1u32 << (idx % 32);
        }
    }
    bits
}

struct LayerWeights {
    attn_norm: wgpu::Buffer,
    q_w: MatMulWeight,
    q_b: wgpu::Buffer,
    k_w: MatMulWeight,
    k_b: wgpu::Buffer,
    v_w: MatMulWeight,
    v_b: wgpu::Buffer,
    o_w: MatMulWeight,
    o_b: wgpu::Buffer, // zero, attn_output has no bias
    ffn_norm: wgpu::Buffer,
    /// `ffn_gate.weight` and `ffn_up.weight` concatenated at load time into
    /// one `[2*intermediate_size, hidden_size]` weight (see
    /// `gguf_matmul_concat2`'s doc comment) - one `linear()` dispatch
    /// produces both halves, consumed by `silu_mul_fused` instead of two
    /// separate matmuls into two separate buffers.
    gate_up_w: MatMulWeight,
    gate_up_b: wgpu::Buffer, // zero, len 2*intermediate_size
    down_w: MatMulWeight,
    down_b: wgpu::Buffer, // zero
    /// Qwen3-only (`cfg.qk_norm`): per-head RMSNorm gamma, `[head_dim]`
    /// each, applied to q/k right after projection, before RoPE. `None` for
    /// Qwen2 layers.
    q_norm: Option<wgpu::Buffer>,
    k_norm: Option<wgpu::Buffer>,
    /// `attn_q`/`attn_k`/`attn_v` weights concatenated at load time into one
    /// `[q_dim+2*kv_dim, hidden_size]` weight (`gguf_matmul_qkv_fused`) -
    /// `qkv_proj`'s decode-shaped (`rows == 1`, no LoRA) fast path runs one
    /// `linear()` dispatch against this instead of three, and reads q/k/v
    /// back out of the one result buffer via offset bindings (`BufView`) -
    /// no copy. `q_w`/`k_w`/`v_w` above are kept alongside this (not
    /// replaced) for the prefill/chunked-forward paths, which stay on the
    /// three-separate-matmul path unconditionally (`rows > 1`; see
    /// `qkv_proj`'s doc comment on why a fused *strided* per-row layout isn't
    /// safe to read back with a plain offset+size binding). `None` when
    /// `gguf_matmul_qkv_fused` found a dtype mismatch across the three
    /// tensors - `qkv_proj` falls back to the three-matmul path in that case
    /// too, at every `rows`.
    qkv_w: Option<MatMulWeight>,
    /// Bias matching `qkv_w`'s row order (`q_b` then `k_b` then `v_b`
    /// concatenated) - a real per-tensor bias on Qwen2 (`cfg.has_qkv_bias`),
    /// all-zero otherwise. `Some` exactly when `qkv_w` is.
    qkv_b: Option<wgpu::Buffer>,
}

pub struct GpuModel {
    pub config: Qwen2Config,
    embed: EmbeddingTable,
    layers: Vec<LayerWeights>,
    out_norm: wgpu::Buffer,
    lm_head: MatMulWeight,
    zero_bias_vocab: wgpu::Buffer,
    /// Selects, inside `linear()`, between the naive reference kernel
    /// (`false`, always `linear_q4.wgsl`) and the tiled/coalesced
    /// fast kernels ported in slice 2 (`true`). Set at `load()` time so a
    /// single `GpuModel` doesn't mix the two - the fixture check runs both.
    pub fast_kernels: bool,
    pub pool: Pool,
    /// Runtime LoRA, applied alongside the frozen Q4_0 base on every
    /// subsequent forward call (q/k/v/o only: see `lora.rs`'s module doc).
    /// `None` means no adapter: identical dispatch sequence to before this
    /// feature existed. Switchable/removable via `apply_lora`/`clear_lora`
    /// with no base reload.
    pub lora: Option<crate::lora::LoraAdapter>,
}

/// One KV cache buffer pair per layer. See this file's top doc comment for
/// the layout. `max_ctx` is the CLI's fixed context length (prompt +
/// max_new_tokens known up front), not the model's full 32768 window.
pub struct KvCache {
    pub k: Vec<wgpu::Buffer>,
    pub v: Vec<wgpu::Buffer>,
    pub max_ctx: u32,
    pub kv_len: u32,
    num_kv_heads: u32,
    head_dim: u32,
    /// Process-unique id, so `Pool::use_kv_cache` can tell a new cache (whose
    /// buffers no cached bind group points at yet) from the one it last saw.
    id: u64,
}

static NEXT_KV_CACHE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl KvCache {
    pub fn new(engine: &Engine, config: &Qwen2Config, max_ctx: u32) -> Self {
        let per_layer = (config.num_kv_heads * config.head_dim) as u32 * max_ctx;
        let k = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.k"))).collect();
        let v = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.v"))).collect();
        let id = NEXT_KV_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        KvCache { k, v, max_ctx, kv_len: 0, num_kv_heads: config.num_kv_heads as u32, head_dim: config.head_dim as u32, id }
    }

    /// Sum of every layer's K+V buffer size, at this cache's `max_ctx` (the
    /// buffers are allocated at full `max_ctx` up front - see `new` above -
    /// so this is constant across the cache's lifetime, not `kv_len`-scaled).
    pub fn gpu_bytes(&self) -> u64 {
        self.k.iter().map(|b| b.size()).sum::<u64>() + self.v.iter().map(|b| b.size()).sum::<u64>()
    }

    /// Reads back positions `[0, kv_len)` of every layer's K/V buffers,
    /// compacted (no `max_ctx` padding) into a max_ctx-independent snapshot
    /// that can be restored into a cache with a *different* `max_ctx` (e.g.
    /// a resident prefix snapshot restored into a fresh, longer-context
    /// cache for a new turn). Async readback only (`Engine::read_buffer`) -
    /// safe to call from the browser.
    pub async fn snapshot(&self, engine: &Engine) -> KvSnapshot {
        let mut k_out = Vec::with_capacity(self.k.len());
        let mut v_out = Vec::with_capacity(self.v.len());
        for i in 0..self.k.len() {
            k_out.push(compact_kv_prefix(engine, &self.k[i], self.num_kv_heads, self.head_dim, self.kv_len, self.max_ctx).await);
            v_out.push(compact_kv_prefix(engine, &self.v[i], self.num_kv_heads, self.head_dim, self.kv_len, self.max_ctx).await);
        }
        KvSnapshot { kv_len: self.kv_len, num_kv_heads: self.num_kv_heads, head_dim: self.head_dim, num_layers: self.k.len() as u32, k: k_out, v: v_out }
    }

    /// Writes a snapshot's compacted K/V back into this cache's own buffers
    /// (`queue.write_buffer`, no readback - safe on any target) and sets
    /// `kv_len` to the snapshot's, so generation can resume from the
    /// restored prefix without re-prefilling it. Restoring into the *same*
    /// `KvCache` instance the snapshot was taken from (or any instance
    /// whose buffers were already bound into a `Pool`'s cached bind groups)
    /// needs no `pool.reset()` - only allocating a *new* `KvCache` does
    /// (see `pool.rs`'s bug note).
    pub fn restore(&mut self, engine: &Engine, snapshot: &KvSnapshot) {
        assert_eq!(snapshot.num_layers as usize, self.k.len(), "kv snapshot layer count mismatch");
        assert_eq!(snapshot.num_kv_heads, self.num_kv_heads, "kv snapshot num_kv_heads mismatch");
        assert_eq!(snapshot.head_dim, self.head_dim, "kv snapshot head_dim mismatch");
        assert!(snapshot.kv_len <= self.max_ctx, "kv snapshot kv_len {} exceeds max_ctx {}", snapshot.kv_len, self.max_ctx);
        for i in 0..self.k.len() {
            write_kv_prefix(engine, &self.k[i], &snapshot.k[i], self.num_kv_heads, self.head_dim, snapshot.kv_len, self.max_ctx);
            write_kv_prefix(engine, &self.v[i], &snapshot.v[i], self.num_kv_heads, self.head_dim, snapshot.kv_len, self.max_ctx);
        }
        self.kv_len = snapshot.kv_len;
    }
}

/// A resident-prefix KV image: every layer's K/V for positions `[0,
/// kv_len)`, compacted (`[kv_heads, kv_len, head_dim]`, no `max_ctx`
/// padding) so it can be exported/imported as bytes and stored by a
/// consumer keyed by its own prompt/tool-schema hash (see this crate's
/// consumer survey, gap #1). Layout must match `KvCache`'s per-head
/// contiguous convention (this file's top doc comment) for `restore` to be
/// a plain per-head copy.
pub struct KvSnapshot {
    pub kv_len: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_layers: u32,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvSnapshot {
    /// Flat little-endian format: `[kv_len, num_kv_heads, head_dim,
    /// num_layers]` (4 x u32) followed by, per layer, `k` then `v` as raw
    /// f32 bytes (`num_kv_heads * kv_len * head_dim` each). No length
    /// prefix per tensor - every tensor's length is derivable from the
    /// header, matching `from_bytes`'s parse.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.k.iter().map(|v| v.len() * 4).sum::<usize>() * 2);
        out.extend_from_slice(&self.kv_len.to_le_bytes());
        out.extend_from_slice(&self.num_kv_heads.to_le_bytes());
        out.extend_from_slice(&self.head_dim.to_le_bytes());
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        for i in 0..self.num_layers as usize {
            out.extend_from_slice(bytemuck::cast_slice(&self.k[i]));
            out.extend_from_slice(bytemuck::cast_slice(&self.v[i]));
        }
        out
    }

    /// Inverse of `to_bytes`. Parses f32s with `f32::from_le_bytes` on
    /// individually-sliced 4-byte chunks rather than `bytemuck::cast_slice`
    /// on the input `&[u8]` - a `Vec<u8>` handed across a JS/wasm boundary
    /// (or read from a file) is not guaranteed 4-byte aligned, and
    /// `cast_slice` panics on misalignment.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(bytes.len() >= 16, "kv snapshot bytes too short for header");
        let kv_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let num_kv_heads = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let head_dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let num_layers = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let per_tensor = (num_kv_heads as usize) * (kv_len as usize) * (head_dim as usize);
        let mut off = 16usize;
        let mut k = Vec::with_capacity(num_layers as usize);
        let mut v = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            anyhow::ensure!(bytes.len() >= off + per_tensor * 4 * 2, "kv snapshot bytes truncated");
            k.push(read_f32_le(&bytes[off..off + per_tensor * 4]));
            off += per_tensor * 4;
            v.push(read_f32_le(&bytes[off..off + per_tensor * 4]));
            off += per_tensor * 4;
        }
        Ok(KvSnapshot { kv_len, num_kv_heads, head_dim, num_layers, k, v })
    }
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Reads a KV buffer's full `[kv_heads, max_ctx, head_dim]` contents back
/// and compacts it to `[kv_heads, kv_len, head_dim]` (dropping the unused
/// `[kv_len, max_ctx)` padding per head).
async fn compact_kv_prefix(engine: &Engine, buf: &wgpu::Buffer, kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32) -> Vec<f32> {
    let full = engine.read_buffer(buf, (kv_heads * max_ctx * head_dim) as usize).await;
    let mut out = Vec::with_capacity((kv_heads * kv_len * head_dim) as usize);
    for h in 0..kv_heads {
        let base = (h * max_ctx * head_dim) as usize;
        let len = (kv_len * head_dim) as usize;
        out.extend_from_slice(&full[base..base + len]);
    }
    out
}

/// Inverse of `compact_kv_prefix`: writes a compacted `[kv_heads, kv_len,
/// head_dim]` snapshot back into a `[kv_heads, max_ctx, head_dim]` cache
/// buffer's `[0, kv_len)` prefix, one `queue.write_buffer` per head (queued
/// CPU->GPU writes, no mapping/readback - safe in the browser).
fn write_kv_prefix(engine: &Engine, buf: &wgpu::Buffer, compact: &[f32], kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32) {
    for h in 0..kv_heads {
        let src_base = (h * kv_len * head_dim) as usize;
        let src_len = (kv_len * head_dim) as usize;
        let dst_offset = ((h * max_ctx * head_dim) as u64) * 4;
        engine.queue.write_buffer(buf, dst_offset, bytemuck::cast_slice(&compact[src_base..src_base + src_len]));
    }
}

/// CPU argmax over a full logits vector, ties broken toward the lower index -
/// matches `shaders/argmax.wgsl`'s tie-break (`forward_decode_step_argmax`'s
/// GPU path) so a caller can freely mix the two without behavior drift
/// (`generate.rs::decode_loop`'s greedy path uses this on `forward_prefill`'s
/// CPU-readback logits, then switches to the GPU path for every decode step).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Packs `allowed` token ids into a bitset (32 ids per `u32`, bit `i % 32`
/// of word `i / 32`) for `mask_logits.wgsl`. `vocab` sizes the bitset (any
/// id `>= vocab` is meaningless to the kernel, which never reads past
/// `dims.n`, but is rejected here to catch a caller bug early).
pub fn build_mask_bitset(vocab: usize, allowed: &[u32]) -> Vec<u32> {
    let mut bits = vec![0u32; vocab.div_ceil(32)];
    for &t in allowed {
        assert!((t as usize) < vocab, "mask token id {t} >= vocab {vocab}");
        bits[(t / 32) as usize] |= 1u32 << (t % 32);
    }
    bits
}

/// Dequantized host-side data for an f32-ish GGUF tensor (norm gamma, bias),
/// without uploading it - used where a caller needs to concatenate several
/// small tensors (see `qkv_b` below) before making one GPU buffer out of the
/// result, so the individual tensors are never uploaded twice.
fn gguf_f32_raw<R: std::io::Read + std::io::Seek>(reader: &mut GgufReader<R>, name: &str) -> Result<Vec<f32>> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let n: usize = info.shape().iter().product();
    let bytes = reader.tensor_data(name)?;
    Ok(crate::gguf::dequantize_for(info.dtype(), &bytes, n))
}

fn gguf_f32<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<wgpu::Buffer> {
    let data = gguf_f32_raw(reader, name)?;
    Ok(engine.buf_f32(&data, name))
}

fn gguf_matmul<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<MatMulWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let bytes = reader.tensor_data(name)?;
    load_matmul_weight_gguf(engine, name, &shape, info.dtype(), &bytes)
}

/// Step 3 of this session's brief (QKV/gate-up fusion): loads two
/// same-shape, same-dtype matmul weights (`ffn_gate.weight`/
/// `ffn_up.weight`) and concatenates their raw on-disk bytes end to end
/// (tensor `a`'s rows first, then tensor `b`'s) before repacking, producing
/// one `MatMulWeight` covering both. This is valid because a Q4_0/Q8_0
/// tensor's bytes are already row-major, block-contiguous per output row
/// (see `quant.rs::chunk_rows`'s doc comment): concatenating two tensors'
/// byte streams with identical `bytes_per_row` (guaranteed here by the
/// `in_dim`/dtype equality check) is bit-identical to loading one tensor
/// whose rows are `a`'s rows followed by `b`'s - no dequantize/requantize
/// round trip, no reshuffling beyond the byte-level `extend_from_slice`.
/// One `linear()` call against the result produces both tensors' outputs
/// in one dispatch instead of two; downstream (`silu_mul_fused.wgsl`)
/// reads the fused `[rows, 2*out_dim_each]` layout directly.
fn gguf_matmul_concat2<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name_a: &str, name_b: &str, label: &str) -> Result<MatMulWeight> {
    let info_a = reader.tensor_info(name_a).with_context(|| format!("missing tensor {name_a}"))?.clone();
    let info_b = reader.tensor_info(name_b).with_context(|| format!("missing tensor {name_b}"))?.clone();
    let (shape_a, shape_b) = (info_a.shape(), info_b.shape());
    anyhow::ensure!(info_a.dtype() == info_b.dtype(), "gguf_matmul_concat2: {name_a} dtype {:?} != {name_b} dtype {:?}", info_a.dtype(), info_b.dtype());
    anyhow::ensure!(shape_a[1] == shape_b[1], "gguf_matmul_concat2: {name_a} in_dim {} != {name_b} in_dim {}", shape_a[1], shape_b[1]);
    let mut bytes = reader.tensor_data(name_a)?;
    bytes.extend_from_slice(&reader.tensor_data(name_b)?);
    let shape = vec![shape_a[0] + shape_b[0], shape_a[1]];
    load_matmul_weight_gguf(engine, label, &shape, info_a.dtype(), &bytes)
}

/// Inverse of llama.cpp `convert_hf_to_gguf.py`'s `LlamaModel.permute()`,
/// applied row-wise (dtype-agnostic: every row of a GGUF matmul weight is
/// the same number of bytes regardless of quant format, so this reorders
/// whole `bytes_per_row` chunks, never touching intra-row block encoding).
/// See `config.rs`'s module doc comment for the derivation and the
/// empirical check against SmolLM2-360M-Instruct's own GGUF. Applied only
/// to `attn_q.weight` (`n_heads = config.num_heads`) and `attn_k.weight`
/// (`n_heads = config.num_kv_heads`) for `Architecture::Llama`; a no-op for
/// Qwen2/Qwen3, which are never passed through this function.
///
/// `pub(crate)` so `cpu.rs`'s CPU rung can apply the exact same row
/// reordering to `attn_q.weight`/`attn_k.weight` before building its
/// `CpuWeight` - the GPU and CPU paths must agree on RoPE row order for a
/// Llama-architecture GGUF, or the two rungs diverge (see
/// `cpu.rs::gguf_weight_qk`).
pub(crate) fn unpermute_rope_rows(bytes: &[u8], n_heads: usize, head_dim: usize, out_dim: usize) -> Vec<u8> {
    assert_eq!(bytes.len() % out_dim, 0, "unpermute_rope_rows: bytes.len() not a multiple of out_dim");
    assert_eq!(n_heads * head_dim, out_dim, "unpermute_rope_rows: n_heads * head_dim != out_dim");
    let bytes_per_row = bytes.len() / out_dim;
    let half = head_dim / 2;
    let mut out = vec![0u8; bytes.len()];
    for h in 0..n_heads {
        let base = h * head_dim;
        for row in 0..head_dim {
            let p = if row < half { 2 * row } else { 2 * (row - half) + 1 };
            let src = (base + p) * bytes_per_row;
            let dst = (base + row) * bytes_per_row;
            out[dst..dst + bytes_per_row].copy_from_slice(&bytes[src..src + bytes_per_row]);
        }
    }
    out
}

/// Step 2 of this session's brief (QKV fusion): concatenates
/// `attn_q.weight`, `attn_k.weight` and `attn_v.weight`'s raw on-disk bytes
/// end to end (q's rows, then k's, then v's), same reasoning as
/// `gguf_matmul_concat2` generalized to three tensors. `attn_q`/`attn_k` are
/// un-permuted first when `config.architecture == Architecture::Llama`
/// (`unpermute_rope_rows`, each with its own head count - exactly what
/// `gguf_matmul_qk` does per tensor) so the concatenated bytes are already in
/// RoPE-ready row order before `load_matmul_weight_gguf` repacks them; `v` is
/// never permuted (`gguf_matmul_qk` is never called for it either). Returns
/// `None` when the three tensors don't share a dtype - every GGUF this crate
/// has loaded so far quantizes all attn weights uniformly, so this is a
/// defensive fallback, not an observed case: callers (`qkv_proj`) fall back
/// to three separate matmuls when it fires.
fn gguf_matmul_qkv_fused<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, config: &Qwen2Config, p: &str) -> Result<Option<MatMulWeight>> {
    let (name_q, name_k, name_v) = (format!("{p}.attn_q.weight"), format!("{p}.attn_k.weight"), format!("{p}.attn_v.weight"));
    let info_q = reader.tensor_info(&name_q).with_context(|| format!("missing tensor {name_q}"))?.clone();
    let info_k = reader.tensor_info(&name_k).with_context(|| format!("missing tensor {name_k}"))?.clone();
    let info_v = reader.tensor_info(&name_v).with_context(|| format!("missing tensor {name_v}"))?.clone();
    if info_q.dtype() != info_k.dtype() || info_k.dtype() != info_v.dtype() {
        return Ok(None);
    }
    let (shape_q, shape_k, shape_v) = (info_q.shape(), info_k.shape(), info_v.shape());
    if shape_q[1] != shape_k[1] || shape_k[1] != shape_v[1] {
        return Ok(None);
    }
    let is_llama = config.architecture == Architecture::Llama;
    let bytes_q = reader.tensor_data(&name_q)?;
    let bytes_q = if is_llama { unpermute_rope_rows(&bytes_q, config.num_heads, config.head_dim, shape_q[0]) } else { bytes_q };
    let bytes_k = reader.tensor_data(&name_k)?;
    let bytes_k = if is_llama { unpermute_rope_rows(&bytes_k, config.num_kv_heads, config.head_dim, shape_k[0]) } else { bytes_k };
    let bytes_v = reader.tensor_data(&name_v)?;
    let mut bytes = bytes_q;
    bytes.extend_from_slice(&bytes_k);
    bytes.extend_from_slice(&bytes_v);
    let shape = vec![shape_q[0] + shape_k[0] + shape_v[0], shape_q[1]];
    Ok(Some(load_matmul_weight_gguf(engine, &format!("{p}.attn_qkv_fused.weight"), &shape, info_q.dtype(), &bytes)?))
}

/// Like [`gguf_matmul`], but un-permutes RoPE row order first when
/// `config.architecture == Architecture::Llama` (see
/// [`unpermute_rope_rows`]). `n_heads` is the tensor's own head count:
/// `config.num_heads` for `attn_q.weight`, `config.num_kv_heads` for
/// `attn_k.weight`.
fn gguf_matmul_qk<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str, config: &Qwen2Config, n_heads: usize) -> Result<MatMulWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let bytes = reader.tensor_data(name)?;
    let bytes = if config.architecture == Architecture::Llama { unpermute_rope_rows(&bytes, n_heads, config.head_dim, shape[0]) } else { bytes };
    load_matmul_weight_gguf(engine, name, &shape, info.dtype(), &bytes)
}

impl GpuModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(engine: &Engine, gguf_path: &str, fast_kernels: bool) -> Result<Self> {
        let file = File::open(gguf_path).with_context(|| format!("opening {gguf_path}"))?;
        Self::load_from_reader(engine, BufReader::new(file), fast_kernels)
    }

    /// Same loading logic as [`GpuModel::load`], generic over any
    /// `Read + Seek` - the browser surface (`web.rs`) calls this with a
    /// `std::io::Cursor` over the whole GGUF file's bytes (fetched by JS and
    /// handed across the wasm boundary as one `Vec<u8>`; this model's
    /// Q4_0 GGUF is ~350MB, comfortably under both the 2GB single -
    /// `ArrayBuffer` limit and wasm32's 4GB address space, so no sharded
    /// reader is needed here - see `gguf.rs`'s module doc comment on that).
    pub fn load_from_reader<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: R, fast_kernels: bool) -> Result<Self> {
        let mut reader = GgufReader::open(reader)?;
        let config = config_from_gguf(&reader)?;

        let embed_info = reader.tensor_info("token_embd.weight").context("missing token_embd.weight")?.clone();
        let embed_shape = embed_info.shape();
        let embed_dtype = embed_info.dtype();
        anyhow::ensure!(
            matches!(embed_dtype, GgmlDtype::Q4_0 | GgmlDtype::Q8_0 | GgmlDtype::Q6_K),
            "expected token_embd.weight to be Q4_0, Q8_0 or Q6_K, got {embed_dtype:?}"
        );
        let embed_bytes = reader.tensor_data("token_embd.weight")?;
        // Tied embeddings (`config.tied_embeddings`, e.g. Qwen3-0.6B/1.7B):
        // the lm head is built from the *same* bytes, straight through
        // `load_matmul_weight_gguf`'s row-chunking path (so it still splits
        // across bindings under the device's storage-buffer-binding limit
        // exactly like an untied `output.weight` would) - built before the
        // embedding-gather table below so both share `embed_bytes` while
        // it's still resident, then it's dropped once, not read twice from
        // disk.
        let tied_lm_head = config.tied_embeddings.then(|| load_matmul_weight_gguf(engine, "token_embd(lm_head)", &embed_shape, embed_dtype, &embed_bytes)).transpose()?;
        let embed = load_embedding_table_gguf(engine, "token_embd", &embed_shape, embed_dtype, &embed_bytes);
        drop(embed_bytes);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{i}");
            let q_dim = (config.num_heads * config.head_dim) as usize;
            let kv_dim = (config.num_kv_heads * config.head_dim) as usize;
            let (q_b, k_b, v_b, qkv_b_data) = if config.has_qkv_bias {
                let q_data = gguf_f32_raw(&mut reader, &format!("{p}.attn_q.bias"))?;
                let k_data = gguf_f32_raw(&mut reader, &format!("{p}.attn_k.bias"))?;
                let v_data = gguf_f32_raw(&mut reader, &format!("{p}.attn_v.bias"))?;
                let mut fused = q_data.clone();
                fused.extend_from_slice(&k_data);
                fused.extend_from_slice(&v_data);
                (engine.buf_f32(&q_data, "attn_q.bias"), engine.buf_f32(&k_data, "attn_k.bias"), engine.buf_f32(&v_data, "attn_v.bias"), fused)
            } else {
                (engine.buf_f32(&vec![0f32; q_dim], "q_b_zero"), engine.buf_f32(&vec![0f32; kv_dim], "k_b_zero"), engine.buf_f32(&vec![0f32; kv_dim], "v_b_zero"), vec![0f32; q_dim + 2 * kv_dim])
            };
            let qkv_w = gguf_matmul_qkv_fused(engine, &mut reader, &config, &p)?;
            let qkv_b = qkv_w.is_some().then(|| engine.buf_f32(&qkv_b_data, "qkv_b_fused"));
            let (q_norm, k_norm) = if config.qk_norm {
                (Some(gguf_f32(engine, &mut reader, &format!("{p}.attn_q_norm.weight"))?), Some(gguf_f32(engine, &mut reader, &format!("{p}.attn_k_norm.weight"))?))
            } else {
                (None, None)
            };
            layers.push(LayerWeights {
                attn_norm: gguf_f32(engine, &mut reader, &format!("{p}.attn_norm.weight"))?,
                q_w: gguf_matmul_qk(engine, &mut reader, &format!("{p}.attn_q.weight"), &config, config.num_heads)?,
                q_b,
                k_w: gguf_matmul_qk(engine, &mut reader, &format!("{p}.attn_k.weight"), &config, config.num_kv_heads)?,
                k_b,
                v_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_v.weight"))?,
                v_b,
                o_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_output.weight"))?,
                o_b: engine.buf_f32(&vec![0f32; config.hidden_size], "o_b_zero"),
                ffn_norm: gguf_f32(engine, &mut reader, &format!("{p}.ffn_norm.weight"))?,
                gate_up_w: gguf_matmul_concat2(engine, &mut reader, &format!("{p}.ffn_gate.weight"), &format!("{p}.ffn_up.weight"), &format!("{p}.ffn_gate_up.weight"))?,
                gate_up_b: engine.buf_f32(&vec![0f32; 2 * config.intermediate_size], "gate_up_b_zero"),
                down_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_down.weight"))?,
                down_b: engine.buf_f32(&vec![0f32; config.hidden_size], "down_b_zero"),
                q_norm,
                k_norm,
                qkv_w,
                qkv_b,
            });
        }

        let out_norm = gguf_f32(engine, &mut reader, "output_norm.weight")?;
        let lm_head = match tied_lm_head {
            Some(w) => w,
            None => gguf_matmul(engine, &mut reader, "output.weight")?,
        };
        let zero_bias_vocab = engine.buf_f32(&vec![0f32; config.vocab_size], "zero_bias_vocab");

        // `reader` (and its underlying `File`) drops here - two-phase
        // loading: no raw GGUF bytes remain in memory past this point,
        // only the GPU-resident buffers built above.
        drop(reader);

        Ok(GpuModel {
            config,
            embed,
            layers,
            out_norm,
            lm_head,
            zero_bias_vocab,
            fast_kernels,
            pool: Pool::new(engine.device.clone(), engine.queue.clone()),
            lora: None,
        })
    }

    /// Loads and applies a runtime LoRA adapter (LLMLIFE2 format, see
    /// `lora.rs`), replacing any adapter loaded earlier: adapters don't
    /// stack. Does not touch the base weights.
    pub fn apply_lora(&mut self, engine: &Engine, bytes: &[u8]) -> Result<()> {
        let adapter = crate::lora::LoraAdapter::from_bytes(engine, bytes, self.config.num_layers)?;
        self.lora = Some(adapter);
        Ok(())
    }

    /// Removes the currently-applied LoRA adapter, if any: subsequent
    /// forward calls run the frozen base only, with no base reload.
    pub fn clear_lora(&mut self) {
        self.lora = None;
    }

    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    /// Sum of every persistent weight buffer's GPU size (embed table,
    /// per-layer norms/matmuls, lm head) - the model's fixed GPU residency,
    /// independent of `kv_len`/context. Does not include `KvCache` (see
    /// `KvCache::gpu_bytes`) or `Pool`'s per-forward scratch (see
    /// `Pool::resident_bytes`) - a caller wanting total GPU footprint sums
    /// all three. Added for this session's browser memory investigation
    /// (docs/runs/2026-09-28-lean-decode-breakdown.md).
    pub fn weight_gpu_bytes(&self) -> u64 {
        let mut total = self.embed.gpu_bytes() + self.out_norm.size() + self.lm_head.gpu_bytes() + self.zero_bias_vocab.size();
        for l in &self.layers {
            total += l.attn_norm.size()
                + l.q_w.gpu_bytes()
                + l.q_b.size()
                + l.k_w.gpu_bytes()
                + l.k_b.size()
                + l.v_w.gpu_bytes()
                + l.v_b.size()
                + l.o_w.gpu_bytes()
                + l.o_b.size()
                + l.ffn_norm.size()
                + l.gate_up_w.gpu_bytes()
                + l.gate_up_b.size()
                + l.down_w.gpu_bytes()
                + l.down_b.size();
            if let Some(b) = &l.q_norm {
                total += b.size();
            }
            if let Some(b) = &l.k_norm {
                total += b.size();
            }
        }
        total
    }

    /// Logits for a caller-chosen subset of vocab ids, at every row of
    /// `hidden_states`: the sliced lm-head mechanism (this crate's
    /// consumer survey, gap #5): llm-life reads exactly `[dead, alive]`
    /// logits at every cell's answer position instead of materializing a
    /// `[rows, vocab_size]` buffer. Returns `[rows, token_ids.len()]`,
    /// row-major, already read back to the CPU.
    pub async fn lm_head_sliced(&self, engine: &Engine, hidden_states: &wgpu::Buffer, rows: u32, token_ids: &[u32]) -> Vec<f32> {
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lm_head_sliced") });
        let n = token_ids.len() as u32;
        let mut pass = engine.begin_pass(&mut encoder, "lm_head_sliced");
        let (gathered, hidden_dim) = gather_dequant_head_rows(engine, &self.pool, &mut pass, "head_slice", &self.lm_head, token_ids);
        let w = MatMulWeight::F32 { w: gathered };
        let zero_b = zero_bias(engine, &self.pool, "head_slice.bias", n);
        let logits = linear(engine, &self.pool, &mut pass, "head_slice.linear", hidden_states, rows, hidden_dim, &w, &zero_b, n, false);
        drop(pass);
        engine.queue.submit(Some(encoder.finish()));
        engine.read_buffer(&logits, (rows * n) as usize).await
    }
}

/// RoPE cos/sin tables for absolute positions `[0, max_pos)`, `half =
/// head_dim/2` columns each. `inv_freq[j] = theta^(-2j/head_dim)` - HF
/// Qwen2RotaryEmbedding's convention (`gen_fixture.py`'s reference reads the
/// same `qwen2.rope.freq_base` metadata key).
pub fn build_rope_tables(head_dim: usize, theta: f32, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|j| theta.powf(-2.0 * j as f32 / head_dim as f32)).collect();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for j in 0..half {
            let angle = pos as f32 * inv_freq[j];
            cos[pos * half + j] = angle.cos();
            sin[pos * half + j] = angle.sin();
        }
    }
    (cos, sin)
}

/// Maps a per-layer call-site key (`"layer5.mlp.gate_up"`,
/// `"dec_layer12.attn"`) onto the SAME string for every layer
/// (`"layer.mlp.gate_up"`, `"dec_layer.attn"`) by stripping the trailing
/// digits off the key's first `.`-delimited segment. Used ONLY for
/// `Pool::data`/`Pool::uniform`/`Pool::upload_*` calls (the actual
/// scratch-buffer allocations), never for `Pool::bind_group` calls, which
/// must stay keyed per layer (each layer's bind group references that
/// layer's own weight tensor - a different GPU buffer every layer - so a
/// shared bind-group cache entry would silently keep pointing at whichever
/// layer built it first).
///
/// Why sharing the underlying buffer across layers is safe: every scratch
/// buffer this maps (rmsnorm's output, q/k/v, attention output, the fused
/// gate/up matmul's output, etc.) is written by one dispatch and consumed
/// by the next within the SAME layer's own dispatch chain, then dead -
/// nothing downstream of layer `i` ever reads layer `i`'s copy of e.g.
/// "layer.mlp.gate_up.out" after layer `i`'s own `silu_mul_fused` call
/// consumes it. `forward_prefill`/`decode_layers`/`forward_chunk_spec`
/// record every layer's dispatches into the SAME open `wgpu::ComputePass`
/// (or the same command encoder) in program order, and a WebGPU/wgpu
/// compute pass guarantees a later dispatch observes an earlier one's
/// writes to a resource it reads or writes - the same guarantee every
/// intra-layer chain here already depends on (rmsnorm's output feeding
/// straight into qkv's input, RoPE mutating q/k in place, etc.). So layer
/// `i+1`'s write to the shared buffer cannot execute (in program order)
/// until after layer `i`'s last read of it has already been recorded and
/// therefore already executes first.
///
/// Root cause this fixes: before this function existed, every layer got
/// its OWN distinct `Pool`-cached buffer for these call sites (Pool is
/// grow-only and never frees - see pool.rs's module doc), so one prefill
/// over a long prompt permanently pinned `num_layers` copies of every
/// per-layer scratch buffer, sized to that prompt's length, for the rest
/// of the page's life. Measured on Qwen2.5-0.5B-Instruct (24 layers,
/// hidden 896): `Pool::resident_bytes()` after a 2225-token prefill was
/// 5.69GB, of which the great majority is these per-layer duplicates (a
/// single layer's worth of scratch at that shape is a small fraction of
/// that) - see docs/runs/2026-09-28-lean-decode-breakdown.md's memory
/// section. On the far larger Qwen2.5-3B (36 layers, hidden 2048,
/// intermediate 11008) this is the dominant cause of the multi-GB browser
/// renderer footprint that produced swap thrashing during this session's
/// long-context timing runs - not a kernel or dispatch-count regression.
fn scratch_key(key: &str) -> std::borrow::Cow<'_, str> {
    let (head, rest) = match key.split_once('.') {
        Some((h, r)) => (h, Some(r)),
        None => (key, None),
    };
    let stripped = head.trim_end_matches(|c: char| c.is_ascii_digit());
    if stripped == head {
        return std::borrow::Cow::Borrowed(key);
    }
    match rest {
        Some(r) => std::borrow::Cow::Owned(format!("{stripped}.{r}")),
        None => std::borrow::Cow::Owned(stripped.to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn linear(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, x: &wgpu::Buffer, rows: u32, in_dim: u32, w: &MatMulWeight, b: &wgpu::Buffer, out_dim: u32, fast: bool) -> wgpu::Buffer {
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (rows * out_dim) as usize);
    let wgs = (out_dim.div_ceil(16), rows.div_ceil(16), 1);
    match w {
        MatMulWeight::F32 { w } => {
            let dims = pool.uniform(&format!("{skey}.dims"), LinearDims { m: rows, k: in_dim, n: out_dim, act: 0 });
            let entries = [
                BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ];
            if fast && rows == 1 {
                // Decode: coalesced matvec (see linear_f32_decode.wgsl's
                // header) - some GGUFs keep a handful of tensors at F32
                // residency even in an otherwise-quantized file, which
                // previously hit the naive per-output-element kernel below.
                let bg = pool.bind_group(&format!("{key}.decode"), &engine.linear_f32_decode, &entries);
                engine.dispatch(pass, &engine.linear_f32_decode, &bg, (out_dim.div_ceil(4), 1, 1), key);
            } else {
                let bg = pool.bind_group(key, &engine.linear, &entries);
                engine.dispatch(pass, &engine.linear, &bg, wgs, key);
            }
        }
        MatMulWeight::Q8_0 { chunks, blocks_per_row, out_dim: n_total } => {
            // Q8_0-resident weights appear on every tensor for Qwen3's
            // official GGUFs (no Q4_0 quant published - qwen3 survey), not
            // just the lm head, so decode (rows == 1) gets the same
            // coalesced-matvec treatment as Q4_0 below. `chunks` exists
            // because `output.weight`'s Q8_0 qs buffer (~130MB) can exceed
            // a browser's storage-buffer-binding limit (see
            // `quant.rs::chunk_rows`) - one dispatch per chunk, each
            // writing its own column range of `out` (see linear_q8.wgsl's
            // Dims doc comment).
            for chunk in chunks {
                let ckey = format!("{key}.{}", chunk.row_start);
                let sckey = scratch_key(&ckey);
                let dims = pool.uniform(&format!("{sckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
                let entries = [
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: chunk.qs.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: chunk.scales.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
                ];
                if fast && rows == 1 {
                    // Decode: coalesced matvec, same kernel shape as
                    // linear_q4_decode below, adapted to Q8_0 blocks.
                    let bg = pool.bind_group(&format!("{ckey}.decode"), &engine.linear_q8_decode, &entries);
                    engine.dispatch(pass, &engine.linear_q8_decode, &bg, (chunk.rows.div_ceil(16), 1, 1), &ckey);
                } else if fast && rows < SMALL_M_MAX_ROWS {
                    // Short prefill: same small-M kernel and row rule as
                    // Q4_0 (linear_q4_small_m.wgsl with its Q8 override).
                    let bg = pool.bind_group(&format!("{ckey}.small_m"), &engine.linear_q8_small_m, &entries);
                    engine.dispatch(pass, &engine.linear_q8_small_m, &bg, (chunk.rows.div_ceil(32), rows.div_ceil(8), 1), &ckey);
                } else if fast {
                    // Prefill, M >= SMALL_M_MAX_ROWS: the 64x64 tiled kernel
                    // (linear_q4_tiled_rb.wgsl with its Q8 override).
                    let bg = pool.bind_group(&format!("{ckey}.tiled_rb"), &engine.linear_q8_tiled_rb, &entries);
                    engine.dispatch(pass, &engine.linear_q8_tiled_rb, &bg, (chunk.rows.div_ceil(64), rows.div_ceil(64), 1), &ckey);
                } else {
                    let bg = pool.bind_group(&ckey, &engine.linear_q8, &entries);
                    engine.dispatch(pass, &engine.linear_q8, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                }
            }
        }
        MatMulWeight::Q6_K { chunks, blocks_per_row, out_dim: n_total } => {
            // Q6_K is only ever token_embd/output.weight in this crate's
            // models, one matmul per forward - but decode (rows == 1) still
            // gets the coalesced matvec below (see linear_q6k_decode.wgsl's
            // header): the naive kernel below stayed at 7.6% of peak
            // bandwidth on the official Qwen2.5-3B GGUF's Q6_K lm head
            // (docs/runs/2026-09-29-lean-vs-llamacpp-profile.md).
            for chunk in chunks {
                let ckey = format!("{key}.{}", chunk.row_start);
                let sckey = scratch_key(&ckey);
                let dims = pool.uniform(&format!("{sckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
                let entries = [
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: chunk.ql.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: chunk.qh.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: chunk.scales.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: chunk.d.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 6, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 7, resource: dims.as_entire_binding() },
                ];
                if fast && rows == 1 {
                    let bg = pool.bind_group(&format!("{ckey}.decode"), &engine.linear_q6k_decode, &entries);
                    engine.dispatch(pass, &engine.linear_q6k_decode, &bg, (chunk.rows.div_ceil(4), 1, 1), &ckey);
                } else {
                    let bg = pool.bind_group(&ckey, &engine.linear_q6k, &entries);
                    engine.dispatch(pass, &engine.linear_q6k, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                }
            }
        }
        MatMulWeight::Q4_0 { chunks, blocks_per_row, out_dim: n_total } => {
            for chunk in chunks {
                let ckey = format!("{key}.{}", chunk.row_start);
                let sckey = scratch_key(&ckey);
                let dims = pool.uniform(&format!("{sckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
                let entries = [
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: chunk.qs.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: chunk.scales.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
                ];
                if !fast {
                    // Reference/naive path - always linear_q4.wgsl regardless
                    // of `rows`, kept for the fixture-parity gate and as the
                    // fallback when a fast kernel's correctness is in doubt.
                    let bg = pool.bind_group(&ckey, &engine.linear_q4, &entries);
                    engine.dispatch(pass, &engine.linear_q4, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                } else if rows == 1 {
                    // Decode: coalesced matvec (llm-wasm's shader_q4_matvec_coalesced.wgsl port).
                    let bg = pool.bind_group(&format!("{ckey}.decode"), &engine.linear_q4_decode, &entries);
                    engine.dispatch(pass, &engine.linear_q4_decode, &bg, (chunk.rows.div_ceil(16), 1, 1), &ckey);
                } else if rows < SMALL_M_MAX_ROWS {
                    // Short prefill: weight words read and dequantised once
                    // per group of 8 query rows (see linear_q4_small_m.wgsl).
                    let bg = pool.bind_group(&format!("{ckey}.small_m"), &engine.linear_q4_small_m, &entries);
                    engine.dispatch(pass, &engine.linear_q4_small_m, &bg, (chunk.rows.div_ceil(32), rows.div_ceil(8), 1), &ckey);
                } else {
                    // Prefill, M >= SMALL_M_MAX_ROWS: 64x64 register-blocked
                    // tiled kernel (see linear_q4_tiled_rb.wgsl's header).
                    let bg = pool.bind_group(&format!("{ckey}.tiled_rb"), &engine.linear_q4_tiled_rb, &entries);
                    engine.dispatch(pass, &engine.linear_q4_tiled_rb, &bg, (chunk.rows.div_ceil(64), rows.div_ceil(64), 1), &ckey);
                }
            }
        }
    }
    out
}

/// Decode's fused gate/up matvec with silu(gate) * up in the same dispatch
/// (`linear_q4_decode_swiglu`), returning the `inter`-long gated
/// activations. `None` when the weight is not one Q4_0 binding (or fast
/// kernels are off): the caller then runs `linear` + `silu_mul_fused`.
#[allow(clippy::too_many_arguments)]
fn gate_up_swiglu_decode(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, x: &wgpu::Buffer, in_dim: u32, w: &MatMulWeight, b: &wgpu::Buffer, inter: u32, fast: bool) -> Option<wgpu::Buffer> {
    let MatMulWeight::Q4_0 { chunks, blocks_per_row, out_dim } = w else { return None };
    if !fast || chunks.len() != 1 || *out_dim != 2 * inter {
        return None;
    }
    let chunk = &chunks[0];
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.swiglu"), inter as usize);
    let dims = pool.uniform(&format!("{skey}.swiglu_dims"), LinearQDims { m: 1, k: in_dim, n: 2 * inter, act: 0, blocks_per_row: *blocks_per_row, n_offset: 0, n_total: 2 * inter, _p2: 0 });
    let bg = pool.bind_group(
        &format!("{key}.swiglu"),
        &engine.linear_q4_decode_swiglu,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: chunk.qs.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: chunk.scales.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.linear_q4_decode_swiglu, &bg, (inter.div_ceil(8), 1, 1), key);
    Some(out)
}

/// A pool-owned all-zero bias buffer of `len` f32s. LoRA's two internal
/// matmuls (`x -> rank`, `rank -> out`) have no bias of their own :
/// `linear()` always takes one, so this is cheaper than adding a
/// bias-optional code path to that shared helper.
fn zero_bias(engine: &Engine, pool: &Pool, key: &str, len: u32) -> wgpu::Buffer {
    let buf = pool.data(key, len as usize);
    engine.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&vec![0f32; len as usize]));
    buf
}

/// Adds one LoRA projection's delta onto `out_buf` in place: `out_buf +=
/// (x @ proj.a) @ proj.b` (the `alpha/rank` scale is already folded into
/// `proj.b` at upload time: see `lora.rs`). Two plain F32 `linear()` calls
/// plus one `add_inplace`, no dedicated LoRA kernel.
#[allow(clippy::too_many_arguments)]
fn apply_lora_proj(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, x: &wgpu::Buffer, rows: u32, in_dim: u32, out_buf: &wgpu::Buffer, out_dim: u32, proj: &crate::lora::LoraProj) {
    let zero_r = zero_bias(engine, pool, &format!("{key}.zr"), proj.rank);
    let ab = linear(engine, pool, pass, &format!("{key}.a"), x, rows, in_dim, &proj.a, &zero_r, proj.rank, false);
    let zero_o = zero_bias(engine, pool, &format!("{key}.zo"), out_dim);
    let delta = linear(engine, pool, pass, &format!("{key}.b"), &ab, rows, proj.rank, &proj.b, &zero_o, out_dim, false);
    add_inplace(engine, pool, pass, &format!("{key}.add"), out_buf, &delta, rows * out_dim);
}

/// `linear()` plus, when `lora` is `Some`, that projection's LoRA delta
/// added in place onto the result: the single call site every q/k/v/o
/// projection in this file goes through, so LoRA is applied uniformly
/// across prefill, decode and the chunked/masked forward path.
#[allow(clippy::too_many_arguments)]
fn linear_lora(
    engine: &Engine,
    pool: &Pool,
    pass: &mut wgpu::ComputePass<'_>,
    key: &str,
    x: &wgpu::Buffer,
    rows: u32,
    in_dim: u32,
    w: &MatMulWeight,
    b: &wgpu::Buffer,
    out_dim: u32,
    fast: bool,
    lora: Option<&crate::lora::LoraProj>,
) -> wgpu::Buffer {
    let out = linear(engine, pool, pass, key, x, rows, in_dim, w, b, out_dim, fast);
    if let Some(proj) = lora {
        apply_lora_proj(engine, pool, pass, &format!("{key}.lora"), x, rows, in_dim, &out, out_dim, proj);
    }
    out
}

/// A view into a `wgpu::Buffer`: either the whole buffer (`From<&wgpu::Buffer>`,
/// `binding()` is then exactly the `as_entire_binding()` every call site used
/// before this type existed) or an offset+length sub-range (`BufView::slice`).
/// Lets `rmsnorm`/`rope`/`attn_decode` read q/k/v straight out of
/// `qkv_proj`'s fused matmul output (see `QkvSlot`, `LayerWeights::qkv_w`)
/// through an offset binding instead of three separate buffers - no extra
/// dispatch, no copy. A storage-buffer binding's declared `size` bounds the
/// shader's `arrayLength()`/index range to that sub-range (standard
/// WebGPU/wgpu behavior), so this is safe for read-write kernels like `rope`
/// that mutate their input in place: the mutation only ever touches the
/// bound sub-range.
#[derive(Clone, Copy)]
struct BufView<'a> {
    buffer: &'a wgpu::Buffer,
    elem_offset: u32,
    elem_len: Option<u32>,
}

impl<'a> From<&'a wgpu::Buffer> for BufView<'a> {
    fn from(buffer: &'a wgpu::Buffer) -> Self {
        BufView { buffer, elem_offset: 0, elem_len: None }
    }
}

impl<'a> BufView<'a> {
    fn slice(buffer: &'a wgpu::Buffer, elem_offset: u32, elem_len: u32) -> Self {
        BufView { buffer, elem_offset, elem_len: Some(elem_len) }
    }
    fn binding(&self) -> wgpu::BindingResource<'a> {
        if self.elem_offset == 0 && self.elem_len.is_none() {
            return self.buffer.as_entire_binding();
        }
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: (self.elem_offset as u64) * 4,
            size: self.elem_len.map(|n| wgpu::BufferSize::new((n as u64) * 4).expect("BufView::slice: elem_len must be nonzero")),
        })
    }
}

/// `qkv_proj`'s per-tensor result: either its own whole buffer (the ordinary
/// three-separate-matmul path, `Whole`, byte-identical to before this type
/// existed) or a view into the one buffer `qkv_proj`'s fused decode path
/// produced (`View` - see `LayerWeights::qkv_w`'s doc comment). `&QkvSlot`
/// converts to a `BufView` so every downstream call site
/// (`rmsnorm`/`rope`/`scatter_kv_gpu`/`attn_decode`) is unchanged syntax
/// whichever variant it's holding.
enum QkvSlot {
    Whole(wgpu::Buffer),
    View { buf: wgpu::Buffer, elem_offset: u32, elem_len: u32 },
}

impl<'a> From<&'a QkvSlot> for BufView<'a> {
    fn from(slot: &'a QkvSlot) -> Self {
        match slot {
            QkvSlot::Whole(b) => BufView::from(b),
            QkvSlot::View { buf, elem_offset, elem_len } => BufView::slice(buf, *elem_offset, *elem_len),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, x: impl Into<BufView<'a>>, scale: &wgpu::Buffer, rows: u32, dim: u32, eps: f32) -> wgpu::Buffer {
    let x = x.into();
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (rows * dim) as usize);
    let dims = pool.uniform(&format!("{skey}.dims"), RmsDims { rows, dim, eps, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rmsnorm,
        &[
            BindGroupEntry { binding: 0, resource: x.binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    // One workgroup per row (see shaders/rmsnorm.wgsl's header comment).
    engine.dispatch(pass, &engine.rmsnorm, &bg, (rows, 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn rope<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, buf: impl Into<BufView<'a>>, cos: &wgpu::Buffer, sin: &wgpu::Buffer, rows: u32, heads: u32, head_dim: u32, pos_base: u32) {
    let buf = buf.into();
    let skey = scratch_key(key);
    let dims = pool.uniform(&format!("{skey}.dims"), RopeDims { rows, heads, head_dim, pos_base });
    let bg = pool.bind_group(
        key,
        &engine.rope,
        &[
            BindGroupEntry { binding: 0, resource: buf.binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(pass, &engine.rope, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

/// Same RoPE as `rope()` but each row's absolute position comes from a
/// caller-supplied buffer (`ForwardSpec::positions`) instead of a
/// contiguous `pos_base + row` run: see `shaders/rope_positions.wgsl`.
/// Decode (one row): RoPE on q and k plus the K/V cache write at `pos`, one
/// dispatch (see `rope_kv_decode.wgsl`). Returns the rotated q.
#[allow(clippy::too_many_arguments)]
fn rope_kv_decode<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, q: impl Into<BufView<'a>>, k: impl Into<BufView<'a>>, v: impl Into<BufView<'a>>, cos: &wgpu::Buffer, sin: &wgpu::Buffer, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, pos: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let (q, k, v) = (q.into(), k.into(), v.into());
    let skey = scratch_key(key);
    let n_heads = cfg.num_heads as u32;
    let n_kv_heads = cfg.num_kv_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let q_out = pool.data(&format!("{skey}.q"), (n_heads * head_dim) as usize);
    let dims = pool.uniform(&format!("{skey}.dims"), RopeKvDims { n_heads, n_kv_heads, head_dim, pos, max_ctx, _p0: 0, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rope_kv_decode,
        &[
            BindGroupEntry { binding: 0, resource: q.binding() },
            BindGroupEntry { binding: 1, resource: k.binding() },
            BindGroupEntry { binding: 2, resource: v.binding() },
            BindGroupEntry { binding: 3, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: q_out.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 7, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 8, resource: dims.as_entire_binding() },
        ],
    );
    let total = (n_heads + n_kv_heads) * (head_dim / 2) + n_kv_heads * head_dim;
    engine.dispatch(pass, &engine.rope_kv_decode, &bg, (total.div_ceil(64), 1, 1), key);
    q_out
}

#[allow(clippy::too_many_arguments)]
fn rope_positions<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, buf: impl Into<BufView<'a>>, cos: &wgpu::Buffer, sin: &wgpu::Buffer, positions: &wgpu::Buffer, rows: u32, heads: u32, head_dim: u32) {
    let buf = buf.into();
    let skey = scratch_key(key);
    let dims = pool.uniform(&format!("{skey}.dims"), RopePosDims { rows, heads, head_dim, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rope_positions,
        &[
            BindGroupEntry { binding: 0, resource: buf.binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: positions.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(pass, &engine.rope_positions, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

/// Fused `add_inplace` (residual `a += delta`) + `rmsnorm` read of the
/// updated `a` - see `shaders/add_rmsnorm.wgsl`'s header. `a` is mutated in
/// place exactly as `add_inplace` would leave it (so later reads of `a`,
/// e.g. the next `add_rmsnorm`/`add_inplace` call, see the same value); the
/// normalized result is returned as a fresh buffer, exactly as `rmsnorm`
/// returns one. Decode-only (see call sites in `forward_decode_step*`) -
/// prefill's add+norm pairs stay unfused, already amortized across rows.
#[allow(clippy::too_many_arguments)]
fn add_rmsnorm(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, a: &wgpu::Buffer, delta: &wgpu::Buffer, scale: &wgpu::Buffer, rows: u32, dim: u32, eps: f32) -> wgpu::Buffer {
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (rows * dim) as usize);
    let dims = pool.uniform(&format!("{skey}.dims"), RmsDims { rows, dim, eps, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.add_rmsnorm,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: delta.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    // One workgroup per row (see shaders/add_rmsnorm.wgsl's header comment).
    engine.dispatch(pass, &engine.add_rmsnorm, &bg, (rows, 1, 1), key);
    out
}

fn add_inplace(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) {
    let skey = scratch_key(key);
    let (gx, gy, stride_x) = grid1d(len);
    let dims = pool.uniform(&format!("{skey}.dims"), AddDims { len, stride_x, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.add_inplace,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.add_inplace, &bg, (gx, gy, 1), key);
}

/// SwiGLU over a fused `[rows, 2*hidden]` gate/up matmul output (see
/// `gguf_matmul_concat2`'s doc comment and `silu_mul_fused.wgsl`).
fn silu_mul_fused(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, gate_up: &wgpu::Buffer, rows: u32, hidden: u32) -> wgpu::Buffer {
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (rows * hidden) as usize);
    let (gx, gy, stride_x) = grid1d(rows * hidden);
    let dims = pool.uniform(&format!("{skey}.dims"), SiluDims { rows, hidden, stride_x, _p1: 0 });
    let bg = pool.bind_group(
        key,
        &engine.silu_mul_fused,
        &[
            BindGroupEntry { binding: 0, resource: gate_up.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.silu_mul_fused, &bg, (gx, gy, 1), key);
    out
}

/// Step 2 of this session's brief (QKV fusion). `rows == 1` (decode) with no
/// LoRA adapter active and a fused weight available (`layer.qkv_w`) runs
/// **one** `linear()` dispatch against the concatenated q/k/v weight instead
/// of three, and returns `q`/`k`/`v` as `BufView`s into that single output
/// buffer (`QkvSlot::View`, offsets `0` / `q_dim` / `q_dim+kv_dim`) - no copy,
/// downstream (`rope`/`scatter_kv_gpu`/`attn_decode`) read the fused buffer
/// directly through those offsets. Every other case (`rows > 1` - prefill and
/// the chunked/masked forward path, which need q/k/v as separately
/// addressable `[rows, dim]` buffers and can't safely read a fused *strided*
/// per-row layout back with a plain offset+size binding; LoRA active; or
/// `layer.qkv_w` is `None`, i.e. `gguf_matmul_qkv_fused` found a dtype
/// mismatch) falls back to the original three-matmul path, `QkvSlot::Whole`,
/// byte-identical to this function before fusion existed.
#[allow(clippy::too_many_arguments)]
fn qkv_proj(
    engine: &Engine,
    pool: &Pool,
    pass: &mut wgpu::ComputePass<'_>,
    key: &str,
    x: &wgpu::Buffer,
    rows: u32,
    cfg: &Qwen2Config,
    layer: &LayerWeights,
    fast: bool,
    lora: Option<&crate::lora::LoraLayer>,
) -> (QkvSlot, QkvSlot, QkvSlot) {
    let hidden = cfg.hidden_size as u32;
    let q_dim = (cfg.num_heads * cfg.head_dim) as u32;
    let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u32;

    let (q, k, v) = if rows == 1 && lora.is_none() {
        if let (Some(qkv_w), Some(qkv_b)) = (&layer.qkv_w, &layer.qkv_b) {
            let qkv_out = linear(engine, pool, pass, &format!("{key}.qkv_fused"), x, rows, hidden, qkv_w, qkv_b, q_dim + 2 * kv_dim, fast);
            let q = QkvSlot::View { buf: qkv_out.clone(), elem_offset: 0, elem_len: q_dim };
            let k = QkvSlot::View { buf: qkv_out.clone(), elem_offset: q_dim, elem_len: kv_dim };
            let v = QkvSlot::View { buf: qkv_out, elem_offset: q_dim + kv_dim, elem_len: kv_dim };
            (q, k, v)
        } else {
            (
                QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.q"), x, rows, hidden, &layer.q_w, &layer.q_b, q_dim, fast, None)),
                QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.k"), x, rows, hidden, &layer.k_w, &layer.k_b, kv_dim, fast, None)),
                QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.v"), x, rows, hidden, &layer.v_w, &layer.v_b, kv_dim, fast, None)),
            )
        }
    } else {
        (
            QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.q"), x, rows, hidden, &layer.q_w, &layer.q_b, q_dim, fast, lora.map(|l| &l.q))),
            QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.k"), x, rows, hidden, &layer.k_w, &layer.k_b, kv_dim, fast, lora.map(|l| &l.k))),
            QkvSlot::Whole(linear_lora(engine, pool, pass, &format!("{key}.v"), x, rows, hidden, &layer.v_w, &layer.v_b, kv_dim, fast, lora.map(|l| &l.v))),
        )
    };

    // Qwen3 only (`layer.q_norm`/`k_norm` set): per-head RMSNorm on q/k
    // before RoPE. `q`/`k` are `[rows, heads*head_dim]` row-major, so
    // reinterpreting the same flat buffer as `[rows*heads, head_dim]` for
    // `rmsnorm()` normalizes each head's slice independently in place - no
    // dedicated per-head kernel needed (see the qwen3 survey's open
    // question, resolved this way). `rmsnorm()`'s output is always a fresh
    // whole buffer (`pool.data`), so a fused `View` slot correctly
    // "materializes" into `QkvSlot::Whole` here regardless of which branch
    // above produced it.
    let q = match &layer.q_norm {
        Some(scale) => QkvSlot::Whole(rmsnorm(engine, pool, pass, &format!("{key}.qnorm"), &q, scale, rows * cfg.num_heads as u32, cfg.head_dim as u32, cfg.rms_norm_eps)),
        None => q,
    };
    let k = match &layer.k_norm {
        Some(scale) => QkvSlot::Whole(rmsnorm(engine, pool, pass, &format!("{key}.knorm"), &k, scale, rows * cfg.num_kv_heads as u32, cfg.head_dim as u32, cfg.rms_norm_eps)),
        None => k,
    };
    (q, k, v)
}

#[allow(clippy::too_many_arguments)]
fn mlp(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, x: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, layer: &LayerWeights, fast: bool) -> wgpu::Buffer {
    let hidden = cfg.hidden_size as u32;
    let inter = cfg.intermediate_size as u32;
    // Fused gate/up matmul (see gguf_matmul_concat2's doc comment): one
    // linear() dispatch (per weight chunk) instead of two, at every M.
    let gate_up = linear(engine, pool, pass, &format!("{key}.gate_up"), x, rows, hidden, &layer.gate_up_w, &layer.gate_up_b, 2 * inter, fast);
    let gated = silu_mul_fused(engine, pool, pass, &format!("{key}.silu"), &gate_up, rows, inter);
    linear(engine, pool, pass, &format!("{key}.down"), &gated, rows, inter, &layer.down_w, &layer.down_b, hidden, fast)
}

fn embed_gather(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, model: &GpuModel, token_ids: &[u32]) -> wgpu::Buffer {
    let rows = token_ids.len() as u32;
    let hidden = model.config.hidden_size as u32;
    let ids_buf = pool.upload_u32("embed.ids", token_ids);
    let out = pool.data("embed.out", (rows * hidden) as usize);
    let (gx, gy, stride_x) = grid1d(rows * hidden);
    if let EmbeddingTable::Q6_K(t) = &model.embed {
        let dims = pool.uniform("embed.dims", GatherDims { rows, hidden, blocks_per_row: t.blocks_per_row, stride_x });
        let bg = pool.bind_group(
            "embed",
            &engine.embed_gather_q6k,
            &[
                BindGroupEntry { binding: 0, resource: ids_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: t.ql.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: t.qh.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: t.scales.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: t.d.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: dims.as_entire_binding() },
            ],
        );
        engine.dispatch(pass, &engine.embed_gather_q6k, &bg, (gx, gy, 1), "embed");
        return out;
    }
    let (pipeline, table) = match &model.embed {
        EmbeddingTable::Q4_0(t) => (&engine.embed_gather_q4, t),
        EmbeddingTable::Q8_0(t) => (&engine.embed_gather_q8, t),
        EmbeddingTable::Q6_K(_) => unreachable!("handled above"),
    };
    let dims = pool.uniform("embed.dims", GatherDims { rows, hidden, blocks_per_row: table.blocks_per_row, stride_x });
    let bg = pool.bind_group(
        "embed",
        pipeline,
        &[
            BindGroupEntry { binding: 0, resource: ids_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: table.qs.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: table.scales.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, pipeline, &bg, (gx, gy, 1), "embed");
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvScatterDims {
    rows: u32,
    kv_heads: u32,
    head_dim: u32,
    kv_base: u32,
    max_ctx: u32,
    src_offset: u32,
    stride_x: u32,
    _p0: u32,
}

/// GPU-side write of `rows` freshly computed K/V rows (`[rows, kv_heads,
/// head_dim]`, row-major) into the cache's `[kv_head, kv_base+row,
/// head_dim]` layout, as one compute dispatch inside the open pass
/// (`shaders/kv_scatter.wgsl`) instead of `rows * kv_heads` encoder-level
/// copies: the multi-row paths (prefill, chunk) use this; the one-row
/// decode path writes K/V in `rope_kv_decode` instead.
#[allow(clippy::too_many_arguments)]
fn scatter_kv_kernel<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, cache_buf: &wgpu::Buffer, src: impl Into<BufView<'a>>, rows: u32, cfg: &Qwen2Config, kv_base: u32, max_ctx: u32) {
    let src = src.into();
    let kv_heads = cfg.num_kv_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let (gx, gy, stride_x) = grid1d(rows * kv_heads * head_dim);
    let skey = scratch_key(key);
    let dims = pool.uniform(&format!("{skey}.dims"), KvScatterDims { rows, kv_heads, head_dim, kv_base, max_ctx, src_offset: src.elem_offset, stride_x, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.kv_scatter,
        &[
            BindGroupEntry { binding: 0, resource: src.buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cache_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.kv_scatter, &bg, (gx, gy, 1), key);
}

#[allow(clippy::too_many_arguments)]
fn attn_prefill<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, q: impl Into<BufView<'a>>, k: impl Into<BufView<'a>>, v: impl Into<BufView<'a>>, seq: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let (q, k, v) = (q.into(), k.into(), v.into());
    // Concatenated-heads width (`num_heads * head_dim`): equal to
    // `cfg.hidden_size` for Qwen2 (head_dim is derived that way) but *not*
    // for Qwen3, where head_dim=128 is explicit and 16*128=2048 != 1024 -
    // see `qkv_proj`'s doc comment on the same distinction.
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (seq * hidden) as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{skey}.dims"),
        AttnPrefillDims { seq, n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, scale, _p0: 0, _p1: 0, _p2: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_prefill,
        &[
            BindGroupEntry { binding: 0, resource: q.binding() },
            BindGroupEntry { binding: 1, resource: k.binding() },
            BindGroupEntry { binding: 2, resource: v.binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    // wg.y = query tile index (256 rows/tile, see attn_prefill.wgsl's doc
    // comment on why this scales with `seq` instead of a fixed dispatch).
    engine.dispatch(pass, &engine.attn_prefill, &bg, (cfg.num_heads as u32, seq.div_ceil(256), 1), key);
    out
}

/// Split-K ("flash-decoding") threshold: below this `kv_len`, one
/// workgroup's O(kv_len) sequential loop (`attn_decode.wgsl`) is already
/// cheap and splitting it would only add the second (reduce) pass's
/// overhead for nothing - matches `shaders/attn_decode_split.wgsl`'s own
/// per-workgroup chunk size. Above it, the single-workgroup-per-head loop
/// becomes the decode step's latency floor even though the GPU has far more
/// concurrent-workgroup capacity than `n_heads` uses, and splitting each
/// head's KV range across `num_splits` workgroups lets that capacity
/// actually shorten the loop.
///
/// `head_dim`-conditioned (session 3): a single global constant isn't a
/// best fit for every `head_dim` this project compiles. Session 2 tried a
/// uniform `SPLIT_CHUNK=64` and found a real win on `head_dim=64` models
/// (Qwen2.5-0.5B, SmolLM2-360M: -4.7% to -9.3%) but a regression on
/// `head_dim=128` (Qwen3-1.7B: +14.1% at the newly-crossed `kv_len=65`
/// threshold - `attn_decode_split_128`'s wider `workgroup_size(128)` split
/// kernel doesn't cover its launch/reduce overhead at the tiny 33-key
/// chunks that threshold produces). Splitting the constant by `head_dim` -
/// still a shape fact known at pipeline-selection time, never a timing
/// measurement - captures the `head_dim=64` win without moving the
/// `head_dim=128` threshold at all.
fn split_chunk(head_dim: u32) -> u32 {
    if head_dim <= 64 { 64 } else { 128 }
}
/// Upper bound on split count and the partial-result buffers' per-head
/// stride; must match `MAX_SPLITS` in `attn_decode_split.wgsl` and
/// `attn_decode_reduce.wgsl` exactly. Fixed so those buffers never need to
/// regrow as `kv_len` grows one token per decode step.
const MAX_SPLITS: u32 = 32;

/// `kv_len <= split_chunk(head_dim)` returns `(1, kv_len)` (no split:
/// `attn_decode` picks the single-workgroup kernel). Otherwise `num_splits =
/// min(MAX_SPLITS, ceil(kv_len / split_chunk(head_dim)))` and `chunk =
/// ceil(kv_len / num_splits)` (recomputed from the actual split count so the
/// chunks partition `[0, kv_len)` exactly, with no split ever going idle).
fn decode_split_plan(kv_len: u32, head_dim: u32) -> (u32, u32) {
    let split_chunk = split_chunk(head_dim);
    if kv_len <= split_chunk {
        return (1, kv_len.max(1));
    }
    let num_splits = kv_len.div_ceil(split_chunk).min(MAX_SPLITS);
    let chunk = kv_len.div_ceil(num_splits).max(1);
    (num_splits, chunk)
}

#[allow(clippy::too_many_arguments)]
fn attn_decode<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, q: impl Into<BufView<'a>>, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, kv_len: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let q = q.into();
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), hidden as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let n_heads = cfg.num_heads as u32;
    let head_dim = cfg.head_dim as u32;

    let (num_splits, chunk) = decode_split_plan(kv_len, head_dim);
    if num_splits <= 1 {
        // Short context: the plain single-workgroup-per-head kernel, exactly
        // as before this change (no split/reduce overhead).
        let dims = pool.uniform(
            &format!("{skey}.dims"),
            AttnDecodeDims { n_heads, n_kv_heads: cfg.num_kv_heads as u32, head_dim, kv_len, max_ctx, scale, _p0: 0, _p1: 0 },
        );
        // attn_decode.wgsl/attn_decode_128.wgsl have `workgroup_size` and a
        // per-thread-owns-one-output-dim design compile-time-fixed to
        // head_dim=64/128 respectively - see engine.rs's doc comment on
        // `attn_decode_128`. Picked by `cfg.head_dim`, not a runtime parameter.
        let pipeline = match cfg.head_dim {
            64 => &engine.attn_decode,
            128 => &engine.attn_decode_128,
            other => panic!("attn_decode: unsupported head_dim {other} (only 64/Qwen2.5 and 128/Qwen3 have a compiled kernel)"),
        };
        let bg = pool.bind_group(
            key,
            pipeline,
            &[
                BindGroupEntry { binding: 0, resource: q.binding() },
                BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ],
        );
        engine.dispatch(pass, pipeline, &bg, (n_heads, 1, 1), key);
        return out;
    }

    // Long context: split-K pass 1 (partial online-softmax per (head, split))
    // then pass 2 (reduce) into the same `out` buffer as the short-context
    // path, so callers never need to know which path ran.
    let (split_pipeline, reduce_pipeline) = match cfg.head_dim {
        64 => (&engine.attn_decode_split, &engine.attn_decode_reduce),
        128 => (&engine.attn_decode_split_128, &engine.attn_decode_reduce_128),
        other => panic!("attn_decode: unsupported head_dim {other} (only 64/Qwen2.5 and 128/Qwen3 have a compiled kernel)"),
    };

    let partial_m = pool.data(&format!("{skey}.partial_m"), (n_heads * MAX_SPLITS) as usize);
    let partial_l = pool.data(&format!("{skey}.partial_l"), (n_heads * MAX_SPLITS) as usize);
    let partial_acc = pool.data(&format!("{skey}.partial_acc"), (n_heads * MAX_SPLITS * head_dim) as usize);

    let split_dims = pool.uniform(
        &format!("{skey}.split_dims"),
        AttnDecodeSplitDims { n_heads, n_kv_heads: cfg.num_kv_heads as u32, head_dim, kv_len, max_ctx, scale, chunk, _p0: 0 },
    );
    let split_bg = pool.bind_group(
        &format!("{key}.split"),
        split_pipeline,
        &[
            BindGroupEntry { binding: 0, resource: q.binding() },
            BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: partial_m.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: partial_l.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: partial_acc.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: split_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, split_pipeline, &split_bg, (n_heads, num_splits, 1), &format!("{key}.split"));

    let reduce_dims = pool.uniform(&format!("{skey}.reduce_dims"), AttnDecodeReduceDims { n_heads, head_dim, num_splits, _p0: 0 });
    let reduce_bg = pool.bind_group(
        &format!("{key}.reduce"),
        reduce_pipeline,
        &[
            BindGroupEntry { binding: 0, resource: partial_m.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: partial_l.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: partial_acc.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: reduce_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, reduce_pipeline, &reduce_bg, (n_heads, 1, 1), &format!("{key}.reduce"));
    out
}

/// Pluggable-mask GQA attention over a resident-prefix KV cache: `t` query
/// rows attend `kv_total = prefix_len + t` keys already scattered into
/// `k_cache`/`v_cache`, gated by an explicit bitset instead of an implicit
/// causal rule: see `shaders/attn_chunk_masked.wgsl`.
#[allow(clippy::too_many_arguments)]
fn attn_chunk_masked<'a>(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, q: impl Into<BufView<'a>>, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, mask: &wgpu::Buffer, t: u32, kv_total: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let q = q.into();
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let skey = scratch_key(key);
    let out = pool.data(&format!("{skey}.out"), (t * hidden) as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{skey}.dims"),
        AttnChunkDims { t, n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, kv_total, max_ctx, scale, _p0: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_chunk_masked,
        &[
            BindGroupEntry { binding: 0, resource: q.binding() },
            BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: mask.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.attn_chunk_masked, &bg, (cfg.num_heads as u32, t.div_ceil(256), 1), key);
    out
}

/// Gathers + dequantizes `token_ids` (absolute vocab ids) out of `w` (must
/// be `MatMulWeight::Q8_0`: this model's lm head) into a small contiguous
/// F32 `[token_ids.len(), hidden]` buffer, one dispatch per id (llm-life's
/// sliced sets are 1-2 ids; not worth a multi-row-per-dispatch path yet).
/// Returns the buffer and `hidden` (the caller already knows `hidden`, but
/// returning it here keeps this fn self-contained for a future non-Qwen2
/// caller). See `shaders/gather_dequant_q8_rows.wgsl`.
fn gather_dequant_head_rows(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, w: &MatMulWeight, token_ids: &[u32]) -> (wgpu::Buffer, u32) {
    let (chunks, blocks_per_row) = match w {
        MatMulWeight::Q8_0 { chunks, blocks_per_row, .. } => (chunks, *blocks_per_row),
        _ => panic!("gather_dequant_head_rows: lm_head must be Q8_0"),
    };
    let hidden = blocks_per_row * 32;
    let n = token_ids.len() as u32;
    let out = pool.data(&format!("{key}.out"), (n * hidden) as usize);
    for (ti, &id) in token_ids.iter().enumerate() {
        let chunk = chunks.iter().find(|c| id >= c.row_start && id < c.row_start + c.rows).unwrap_or_else(|| panic!("gather_dequant_head_rows: token id {id} out of range"));
        let local_row = id - chunk.row_start;
        let ckey = format!("{key}.{ti}");
        let row_ids = pool.upload_u32(&format!("{ckey}.rowid"), &[local_row]);
        let dims = pool.uniform(&format!("{ckey}.dims"), GatherRowsDims { n_selected: 1, k: hidden, blocks_per_row, out_row_offset: ti as u32 });
        let bg = pool.bind_group(
            &ckey,
            &engine.gather_dequant_q8_rows,
            &[
                BindGroupEntry { binding: 0, resource: row_ids.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: chunk.qs.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: chunk.scales.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ],
        );
        engine.dispatch(pass, &engine.gather_dequant_q8_rows, &bg, (hidden.div_ceil(64), 1, 1), &ckey);
    }
    (out, hidden)
}

/// Diagnostics split point: when `Engine::diag_split()` is on, closes the
/// open compute pass and opens a new one labelled `$label`, so the pass-level
/// timestamps time each op group separately. A no-op otherwise: the default
/// command stream keeps its one-pass-per-layer-half shape.
macro_rules! seg {
    ($engine:expr, $encoder:expr, $pass:ident, $label:expr) => {
        if $engine.diag_split() {
            drop($pass);
            $pass = $engine.begin_pass($encoder, $label);
        }
    };
}

/// Prefill: runs every layer over the whole prompt with causal attention
/// (no cache read needed - attends directly over this call's own q/k/v),
/// GPU-scatters every position's K/V into `cache`, and returns the last
/// position's logits ([vocab]).
pub async fn forward_prefill(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32> {
    model.pool.use_kv_cache(cache.id);
    let cfg = &model.config;
    let seq = token_ids.len() as u32;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;
    let t_start = crate::engine::now_ms();
    engine.diag_begin();

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill") });
    // One open ComputePass for the whole prefill outside the per-layer
    // flushes: see `Engine::dispatch`'s doc comment (the K/V cache writes
    // are dispatches too, `scatter_kv_kernel`, so the pass stays open
    // across them).
    let mut pass = engine.begin_pass(&mut encoder, "embed");
    let x = embed_gather(engine, pool, &mut pass, model, token_ids);
    seg!(engine, &mut encoder, pass, "norm");

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("layer{i}");
        let normed = rmsnorm(engine, pool, &mut pass, &format!("{key}.norm"), &x, &layer.attn_norm, seq, hidden, cfg.rms_norm_eps);
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        seg!(engine, &mut encoder, pass, "qkv");
        let (q, k, v) = qkv_proj(engine, pool, &mut pass, &format!("{key}.qkv"), &normed, seq, cfg, layer, model.fast_kernels, lora_layer);
        seg!(engine, &mut encoder, pass, "rope");
        rope(engine, pool, &mut pass, &format!("{key}.ropeq"), &q, cos, sin, seq, cfg.num_heads as u32, cfg.head_dim as u32, 0);
        rope(engine, pool, &mut pass, &format!("{key}.ropek"), &k, cos, sin, seq, cfg.num_kv_heads as u32, cfg.head_dim as u32, 0);
        seg!(engine, &mut encoder, pass, "attn");
        let attn_out = attn_prefill(engine, pool, &mut pass, &format!("{key}.attn"), &q, &k, &v, seq, cfg);
        seg!(engine, &mut encoder, pass, "kv");
        scatter_kv_kernel(engine, pool, &mut pass, &format!("{key}.kscatter"), &cache.k[i], &k, seq, cfg, 0, cache.max_ctx);
        scatter_kv_kernel(engine, pool, &mut pass, &format!("{key}.vscatter"), &cache.v[i], &v, seq, cfg, 0, cache.max_ctx);
        seg!(engine, &mut encoder, pass, "o_proj");

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        let o = linear_lora(engine, pool, &mut pass, &format!("{key}.wo"), &attn_out, seq, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        seg!(engine, &mut encoder, pass, "add+norm");
        add_inplace(engine, pool, &mut pass, &format!("{key}.add1"), &x, &o, seq * hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut pass, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, seq, hidden, cfg.rms_norm_eps);
        // `mlp()`'s three calls, inlined so `seg!` can time them apart.
        let mlp_key = format!("{key}.mlp");
        let inter = cfg.intermediate_size as u32;
        seg!(engine, &mut encoder, pass, "gate_up");
        let gate_up = linear(engine, pool, &mut pass, &format!("{mlp_key}.gate_up"), &ffn_normed, seq, hidden, &layer.gate_up_w, &layer.gate_up_b, 2 * inter, model.fast_kernels);
        seg!(engine, &mut encoder, pass, "silu");
        let gated = silu_mul_fused(engine, pool, &mut pass, &format!("{mlp_key}.silu"), &gate_up, seq, inter);
        seg!(engine, &mut encoder, pass, "down");
        let mlp_out = linear(engine, pool, &mut pass, &format!("{mlp_key}.down"), &gated, seq, inter, &layer.down_w, &layer.down_b, hidden, model.fast_kernels);
        seg!(engine, &mut encoder, pass, "add");
        add_inplace(engine, pool, &mut pass, &format!("{key}.add2"), &x, &mlp_out, seq * hidden);

        // Flush per layer - see `Engine::flush_encoder`'s doc comment for
        // why a single encoder covering every layer hangs on Metal at
        // Qwen3-0.6B's depth (28 layers).
        drop(pass);
        engine.flush_encoder(&mut encoder, "prefill");
        pass = engine.begin_pass(&mut encoder, if i + 1 < model.layers.len() { "norm" } else { "tail" });
    }

    // Only the last position's row ever feeds the next token (this fn's
    // documented return value is just that one row's logits - every caller
    // already only reads `all_logits[(seq-1)*vocab..]`), so slice `x` down
    // to that one row here, BEFORE `out_norm`/`lm_head`, instead of running
    // the final RMSNorm and the lm-head matmul over all `seq` rows and
    // throwing away `seq - 1` of them. `x` is `[seq, hidden]` row-major
    // (embed_gather's/every layer's convention), so the last row is a
    // contiguous `hidden`-f32 byte range at the end of the buffer - one
    // `copy_buffer_to_buffer` (encoder-level, hence the pass close/reopen -
    // same pattern as `scatter_kv_gpu`'s calls above).
    //
    // This matters far more than it looks: `lm_head` is a `[hidden,
    // vocab_size]` matmul, and `output.weight` on this crate's official
    // GGUFs is the naive/no-fast-path Q6_K kernel or an un-tiled Q4_0/Q8_0
    // path at the vocab sizes involved (151936 for Qwen2.5/Qwen3) - running
    // it at `rows = seq` instead of `rows = 1` multiplied both its compute
    // and its output buffer size by `seq` for no benefit. Measured on
    // Qwen2.5-0.5B-Instruct (native, `LEAN_DEBUG_POOL_TOP`): the `lm_head`
    // scratch buffer alone was 1.43GB at `seq = 2225` before this fix -
    // this crate's single largest scratch allocation, dwarfing everything
    // else in `Pool` (see docs/runs/2026-09-28-lean-decode-breakdown.md).
    let last_row = pool.data("prefill_last_row", hidden as usize);
    drop(pass);
    encoder.copy_buffer_to_buffer(&x, ((seq - 1) * hidden * 4) as u64, &last_row, 0, (hidden * 4) as u64);
    pass = engine.begin_pass(&mut encoder, "lm_head");

    let normed_final = rmsnorm(engine, pool, &mut pass, "out_norm", &last_row, &model.out_norm, 1, hidden, cfg.rms_norm_eps);
    let logits = linear(engine, pool, &mut pass, "lm_head", &normed_final, 1, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, model.fast_kernels);
    if let Some(mask) = mask {
        mask_logits_gpu(engine, pool, &mut pass, "prefill_mask", &logits, mask, cfg.vocab_size as u32, 0);
    }
    drop(pass);

    let diag = engine.diag_resolve(&mut encoder);
    engine.queue.submit(Some(encoder.finish()));
    let t_submitted = crate::engine::now_ms();
    let vocab = cfg.vocab_size;
    let all_logits = engine.read_buffer(&logits, vocab).await;
    engine.diag_encode_ms.set(t_submitted - t_start);
    engine.diag_wait_ms.set(crate::engine::now_ms() - t_submitted);
    if let Some((buf, count)) = diag {
        engine.diag_collect(&buf, count).await;
    }
    cache.kv_len = seq;
    all_logits
}

/// Applies an allowed-token bitset to one row of `logits` (`vocab` wide, at
/// element offset `row_offset`) in place, on the GPU, before argmax or
/// readback - the mechanism behind per-step constrained decoding. A no-op
/// dispatch is never recorded when `mask` is `None`: unmasked generation
/// then has exactly the same dispatch sequence (and result) as before this
/// feature existed.
#[allow(clippy::too_many_arguments)]
fn mask_logits_gpu(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, logits: &wgpu::Buffer, mask: &wgpu::Buffer, vocab: u32, row_offset: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), MaskDims { n: vocab, offset: row_offset, _p0: 0, _p1: 0 });
    // Not `pool.bind_group`: a caller driving constrained decoding uploads a
    // *different* mask buffer every step (a fresh `wgpu::Buffer`, not a
    // `Pool`-tracked one written in place) - `Pool`'s bind-group cache only
    // invalidates on its own generation counter (bumped when a pool-owned
    // buffer regrows), so a cached bind group here would keep pointing at
    // the *first* step's mask buffer forever (exactly `pool.rs`'s documented
    // stale-binding hazard, applied to an externally-owned buffer). Building
    // fresh every call is the correct fix, not a missing optimization: the
    // cost is one small bind-group alloc per masked step, only paid when a
    // caller opts into masking at all.
    let bg = engine.bind_group(
        &engine.mask_logits,
        &[
            BindGroupEntry { binding: 0, resource: logits.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: mask.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.mask_logits, &bg, (vocab.div_ceil(256), 1, 1), key);
}

fn argmax_gpu(engine: &Engine, pool: &Pool, pass: &mut wgpu::ComputePass<'_>, key: &str, logits: &wgpu::Buffer, vocab: u32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), 1);
    let dims = pool.uniform(&format!("{key}.dims"), ArgmaxDims { n: vocab, _p0: 0, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.argmax,
        &[
            BindGroupEntry { binding: 0, resource: logits.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(pass, &engine.argmax, &bg, (1, 1, 1), key);
    out
}

/// Records one decode step's whole layer stack (embed through the lm-head
/// logits) into `encoder`, without submitting or reading anything back -
/// shared by `forward_decode_step` (full-logits readback, for samplers) and
/// `forward_decode_step_argmax` (single-index readback, the fast path),
/// so the two only differ in what they append after the layer stack.
/// `argmax` selects whether the GPU argmax (`argmax_gpu`) is folded into this
/// same still-open compute pass before it closes, so a greedy decode step
/// opens no extra pass beyond this function's own one: `Some(idx_buffer)` is
/// returned alongside the logits when requested, `None` otherwise (samplers
/// that need the full vocab never pay for the extra dispatch).
#[allow(clippy::too_many_arguments)]
fn decode_layers(engine: &Engine, model: &GpuModel, encoder: &mut wgpu::CommandEncoder, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>, argmax: bool) -> (wgpu::Buffer, Option<wgpu::Buffer>) {
    model.pool.use_kv_cache(cache.id);
    let cfg = &model.config;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;
    let pos = cache.kv_len;

    // One open ComputePass instead of one per dispatch (same
    // rationale as `forward_prefill` - see `Engine::dispatch`'s doc
    // comment): decode's per-step dispatch count doesn't scale with
    // kv_len, but at ~15 dispatches/layer every one was still its own
    // `MTLComputeCommandEncoder` session before this change (532
    // single-dispatch passes/step measured on Qwen2.5-3B, 36 layers).
    // The K/V cache write is a dispatch (`rope_kv_decode`), so the whole
    // step is one pass, across the layer loop too.
    let mut pass = engine.begin_pass(encoder, "embed");
    let x = embed_gather(engine, pool, &mut pass, model, &[token_id]);
    seg!(engine, encoder, pass, "norm");

    // `normed` carries the *next* dispatch's already-computed rmsnorm input:
    // seeded here from a plain rmsnorm on the fresh embedding (layer 0's
    // attn_norm), then thereafter produced as a side effect of the previous
    // iteration's `add_rmsnorm(add2, next layer's attn_norm)` fusion below -
    // see that call's comment. This removes one dispatch per layer relative
    // to a separate `add_inplace`+`rmsnorm(attn_norm)` pair.
    let mut normed = rmsnorm(engine, pool, &mut pass, "dec_layer0.norm", &x, &model.layers[0].attn_norm, 1, hidden, cfg.rms_norm_eps);

    let num_layers = model.layers.len();
    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("dec_layer{i}");
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        seg!(engine, encoder, pass, "qkv");
        let (q, k, v) = qkv_proj(engine, pool, &mut pass, &format!("{key}.qkv"), &normed, 1, cfg, layer, model.fast_kernels, lora_layer);
        seg!(engine, encoder, pass, "rope");
        let q = rope_kv_decode(engine, pool, &mut pass, &format!("{key}.ropekv"), &q, &k, &v, cos, sin, &cache.k[i], &cache.v[i], pos, cache.max_ctx, cfg);
        seg!(engine, encoder, pass, "attn");
        let attn_out = attn_decode(engine, pool, &mut pass, &format!("{key}.attn"), &q, &cache.k[i], &cache.v[i], pos + 1, cache.max_ctx, cfg);

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        seg!(engine, encoder, pass, "o_proj");
        let o = linear_lora(engine, pool, &mut pass, &format!("{key}.wo"), &attn_out, 1, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        // Fused add1 (x += o) + ffnnorm(x) - one dispatch instead of two
        // (docs/runs/2026-09-29-lean-kernels.md).
        seg!(engine, encoder, pass, "add+norm");
        let ffn_normed = add_rmsnorm(engine, pool, &mut pass, &format!("{key}.add1_ffnnorm"), &x, &o, &layer.ffn_norm, 1, hidden, cfg.rms_norm_eps);
        // `mlp()`'s three calls, inlined so `seg!` can time them apart.
        let mlp_key = format!("{key}.mlp");
        let inter = cfg.intermediate_size as u32;
        seg!(engine, encoder, pass, "gate_up");
        let gated = match gate_up_swiglu_decode(engine, pool, &mut pass, &format!("{mlp_key}.gate_up"), &ffn_normed, hidden, &layer.gate_up_w, &layer.gate_up_b, inter, model.fast_kernels) {
            Some(gated) => gated,
            None => {
                let gate_up = linear(engine, pool, &mut pass, &format!("{mlp_key}.gate_up"), &ffn_normed, 1, hidden, &layer.gate_up_w, &layer.gate_up_b, 2 * inter, model.fast_kernels);
                seg!(engine, encoder, pass, "silu");
                silu_mul_fused(engine, pool, &mut pass, &format!("{mlp_key}.silu"), &gate_up, 1, inter)
            }
        };
        seg!(engine, encoder, pass, "down");
        let mlp_out = linear(engine, pool, &mut pass, &format!("{mlp_key}.down"), &gated, 1, inter, &layer.down_w, &layer.down_b, hidden, model.fast_kernels);
        seg!(engine, encoder, pass, "add+norm");
        // Fused add2 (x += mlp_out) + the *next* dispatch's rmsnorm: the
        // following layer's attn_norm, or (last layer) the final out_norm -
        // either way `x`'s next reader is always exactly one rmsnorm, so
        // this fusion always applies. `normed` here becomes next iteration's
        // `qkv_proj` input (or `normed_final` below, at the last layer).
        let next_scale = if i + 1 < num_layers { &model.layers[i + 1].attn_norm } else { &model.out_norm };
        normed = add_rmsnorm(engine, pool, &mut pass, &format!("{key}.add2_nextnorm"), &x, &mlp_out, next_scale, 1, hidden, cfg.rms_norm_eps);

        // No per-layer flush here (unlike `forward_prefill`'s per-layer
        // flush, which exists to bound the *seq_len*-scaled dispatch count
        // that caused the native `device.poll` timeout documented on
        // `Engine::flush_encoder`). Decode's dispatch count is independent
        // of kv_len and stays in the low hundreds even at the split-K path's
        // worst case (measured: 364-508 total across a whole decode step
        // depending on model/context - see this session's run doc), well
        // under the threshold that produced that timeout. Profiling (this
        // session's run doc) measured a per-layer flush+blocking-wait here
        // costing ~30% of decode time on Qwen2.5-0.5B (submit/poll overhead
        // multiplied by layer count) for zero correctness benefit at this
        // dispatch volume - removed.
    }

    let normed_final = normed;
    seg!(engine, encoder, pass, "lm_head");
    let logits = linear(engine, pool, &mut pass, "dec_lm_head", &normed_final, 1, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, model.fast_kernels);
    if let Some(mask) = mask {
        mask_logits_gpu(engine, pool, &mut pass, "dec_mask", &logits, mask, cfg.vocab_size as u32, 0);
    }
    if argmax {
        seg!(engine, encoder, pass, "argmax");
    }
    let idx = argmax.then(|| argmax_gpu(engine, pool, &mut pass, "dec_argmax", &logits, cfg.vocab_size as u32));
    drop(pass);
    (logits, idx)
}

/// Decode one token against the cache (already populated up to
/// `cache.kv_len`): embed, run every layer (each layer GPU-scatters this
/// step's K/V into the cache at `cache.kv_len` before its own attention
/// dispatch, so the step attends to itself too), return logits ([vocab]).
/// Full-vocab readback (~608KB for this model) - kept for samplers that need
/// more than the top-1 id; greedy decode should prefer
/// `forward_decode_step_argmax` below. `mask`, if given, is a bitset built
/// by `build_mask_bitset` - see `mask_logits_gpu`'s doc comment for how it's
/// applied.
pub async fn forward_decode_step(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32> {
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("decode") });
    let (logits, _) = decode_layers(engine, model, &mut encoder, cache, token_id, cos, sin, mask, false);
    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len += 1;
    engine.read_buffer(&logits, model.config.vocab_size).await
}

/// Same forward as `forward_decode_step`, but the argmax over the logits
/// runs on the GPU (`shaders/argmax.wgsl`, one workgroup) in the same
/// encoder/submit as the rest of the step, so the only readback is 4 bytes
/// (one `u32`) instead of `vocab_size * 4` - the fast path for greedy
/// decoding. Ties broken toward the lower index, matching the CPU `argmax()`
/// helpers in `lean_cli.rs`/`web.rs`. When `mask` is given, the mask is
/// applied (in the same encoder) before argmax, so the returned index is
/// always drawn from the allowed set - constrained greedy decoding with no
/// extra readback over the unconstrained path.
#[allow(clippy::too_many_arguments)]
pub async fn forward_decode_step_argmax(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> u32 {
    let t_start = crate::engine::now_ms();
    engine.diag_begin();
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("decode_argmax") });
    if engine.profiling_enabled() {
        engine.reset_profile();
    }
    let (_, idx) = decode_layers(engine, model, &mut encoder, cache, token_id, cos, sin, mask, true);
    let idx = idx.expect("decode_layers(argmax=true) always returns Some");
    // Resolve on the same encoder as the profiled dispatches, before it's
    // finished/submitted - see `Engine::resolve_profile`'s doc comment.
    let profile = engine.resolve_profile(&mut encoder);
    let diag = engine.diag_resolve(&mut encoder);
    engine.queue.submit(Some(encoder.finish()));
    let t_submitted = crate::engine::now_ms();
    cache.kv_len += 1;
    let result = engine.read_u32(&idx).await;
    engine.diag_encode_ms.set(t_submitted - t_start);
    engine.diag_wait_ms.set(crate::engine::now_ms() - t_submitted);
    if let Some((buf, count)) = diag {
        engine.diag_collect(&buf, count).await;
    }
    if let Some((buf, count)) = profile {
        let data = engine.read_profile(&buf, count).await;
        crate::profile_report::record_step(&data);
    }
    result
}

/// Appends `token_ids` onto a cache already populated up to `cache.kv_len`
/// (typically just after `KvCache::restore` from a resident-prefix
/// snapshot) - the "prefill(suffix)" half of KV snapshot/restore. Each token
/// is forced (no argmax - these are known tokens, not generated ones)
/// through its own `decode_layers` call/submit, exactly `forward_decode_step`'s
/// shape, except intermediate tokens' logits are never read back (nothing
/// needs them) - only the *last* token's logits are, matching
/// `forward_prefill`'s "returns the last position's logits" contract.
///
/// **Deliberately one submit per token, not one encoder/submit for the
/// whole suffix.** An earlier version of this function batched every
/// token's `decode_layers` call into a single encoder before one submit, to
/// save dispatch/submit overhead. That hung on native (Metal) with no CPU
/// spin and no error: `Pool`'s per-call-site uniform buffers (`RopeDims`'s
/// `pos_base`, `AttnDecodeDims`'s `kv_len`, etc.) are updated via
/// `queue.write_buffer`, which is a *queue-timeline* operation, not an
/// encoder-timeline one - every `write_buffer` call for the same
/// pool-cached key before the next `submit()` clobbers the previous one, so
/// batching multiple `decode_layers` calls into one unsubmitted encoder
/// made every dispatch in that encoder read whatever dims the *last*
/// iteration's writes left behind, not its own - undefined/hanging
/// behavior on the KV-cache offsets `scatter_kv_gpu` computes from them.
/// One submit per token keeps every `write_buffer` visible before the
/// dispatches that depend on it run, at the cost of `token_ids.len()`
/// submits instead of 1 (still zero *readbacks* except the last).
pub async fn forward_prefill_suffix(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32> {
    let (last, rest) = token_ids.split_last().expect("forward_prefill_suffix needs at least one token");
    for &tok in rest {
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill_suffix_step") });
        let _ = decode_layers(engine, model, &mut encoder, cache, tok, cos, sin, None, false);
        engine.queue.submit(Some(encoder.finish()));
        cache.kv_len += 1;
    }
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill_suffix_last") });
    let (logits, _) = decode_layers(engine, model, &mut encoder, cache, *last, cos, sin, mask, false);
    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len += 1;
    engine.read_buffer(&logits, model.config.vocab_size).await
}

/// Runs one chunk of `token_ids` against the cache's resident prefix
/// (`cache.kv_len` positions, already populated: typically by an earlier
/// `forward_prefill`/`forward_prefill_suffix` call, then rewound with
/// `KvCache::snapshot`/`restore` so the next chunk starts from the same
/// prefix) with caller-supplied positions and attention topology
/// (`ForwardSpec`) instead of the fixed causal-continuation shape the other
/// `forward_*` functions assume. This is the mechanism behind llm-life
/// variant A's packed block-diagonal per-cell prompts (`spec.positions`
/// restarts RoPE at the prefix length for every block; `spec.allowed_bits`
/// is the block-diagonal mask) and variant B's sparse 9-key stencil.
///
/// Returns the **hidden states** (`[t, hidden]`, post `output_norm`, pre
/// lm-head: not logits): callers needing only a few vocab ids' logits at
/// every position should slice with `GpuModel::lm_head_sliced` rather than
/// materializing a full `[t, vocab]` buffer (see that fn's doc comment).
/// Leaves `cache.kv_len` at `prefix_len + t`: call `KvCache::snapshot`
/// before this and `KvCache::restore` after reading the result back if the
/// next chunk should start from the same prefix again (llm-life's own
/// per-chunk rewind, `LifeEngine::step_ids_a`'s pattern in the Burn
/// engine).
pub async fn forward_chunk_spec(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, spec: &ForwardSpec) -> wgpu::Buffer {
    model.pool.use_kv_cache(cache.id);
    let cfg = &model.config;
    let t = token_ids.len() as u32;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;
    let prefix_len = cache.kv_len;
    let kv_total = prefix_len + t;
    assert!(kv_total <= cache.max_ctx, "forward_chunk_spec: prefix_len {prefix_len} + t {t} exceeds max_ctx {}", cache.max_ctx);

    let positions: Vec<u32> = match &spec.positions {
        Some(p) => {
            assert_eq!(p.len(), t as usize, "ForwardSpec positions must be one per token");
            p.clone()
        }
        None => (0..t).map(|r| prefix_len + r).collect(),
    };
    let pos_buf = pool.upload_u32("chunk.positions", &positions);

    let default_mask;
    let mask_bits: &[u32] = match &spec.allowed_bits {
        Some(b) => b,
        None => {
            default_mask = build_prefix_causal_bits(prefix_len, t);
            &default_mask
        }
    };
    let mask_buf = pool.upload_u32("chunk.mask", mask_bits);

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("chunk") });
    // Same pass-batching as `forward_prefill`/`decode_layers` - see
    // `Engine::dispatch`'s doc comment.
    let mut pass = engine.begin_pass(&mut encoder, "chunk");
    let x = embed_gather(engine, pool, &mut pass, model, token_ids);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("chunk_layer{i}");
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        let normed = rmsnorm(engine, pool, &mut pass, &format!("{key}.norm"), &x, &layer.attn_norm, t, hidden, cfg.rms_norm_eps);
        let (q, k, v) = qkv_proj(engine, pool, &mut pass, &format!("{key}.qkv"), &normed, t, cfg, layer, model.fast_kernels, lora_layer);
        rope_positions(engine, pool, &mut pass, &format!("{key}.ropeq"), &q, cos, sin, &pos_buf, t, cfg.num_heads as u32, cfg.head_dim as u32);
        rope_positions(engine, pool, &mut pass, &format!("{key}.ropek"), &k, cos, sin, &pos_buf, t, cfg.num_kv_heads as u32, cfg.head_dim as u32);

        scatter_kv_kernel(engine, pool, &mut pass, &format!("{key}.kscatter"), &cache.k[i], &k, t, cfg, prefix_len, cache.max_ctx);
        scatter_kv_kernel(engine, pool, &mut pass, &format!("{key}.vscatter"), &cache.v[i], &v, t, cfg, prefix_len, cache.max_ctx);

        let attn_out = attn_chunk_masked(engine, pool, &mut pass, &format!("{key}.attn"), &q, &cache.k[i], &cache.v[i], &mask_buf, t, kv_total, cache.max_ctx, cfg);

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        let o = linear_lora(engine, pool, &mut pass, &format!("{key}.wo"), &attn_out, t, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        add_inplace(engine, pool, &mut pass, &format!("{key}.add1"), &x, &o, t * hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut pass, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, t, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, &mut pass, &format!("{key}.mlp"), &ffn_normed, t, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, &mut pass, &format!("{key}.add2"), &x, &mlp_out, t * hidden);

        // See `Engine::flush_encoder`'s doc comment.
        drop(pass);
        engine.flush_encoder(&mut encoder, "chunk");
        pass = engine.begin_pass(&mut encoder, "chunk");
    }

    let normed_final = rmsnorm(engine, pool, &mut pass, "chunk_out_norm", &x, &model.out_norm, t, hidden, cfg.rms_norm_eps);
    drop(pass);

    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len = kv_total;
    normed_final
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod grid1d_tests {
    //! Regression coverage for the M=1525 crash on Qwen2.5-3B-Instruct
    //! (native wgpu panic, "Encoder is invalid" at `CommandEncoder::finish`,
    //! caused by `silu_mul_fused`'s 1-D dispatch exceeding
    //! `max_compute_workgroups_per_dimension` (65535) at
    //! `(rows*intermediate_size).div_ceil(256)`). Runs `add_inplace`/
    //! `silu_mul_fused` directly at more than `MAX_WORKGROUPS_PER_DIM*256`
    //! elements (so `grid1d` must produce `y > 1`) and checks the GPU
    //! result against a plain CPU computation of the same op.
    use super::*;
    use crate::engine::Engine;
    use crate::pool::Pool;

    fn engine_and_pool() -> (Engine, Pool) {
        let engine = Engine::new().expect("wgpu device for grid1d test");
        let pool = Pool::new(engine.device.clone(), engine.queue.clone());
        (engine, pool)
    }

    #[test]
    fn add_inplace_beyond_max_workgroups_matches_cpu() {
        let (engine, pool) = engine_and_pool();
        let total: u32 = MAX_WORKGROUPS_PER_DIM * 256 + 4321; // forces grid1d's y > 1
        let a: Vec<f32> = (0..total).map(|i| (i as f32) * 0.5 - 100.0).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.0173).sin()).collect();
        let a_buf = engine.buf_f32(&a, "grid1d_test.a");
        let b_buf = engine.buf_f32(&b, "grid1d_test.b");
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = engine.begin_pass(&mut encoder, "test");
        add_inplace(&engine, &pool, &mut pass, "grid1d_test.add", &a_buf, &b_buf, total);
        drop(pass);
        engine.queue.submit(Some(encoder.finish()));
        let got = pollster::block_on(engine.read_buffer(&a_buf, total as usize));
        for i in [0usize, 12345, (total / 2) as usize, (total - 1) as usize] {
            let expected = a[i] + b[i];
            assert!((got[i] - expected).abs() < 1e-4, "index {i}: got {} expected {expected}", got[i]);
        }
    }

    #[test]
    fn silu_mul_fused_beyond_max_workgroups_matches_cpu() {
        let (engine, pool) = engine_and_pool();
        let hidden: u32 = 4096;
        // rows*hidden must exceed MAX_WORKGROUPS_PER_DIM*256 to force y > 1.
        let rows: u32 = (MAX_WORKGROUPS_PER_DIM * 256) / hidden + 2;
        assert!(rows * hidden > MAX_WORKGROUPS_PER_DIM * 256);
        let gate_up: Vec<f32> = (0..rows * 2 * hidden).map(|i| ((i as f32) * 0.0091).cos() * 3.0).collect();
        let gate_up_buf = engine.buf_f32(&gate_up, "grid1d_test.gate_up");
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = engine.begin_pass(&mut encoder, "test");
        let out = silu_mul_fused(&engine, &pool, &mut pass, "grid1d_test.silu", &gate_up_buf, rows, hidden);
        drop(pass);
        engine.queue.submit(Some(encoder.finish()));
        let got = pollster::block_on(engine.read_buffer(&out, (rows * hidden) as usize));
        let check = |row: u32, c: u32| {
            let id = (row * hidden + c) as usize;
            let base = (row * 2 * hidden) as usize;
            let gate = gate_up[base + c as usize];
            let up = gate_up[base + hidden as usize + c as usize];
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            assert!((got[id] - expected).abs() < 1e-3, "row {row} col {c}: got {} expected {expected}", got[id]);
        };
        check(0, 0);
        check(0, hidden - 1);
        check(rows / 2, hidden / 2);
        check(rows - 1, hidden - 1);
    }
}
