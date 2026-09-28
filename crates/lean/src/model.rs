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
//! model's `qwen2.context_length`). Writes into it are plain
//! `copy_buffer_to_buffer` calls recorded in the same encoder as the
//! dispatch that produced the source K/V - GPU-resident, no CPU readback -
//! so a freshly-computed decode step's own K/V is visible to that same
//! step's causal attention (`kv_len` passed to `attn_decode` already
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

/// Row-count (`M`) threshold above which the tiled Q4_0 matmul
/// (`linear_q4_tiled.wgsl`) is used for prefill; below it, the naive
/// per-element kernel (`linear_q4.wgsl`) is faster (see `linear()`'s match
/// arm doc comment) - measured in `docs/runs/2026-09-28-lean-perf.md`.
const TILED_MIN_ROWS: u32 = 16;

/// Row-count (`M`) threshold below which the register-blocked 32x32/TK=16
/// tiled kernel (`linear_q4_tiled_rb.wgsl`/`linear_q8_tiled_rb.wgsl`, this
/// project's own port of t0-web's tile/register-blocking scheme, see that
/// file's header) beats the larger TM=TN=64/MICRO=4 kernel
/// (`linear_q4_tiled.wgsl`); at or above it, the bigger tile's 4x4/16
/// outputs-per-thread reuse wins. Measured on the RTX 3080 (Vulkan) at
/// Qwen2.5-0.5B/Qwen3-1.7B prompt lengths 36/86/256/512/1024/2225 - see
/// this session's run doc for the crossover table. Set from `rows` alone,
/// not the device: the mechanism this threshold tracks (arithmetic
/// intensity per shared-memory tile load crossing over as `M` grows) is a
/// GEMM-shape property, not a vendor-specific one, so the same threshold is
/// expected to hold on other GPUs; only its exact value would need
/// re-measuring if it turned out not to.
const PREFILL_RB_MAX_ROWS: u32 = 512;

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
    _p0: u32,
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
    _p0: u32,
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
    _p0: u32,
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
    gate_w: MatMulWeight,
    gate_b: wgpu::Buffer, // zero
    up_w: MatMulWeight,
    up_b: wgpu::Buffer, // zero
    down_w: MatMulWeight,
    down_b: wgpu::Buffer, // zero
    /// Qwen3-only (`cfg.qk_norm`): per-head RMSNorm gamma, `[head_dim]`
    /// each, applied to q/k right after projection, before RoPE. `None` for
    /// Qwen2 layers.
    q_norm: Option<wgpu::Buffer>,
    k_norm: Option<wgpu::Buffer>,
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
}

impl KvCache {
    pub fn new(engine: &Engine, config: &Qwen2Config, max_ctx: u32) -> Self {
        let per_layer = (config.num_kv_heads * config.head_dim) as u32 * max_ctx;
        let k = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.k"))).collect();
        let v = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.v"))).collect();
        KvCache { k, v, max_ctx, kv_len: 0, num_kv_heads: config.num_kv_heads as u32, head_dim: config.head_dim as u32 }
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

fn gguf_f32<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<wgpu::Buffer> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let n: usize = info.shape().iter().product();
    let bytes = reader.tensor_data(name)?;
    let data = crate::gguf::dequantize_for(info.dtype(), &bytes, n);
    Ok(engine.buf_f32(&data, name))
}

fn gguf_matmul<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<MatMulWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let bytes = reader.tensor_data(name)?;
    Ok(load_matmul_weight_gguf(engine, name, &shape, info.dtype(), &bytes))
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
fn unpermute_rope_rows(bytes: &[u8], n_heads: usize, head_dim: usize, out_dim: usize) -> Vec<u8> {
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
    Ok(load_matmul_weight_gguf(engine, name, &shape, info.dtype(), &bytes))
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
        let tied_lm_head = config.tied_embeddings.then(|| load_matmul_weight_gguf(engine, "token_embd(lm_head)", &embed_shape, embed_dtype, &embed_bytes));
        let embed = load_embedding_table_gguf(engine, "token_embd", &embed_shape, embed_dtype, &embed_bytes);
        drop(embed_bytes);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{i}");
            let (q_b, k_b, v_b) = if config.has_qkv_bias {
                (gguf_f32(engine, &mut reader, &format!("{p}.attn_q.bias"))?, gguf_f32(engine, &mut reader, &format!("{p}.attn_k.bias"))?, gguf_f32(engine, &mut reader, &format!("{p}.attn_v.bias"))?)
            } else {
                let q_dim = (config.num_heads * config.head_dim) as usize;
                let kv_dim = (config.num_kv_heads * config.head_dim) as usize;
                (engine.buf_f32(&vec![0f32; q_dim], "q_b_zero"), engine.buf_f32(&vec![0f32; kv_dim], "k_b_zero"), engine.buf_f32(&vec![0f32; kv_dim], "v_b_zero"))
            };
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
                gate_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_gate.weight"))?,
                gate_b: engine.buf_f32(&vec![0f32; config.intermediate_size], "gate_b_zero"),
                up_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_up.weight"))?,
                up_b: engine.buf_f32(&vec![0f32; config.intermediate_size], "up_b_zero"),
                down_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_down.weight"))?,
                down_b: engine.buf_f32(&vec![0f32; config.hidden_size], "down_b_zero"),
                q_norm,
                k_norm,
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

    /// Logits for a caller-chosen subset of vocab ids, at every row of
    /// `hidden_states`: the sliced lm-head mechanism (this crate's
    /// consumer survey, gap #5): llm-life reads exactly `[dead, alive]`
    /// logits at every cell's answer position instead of materializing a
    /// `[rows, vocab_size]` buffer. Returns `[rows, token_ids.len()]`,
    /// row-major, already read back to the CPU.
    pub async fn lm_head_sliced(&self, engine: &Engine, hidden_states: &wgpu::Buffer, rows: u32, token_ids: &[u32]) -> Vec<f32> {
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lm_head_sliced") });
        let (gathered, hidden_dim) = gather_dequant_head_rows(engine, &self.pool, &mut encoder, "head_slice", &self.lm_head, token_ids);
        let n = token_ids.len() as u32;
        let w = MatMulWeight::F32 { w: gathered };
        let zero_b = zero_bias(engine, &self.pool, "head_slice.bias", n);
        let logits = linear(engine, &self.pool, &mut encoder, "head_slice.linear", hidden_states, rows, hidden_dim, &w, &zero_b, n, false);
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

#[allow(clippy::too_many_arguments)]
fn linear(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, in_dim: u32, w: &MatMulWeight, b: &wgpu::Buffer, out_dim: u32, fast: bool) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * out_dim) as usize);
    let wgs = (out_dim.div_ceil(16), rows.div_ceil(16), 1);
    match w {
        MatMulWeight::F32 { w } => {
            let dims = pool.uniform(&format!("{key}.dims"), LinearDims { m: rows, k: in_dim, n: out_dim, act: 0 });
            let bg = pool.bind_group(
                key,
                &engine.linear,
                &[
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
                ],
            );
            engine.dispatch(encoder, &engine.linear, &bg, wgs, key);
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
                let dims = pool.uniform(&format!("{ckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
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
                    engine.dispatch(encoder, &engine.linear_q8_decode, &bg, (chunk.rows.div_ceil(4), 1, 1), &ckey);
                } else if fast && rows >= TILED_MIN_ROWS {
                    // Prefill (M >= 16): register-blocked tiled kernel (see
                    // linear_q4_tiled_rb.wgsl's header) - Q8_0 prefill had
                    // no tiled kernel before this session, only the naive
                    // one below.
                    let bg = pool.bind_group(&format!("{ckey}.tiled_rb"), &engine.linear_q8_tiled_rb, &entries);
                    engine.dispatch(encoder, &engine.linear_q8_tiled_rb, &bg, (chunk.rows.div_ceil(32), rows.div_ceil(32), 1), &ckey);
                } else {
                    let bg = pool.bind_group(&ckey, &engine.linear_q8, &entries);
                    engine.dispatch(encoder, &engine.linear_q8, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                }
            }
        }
        MatMulWeight::Q6_K { chunks, blocks_per_row, out_dim: n_total } => {
            // Naive kernel only (see linear_q6k.wgsl's doc comment) - Q6_K
            // is only ever token_embd/output.weight in this crate's models,
            // one matmul per forward, not worth a fast-path yet.
            for chunk in chunks {
                let ckey = format!("{key}.{}", chunk.row_start);
                let dims = pool.uniform(&format!("{ckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
                let bg = pool.bind_group(
                    &ckey,
                    &engine.linear_q6k,
                    &[
                        BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: chunk.ql.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: chunk.qh.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: chunk.scales.as_entire_binding() },
                        BindGroupEntry { binding: 4, resource: chunk.d.as_entire_binding() },
                        BindGroupEntry { binding: 5, resource: b.as_entire_binding() },
                        BindGroupEntry { binding: 6, resource: out.as_entire_binding() },
                        BindGroupEntry { binding: 7, resource: dims.as_entire_binding() },
                    ],
                );
                engine.dispatch(encoder, &engine.linear_q6k, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
            }
        }
        MatMulWeight::Q4_0 { chunks, blocks_per_row, out_dim: n_total } => {
            for chunk in chunks {
                let ckey = format!("{key}.{}", chunk.row_start);
                let dims = pool.uniform(&format!("{ckey}.dims"), LinearQDims { m: rows, k: in_dim, n: chunk.rows, act: 0, blocks_per_row: *blocks_per_row, n_offset: chunk.row_start, n_total: *n_total, _p2: 0 });
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
                    engine.dispatch(encoder, &engine.linear_q4, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                } else if rows == 1 {
                    // Decode: coalesced matvec (llm-wasm's shader_q4_matvec_coalesced.wgsl port).
                    let bg = pool.bind_group(&format!("{ckey}.decode"), &engine.linear_q4_decode, &entries);
                    engine.dispatch(encoder, &engine.linear_q4_decode, &bg, (chunk.rows.div_ceil(4), 1, 1), &ckey);
                } else if (TILED_MIN_ROWS..PREFILL_RB_MAX_ROWS).contains(&rows) {
                    // Prefill, TILED_MIN_ROWS <= M < PREFILL_RB_MAX_ROWS:
                    // register-blocked 32x32/TK=16 kernel (see
                    // linear_q4_tiled_rb.wgsl's header) - faster than the
                    // bigger tile below at short-to-medium prefill lengths.
                    let bg = pool.bind_group(&format!("{ckey}.tiled_rb"), &engine.linear_q4_tiled_rb, &entries);
                    engine.dispatch(encoder, &engine.linear_q4_tiled_rb, &bg, (chunk.rows.div_ceil(32), rows.div_ceil(32), 1), &ckey);
                } else if rows >= PREFILL_RB_MAX_ROWS {
                    // Prefill (M >= PREFILL_RB_MAX_ROWS): tiled matmul
                    // (llm-wasm's shader_q4_tiled.wgsl port). llm-wasm
                    // measured this kernel 3-4x slower than the naive one at
                    // M=1 (tile/barrier overhead not amortized) - it only
                    // pays off once weight reuse across enough rows
                    // outweighs that, hence the size gate rather than "any
                    // M > 1"; above PREFILL_RB_MAX_ROWS it also beats the
                    // smaller register-blocked tile above (see
                    // PREFILL_RB_MAX_ROWS's own doc comment).
                    let bg = pool.bind_group(&format!("{ckey}.tiled"), &engine.linear_q4_tiled, &entries);
                    engine.dispatch(encoder, &engine.linear_q4_tiled, &bg, (chunk.rows.div_ceil(64), rows.div_ceil(64), 1), &ckey);
                } else {
                    // 2 <= M < 16: below the tiled kernel's break-even point
                    // and not the M=1 shape the coalesced matvec assumes -
                    // fall back to the naive per-element kernel.
                    let bg = pool.bind_group(&format!("{ckey}.naive_small_m"), &engine.linear_q4, &entries);
                    engine.dispatch(encoder, &engine.linear_q4, &bg, (chunk.rows.div_ceil(16), rows.div_ceil(16), 1), &ckey);
                }
            }
        }
    }
    out
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
fn apply_lora_proj(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, in_dim: u32, out_buf: &wgpu::Buffer, out_dim: u32, proj: &crate::lora::LoraProj) {
    let zero_r = zero_bias(engine, pool, &format!("{key}.zr"), proj.rank);
    let ab = linear(engine, pool, encoder, &format!("{key}.a"), x, rows, in_dim, &proj.a, &zero_r, proj.rank, false);
    let zero_o = zero_bias(engine, pool, &format!("{key}.zo"), out_dim);
    let delta = linear(engine, pool, encoder, &format!("{key}.b"), &ab, rows, proj.rank, &proj.b, &zero_o, out_dim, false);
    add_inplace(engine, pool, encoder, &format!("{key}.add"), out_buf, &delta, rows * out_dim);
}

/// `linear()` plus, when `lora` is `Some`, that projection's LoRA delta
/// added in place onto the result: the single call site every q/k/v/o
/// projection in this file goes through, so LoRA is applied uniformly
/// across prefill, decode and the chunked/masked forward path.
#[allow(clippy::too_many_arguments)]
fn linear_lora(
    engine: &Engine,
    pool: &Pool,
    encoder: &mut wgpu::CommandEncoder,
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
    let out = linear(engine, pool, encoder, key, x, rows, in_dim, w, b, out_dim, fast);
    if let Some(proj) = lora {
        apply_lora_proj(engine, pool, encoder, &format!("{key}.lora"), x, rows, in_dim, &out, out_dim, proj);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, scale: &wgpu::Buffer, rows: u32, dim: u32, eps: f32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * dim) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), RmsDims { rows, dim, eps, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rmsnorm,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm, &bg, (rows.div_ceil(64), 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn rope(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, buf: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer, rows: u32, heads: u32, head_dim: u32, pos_base: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), RopeDims { rows, heads, head_dim, pos_base });
    let bg = pool.bind_group(
        key,
        &engine.rope,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(encoder, &engine.rope, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

/// Same RoPE as `rope()` but each row's absolute position comes from a
/// caller-supplied buffer (`ForwardSpec::positions`) instead of a
/// contiguous `pos_base + row` run: see `shaders/rope_positions.wgsl`.
#[allow(clippy::too_many_arguments)]
fn rope_positions(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, buf: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer, positions: &wgpu::Buffer, rows: u32, heads: u32, head_dim: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), RopePosDims { rows, heads, head_dim, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rope_positions,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: positions.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(encoder, &engine.rope_positions, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

fn add_inplace(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), AddDims { len, _p0: 0, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.add_inplace,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.add_inplace, &bg, (len.div_ceil(256), 1, 1), key);
}

#[allow(clippy::too_many_arguments)]
fn silu_mul(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, gate: &wgpu::Buffer, up: &wgpu::Buffer, rows: u32, hidden: u32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * hidden) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), SiluDims { rows, hidden, _p0: 0, _p1: 0 });
    let bg = pool.bind_group(
        key,
        &engine.silu_mul,
        &[
            BindGroupEntry { binding: 0, resource: gate.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: up.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.silu_mul, &bg, ((rows * hidden).div_ceil(256), 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn qkv_proj(
    engine: &Engine,
    pool: &Pool,
    encoder: &mut wgpu::CommandEncoder,
    key: &str,
    x: &wgpu::Buffer,
    rows: u32,
    cfg: &Qwen2Config,
    layer: &LayerWeights,
    fast: bool,
    lora: Option<&crate::lora::LoraLayer>,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let hidden = cfg.hidden_size as u32;
    let q_dim = (cfg.num_heads * cfg.head_dim) as u32;
    let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u32;
    let q = linear_lora(engine, pool, encoder, &format!("{key}.q"), x, rows, hidden, &layer.q_w, &layer.q_b, q_dim, fast, lora.map(|l| &l.q));
    let k = linear_lora(engine, pool, encoder, &format!("{key}.k"), x, rows, hidden, &layer.k_w, &layer.k_b, kv_dim, fast, lora.map(|l| &l.k));
    let v = linear_lora(engine, pool, encoder, &format!("{key}.v"), x, rows, hidden, &layer.v_w, &layer.v_b, kv_dim, fast, lora.map(|l| &l.v));
    // Qwen3 only (`layer.q_norm`/`k_norm` set): per-head RMSNorm on q/k
    // before RoPE. `q`/`k` are `[rows, heads*head_dim]` row-major, so
    // reinterpreting the same flat buffer as `[rows*heads, head_dim]` for
    // `rmsnorm()` normalizes each head's slice independently in place - no
    // dedicated per-head kernel needed (see the qwen3 survey's open
    // question, resolved this way).
    let q = match &layer.q_norm {
        Some(scale) => rmsnorm(engine, pool, encoder, &format!("{key}.qnorm"), &q, scale, rows * cfg.num_heads as u32, cfg.head_dim as u32, cfg.rms_norm_eps),
        None => q,
    };
    let k = match &layer.k_norm {
        Some(scale) => rmsnorm(engine, pool, encoder, &format!("{key}.knorm"), &k, scale, rows * cfg.num_kv_heads as u32, cfg.head_dim as u32, cfg.rms_norm_eps),
        None => k,
    };
    (q, k, v)
}

#[allow(clippy::too_many_arguments)]
fn mlp(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, layer: &LayerWeights, fast: bool) -> wgpu::Buffer {
    let hidden = cfg.hidden_size as u32;
    let inter = cfg.intermediate_size as u32;
    let gate = linear(engine, pool, encoder, &format!("{key}.gate"), x, rows, hidden, &layer.gate_w, &layer.gate_b, inter, fast);
    let up = linear(engine, pool, encoder, &format!("{key}.up"), x, rows, hidden, &layer.up_w, &layer.up_b, inter, fast);
    let gated = silu_mul(engine, pool, encoder, &format!("{key}.silu"), &gate, &up, rows, inter);
    linear(engine, pool, encoder, &format!("{key}.down"), &gated, rows, inter, &layer.down_w, &layer.down_b, hidden, fast)
}

fn embed_gather(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, model: &GpuModel, token_ids: &[u32]) -> wgpu::Buffer {
    let rows = token_ids.len() as u32;
    let hidden = model.config.hidden_size as u32;
    let ids_buf = pool.upload_u32("embed.ids", token_ids);
    let out = pool.data("embed.out", (rows * hidden) as usize);
    if let EmbeddingTable::Q6_K(t) = &model.embed {
        let dims = pool.uniform("embed.dims", GatherDims { rows, hidden, blocks_per_row: t.blocks_per_row, _p0: 0 });
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
        engine.dispatch(encoder, &engine.embed_gather_q6k, &bg, ((rows * hidden).div_ceil(256), 1, 1), "embed");
        return out;
    }
    let (pipeline, table) = match &model.embed {
        EmbeddingTable::Q4_0(t) => (&engine.embed_gather_q4, t),
        EmbeddingTable::Q8_0(t) => (&engine.embed_gather_q8, t),
        EmbeddingTable::Q6_K(_) => unreachable!("handled above"),
    };
    let dims = pool.uniform("embed.dims", GatherDims { rows, hidden, blocks_per_row: table.blocks_per_row, _p0: 0 });
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
    engine.dispatch(encoder, pipeline, &bg, ((rows * hidden).div_ceil(256), 1, 1), "embed");
    out
}

/// GPU-side scatter of `rows` freshly-computed K/V rows
/// (`[rows, kv_heads, head_dim]`, row-major) into the cache's
/// `[kv_head, kv_base+row, head_dim]` layout, via one `copy_buffer_to_buffer`
/// per (row, kv_head) recorded into the same encoder as the dispatch that
/// produced `src` - no CPU readback, so this can run before the attention
/// dispatch that needs to see it (decode's self-attention).
fn scatter_kv_gpu(encoder: &mut wgpu::CommandEncoder, cache_buf: &wgpu::Buffer, src: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, kv_base: u32, max_ctx: u32) {
    let kv_heads = cfg.num_kv_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let row_bytes = (head_dim * 4) as u64;
    for row in 0..rows {
        for h in 0..kv_heads {
            let src_off = (((row * kv_heads + h) * head_dim) as u64) * 4;
            let dst_pos = kv_base + row;
            let dst_off = (((h * max_ctx + dst_pos) * head_dim) as u64) * 4;
            encoder.copy_buffer_to_buffer(src, src_off, cache_buf, dst_off, row_bytes);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_prefill(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, seq: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    // Concatenated-heads width (`num_heads * head_dim`): equal to
    // `cfg.hidden_size` for Qwen2 (head_dim is derived that way) but *not*
    // for Qwen3, where head_dim=128 is explicit and 16*128=2048 != 1024 -
    // see `qkv_proj`'s doc comment on the same distinction.
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let out = pool.data(&format!("{key}.out"), (seq * hidden) as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{key}.dims"),
        AttnPrefillDims { seq, n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, scale, _p0: 0, _p1: 0, _p2: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_prefill,
        &[
            BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: k.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    // wg.y = query tile index (256 rows/tile, see attn_prefill.wgsl's doc
    // comment on why this scales with `seq` instead of a fixed dispatch).
    engine.dispatch(encoder, &engine.attn_prefill, &bg, (cfg.num_heads as u32, seq.div_ceil(256), 1), key);
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
/// actually shorten the loop. Chosen from `kv_len` alone (a workload-size
/// fact), never from a timing measurement.
const SPLIT_CHUNK: u32 = 128;
/// Upper bound on split count and the partial-result buffers' per-head
/// stride; must match `MAX_SPLITS` in `attn_decode_split.wgsl` and
/// `attn_decode_reduce.wgsl` exactly. Fixed so those buffers never need to
/// regrow as `kv_len` grows one token per decode step.
const MAX_SPLITS: u32 = 32;

/// `kv_len <= SPLIT_CHUNK` returns `(1, kv_len)` (no split: `attn_decode`
/// picks the single-workgroup kernel). Otherwise `num_splits =
/// min(MAX_SPLITS, ceil(kv_len / SPLIT_CHUNK))` and `chunk =
/// ceil(kv_len / num_splits)` (recomputed from the actual split count so the
/// chunks partition `[0, kv_len)` exactly, with no split ever going idle).
fn decode_split_plan(kv_len: u32) -> (u32, u32) {
    if kv_len <= SPLIT_CHUNK {
        return (1, kv_len.max(1));
    }
    let num_splits = kv_len.div_ceil(SPLIT_CHUNK).min(MAX_SPLITS);
    let chunk = kv_len.div_ceil(num_splits).max(1);
    (num_splits, chunk)
}

#[allow(clippy::too_many_arguments)]
fn attn_decode(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, q: &wgpu::Buffer, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, kv_len: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let out = pool.data(&format!("{key}.out"), hidden as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let n_heads = cfg.num_heads as u32;
    let head_dim = cfg.head_dim as u32;

    let (num_splits, chunk) = decode_split_plan(kv_len);
    if num_splits <= 1 {
        // Short context: the plain single-workgroup-per-head kernel, exactly
        // as before this change (no split/reduce overhead).
        let dims = pool.uniform(
            &format!("{key}.dims"),
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
                BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ],
        );
        engine.dispatch(encoder, pipeline, &bg, (n_heads, 1, 1), key);
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

    let partial_m = pool.data(&format!("{key}.partial_m"), (n_heads * MAX_SPLITS) as usize);
    let partial_l = pool.data(&format!("{key}.partial_l"), (n_heads * MAX_SPLITS) as usize);
    let partial_acc = pool.data(&format!("{key}.partial_acc"), (n_heads * MAX_SPLITS * head_dim) as usize);

    let split_dims = pool.uniform(
        &format!("{key}.split_dims"),
        AttnDecodeSplitDims { n_heads, n_kv_heads: cfg.num_kv_heads as u32, head_dim, kv_len, max_ctx, scale, chunk, _p0: 0 },
    );
    let split_bg = pool.bind_group(
        &format!("{key}.split"),
        split_pipeline,
        &[
            BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: partial_m.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: partial_l.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: partial_acc.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: split_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, split_pipeline, &split_bg, (n_heads, num_splits, 1), &format!("{key}.split"));

    let reduce_dims = pool.uniform(&format!("{key}.reduce_dims"), AttnDecodeReduceDims { n_heads, head_dim, num_splits, _p0: 0 });
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
    engine.dispatch(encoder, reduce_pipeline, &reduce_bg, (n_heads, 1, 1), &format!("{key}.reduce"));
    out
}

/// Pluggable-mask GQA attention over a resident-prefix KV cache: `t` query
/// rows attend `kv_total = prefix_len + t` keys already scattered into
/// `k_cache`/`v_cache`, gated by an explicit bitset instead of an implicit
/// causal rule: see `shaders/attn_chunk_masked.wgsl`.
#[allow(clippy::too_many_arguments)]
fn attn_chunk_masked(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, q: &wgpu::Buffer, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, mask: &wgpu::Buffer, t: u32, kv_total: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let hidden = (cfg.num_heads * cfg.head_dim) as u32;
    let out = pool.data(&format!("{key}.out"), (t * hidden) as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{key}.dims"),
        AttnChunkDims { t, n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, kv_total, max_ctx, scale, _p0: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_chunk_masked,
        &[
            BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: mask.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.attn_chunk_masked, &bg, (cfg.num_heads as u32, t.div_ceil(256), 1), key);
    out
}

/// Gathers + dequantizes `token_ids` (absolute vocab ids) out of `w` (must
/// be `MatMulWeight::Q8_0`: this model's lm head) into a small contiguous
/// F32 `[token_ids.len(), hidden]` buffer, one dispatch per id (llm-life's
/// sliced sets are 1-2 ids; not worth a multi-row-per-dispatch path yet).
/// Returns the buffer and `hidden` (the caller already knows `hidden`, but
/// returning it here keeps this fn self-contained for a future non-Qwen2
/// caller). See `shaders/gather_dequant_q8_rows.wgsl`.
fn gather_dequant_head_rows(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, w: &MatMulWeight, token_ids: &[u32]) -> (wgpu::Buffer, u32) {
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
        engine.dispatch(encoder, &engine.gather_dequant_q8_rows, &bg, (hidden.div_ceil(64), 1, 1), &ckey);
    }
    (out, hidden)
}

/// Prefill: runs every layer over the whole prompt with causal attention
/// (no cache read needed - attends directly over this call's own q/k/v),
/// GPU-scatters every position's K/V into `cache`, and returns the last
/// position's logits ([vocab]).
pub async fn forward_prefill(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32> {
    let cfg = &model.config;
    let seq = token_ids.len() as u32;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill") });
    let x = embed_gather(engine, pool, &mut encoder, model, token_ids);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("layer{i}");
        let normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.norm"), &x, &layer.attn_norm, seq, hidden, cfg.rms_norm_eps);
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        let (q, k, v) = qkv_proj(engine, pool, &mut encoder, &format!("{key}.qkv"), &normed, seq, cfg, layer, model.fast_kernels, lora_layer);
        rope(engine, pool, &mut encoder, &format!("{key}.ropeq"), &q, cos, sin, seq, cfg.num_heads as u32, cfg.head_dim as u32, 0);
        rope(engine, pool, &mut encoder, &format!("{key}.ropek"), &k, cos, sin, seq, cfg.num_kv_heads as u32, cfg.head_dim as u32, 0);
        let attn_out = attn_prefill(engine, pool, &mut encoder, &format!("{key}.attn"), &q, &k, &v, seq, cfg);
        scatter_kv_gpu(&mut encoder, &cache.k[i], &k, seq, cfg, 0, cache.max_ctx);
        scatter_kv_gpu(&mut encoder, &cache.v[i], &v, seq, cfg, 0, cache.max_ctx);

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        let o = linear_lora(engine, pool, &mut encoder, &format!("{key}.wo"), &attn_out, seq, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add1"), &x, &o, seq * hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, seq, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, &mut encoder, &format!("{key}.mlp"), &ffn_normed, seq, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add2"), &x, &mlp_out, seq * hidden);

        // Flush per layer - see `Engine::flush_encoder`'s doc comment for
        // why a single encoder covering every layer hangs on Metal at
        // Qwen3-0.6B's depth (28 layers).
        engine.flush_encoder(&mut encoder, "prefill");
    }

    let normed_final = rmsnorm(engine, pool, &mut encoder, "out_norm", &x, &model.out_norm, seq, hidden, cfg.rms_norm_eps);
    let logits = linear(engine, pool, &mut encoder, "lm_head", &normed_final, seq, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, false);
    // Only the last position's row feeds the next token (see this fn's
    // return value below), so only it needs masking - `row_offset =
    // (seq-1)*vocab` into the `[seq, vocab]` logits buffer.
    if let Some(mask) = mask {
        mask_logits_gpu(engine, pool, &mut encoder, "prefill_mask", &logits, mask, cfg.vocab_size as u32, (seq - 1) * cfg.vocab_size as u32);
    }

    engine.queue.submit(Some(encoder.finish()));
    let vocab = cfg.vocab_size;
    let all_logits = engine.read_buffer(&logits, (seq as usize) * vocab).await;
    cache.kv_len = seq;
    all_logits[(seq as usize - 1) * vocab..].to_vec()
}

/// Applies an allowed-token bitset to one row of `logits` (`vocab` wide, at
/// element offset `row_offset`) in place, on the GPU, before argmax or
/// readback - the mechanism behind per-step constrained decoding. A no-op
/// dispatch is never recorded when `mask` is `None`: unmasked generation
/// then has exactly the same dispatch sequence (and result) as before this
/// feature existed.
#[allow(clippy::too_many_arguments)]
fn mask_logits_gpu(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, logits: &wgpu::Buffer, mask: &wgpu::Buffer, vocab: u32, row_offset: u32) {
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
    engine.dispatch(encoder, &engine.mask_logits, &bg, (vocab.div_ceil(256), 1, 1), key);
}

fn argmax_gpu(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, logits: &wgpu::Buffer, vocab: u32) -> wgpu::Buffer {
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
    engine.dispatch(encoder, &engine.argmax, &bg, (1, 1, 1), key);
    out
}

/// Records one decode step's whole layer stack (embed through the lm-head
/// logits) into `encoder`, without submitting or reading anything back -
/// shared by `forward_decode_step` (full-logits readback, for samplers) and
/// `forward_decode_step_argmax` (single-index readback, the fast path),
/// so the two only differ in what they append after the layer stack.
#[allow(clippy::too_many_arguments)]
fn decode_layers(engine: &Engine, model: &GpuModel, encoder: &mut wgpu::CommandEncoder, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> wgpu::Buffer {
    let cfg = &model.config;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;
    let pos = cache.kv_len;

    let x = embed_gather(engine, pool, encoder, model, &[token_id]);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("dec_layer{i}");
        let normed = rmsnorm(engine, pool, encoder, &format!("{key}.norm"), &x, &layer.attn_norm, 1, hidden, cfg.rms_norm_eps);
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        let (q, k, v) = qkv_proj(engine, pool, encoder, &format!("{key}.qkv"), &normed, 1, cfg, layer, model.fast_kernels, lora_layer);
        rope(engine, pool, encoder, &format!("{key}.ropeq"), &q, cos, sin, 1, cfg.num_heads as u32, cfg.head_dim as u32, pos);
        rope(engine, pool, encoder, &format!("{key}.ropek"), &k, cos, sin, 1, cfg.num_kv_heads as u32, cfg.head_dim as u32, pos);

        scatter_kv_gpu(encoder, &cache.k[i], &k, 1, cfg, pos, cache.max_ctx);
        scatter_kv_gpu(encoder, &cache.v[i], &v, 1, cfg, pos, cache.max_ctx);
        let attn_out = attn_decode(engine, pool, encoder, &format!("{key}.attn"), &q, &cache.k[i], &cache.v[i], pos + 1, cache.max_ctx, cfg);

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        let o = linear_lora(engine, pool, encoder, &format!("{key}.wo"), &attn_out, 1, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        add_inplace(engine, pool, encoder, &format!("{key}.add1"), &x, &o, hidden);

        let ffn_normed = rmsnorm(engine, pool, encoder, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, 1, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, encoder, &format!("{key}.mlp"), &ffn_normed, 1, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, encoder, &format!("{key}.add2"), &x, &mlp_out, hidden);

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

    let normed_final = rmsnorm(engine, pool, encoder, "dec_out_norm", &x, &model.out_norm, 1, hidden, cfg.rms_norm_eps);
    let logits = linear(engine, pool, encoder, "dec_lm_head", &normed_final, 1, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, model.fast_kernels);
    if let Some(mask) = mask {
        mask_logits_gpu(engine, pool, encoder, "dec_mask", &logits, mask, cfg.vocab_size as u32, 0);
    }
    logits
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
    let logits = decode_layers(engine, model, &mut encoder, cache, token_id, cos, sin, mask);
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
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("decode_argmax") });
    let logits = decode_layers(engine, model, &mut encoder, cache, token_id, cos, sin, mask);
    let idx = argmax_gpu(engine, &model.pool, &mut encoder, "dec_argmax", &logits, model.config.vocab_size as u32);
    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len += 1;
    engine.read_u32(&idx).await
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
        let _ = decode_layers(engine, model, &mut encoder, cache, tok, cos, sin, None);
        engine.queue.submit(Some(encoder.finish()));
        cache.kv_len += 1;
    }
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill_suffix_last") });
    let logits = decode_layers(engine, model, &mut encoder, cache, *last, cos, sin, mask);
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
    let x = embed_gather(engine, pool, &mut encoder, model, token_ids);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("chunk_layer{i}");
        let lora_layer = model.lora.as_ref().map(|l| &l.layers[i]);
        let normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.norm"), &x, &layer.attn_norm, t, hidden, cfg.rms_norm_eps);
        let (q, k, v) = qkv_proj(engine, pool, &mut encoder, &format!("{key}.qkv"), &normed, t, cfg, layer, model.fast_kernels, lora_layer);
        rope_positions(engine, pool, &mut encoder, &format!("{key}.ropeq"), &q, cos, sin, &pos_buf, t, cfg.num_heads as u32, cfg.head_dim as u32);
        rope_positions(engine, pool, &mut encoder, &format!("{key}.ropek"), &k, cos, sin, &pos_buf, t, cfg.num_kv_heads as u32, cfg.head_dim as u32);

        scatter_kv_gpu(&mut encoder, &cache.k[i], &k, t, cfg, prefix_len, cache.max_ctx);
        scatter_kv_gpu(&mut encoder, &cache.v[i], &v, t, cfg, prefix_len, cache.max_ctx);
        let attn_out = attn_chunk_masked(engine, pool, &mut encoder, &format!("{key}.attn"), &q, &cache.k[i], &cache.v[i], &mask_buf, t, kv_total, cache.max_ctx, cfg);

        let attn_dim = (cfg.num_heads * cfg.head_dim) as u32;
        let o = linear_lora(engine, pool, &mut encoder, &format!("{key}.wo"), &attn_out, t, attn_dim, &layer.o_w, &layer.o_b, hidden, model.fast_kernels, lora_layer.map(|l| &l.o));
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add1"), &x, &o, t * hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, t, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, &mut encoder, &format!("{key}.mlp"), &ffn_normed, t, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add2"), &x, &mlp_out, t * hidden);

        // See `Engine::flush_encoder`'s doc comment.
        engine.flush_encoder(&mut encoder, "chunk");
    }

    let normed_final = rmsnorm(engine, pool, &mut encoder, "chunk_out_norm", &x, &model.out_norm, t, hidden, cfg.rms_norm_eps);

    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len = kv_total;
    normed_final
}
