//! Single-threaded CPU forward pass: the CPU fallback rung below the WebGPU
//! rung in `model.rs`. Same op list, same config (`Qwen2Config`), same GGUF parsing
//! (`GgufReader`), same tokenizer/chat-template code as the GPU path - only
//! the destination of each weight tensor differs: a plain `Vec<u8>` holding
//! the GGUF's on-disk Q4_0/Q8_0/Q6_K block bytes unchanged (no repack, no
//! dequantize-to-f32-then-requantize), dequantized in-kernel per dot
//! product by `cpu_kernels::dot_q4_0`/`dot_q8_0`/`dot_q6_k`. This is a second rung, not
//! a second model definition: everything architecture-specific lives in
//! `config.rs`/`gguf.rs`, shared unchanged with `model.rs`'s GPU path.
//!
//! KV cache layout matches `model.rs`'s doc comment exactly: per layer, two
//! flat `Vec<f32>` buffers `[kv_heads, max_ctx, head_dim]`, head-major and
//! contiguous per head - not a ring buffer, `kv_len` only grows.
//!
//! Numerics reference: same as `model.rs` (`reference/gen_fixture.py`'s HF
//! transformers forward). RoPE is split-half ("NeoX-style", `rotate_half`),
//! matching `shaders/rope_neox.wgsl`'s convention exactly - see that file's
//! doc comment for why split-half and not interleaved pairs.

use anyhow::{Context, Result};
use std::io::{Read, Seek};

use crate::config::{config_from_gguf, Architecture, Qwen2Config};
use crate::cpu_kernels::{dot_q4_0, dot_q6_k, dot_q8_0, F4};
use crate::gguf::{dequantize_for, GgmlDtype, GgufReader};
use crate::model::unpermute_rope_rows;

/// A CPU-resident matmul weight (`shape = [out_dim, in_dim]`), holding the
/// GGUF's on-disk bytes for Q4_0/Q8_0 unchanged (row-major, block-contiguous,
/// the same bytes `quant.rs::load_matmul_weight_gguf` repacks for the GPU
/// path, here left exactly as read). F32/F16 tensors are dequantized once at
/// load time (same as the GPU path's `gguf_f32` helper) since there is no
/// per-dot-product win to chasing their (already dense) bytes further.
#[allow(non_camel_case_types)]
pub enum CpuWeight {
    F32 { data: Vec<f32>, out_dim: usize, in_dim: usize },
    Q4_0 { bytes: Vec<u8>, out_dim: usize, in_dim: usize, blocks_per_row: usize },
    Q8_0 { bytes: Vec<u8>, out_dim: usize, in_dim: usize, blocks_per_row: usize },
    Q6_K { bytes: Vec<u8>, out_dim: usize, in_dim: usize, blocks_per_row: usize },
}

impl CpuWeight {
    fn dims(&self) -> (usize, usize) {
        match self {
            CpuWeight::F32 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
            CpuWeight::Q4_0 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
            CpuWeight::Q8_0 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
            CpuWeight::Q6_K { out_dim, in_dim, .. } => (*out_dim, *in_dim),
        }
    }

    /// Row `row`'s output value: dot product of that output row's weights
    /// against `x` (`x.len() == in_dim`). One row = one output scalar - the
    /// same op boundary `linear_q4.wgsl`/`linear_q8.wgsl`/`linear_q6k.wgsl`'s
    /// per-thread work unit uses on the GPU path.
    #[inline]
    fn dot_row(&self, row: usize, x: &[f32]) -> f32 {
        match self {
            CpuWeight::F32 { data, in_dim, .. } => {
                let base = row * in_dim;
                data[base..base + in_dim].iter().zip(x).map(|(w, v)| w * v).sum()
            }
            CpuWeight::Q4_0 { bytes, blocks_per_row, .. } => {
                let bytes_per_row = blocks_per_row * 18;
                dot_q4_0(&bytes[row * bytes_per_row..(row + 1) * bytes_per_row], x)
            }
            CpuWeight::Q8_0 { bytes, blocks_per_row, .. } => {
                let bytes_per_row = blocks_per_row * 34;
                dot_q8_0(&bytes[row * bytes_per_row..(row + 1) * bytes_per_row], x)
            }
            CpuWeight::Q6_K { bytes, blocks_per_row, .. } => {
                let bytes_per_row = blocks_per_row * 210;
                dot_q6_k(&bytes[row * bytes_per_row..(row + 1) * bytes_per_row], x)
            }
        }
    }
}

impl CpuWeight {
    /// Output row `row`'s weights as f32 into `out` (`out.len() == in_dim`),
    /// the same per-element values as `gguf::dequantize_for` (`(q - 8) *
    /// scale` for Q4_0, `q * scale` for Q8_0).
    fn dequant_row(&self, row: usize, out: &mut [f32]) {
        match self {
            CpuWeight::F32 { data, in_dim, .. } => out.copy_from_slice(&data[row * in_dim..(row + 1) * in_dim]),
            CpuWeight::Q4_0 { bytes, blocks_per_row, .. } => {
                let n = blocks_per_row * 18;
                for (block, o) in bytes[row * n..(row + 1) * n].chunks_exact(18).zip(out.chunks_exact_mut(32)) {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..16 {
                        let byte = block[2 + j];
                        o[j] = ((byte & 0x0F) as f32 - 8.0) * scale;
                        o[16 + j] = ((byte >> 4) as f32 - 8.0) * scale;
                    }
                }
            }
            CpuWeight::Q8_0 { bytes, blocks_per_row, .. } => {
                let n = blocks_per_row * 34;
                for (block, o) in bytes[row * n..(row + 1) * n].chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..32 {
                        o[j] = (block[2 + j] as i8) as f32 * scale;
                    }
                }
            }
            CpuWeight::Q6_K { bytes, blocks_per_row, .. } => {
                let n = blocks_per_row * 210;
                out.copy_from_slice(&crate::gguf::dequantize_q6_k(&bytes[row * n..(row + 1) * n], out.len()));
            }
        }
    }
}

/// f32 dot product with four lane accumulators (`acc[l]` sums the
/// elements `k = l mod 4`, in order), reduced as `(a0 + a1) + (a2 + a3)`.
/// `dot_tile` below computes every element of its tile with exactly this
/// sequence of operations, so an element's value does not depend on whether
/// it landed in a full tile or an edge. `a.len()` is a multiple of 4.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = F4::zero();
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        acc = acc.add_mul(F4::load(cb), F4::load(ca));
    }
    acc.sum()
}

/// 4 activation rows by 4 weight rows: sixteen `dot_f32`s sharing their
/// loads (each 4-lane chunk of a row is read once for four products), the
/// register-blocked inner kernel of the multi-row path; 16 accumulators in
/// SIMD registers (`cpu_kernels::F4`).
#[inline]
fn dot_tile(x: [&[f32]; 4], w: [&[f32]; 4]) -> [[f32; 4]; 4] {
    let mut acc = [[F4::zero(); 4]; 4];
    let n = x[0].len();
    let mut k = 0;
    while k + 4 <= n {
        let xs = [F4::load(&x[0][k..]), F4::load(&x[1][k..]), F4::load(&x[2][k..]), F4::load(&x[3][k..])];
        let ws = [F4::load(&w[0][k..]), F4::load(&w[1][k..]), F4::load(&w[2][k..]), F4::load(&w[3][k..])];
        for i in 0..4 {
            for j in 0..4 {
                acc[i][j] = acc[i][j].add_mul(xs[i], ws[j]);
            }
        }
        k += 4;
    }
    acc.map(|row| row.map(F4::sum))
}

/// Output columns per unit of work of the multi-row path: their weight
/// rows are dequantized once (`COLS_PER_TASK * in_dim` floats, cache
/// resident) and reused for every activation row. A multiple of
/// `dot_tile`'s 4.
const COLS_PER_TASK: usize = 16;

/// Output columns `cols` of `linear` for every row, written column-major
/// into `out_t` (`[cols.len(), rows]`): the multi-row path's unit of work.
/// Each element is `dot_f32(weight row, activation row) + bias` computed by
/// the same operation sequence whether it falls in a 4x4 `dot_tile` or an
/// edge, so any split of rows or columns (tiles, threads) gives the same
/// bits.
fn linear_cols(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], cols: std::ops::Range<usize>, out_t: &mut [f32]) {
    let ncols = cols.len();
    let mut wrows = vec![0f32; ncols * in_dim];
    for (j, c) in cols.clone().enumerate() {
        w.dequant_row(c, &mut wrows[j * in_dim..(j + 1) * in_dim]);
    }
    let wrow = |j: usize| &wrows[j * in_dim..(j + 1) * in_dim];
    let xrow = |r: usize| &x[r * in_dim..(r + 1) * in_dim];
    let bias = |j: usize| b[cols.start + j];
    let full_c = ncols / 4 * 4;
    let mut r = 0;
    while r + 4 <= rows {
        let xs = [xrow(r), xrow(r + 1), xrow(r + 2), xrow(r + 3)];
        for c0 in (0..full_c).step_by(4) {
            let t = dot_tile(xs, [wrow(c0), wrow(c0 + 1), wrow(c0 + 2), wrow(c0 + 3)]);
            for (i, ti) in t.iter().enumerate() {
                for (j, v) in ti.iter().enumerate() {
                    out_t[(c0 + j) * rows + r + i] = v + bias(c0 + j);
                }
            }
        }
        for j in full_c..ncols {
            for (i, xr) in xs.iter().enumerate() {
                out_t[j * rows + r + i] = dot_f32(wrow(j), xr) + bias(j);
            }
        }
        r += 4;
    }
    for rr in r..rows {
        for j in 0..ncols {
            out_t[j * rows + rr] = dot_f32(wrow(j), xrow(rr)) + bias(j);
        }
    }
}

/// Multi-row `linear` (prefill, chunks): dequantize each weight row once
/// per call instead of once per output element, and run 4x4 register
/// tiles. Column blocks are spread over rayon's pool under the `threads`
/// feature (pure capability read, as in `linear_threads`); same bits either
/// way (`linear_cols`).
fn linear_multi_row(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out_t = vec![0f32; rows * out_dim];
    let block = |(i, chunk): (usize, &mut [f32])| {
        let c0 = i * COLS_PER_TASK;
        linear_cols(x, rows, in_dim, w, b, c0..(c0 + chunk.len() / rows), chunk);
    };
    #[cfg(feature = "threads")]
    {
        // Inside a team (`cpu_team::with_team`): the same column blocks as
        // team items.
        struct OutPtr(*mut f32);
        // SAFETY: items write disjoint column blocks of `out_t`.
        unsafe impl Sync for OutPtr {}
        impl OutPtr {
            fn get(&self) -> *mut f32 {
                self.0
            }
        }
        let (len, task) = (out_t.len(), COLS_PER_TASK * rows);
        let ptr = OutPtr(out_t.as_mut_ptr());
        let item = |i: usize| {
            let start = i * task;
            // SAFETY: block `i` is `[start, min(start + task, len))`, in
            // bounds and disjoint from every other block.
            let chunk = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(start), task.min(len - start)) };
            block((i, chunk));
        };
        if !crate::cpu_team::team_for(len.div_ceil(task), &item) {
            use rayon::prelude::*;
            out_t.par_chunks_mut(task).enumerate().for_each(block);
        }
    }
    #[cfg(not(feature = "threads"))]
    out_t.chunks_mut(COLS_PER_TASK * rows).enumerate().for_each(block);
    let mut out = vec![0f32; rows * out_dim];
    for c in 0..out_dim {
        for r in 0..rows {
            out[r * out_dim + c] = out_t[c * rows + r];
        }
    }
    out
}

/// `x`: `[rows, in_dim]` row-major. Returns `[rows, out_dim]` row-major, `+
/// bias` per output column - same shape/semantics as `model.rs::linear`'s
/// GPU kernel, computed as `rows * out_dim` independent dot products (no
/// blocking/tiling: this is the reference-correctness rung, see this
/// crate's CPU-fallback plan on why a fast CPU kernel is out of scope for
/// this pass). Each output element `out[r*out_dim+c]` depends only on `x`'s
/// row `r` and weight row `c`, never on any other output element - so
/// splitting this loop across threads (see `linear_threads` below) changes
/// only which thread computes which element, not the arithmetic each
/// element does. Unlike a tree-reduction split, this makes the threaded and
/// single-thread paths bit-for-bit identical, not just token-exact.
///
/// Two or more rows (`in_dim` a multiple of 4) take `linear_multi_row` instead (same values up to
/// float summation order; a one-row decode step keeps the per-row dot
/// kernels the fixtures were gated on).
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Vec<f32> {
    if rows >= 2 && in_dim.is_multiple_of(4) {
        return linear_multi_row(x, rows, in_dim, w, b, out_dim);
    }
    #[cfg(feature = "threads")]
    {
        if let Some(out) = linear_threads(x, rows, in_dim, w, b, out_dim) {
            return out;
        }
    }
    linear_serial(x, rows, in_dim, w, b, out_dim)
}

fn linear_serial(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Vec<f32> {
    let (w_out, w_in) = w.dims();
    debug_assert_eq!(w_out, out_dim);
    debug_assert_eq!(w_in, in_dim);
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for c in 0..out_dim {
            out[r * out_dim + c] = w.dot_row(c, xr) + b[c];
        }
    }
    out
}

/// Threaded rung: the same `rows * out_dim` independent dot products as
/// `linear_serial`, spread across rayon's global thread pool. Thread count
/// is a pure capability read, never tuned or measured at runtime -
/// `rayon::current_num_threads()` reflects `std::thread::available_parallelism()`
/// on native (rayon's own default pool sizing) and whatever
/// `initThreadPool(n)` set on wasm (`lib.rs`'s `wasm-mt`-gated re-export) -
/// so this function makes no bandwidth/timing measurement of its own, only
/// a capability check plus a fixed, shape-based minimum-work threshold
/// (`MIN_WORK_PER_THREAD`) so small matvecs (e.g. a future small per-head
/// op) aren't handed to the pool for less work than the dispatch itself
/// costs. Returns `None` when the threaded path isn't worth taking
/// (pool size 1, or below the threshold) so the caller falls back to
/// `linear_serial` - same output either way, see `linear`'s doc comment on
/// why the two paths are bit-identical, not just token-exact.
#[cfg(feature = "threads")]
fn linear_threads(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Option<Vec<f32>> {
    use rayon::prelude::*;

    const MIN_WORK_PER_THREAD: usize = 64;

    let (w_out, w_in) = w.dims();
    debug_assert_eq!(w_out, out_dim);
    debug_assert_eq!(w_in, in_dim);

    // Inside a decode step's team (`cpu_team::with_team`): blocks of
    // TEAM_ITEM output columns as team items, no rayon call.
    if rows == 1 {
        const TEAM_ITEM: usize = 32;
        struct OutPtr(*mut f32);
        // SAFETY: items write disjoint column ranges of `out`.
        unsafe impl Sync for OutPtr {}
        impl OutPtr {
            fn get(&self) -> *mut f32 {
                self.0
            }
        }
        let mut out = vec![0f32; out_dim];
        let ptr = OutPtr(out.as_mut_ptr());
        let item = |i: usize| {
            let c0 = i * TEAM_ITEM;
            for (c, bc) in b.iter().enumerate().take(((i + 1) * TEAM_ITEM).min(out_dim)).skip(c0) {
                // SAFETY: `c < out_dim`, and each `c` belongs to one item.
                unsafe { *ptr.get().add(c) = w.dot_row(c, x) + bc };
            }
        };
        if crate::cpu_team::team_for(out_dim.div_ceil(TEAM_ITEM), &item) {
            return Some(out);
        }
    }

    let n_threads = rayon::current_num_threads();
    let total = rows * out_dim;
    if n_threads <= 1 || total < n_threads * MIN_WORK_PER_THREAD {
        return None;
    }

    let mut out = vec![0f32; total];
    out.par_iter_mut().enumerate().for_each(|(idx, o)| {
        let r = idx / out_dim;
        let c = idx % out_dim;
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        *o = w.dot_row(c, xr) + b[c];
    });
    Some(out)
}

fn rmsnorm(x: &[f32], scale: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..dim {
            out[r * dim + j] = row[j] * inv * scale[j];
        }
    }
    out
}

fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// Split-half RoPE (`rotate_half`, matches `shaders/rope_neox.wgsl`), in
/// place on `buf` (`[rows, heads, head_dim]` row-major). `positions[row]` is
/// each row's absolute position: `pos_base + row` for a causal
/// continuation (`model.rs::rope`), caller-chosen for a `ForwardSpec` chunk
/// (`model.rs::rope_positions`).
fn rope_inplace(buf: &mut [f32], rows: usize, heads: usize, head_dim: usize, positions: &[u32], theta: f32) {
    let half = head_dim / 2;
    for (row, &pos) in positions.iter().enumerate().take(rows) {
        let pos = pos as f32;
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for j in 0..half {
                let inv_freq = theta.powf(-2.0 * j as f32 / head_dim as f32);
                let angle = pos * inv_freq;
                let (s, c) = angle.sin_cos();
                let x0 = buf[base + j];
                let x1 = buf[base + half + j];
                buf[base + j] = x0 * c - x1 * s;
                buf[base + half + j] = x1 * c + x0 * s;
            }
        }
    }
}

struct CpuLayer {
    attn_norm: Vec<f32>,
    q_w: CpuWeight,
    q_b: Vec<f32>,
    k_w: CpuWeight,
    k_b: Vec<f32>,
    v_w: CpuWeight,
    v_b: Vec<f32>,
    o_w: CpuWeight,
    o_b: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_w: CpuWeight,
    gate_b: Vec<f32>,
    up_w: CpuWeight,
    up_b: Vec<f32>,
    down_w: CpuWeight,
    down_b: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

#[allow(non_camel_case_types)]
enum CpuEmbed {
    Q4_0 { bytes: Vec<u8>, blocks_per_row: usize, hidden: usize },
    Q8_0 { bytes: Vec<u8>, blocks_per_row: usize, hidden: usize },
    Q6_K { bytes: Vec<u8>, blocks_per_row: usize, hidden: usize },
}

impl CpuEmbed {
    /// Dequantizes one vocab row (`token_id`) to a dense `[hidden]` f32
    /// vector - the CPU op boundary matching `embed_gather_q4.wgsl`/
    /// `embed_gather_q8.wgsl`/`embed_gather_q6k.wgsl`.
    fn row(&self, token_id: u32) -> Vec<f32> {
        match self {
            CpuEmbed::Q4_0 { bytes, blocks_per_row, hidden } => {
                let bytes_per_row = blocks_per_row * 18;
                let start = token_id as usize * bytes_per_row;
                crate::gguf::dequantize_q4_0(&bytes[start..start + bytes_per_row], *hidden)
            }
            CpuEmbed::Q8_0 { bytes, blocks_per_row, hidden } => {
                let bytes_per_row = blocks_per_row * 34;
                let start = token_id as usize * bytes_per_row;
                crate::gguf::dequantize_q8_0(&bytes[start..start + bytes_per_row], *hidden)
            }
            CpuEmbed::Q6_K { bytes, blocks_per_row, hidden } => {
                let bytes_per_row = blocks_per_row * 210;
                let start = token_id as usize * bytes_per_row;
                crate::gguf::dequantize_q6_k(&bytes[start..start + bytes_per_row], *hidden)
            }
        }
    }
}

/// One projection's runtime LoRA delta on the CPU backend: `a` is
/// `[rank, in]`, `b` is `[out, rank]` with `alpha / rank` folded in, both
/// plain F32 weights run through `linear()` - the same two-matmul-plus-add
/// the GPU backend does (`model.rs::apply_lora_proj`), from the same
/// host-side factors (`lora::HostLoraProj`).
struct CpuLoraProj {
    a: CpuWeight,
    b: CpuWeight,
    rank: usize,
    out_features: usize,
}

/// `[q, k, v, o]` per layer.
struct CpuLora {
    layers: Vec<[CpuLoraProj; 4]>,
}

pub struct CpuModel {
    pub config: Qwen2Config,
    embed: CpuEmbed,
    layers: Vec<CpuLayer>,
    out_norm: Vec<f32>,
    lm_head: CpuWeight,
    /// Runtime LoRA adapter (q/k/v/o), added to each projection's output on
    /// every forward; the base weights are never touched. See `lora.rs`.
    lora: Option<CpuLora>,
}

fn gguf_f32_vec<R: Read + Seek>(reader: &mut GgufReader<R>, name: &str) -> Result<Vec<f32>> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let n: usize = info.shape().iter().product();
    let bytes = reader.tensor_data(name)?;
    Ok(dequantize_for(info.dtype(), &bytes, n))
}

fn weight_from_bytes(bytes: Vec<u8>, dtype: GgmlDtype, out_dim: usize, in_dim: usize) -> CpuWeight {
    match dtype {
        GgmlDtype::Q4_0 => CpuWeight::Q4_0 { bytes, out_dim, in_dim, blocks_per_row: in_dim / 32 },
        GgmlDtype::Q8_0 => CpuWeight::Q8_0 { bytes, out_dim, in_dim, blocks_per_row: in_dim / 32 },
        GgmlDtype::Q6_K => CpuWeight::Q6_K { bytes, out_dim, in_dim, blocks_per_row: in_dim / 256 },
        other => CpuWeight::F32 { data: dequantize_for(other, &bytes, out_dim * in_dim), out_dim, in_dim },
    }
}

fn gguf_weight<R: Read + Seek>(reader: &mut GgufReader<R>, name: &str) -> Result<CpuWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let (out_dim, in_dim) = (shape[0], shape[1]);
    let bytes = reader.tensor_data(name)?;
    Ok(weight_from_bytes(bytes, info.dtype(), out_dim, in_dim))
}

/// Like [`gguf_weight`], but un-permutes RoPE row order first when
/// `config.architecture == Architecture::Llama` - the CPU mirror of
/// `model.rs::gguf_matmul_qk`. Without this, a Llama-family GGUF's
/// `attn_q.weight`/`attn_k.weight` rows are in llama.cpp's permuted order
/// (interleaved-pairs-derived), which this crate's split-half `rope_inplace`
/// does not expect - the GPU path already un-permutes these two tensors
/// (`model.rs::gguf_matmul_qk`); the CPU rung must apply the exact same
/// reordering or the two rungs diverge on any Llama-architecture model
/// (observed on SmolLM2-360M-Instruct, not on Qwen2/Qwen3, which never take
/// this branch). `n_heads` is the tensor's own head count: `config.num_heads`
/// for `attn_q.weight`, `config.num_kv_heads` for `attn_k.weight`.
fn gguf_weight_qk<R: Read + Seek>(reader: &mut GgufReader<R>, name: &str, config: &Qwen2Config, n_heads: usize) -> Result<CpuWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let (out_dim, in_dim) = (shape[0], shape[1]);
    let bytes = reader.tensor_data(name)?;
    let bytes = if config.architecture == Architecture::Llama { unpermute_rope_rows(&bytes, n_heads, config.head_dim, out_dim) } else { bytes };
    Ok(weight_from_bytes(bytes, info.dtype(), out_dim, in_dim))
}

impl CpuModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(gguf_path: &str) -> Result<Self> {
        let file = std::fs::File::open(gguf_path).with_context(|| format!("opening {gguf_path}"))?;
        Self::load_from_reader(std::io::BufReader::new(file))
    }

    /// Same loading logic as [`CpuModel::load`], generic over any `Read +
    /// Seek` - the wasm surface hands this a `Cursor` over the whole GGUF's
    /// bytes, exactly like `GpuModel::load_from_reader`. Two-phase loading
    /// applies here too even though there's no GPU upload step: `reader`
    /// (and its underlying byte source) drops at the end of this function,
    /// so the raw GGUF bytes it isn't reusing (already-copied-out tensors)
    /// don't linger.
    pub fn load_from_reader<R: Read + Seek>(reader: R) -> Result<Self> {
        let mut reader = GgufReader::open(reader)?;
        let config = config_from_gguf(&reader)?;

        let embed_info = reader.tensor_info("token_embd.weight").context("missing token_embd.weight")?.clone();
        let embed_shape = embed_info.shape();
        let embed_dtype = embed_info.dtype();
        anyhow::ensure!(
            matches!(embed_dtype, GgmlDtype::Q4_0 | GgmlDtype::Q8_0 | GgmlDtype::Q6_K),
            "expected token_embd.weight to be Q4_0, Q8_0 or Q6_K, got {embed_dtype:?}"
        );
        let hidden = embed_shape[1];
        let embed_bytes = reader.tensor_data("token_embd.weight")?;
        let tied_lm_head = config.tied_embeddings.then(|| match embed_dtype {
            GgmlDtype::Q4_0 => CpuWeight::Q4_0 { bytes: embed_bytes.clone(), out_dim: embed_shape[0], in_dim: hidden, blocks_per_row: hidden / 32 },
            GgmlDtype::Q8_0 => CpuWeight::Q8_0 { bytes: embed_bytes.clone(), out_dim: embed_shape[0], in_dim: hidden, blocks_per_row: hidden / 32 },
            GgmlDtype::Q6_K => CpuWeight::Q6_K { bytes: embed_bytes.clone(), out_dim: embed_shape[0], in_dim: hidden, blocks_per_row: hidden / 256 },
            _ => unreachable!(),
        });
        let embed = match embed_dtype {
            GgmlDtype::Q4_0 => CpuEmbed::Q4_0 { bytes: embed_bytes, blocks_per_row: hidden / 32, hidden },
            GgmlDtype::Q8_0 => CpuEmbed::Q8_0 { bytes: embed_bytes, blocks_per_row: hidden / 32, hidden },
            GgmlDtype::Q6_K => CpuEmbed::Q6_K { bytes: embed_bytes, blocks_per_row: hidden / 256, hidden },
            _ => unreachable!(),
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{i}");
            let (q_b, k_b, v_b) = if config.has_qkv_bias {
                (gguf_f32_vec(&mut reader, &format!("{p}.attn_q.bias"))?, gguf_f32_vec(&mut reader, &format!("{p}.attn_k.bias"))?, gguf_f32_vec(&mut reader, &format!("{p}.attn_v.bias"))?)
            } else {
                let q_dim = config.num_heads * config.head_dim;
                let kv_dim = config.num_kv_heads * config.head_dim;
                (vec![0f32; q_dim], vec![0f32; kv_dim], vec![0f32; kv_dim])
            };
            let (q_norm, k_norm) = if config.qk_norm {
                (Some(gguf_f32_vec(&mut reader, &format!("{p}.attn_q_norm.weight"))?), Some(gguf_f32_vec(&mut reader, &format!("{p}.attn_k_norm.weight"))?))
            } else {
                (None, None)
            };
            layers.push(CpuLayer {
                attn_norm: gguf_f32_vec(&mut reader, &format!("{p}.attn_norm.weight"))?,
                q_w: gguf_weight_qk(&mut reader, &format!("{p}.attn_q.weight"), &config, config.num_heads)?,
                q_b,
                k_w: gguf_weight_qk(&mut reader, &format!("{p}.attn_k.weight"), &config, config.num_kv_heads)?,
                k_b,
                v_w: gguf_weight(&mut reader, &format!("{p}.attn_v.weight"))?,
                v_b,
                o_w: gguf_weight(&mut reader, &format!("{p}.attn_output.weight"))?,
                o_b: vec![0f32; config.hidden_size],
                ffn_norm: gguf_f32_vec(&mut reader, &format!("{p}.ffn_norm.weight"))?,
                gate_w: gguf_weight(&mut reader, &format!("{p}.ffn_gate.weight"))?,
                gate_b: vec![0f32; config.intermediate_size],
                up_w: gguf_weight(&mut reader, &format!("{p}.ffn_up.weight"))?,
                up_b: vec![0f32; config.intermediate_size],
                down_w: gguf_weight(&mut reader, &format!("{p}.ffn_down.weight"))?,
                down_b: vec![0f32; config.hidden_size],
                q_norm,
                k_norm,
            });
        }

        let out_norm = gguf_f32_vec(&mut reader, "output_norm.weight")?;
        let lm_head = match tied_lm_head {
            Some(w) => w,
            None => gguf_weight(&mut reader, "output.weight")?,
        };

        drop(reader);

        Ok(CpuModel { config, embed, layers, out_norm, lm_head, lora: None })
    }

    /// Parse an LLMLIFE2 adapter (q/k/v/o, see `lora.rs`) and apply it,
    /// replacing any adapter applied earlier: CPU mirror of
    /// `GpuModel::apply_lora`.
    pub fn apply_lora(&mut self, bytes: &[u8]) -> Result<()> {
        let raw = crate::lora::RawLoraAdapter::parse(bytes, self.config.num_layers)?;
        let proj = |h: crate::lora::HostLoraProj| CpuLoraProj {
            a: CpuWeight::F32 { data: h.a_t, out_dim: h.rank, in_dim: h.in_features },
            b: CpuWeight::F32 { data: h.b_t, out_dim: h.out_features, in_dim: h.rank },
            rank: h.rank,
            out_features: h.out_features,
        };
        let layers = raw.host_layers().into_iter().map(|l| l.map(proj)).collect();
        self.lora = Some(CpuLora { layers });
        Ok(())
    }

    /// Back to the base model: CPU mirror of `GpuModel::clear_lora`.
    pub fn clear_lora(&mut self) {
        self.lora = None;
    }

    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    /// Logits for a few vocab ids at every row of `hidden` (`[rows,
    /// hidden]`, as `forward_chunk_spec` returns it), read through the token
    /// embedding rows: CPU mirror of `GpuModel::embed_head_sliced` (see its
    /// doc comment on why the embedding rows and not `output.weight`).
    /// Returns `[rows, token_ids.len()]`, row-major.
    pub fn embed_head_sliced(&self, hidden: &[f32], rows: usize, token_ids: &[u32]) -> Vec<f32> {
        let h = self.config.hidden_size;
        let head: Vec<Vec<f32>> = token_ids.iter().map(|&id| self.embed.row(id)).collect();
        let mut out = vec![0f32; rows * token_ids.len()];
        for r in 0..rows {
            let x = &hidden[r * h..(r + 1) * h];
            for (k, w) in head.iter().enumerate() {
                out[r * token_ids.len() + k] = x.iter().zip(w).map(|(a, b)| a * b).sum();
            }
        }
        out
    }
}

/// `linear()` plus, when `lora` is `Some`, that projection's LoRA delta
/// `(x A^T) B^T` added to the output - CPU mirror of `model.rs::linear_lora`.
fn linear_lora(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize, lora: Option<&CpuLoraProj>) -> Vec<f32> {
    let mut out = linear(x, rows, in_dim, w, b, out_dim);
    if let Some(p) = lora {
        let mid = linear(x, rows, in_dim, &p.a, &vec![0f32; p.rank], p.rank);
        let delta = linear(&mid, rows, p.rank, &p.b, &vec![0f32; p.out_features], p.out_features);
        add_inplace(&mut out, &delta);
    }
    out
}

/// Same layout convention as `model.rs::KvCache` (see this file's top doc
/// comment): per layer, `[kv_heads, max_ctx, head_dim]`, head-major and
/// contiguous per head. Plain `Vec<f32>`, grown up front to `max_ctx` (not a
/// ring buffer) - `kv_len` tracks how much of it is valid.
pub struct CpuKvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    max_ctx: usize,
    pub kv_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl CpuKvCache {
    pub fn new(config: &Qwen2Config, max_ctx: usize) -> Self {
        let per_layer = config.num_kv_heads * max_ctx * config.head_dim;
        CpuKvCache {
            k: (0..config.num_layers).map(|_| vec![0f32; per_layer]).collect(),
            v: (0..config.num_layers).map(|_| vec![0f32; per_layer]).collect(),
            max_ctx,
            kv_len: 0,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn remaining_capacity(&self) -> usize {
        self.max_ctx.saturating_sub(self.kv_len)
    }

    fn assert_matches(&self, cfg: &Qwen2Config) {
        assert_eq!(self.num_kv_heads, cfg.num_kv_heads, "CpuKvCache built for a different num_kv_heads than this model");
        assert_eq!(self.head_dim, cfg.head_dim, "CpuKvCache built for a different head_dim than this model");
    }

    /// Scatters `rows` freshly-computed K or V rows (`[rows, kv_heads,
    /// head_dim]` row-major, `cache_buf`'s own layer) into
    /// `[kv_head, kv_base+row, head_dim]` - the CPU mirror of
    /// `model.rs::scatter_kv_gpu`.
    fn scatter(cache_buf: &mut [f32], src: &[f32], rows: usize, kv_heads: usize, head_dim: usize, kv_base: usize, max_ctx: usize) {
        for row in 0..rows {
            for h in 0..kv_heads {
                let src_base = (row * kv_heads + h) * head_dim;
                let dst_pos = kv_base + row;
                let dst_base = (h * max_ctx + dst_pos) * head_dim;
                cache_buf[dst_base..dst_base + head_dim].copy_from_slice(&src[src_base..src_base + head_dim]);
            }
        }
    }
}

/// GQA causal attention, `t` new query rows against a resident KV cache
/// already holding `kv_len` positions (the new rows' own K/V must already be
/// scattered into the cache before calling this - see the call sites below,
/// matching `attn_prefill.wgsl`'s/`attn_decode.wgsl`'s online-softmax
/// convention and `model.rs`'s doc comment: "kv_len passed to attn_decode
/// already includes the current position"). `q`: `[t, n_heads, head_dim]`.
/// Returns `[t, n_heads, head_dim]`.
///
/// `allowed`: `None` is the causal rule above; `Some(bits)` is a
/// `ForwardSpec` bitset (`[t, kv_len]`, bit `i * kv_len + j` set when row
/// `i` may attend key `j`) - the CPU mirror of `attn_chunk_masked.wgsl`.
#[allow(clippy::too_many_arguments)]
fn attention(q: &[f32], k_cache: &[f32], v_cache: &[f32], t: usize, kv_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_ctx: usize, query_pos_base: usize, allowed: Option<&[u32]>) -> Vec<f32> {
    let n_rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; t * n_heads * head_dim];
    let mut keys: Vec<usize> = Vec::with_capacity(kv_len);
    for i in 0..t {
        keys.clear();
        match allowed {
            None => {
                let causal_len = query_pos_base + i + 1; // this row may attend keys [0, causal_len)
                keys.extend(0..causal_len.min(kv_len));
            }
            Some(bits) => keys.extend((0..kv_len).filter(|j| {
                let idx = i * kv_len + j;
                (bits[idx / 32] >> (idx % 32)) & 1 != 0
            })),
        }
        for head in 0..n_heads {
            let kv_head = head / n_rep;
            let q_base = (i * n_heads + head) * head_dim;
            let qr = &q[q_base..q_base + head_dim];
            let mut scores = vec![0f32; keys.len()];
            let mut max_score = f32::NEG_INFINITY;
            for (&j, score) in keys.iter().zip(scores.iter_mut()) {
                let k_base = (kv_head * max_ctx + j) * head_dim;
                let kr = &k_cache[k_base..k_base + head_dim];
                let dot: f32 = qr.iter().zip(kr).map(|(a, b)| a * b).sum();
                *score = dot * scale;
                max_score = max_score.max(*score);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let out_base = (i * n_heads + head) * head_dim;
            for (&j, &score) in keys.iter().zip(scores.iter()) {
                let w = score / sum;
                let v_base = (kv_head * max_ctx + j) * head_dim;
                let vr = &v_cache[v_base..v_base + head_dim];
                for d in 0..head_dim {
                    out[out_base + d] += w * vr[d];
                }
            }
        }
    }
    out
}

/// One layer's forward, in place on `x` (`[rows, hidden]`, returned as a new
/// `Vec<f32>` - residual add is internal). `kv_base` is the cache slot of
/// `x`'s first row (0 for a from-scratch prefill, `cache.kv_len` for a
/// decode/suffix/chunk step); `positions` holds each row's RoPE position
/// (`kv_base + row` for a causal continuation, caller-chosen for a
/// `ForwardSpec` chunk); `allowed` is the chunk's attention bitset (`None`:
/// causal) - see `attention`. `lora` is this layer's `[q, k, v, o]` adapter.
#[allow(clippy::too_many_arguments)]
fn layer_forward(
    layer: &CpuLayer,
    x: &[f32],
    rows: usize,
    cfg: &Qwen2Config,
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    kv_base: usize,
    positions: &[u32],
    allowed: Option<&[u32]>,
    lora: Option<&[CpuLoraProj; 4]>,
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let lora_proj = |i: usize| lora.map(|l| &l[i]);

    let normed = rmsnorm(x, &layer.attn_norm, rows, hidden, cfg.rms_norm_eps);
    let mut q = linear_lora(&normed, rows, hidden, &layer.q_w, &layer.q_b, q_dim, lora_proj(0));
    let mut k = linear_lora(&normed, rows, hidden, &layer.k_w, &layer.k_b, kv_dim, lora_proj(1));
    let v = linear_lora(&normed, rows, hidden, &layer.v_w, &layer.v_b, kv_dim, lora_proj(2));

    if let Some(scale) = &layer.q_norm {
        q = rmsnorm(&q, scale, rows * cfg.num_heads, cfg.head_dim, cfg.rms_norm_eps);
    }
    if let Some(scale) = &layer.k_norm {
        k = rmsnorm(&k, scale, rows * cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps);
    }

    rope_inplace(&mut q, rows, cfg.num_heads, cfg.head_dim, positions, cfg.rope_theta);
    rope_inplace(&mut k, rows, cfg.num_kv_heads, cfg.head_dim, positions, cfg.rope_theta);

    let max_ctx = k_cache.len() / (cfg.num_kv_heads * cfg.head_dim);
    CpuKvCache::scatter(k_cache, &k, rows, cfg.num_kv_heads, cfg.head_dim, kv_base, max_ctx);
    CpuKvCache::scatter(v_cache, &v, rows, cfg.num_kv_heads, cfg.head_dim, kv_base, max_ctx);

    let kv_len = kv_base + rows;
    let attn_out = attention(&q, k_cache, v_cache, rows, kv_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, max_ctx, kv_base, allowed);

    let o = linear_lora(&attn_out, rows, q_dim, &layer.o_w, &layer.o_b, hidden, lora_proj(3));
    let mut x = x.to_vec();
    add_inplace(&mut x, &o);

    let ffn_normed = rmsnorm(&x, &layer.ffn_norm, rows, hidden, cfg.rms_norm_eps);
    let gate = linear(&ffn_normed, rows, hidden, &layer.gate_w, &layer.gate_b, cfg.intermediate_size);
    let up = linear(&ffn_normed, rows, hidden, &layer.up_w, &layer.up_b, cfg.intermediate_size);
    let gated = silu_mul(&gate, &up);
    let down = linear(&gated, rows, cfg.intermediate_size, &layer.down_w, &layer.down_b, hidden);
    add_inplace(&mut x, &down);
    x
}

fn embed_gather(model: &CpuModel, token_ids: &[u32]) -> Vec<f32> {
    let hidden = model.config.hidden_size;
    let mut out = vec![0f32; token_ids.len() * hidden];
    for (i, &id) in token_ids.iter().enumerate() {
        out[i * hidden..(i + 1) * hidden].copy_from_slice(&model.embed.row(id));
    }
    out
}

/// Every layer plus `output_norm` over `token_ids` written at cache slots
/// `[kv_len, kv_len + t)`; returns the normed hidden states `[t, hidden]`
/// and leaves `cache.kv_len` at `kv_len + t`.
fn forward_hidden(model: &CpuModel, cache: &mut CpuKvCache, token_ids: &[u32], positions: &[u32], allowed: Option<&[u32]>) -> Vec<f32> {
    let cfg = &model.config;
    cache.assert_matches(cfg);
    let rows = token_ids.len();
    let kv_base = cache.kv_len;
    assert!(kv_base + rows <= cache.max_ctx, "CPU forward: kv_len {kv_base} + {rows} rows exceeds max_ctx {}", cache.max_ctx);
    let mut x = embed_gather(model, token_ids);
    for (i, layer) in model.layers.iter().enumerate() {
        let lora = model.lora.as_ref().map(|l| &l.layers[i]);
        x = layer_forward(layer, &x, rows, cfg, &mut cache.k[i], &mut cache.v[i], kv_base, positions, allowed, lora);
    }
    cache.kv_len = kv_base + rows;
    rmsnorm(&x, &model.out_norm, rows, cfg.hidden_size, cfg.rms_norm_eps)
}

fn forward_layers(model: &CpuModel, cache: &mut CpuKvCache, token_ids: &[u32]) -> Vec<f32> {
    let cfg = &model.config;
    let rows = token_ids.len();
    let positions: Vec<u32> = (0..rows).map(|r| (cache.kv_len + r) as u32).collect();
    let normed = forward_hidden(model, cache, token_ids, &positions, None);
    linear(&normed, rows, cfg.hidden_size, &model.lm_head, &vec![0f32; cfg.vocab_size], cfg.vocab_size)
}

/// CPU mirror of `model.rs::forward_chunk_spec`: one chunk of `token_ids`
/// against the cache's resident prefix (`cache.kv_len` positions), with
/// the caller's positions and attention bitset (`ForwardSpec`; defaults:
/// continue counting from `kv_len`, causal over prefix plus chunk).
/// Returns the hidden states `[t, hidden]` (post `output_norm`, pre head:
/// slice with `CpuModel::embed_head_sliced`) and leaves `cache.kv_len` at
/// `prefix_len + t`; rewind by setting `kv_len` back to the prefix length.
pub fn forward_chunk_spec(model: &CpuModel, cache: &mut CpuKvCache, token_ids: &[u32], spec: &crate::model::ForwardSpec) -> Vec<f32> {
    let t = token_ids.len();
    let prefix_len = cache.kv_len;
    let positions: Vec<u32> = match &spec.positions {
        Some(p) => {
            assert_eq!(p.len(), t, "ForwardSpec positions must be one per token");
            p.clone()
        }
        None => (0..t).map(|r| (prefix_len + r) as u32).collect(),
    };
    if let Some(bits) = &spec.allowed_bits {
        assert!(bits.len() * 32 >= t * (prefix_len + t), "ForwardSpec mask must cover [t, prefix_len + t]");
    }
    forward_hidden(model, cache, token_ids, &positions, spec.allowed_bits.as_deref())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Prefill `token_ids` against a fresh (or already-resident-prefix) cache;
/// returns the last position's logits (`[vocab]`) - same contract as
/// `model.rs::forward_prefill`, minus the `engine`/`cos`/`sin` parameters
/// the GPU path needs and this one doesn't (RoPE angles are computed
/// on the fly per call, no precomputed table - see `rope_inplace`).
pub fn forward_prefill(model: &CpuModel, cache: &mut CpuKvCache, token_ids: &[u32]) -> Vec<f32> {
    let vocab = model.config.vocab_size;
    let rows = token_ids.len();
    #[cfg(feature = "threads")]
    let all_logits = crate::cpu_team::with_team(|| forward_layers(model, cache, token_ids));
    #[cfg(not(feature = "threads"))]
    let all_logits = forward_layers(model, cache, token_ids);
    all_logits[(rows - 1) * vocab..].to_vec()
}

/// Decode one token against the cache; returns full-vocab logits - CPU
/// mirror of `model.rs::forward_decode_step`.
pub fn forward_decode_step(model: &CpuModel, cache: &mut CpuKvCache, token_id: u32) -> Vec<f32> {
    #[cfg(feature = "threads")]
    return crate::cpu_team::with_team(|| forward_layers(model, cache, &[token_id]));
    #[cfg(not(feature = "threads"))]
    forward_layers(model, cache, &[token_id])
}

/// Decode one token, returning only the argmax id - CPU mirror of
/// `model.rs::forward_decode_step_argmax` (no separate GPU-vs-CPU readback
/// cost to avoid here, but kept as its own function for API-shape parity: a
/// caller picking the rung at runtime calls the same-named function either
/// way).
pub fn forward_decode_step_argmax(model: &CpuModel, cache: &mut CpuKvCache, token_id: u32) -> u32 {
    argmax(&forward_decode_step(model, cache, token_id))
}

/// In-process parity gate for the threads rung, no GGUF/model needed - the
/// fixture-level gate (`lean-cli --engine cpu` under `RAYON_NUM_THREADS=1`
/// vs the default pool size, see docs/runs/2026-09-29-lean-threads.md) is
/// the end-to-end check; this covers `linear()` itself, at both a
/// decode-shaped (rows=1) and prefill-shaped (rows>1) size, against every
/// weight dtype `linear_threads` dispatches over.
#[cfg(all(test, feature = "threads"))]
mod threads_tests {
    use super::*;

    fn synth_x(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| ((i as u32 * 7 + seed * 13) % 251) as f32 * 0.01 - 1.0).collect()
    }

    fn synth_f32_weight(out_dim: usize, in_dim: usize) -> CpuWeight {
        let data = (0..out_dim * in_dim).map(|i| ((i as u32 * 3) % 251) as f32 * 0.005 - 0.6).collect();
        CpuWeight::F32 { data, out_dim, in_dim }
    }

    fn synth_q4_weight(out_dim: usize, in_dim: usize) -> CpuWeight {
        let blocks_per_row = in_dim / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut bytes = vec![0u8; out_dim * bytes_per_row];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i as u32 * 11 + 5) % 256) as u8;
        }
        // give every block a non-degenerate f16 scale (bytes[0..2] of each 18-byte block)
        for row in 0..out_dim {
            for block in 0..blocks_per_row {
                let base = row * bytes_per_row + block * 18;
                let scale = half::f16::from_f32(0.01 + (block % 5) as f32 * 0.004);
                bytes[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            }
        }
        CpuWeight::Q4_0 { bytes, out_dim, in_dim, blocks_per_row }
    }

    /// Runs `linear` inside a fresh, scoped rayon pool of `n_threads` (never
    /// touching the process-global pool), so this test controls thread count
    /// deterministically instead of depending on `available_parallelism()`.
    fn linear_with_pool(n_threads: usize, x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Vec<f32> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
        pool.install(|| linear(x, rows, in_dim, w, b, out_dim))
    }

    #[test]
    fn decode_shaped_threads_match_single_thread() {
        let (out_dim, in_dim) = (896, 896);
        let w = synth_f32_weight(out_dim, in_dim);
        let b = vec![0f32; out_dim];
        let x = synth_x(in_dim, 1);
        let single = linear_with_pool(1, &x, 1, in_dim, &w, &b, out_dim);
        for n in [2usize, 4, 8] {
            let threaded = linear_with_pool(n, &x, 1, in_dim, &w, &b, out_dim);
            assert_eq!(single, threaded, "n_threads={n} diverged from single-thread (decode-shaped, F32)");
        }
    }

    #[test]
    fn prefill_shaped_threads_match_single_thread() {
        let (out_dim, in_dim) = (896, 896);
        let rows = 37;
        let w = synth_f32_weight(out_dim, in_dim);
        let b = vec![0f32; out_dim];
        let x = synth_x(rows * in_dim, 2);
        let single = linear_with_pool(1, &x, rows, in_dim, &w, &b, out_dim);
        for n in [2usize, 4, 8] {
            let threaded = linear_with_pool(n, &x, rows, in_dim, &w, &b, out_dim);
            assert_eq!(single, threaded, "n_threads={n} diverged from single-thread (prefill-shaped, F32)");
        }
    }

    #[test]
    fn q4_0_weight_threads_match_single_thread() {
        let (out_dim, in_dim) = (256, 256);
        let w = synth_q4_weight(out_dim, in_dim);
        let b = vec![0f32; out_dim];
        let x = synth_x(in_dim, 3);
        let single = linear_with_pool(1, &x, 1, in_dim, &w, &b, out_dim);
        let threaded = linear_with_pool(8, &x, 1, in_dim, &w, &b, out_dim);
        assert_eq!(single, threaded, "Q4_0 decode-shaped linear diverged under threads");
    }

    #[test]
    fn below_threshold_falls_back_to_serial() {
        // A tiny matrix (well under MIN_WORK_PER_THREAD * n_threads) must
        // still produce the exact serial result even when a large pool is
        // available - `linear_threads` should decline it, not divide it
        // pointlessly across threads.
        let (out_dim, in_dim) = (4, 8);
        let w = synth_f32_weight(out_dim, in_dim);
        let b = vec![0.1f32; out_dim];
        let x = synth_x(in_dim, 4);
        let single = linear_with_pool(1, &x, 1, in_dim, &w, &b, out_dim);
        let threaded = linear_with_pool(8, &x, 1, in_dim, &w, &b, out_dim);
        assert_eq!(single, threaded);
    }
}

/// The multi-row path (`linear_multi_row`): every element equals
/// `dot_f32(weight row, activation row) + bias` bit for bit, in a full 4x4
/// tile or an edge, and agrees with the per-row decode kernels up to float
/// summation order.
#[cfg(test)]
mod multi_row_tests {
    use super::*;

    #[test]
    fn multi_row_matches_per_element_dot_and_per_row_kernel() {
        let (out_dim, in_dim, rows) = (37usize, 64usize, 11usize); // edges in both rows and columns
        let blocks_per_row = in_dim / 32;
        let mut bytes = vec![0u8; out_dim * blocks_per_row * 18];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 11 + 5) % 256) as u8;
        }
        for blk in 0..out_dim * blocks_per_row {
            bytes[blk * 18..blk * 18 + 2].copy_from_slice(&half::f16::from_f32(0.01 + (blk % 5) as f32 * 0.004).to_le_bytes());
        }
        let w = CpuWeight::Q4_0 { bytes, out_dim, in_dim, blocks_per_row };
        let b: Vec<f32> = (0..out_dim).map(|c| c as f32 * 0.01).collect();
        let x: Vec<f32> = (0..rows * in_dim).map(|i| ((i * 7) % 251) as f32 * 0.01 - 1.0).collect();
        let got = linear_multi_row(&x, rows, in_dim, &w, &b, out_dim);
        let reference = linear_serial(&x, rows, in_dim, &w, &b, out_dim);
        let mut wrow = vec![0f32; in_dim];
        for c in 0..out_dim {
            w.dequant_row(c, &mut wrow);
            for r in 0..rows {
                let want = dot_f32(&wrow, &x[r * in_dim..(r + 1) * in_dim]) + b[c];
                assert_eq!(got[r * out_dim + c].to_bits(), want.to_bits(), "row {r} col {c}");
                assert!((got[r * out_dim + c] - reference[r * out_dim + c]).abs() < 1e-4, "row {r} col {c} vs per-row kernel");
            }
        }
    }
}
