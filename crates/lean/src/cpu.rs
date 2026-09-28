//! Single-threaded CPU forward pass: the CPU fallback rung below the WebGPU
//! rung in `model.rs`. Same op list, same config (`Qwen2Config`), same GGUF parsing
//! (`GgufReader`), same tokenizer/chat-template code as the GPU path - only
//! the destination of each weight tensor differs: a plain `Vec<u8>` holding
//! the GGUF's on-disk Q4_0/Q8_0 block bytes unchanged (no repack, no
//! dequantize-to-f32-then-requantize), dequantized in-kernel per dot
//! product by `cpu_kernels::dot_q4_0`/`dot_q8_0`. This is a second rung, not
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

use crate::config::{config_from_gguf, Qwen2Config};
use crate::cpu_kernels::{dot_q4_0, dot_q8_0};
use crate::gguf::{dequantize_for, GgmlDtype, GgufReader};

/// A CPU-resident matmul weight (`shape = [out_dim, in_dim]`), holding the
/// GGUF's on-disk bytes for Q4_0/Q8_0 unchanged (row-major, block-contiguous,
/// the same bytes `quant.rs::load_matmul_weight_gguf` repacks for the GPU
/// path, here left exactly as read). F32/F16 tensors are dequantized once at
/// load time (same as the GPU path's `gguf_f32` helper) since there is no
/// per-dot-product win to chasing their (already dense) bytes further.
pub enum CpuWeight {
    F32 { data: Vec<f32>, out_dim: usize, in_dim: usize },
    Q4_0 { bytes: Vec<u8>, out_dim: usize, in_dim: usize, blocks_per_row: usize },
    Q8_0 { bytes: Vec<u8>, out_dim: usize, in_dim: usize, blocks_per_row: usize },
}

impl CpuWeight {
    fn dims(&self) -> (usize, usize) {
        match self {
            CpuWeight::F32 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
            CpuWeight::Q4_0 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
            CpuWeight::Q8_0 { out_dim, in_dim, .. } => (*out_dim, *in_dim),
        }
    }

    /// Row `row`'s output value: dot product of that output row's weights
    /// against `x` (`x.len() == in_dim`). One row = one output scalar - the
    /// same op boundary `linear_q4.wgsl`/`linear_q8.wgsl`'s per-thread work
    /// unit uses on the GPU path.
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
        }
    }
}

/// `x`: `[rows, in_dim]` row-major. Returns `[rows, out_dim]` row-major, `+
/// bias` per output column - same shape/semantics as `model.rs::linear`'s
/// GPU kernel, computed as `rows * out_dim` independent dot products (no
/// blocking/tiling: this is the reference-correctness rung, see this
/// crate's CPU-fallback plan on why a fast CPU kernel is out of scope for
/// this pass).
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &CpuWeight, b: &[f32], out_dim: usize) -> Vec<f32> {
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
/// place on `buf` (`[rows, heads, head_dim]` row-major). `pos_base + row` is
/// each row's absolute position - same convention as `model.rs::rope`.
fn rope_inplace(buf: &mut [f32], rows: usize, heads: usize, head_dim: usize, pos_base: usize, theta: f32) {
    let half = head_dim / 2;
    for row in 0..rows {
        let pos = (pos_base + row) as f32;
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

enum CpuEmbed {
    Q4_0 { bytes: Vec<u8>, blocks_per_row: usize, hidden: usize },
    Q8_0 { bytes: Vec<u8>, blocks_per_row: usize, hidden: usize },
}

impl CpuEmbed {
    /// Dequantizes one vocab row (`token_id`) to a dense `[hidden]` f32
    /// vector - the CPU op boundary matching `embed_gather_q4.wgsl`/
    /// `embed_gather_q8.wgsl`.
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
        }
    }
}

pub struct CpuModel {
    pub config: Qwen2Config,
    embed: CpuEmbed,
    layers: Vec<CpuLayer>,
    out_norm: Vec<f32>,
    lm_head: CpuWeight,
}

fn gguf_f32_vec<R: Read + Seek>(reader: &mut GgufReader<R>, name: &str) -> Result<Vec<f32>> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let n: usize = info.shape().iter().product();
    let bytes = reader.tensor_data(name)?;
    Ok(dequantize_for(info.dtype(), &bytes, n))
}

fn gguf_weight<R: Read + Seek>(reader: &mut GgufReader<R>, name: &str) -> Result<CpuWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let out_dim = shape[0];
    let in_dim = shape[1];
    let bytes = reader.tensor_data(name)?;
    Ok(match info.dtype() {
        GgmlDtype::Q4_0 => CpuWeight::Q4_0 { bytes, out_dim, in_dim, blocks_per_row: in_dim / 32 },
        GgmlDtype::Q8_0 => CpuWeight::Q8_0 { bytes, out_dim, in_dim, blocks_per_row: in_dim / 32 },
        other => CpuWeight::F32 { data: dequantize_for(other, &bytes, out_dim * in_dim), out_dim, in_dim },
    })
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
        anyhow::ensure!(matches!(embed_dtype, GgmlDtype::Q4_0 | GgmlDtype::Q8_0), "expected token_embd.weight to be Q4_0 or Q8_0, got {embed_dtype:?}");
        let hidden = embed_shape[1];
        let embed_bytes = reader.tensor_data("token_embd.weight")?;
        let tied_lm_head = config.tied_embeddings.then(|| match embed_dtype {
            GgmlDtype::Q4_0 => CpuWeight::Q4_0 { bytes: embed_bytes.clone(), out_dim: embed_shape[0], in_dim: hidden, blocks_per_row: hidden / 32 },
            GgmlDtype::Q8_0 => CpuWeight::Q8_0 { bytes: embed_bytes.clone(), out_dim: embed_shape[0], in_dim: hidden, blocks_per_row: hidden / 32 },
            _ => unreachable!(),
        });
        let embed = match embed_dtype {
            GgmlDtype::Q4_0 => CpuEmbed::Q4_0 { bytes: embed_bytes, blocks_per_row: hidden / 32, hidden },
            GgmlDtype::Q8_0 => CpuEmbed::Q8_0 { bytes: embed_bytes, blocks_per_row: hidden / 32, hidden },
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
                q_w: gguf_weight(&mut reader, &format!("{p}.attn_q.weight"))?,
                q_b,
                k_w: gguf_weight(&mut reader, &format!("{p}.attn_k.weight"))?,
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

        Ok(CpuModel { config, embed, layers, out_norm, lm_head })
    }
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
#[allow(clippy::too_many_arguments)]
fn attention(q: &[f32], k_cache: &[f32], v_cache: &[f32], t: usize, kv_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_ctx: usize, query_pos_base: usize) -> Vec<f32> {
    let n_rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; t * n_heads * head_dim];
    for i in 0..t {
        let causal_len = query_pos_base + i + 1; // this row may attend keys [0, causal_len)
        let causal_len = causal_len.min(kv_len);
        for head in 0..n_heads {
            let kv_head = head / n_rep;
            let q_base = (i * n_heads + head) * head_dim;
            let qr = &q[q_base..q_base + head_dim];
            let mut scores = vec![0f32; causal_len];
            let mut max_score = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
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
            for (j, &score) in scores.iter().enumerate() {
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
/// `Vec<f32>` - residual add is internal). `query_pos_base` is `x`'s first
/// row's absolute position (0 for a from-scratch prefill, `cache.kv_len`
/// for a decode/suffix step) - same role as `model.rs::attn_decode`'s
/// `pos`/`rope`'s `pos_base`.
#[allow(clippy::too_many_arguments)]
fn layer_forward(layer: &CpuLayer, x: &[f32], rows: usize, cfg: &Qwen2Config, k_cache: &mut [f32], v_cache: &mut [f32], query_pos_base: usize) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    let normed = rmsnorm(x, &layer.attn_norm, rows, hidden, cfg.rms_norm_eps);
    let mut q = linear(&normed, rows, hidden, &layer.q_w, &layer.q_b, q_dim);
    let mut k = linear(&normed, rows, hidden, &layer.k_w, &layer.k_b, kv_dim);
    let v = linear(&normed, rows, hidden, &layer.v_w, &layer.v_b, kv_dim);

    if let Some(scale) = &layer.q_norm {
        q = rmsnorm(&q, scale, rows * cfg.num_heads, cfg.head_dim, cfg.rms_norm_eps);
    }
    if let Some(scale) = &layer.k_norm {
        k = rmsnorm(&k, scale, rows * cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps);
    }

    rope_inplace(&mut q, rows, cfg.num_heads, cfg.head_dim, query_pos_base, cfg.rope_theta);
    rope_inplace(&mut k, rows, cfg.num_kv_heads, cfg.head_dim, query_pos_base, cfg.rope_theta);

    CpuKvCache::scatter(k_cache, &k, rows, cfg.num_kv_heads, cfg.head_dim, query_pos_base, k_cache.len() / (cfg.num_kv_heads * cfg.head_dim));
    CpuKvCache::scatter(v_cache, &v, rows, cfg.num_kv_heads, cfg.head_dim, query_pos_base, v_cache.len() / (cfg.num_kv_heads * cfg.head_dim));

    let max_ctx = k_cache.len() / (cfg.num_kv_heads * cfg.head_dim);
    let kv_len = query_pos_base + rows;
    let attn_out = attention(&q, k_cache, v_cache, rows, kv_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, max_ctx, query_pos_base);

    let o = linear(&attn_out, rows, q_dim, &layer.o_w, &layer.o_b, hidden);
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

fn forward_layers(model: &CpuModel, cache: &mut CpuKvCache, token_ids: &[u32]) -> Vec<f32> {
    let cfg = &model.config;
    cache.assert_matches(cfg);
    let rows = token_ids.len();
    let query_pos_base = cache.kv_len;
    let mut x = embed_gather(model, token_ids);
    for (i, layer) in model.layers.iter().enumerate() {
        x = layer_forward(layer, &x, rows, cfg, &mut cache.k[i], &mut cache.v[i], query_pos_base);
    }
    cache.kv_len = query_pos_base + rows;
    let normed = rmsnorm(&x, &model.out_norm, rows, cfg.hidden_size, cfg.rms_norm_eps);
    linear(&normed, rows, cfg.hidden_size, &model.lm_head, &vec![0f32; cfg.vocab_size], cfg.vocab_size)
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
    let all_logits = forward_layers(model, cache, token_ids);
    all_logits[(rows - 1) * vocab..].to_vec()
}

/// Decode one token against the cache; returns full-vocab logits - CPU
/// mirror of `model.rs::forward_decode_step`.
pub fn forward_decode_step(model: &CpuModel, cache: &mut CpuKvCache, token_id: u32) -> Vec<f32> {
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
