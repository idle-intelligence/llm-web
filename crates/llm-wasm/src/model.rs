//! Qwen2 decoder-only transformer: RMSNorm, GQA attention with q/k/v bias,
//! RoPE (rotate-half convention), SwiGLU MLP, tied lm_head. Ported from
//! HF's `Qwen2Model`/`Qwen2Attention` semantics (transformers
//! `modeling_qwen2.py`), matched numerically against `fixtures/reference`
//! (see tests/full_forward.rs), not from stt-web's structurally-similar but
//! differently-shaped `SttModel` (docs/ENGINE.md §3-4 catalogued the deltas:
//! separate q/k/v/bias instead of one `in_proj`, 3-matrix SwiGLU instead of
//! 2-matrix gating, rotate-half RoPE instead of interleaved-pair, no sliding
//! window, tied lm_head instead of an independent `text_linear`).

use anyhow::Result;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor};

use crate::gguf::{EmbeddingStore, Q4Linear};
use crate::grammar::Constraint;
use crate::kv::KvCache;
use crate::LlmConfig;

// ---------------------------------------------------------------------------
// RoPE — Rotary Position Embeddings (rotate-half / HF Llama-Qwen2 convention)
// ---------------------------------------------------------------------------

/// Rotary Position Embeddings with precomputed cos/sin tables, HF
/// `rotate_half` convention: `head_dim` splits into two contiguous halves
/// `[x0..x_{d/2})` / `[x_{d/2}..x_d)`, rotated against each other — **not**
/// stt-wasm's interleaved-pair convention (docs/ENGINE.md §3). `cos`/`sin`
/// are `[max_seq_len, head_dim]` (the per-half-dim frequency table
/// concatenated with itself, matching HF's `emb = cat([freqs, freqs],
/// dim=-1)`) so they can be sliced and broadcast-multiplied directly against
/// `[.., head_dim]`-shaped q/k tensors.
pub struct RoPE {
    cos: Tensor<Wgpu, 2>,
    sin: Tensor<Wgpu, 2>,
}

impl RoPE {
    /// Create RoPE with precomputed frequencies.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &WgpuDevice) -> Self {
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();

        // freqs[pos, j] = pos * inv_freq[j], then duplicated across the
        // second half: emb = [freqs, freqs] (HF `Qwen2RotaryEmbedding`).
        let mut emb = vec![0.0f32; max_seq_len * head_dim];
        for pos in 0..max_seq_len {
            for j in 0..half_dim {
                let v = pos as f32 * inv_freq[j];
                emb[pos * head_dim + j] = v;
                emb[pos * head_dim + half_dim + j] = v;
            }
        }

        let emb = Tensor::<Wgpu, 1>::from_floats(emb.as_slice(), device)
            .reshape([max_seq_len, head_dim]);

        let cos = emb.clone().cos();
        let sin = emb.sin();

        RoPE { cos, sin }
    }

    /// cos/sin slices for absolute positions `[offset, offset+len)`, shaped
    /// `[1, 1, len, head_dim]` for broadcast against `[B, H, len, head_dim]`.
    /// F1 (docs/BENCHMARKS.md Session 9): no longer used on the production
    /// forward path (superseded by `gguf::rope_fused`'s single dispatch) —
    /// kept `#[cfg(test)]`-only as the reference this crate tests the fused
    /// kernel against, see `rope_fused_matches_apply_rope` below.
    #[cfg(test)]
    fn slice(&self, offset: usize, len: usize) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let cos = self.cos.clone().narrow(0, offset, len).unsqueeze::<4>();
        let sin = self.sin.clone().narrow(0, offset, len).unsqueeze::<4>();
        (cos, sin)
    }
}

/// `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`. F1: superseded on
/// the production path by `gguf::rope_fused` — kept `#[cfg(test)]`-only,
/// see `RoPE::slice`'s doc comment.
#[cfg(test)]
fn rotate_half(x: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 4> {
    let d = x.dims()[3];
    let half = d / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3)
}

/// `x_rope = x * cos + rotate_half(x) * sin`. F1: superseded on the
/// production path by `gguf::rope_fused` — kept `#[cfg(test)]`-only, see
/// `RoPE::slice`'s doc comment.
#[cfg(test)]
fn apply_rope(x: Tensor<Wgpu, 4>, cos: Tensor<Wgpu, 4>, sin: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 4> {
    x.clone() * cos + rotate_half(x) * sin
}

// ---------------------------------------------------------------------------
// RmsNorm wrapper
// ---------------------------------------------------------------------------

/// RMSNorm layer wrapping burn::nn::RmsNorm for GGUF weight loading.
pub struct RmsNormLayer {
    pub inner: burn::nn::RmsNorm<Wgpu>,
}

impl RmsNormLayer {
    /// D2 (docs/BENCHMARKS.md Session 3): dispatches the fused WGSL kernel
    /// (`gguf.rs::rmsnorm_fused`) instead of burn-nn's unfused
    /// cast/square/mean_dim/add/sqrt/div/mul chain (~8 dispatches). Matches
    /// `burn::nn::RmsNorm::forward`'s math exactly — see
    /// `wgsl/shader_rmsnorm.wgsl`'s doc comment and
    /// `tests/full_forward.rs`'s unit test comparing the two.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        crate::gguf::rmsnorm_fused(x, self.inner.gamma.val(), self.inner.epsilon as f32)
    }
}

// ---------------------------------------------------------------------------
// Q4Attention — GQA with q/k/v bias
// ---------------------------------------------------------------------------

/// Grouped-query attention: `n_heads` query heads, `n_kv_heads` key/value
/// heads (`n_heads / n_kv_heads` query heads share each KV head). Qwen2's
/// q/k/v projections carry bias; `attn_output` doesn't (docs/MODELS.md §2).
pub struct Q4Attention {
    q_proj: Q4Linear,
    k_proj: Q4Linear,
    v_proj: Q4Linear,
    o_proj: Q4Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Q4Attention {
    pub fn new(
        q_proj: Q4Linear,
        k_proj: Q4Linear,
        v_proj: Q4Linear,
        o_proj: Q4Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    /// `x`: `[1, T, hidden]`. `cache`/`layer_idx`/`offset` give the absolute
    /// position (for RoPE) and prior KV length (for the causal mask) — see
    /// `kv.rs` for cache layout. Returns `[1, T, hidden]`.
    fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KvCache,
        layer_idx: usize,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let [b, t, _] = x.dims();
        assert_eq!(b, 1, "batch size 1 only (single-session MCP agent)");

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // F1 (docs/BENCHMARKS.md Session 9): apply fused RoPE in the
        // natural [1, T, H, Dh] reshape() layout (one dispatch for both q
        // and k), *then* permute to [1, H, T, Dh] — see
        // `gguf::rope_fused`'s doc comment for why this ordering avoids an
        // extra `into_contiguous` versus fusing after permute.
        let q = q.reshape([b, t, self.n_heads, self.head_dim]);
        let k = k.reshape([b, t, self.n_kv_heads, self.head_dim]);
        let v = v
            .reshape([b, t, self.n_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let (q, k) = crate::gguf::rope_fused(q, k, &rope.cos, &rope.sin, offset);
        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);

        let (k_all, v_all) = cache.append(layer_idx, k, v);
        let kv_len = offset + t;

        let n_rep = self.n_heads / self.n_kv_heads;
        let k_all = repeat_kv(k_all, n_rep);
        let v_all = repeat_kv(v_all, n_rep);

        let out = attention_scores_and_values(q, k_all, v_all, t, kv_len, offset, self.scale);
        let out = out.permute([0, 2, 1, 3]).reshape([b, t, self.n_heads * self.head_dim]);

        self.o_proj.forward(out)
    }
}

/// Repeat each of `n_kv_heads` KV heads `n_rep` times along the head axis
/// (axis 1) so shapes line up with `n_heads = n_kv_heads * n_rep` query
/// heads — HF `repeat_kv`.
/// Largest query-chunk size for QK^T/softmax/PV. Burn 0.20's wgpu backend
/// (built here without the `autotune` cubecl feature, see workspace
/// Cargo.toml) falls back to a fixed `Strategy::Auto` matmul kernel
/// (`cubek-matmul`'s `SimpleCyclicCmma`) that panics with "shared memory ...
/// hardware limit" on this M2/Metal adapter once the query dimension of
/// QK^T gets into the low thousands (hit at T=2225 while prefilling C3's
/// `02_tools_single` fixture — `SimpleCyclicCmma` doesn't fall back
/// gracefully on an `InvalidConfig` error, only on `Unavailable`). Chunking
/// the query dimension keeps each individual `matmul` call's shape inside
/// the region the fixed strategy handles, at the cost of
/// `ceil(T/ATTN_QUERY_CHUNK)` separate score/softmax/PV passes per layer
/// during prefill (decode, T=1, never chunks).
///
/// K3 (docs/BENCHMARKS.md): 256 doesn't panic on this M2/Metal adapter and
/// is bit-identical to 128 (full_forward's greedy-match tests still pass);
/// prefill tok/s is unchanged within noise (26.3 vs 26.5 tok/s on
/// `02_tools_single`) since attention chunking overhead isn't prefill's
/// bottleneck — the naive Q4_0 matmul kernel (K2) is. 512 still panics
/// (unverified after this change, carried over from C3's finding).
const ATTN_QUERY_CHUNK: usize = 256;

/// Largest single contraction (K) block for the P@V matmul (`probs`:
/// `[.., Tq, kv_len]`, `v`: `[.., kv_len, Dh]`). Session 7 root-caused a
/// Burn/cubecl wgpu matmul correctness bug (not a shape *panic* like
/// `ATTN_QUERY_CHUNK`'s, a silently wrong result) that appears once the
/// contraction dimension `kv_len` is in the low thousands while the output
/// width `Dh` (head_dim) is comparatively small — confirmed with a plain
/// random-data `matmul([1,H,256,1256],[1,H,1256,16])` outside any
/// attention/softmax code (worst_abs_diff ~34 against a CPU f64 reference;
/// QK^T and softmax on the same inputs matched the CPU reference to
/// ~1e-6/~1e-8). Splitting the contraction into `PV_KV_CHUNK`-sized blocks
/// and summing partial products (`pv_matmul` below) keeps every individual
/// `matmul` call's K dimension small and avoids the bad kernel selection;
/// verified against a CPU reference at K=1256 with block=128 (bit-close,
/// max-abs ~1e-6) where the single-call matmul was off by ~0.1.
const PV_KV_CHUNK: usize = 256;

/// `probs`: `[1, H, Tq, kv_len]`, `v`: `[1, H, kv_len, Dh]`. Returns
/// `[1, H, Tq, Dh]`. See `PV_KV_CHUNK`'s doc comment for why this chunks the
/// contraction dimension instead of calling `probs.matmul(v)` directly.
fn pv_matmul(probs: Tensor<Wgpu, 4>, v: Tensor<Wgpu, 4>) -> Tensor<Wgpu, 4> {
    let kv_len = probs.dims()[3];
    if kv_len <= PV_KV_CHUNK {
        return probs.matmul(v);
    }
    let mut acc: Option<Tensor<Wgpu, 4>> = None;
    let mut start = 0usize;
    while start < kv_len {
        let len = PV_KV_CHUNK.min(kv_len - start);
        let p_chunk = probs.clone().narrow(3, start, len);
        let v_chunk = v.clone().narrow(2, start, len);
        let part = p_chunk.matmul(v_chunk);
        acc = Some(match acc {
            Some(prev) => prev + part,
            None => part,
        });
        start += len;
    }
    acc.unwrap()
}

/// QK^T -> causal mask -> softmax -> PV, chunked over the query (T)
/// dimension — see `ATTN_QUERY_CHUNK`'s doc comment. `q`: `[1, H, T, Dh]`,
/// `k_all`/`v_all`: `[1, H, kv_len, Dh]`. Returns `[1, H, T, Dh]`.
fn attention_scores_and_values(
    q: Tensor<Wgpu, 4>,
    k_all: Tensor<Wgpu, 4>,
    v_all: Tensor<Wgpu, 4>,
    t: usize,
    kv_len: usize,
    offset: usize,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    if t <= ATTN_QUERY_CHUNK {
        let scores = q.matmul(k_all.swap_dims(2, 3)) * scale;
        let scores = apply_causal_mask(scores, t, kv_len, offset);
        let probs = softmax(scores, 3);
        return pv_matmul(probs, v_all);
    }

    let mut chunks = Vec::with_capacity(t.div_ceil(ATTN_QUERY_CHUNK));
    let mut start = 0usize;
    while start < t {
        let len = ATTN_QUERY_CHUNK.min(t - start);
        let q_chunk = q.clone().narrow(2, start, len);
        // This chunk's queries are at absolute positions
        // [offset+start, offset+start+len); they may attend to keys
        // [0, offset+start+len).
        let chunk_kv_len = offset + start + len;
        let k_chunk = k_all.clone().narrow(2, 0, chunk_kv_len);
        let v_chunk = v_all.clone().narrow(2, 0, chunk_kv_len);
        let scores = q_chunk.matmul(k_chunk.swap_dims(2, 3)) * scale;
        let scores = apply_causal_mask(scores, len, chunk_kv_len, offset + start);
        let probs = softmax(scores, 3);
        chunks.push(pv_matmul(probs, v_chunk));
        start += len;
    }
    Tensor::cat(chunks, 2)
}

fn repeat_kv(x: Tensor<Wgpu, 4>, n_rep: usize) -> Tensor<Wgpu, 4> {
    if n_rep == 1 {
        return x;
    }
    let n_kv_heads = x.dims()[1];
    let mut heads = Vec::with_capacity(n_kv_heads * n_rep);
    for h in 0..n_kv_heads {
        let slice = x.clone().narrow(1, h, 1);
        for _ in 0..n_rep {
            heads.push(slice.clone());
        }
    }
    Tensor::cat(heads, 1)
}

/// Additive-mask-free causal mask via `mask_fill`: position `i` (absolute
/// `offset + i`) may attend to key `j` iff `j <= offset + i`. `scores`:
/// `[1, H, T, kv_len]`.
fn apply_causal_mask(
    scores: Tensor<Wgpu, 4>,
    t: usize,
    kv_len: usize,
    offset: usize,
) -> Tensor<Wgpu, 4> {
    let device = scores.device();
    let q_pos = Tensor::<Wgpu, 1, Int>::arange(0..t as i64, &device)
        .reshape([t, 1])
        .add_scalar(offset as i64);
    let k_pos = Tensor::<Wgpu, 1, Int>::arange(0..kv_len as i64, &device).reshape([1, kv_len]);
    // true where key position is in the future (masked out)
    let mask = k_pos.greater(q_pos).unsqueeze::<4>();
    scores.mask_fill(mask, f32::NEG_INFINITY)
}

// ---------------------------------------------------------------------------
// Q4FeedForward — 3-matrix SwiGLU
// ---------------------------------------------------------------------------

/// SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
pub struct Q4FeedForward {
    gate_proj: Q4Linear,
    up_proj: Q4Linear,
    down_proj: Q4Linear,
}

impl Q4FeedForward {
    pub fn new(gate_proj: Q4Linear, up_proj: Q4Linear, down_proj: Q4Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// F2 (docs/BENCHMARKS.md Session 9): `gguf::silu_mul_fused` replaces
    /// Burn's separate `silu` + `mul` chain with one dispatch.
    fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        let fused = crate::gguf::silu_mul_fused(gate, up);
        self.down_proj.forward(fused)
    }
}

// ---------------------------------------------------------------------------
// Q4TransformerBlock
// ---------------------------------------------------------------------------

/// Pre-LN (RMSNorm) transformer block: `x += attn(norm(x)); x += ffn(norm(x))`.
pub struct Q4TransformerBlock {
    attention_norm: RmsNormLayer,
    attention: Q4Attention,
    ffn_norm: RmsNormLayer,
    ffn: Q4FeedForward,
}

impl Q4TransformerBlock {
    pub fn new(
        attention_norm: RmsNormLayer,
        attention: Q4Attention,
        ffn_norm: RmsNormLayer,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KvCache,
        layer_idx: usize,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let attn_out = self
            .attention
            .forward(self.attention_norm.forward(x.clone()), rope, cache, layer_idx, offset);
        let x = x + attn_out;
        let ffn_out = self.ffn.forward(self.ffn_norm.forward(x.clone()));
        x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// LlmModel
// ---------------------------------------------------------------------------

/// The complete Qwen2 decoder. `lm_head` is tied to `embed`'s Q4 buffer (same
/// GPU handle, shared via `Q4Tensor::clone` at load time — docs/MODELS.md §2:
/// no independent `output.weight` tensor exists in this GGUF).
/// A forced run shorter than this many tokens is decoded one token at a
/// time via ordinary masked argmax instead of the batched jump-forward
/// prefill path (`LlmModel::decode_with_constraint`) — exact either way,
/// since the mask already forces the same token(s); below this length the
/// batched path's own overhead (the forced-run's tokenizer encode/decode
/// round trip in `GrammarConstraint::forced_run`) isn't worth it. `8` is a
/// starting value, not derived from a model (session 12 speed addendum,
/// `docs/BENCHMARKS.md`).
const JUMP_MIN_TOKENS: usize = 8;

pub struct LlmModel {
    embed: EmbeddingStore,
    layers: Vec<Q4TransformerBlock>,
    rope: RoPE,
    out_norm: RmsNormLayer,
    lm_head: Q4Linear,
    config: LlmConfig,
    device: WgpuDevice,
}

impl LlmModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed: EmbeddingStore,
        layers: Vec<Q4TransformerBlock>,
        rope: RoPE,
        out_norm: RmsNormLayer,
        lm_head: Q4Linear,
        config: LlmConfig,
        device: WgpuDevice,
    ) -> Self {
        Self {
            embed,
            layers,
            rope,
            out_norm,
            lm_head,
            config,
            device,
        }
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// New KV cache sized for this model, `max_ctx` timesteps.
    pub fn new_cache(&self, max_ctx: usize) -> KvCache {
        KvCache::new(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.hidden_size / self.config.num_heads,
            max_ctx,
            &self.device,
        )
    }

    /// Embed `token_ids` on CPU (per-row Q4 dequant) and upload as `[1, T, hidden]`.
    /// `pub` (not just used internally) so `llm-agent bench` can time the
    /// CPU dequant + upload step in isolation (docs/BENCHMARKS.md P1c).
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor<Wgpu, 3>> {
        let hidden = self.config.hidden_size;
        let mut data = vec![0.0f32; token_ids.len() * hidden];
        for (i, &id) in token_ids.iter().enumerate() {
            self.embed
                .embed_id_add_cpu(id, &mut data[i * hidden..(i + 1) * hidden])?;
        }
        Ok(
            Tensor::<Wgpu, 1>::from_floats(data.as_slice(), &self.device).reshape([
                1,
                token_ids.len(),
                hidden,
            ]),
        )
    }

    /// Run all transformer layers + final norm over `token_ids`, appending to
    /// `cache` at its current length. Returns hidden states `[1, T, hidden]`
    /// (post `output_norm`, pre lm_head — callers slice before calling
    /// `lm_head` to avoid materializing `T x 151936` logits when only a few
    /// positions are needed, e.g. prefill's last-token generation step).
    pub fn forward_hidden(&self, token_ids: &[u32], cache: &mut KvCache) -> Result<Tensor<Wgpu, 3>> {
        let offset = cache.len();
        let mut x = self.embed_tokens(token_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, &self.rope, cache, i, offset);
        }
        cache.advance(token_ids.len());
        Ok(self.out_norm.forward(x))
    }

    /// lm_head over hidden states `[1, T, hidden]` -> logits `[1, T, vocab]`.
    /// Callers should narrow `hidden` to only the positions they need first
    /// (the 151936-wide head is 0.6MB/row of f32 output).
    pub fn lm_head(&self, hidden: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        self.lm_head.forward(hidden)
    }

    /// Convenience: full forward + full-width logits for every position.
    /// Only use for small T (tests); for real prefill, slice `forward_hidden`'s
    /// output to the positions you need before calling `lm_head`.
    pub fn forward_logits(&self, token_ids: &[u32], cache: &mut KvCache) -> Result<Tensor<Wgpu, 3>> {
        let hidden = self.forward_hidden(token_ids, cache)?;
        Ok(self.lm_head(hidden))
    }

    /// Greedy-decode up to `max_new` tokens after prefilling `prompt_ids`
    /// into `cache` (which may already hold a restored prefix — see
    /// `KvCache::snapshot`/`restore`). Stops after appending a generated id
    /// that's in `stop_ids` (the stop token IS included in the returned
    /// vec — matches `fixtures/reference/logits/*.json`'s
    /// `greedy_first_32_token_ids` convention). Only the last prefill
    /// position's logits are computed (never the full `T x vocab` matrix)
    /// — see module doc comment on `forward_hidden`. Unconstrained
    /// convenience wrapper over `generate_with_constraint`.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        stop_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<u32>> {
        let (ids, _stats) = self.generate_with_constraint(prompt_ids, max_new, stop_ids, cache, None)?;
        Ok(ids)
    }

    /// Like `generate`, but drives an optional [`Constraint`] (see
    /// `grammar.rs`) through the decode loop: at every step, if
    /// `constraint.forced_run()` returns a non-empty run (the mask allows
    /// exactly one token at each of the next several positions), those
    /// tokens are appended without individual argmax/forward-pass-per-token
    /// decode steps — instead, one `forward_hidden` prefill call of `M =
    /// run.len()` runs them all through in a single model step, appending
    /// to `cache` the same way prefill does (`docs/ENGINE.md` "Schema-
    /// constrained decoding", jump-forward semantics). Otherwise, one
    /// normal masked-argmax decode step runs (`sample::greedy_masked` when
    /// the constraint has a mask, plain `sample::greedy` when
    /// unconstrained). Returns the generated ids plus a step/token
    /// breakdown for eval reporting.
    pub fn generate_with_constraint(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        stop_ids: &[u32],
        cache: &mut KvCache,
        constraint: Option<&mut dyn Constraint>,
    ) -> Result<(Vec<u32>, GenerateStats)> {
        assert!(!prompt_ids.is_empty());
        let hidden = self.forward_hidden(prompt_ids, cache)?;
        let last = hidden.narrow(1, prompt_ids.len() - 1, 1);
        let logits = self.lm_head(last);
        let logits_vec = logits_to_vec(logits)?;
        self.decode_with_constraint(logits_vec, max_new, stop_ids, cache, constraint)
    }

    /// The decode half of `generate_with_constraint`, split out so a
    /// caller that already has its own prefill logic (e.g.
    /// `bin/llm-agent.rs`'s `NativeGenerator`, which prefills only the
    /// suffix not already resident via its own KV-prefix-reuse bookkeeping)
    /// can reuse the jump-forward decode loop without re-prefilling
    /// `prompt_ids` from scratch. `logits_vec` is the last prefill
    /// position's logits (`[vocab]`, already read back to CPU).
    pub fn decode_with_constraint(
        &self,
        mut logits_vec: Vec<f32>,
        max_new: usize,
        stop_ids: &[u32],
        cache: &mut KvCache,
        mut constraint: Option<&mut dyn Constraint>,
    ) -> Result<(Vec<u32>, GenerateStats)> {
        let mut out = Vec::with_capacity(max_new);
        let mut stats = GenerateStats::default();

        while out.len() < max_new {
            let forced = constraint.as_deref().and_then(Constraint::forced_run).unwrap_or_default();
            // Below `JUMP_MIN_TOKENS`, a masked-argmax single step is exact
            // and free (the mask already allows exactly the tokens the
            // forced run would have picked), so the batched-prefill path
            // below buys nothing worth its own overhead (the forced-run
            // computation itself: DFA walk + tokenizer encode/decode
            // round trip) for a run this short — only take it once it's
            // long enough to actually save forward passes (session 12
            // speed addendum, `docs/BENCHMARKS.md`).
            if forced.len() >= JUMP_MIN_TOKENS {
                let take = forced.len().min(max_new - out.len());
                let run = &forced[..take];
                let hidden = self.forward_hidden(run, cache)?;
                for &t in run {
                    if let Some(c) = constraint.as_deref_mut() {
                        c.advance(t);
                    }
                }
                out.extend_from_slice(run);
                stats.model_steps += 1;
                stats.forced_tokens += run.len();
                if take < forced.len() {
                    // Hit max_new mid-run: stop without sampling further.
                    break;
                }
                let last = hidden.narrow(1, run.len() - 1, 1);
                let logits = self.lm_head(last);
                logits_vec = logits_to_vec(logits)?;
                continue;
            }

            let next = match constraint.as_deref().and_then(Constraint::allowed) {
                Some(mask) => crate::sample::greedy_masked(&logits_vec, mask),
                None => crate::sample::greedy(&logits_vec),
            };
            out.push(next);
            stats.model_steps += 1;
            if let Some(c) = constraint.as_deref_mut() {
                c.advance(next);
            }
            if stop_ids.contains(&next) {
                break;
            }
            let hidden = self.forward_hidden(&[next], cache)?;
            let logits = self.lm_head(hidden);
            logits_vec = logits_to_vec(logits)?;
        }
        stats.total_tokens = out.len();
        Ok((out, stats))
    }
}

/// Model-step / forced-token breakdown for `generate_with_constraint`
/// (`docs/ENGINE.md` "Schema-constrained decoding"). `model_steps` counts
/// forward passes (one jump-forward run of `k` forced tokens is 1 step, not
/// `k`); `forced_tokens` + the sampled-token count equal `total_tokens`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateStats {
    pub model_steps: usize,
    pub forced_tokens: usize,
    pub total_tokens: usize,
}

/// Extract a `[1, 1, vocab]` (or `[1, T=1, vocab]`) logits tensor to a flat
/// `Vec<f32>`. Native-only sync readback (`into_data()`); WASM callers must
/// use `into_data_async().await` instead (see crate-level WASM constraints).
pub fn logits_to_vec(logits: Tensor<Wgpu, 3>) -> Result<Vec<f32>> {
    logits
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to read back f32 logits: {e:?}"))
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn causal_mask_pattern() {
        let device = WgpuDevice::default();
        let scores = Tensor::<Wgpu, 4>::zeros([1, 1, 3, 3], &device);
        let masked = apply_causal_mask(scores, 3, 3, 0);
        let data = masked.into_data().into_vec::<f32>().unwrap();
        println!("mask pattern: {data:?}");
        // Expect row i can see cols 0..=i: masked (-inf) where col>row.
        assert_eq!(data[0], 0.0); // (0,0)
        assert!(data[1].is_infinite() && data[1] < 0.0); // (0,1) masked
        assert!(data[2].is_infinite() && data[2] < 0.0); // (0,2) masked
        assert_eq!(data[3], 0.0); // (1,0)
        assert_eq!(data[4], 0.0); // (1,1)
        assert!(data[5].is_infinite() && data[5] < 0.0); // (1,2) masked
        assert_eq!(data[6], 0.0); // (2,0)
        assert_eq!(data[7], 0.0); // (2,1)
        assert_eq!(data[8], 0.0); // (2,2)
    }

    #[test]
    fn causal_mask_with_offset() {
        let device = WgpuDevice::default();
        // offset=2 (2 cached tokens), T=1 new query at absolute pos 2, kv_len=3.
        let scores = Tensor::<Wgpu, 4>::zeros([1, 1, 1, 3], &device);
        let masked = apply_causal_mask(scores, 1, 3, 2);
        let data = masked.into_data().into_vec::<f32>().unwrap();
        println!("offset mask pattern: {data:?}");
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
    }

    /// Session 7 regression test for the chunked-attention divergence
    /// (docs/BENCHMARKS.md Session 5/6 addenda, `split_prefill_matches_single_prefill`):
    /// synthetic q/k/v at offset=1000, T=1225 (mirrors the real 1000/1225
    /// split). Compares `attention_scores_and_values`'s chunked branch
    /// (T=1225 > ATTN_QUERY_CHUNK=256) against the same math computed in one
    /// unchunked shot (the T<=256 branch's formula, called directly) on
    /// *identical* q/k/v tensors — isolated from gguf.rs/RoPE/KV-cache
    /// entirely. Root cause (proven by bisecting QK^T / softmax / PV matmul
    /// independently against a CPU f64 reference, and reproducing with
    /// plain random-data matmul calls with no attention involved at all):
    /// Burn/cubecl's wgpu matmul kernel silently mis-computes
    /// `probs.matmul(v)` once the contraction dim (`kv_len`, here 1256) is
    /// in the low thousands while the output width (`Dh`, head_dim) is
    /// comparatively small — QK^T and softmax on the same inputs matched a
    /// CPU reference to ~1e-6/~1e-8, only the PV matmul was wrong (~0.1
    /// abs, matching this test's failure before the fix). Fixed by
    /// `pv_matmul` chunking the contraction dimension into `PV_KV_CHUNK`
    /// blocks and summing partial products.
    #[test]
    fn chunked_attention_matches_unchunked_synthetic() {
        let device = WgpuDevice::default();
        let h = 4usize;
        let dh = 16usize;
        let offset = 1000usize;
        let t = 1225usize;
        let kv_len = offset + t;

        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng as i64 % 2000) as f32) / 1000.0 - 1.0
        };

        let q_data: Vec<f32> = (0..h * t * dh).map(|_| next()).collect();
        let k_data: Vec<f32> = (0..h * kv_len * dh).map(|_| next()).collect();
        let v_data: Vec<f32> = (0..h * kv_len * dh).map(|_| next()).collect();

        let q = Tensor::<Wgpu, 1>::from_floats(q_data.as_slice(), &device).reshape([1, h, t, dh]);
        let k_all =
            Tensor::<Wgpu, 1>::from_floats(k_data.as_slice(), &device).reshape([1, h, kv_len, dh]);
        let v_all =
            Tensor::<Wgpu, 1>::from_floats(v_data.as_slice(), &device).reshape([1, h, kv_len, dh]);

        let scale = 1.0 / (dh as f32).sqrt();

        // Chunked path (real code path: T=1225 > ATTN_QUERY_CHUNK).
        let chunked = attention_scores_and_values(
            q.clone(),
            k_all.clone(),
            v_all.clone(),
            t,
            kv_len,
            offset,
            scale,
        );

        // Unchunked reference: identical formula to the T<=ATTN_QUERY_CHUNK
        // branch, called directly on the same (whole, unsliced) tensors.
        let scores = q.matmul(k_all.swap_dims(2, 3)) * scale;
        let scores = apply_causal_mask(scores, t, kv_len, offset);
        let probs = softmax(scores, 3);
        let unchunked = pv_matmul(probs, v_all);

        let chunked_data = chunked.into_data().into_vec::<f32>().unwrap();
        let unchunked_data = unchunked.into_data().into_vec::<f32>().unwrap();
        assert_eq!(chunked_data.len(), unchunked_data.len());

        let mut max_abs_diff = 0f32;
        for (a, b) in chunked_data.iter().zip(unchunked_data.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        println!("chunked vs unchunked synthetic attention: max_abs_diff={max_abs_diff}");
        assert!(
            max_abs_diff < 3e-4,
            "chunked attention diverges from unchunked reference by {max_abs_diff} \
             (offset={offset}, t={t}, kv_len={kv_len})"
        );
    }

    /// F1 (docs/BENCHMARKS.md Session 9): `gguf::rope_fused` must match the
    /// old `apply_rope`/`rotate_half`/`RoPE::slice` Burn-op chain this
    /// crate was previously using on the production path, to within float
    /// reassociation tolerance (task brief: <=1e-6).
    #[test]
    fn rope_fused_matches_apply_rope() {
        let device = WgpuDevice::default();
        let head_dim = 128usize;
        let h = 16usize;
        let hkv = 2usize;
        let t = 5usize;
        let offset = 37usize;
        let max_seq_len = 64usize;

        let rope = RoPE::new(head_dim, max_seq_len, 1_000_000.0, &device);

        let mut rng = 999u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng as i64 % 2000) as f32) / 1000.0 - 1.0
        };
        let q_data: Vec<f32> = (0..t * h * head_dim).map(|_| next()).collect();
        let k_data: Vec<f32> = (0..t * hkv * head_dim).map(|_| next()).collect();

        // Reference: old apply_rope on [1, H, T, Dh] (post-permute) layout.
        let q_ref = Tensor::<Wgpu, 1>::from_floats(q_data.as_slice(), &device)
            .reshape([1, t, h, head_dim])
            .permute([0, 2, 1, 3]);
        let k_ref = Tensor::<Wgpu, 1>::from_floats(k_data.as_slice(), &device)
            .reshape([1, t, hkv, head_dim])
            .permute([0, 2, 1, 3]);
        let (cos, sin) = rope.slice(offset, t);
        let q_ref = apply_rope(q_ref, cos.clone(), sin.clone());
        let k_ref = apply_rope(k_ref, cos, sin);
        let q_ref = q_ref.permute([0, 2, 1, 3]); // back to [1, T, H, Dh]
        let k_ref = k_ref.permute([0, 2, 1, 3]);

        // Fused: gguf::rope_fused on [1, T, H, Dh] (pre-permute) layout.
        let q_fused = Tensor::<Wgpu, 1>::from_floats(q_data.as_slice(), &device)
            .reshape([1, t, h, head_dim]);
        let k_fused = Tensor::<Wgpu, 1>::from_floats(k_data.as_slice(), &device)
            .reshape([1, t, hkv, head_dim]);
        let (q_fused, k_fused) =
            crate::gguf::rope_fused(q_fused, k_fused, &rope.cos, &rope.sin, offset);

        let q_ref_data = q_ref.into_data().into_vec::<f32>().unwrap();
        let q_fused_data = q_fused.into_data().into_vec::<f32>().unwrap();
        let k_ref_data = k_ref.into_data().into_vec::<f32>().unwrap();
        let k_fused_data = k_fused.into_data().into_vec::<f32>().unwrap();

        let mut max_abs_diff = 0f32;
        for (a, b) in q_ref_data.iter().zip(q_fused_data.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        for (a, b) in k_ref_data.iter().zip(k_fused_data.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        println!("rope_fused vs apply_rope: max_abs_diff={max_abs_diff}");
        assert!(max_abs_diff < 1e-6, "rope_fused diverges by {max_abs_diff}");
    }

    /// F2 (docs/BENCHMARKS.md Session 9): `gguf::silu_mul_fused` must match
    /// `silu(gate) * up` computed via Burn's own ops.
    #[test]
    fn silu_mul_fused_matches_burn() {
        use burn::tensor::activation::silu;

        let device = WgpuDevice::default();
        let m = 3usize;
        let n = 4096usize;

        let mut rng = 4242u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng as i64 % 4000) as f32) / 1000.0 - 2.0
        };
        let gate_data: Vec<f32> = (0..m * n).map(|_| next()).collect();
        let up_data: Vec<f32> = (0..m * n).map(|_| next()).collect();

        let gate = Tensor::<Wgpu, 1>::from_floats(gate_data.as_slice(), &device).reshape([1, m, n]);
        let up = Tensor::<Wgpu, 1>::from_floats(up_data.as_slice(), &device).reshape([1, m, n]);

        let reference = silu(gate.clone()) * up.clone();
        let fused = crate::gguf::silu_mul_fused(gate, up);

        let ref_data = reference.into_data().into_vec::<f32>().unwrap();
        let fused_data = fused.into_data().into_vec::<f32>().unwrap();

        let mut max_abs_diff = 0f32;
        for (a, b) in ref_data.iter().zip(fused_data.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        println!("silu_mul_fused vs burn silu*mul: max_abs_diff={max_abs_diff}");
        // 5e-6, not 1e-6: burn-nn's `silu` uses a different but
        // mathematically equivalent formula (sigmoid(x)*x via a fused
        // sigmoid op) than this kernel's `x/(1+exp(-x))`; the two
        // reassociate f32 rounding differently. Measured max diff here is
        // ~1.9e-6 (well under 5e-6), consistent with float32 ULP-scale
        // reassociation noise, not a formula bug.
        assert!(max_abs_diff < 5e-6, "silu_mul_fused diverges by {max_abs_diff}");
    }
}
