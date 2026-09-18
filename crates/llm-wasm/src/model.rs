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
use burn::backend::wgpu::{into_contiguous, CubeTensor, Wgpu, WgpuDevice, WgpuRuntime};
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, DType, Int, Tensor, TensorData, TensorPrimitive};

use cubecl::server::Handle;
use cubecl::Runtime;

use crate::gguf::{EmbeddingStore, Q4Linear};
use crate::grammar::Constraint;
use crate::kv::{KvCache, KvDtype};
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
// ForwardSpec — caller-supplied positions and attention topology
// ---------------------------------------------------------------------------

/// Per-call overrides of the two things a decoder-only LM normally hard-codes:
/// "my position is my index" and "I attend to everything before me".
///
/// Added for `llm-life`, which uses the model as a cellular-automaton update
/// rule: one forward pass carries many independent cells, so RoPE positions
/// restart (or collapse to a constant) per cell and attention follows a 2D
/// stencil instead of a causal chain. Both fields are `None` by default and
/// that reproduces the previous behavior exactly — positions
/// `offset..offset+T`, causal mask.
/// A sparse attention topology: query `i` attends to the contiguous key
/// range `[0, prefix_len[i])` plus `n_keys[i]` explicit key indices taken
/// from row `i` of `keys` (row stride `max_keys`).
///
/// This is the same information a dense `[T, kv_len]` bool mask carries, in
/// the form the stencil actually has it — and in a form that does not cost
/// `T * kv_len` bytes to build or to hold. At `llm-life`'s 128x128 grid the
/// dense mask alone is a 271 MB `Vec<bool>` plus the same again on the GPU,
/// before any attention work happens; this is 16452 * (2 + 9) u32.
///
/// The three vectors are uploaded once here, not once per layer: all 24
/// layers of one forward read the same two buffers.
///
/// Consumed by `wgsl/shader_attn_sparse.wgsl` — see that file for the
/// kernel's limits (`gguf::MAX_SPARSE_KEYS`, `MAX_SPARSE_HEAD_DIM`,
/// `MAX_SPARSE_QUERIES`). `ForwardSpec` may carry both this and `mask_out`;
/// the sparse path wins where it applies and the dense mask stays as the
/// fallback and as the test oracle (`tests/stencil.rs`).
#[derive(Clone)]
pub struct SparseMask {
    meta: Handle,
    keys: Handle,
    t: usize,
    kv_len: usize,
    max_keys: usize,
}

impl SparseMask {
    /// `prefix_len` and `n_keys` are one entry per query; `keys` is
    /// row-major `[t, max_keys]` (entries past `n_keys[i]` are never read,
    /// so they may be anything). Panics on a mask the kernel cannot run:
    /// an empty row (softmax over no keys is NaN), an out-of-range key, or
    /// a row with more than `gguf::MAX_SPARSE_KEYS` attended keys.
    pub fn new(
        prefix_len: &[u32],
        n_keys: &[u32],
        keys: &[u32],
        max_keys: usize,
        kv_len: usize,
        device: &WgpuDevice,
    ) -> Self {
        let t = prefix_len.len();
        assert_eq!(n_keys.len(), t, "n_keys must be one per query");
        assert_eq!(keys.len(), t * max_keys, "keys must be [t, max_keys]");
        let mut meta = Vec::with_capacity(t * 2);
        for i in 0..t {
            let total = prefix_len[i] as usize + n_keys[i] as usize;
            assert!(total > 0, "query {i} attends to no key (softmax would be NaN)");
            assert!(
                total <= crate::gguf::MAX_SPARSE_KEYS,
                "query {i} attends to {total} keys, kernel limit is {}",
                crate::gguf::MAX_SPARSE_KEYS
            );
            assert!(prefix_len[i] as usize <= kv_len, "query {i}'s prefix exceeds kv_len");
            assert!(n_keys[i] as usize <= max_keys, "query {i} has more keys than max_keys");
            for j in 0..n_keys[i] as usize {
                assert!(
                    (keys[i * max_keys + j] as usize) < kv_len,
                    "query {i} key {j} is out of range"
                );
            }
            meta.push(prefix_len[i]);
            meta.push(n_keys[i]);
        }
        let client = WgpuRuntime::client(device);
        let meta_bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        // A zero-length storage buffer is not bindable; `max_keys == 0`
        // (a mask that is nothing but prefix ranges) still needs one word.
        let keys_bytes: Vec<u8> = if keys.is_empty() {
            vec![0u8; 4]
        } else {
            keys.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        Self {
            meta: client.create_from_slice(&meta_bytes),
            keys: client.create_from_slice(&keys_bytes),
            t,
            kv_len,
            max_keys,
        }
    }

    /// Build from the same row-major `[t, kv_len]` "may attend" mask
    /// `ForwardSpec::with_allowed` takes, choosing the longest leading run
    /// of allowed keys as the prefix range and listing the rest explicitly.
    /// This is the bridge the tests use to check the kernel against the
    /// dense path on one and the same mask; production callers build the
    /// sparse form directly and never materialize `allowed`.
    pub fn from_allowed(allowed: &[bool], t: usize, kv_len: usize, device: &WgpuDevice) -> Self {
        assert_eq!(allowed.len(), t * kv_len, "allowed mask is not [t, kv_len]");
        let mut prefix_len = Vec::with_capacity(t);
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(t);
        for i in 0..t {
            let row = &allowed[i * kv_len..(i + 1) * kv_len];
            let p = row.iter().take_while(|&&a| a).count();
            prefix_len.push(p as u32);
            rows.push(
                (p..kv_len)
                    .filter(|&j| row[j])
                    .map(|j| j as u32)
                    .collect(),
            );
        }
        let max_keys = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let n_keys: Vec<u32> = rows.iter().map(|r| r.len() as u32).collect();
        let mut keys = vec![0u32; t * max_keys];
        for (i, r) in rows.iter().enumerate() {
            keys[i * max_keys..i * max_keys + r.len()].copy_from_slice(r);
        }
        Self::new(&prefix_len, &n_keys, &keys, max_keys, kv_len, device)
    }

    pub fn t(&self) -> usize {
        self.t
    }

    pub fn kv_len(&self) -> usize {
        self.kv_len
    }
}

#[derive(Clone, Default)]
pub struct ForwardSpec {
    /// Absolute RoPE position of each token in this call; length must equal
    /// the token count. `None` = `offset..offset+T`.
    pub positions: Option<Vec<u32>>,
    /// `[T, kv_len]`, `true` where the attention score must be masked OUT.
    /// Stored pre-inverted (not as "allowed") because every layer reuses this
    /// one tensor and the inversion would otherwise be paid per layer.
    pub mask_out: Option<Tensor<Wgpu, 2, Bool>>,
    /// The same topology in sparse form — see [`SparseMask`]. Takes
    /// precedence over `mask_out` wherever the fused kernel applies.
    pub sparse: Option<SparseMask>,
}

impl ForwardSpec {
    /// True when this spec asks for nothing, i.e. the fast paths (fused RoPE,
    /// fused decode attention) are still valid.
    pub fn is_default(&self) -> bool {
        self.positions.is_none() && self.mask_out.is_none() && self.sparse.is_none()
    }

    pub fn with_sparse(mut self, sparse: SparseMask) -> Self {
        self.sparse = Some(sparse);
        self
    }

    pub fn with_positions(mut self, positions: Vec<u32>) -> Self {
        self.positions = Some(positions);
        self
    }

    /// `allowed` is row-major `[T, kv_len]`, `true` where query `i` may attend
    /// to key `j`. Every row must allow at least one key — a fully masked row
    /// softmaxes to NaN.
    pub fn with_allowed(
        mut self,
        allowed: &[bool],
        t: usize,
        kv_len: usize,
        device: &WgpuDevice,
    ) -> Self {
        assert_eq!(allowed.len(), t * kv_len, "allowed mask is not [T, kv_len]");
        let masked_out: Vec<bool> = allowed.iter().map(|&a| !a).collect();
        self.mask_out = Some(Tensor::<Wgpu, 2, Bool>::from_data(
            TensorData::new(masked_out, [t, kv_len]),
            device,
        ));
        self
    }
}

/// `x * cos + rotate_half(x) * sin` over `[1, T, H, Dh]` with `cos`/`sin`
/// broadcast as `[1, T, 1, Dh]`.
fn apply_rope_rows(
    x: Tensor<Wgpu, 4>,
    cos: &Tensor<Wgpu, 4>,
    sin: &Tensor<Wgpu, 4>,
) -> Tensor<Wgpu, 4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
    x * cos.clone() + rotated * sin.clone()
}

/// RoPE at caller-supplied absolute positions. `gguf::rope_fused` bakes
/// `pos = offset + row` into its kernel, so arbitrary positions take this
/// pure-Burn path instead: gather the cos/sin rows by index, then apply.
/// Same layout as `rope_fused`: q `[1, T, H, Dh]`, k `[1, T, Hkv, Dh]`.
fn rope_positions(
    q: Tensor<Wgpu, 4>,
    k: Tensor<Wgpu, 4>,
    rope: &RoPE,
    positions: &[u32],
) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
    let device = q.device();
    let t = positions.len();
    let dh = q.dims()[3];
    assert_eq!(q.dims()[1], t, "positions length must equal the token count");
    let idx: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
    let idx = Tensor::<Wgpu, 1, Int>::from_data(TensorData::new(idx, [t]), &device);
    let cos = rope.cos.clone().select(0, idx.clone()).reshape([1, t, 1, dh]);
    let sin = rope.sin.clone().select(0, idx).reshape([1, t, 1, dh]);
    (apply_rope_rows(q, &cos, &sin), apply_rope_rows(k, &cos, &sin))
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
        spec: &ForwardSpec,
    ) -> Tensor<Wgpu, 3> {
        let [b, t, _] = x.dims();
        assert_eq!(b, 1, "batch size 1 only (single-session MCP agent)");

        let dev = x.device();
        let (q, k, v) = crate::profile::scope("attn.qkv_proj", &dev, || {
            (
                self.q_proj.forward(x.clone()),
                self.k_proj.forward(x.clone()),
                self.v_proj.forward(x.clone()),
            )
        });

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

        let (q, k) = crate::profile::scope("attn.rope", &dev, || match &spec.positions {
            Some(positions) => rope_positions(q, k, rope, positions),
            None => crate::gguf::rope_fused(q, k, &rope.cos, &rope.sin, offset),
        });
        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);

        crate::profile::scope("attn.cache_write", &dev, || cache.write(layer_idx, k, v));
        let kv_len = offset + t;

        // Session 13 (docs/BENCHMARKS.md): decode (t==1) against a
        // Q8_0-backed cache takes the fused kernel that reads K/V directly
        // out of the quantized cache (`gguf::attn_decode_q8_dispatch`,
        // `wgsl/shader_attn_decode_q8.wgsl`) — no dequant-to-f32 round trip,
        // no repeat_kv materialization, no Burn matmul/`pv_matmul` chunking
        // workaround. Session 14: decode against an F32-backed cache (the
        // production default, `KvDtype::F32`) takes the analogous fused
        // kernel with no dequant at all (`gguf::attn_decode_f32_dispatch`,
        // `wgsl/shader_attn_decode_f32.wgsl`), gated by `FUSED_DECODE_ATTN`
        // for A/B against the Burn-matmul path. Prefill (t>1) always keeps
        // the existing Burn-matmul attention path — see `kv.rs`'s module
        // doc comment for why prefill isn't fused yet.
        // A `ForwardSpec` mask replaces the causal mask entirely, so neither
        // fused decode kernel (both of which assume "attend to all of
        // kv_len") is valid; those paths stay on the default spec.
        let out = if let Some(sparse) = spec.sparse.as_ref().filter(|s| {
            cache.dtype() == KvDtype::F32
                && s.t == t
                && s.kv_len == kv_len
                && self.head_dim <= crate::gguf::MAX_SPARSE_HEAD_DIM
                && t <= crate::gguf::MAX_SPARSE_QUERIES
        }) {
            crate::profile::scope("attn.sparse", &dev, || {
                attn_sparse(
                    &q,
                    cache,
                    layer_idx,
                    sparse,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    self.scale,
                )
            })
        } else if let Some(mask_out) = &spec.mask_out {
            let n_rep = self.n_heads / self.n_kv_heads;
            let (k_all, v_all) = crate::profile::scope("attn.kv_read_repeat", &dev, || {
                let (k_all, v_all) = cache.read_or_dequant_f32(layer_idx, kv_len);
                (repeat_kv(k_all, n_rep), repeat_kv(v_all, n_rep))
            });
            assert_eq!(
                mask_out.dims(),
                [t, kv_len],
                "ForwardSpec mask must be [T, kv_len]"
            );
            crate::profile::scope("attn.dense_masked", &dev, || {
                attention_with_mask(q, k_all, v_all, t, mask_out, self.scale)
            })
        } else if t == 1 && cache.dtype() == KvDtype::Q8_0 {
            attn_decode_q8(&q, cache, layer_idx, self.n_heads, self.n_kv_heads, self.head_dim, kv_len, self.scale)
        } else if t == 1 && cache.dtype() == KvDtype::F32 && FUSED_DECODE_ATTN && kv_len <= FUSED_DECODE_ATTN_MAX_KV_LEN {
            attn_decode_f32(&q, cache, layer_idx, self.n_heads, self.n_kv_heads, self.head_dim, kv_len, self.scale)
        } else {
            let n_rep = self.n_heads / self.n_kv_heads;
            let (k_all, v_all) = cache.read_or_dequant_f32(layer_idx, kv_len);
            let k_all = repeat_kv(k_all, n_rep);
            let v_all = repeat_kv(v_all, n_rep);
            attention_scores_and_values(q, k_all, v_all, t, kv_len, offset, self.scale)
        };
        let out = out.permute([0, 2, 1, 3]).reshape([b, t, self.n_heads * self.head_dim]);

        crate::profile::scope("attn.o_proj", &dev, || self.o_proj.forward(out))
    }
}

/// Session 13 (docs/BENCHMARKS.md): decode-time (t==1) fused QK^T ->
/// softmax -> PV over a `KvDtype::Q8_0` cache. `q`: `[1, n_heads, 1,
/// head_dim]` (already RoPE'd). Returns `[1, n_heads, 1, head_dim]`
/// (matches `attention_scores_and_values`'s output shape so both branches
/// of `Q4Attention::forward` feed the same permute/reshape). Reuses one
/// thread-local scratch buffer (`n_heads * max_ctx` f32) across calls,
/// resized only when a larger cache is used — mirrors `gguf.rs`'s
/// `DEQUANT_SCRATCH` pattern.
#[allow(clippy::too_many_arguments)]
fn attn_decode_q8(
    q: &Tensor<Wgpu, 4>,
    cache: &KvCache,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    thread_local! {
        static ATTN_SCRATCH: std::cell::RefCell<Option<(cubecl::server::Handle, usize)>> =
            const { std::cell::RefCell::new(None) };
    }

    let q_cube: CubeTensor<WgpuRuntime> = into_contiguous(q.clone().into_primitive().tensor());
    let client = q_cube.client.clone();
    let device = q_cube.device.clone();
    let max_ctx = cache.max_ctx();
    let (k_scales, k_words, v_scales, v_words) = cache.q8_layer(layer_idx);

    let needed = n_heads * max_ctx;
    let scratch = ATTN_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let reuse = matches!(&*slot, Some((_, cap)) if *cap >= needed);
        if !reuse {
            *slot = Some((client.empty(needed * 4), needed));
        }
        slot.as_ref().unwrap().0.clone()
    });

    let out_handle = crate::gguf::attn_decode_q8_dispatch(
        &client, &q_cube.handle, k_scales, k_words, v_scales, v_words, &scratch, n_heads, n_kv_heads, head_dim, kv_len,
        max_ctx, scale,
    );
    let shape = burn::prelude::Shape::from(vec![1, n_heads, 1, head_dim]);
    let cube_tensor = CubeTensor::new_contiguous(client, device, shape, out_handle, DType::F32);
    Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
}

/// Session 14 (docs/BENCHMARKS.md): toggles the fused decode-attention
/// kernel (`attn_decode_f32`) on for the production `KvDtype::F32` cache.
/// Kept as a const (not a runtime flag) so the Burn-matmul path stays
/// compiled in and reachable by flipping this one bool — the A/B fallback
/// the task brief asked for — without a second code path to wire through
/// `LlmConfig`.
const FUSED_DECODE_ATTN: bool = true;

/// Session 14 (docs/BENCHMARKS.md): the fused kernel's one-workgroup-per-
/// head design (16 workgroups total, phase C's V-accumulation loop
/// unstrided over `kv_len` per thread) wins at kv_len~2225 (median 90.8 vs
/// 100.8 ms/token, fused vs Burn-matmul path) but *regresses* at
/// kv_len~8140 (median 376.8 vs 334.6 ms/token) — too few workgroups to
/// saturate the GPU once each thread's serial V-accumulation loop dominates.
/// Gate the fused path to context lengths where it's measured to win;
/// longer contexts fall back to the Burn-matmul path (chunked via
/// `PV_KV_CHUNK`, which scales better here). A tiled/two-pass kernel with
/// more workgroups per head would likely fix this at long context but is
/// unimplemented this session — see docs/BENCHMARKS.md Session 14 "what's
/// left".
const FUSED_DECODE_ATTN_MAX_KV_LEN: usize = 4096;

/// Session 14 (docs/BENCHMARKS.md): decode-time (t==1) fused QK^T ->
/// softmax -> PV over a `KvDtype::F32` cache — the no-dequant counterpart
/// of `attn_decode_q8` above. `q`: `[1, n_heads, 1, head_dim]` (already
/// RoPE'd). Returns `[1, n_heads, 1, head_dim]`. Reuses the same
/// thread-local scratch buffer as `attn_decode_q8` (disjoint call sites —
/// exactly one of the two dtypes is active per `KvCache`, and the scratch
/// buffer is generic float storage with no dtype-specific layout).
#[allow(clippy::too_many_arguments)]
fn attn_decode_f32(
    q: &Tensor<Wgpu, 4>,
    cache: &KvCache,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    thread_local! {
        static ATTN_SCRATCH_F32: std::cell::RefCell<Option<(cubecl::server::Handle, usize)>> =
            const { std::cell::RefCell::new(None) };
    }

    let q_cube: CubeTensor<WgpuRuntime> = into_contiguous(q.clone().into_primitive().tensor());
    let client = q_cube.client.clone();
    let device = q_cube.device.clone();
    let max_ctx = cache.max_ctx();
    let (k_cache, v_cache) = cache.f32_layer(layer_idx);

    let needed = n_heads * max_ctx;
    let scratch = ATTN_SCRATCH_F32.with(|cell| {
        let mut slot = cell.borrow_mut();
        let reuse = matches!(&*slot, Some((_, cap)) if *cap >= needed);
        if !reuse {
            *slot = Some((client.empty(needed * 4), needed));
        }
        slot.as_ref().unwrap().0.clone()
    });

    let out_handle = crate::gguf::attn_decode_f32_dispatch(
        &client, &q_cube.handle, &k_cache, &v_cache, &scratch, n_heads, n_kv_heads, head_dim, kv_len, max_ctx, scale,
    );
    let shape = burn::prelude::Shape::from(vec![1, n_heads, 1, head_dim]);
    let cube_tensor = CubeTensor::new_contiguous(client, device, shape, out_handle, DType::F32);
    Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
}

/// Prefill-time sparse attention through `wgsl/shader_attn_sparse.wgsl`:
/// QK^T -> softmax -> PV where each query reads only the keys its
/// [`SparseMask`] names. `q`: `[1, n_heads, T, head_dim]` (already RoPE'd).
/// Returns `[1, n_heads, T, head_dim]` — the same shape
/// `attention_with_mask` returns, so the caller's permute/reshape is
/// unchanged. GQA is handled inside the kernel (no `repeat_kv`), and K/V
/// are read straight out of the F32 cache (no `read_or_dequant_f32`).
#[allow(clippy::too_many_arguments)]
fn attn_sparse(
    q: &Tensor<Wgpu, 4>,
    cache: &KvCache,
    layer_idx: usize,
    sparse: &SparseMask,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    let q_cube: CubeTensor<WgpuRuntime> = into_contiguous(q.clone().into_primitive().tensor());
    let client = q_cube.client.clone();
    let device = q_cube.device.clone();
    let (k_cache, v_cache) = cache.f32_layer(layer_idx);
    let out_handle = crate::gguf::attn_sparse_dispatch(
        &client,
        &q_cube.handle,
        &k_cache,
        &v_cache,
        &sparse.meta,
        &sparse.keys,
        n_heads,
        n_kv_heads,
        head_dim,
        sparse.t,
        cache.max_ctx(),
        sparse.max_keys,
        scale,
    );
    let shape = burn::prelude::Shape::from(vec![1, n_heads, sparse.t, head_dim]);
    let cube_tensor = CubeTensor::new_contiguous(client, device, shape, out_handle, DType::F32);
    Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
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

/// QK^T -> caller-supplied mask -> softmax -> PV, chunked over the query
/// dimension only.
///
/// Unlike `attention_scores_and_values`, the key range is never truncated to
/// the causal prefix: a stencil mask deliberately lets a query attend to keys
/// *after* it (a CA cell's south and east neighbors come later in row-major
/// order), so every query chunk sees all `kv_len` keys and the mask alone
/// decides. `mask_out` is `[T, kv_len]`, `true` = masked out.
fn attention_with_mask(
    q: Tensor<Wgpu, 4>,
    k_all: Tensor<Wgpu, 4>,
    v_all: Tensor<Wgpu, 4>,
    t: usize,
    mask_out: &Tensor<Wgpu, 2, Bool>,
    scale: f32,
) -> Tensor<Wgpu, 4> {
    let kt = k_all.swap_dims(2, 3);
    let mut chunks = Vec::with_capacity(t.div_ceil(ATTN_QUERY_CHUNK));
    let mut start = 0usize;
    while start < t {
        let len = ATTN_QUERY_CHUNK.min(t - start);
        let scores = q.clone().narrow(2, start, len).matmul(kt.clone()) * scale;
        let m = mask_out.clone().narrow(0, start, len).unsqueeze::<4>();
        let probs = softmax(scores.mask_fill(m, f32::NEG_INFINITY), 3);
        chunks.push(pv_matmul(probs, v_all.clone()));
        start += len;
    }
    if chunks.len() == 1 {
        chunks.pop().unwrap()
    } else {
        Tensor::cat(chunks, 2)
    }
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
        let dev = x.device();
        let gate = crate::profile::scope("ffn.gate_up", &dev, || {
            let gate = self.gate_proj.forward(x.clone());
            let up = self.up_proj.forward(x.clone());
            (gate, up)
        });
        let fused = crate::profile::scope("ffn.silu_mul", &dev, || {
            crate::gguf::silu_mul_fused(gate.0, gate.1)
        });
        crate::profile::scope("ffn.down", &dev, || self.down_proj.forward(fused))
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
        spec: &ForwardSpec,
    ) -> Tensor<Wgpu, 3> {
        let dev = x.device();
        let normed = crate::profile::scope("norm", &dev, || self.attention_norm.forward(x.clone()));
        let attn_out = self.attention.forward(normed, rope, cache, layer_idx, offset, spec);
        let x = x + attn_out;
        let normed = crate::profile::scope("norm", &dev, || self.ffn_norm.forward(x.clone()));
        let ffn_out = crate::profile::scope("ffn", &dev, || self.ffn.forward(normed));
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
        self.forward_hidden_spec(token_ids, cache, &ForwardSpec::default())
    }

    /// `forward_hidden` with caller-supplied RoPE positions and/or attention
    /// mask — see [`ForwardSpec`]. With the default spec this is
    /// bit-identical to `forward_hidden` (it is the same code path).
    pub fn forward_hidden_spec(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
        spec: &ForwardSpec,
    ) -> Result<Tensor<Wgpu, 3>> {
        if let Some(positions) = &spec.positions {
            assert_eq!(
                positions.len(),
                token_ids.len(),
                "ForwardSpec positions must be one per token"
            );
        }
        let offset = cache.len();
        let mut x = crate::profile::scope("embed", &self.device, || self.embed_tokens(token_ids))?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, &self.rope, cache, i, offset, spec);
        }
        cache.advance(token_ids.len());
        Ok(crate::profile::scope("out_norm", &self.device, || {
            self.out_norm.forward(x)
        }))
    }

    /// lm_head over hidden states `[1, T, hidden]` -> logits `[1, T, vocab]`.
    /// Callers should narrow `hidden` to only the positions they need first
    /// (the 151936-wide head is 0.6MB/row of f32 output).
    pub fn lm_head(&self, hidden: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        self.lm_head.forward(hidden)
    }

    /// Dequantized `[K, hidden]` rows of the tied embedding matrix — the
    /// weight of a **sliced lm-head** over just `token_ids`.
    ///
    /// `llm-life` (CONCEPT.md §1) reads p(alive) from two logits (`0`/`1`) at
    /// every one of thousands of positions; the full head would materialize
    /// `T x 151936` floats to read two columns. The rows come from the same
    /// Q4 buffer `lm_head` is tied to (docs/MODELS.md §2: no independent
    /// `output.weight` exists), so this is the same projection, sliced.
    /// Build it once per generation and reuse it across positions.
    pub fn head_slice(&self, token_ids: &[u32]) -> Result<Tensor<Wgpu, 2>> {
        let hidden = self.config.hidden_size;
        let mut data = vec![0.0f32; token_ids.len() * hidden];
        for (i, &id) in token_ids.iter().enumerate() {
            self.embed
                .embed_id_add_cpu(id, &mut data[i * hidden..(i + 1) * hidden])?;
        }
        Ok(Tensor::<Wgpu, 1>::from_floats(data.as_slice(), &self.device)
            .reshape([token_ids.len(), hidden]))
    }

    /// `hidden` `[1, T, hidden]` projected onto a `head_slice` `[K, hidden]`
    /// -> `[1, T, K]`, for **every** position (that is the point: all-position
    /// logits are affordable once the head is K-wide instead of 151936-wide).
    ///
    /// Broadcast-multiply-and-sum rather than `matmul`: at K=2 the output
    /// width is tiny while the contraction is `hidden`, which is the shape
    /// regime where this backend's matmul kernel was found to be silently
    /// wrong (see `PV_KV_CHUNK`'s doc comment). The elementwise path costs
    /// `T*K*hidden` floats of scratch, which at K=2 is cheaper than the
    /// full-width logits it replaces.
    pub fn lm_head_sliced(&self, hidden: Tensor<Wgpu, 3>, head: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 3> {
        let [_, t, d] = hidden.dims();
        let k = head.dims()[0];
        assert_eq!(head.dims()[1], d, "head slice width must be hidden_size");
        let h = hidden.reshape([1, t, 1, d]);
        let w = head.clone().reshape([1, 1, k, d]);
        (h * w).sum_dim(3).reshape([1, t, k])
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

/// Async counterpart of [`logits_to_vec`]. **WASM callers must use this** —
/// `into_data()` deadlocks the browser (crate-level WASM constraints).
pub async fn logits_to_vec_async(logits: Tensor<Wgpu, 3>) -> Result<Vec<f32>> {
    logits
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("async logits readback failed: {e:?}"))?
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

#[cfg(all(test, feature = "wgpu"))]
mod bench_q8 {
    use super::*;
    use crate::kv::{KvCache, KvDtype};

    /// Session 13 (docs/BENCHMARKS.md): isolates the changed code path —
    /// one layer's decode-time (T=1) attention (cache write + QK^T/softmax/
    /// PV) — at a given `kv_len`, `KvDtype::F32` (old Burn-matmul path) vs
    /// `KvDtype::Q8_0` (new fused kernel). Everything else in a real
    /// decode step (q/k/v/o Q4 matmuls, RMSNorm, MLP) is unchanged by this
    /// session's work and not included; multiplying by `num_layers=36`
    /// gives an estimate of the KV-cache-attributable share of decode
    /// time, not a full end-to-end ms/token number (see `llm-agent bench`
    /// for that, native-only, not run here since this crate doesn't own
    /// that binary this session). Each iteration re-writes the same row
    /// (cache length held fixed at `kv_len`) rather than growing the
    /// cache, so the bench measures steady-state cost at that `kv_len`,
    /// not `max_ctx` accumulation. Run with: `cargo test --release
    /// --features wgpu --lib model::bench_q8::bench_decode_attention_layer
    /// -- --ignored --nocapture`
    fn bench_one(kv_len: usize, dtype: KvDtype, iters: usize) -> f64 {
        let device = WgpuDevice::default();
        let n_heads = 16;
        let n_kv_heads = 2;
        let head_dim = 128;
        let scale = (head_dim as f32).powf(-0.5);
        let max_ctx = kv_len + 8;

        let mut cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, dtype);
        let bulk_shape = [1, n_kv_heads, kv_len - 1, head_dim];
        let k_bulk = Tensor::<Wgpu, 4>::zeros(bulk_shape, &device);
        let v_bulk = Tensor::<Wgpu, 4>::zeros(bulk_shape, &device);
        cache.write(0, k_bulk, v_bulk);
        cache.advance(kv_len - 1);
        let offset = cache.len();

        let q = Tensor::<Wgpu, 4>::zeros([1, n_heads, 1, head_dim], &device);
        let k_new = Tensor::<Wgpu, 4>::zeros([1, n_kv_heads, 1, head_dim], &device);
        let v_new = Tensor::<Wgpu, 4>::zeros([1, n_kv_heads, 1, head_dim], &device);

        // Warm up (first dispatch of each kernel shape pays a one-time
        // pipeline-compilation cost — docs/BENCHMARKS.md Session 11).
        for _ in 0..3 {
            cache.write(0, k_new.clone(), v_new.clone());
            let kv_len_now = offset + 1;
            let out = if dtype == KvDtype::Q8_0 {
                attn_decode_q8(&q, &cache, 0, n_heads, n_kv_heads, head_dim, kv_len_now, scale)
            } else {
                let n_rep = n_heads / n_kv_heads;
                let (k_all, v_all) = cache.read_or_dequant_f32(0, kv_len_now);
                let k_all = repeat_kv(k_all, n_rep);
                let v_all = repeat_kv(v_all, n_rep);
                attention_scores_and_values(q.clone(), k_all, v_all, 1, kv_len_now, offset, scale)
            };
            let _ = out.into_data().into_vec::<f32>().unwrap();
        }

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            cache.write(0, k_new.clone(), v_new.clone());
            let kv_len_now = offset + 1;
            let out = if dtype == KvDtype::Q8_0 {
                attn_decode_q8(&q, &cache, 0, n_heads, n_kv_heads, head_dim, kv_len_now, scale)
            } else {
                let n_rep = n_heads / n_kv_heads;
                let (k_all, v_all) = cache.read_or_dequant_f32(0, kv_len_now);
                let k_all = repeat_kv(k_all, n_rep);
                let v_all = repeat_kv(v_all, n_rep);
                attention_scores_and_values(q.clone(), k_all, v_all, 1, kv_len_now, offset, scale)
            };
            let _ = out.into_data().into_vec::<f32>().unwrap();
        }
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    #[test]
    #[ignore]
    fn bench_decode_attention_layer() {
        let num_layers = 36;
        for &kv_len in &[2300usize, 8000] {
            let f32_ms = bench_one(kv_len, KvDtype::F32, 20);
            let q8_ms = bench_one(kv_len, KvDtype::Q8_0, 20);
            println!(
                "kv_len={kv_len}: F32={f32_ms:.4} ms/layer-step, Q8_0={q8_ms:.4} ms/layer-step \
                 (x{num_layers} layers, attention-only estimate: F32~{:.2}ms Q8_0~{:.2}ms)",
                f32_ms * num_layers as f64,
                q8_ms * num_layers as f64
            );
        }
    }
}
