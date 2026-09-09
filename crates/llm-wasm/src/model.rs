//! Qwen2 transformer model: embeddings, attention, RoPE, KV cache. Owned by phase 1b.
//!
//! This is a *type skeleton*, not a one-line stub: `crate::gguf` was copied
//! working from stt-wasm (see gguf.rs's header comment) and constructs these
//! types directly (`Q4Attention::new`, `SttModel::new`, ...), so their fields
//! and constructors are mirrored here **verbatim from stt-web's `model.rs`**
//! (only `SttConfig`/`SttModel` renamed to `LlmConfig`/`LlmModel`) purely to
//! keep `gguf.rs` compiling as copied. This is deliberately *not* yet
//! Qwen2-shaped: stt-web's `Q4Attention` has one combined `in_proj` (no q/k/v
//! bias) and a sliding window; `Q4FeedForward` is the 2-matrix STT variant, not
//! Qwen2/Llama's 3-matrix gate/up/down; `LlmModel` still carries a leftover
//! `audio_emb` field from STT's per-codebook embeddings. All of that — plus
//! `forward`/`forward_with_cache`/KV cache wiring, entirely absent here — is
//! real phase-1b work: rewrite `gguf.rs`'s tensor-name table AND these structs
//! together for Qwen2's `blk.N.attn_{q,k,v,output}` naming, q/k/v bias, 3-matrix
//! SwiGLU, tied lm_head, and no sliding window. See docs/ENGINE.md §1/§3/§4 and
//! docs/MODELS.md §1-2 for the target shapes.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use crate::gguf::{EmbeddingStore, Q4Linear};
use crate::LlmConfig;

// ---------------------------------------------------------------------------
// RoPE — Rotary Position Embeddings
// ---------------------------------------------------------------------------

/// Rotary Position Embeddings with precomputed cos/sin tables.
///
/// NOTE: stt-wasm's (unported) `apply()` used the interleaved/GPT-NeoX-pair
/// convention. Qwen2 (HF `rotate_half`) needs the contiguous-half-split
/// convention instead — see docs/ENGINE.md §3.
pub struct RoPE {
    #[allow(dead_code)]
    cos: Tensor<Wgpu, 2>,
    #[allow(dead_code)]
    sin: Tensor<Wgpu, 2>,
}

impl RoPE {
    /// Create RoPE with precomputed frequencies.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &WgpuDevice) -> Self {
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        let mut freqs = vec![0.0f32; max_seq_len * half_dim];
        for i in 0..max_seq_len {
            for j in 0..half_dim {
                freqs[i * half_dim + j] = positions[i] * inv_freq[j];
            }
        }

        let freqs = Tensor::<Wgpu, 1>::from_floats(freqs.as_slice(), device)
            .reshape([max_seq_len, half_dim]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        RoPE { cos, sin }
    }
}

// ---------------------------------------------------------------------------
// RmsNorm wrapper
// ---------------------------------------------------------------------------

/// RMSNorm layer wrapping burn::nn::RmsNorm for GGUF weight loading.
pub struct RmsNormLayer {
    pub inner: burn::nn::RmsNorm<Wgpu>,
}

impl RmsNormLayer {
    pub fn forward<const D: usize>(&self, x: Tensor<Wgpu, D>) -> Tensor<Wgpu, D> {
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Q4Attention
// ---------------------------------------------------------------------------

/// Multi-head attention with combined QKV projection (stt-web shape, see the
/// module doc comment above — needs reworking to separate q/k/v + bias).
pub struct Q4Attention {
    #[allow(dead_code)]
    in_proj: Q4Linear,
    #[allow(dead_code)]
    out_proj: Q4Linear,
    #[allow(dead_code)]
    n_heads: usize,
    #[allow(dead_code)]
    n_kv_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    scale: f32,
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl Q4Attention {
    pub fn new(
        in_proj: Q4Linear,
        out_proj: Q4Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        let dim = n_heads * head_dim;
        Self {
            in_proj,
            out_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window,
        }
    }
}

// ---------------------------------------------------------------------------
// Q4FeedForward
// ---------------------------------------------------------------------------

/// Gated MLP, 2-matrix STT form (see module doc comment — Qwen2/Llama need the
/// 3-matrix gate_proj/up_proj/down_proj form instead).
pub struct Q4FeedForward {
    #[allow(dead_code)]
    linear_in: Q4Linear,
    #[allow(dead_code)]
    linear_out: Q4Linear,
}

impl Q4FeedForward {
    pub fn new(linear_in: Q4Linear, linear_out: Q4Linear) -> Self {
        Self {
            linear_in,
            linear_out,
        }
    }
}

// ---------------------------------------------------------------------------
// Q4TransformerBlock
// ---------------------------------------------------------------------------

/// Pre-LN transformer block with Q4 weights.
pub struct Q4TransformerBlock {
    #[allow(dead_code)]
    attention_norm: RmsNormLayer,
    #[allow(dead_code)]
    attention: Q4Attention,
    #[allow(dead_code)]
    ffn_norm: RmsNormLayer,
    #[allow(dead_code)]
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
}

// ---------------------------------------------------------------------------
// LlmModel
// ---------------------------------------------------------------------------

/// The complete transformer model (still carrying stt-web's `audio_emb` field
/// — see module doc comment; xLAM-2-3b-fc-r has no audio embeddings, and its
/// `text_emb`/`text_emb_gpu` double as the tied lm_head instead of a separate
/// `text_linear`).
pub struct LlmModel {
    #[allow(dead_code)]
    audio_emb: Vec<EmbeddingStore>,
    #[allow(dead_code)]
    text_emb: EmbeddingStore,
    #[allow(dead_code)]
    text_emb_gpu: Tensor<Wgpu, 2>,
    #[allow(dead_code)]
    layers: Vec<Q4TransformerBlock>,
    #[allow(dead_code)]
    rope: RoPE,
    #[allow(dead_code)]
    out_norm: RmsNormLayer,
    #[allow(dead_code)]
    text_linear: Q4Linear,
    #[allow(dead_code)]
    config: LlmConfig,
    #[allow(dead_code)]
    device: WgpuDevice,
}

impl LlmModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        audio_emb: Vec<EmbeddingStore>,
        text_emb: EmbeddingStore,
        text_emb_gpu: Tensor<Wgpu, 2>,
        layers: Vec<Q4TransformerBlock>,
        rope: RoPE,
        out_norm: RmsNormLayer,
        text_linear: Q4Linear,
        config: LlmConfig,
        device: WgpuDevice,
    ) -> Self {
        Self {
            audio_emb,
            text_emb,
            text_emb_gpu,
            layers,
            rope,
            out_norm,
            text_linear,
            config,
            device,
        }
    }
}
