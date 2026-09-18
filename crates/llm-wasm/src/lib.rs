//! xLAM-2-3b-fc-r (Qwen2 architecture) — browser-native tool-calling LLM.
//!
//! Decoder-only transformer (GQA + q/k/v bias, RoPE, RMSNorm, SwiGLU, tied
//! embeddings, 151936-token vocab) driving MCP tool calls in-browser, mirroring
//! the stt-wasm engine's Burn+wgpu architecture (see docs/ENGINE.md).

#[cfg(feature = "wgpu")]
pub mod gguf;

#[cfg(feature = "wgpu")]
pub mod model;

#[cfg(feature = "wgpu")]
pub mod kv;

#[cfg(feature = "wgpu")]
pub mod profile;

pub mod kvimg;

#[cfg(feature = "wgpu")]
pub mod sample;

pub mod grammar;
pub mod schemadiet;
pub mod template;
pub mod tools;
pub mod agent;
pub mod tokenizer;
pub mod eval;

#[cfg(feature = "web")]
pub mod web;

/// Model configuration for xLAM-2-3b-fc-r (Qwen2 architecture).
///
/// See `docs/MODELS.md` for the values verified against the GGUF header and
/// `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads (queries).
    pub num_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size (padded embedding-matrix size).
    pub vocab_size: usize,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// Maximum sequence length / context length.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// BOS token id.
    pub bos_token_id: u32,
    /// EOS token id(s) — generation stops on any of these.
    pub eos_token_ids: Vec<u32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        // Verified values from xLAM-2-3b-fc-r's config.json / GGUF header,
        // see docs/MODELS.md §1-2.
        Self {
            num_layers: 36,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 2,
            intermediate_size: 11008,
            vocab_size: 151936,
            rope_theta: 1_000_000.0,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            bos_token_id: 151643,
            eos_token_ids: vec![151645, 151643],
        }
    }
}
