//! Qwen2/Qwen3/Llama architecture config, read from `{qwen2,qwen3,llama}.*`
//! GGUF metadata keys. Same key set as `llm-wasm/src/gguf.rs::config_from_gguf`
//! for the qwen2 prefix; the qwen3 prefix adds an explicit `attention.key_length`
//! (head_dim is not derivable from `hidden_size / num_heads` for Qwen3 -
//! see the qwen3 survey) and per-head q/k RMSNorm weights (`attn_q_norm`/
//! `attn_k_norm`, read as tensors in `model.rs`, not metadata). Vocab size
//! comes off `token_embd.weight`'s tensor shape (padded vocab, not the
//! tokenizer's smaller live vocab - see llm-wasm's doc comment on this).
//!
//! Whether the lm head shares `token_embd.weight` (`tied_embeddings`) is
//! read from the GGUF's tensor index directly (`output.weight` present or
//! not), never assumed from `general.architecture`: Qwen2.5-0.5B's GGUF
//! carries a separate `output.weight` despite `tie_word_embeddings: true`
//! in its `config.json` (verified against the file - see `model.rs`'s doc
//! comment), while Qwen3-0.6B's GGUF carries no `output.weight` tensor at
//! all (verified against the file: `token_embd.weight` is the only
//! embedding/head tensor). SmolLM2's `llama`-architecture GGUFs also carry
//! no `output.weight` tensor (verified against both SmolLM2-360M-Instruct
//! and SmolLM2-1.7B-Instruct's GGUFs), matching their `config.json`'s
//! `tie_word_embeddings: true`.
//!
//! `llama.rope.dimension_count` is read explicitly for head_dim (present in
//! both SmolLM2 GGUFs, value 64 for both sizes, matching `hidden_size /
//! num_heads` in each case) rather than assumed, the same caution applied
//! to Qwen3's `attention.key_length`.
//!
//! RoPE convention for `llama`-architecture GGUFs: llama.cpp's
//! `convert_hf_to_gguf.py` permutes `attn_q.weight`/`attn_k.weight` row
//! order per head (`LlamaModel`'s `permute()`, splitting each head's
//! `head_dim` rows into two `head_dim/2` halves and interleaving them) so
//! that ggml's "normal" (interleaved-pair) RoPE kernel, applied to the
//! permuted weight, produces the same result as HF's own split-half
//! `rotate_half` convention applied to the unpermuted weight - this is
//! *not* done for `qwen2`/`qwen3` GGUFs (ggml applies its NEOX/split-half
//! RoPE kernel directly to those, no permutation). Verified empirically
//! against SmolLM2-360M-Instruct's Q8_0 GGUF: dequantizing
//! `blk.0.attn_q.weight` directly does not match
//! `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`'s own
//! `q_proj.weight` (max abs diff 6.6), but applying the inverse of that
//! permutation (`model.rs::unpermute_rope_rows`) matches it exactly (max
//! abs diff 0.0) - HF's own GGUF loader reverses this permutation on load,
//! so this crate must too to reuse the same split-half RoPE kernel Qwen2/
//! Qwen3 already use (`config.json`'s `rope_interleaved: false` for
//! SmolLM2-360M-Instruct independently confirms the split-half/rotate_half
//! convention on the HF side). `model.rs` applies `unpermute_rope_rows` to
//! `attn_q.weight`/`attn_k.weight` only when `architecture ==
//! Architecture::Llama`; Qwen2/Qwen3 loading is unchanged.

use anyhow::{ensure, Context, Result};
use std::io::{Read, Seek};

use crate::gguf::GgufReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Qwen3,
    Llama,
}

impl Architecture {
    fn meta_prefix(self) -> &'static str {
        match self {
            Architecture::Qwen2 => "qwen2",
            Architecture::Qwen3 => "qwen3",
            Architecture::Llama => "llama",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub architecture: Architecture,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    /// Qwen2's GGUF carries an attn q/k/v bias tensor per layer; Qwen3's
    /// `attention_bias: false` means none exists (`model.rs`'s per-layer
    /// load becomes a zero buffer instead of reading `attn_{q,k,v}.bias`).
    pub has_qkv_bias: bool,
    /// Qwen3-only: per-head RMSNorm on q and k (`attn_q_norm.weight`/
    /// `attn_k_norm.weight`, `[head_dim]` each) applied after projection,
    /// before RoPE. `false` for Qwen2 - no such tensors exist in its GGUF.
    pub qk_norm: bool,
    /// Whether the lm head reuses `token_embd.weight` instead of its own
    /// `output.weight` tensor - read off the GGUF's tensor index in
    /// `config_from_gguf` (see this module's doc comment), not derived from
    /// `architecture`.
    pub tied_embeddings: bool,
}

pub fn config_from_gguf<R: Read + Seek>(reader: &GgufReader<R>) -> Result<Qwen2Config> {
    let arch_str = reader.meta_string("general.architecture").unwrap_or("");
    let architecture = match arch_str {
        "qwen2" => Architecture::Qwen2,
        "qwen3" => Architecture::Qwen3,
        "llama" => Architecture::Llama,
        other => anyhow::bail!("unsupported general.architecture '{other}' (this crate reads 'qwen2', 'qwen3', or 'llama')"),
    };
    let p = architecture.meta_prefix();

    let num_layers = reader.meta_u32(&format!("{p}.block_count")).with_context(|| format!("missing {p}.block_count"))? as usize;
    let hidden_size = reader.meta_u32(&format!("{p}.embedding_length")).with_context(|| format!("missing {p}.embedding_length"))? as usize;
    let intermediate_size = reader.meta_u32(&format!("{p}.feed_forward_length")).with_context(|| format!("missing {p}.feed_forward_length"))? as usize;
    let num_heads = reader.meta_u32(&format!("{p}.attention.head_count")).with_context(|| format!("missing {p}.attention.head_count"))? as usize;
    let num_kv_heads = reader.meta_u32(&format!("{p}.attention.head_count_kv")).with_context(|| format!("missing {p}.attention.head_count_kv"))? as usize;
    let rms_norm_eps = reader.meta_f32(&format!("{p}.attention.layer_norm_rms_epsilon")).with_context(|| format!("missing {p}.attention.layer_norm_rms_epsilon"))?;
    let rope_theta = reader.meta_f32(&format!("{p}.rope.freq_base")).with_context(|| format!("missing {p}.rope.freq_base"))?;
    let max_seq_len = reader.meta_u32(&format!("{p}.context_length")).with_context(|| format!("missing {p}.context_length"))? as usize;

    // Qwen2/Qwen3's own special-token ids (151643/151645) are only a safe
    // default for those two architectures; a Llama-family GGUF (e.g.
    // SmolLM2) missing these keys gets a clear parse error instead of
    // silently inheriting Qwen's ids, which would be wrong for that vocab.
    let (default_bos, default_eos) = match architecture {
        Architecture::Qwen2 | Architecture::Qwen3 => (Some(151643), Some(151645)),
        Architecture::Llama => (None, None),
    };
    let bos_token_id = reader
        .meta_u32("tokenizer.ggml.bos_token_id")
        .or(default_bos)
        .with_context(|| format!("missing tokenizer.ggml.bos_token_id (no safe default for architecture '{arch_str}')"))?;
    let eos_token_id = reader
        .meta_u32("tokenizer.ggml.eos_token_id")
        .or(default_eos)
        .with_context(|| format!("missing tokenizer.ggml.eos_token_id (no safe default for architecture '{arch_str}')"))?;
    let pad_token_id = reader.meta_u32("tokenizer.ggml.padding_token_id").or(default_bos).unwrap_or(bos_token_id);
    let mut eos_token_ids = vec![eos_token_id];
    if pad_token_id != eos_token_id {
        eos_token_ids.push(pad_token_id);
    }

    let embd_info = reader.tensor_info("token_embd.weight").context("missing tensor 'token_embd.weight'")?;
    let embd_shape = embd_info.shape();
    ensure!(embd_shape.len() == 2, "expected 2D token_embd.weight, got {embd_shape:?}");
    let vocab_size = embd_shape[0];
    ensure!(embd_shape[1] == hidden_size, "token_embd.weight hidden dim {} != {p}.embedding_length {hidden_size}", embd_shape[1]);

    let (head_dim, qk_norm, has_qkv_bias) = match architecture {
        Architecture::Qwen2 => (hidden_size / num_heads, false, true),
        Architecture::Qwen3 => {
            let head_dim = reader.meta_u32(&format!("{p}.attention.key_length")).with_context(|| format!("missing {p}.attention.key_length"))? as usize;
            (head_dim, true, false)
        }
        Architecture::Llama => {
            let head_dim = reader.meta_u32(&format!("{p}.rope.dimension_count")).with_context(|| format!("missing {p}.rope.dimension_count"))? as usize;
            (head_dim, false, false)
        }
    };

    let tied_embeddings = reader.tensor_info("output.weight").is_none();

    Ok(Qwen2Config {
        architecture,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rope_theta,
        max_seq_len,
        rms_norm_eps,
        bos_token_id,
        eos_token_ids,
        has_qkv_bias,
        qk_norm,
        tied_embeddings,
    })
}
