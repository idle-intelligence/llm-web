//! Qwen2/Qwen3 architecture config, read from `qwen{2,3}.*` GGUF metadata
//! keys. Same key set as `llm-wasm/src/gguf.rs::config_from_gguf` for the
//! qwen2 prefix; the qwen3 prefix adds an explicit `attention.key_length`
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
//! embedding/head tensor).

use anyhow::{ensure, Context, Result};
use std::io::{Read, Seek};

use crate::gguf::GgufReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Qwen3,
}

impl Architecture {
    fn meta_prefix(self) -> &'static str {
        match self {
            Architecture::Qwen2 => "qwen2",
            Architecture::Qwen3 => "qwen3",
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
        other => anyhow::bail!("unsupported general.architecture '{other}' (this crate reads 'qwen2' or 'qwen3')"),
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

    let bos_token_id = reader.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(151643);
    let eos_token_id = reader.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(151645);
    let pad_token_id = reader.meta_u32("tokenizer.ggml.padding_token_id").unwrap_or(151643);
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
