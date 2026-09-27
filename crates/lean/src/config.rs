//! Qwen2 architecture config, read from `qwen2.*` GGUF metadata keys. Same
//! key set as `llm-wasm/src/gguf.rs::config_from_gguf`; vocab size comes off
//! `token_embd.weight`'s tensor shape (padded vocab, not the tokenizer's
//! smaller live vocab — see llm-wasm's doc comment on this).

use anyhow::{ensure, Context, Result};
use std::io::{Read, Seek};

use crate::gguf::GgufReader;

#[derive(Debug, Clone)]
pub struct Qwen2Config {
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
}

pub fn config_from_gguf<R: Read + Seek>(reader: &GgufReader<R>) -> Result<Qwen2Config> {
    let arch = reader.meta_string("general.architecture").unwrap_or("");
    ensure!(arch == "qwen2", "expected general.architecture = 'qwen2', got '{arch}'");

    let num_layers = reader.meta_u32("qwen2.block_count").context("missing qwen2.block_count")? as usize;
    let hidden_size = reader.meta_u32("qwen2.embedding_length").context("missing qwen2.embedding_length")? as usize;
    let intermediate_size = reader.meta_u32("qwen2.feed_forward_length").context("missing qwen2.feed_forward_length")? as usize;
    let num_heads = reader.meta_u32("qwen2.attention.head_count").context("missing qwen2.attention.head_count")? as usize;
    let num_kv_heads = reader.meta_u32("qwen2.attention.head_count_kv").context("missing qwen2.attention.head_count_kv")? as usize;
    let rms_norm_eps = reader.meta_f32("qwen2.attention.layer_norm_rms_epsilon").context("missing qwen2.attention.layer_norm_rms_epsilon")?;
    let rope_theta = reader.meta_f32("qwen2.rope.freq_base").context("missing qwen2.rope.freq_base")?;
    let max_seq_len = reader.meta_u32("qwen2.context_length").context("missing qwen2.context_length")? as usize;

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
    ensure!(embd_shape[1] == hidden_size, "token_embd.weight hidden dim {} != qwen2.embedding_length {hidden_size}", embd_shape[1]);

    let head_dim = hidden_size / num_heads;

    Ok(Qwen2Config {
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
    })
}
