//! A zero-valued runtime LoRA adapter must be a numerical no-op: this is
//! the "no adapter -> identical behaviour and identical outputs" guarantee
//! the runtime-LoRA task asked for, exercised end to end against a real
//! model (complementing `lora::tests`' pure-format unit tests, which never
//! touch the GPU).
//!
//! Env vars (default is relative to the repo root, same convention as
//! `tests/full_forward.rs`):
//! - `LLM_MODEL_DIR` -> GGUF directory, default
//!   `./models/gguf/xlam-2-3b-fc-r`
//! - `LLM_MODEL_FILE` -> GGUF filename within that directory, default
//!   `xLAM-2-3b-fc-r-q4_0.gguf`
//!
//! Skips (prints + returns) instead of failing when the GGUF isn't present.
#![cfg(feature = "wgpu")]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use burn::backend::wgpu::WgpuDevice;
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::lora::LoraAdapter;
use llm_wasm::model::logits_to_vec;

fn model_path() -> String {
    let dir = std::env::var("LLM_MODEL_DIR")
        .unwrap_or_else(|_| "./models/gguf/xlam-2-3b-fc-r".to_string());
    let file = std::env::var("LLM_MODEL_FILE").unwrap_or_else(|_| "xLAM-2-3b-fc-r-q4_0.gguf".to_string());
    format!("{dir}/{file}")
}

/// A zero-valued LLMLIFE2 adapter for `num_layers` layers, with each
/// projection's `(in_features, out_features)` given explicitly (q/k/v/o
/// don't all share the same out_features under GQA — k/v project to
/// `num_kv_heads * head_dim`, not `hidden_size`).
fn zero_adapter_bytes(num_layers: usize, dims: [(usize, usize); 4], rank: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LLMLIFE2");
    out.extend_from_slice(&(rank as u32).to_le_bytes());
    out.extend_from_slice(&16.0f32.to_le_bytes());
    out.push(0); // mlp = false
    let n_tensors = (num_layers * dims.len() * 2) as u32;
    out.extend_from_slice(&n_tensors.to_le_bytes());
    for _ in 0..num_layers {
        for &(in_f, out_f) in &dims {
            for &(rows, cols) in &[(in_f, rank), (rank, out_f)] {
                out.extend_from_slice(&(rows as u32).to_le_bytes());
                out.extend_from_slice(&(cols as u32).to_le_bytes());
                out.extend(std::iter::repeat_n(0.0f32.to_le_bytes(), rows * cols).flatten());
            }
        }
    }
    out
}

#[test]
fn zero_adapter_is_a_no_op() {
    let path = model_path();
    if !Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }
    let device = WgpuDevice::default();
    let file = File::open(&path).expect("open gguf");
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader).expect("parse gguf header");
    let parts = loader.load_deferred(&device).expect("load_deferred");
    drop(loader);
    let mut model = parts.finalize(&device).expect("finalize");

    let tokens = vec![1u32, 2, 3, 4, 5];
    let mut cache = model.new_cache(64);
    let hidden = model.forward_hidden(&tokens, &mut cache).expect("forward (no adapter)");
    let before = logits_to_vec(hidden).unwrap();

    let cfg = model.config();
    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let head_dim = hidden_size / cfg.num_heads;
    let kv_dim = cfg.num_kv_heads * head_dim;
    let dims = [
        (hidden_size, hidden_size), // q
        (hidden_size, kv_dim),      // k
        (hidden_size, kv_dim),      // v
        (hidden_size, hidden_size), // o
    ];
    let bytes = zero_adapter_bytes(num_layers, dims, 8);
    let adapter = LoraAdapter::from_bytes(&bytes, num_layers, &device).expect("parse zero adapter");
    model.apply_lora(adapter).expect("apply zero adapter");

    let mut cache2 = model.new_cache(64);
    let hidden2 = model.forward_hidden(&tokens, &mut cache2).expect("forward (zero adapter)");
    let after = logits_to_vec(hidden2).unwrap();

    assert_eq!(before.len(), after.len());
    let max_diff = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "zero-valued adapter changed the forward output: max diff {max_diff}");
}
