//! CPU-rung parity test against `crates/lean/reference/fixture_llama_360m_{q4_0,q8_0}.json`
//! (SmolLM2-360M-Instruct, `general.architecture = llama`) - the CPU mirror
//! of `fixture_parity_llama_360m.rs`, exercising `cpu.rs::CpuModel` instead
//! of `model.rs::GpuModel`.
//!
//! This is the regression test for a bug where `cpu.rs::gguf_weight`
//! loaded `attn_q.weight`/`attn_k.weight` straight off disk for every
//! architecture, missing the RoPE row un-permutation
//! (`model.rs::unpermute_rope_rows`) that the GPU path already applies for
//! `Architecture::Llama` GGUFs (llama.cpp's `convert_hf_to_gguf.py`
//! permutes those two tensors' rows on Llama-family conversions so its own
//! kernels can consume them; this crate's split-half `rope_inplace`/
//! `rope_neox.wgsl` convention expects the un-permuted order). Qwen2/Qwen3
//! GGUFs never take this branch, which is why only Llama-architecture
//! models (SmolLM2-360M here) were affected - the CPU rung's greedy
//! continuation diverged from the reference almost immediately, with
//! `top20_maxdiff` in the single digits instead of the ~1e-5 quantization
//! noise every other model/rung combination shows.
//!
//! Ignored by default because it needs the actual GGUF + tokenizer files
//! on disk, which are never committed to this repo. Run explicitly:
//!
//! ```sh
//! LEAN_GGUF_LLAMA_360M_Q4_0=/path/to/SmolLM2-360M-Instruct-Q4_0.gguf \
//! LEAN_GGUF_LLAMA_360M_Q8_0=/path/to/SmolLM2-360M-Instruct-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_360M=/path/to/SmolLM2-360M-Instruct \
//! cargo test -p lean --release -- --ignored fixture_parity_llama_360m_cpu
//! ```

use lean::chat_template::{read_chat_template, render_user_prompt};
use lean::cpu::{forward_decode_step, forward_prefill, CpuKvCache, CpuModel};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Top20 {
    ids: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    prompt: Option<String>,
    input_ids: Vec<u32>,
    prefill_top20: Top20,
    greedy_continuation: Vec<u32>,
    #[serde(default)]
    no_retokenize: bool,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
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

fn run_fixture(gguf_path: &str, fixture_json: &str, tokenizer: &Tokenizer, chat_template: &str) {
    let fixture: Fixture = serde_json::from_str(fixture_json).expect("parsing fixture json");
    let model = CpuModel::load(gguf_path).expect("loading model");

    for case in &fixture.cases {
        if !case.no_retokenize {
            let prompt = case.prompt.as_deref().expect("case has neither prompt nor no_retokenize");
            let rendered = render_user_prompt(chat_template, prompt).unwrap();
            let our_ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
            assert_eq!(our_ids, case.input_ids, "[gguf={gguf_path} case={}] tokenization mismatch", case.name);
        }

        let seq = case.input_ids.len();
        let max_ctx = seq + case.greedy_continuation.len() + 4;
        let mut cache = CpuKvCache::new(&model.config, max_ctx);

        let mut logits = forward_prefill(&model, &mut cache, &case.input_ids);

        let mut top20: Vec<(u32, f32)> = (0..logits.len() as u32).map(|i| (i, logits[i as usize])).collect();
        top20.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(top20[0].0, case.prefill_top20.ids[0], "[gguf={gguf_path} case={}] prefill top1 mismatch", case.name);
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (logits[id as usize] - case.prefill_top20.values[i]).abs();
            assert!(diff < 1e-3, "[gguf={gguf_path} case={}] top20 id {id} diff {diff} >= 1e-3", case.name);
        }

        let mut got_tokens = Vec::with_capacity(case.greedy_continuation.len());
        for _ in 0..case.greedy_continuation.len() {
            let next_id = argmax(&logits);
            got_tokens.push(next_id);
            logits = forward_decode_step(&model, &mut cache, next_id);
        }
        assert_eq!(got_tokens, case.greedy_continuation, "[gguf={gguf_path} case={}] greedy continuation mismatch", case.name);
    }
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_360M_Q4_0/LEAN_GGUF_LLAMA_360M_Q8_0 (GGUF paths) and LEAN_TOKENIZER_DIR_LLAMA_360M (dir with tokenizer.json + tokenizer_config.json) on disk; never committed to this repo"]
fn fixture_parity_llama_360m_cpu() {
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_LLAMA_360M").expect("set LEAN_TOKENIZER_DIR_LLAMA_360M to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let q4_gguf = std::env::var("LEAN_GGUF_LLAMA_360M_Q4_0").expect("set LEAN_GGUF_LLAMA_360M_Q4_0 to run this test");
    let q4_fixture = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_llama_360m_q4_0.json")).expect("reading fixture_llama_360m_q4_0.json");
    run_fixture(&q4_gguf, &q4_fixture, &tokenizer, &chat_template);

    let q8_gguf = std::env::var("LEAN_GGUF_LLAMA_360M_Q8_0").expect("set LEAN_GGUF_LLAMA_360M_Q8_0 to run this test");
    let q8_fixture = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_llama_360m_q8_0.json")).expect("reading fixture_llama_360m_q8_0.json");
    run_fixture(&q8_gguf, &q8_fixture, &tokenizer, &chat_template);
}
