//! CPU-backend mirror of `fixture_parity_llama32.rs`, exercising
//! `cpu.rs::CpuModel` (single thread by default, rayon threads with
//! `--features threads`) against the same fixtures.
//!
//! The `past_8k` case (about 9k prompt tokens) runs only with
//! `LEAN_LLAMA32_CPU_LONG=1`: a 9k-token prefill is minutes of CPU time on
//! one thread, so it is meant for the threads build.
//!
//! Ignored by default because it needs the actual GGUF + tokenizer files
//! on disk, which are never committed to this repo. Run explicitly:
//!
//! ```sh
//! LEAN_GGUF_LLAMA32_1B_Q4_0=/path/to/Llama-3.2-1B-Instruct-Q4_0.gguf \
//! LEAN_GGUF_LLAMA32_1B_Q8_0=/path/to/Llama-3.2-1B-Instruct-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA32_1B=/path/to/Llama-3.2-1B-Instruct \
//! cargo test -p lean-core --release --features threads --test fixture_parity_llama32_cpu -- --ignored fixture_parity_llama32_1b_cpu
//! ```
//!
//! and the same with `3B`/`3b` for `fixture_parity_llama32_3b_cpu`.

use lean_core::cpu::{forward_decode_step, forward_prefill, CpuKvCache, CpuModel};
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
    rendered: String,
    input_ids: Vec<u32>,
    prefill_top20: Top20,
    greedy_continuation: Vec<u32>,
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

fn run_fixture(gguf_path: &str, fixture_json: &str, tokenizer: &Tokenizer, long: bool) {
    let fixture: Fixture = serde_json::from_str(fixture_json).expect("parsing fixture json");
    let model = CpuModel::load(gguf_path).expect("loading model");
    assert!(model.config.rope_freq_factors.is_some(), "[gguf={gguf_path}] no rope_freqs.weight read");

    for case in &fixture.cases {
        if case.name == "past_8k" && !long {
            eprintln!("[gguf={gguf_path} case={}] skipped (set LEAN_LLAMA32_CPU_LONG=1)", case.name);
            continue;
        }
        let our_ids: Vec<u32> = tokenizer.encode(case.rendered.as_str(), false).unwrap().get_ids().to_vec();
        assert_eq!(our_ids, case.input_ids, "[gguf={gguf_path} case={}] tokenization mismatch", case.name);

        let seq = case.input_ids.len();
        let max_ctx = seq + case.greedy_continuation.len() + 4;
        let mut cache = CpuKvCache::new(&model.config, max_ctx);

        let mut logits = forward_prefill(&model, &mut cache, &case.input_ids);

        let mut top20: Vec<(u32, f32)> = (0..logits.len() as u32).map(|i| (i, logits[i as usize])).collect();
        top20.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let maxdiff = case.prefill_top20.ids.iter().zip(&case.prefill_top20.values).map(|(&id, &v)| (logits[id as usize] - v).abs()).fold(0f32, f32::max);
        eprintln!("[gguf={gguf_path} case={} seq={seq}] top20_maxdiff={maxdiff:.2e}", case.name);
        assert_eq!(top20[0].0, case.prefill_top20.ids[0], "[gguf={gguf_path} case={}] prefill top1 mismatch", case.name);
        assert!(maxdiff < 1e-3, "[gguf={gguf_path} case={}] top20 maxdiff {maxdiff} >= 1e-3", case.name);

        let mut got_tokens = Vec::with_capacity(case.greedy_continuation.len());
        for _ in 0..case.greedy_continuation.len() {
            let next_id = argmax(&logits);
            got_tokens.push(next_id);
            logits = forward_decode_step(&model, &mut cache, next_id);
        }
        assert_eq!(got_tokens, case.greedy_continuation, "[gguf={gguf_path} case={}] greedy continuation mismatch", case.name);
    }
}

fn run_size(size: &str) {
    let upper = size.to_uppercase();
    let tokenizer_dir = std::env::var(format!("LEAN_TOKENIZER_DIR_LLAMA32_{upper}")).unwrap_or_else(|_| panic!("set LEAN_TOKENIZER_DIR_LLAMA32_{upper} to run this test"));
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let long = std::env::var("LEAN_LLAMA32_CPU_LONG").is_ok_and(|v| v == "1");
    for quant in ["q4_0", "q8_0"] {
        let var = format!("LEAN_GGUF_LLAMA32_{upper}_{}", quant.to_uppercase());
        let gguf = std::env::var(&var).unwrap_or_else(|_| panic!("set {var} to run this test"));
        let path = format!("{}/reference/fixture_llama32_{size}_{quant}.json", env!("CARGO_MANIFEST_DIR"));
        let fixture = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("reading {path}"));
        run_fixture(&gguf, &fixture, &tokenizer, long);
    }
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA32_1B_Q4_0/LEAN_GGUF_LLAMA32_1B_Q8_0 (GGUF paths) and LEAN_TOKENIZER_DIR_LLAMA32_1B (dir with tokenizer.json) on disk; never committed to this repo"]
fn fixture_parity_llama32_1b_cpu() {
    run_size("1b");
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA32_3B_Q4_0/LEAN_GGUF_LLAMA32_3B_Q8_0 (GGUF paths) and LEAN_TOKENIZER_DIR_LLAMA32_3B (dir with tokenizer.json) on disk; never committed to this repo"]
fn fixture_parity_llama32_3b_cpu() {
    run_size("3b");
}
