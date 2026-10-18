//! Gate for the f16 path. The F32 path is the HF-exact reference (the
//! `fixture_parity*` tests pin it to transformers with `LEAN_F16=0`); this
//! test compares the f16 path to it, in two parts:
//!
//! 1. the greedy tokens of the f16 path are identical to the F32 path and to
//!    the fixture, and the prefill top-1 is the fixture's;
//! 2. at every generated position (the prefill position included), the token
//!    chosen by each path is in the other path's top-5 (the logprob-closeness
//!    check vLLM uses between implementations).
//!
//! Prints the largest |logit| difference of the last prefill position per
//! case. Skips with a message when the adapter has no `shader-f16`.
//!
//! ```sh
//! LEAN_GGUF=/path/to/model.gguf LEAN_FIXTURE=fixture.json \
//! cargo test -p lean-core --release --test f16_parity -- --ignored --nocapture
//! ```
//!
//! `LEAN_FIXTURE` names a file in `reference/` (default `fixture.json`, which
//! includes the two long tool-calling prompts). Cases longer than
//! `LEAN_F16_MAX_TOKENS` (default 3000) are skipped.

use lean_core::engine::Engine;
use lean_core::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};
use serde::Deserialize;

#[derive(Deserialize)]
struct Top20 {
    ids: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
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

const TOP_K: usize = 5;

fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.select_nth_unstable_by(k - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    ids.truncate(k);
    ids
}

struct Out {
    logits: Vec<f32>,
    tokens: Vec<u32>,
    /// Top-5 token ids of the logits each generated token was chosen from.
    tops: Vec<Vec<u32>>,
}

fn run(cases: &[Case], gguf: &str, max_tokens: usize) -> (bool, Vec<Option<Out>>) {
    let engine = Engine::new().expect("wgpu engine init");
    eprintln!("adapter {}", engine.adapter_report);
    let f16 = engine.adapter_report.split("\"deviceFeatures\":").nth(1).is_some_and(|r| r.split("\"maxStorage").next().unwrap_or("").contains("SHADER_F16"));
    let model = GpuModel::load(&engine, gguf, true).expect("loading model");
    let mut outs = Vec::new();
    for case in cases {
        if case.input_ids.len() > max_tokens {
            outs.push(None);
            continue;
        }
        model.pool.reset();
        let max_ctx = case.input_ids.len() as u32 + case.greedy_continuation.len() as u32 + 4;
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
        let cos_buf = engine.buf_f32(&cos, "rope_cos");
        let sin_buf = engine.buf_f32(&sin, "rope_sin");
        let mut logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &case.input_ids, &cos_buf, &sin_buf, None));
        let first = logits.clone();
        let mut tokens = Vec::new();
        let mut tops = Vec::new();
        for _ in 0..case.greedy_continuation.len() {
            let id = argmax(&logits);
            tops.push(top_k(&logits, TOP_K));
            tokens.push(id);
            logits = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, id, &cos_buf, &sin_buf, None));
        }
        outs.push(Some(Out { logits: first, tokens, tops }));
    }
    (f16, outs)
}

#[test]
#[ignore = "needs LEAN_GGUF (LEAN_FIXTURE optional); model files are never committed"]
fn f16_prefill_matches_f32() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let max_tokens: usize = std::env::var("LEAN_F16_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let fixture = std::env::var("LEAN_FIXTURE").unwrap_or_else(|_| "fixture.json".into());
    let json = std::fs::read_to_string(format!("{}/reference/{fixture}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let cases = serde_json::from_str::<Fixture>(&json).unwrap().cases;

    std::env::set_var("LEAN_F16", "0");
    let (f16_off, f32_out) = run(&cases, &gguf, max_tokens);
    assert!(!f16_off);
    std::env::remove_var("LEAN_F16");
    let (f16_on, f16_out) = run(&cases, &gguf, max_tokens);
    if !f16_on {
        eprintln!("adapter has no shader-f16: nothing to compare");
        return;
    }

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let (Some(a), Some(b)) = (&f32_out[i], &f16_out[i]) else { continue };
        let vs_f32 = a.logits.iter().zip(&b.logits).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let hf = |o: &Out| case.prefill_top20.ids.iter().zip(&case.prefill_top20.values).map(|(&id, &v)| (o.logits[id as usize] - v).abs()).fold(0.0f32, f32::max);
        let top1_ok = argmax(&b.logits) == case.prefill_top20.ids[0];
        let diverge = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
        let hf_diverge = case.greedy_continuation.iter().zip(&b.tokens).position(|(x, y)| x != y);
        eprintln!(
            "case {} ({} tokens): max|logit diff| f16 vs f32 {vs_f32:.5}, top20 vs HF f32 {:.5} f16 {:.5}, top1 ok {top1_ok}, greedy first divergence f16 vs f32 {diverge:?}, f16 vs HF {hf_diverge:?}",
            case.name,
            case.input_ids.len(),
            hf(a),
            hf(b)
        );
        // Each side's chosen token must be in the other side's top-5 at every position.
        let topk_miss = (0..a.tokens.len()).find(|&p| !b.tops[p].contains(&a.tokens[p]) || !a.tops[p].contains(&b.tokens[p]));
        eprintln!("  top-{TOP_K} cross-membership first miss: {topk_miss:?}");
        if !top1_ok || diverge.is_some() || hf_diverge.is_some() || topk_miss.is_some() {
            failures.push(case.name.clone());
        }
    }
    assert!(failures.is_empty(), "token mismatch in cases {failures:?}");
}
