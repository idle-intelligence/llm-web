//! The CPU sampler's repetition penalty against transformers'
//! `RepetitionPenaltyLogitsProcessor` (`lean-core/reference/fixture_rep_penalty.json`,
//! from `gen_fixture_rep_penalty.py`): real next-token logits of SmolLM2-360M
//! and Qwen2.5-0.5B after a prompt plus a greedy continuation, so the history
//! repeats ids. The processor penalizes each distinct id of the history once.
//! With temperature 1 and no top-k or top-p, `sample` leaves the penalized
//! logits in place; they must equal the reference bit for bit, and every id
//! outside the history must keep its logit.

use crate::sampling::{sample, Rng64, SamplingParams};

#[derive(serde::Deserialize)]
struct Case {
    model: String,
    penalty: f32,
    vocab: usize,
    input_ids: Vec<u32>,
    distinct: Vec<(u32, u32, u32)>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

/// Logits outside the history: any finite values, positive and negative.
fn filler(i: usize) -> f32 {
    ((i * 7919) % 2001) as f32 / 100.0 - 10.0
}

#[test]
fn cpu_penalty_matches_transformers() {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../lean-core/reference/fixture_rep_penalty.json")).unwrap();
    let fx: Fixture = serde_json::from_str(&s).unwrap();
    assert!(!fx.cases.is_empty());
    let mut failed = Vec::new();
    for case in &fx.cases {
        let mut logits: Vec<f32> = (0..case.vocab).map(filler).collect();
        let mut want = logits.clone();
        for &(id, raw, hf) in &case.distinct {
            logits[id as usize] = f32::from_bits(raw);
            want[id as usize] = f32::from_bits(hf);
        }
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: case.penalty, seed: 1 };
        sample(&mut logits, &params, &case.input_ids, &mut Rng64::new(1));
        let max_diff = logits.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let differing = logits.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("cpu {} penalty {}: max abs diff {max_diff:e}, {differing} logits differ", case.model, case.penalty);
        if differing != 0 {
            failed.push(format!("{} penalty {}", case.model, case.penalty));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
