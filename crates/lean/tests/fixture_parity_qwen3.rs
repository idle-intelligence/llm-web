//! Full-model parity test against `crates/lean/reference/fixture_qwen3.json`
//! (Qwen3-0.6B, Q8_0 - see `reference/gen_fixture_qwen3.py`). Same shape as
//! `fixture_parity.rs`, minus the `fast_kernels` loop: Qwen3's official
//! GGUFs carry no Q4_0 tensors at all (Q8_0 only, per the qwen3 survey), so
//! `model::linear()`'s `fast_kernels` flag never changes which kernel runs
//! for this model's `MatMulWeight::Q8_0` weights - a single load covers
//! every code path this fixture exercises.
//!
//! Ignored by default because it needs the actual Q8_0 GGUF + tokenizer
//! files on disk, which are never committed to this repo. Run explicitly:
//!
//! ```sh
//! LEAN_GGUF_QWEN3=/path/to/Qwen3-0.6B-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_QWEN3=/path/to/Qwen3-0.6B \
//! cargo test -p lean --release -- --ignored fixture_parity_qwen3
//! ```

use lean::chat_template::{read_chat_template, render_user_prompt};
use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};
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

#[test]
#[ignore = "needs LEAN_GGUF_QWEN3 (Q8_0 GGUF path) and LEAN_TOKENIZER_DIR_QWEN3 (dir with tokenizer.json + tokenizer_config.json) on disk; never committed to this repo"]
fn fixture_parity_qwen3() {
    let gguf_path = std::env::var("LEAN_GGUF_QWEN3").expect("set LEAN_GGUF_QWEN3 to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_QWEN3").expect("set LEAN_TOKENIZER_DIR_QWEN3 to run this test");

    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let fixture_json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_qwen3.json")).expect("reading fixture_qwen3.json");
    let fixture: Fixture = serde_json::from_str(&fixture_json).expect("parsing fixture_qwen3.json");

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, false).expect("loading model");

    for case in &fixture.cases {
        if !case.no_retokenize {
            let prompt = case.prompt.as_deref().expect("case has neither prompt nor no_retokenize");
            let rendered = render_user_prompt(&chat_template, prompt).unwrap();
            let our_ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
            assert_eq!(our_ids, case.input_ids, "[case={}] tokenization mismatch", case.name);
        }

        model.pool.reset();
        let seq = case.input_ids.len() as u32;
        let max_ctx = seq + case.greedy_continuation.len() as u32 + 4;
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
        let cos_buf = engine.buf_f32(&cos, "rope_cos");
        let sin_buf = engine.buf_f32(&sin, "rope_sin");

        let mut logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &case.input_ids, &cos_buf, &sin_buf, None));

        let mut top20: Vec<(u32, f32)> = (0..logits.len() as u32).map(|i| (i, logits[i as usize])).collect();
        top20.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(top20[0].0, case.prefill_top20.ids[0], "[case={}] prefill top1 mismatch", case.name);
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (logits[id as usize] - case.prefill_top20.values[i]).abs();
            assert!(diff < 1e-3, "[case={}] top20 id {id} diff {diff} >= 1e-3", case.name);
        }

        let mut got_tokens = Vec::with_capacity(case.greedy_continuation.len());
        for _ in 0..case.greedy_continuation.len() {
            let next_id = argmax(&logits);
            got_tokens.push(next_id);
            logits = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, next_id, &cos_buf, &sin_buf, None));
        }
        assert_eq!(got_tokens, case.greedy_continuation, "[case={}] greedy continuation mismatch", case.name);
    }
}
