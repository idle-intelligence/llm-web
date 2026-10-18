//! Full-model parity of one GGUF against a reference fixture produced by
//! transformers (`reference/gen_fixture_llama.py`: the GGUF dequantized on
//! load by transformers, prefill top-20 logits and a 32-token greedy
//! continuation for three chat prompts). Runs the model through the naive
//! reference kernels (`fast_kernels = false`) and through the fast f32
//! kernels (shader-f16 left ungranted), and requires every top-20 logit
//! within 1e-3 of transformers and the greedy continuation to be equal. The
//! f16 prefill path is checked separately by `f16_parity.rs` against the same
//! fixture. `fixture_parity_gguf_cpu` runs the same checks on the CPU path.
//!
//! Ignored by default (needs the GGUF and the tokenizer files, never committed):
//!
//! ```sh
//! LEAN_GGUF=/path/to/model.gguf LEAN_FIXTURE=fixture_qwen3_0_6b_q4_k_m.json \
//! LEAN_TOKENIZER_DIR=/path/to/hf-model-dir \
//! cargo test -p lean-core --release --test fixture_parity_gguf -- --ignored --nocapture
//! ```

use lean_chat::{read_chat_template, render, Message};
use lean_core::cpu::{forward_decode_step as cpu_decode_step, forward_prefill as cpu_prefill, CpuKvCache, CpuModel};
use lean_core::engine::Engine;
use lean_core::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};
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

fn run(engine: &Engine, gguf: &str, fixture: &Fixture, tokenizer: &Tokenizer, template: &lean_chat::ChatTemplate, fast: bool) {
    let model = GpuModel::load(engine, gguf, fast).expect("loading model");
    let mode = if fast { "fast" } else { "naive" };
    for case in &fixture.cases {
        let prompt = case.prompt.as_deref().expect("case without prompt");
        let rendered = render(template, &[Message::user(prompt)], None, true).unwrap();
        let ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
        assert_eq!(ids, case.input_ids, "[{mode} {}] tokenization mismatch", case.name);

        model.pool.reset();
        let max_ctx = case.input_ids.len() as u32 + case.greedy_continuation.len() as u32 + 4;
        let mut cache = KvCache::new(engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
        let cos_buf = engine.buf_f32(&cos, "rope_cos");
        let sin_buf = engine.buf_f32(&sin, "rope_sin");
        let mut logits = pollster::block_on(forward_prefill(engine, &model, &mut cache, &case.input_ids, &cos_buf, &sin_buf, None));

        assert_eq!(argmax(&logits), case.prefill_top20.ids[0], "[{mode} {}] prefill top1 mismatch", case.name);
        let mut worst = 0f32;
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (logits[id as usize] - case.prefill_top20.values[i]).abs();
            worst = worst.max(diff);
            assert!(diff < 1e-3, "[{mode} {}] top20 id {id}: diff {diff} >= 1e-3", case.name);
        }
        let mut got = Vec::new();
        for _ in 0..case.greedy_continuation.len() {
            let id = argmax(&logits);
            got.push(id);
            logits = pollster::block_on(forward_decode_step(engine, &model, &mut cache, id, &cos_buf, &sin_buf, None));
        }
        assert_eq!(got, case.greedy_continuation, "[{mode} {}] greedy continuation mismatch", case.name);
        eprintln!("[{mode} {}] seq {} max |top20 diff| {worst:.2e}, {} greedy tokens equal", case.name, case.input_ids.len(), got.len());
    }
}

#[test]
#[ignore = "needs LEAN_GGUF, LEAN_FIXTURE and LEAN_TOKENIZER_DIR; model files are never committed"]
fn fixture_parity_gguf() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let fixture = std::env::var("LEAN_FIXTURE").expect("set LEAN_FIXTURE");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let json = std::fs::read_to_string(format!("{}/reference/{fixture}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let fixture: Fixture = serde_json::from_str(&json).unwrap();

    std::env::set_var("LEAN_F16", "0");
    let engine = Engine::new().expect("wgpu engine init");
    run(&engine, &gguf, &fixture, &tokenizer, &template, false);
    run(&engine, &gguf, &fixture, &tokenizer, &template, true);
}

#[test]
#[ignore = "needs LEAN_GGUF, LEAN_FIXTURE and LEAN_TOKENIZER_DIR; model files are never committed"]
fn fixture_parity_gguf_cpu() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let fixture = std::env::var("LEAN_FIXTURE").expect("set LEAN_FIXTURE");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let json = std::fs::read_to_string(format!("{}/reference/{fixture}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let fixture: Fixture = serde_json::from_str(&json).unwrap();

    let model = CpuModel::load(&gguf).expect("loading model");
    for case in &fixture.cases {
        let prompt = case.prompt.as_deref().expect("case without prompt");
        let rendered = render(&template, &[Message::user(prompt)], None, true).unwrap();
        let ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
        assert_eq!(ids, case.input_ids, "[cpu {}] tokenization mismatch", case.name);

        let mut cache = CpuKvCache::new(&model.config, case.input_ids.len() + case.greedy_continuation.len() + 4);
        let mut logits = cpu_prefill(&model, &mut cache, &case.input_ids);
        assert_eq!(argmax(&logits), case.prefill_top20.ids[0], "[cpu {}] prefill top1 mismatch", case.name);
        let mut worst = 0f32;
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (logits[id as usize] - case.prefill_top20.values[i]).abs();
            worst = worst.max(diff);
            assert!(diff < 1e-3, "[cpu {}] top20 id {id}: diff {diff} >= 1e-3", case.name);
        }
        let mut got = Vec::new();
        for _ in 0..case.greedy_continuation.len() {
            let id = argmax(&logits);
            got.push(id);
            logits = cpu_decode_step(&model, &mut cache, id);
        }
        assert_eq!(got, case.greedy_continuation, "[cpu {}] greedy continuation mismatch", case.name);
        eprintln!("[cpu {}] seq {} max |top20 diff| {worst:.2e}, {} greedy tokens equal", case.name, case.input_ids.len(), got.len());
    }
}
