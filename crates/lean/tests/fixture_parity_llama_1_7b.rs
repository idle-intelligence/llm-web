//! Full-model parity test against
//! `crates/lean/reference/fixture_llama_1_7b_q8_0.json` (SmolLM2-1.7B-Instruct,
//! `general.architecture = llama` - see `reference/gen_fixture_llama.py`).
//! Q8_0 only, unlike `fixture_parity_llama_360m.rs`: SmolLM2-1.7B-Instruct's
//! "Q4_0" GGUF carries a Q6_K `token_embd.weight` (verified against the
//! file - see `gguf.rs`'s `GgmlDtype` doc comment), a K-quant this crate
//! does not read, so that file fails to load at all (the same class of gap
//! already flagged for Qwen2.5-3B-Instruct's GGUF). Its Q8_0 GGUF is pure
//! Q8_0 (no K-quants), so it loads and is covered here.
//!
//! Ignored by default because it needs the actual GGUF + tokenizer files
//! on disk, which are never committed to this repo. Run explicitly:
//!
//! ```sh
//! LEAN_GGUF_LLAMA_1_7B_Q8_0=/path/to/SmolLM2-1.7B-Instruct-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_1_7B=/path/to/SmolLM2-1.7B-Instruct \
//! cargo test -p lean --release -- --ignored fixture_parity_llama_1_7b
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

fn run_fixture(engine: &Engine, gguf_path: &str, fixture_json: &str, tokenizer: &Tokenizer, chat_template: &str) {
    let fixture: Fixture = serde_json::from_str(fixture_json).expect("parsing fixture json");
    let model = GpuModel::load(engine, gguf_path, false).expect("loading model");

    for case in &fixture.cases {
        if !case.no_retokenize {
            let prompt = case.prompt.as_deref().expect("case has neither prompt nor no_retokenize");
            let rendered = render_user_prompt(chat_template, prompt).unwrap();
            let our_ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
            assert_eq!(our_ids, case.input_ids, "[gguf={gguf_path} case={}] tokenization mismatch", case.name);
        }

        model.pool.reset();
        let seq = case.input_ids.len() as u32;
        let max_ctx = seq + case.greedy_continuation.len() as u32 + 4;
        let mut cache = KvCache::new(engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
        let cos_buf = engine.buf_f32(&cos, "rope_cos");
        let sin_buf = engine.buf_f32(&sin, "rope_sin");

        let mut logits = pollster::block_on(forward_prefill(engine, &model, &mut cache, &case.input_ids, &cos_buf, &sin_buf, None));

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
            logits = pollster::block_on(forward_decode_step(engine, &model, &mut cache, next_id, &cos_buf, &sin_buf, None));
        }
        assert_eq!(got_tokens, case.greedy_continuation, "[gguf={gguf_path} case={}] greedy continuation mismatch", case.name);
    }
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_1_7B_Q8_0 (GGUF path) and LEAN_TOKENIZER_DIR_LLAMA_1_7B (dir with tokenizer.json + tokenizer_config.json) on disk; never committed to this repo"]
fn fixture_parity_llama_1_7b() {
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_LLAMA_1_7B").expect("set LEAN_TOKENIZER_DIR_LLAMA_1_7B to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let engine = Engine::new().expect("wgpu engine init");

    let q8_gguf = std::env::var("LEAN_GGUF_LLAMA_1_7B_Q8_0").expect("set LEAN_GGUF_LLAMA_1_7B_Q8_0 to run this test");
    let q8_fixture = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_llama_1_7b_q8_0.json")).expect("reading fixture_llama_1_7b_q8_0.json");
    run_fixture(&engine, &q8_gguf, &q8_fixture, &tokenizer, &chat_template);
}
