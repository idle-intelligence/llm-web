//! Full-model parity test against
//! `crates/lean/reference/fixture_qwen25_3b.json` (Qwen2.5-3B-Instruct,
//! official "q4_0" GGUF - see `reference/gen_fixture_qwen25_3b.py`). This
//! GGUF's `output.weight` is Q6_K (`token_embd.weight` stays Q4_0) - see
//! `gguf.rs`'s `GgmlDtype` doc comment - the case this crate's `Q6_K`
//! support (`quant.rs::MatMulWeight::Q6_K`, `shaders/linear_q6k.wgsl`) was
//! added for: before that support existed, loading this file failed with
//! "unsupported GGML dtype code: 14".
//!
//! Ignored by default because it needs the actual GGUF + tokenizer files on
//! disk, which are never committed to this repo. Run explicitly:
//!
//! ```sh
//! LEAN_GGUF_QWEN25_3B=/path/to/qwen2.5-3b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR_QWEN25_3B=/path/to/Qwen2.5-3B-Instruct \
//! cargo test -p lean --release -- --ignored fixture_parity_qwen25_3b
//! ```
//!
//! Checks both kernel paths (`fast_kernels = false` and `true`, same as
//! `fixture_parity.rs`): our own tokenizer + chat-template path reproduces
//! the fixture's `input_ids`, prefill top-20 logits agree within 1e-3
//! (through the Q6_K lm head), and the 32-token greedy continuation is
//! identical - only Q6_K's naive kernel exists (see `linear_q6k.wgsl`'s doc
//! comment), so `fast_kernels` only changes the q/k/v/o/gate/up/down (Q4_0)
//! kernels here, not the lm head.

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
    prompt: String,
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

#[test]
#[ignore = "needs LEAN_GGUF_QWEN25_3B (GGUF path) and LEAN_TOKENIZER_DIR_QWEN25_3B (dir with tokenizer.json + tokenizer_config.json) on disk; never committed to this repo"]
fn fixture_parity_qwen25_3b() {
    let gguf_path = std::env::var("LEAN_GGUF_QWEN25_3B").expect("set LEAN_GGUF_QWEN25_3B to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_QWEN25_3B").expect("set LEAN_TOKENIZER_DIR_QWEN25_3B to run this test");

    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let fixture_json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_qwen25_3b.json")).expect("reading fixture_qwen25_3b.json");
    let fixture: Fixture = serde_json::from_str(&fixture_json).expect("parsing fixture_qwen25_3b.json");

    let engine = Engine::new().expect("wgpu engine init");

    for fast_kernels in [false, true] {
        let model = GpuModel::load(&engine, &gguf_path, fast_kernels).expect("loading model");
        for case in &fixture.cases {
            let rendered = render_user_prompt(&chat_template, &case.prompt).unwrap();
            let our_ids: Vec<u32> = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
            assert_eq!(our_ids, case.input_ids, "[fast={fast_kernels} case={}] tokenization mismatch", case.name);

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
            assert_eq!(top20[0].0, case.prefill_top20.ids[0], "[fast={fast_kernels} case={}] prefill top1 mismatch", case.name);
            for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
                let diff = (logits[id as usize] - case.prefill_top20.values[i]).abs();
                assert!(diff < 1e-3, "[fast={fast_kernels} case={}] top20 id {id} diff {diff} >= 1e-3", case.name);
            }

            let mut got_tokens = Vec::with_capacity(case.greedy_continuation.len());
            for _ in 0..case.greedy_continuation.len() {
                let next_id = argmax(&logits);
                got_tokens.push(next_id);
                logits = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, next_id, &cos_buf, &sin_buf, None));
            }
            assert_eq!(got_tokens, case.greedy_continuation, "[fast={fast_kernels} case={}] greedy continuation mismatch", case.name);
        }
    }
}
