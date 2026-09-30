//! Native tests for this session's streaming/sampling/multi-turn additions
//! (`generate::decode_loop`, `sampling::{SamplingParams, sample}`,
//! `chat_template::render_conversation`) - the plain-Rust core that
//! `web.rs`'s `generateStream`/`chatGenerate` wrap for JS. Same fixture/env
//! convention as `kv_snapshot.rs`/`logit_mask.rs`: needs the actual GGUF +
//! tokenizer files on disk, never committed to this repo.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release -- --ignored streaming_sampling
//! ```

use lean::chat_template::{read_chat_template, render_conversation, render_user_prompt};
use lean::engine::Engine;
use lean::generate::decode_loop;
use lean::model::{argmax, build_rope_tables, forward_prefill, forward_prefill_suffix, GpuModel, KvCache};
use lean::sampling::SamplingParams;
use tokenizers::Tokenizer;

fn tokenize(tokenizer: &Tokenizer, chat_template: &str, prompt: &str) -> Vec<u32> {
    let rendered = render_user_prompt(chat_template, prompt).unwrap();
    tokenizer.encode(rendered, false).unwrap().get_ids().to_vec()
}

fn setup() -> (Engine, GpuModel, Tokenizer, String) {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    (engine, model, tokenizer, chat_template)
}

/// Runs the pre-existing whole-reply greedy path exactly as `LeanEngine::generate`
/// does natively (prefill -> CPU argmax -> loop of `forward_decode_step_argmax`),
/// for comparison against `decode_loop`'s greedy path.
fn whole_reply_greedy(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, max_new_tokens: u32) -> Vec<u32> {
    let logits = pollster::block_on(forward_prefill(engine, model, cache, token_ids, cos, sin, None));
    let mut next_id = argmax(&logits);
    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        if model.config.eos_token_ids.contains(&next_id) {
            break;
        }
        generated.push(next_id);
        next_id = pollster::block_on(lean::model::forward_decode_step_argmax(engine, model, cache, next_id, cos, sin, None));
    }
    generated
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn greedy_streaming_matches_whole_reply() {
    let (engine, model, tokenizer, chat_template) = setup();
    let prompt_ids = tokenize(&tokenizer, &chat_template, "What is the capital of France?");
    let max_ctx = prompt_ids.len() as u32 + 64;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    let mut cache_a = KvCache::new(&engine, &model.config, max_ctx);
    let expected = whole_reply_greedy(&engine, &model, &mut cache_a, &prompt_ids, &cos_buf, &sin_buf, 24);

    let mut cache_b = KvCache::new(&engine, &model.config, max_ctx);
    let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache_b, &prompt_ids, &cos_buf, &sin_buf, None));
    let mut streamed = Vec::new();
    let mut history = prompt_ids.clone();
    let params = SamplingParams::default();
    let got = pollster::block_on(decode_loop(&engine, &model, &mut cache_b, logits, &cos_buf, &sin_buf, None, 24, &params, &mut history, |id| streamed.push(id), || false));

    assert_eq!(got, expected, "decode_loop's greedy output must match the pre-existing whole-reply greedy path token for token");
    assert_eq!(streamed, expected, "on_token callback must fire with exactly the tokens returned");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn seeded_sampling_is_reproducible_across_runs() {
    let (engine, model, tokenizer, chat_template) = setup();
    let prompt_ids = tokenize(&tokenizer, &chat_template, "Tell me a short story about a fox.");
    let max_ctx = prompt_ids.len() as u32 + 64;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let params = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.1, seed: 12345 };

    let run = || {
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &prompt_ids, &cos_buf, &sin_buf, None));
        let mut history = prompt_ids.clone();
        pollster::block_on(decode_loop(&engine, &model, &mut cache, logits, &cos_buf, &sin_buf, None, 20, &params, &mut history, |_| {}, || false))
    };

    let out_a = run();
    let out_b = run();
    assert_eq!(out_a, out_b, "same seed + same params must reproduce the exact same generated token ids");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn multi_turn_append_matches_full_reprefill() {
    let (engine, model, tokenizer, chat_template) = setup();
    let p1 = "What is the capital of France?";
    let p2 = "And what is its population?";
    let max_ctx = 256u32;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // --- Append path: turn 1 (fresh prefill) -> greedy reply -> close turn
    // -> turn 2 appended onto the existing cache. ---
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let ids1 = tokenizer.encode(render_conversation(&chat_template, &[("user", p1)], true).unwrap(), false).unwrap().get_ids().to_vec();
    let logits1 = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids1, &cos_buf, &sin_buf, None));
    let mut history = ids1.clone();
    let params = SamplingParams::default();
    let gen1 = pollster::block_on(decode_loop(&engine, &model, &mut cache, logits1, &cos_buf, &sin_buf, None, 12, &params, &mut history, |_| {}, || false));
    let text1 = tokenizer.decode(&gen1, true).unwrap();

    let closed1 = tokenizer.encode(render_conversation(&chat_template, &[("user", p1), ("assistant", &text1)], false).unwrap(), false).unwrap().get_ids().to_vec();
    assert!(closed1.len() as u32 > cache.kv_len, "closing text must add at least the eos/newline tokens back onto the cache");
    let closing_suffix = &closed1[cache.kv_len as usize..];
    let _ = pollster::block_on(forward_prefill_suffix(&engine, &model, &mut cache, closing_suffix, &cos_buf, &sin_buf, None));

    let full_ids2 = tokenizer
        .encode(render_conversation(&chat_template, &[("user", p1), ("assistant", &text1), ("user", p2)], true).unwrap(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(full_ids2.len() as u32 > cache.kv_len, "turn 2 must add new tokens past the cached prefix");
    let new_suffix = &full_ids2[cache.kv_len as usize..];
    let logits_appended = pollster::block_on(forward_prefill_suffix(&engine, &model, &mut cache, new_suffix, &cos_buf, &sin_buf, None));

    // --- Reference path: the exact same final token sequence, prefilled
    // from scratch on a fresh cache in one batched call. ---
    let mut cache_full = KvCache::new(&engine, &model.config, max_ctx);
    let logits_full = pollster::block_on(forward_prefill(&engine, &model, &mut cache_full, &full_ids2, &cos_buf, &sin_buf, None));

    assert_eq!(logits_appended.len(), logits_full.len());
    let top_appended = argmax(&logits_appended);
    let top_full = argmax(&logits_full);
    assert_eq!(top_appended, top_full, "append path and full-reprefill path must agree on the argmax token");
    let mut max_diff = 0.0f32;
    for (a, b) in logits_appended.iter().zip(logits_full.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff < 1e-2, "append-path logits must match full-reprefill logits within tolerance, max_diff={max_diff}");
}
