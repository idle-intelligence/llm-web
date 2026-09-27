//! Per-step logit mask (constrained decoding) parity + timing.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release -- --ignored logit_mask
//! ```
//!
//! Checks: (1) a per-step singleton mask (only one allowed token id) forces
//! greedy decoding to reproduce an exact fixed string regardless of what the
//! model would have generated unconstrained - the mechanism a consumer's
//! grammar/schema loop drives (`build_mask_bitset` + `forward_prefill`'s and
//! `forward_decode_step_argmax`'s `mask` parameter); (2) an all-allowed mask
//! produces byte-identical output to no mask at all; (3) the added ms/step
//! of uploading + applying a mask vs an unmasked decode step.

use lean::chat_template::{read_chat_template, render_user_prompt};
use lean::engine::Engine;
use lean::model::{build_mask_bitset, build_rope_tables, forward_decode_step_argmax, forward_prefill, GpuModel, KvCache};
use std::time::Instant;
use tokenizers::Tokenizer;

fn tokenize(tokenizer: &Tokenizer, chat_template: &str, prompt: &str) -> Vec<u32> {
    let rendered = render_user_prompt(chat_template, prompt).unwrap();
    tokenizer.encode(rendered, false).unwrap().get_ids().to_vec()
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn singleton_mask_forces_exact_string() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let vocab = model.config.vocab_size;

    let prompt_ids = tokenize(&tokenizer, &chat_template, "Reply with a short JSON object.");
    // A target the model would not greedily produce unconstrained (checked
    // below by also running the unconstrained continuation and asserting it
    // differs) - the singleton mask must win regardless.
    let target = "{\"ok\":true}";
    let target_ids = tokenizer.encode(target, false).expect("tokenizer encode failed").get_ids().to_vec();
    assert!(!target_ids.is_empty());

    let max_ctx = prompt_ids.len() as u32 + target_ids.len() as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // Baseline: unconstrained continuation, to confirm the mask below is
    // actually doing something (not incidentally matching the model's own
    // top pick at every step).
    model.pool.reset();
    let mut cache_free = KvCache::new(&engine, &model.config, max_ctx);
    let logits_free = pollster::block_on(forward_prefill(&engine, &model, &mut cache_free, &prompt_ids, &cos_buf, &sin_buf, None));
    let mut unconstrained = Vec::with_capacity(target_ids.len());
    let mut next = argmax(&logits_free);
    for _ in 0..target_ids.len() {
        unconstrained.push(next);
        next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache_free, next, &cos_buf, &sin_buf, None));
    }
    assert_ne!(unconstrained, target_ids, "test fixture prompt happens to already produce the target string unconstrained - pick a different target");

    // Constrained: mask each step down to exactly one allowed token.
    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let mask0 = engine.buf_u32(&build_mask_bitset(vocab, &[target_ids[0]]), "mask0");
    let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &prompt_ids, &cos_buf, &sin_buf, Some(&mask0)));
    let mut got = Vec::with_capacity(target_ids.len());
    let mut next = argmax(&logits);
    got.push(next);
    for &want in &target_ids[1..] {
        let mask = engine.buf_u32(&build_mask_bitset(vocab, &[want]), "mask_step");
        next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, Some(&mask)));
        got.push(next);
    }
    assert_eq!(got, target_ids, "singleton mask at every step must force the exact target string");

    let text = tokenizer.decode(&got, true).expect("decode failed");
    eprintln!("[logit_mask] forced string = {text:?}");
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
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn all_allowed_mask_matches_unmasked() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let vocab = model.config.vocab_size;
    let n_steps = 16;

    let prompt_ids = tokenize(&tokenizer, &chat_template, "What is the capital of France?");
    let max_ctx = prompt_ids.len() as u32 + n_steps as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    model.pool.reset();
    let mut cache_unmasked = KvCache::new(&engine, &model.config, max_ctx);
    let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache_unmasked, &prompt_ids, &cos_buf, &sin_buf, None));
    let mut unmasked = Vec::with_capacity(n_steps);
    let mut next = argmax(&logits);
    for _ in 0..n_steps {
        unmasked.push(next);
        next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache_unmasked, next, &cos_buf, &sin_buf, None));
    }

    let all_allowed: Vec<u32> = (0..vocab as u32).collect();
    let all_mask_bits = build_mask_bitset(vocab, &all_allowed);

    model.pool.reset();
    let mut cache_masked = KvCache::new(&engine, &model.config, max_ctx);
    let mask0 = engine.buf_u32(&all_mask_bits, "mask_all_0");
    let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache_masked, &prompt_ids, &cos_buf, &sin_buf, Some(&mask0)));
    let mut masked = Vec::with_capacity(n_steps);
    let mut next = argmax(&logits);
    let mask_step = engine.buf_u32(&all_mask_bits, "mask_all_step");
    for _ in 0..n_steps {
        masked.push(next);
        next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache_masked, next, &cos_buf, &sin_buf, Some(&mask_step)));
    }

    assert_eq!(unmasked, masked, "an all-allowed mask must produce identical output to no mask");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn mask_upload_per_step_cost() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let vocab = model.config.vocab_size;
    let n_steps = 24;

    let prompt_ids = tokenize(&tokenizer, &chat_template, "Write one sentence about the ocean.");
    let max_ctx = prompt_ids.len() as u32 + n_steps as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let all_mask_bits = build_mask_bitset(vocab, &(0..vocab as u32).collect::<Vec<_>>());

    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let mut next = argmax(&pollster::block_on(forward_prefill(&engine, &model, &mut cache, &prompt_ids, &cos_buf, &sin_buf, None)));
    let unmasked_start = Instant::now();
    for _ in 0..n_steps {
        next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
    }
    let unmasked_ms_per_step = unmasked_start.elapsed().as_secs_f64() * 1000.0 / n_steps as f64;

    model.pool.reset();
    let mut cache2 = KvCache::new(&engine, &model.config, max_ctx);
    let mut next2 = argmax(&pollster::block_on(forward_prefill(&engine, &model, &mut cache2, &prompt_ids, &cos_buf, &sin_buf, None)));
    let masked_start = Instant::now();
    for _ in 0..n_steps {
        let mask = engine.buf_u32(&all_mask_bits, "mask_bench_step");
        next2 = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache2, next2, &cos_buf, &sin_buf, Some(&mask)));
    }
    let masked_ms_per_step = masked_start.elapsed().as_secs_f64() * 1000.0 / n_steps as f64;

    eprintln!(
        "[mask_upload_cost] vocab={vocab} unmasked_ms_per_step={unmasked_ms_per_step:.3} masked_ms_per_step={masked_ms_per_step:.3} overhead_ms={:.3}",
        masked_ms_per_step - unmasked_ms_per_step
    );
}
