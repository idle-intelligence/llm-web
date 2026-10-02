//! One `KvCache` reused for several independent generations with no
//! `pool.reset()` in between (what the browser engine does since
//! `Pool::use_kv_cache`), longest prompt first so no pooled buffer regrows,
//! must give the same greedy tokens as each prompt on its own fresh cache.
//! Also switches back and forth between two caches on one pool, the case
//! `Pool::use_kv_cache` exists for.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release --test pool_reuse -- --ignored
//! ```

use lean::chat_template::{read_chat_template, render_user_prompt};
use lean::engine::Engine;
use lean::model::{argmax, build_rope_tables, forward_decode_step_argmax, forward_prefill, GpuModel, KvCache};
use tokenizers::Tokenizer;

const N_NEW: usize = 16;
const PROMPTS: [&str; 3] = [
    "Write two sentences about the history of the printing press in Europe.",
    "What is the capital of France?",
    "Name a prime number.",
];

fn greedy(engine: &Engine, model: &GpuModel, cache: &mut KvCache, ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer) -> Vec<u32> {
    cache.kv_len = 0;
    let logits = pollster::block_on(forward_prefill(engine, model, cache, ids, cos, sin, None));
    let mut next = argmax(&logits);
    let mut out = vec![next];
    for _ in 1..N_NEW {
        next = pollster::block_on(forward_decode_step_argmax(engine, model, cache, next, cos, sin, None));
        out.push(next);
    }
    out
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reused_cache_matches_fresh_caches() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");

    let prompts: Vec<Vec<u32>> = PROMPTS.iter().map(|p| tokenizer.encode(render_user_prompt(&chat_template, p).unwrap(), false).unwrap().get_ids().to_vec()).collect();
    let max_ctx = prompts.iter().map(|p| p.len()).max().unwrap() as u32 + N_NEW as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    let fresh: Vec<Vec<u32>> = prompts
        .iter()
        .map(|ids| {
            let mut cache = KvCache::new(&engine, &model.config, max_ctx);
            greedy(&engine, &model, &mut cache, ids, &cos_buf, &sin_buf)
        })
        .collect();

    let mut shared = KvCache::new(&engine, &model.config, max_ctx);
    for (ids, want) in prompts.iter().zip(&fresh) {
        assert_eq!(&greedy(&engine, &model, &mut shared, ids, &cos_buf, &sin_buf), want, "reused cache diverged");
    }

    let mut other = KvCache::new(&engine, &model.config, max_ctx);
    for (i, (ids, want)) in prompts.iter().zip(&fresh).enumerate().rev() {
        let cache = if i % 2 == 0 { &mut shared } else { &mut other };
        assert_eq!(&greedy(&engine, &model, cache, ids, &cos_buf, &sin_buf), want, "alternating caches diverged");
    }
}
