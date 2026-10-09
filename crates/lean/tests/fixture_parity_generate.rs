//! Token-exact parity for the one-turn `generate` path specifically
//! (`chat::gpu_generate` / `chat::cpu_generate`), as opposed to
//! `fixture_parity.rs`/`fixture_parity_llama_360m.rs`, which drive the
//! lower-level `forward_prefill`/`forward_decode_step` primitives directly
//! and already prove those are token-exact against HF transformers'
//! greedy continuation (`crates/lean/reference/gen_fixture*.py`'s output,
//! loading the real GGUF via `gguf_file=`).
//!
//! `gpu_generate`/`cpu_generate` return text, not ids (the one-turn API's
//! contract is text-in/text-out), so this test's assertion is text-exact:
//! the reply each one produces for a fixture's `prompt`, wrapped as a
//! single user `Message` and run through `generate`'s full path (render
//! -> tokenize -> fresh `PrefixCache` prefill -> greedy decode loop ->
//! `TextStream`), must equal `tokenizer.decode(case.greedy_continuation,
//! skip_special=true)` - the same text the fixture's HF-reference token
//! ids decode to. Combined with `fixture_parity.rs`'s direct id-level
//! proof that the exact same `forward_prefill`/`forward_decode_step_argmax`
//! calls `gpu_generate`/`cpu_generate` wrap are themselves token-exact vs
//! HF, this closes the loop for the new message-level path end to end.
//!
//! Only the `prompt`-bearing cases run (`short`/`long`/`non_english`) -
//! the `no_retokenize` long-tools cases have no `prompt` to wrap as a
//! `Message` (this crate's chat template has no tools support, same
//! limitation `fixture_parity.rs` documents).
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release --test fixture_parity_generate -- --ignored qwen25_0_5b
//!
//! LEAN_GGUF_LLAMA_360M_Q4_0=/path/to/SmolLM2-360M-Instruct-Q4_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_360M=/path/to/SmolLM2-360M-Instruct \
//! cargo test -p lean --release --test fixture_parity_generate -- --ignored smollm2_360m
//! ```

use lean::chat::{cpu_generate, gpu_generate, GenParams, Message, PrefixCache};
use lean::chat_template::read_chat_template;
use lean::cpu::{CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::model::{build_rope_tables, GpuModel, KvCache};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    prompt: Option<String>,
    greedy_continuation: Vec<u32>,
    #[serde(default)]
    no_retokenize: bool,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn load_fixture(path: &str) -> Fixture {
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parsing {path}: {e}"))
}

/// Runs every `prompt`-bearing case in `fixture_path` through
/// `gpu_generate` (fresh `KvCache`/`PrefixCache` per case, greedy,
/// `max_new_tokens` = the fixture's own continuation length) and through
/// `cpu_generate`, and checks both replies equal the HF-reference
/// continuation's decoded text.
fn check_generate_matches_fixture(gguf: &str, tokenizer_dir: &str, fixture_path: &str) {
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let fixture = load_fixture(fixture_path);

    let engine = Engine::new().expect("wgpu engine init");
    let gpu_model = GpuModel::load(&engine, gguf, true).expect("loading GPU model");
    let cpu_model = CpuModel::load(gguf).expect("loading CPU model");

    let mut checked = 0;
    for case in &fixture.cases {
        if case.no_retokenize {
            continue; // no `prompt` to wrap as a Message - see module doc.
        }
        let prompt = case.prompt.as_deref().expect("case has neither prompt nor no_retokenize");
        let expected = tokenizer.decode(&case.greedy_continuation, true).expect("decoding fixture's greedy_continuation");
        let messages = [Message::new("user", prompt)];
        let params = GenParams { max_new_tokens: case.greedy_continuation.len() as u32, ..GenParams::default() };
        let max_ctx = 256u32;

        gpu_model.pool.reset();
        let mut gpu_cache = KvCache::new(&engine, &gpu_model.config, max_ctx);
        let (cos, sin) = build_rope_tables(gpu_model.config.head_dim, gpu_model.config.rope_theta, max_ctx as usize);
        let cos_buf = engine.buf_f32(&cos, "rope_cos");
        let sin_buf = engine.buf_f32(&sin, "rope_sin");
        let mut gpu_prefix = PrefixCache::new();
        let mut gpu_streamed = String::new();
        let gpu_reply = pollster::block_on(gpu_generate(&engine, &gpu_model, &mut gpu_cache, &cos_buf, &sin_buf, &mut gpu_prefix, &tokenizer, &chat_template, &messages, &params, None, |t| gpu_streamed.push_str(t), || false)).unwrap();
        assert_eq!(gpu_streamed, gpu_reply, "[{}] on_token deltas must concatenate to the reply", case.name);
        assert_eq!(gpu_reply, expected, "[{}] gpu_generate reply != HF-reference continuation text", case.name);

        let mut cpu_cache = CpuKvCache::new(&cpu_model.config, max_ctx as usize);
        let mut cpu_prefix = PrefixCache::new();
        let cpu_reply = pollster::block_on(cpu_generate(&cpu_model, &mut cpu_cache, &mut cpu_prefix, &tokenizer, &chat_template, &messages, &params, |_| {}, || false, || async {})).unwrap();
        assert_eq!(cpu_reply, expected, "[{}] cpu_generate reply != HF-reference continuation text", case.name);

        println!("[{}] gpu/cpu generate() == HF reference: {:?}", case.name, expected);
        checked += 1;
    }
    assert!(checked > 0, "no prompt-bearing cases found in {fixture_path}");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn generate_matches_hf_fixture_qwen25_0_5b() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    check_generate_matches_fixture(&gguf, &tokenizer_dir, concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture.json"));
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_360M_Q4_0 and LEAN_TOKENIZER_DIR_LLAMA_360M on disk; never committed to this repo"]
fn generate_matches_hf_fixture_smollm2_360m() {
    let gguf = std::env::var("LEAN_GGUF_LLAMA_360M_Q4_0").expect("set LEAN_GGUF_LLAMA_360M_Q4_0 to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_LLAMA_360M").expect("set LEAN_TOKENIZER_DIR_LLAMA_360M to run this test");
    check_generate_matches_fixture(&gguf, &tokenizer_dir, concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_llama_360m_q4_0.json"));
}
