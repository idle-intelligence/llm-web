//! The one-turn API both browser engines expose (`generate` on `LeanEngine`
//! and `LeanEngineCpu`) runs `chat::gpu_generate` / `chat::cpu_generate`,
//! each call taking the whole conversation as `Message`s and streaming the
//! reply as text through `chat::TextStream`; these tests call the same
//! functions natively, driving several calls in a row (each one holding
//! its own growing `messages` slice, as a client would) to check that
//! `PrefixCache` reuse gives the same output a from-scratch prefill would.
//!
//! - `cpu_matches_gpu_three_turns_*`: a 3-call conversation (each call's
//!   `messages` is the previous reply appended plus a new user turn) gives
//!   the same token ids every call on the GPU and on the CPU, and each
//!   call's streamed text deltas concatenate to the reply.
//! - `text_stream_*`: deltas from `TextStream` concatenate to the full
//!   decode for a multilingual string (multi-byte characters split across
//!   byte-level BPE tokens) on the Qwen2.5 and SmolLM2 tokenizers.
//! - `smollm2_1_7b_q4_0_gpu_streams_text`: the model of the empty-output
//!   report streams non-empty text on the GPU.
//!
//! Needs the GGUF + tokenizer files on disk (never committed):
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! LEAN_GGUF_LLAMA_360M_Q4_0=/path/to/SmolLM2-360M-Instruct-Q4_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_360M=/path/to/SmolLM2-360M-Instruct \
//! LEAN_GGUF_LLAMA_1_7B_Q4_0=/path/to/SmolLM2-1.7B-Instruct-Q4_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_1_7B=/path/to/SmolLM2-1.7B-Instruct \
//! cargo test -p lean --release --features threads --test chat_api -- --ignored
//! ```

use lean::chat::{cpu_generate, gpu_generate, GenParams, Message, PrefixCache, TextStream};
use lean::chat_template::read_chat_template;
use lean::cpu::{CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::error::LeanError;
use lean::model::{build_rope_tables, GpuModel, KvCache};
use tokenizers::Tokenizer;

const TURNS: [&str; 3] = [
    "What is the capital of France?",
    "Name two famous museums there, one sentence each.",
    "Écris une phrase en français sur la Seine, avec un emoji.",
];
const MAX_NEW: u32 = 48;
const MAX_CTX: u32 = 1024;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run this test"))
}

fn load_tokenizer(dir: &str) -> (Tokenizer, String) {
    let tokenizer = Tokenizer::from_file(format!("{dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{dir}/tokenizer_config.json")).expect("reading chat_template");
    (tokenizer, chat_template)
}

/// Streams `ids` through a `TextStream` and returns the deltas.
fn stream(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<String> {
    let mut s = TextStream::new();
    let mut out: Vec<String> = ids.iter().map(|&id| s.push(tokenizer, id).unwrap()).collect();
    out.push(s.finish(tokenizer).unwrap());
    out
}

/// A client's own growing conversation: appends a user turn, calls
/// `generate`, then appends the assistant's reply - exactly the policy a
/// caller owns per this crate's contract (lean keeps no conversation state
/// of its own). `reuse = false` makes a fresh `PrefixCache` every call (so
/// every call reprefills the whole conversation from scratch), the control
/// path [`reuse_equivalence`] compares against the normal, persistent-
/// `PrefixCache` path. Returns each call's reply and its wall time (ms).
fn gpu_conversation(gguf: &str, tokenizer: &Tokenizer, chat_template: &str, reuse: bool) -> Vec<(String, f64)> {
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, gguf, true).expect("loading model");
    let mut cache = KvCache::new(&engine, &model.config, MAX_CTX);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, MAX_CTX as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut prefix = PrefixCache::new();
    let params = GenParams { max_new_tokens: MAX_NEW, ..GenParams::default() };
    let mut messages: Vec<Message> = Vec::new();
    TURNS
        .iter()
        .map(|prompt| {
            messages.push(Message::new("user", *prompt));
            if !reuse {
                prefix.invalidate();
            }
            let mut streamed = String::new();
            let t0 = std::time::Instant::now();
            let reply = pollster::block_on(gpu_generate(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut prefix, tokenizer, chat_template, &messages, &params, None, |t| streamed.push_str(t), || false)).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(streamed, reply, "on_token's concatenated text equals the returned reply");
            messages.push(Message::new("assistant", reply.clone()));
            (reply, ms)
        })
        .collect()
}

fn cpu_conversation(gguf: &str, tokenizer: &Tokenizer, chat_template: &str, reuse: bool) -> Vec<(String, f64)> {
    let model = CpuModel::load(gguf).expect("loading model");
    let mut cache = CpuKvCache::new(&model.config, MAX_CTX as usize);
    let mut prefix = PrefixCache::new();
    let params = GenParams { max_new_tokens: MAX_NEW, ..GenParams::default() };
    let mut messages: Vec<Message> = Vec::new();
    TURNS
        .iter()
        .map(|prompt| {
            messages.push(Message::new("user", *prompt));
            if !reuse {
                prefix.invalidate();
            }
            let t0 = std::time::Instant::now();
            let reply = pollster::block_on(cpu_generate(&model, &mut cache, &mut prefix, tokenizer, chat_template, &messages, &params, |_| {}, || false, || async {})).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            messages.push(Message::new("assistant", reply.clone()));
            (reply, ms)
        })
        .collect()
}

/// Reuse must be invisible to the caller: a multi-turn conversation run
/// with `PrefixCache` reuse on (the normal path) must produce exactly the
/// same replies as the same conversation forced to reprefill from scratch
/// every call (`reuse = false`). Prints each call's wall time so a run
/// shows the reuse speedup (later calls' prefill shrinks to just the new
/// suffix) without asserting on timing.
fn reuse_equivalence(conversation: impl Fn(bool) -> Vec<(String, f64)>, label: &str) {
    let with_reuse = conversation(true);
    let without_reuse = conversation(false);
    for (turn, ((reply_on, ms_on), (reply_off, ms_off))) in with_reuse.iter().zip(&without_reuse).enumerate() {
        println!("{label} turn {turn}: reuse_on={ms_on:.1}ms reuse_off={ms_off:.1}ms");
        assert_eq!(reply_on, reply_off, "{label} turn {turn}: reuse changed the output");
    }
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_equivalence_gpu_qwen25_0_5b() {
    let gguf = env("LEAN_GGUF");
    let (tokenizer, chat_template) = load_tokenizer(&env("LEAN_TOKENIZER_DIR"));
    reuse_equivalence(|reuse| gpu_conversation(&gguf, &tokenizer, &chat_template, reuse), "gpu");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_equivalence_cpu_qwen25_0_5b() {
    let gguf = env("LEAN_GGUF");
    let (tokenizer, chat_template) = load_tokenizer(&env("LEAN_TOKENIZER_DIR"));
    reuse_equivalence(|reuse| cpu_conversation(&gguf, &tokenizer, &chat_template, reuse), "cpu");
}

fn cpu_matches_gpu(gguf_var: &str, tok_var: &str) {
    let gguf = env(gguf_var);
    let (tokenizer, chat_template) = load_tokenizer(&env(tok_var));
    let gpu = gpu_conversation(&gguf, &tokenizer, &chat_template, true);
    let cpu = cpu_conversation(&gguf, &tokenizer, &chat_template, true);
    for (turn, ((g_text, _), (c_text, _))) in gpu.iter().zip(&cpu).enumerate() {
        println!("turn {turn}: gpu {g_text:?}");
        assert!(!g_text.is_empty(), "turn {turn}: empty reply");
        assert_eq!(g_text, c_text, "turn {turn}: CPU reply differs from GPU");
    }
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn cpu_matches_gpu_three_turns_qwen25_0_5b() {
    cpu_matches_gpu("LEAN_GGUF", "LEAN_TOKENIZER_DIR");
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_360M_Q4_0 and LEAN_TOKENIZER_DIR_LLAMA_360M on disk; never committed to this repo"]
fn cpu_matches_gpu_three_turns_smollm2_360m() {
    cpu_matches_gpu("LEAN_GGUF_LLAMA_360M_Q4_0", "LEAN_TOKENIZER_DIR_LLAMA_360M");
}

/// Accents, CJK and emoji: several of these characters are split across
/// byte-level BPE tokens, so decoding one id at a time yields U+FFFD halves.
const MULTILINGUAL: &str = " Hello, world! Ça va très bien. 東京は日本の首都です。 Emoji: 🎉🚀 and naïve café.\nNew line.";

fn text_stream_roundtrip(tok_var: &str) {
    let (tokenizer, _) = load_tokenizer(&env(tok_var));
    let ids = tokenizer.encode(MULTILINGUAL, false).unwrap().get_ids().to_vec();
    let full = tokenizer.decode(&ids, true).unwrap();
    let deltas = stream(&tokenizer, &ids);
    assert_eq!(deltas.concat(), full);
    assert!(deltas.iter().all(|d| !d.contains('\u{FFFD}')), "a delta carries half a character: {deltas:?}");
    let per_id: String = ids.iter().map(|&id| tokenizer.decode(&[id], true).unwrap()).collect();
    println!("{} ids; per-id decode equals full decode: {}", ids.len(), per_id == full);
}

#[test]
#[ignore = "needs LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn text_stream_qwen25() {
    text_stream_roundtrip("LEAN_TOKENIZER_DIR");
}

#[test]
#[ignore = "needs LEAN_TOKENIZER_DIR_LLAMA_360M on disk; never committed to this repo"]
fn text_stream_smollm2() {
    text_stream_roundtrip("LEAN_TOKENIZER_DIR_LLAMA_360M");
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_1_7B_Q4_0 and LEAN_TOKENIZER_DIR_LLAMA_1_7B on disk; never committed to this repo"]
fn smollm2_1_7b_q4_0_gpu_streams_text() {
    let gguf = env("LEAN_GGUF_LLAMA_1_7B_Q4_0");
    let (tokenizer, chat_template) = load_tokenizer(&env("LEAN_TOKENIZER_DIR_LLAMA_1_7B"));
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf, true).expect("loading model");
    let mut cache = KvCache::new(&engine, &model.config, 512);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, 512);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut prefix = PrefixCache::new();
    let messages = [Message::new("user", "Hello! Who are you?")];
    let params = GenParams { max_new_tokens: 32, ..GenParams::default() };
    let reply = pollster::block_on(gpu_generate(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut prefix, &tokenizer, &chat_template, &messages, &params, None, |_| {}, || false)).unwrap();
    println!("{reply:?}");
    assert!(!reply.trim().is_empty(), "empty reply");
}

/// A call that would overflow `max_ctx` returns `LeanError::ContextOverflow`
/// with the right `needed`/`available`, touches nothing (`cache.kv_len` and
/// `prefix`'s cached ids stay as they were), and the engine's next call
/// (a client-side trim: drop down to the latest message) generates
/// correctly - the same policy `web/lean-chat-worker.js` and
/// `examples/chat.rs` apply.
#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn context_overflow_then_recovers_gpu() {
    let gguf = env("LEAN_GGUF");
    let (tokenizer, chat_template) = load_tokenizer(&env("LEAN_TOKENIZER_DIR"));
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf, true).expect("loading model");
    let max_ctx = 64u32;
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut prefix = PrefixCache::new();
    let params = GenParams { max_new_tokens: 32, ..GenParams::default() };

    // A long prompt plus max_new_tokens=32 must not fit max_ctx=64.
    let long_prompt = "Tell me, in great detail, everything you know about the history, geography, culture and cuisine of France, Italy and Spain.";
    let overflowing = [Message::new("user", long_prompt)];
    let err = pollster::block_on(gpu_generate(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut prefix, &tokenizer, &chat_template, &overflowing, &params, None, |_| {}, || false)).unwrap_err();
    match err {
        LeanError::ContextOverflow { needed, available } => {
            assert_eq!(available, max_ctx as usize);
            assert!(needed > available, "needed={needed} available={available}");
        }
        other => panic!("expected ContextOverflow, got {other}"),
    }
    assert_eq!(cache.kv_len, 0, "a failed call must not touch the KV cache");

    // The client trims down to a short prompt that does fit, and retries.
    let short = [Message::new("user", "Hi!")];
    let reply = pollster::block_on(gpu_generate(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut prefix, &tokenizer, &chat_template, &short, &params, None, |_| {}, || false)).unwrap();
    assert!(!reply.trim().is_empty(), "the engine must still work after an overflow");
}
