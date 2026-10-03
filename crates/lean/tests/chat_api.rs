//! The chat API both browser engines expose (`chatGenerate`/`chatReset` on
//! `LeanEngine` and `LeanEngineCpu`) runs `chat::gpu_chat_turn` /
//! `chat::cpu_chat_turn` and streams text through `chat::TextStream`; these
//! tests call the same functions natively.
//!
//! - `cpu_matches_gpu_three_turns_*`: one 3-turn greedy conversation on the
//!   GPU and on the CPU gives the same token ids every turn, and each turn's
//!   streamed text deltas concatenate to the reply.
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

use lean::chat::{cpu_chat_turn, gpu_chat_turn, ChatSession, TextStream};
use lean::chat_template::read_chat_template;
use lean::cpu::{CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::model::{build_rope_tables, GpuModel, KvCache};
use lean::sampling::SamplingParams;
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

fn gpu_conversation(gguf: &str, tokenizer: &Tokenizer, chat_template: &str) -> Vec<(Vec<u32>, String)> {
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, gguf, true).expect("loading model");
    let mut cache = KvCache::new(&engine, &model.config, MAX_CTX);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, MAX_CTX as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut session = ChatSession::new();
    let params = SamplingParams::default();
    TURNS
        .iter()
        .map(|prompt| {
            let mut streamed = Vec::new();
            let (ids, reply) = pollster::block_on(gpu_chat_turn(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut session, tokenizer, chat_template, prompt, MAX_NEW, &params, None, |id| streamed.push(id), || false)).unwrap();
            assert_eq!(streamed, ids, "on_token saw every generated id");
            (ids, reply)
        })
        .collect()
}

fn cpu_conversation(gguf: &str, tokenizer: &Tokenizer, chat_template: &str) -> Vec<(Vec<u32>, String)> {
    let model = CpuModel::load(gguf).expect("loading model");
    let mut cache = CpuKvCache::new(&model.config, MAX_CTX as usize);
    let mut session = ChatSession::new();
    let params = SamplingParams::default();
    TURNS
        .iter()
        .map(|prompt| pollster::block_on(cpu_chat_turn(&model, &mut cache, &mut session, tokenizer, chat_template, prompt, MAX_NEW, &params, None, |_| {}, || false, || async {})).unwrap())
        .collect()
}

fn cpu_matches_gpu(gguf_var: &str, tok_var: &str) {
    let gguf = env(gguf_var);
    let (tokenizer, chat_template) = load_tokenizer(&env(tok_var));
    let gpu = gpu_conversation(&gguf, &tokenizer, &chat_template);
    let cpu = cpu_conversation(&gguf, &tokenizer, &chat_template);
    for (turn, ((g_ids, g_text), (c_ids, c_text))) in gpu.iter().zip(&cpu).enumerate() {
        println!("turn {turn}: {} tokens, gpu {g_text:?}", g_ids.len());
        assert!(!g_ids.is_empty(), "turn {turn}: empty reply");
        assert_eq!(g_ids, c_ids, "turn {turn}: CPU tokens differ from GPU (cpu {c_text:?})");
        assert_eq!(g_text, c_text);
        assert_eq!(stream(&tokenizer, g_ids).concat(), *g_text, "turn {turn}: streamed deltas != reply");
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
    let mut session = ChatSession::new();
    let (ids, reply) = pollster::block_on(gpu_chat_turn(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut session, &tokenizer, &chat_template, "Hello! Who are you?", 32, &SamplingParams::default(), None, |_| {}, || false)).unwrap();
    println!("{} ids {ids:?}: {reply:?}", ids.len());
    assert!(!reply.trim().is_empty(), "empty reply for ids {ids:?}");
    assert_eq!(stream(&tokenizer, &ids).concat(), reply);
}
