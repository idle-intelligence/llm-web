//! The engine's contract, through `Lean` on both backends:
//!
//! - `reuse_equivalence_*`: a three-turn conversation run on one engine
//!   (each call reuses the previous call's KV prefix) gives exactly the
//!   replies of the same conversation where every call gets a freshly
//!   loaded engine.
//! - `reuse_diverging_prefix_*`: a sequence of prompts that diverge from
//!   the cached prefix at different points (a different user turn, a
//!   different system message, the same prompt twice, back to the first)
//!   gives on one engine exactly what a fresh engine gives for each.
//! - `cpu_matches_gpu_three_turns_*`: the three-turn conversation gives the
//!   same replies on the GPU and on the CPU.
//! - `context_overflow_then_recovers_*`: a prompt that does not fit returns
//!   `ContextOverflow` with `needed` = its token count + `max_new_tokens`
//!   and `available` = `max_ctx`, and the next call returns what a fresh
//!   engine returns.
//! - `text_stream_*`: `TextStream` deltas concatenate to the full decode
//!   for a multilingual string (multi-byte characters split across
//!   byte-level BPE tokens) on the Qwen2.5 and SmolLM2 tokenizers.
//! - `smollm2_1_7b_q4_0_gpu_streams_text`: that model streams non-empty
//!   text on the GPU.
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
//! cargo test -p lean --release --features threads --test engine_api -- --ignored
//! ```

use lean::chat_template::{read_chat_template, render_conversation};
use lean::{Backend, GenParams, GgufSource, Lean, LeanError, LoadOptions, TextStream, TokenizerSource};
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

fn load(gguf: &str, tokenizer_dir: &str, max_ctx: u32, backend: Backend) -> Lean {
    let opts = LoadOptions { gguf: GgufSource::Path(gguf.into()), tokenizer: TokenizerSource::Path(format!("{tokenizer_dir}/tokenizer.json")), max_ctx, backend };
    pollster::block_on(Lean::load(opts)).expect("loading lean")
}

fn template(tokenizer_dir: &str) -> String {
    read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template")
}

fn render(template: &str, turns: &[(&str, &str)]) -> String {
    render_conversation(template, turns, true).expect("rendering")
}

fn params() -> GenParams {
    GenParams { max_new_tokens: MAX_NEW, ..GenParams::default() }
}

fn generate(lean: &mut Lean, text: &str) -> String {
    let mut streamed = String::new();
    let reply = pollster::block_on(lean.generate(text, &params(), |t| streamed.push_str(t), None)).unwrap();
    assert_eq!(streamed, reply, "on_token pieces must concatenate to the reply");
    reply
}

/// A client's growing conversation: each call sends the whole conversation
/// so far plus a new user turn. `reuse = false` loads a fresh engine for
/// every call, so nothing is reused.
fn conversation(gguf: &str, tokenizer_dir: &str, backend: Backend, reuse: bool) -> Vec<String> {
    let template = template(tokenizer_dir);
    let mut shared = reuse.then(|| load(gguf, tokenizer_dir, MAX_CTX, backend));
    let mut history: Vec<(String, String)> = Vec::new();
    let mut replies = Vec::new();
    for prompt in TURNS {
        history.push(("user".into(), prompt.into()));
        let turns: Vec<(&str, &str)> = history.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
        let text = render(&template, &turns);
        let t = std::time::Instant::now();
        let reply = match shared.as_mut() {
            Some(lean) => generate(lean, &text),
            None => generate(&mut load(gguf, tokenizer_dir, MAX_CTX, backend), &text),
        };
        println!("{backend:?} reuse={reuse} turn {}: {:.1} ms (includes load when reuse=false)", replies.len(), t.elapsed().as_secs_f64() * 1e3);
        history.push(("assistant".into(), reply.clone()));
        replies.push(reply);
    }
    replies
}

fn reuse_equivalence(backend: Backend) {
    let (gguf, dir) = (env("LEAN_GGUF"), env("LEAN_TOKENIZER_DIR"));
    let with_reuse = conversation(&gguf, &dir, backend, true);
    let fresh = conversation(&gguf, &dir, backend, false);
    for (turn, (a, b)) in with_reuse.iter().zip(&fresh).enumerate() {
        assert!(!a.is_empty(), "turn {turn}: empty reply");
        assert_eq!(a, b, "{backend:?} turn {turn}: reuse changed the output");
    }
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_equivalence_gpu_qwen25_0_5b() {
    reuse_equivalence(Backend::WebGpu);
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_equivalence_cpu_qwen25_0_5b() {
    reuse_equivalence(Backend::Cpu);
}

fn reuse_diverging_prefix(backend: Backend) {
    let (gguf, dir) = (env("LEAN_GGUF"), env("LEAN_TOKENIZER_DIR"));
    let template = template(&dir);
    let first = render(&template, &[("user", TURNS[0])]);
    let texts = [
        first.clone(),
        // Same template prefix, different user turn.
        render(&template, &[("user", TURNS[1])]),
        // Diverges a few tokens in, at the system message.
        render(&template, &[("system", "You answer in French."), ("user", TURNS[0])]),
        // Back to the first prompt, then the same prompt again (the cached
        // ids then cover the whole prompt).
        first.clone(),
        first,
    ];
    let mut lean = load(&gguf, &dir, MAX_CTX, backend);
    for (i, text) in texts.iter().enumerate() {
        let shared = generate(&mut lean, text);
        let fresh = generate(&mut load(&gguf, &dir, MAX_CTX, backend), text);
        assert!(!shared.is_empty(), "call {i}: empty reply");
        assert_eq!(shared, fresh, "{backend:?} call {i}: reuse changed the output");
        println!("{backend:?} call {i}: {shared:?}");
    }
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_diverging_prefix_gpu_qwen25_0_5b() {
    reuse_diverging_prefix(Backend::WebGpu);
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn reuse_diverging_prefix_cpu_qwen25_0_5b() {
    reuse_diverging_prefix(Backend::Cpu);
}

fn cpu_matches_gpu(gguf_var: &str, tok_var: &str) {
    let (gguf, dir) = (env(gguf_var), env(tok_var));
    let gpu = conversation(&gguf, &dir, Backend::WebGpu, true);
    let cpu = conversation(&gguf, &dir, Backend::Cpu, true);
    for (turn, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        println!("turn {turn}: gpu {g:?}");
        assert!(!g.is_empty(), "turn {turn}: empty reply");
        assert_eq!(g, c, "turn {turn}: CPU reply differs from GPU");
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

fn context_overflow_then_recovers(backend: Backend) {
    let (gguf, dir) = (env("LEAN_GGUF"), env("LEAN_TOKENIZER_DIR"));
    let template = template(&dir);
    let max_ctx = 64u32;
    let params = GenParams { max_new_tokens: 32, ..GenParams::default() };
    let mut lean = load(&gguf, &dir, max_ctx, backend);

    // A short call first, so the engine holds a cached prefix.
    let short = render(&template, &[("user", "Hi!")]);
    let before = pollster::block_on(lean.generate(&short, &params, |_| {}, None)).unwrap();

    let long = render(&template, &[("user", "Tell me, in great detail, everything you know about the history, geography, culture and cuisine of France, Italy and Spain.")]);
    let long_tokens = lean.token_count(&long).unwrap();
    match pollster::block_on(lean.generate(&long, &params, |_| panic!("an overflowing call must not stream"), None)) {
        Err(LeanError::ContextOverflow { needed, available }) => {
            assert_eq!(needed, long_tokens + 32, "needed = prompt tokens + max_new_tokens");
            assert_eq!(available, max_ctx as usize);
            println!("{backend:?}: ContextOverflow {{ needed: {needed}, available: {available} }}");
        }
        Err(other) => panic!("expected ContextOverflow, got {other}"),
        Ok(reply) => panic!("expected ContextOverflow, got a reply {reply:?}"),
    }

    let after = pollster::block_on(lean.generate(&short, &params, |_| {}, None)).unwrap();
    let fresh = pollster::block_on(load(&gguf, &dir, max_ctx, backend).generate(&short, &params, |_| {}, None)).unwrap();
    assert!(!after.trim().is_empty(), "the engine must still work after an overflow");
    assert_eq!(after, before, "the same call before and after an overflow must agree");
    assert_eq!(after, fresh, "after an overflow the engine must answer like a fresh one");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn context_overflow_then_recovers_gpu() {
    context_overflow_then_recovers(Backend::WebGpu);
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn context_overflow_then_recovers_cpu() {
    context_overflow_then_recovers(Backend::Cpu);
}

/// Accents, CJK and emoji: several of these characters are split across
/// byte-level BPE tokens, so decoding one id at a time yields U+FFFD halves.
const MULTILINGUAL: &str = " Hello, world! Ça va très bien. 東京は日本の首都です。 Emoji: 🎉🚀 and naïve café.\nNew line.";

fn text_stream_roundtrip(tok_var: &str) {
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", env(tok_var))).expect("loading tokenizer.json");
    let ids = tokenizer.encode(MULTILINGUAL, false).unwrap().get_ids().to_vec();
    let full = tokenizer.decode(&ids, true).unwrap();
    let mut s = TextStream::new();
    let mut deltas: Vec<String> = ids.iter().map(|&id| s.push(&tokenizer, id).unwrap()).collect();
    deltas.push(s.finish(&tokenizer).unwrap());
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
    let dir = env("LEAN_TOKENIZER_DIR_LLAMA_1_7B");
    let mut lean = load(&env("LEAN_GGUF_LLAMA_1_7B_Q4_0"), &dir, 512, Backend::WebGpu);
    let text = render(&template(&dir), &[("user", "Hello! Who are you?")]);
    let params = GenParams { max_new_tokens: 32, ..GenParams::default() };
    let reply = pollster::block_on(lean.generate(&text, &params, |_| {}, None)).unwrap();
    println!("{reply:?}");
    assert!(!reply.trim().is_empty(), "empty reply");
}
