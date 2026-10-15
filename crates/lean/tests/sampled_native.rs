//! Sampled decode through `Lean::generate` with the chat page's sampler
//! settings and a fixed seed: tokens per second for 128 and 256 new tokens,
//! and the reply written to `LEAN_SAMPLED_OUT` (when set) so its hash can be
//! compared with the browser bench's `sampled` phase. Also asserts that two
//! runs with the same seed give the same reply.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release --test sampled_native -- --ignored --nocapture
//! ```
//! `LEAN_BACKEND=cpu` runs the CPU backend.

use std::time::Instant;

use lean::{Backend, GenParams, GgufSource, Lean, LoadOptions, TokenizerSource};
use lean_chat::{read_chat_template, Message};

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn sampled_decode_speed_and_determinism() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR");
    let backend = if std::env::var("LEAN_BACKEND").as_deref() == Ok("cpu") { Backend::Cpu } else { Backend::WebGpu };
    let opts = LoadOptions { gguf: GgufSource::Path(gguf), tokenizer: TokenizerSource::Path(format!("{dir}/tokenizer.json")), max_ctx: 2048, backend };
    let mut lean = pollster::block_on(Lean::load(opts)).expect("loading lean");
    let template = read_chat_template(&format!("{dir}/tokenizer_config.json")).expect("chat template");
    let text = lean_chat::render(&template, &[Message::user("Write a long story about a lighthouse keeper.")], None, true).unwrap();
    let mut out = String::new();
    for n in [128u32, 256] {
        let params = GenParams { max_new_tokens: n, temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.1, seed: 0, mask_bits: None };
        let mut replies = Vec::new();
        for rep in 0..3 {
            let mut stamps: Vec<Instant> = Vec::new();
            let reply = pollster::block_on(lean.generate(&text, &params, |_| stamps.push(Instant::now()), None)).unwrap();
            let dt = stamps.last().unwrap().duration_since(stamps[0]).as_secs_f64();
            println!("sampled n={n} rep={rep} pieces={} tok/s={:.1}", stamps.len(), (stamps.len() - 1) as f64 / dt);
            replies.push(reply);
        }
        assert!(replies.iter().all(|r| r == &replies[0]), "same seed must give the same reply");
        out.push_str(&format!("=== {n}\n{}\n", replies[0]));
    }
    if let Ok(path) = std::env::var("LEAN_SAMPLED_OUT") {
        std::fs::write(path, out).unwrap();
    }
}
