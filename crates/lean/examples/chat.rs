//! A multi-turn terminal chat on lean's engine. The conversation lives
//! here, not in the engine: every turn renders the whole history with
//! lean-chat (the model's own chat template) and passes that text to
//! `Lean::generate`. On `LeanError::ContextOverflow` it drops the oldest
//! exchange with `lean_chat::trim_oldest_exchange` (the system message
//! stays) and retries, the same policy the web demo applies.
//!
//! ```sh
//! LEAN_GGUF=/path/to/model.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/tokenizer_dir \
//! cargo run -p lean --release --example chat -- [--backend auto|webgpu|cpu] [--max-ctx 2048] [--max-new-tokens 256] [--system "..."]
//! ```
//!
//! Reads one prompt per line from stdin, streams the reply to stdout, and
//! repeats until EOF (Ctrl-D) or a line that is exactly `/exit`.

use lean::{Backend, GenParams, GgufSource, Lean, LeanError, LoadOptions, TokenizerSource};
use lean_chat::{read_chat_template, render, trim_oldest_exchange, Message};
use std::io::Write;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn print_token(text: &str) {
    print!("{text}");
    let _ = std::io::stdout().flush();
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gguf = arg(&args, "--gguf").or_else(|| std::env::var("LEAN_GGUF").ok()).ok_or_else(|| anyhow::anyhow!("set LEAN_GGUF or pass --gguf"))?;
    let tokenizer_dir = arg(&args, "--tokenizer-dir").or_else(|| std::env::var("LEAN_TOKENIZER_DIR").ok()).ok_or_else(|| anyhow::anyhow!("set LEAN_TOKENIZER_DIR or pass --tokenizer-dir"))?;
    let max_ctx: u32 = arg(&args, "--max-ctx").map(|s| s.parse()).transpose()?.unwrap_or(2048);
    let max_new_tokens: u32 = arg(&args, "--max-new-tokens").map(|s| s.parse()).transpose()?.unwrap_or(256);
    let system = arg(&args, "--system");
    let backend = match arg(&args, "--backend").as_deref() {
        None | Some("auto") => Backend::Auto,
        Some("webgpu") => Backend::WebGpu,
        Some("cpu") => Backend::Cpu,
        Some(other) => anyhow::bail!("unknown --backend {other} (auto, webgpu or cpu)"),
    };

    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json"))?;
    let opts = LoadOptions { gguf: GgufSource::Path(gguf), tokenizer: TokenizerSource::Path(format!("{tokenizer_dir}/tokenizer.json")), max_ctx, backend };
    let mut lean = pollster::block_on(Lean::load(opts))?;
    eprintln!("[backend: {}]", lean.backend());
    let params = GenParams { max_new_tokens, ..GenParams::default() };

    let mut messages: Vec<Message> = Vec::new();
    if let Some(s) = system {
        messages.push(Message::system(s));
    }

    for line in std::io::stdin().lines() {
        let line = line?;
        if line == "/exit" {
            break;
        }
        messages.push(Message::user(line));
        loop {
            let prompt = render(&chat_template, &messages, None, true)?;
            match pollster::block_on(lean.generate(&prompt, &params, print_token, None)) {
                Ok(reply) => {
                    println!();
                    messages.push(Message::assistant(reply));
                    break;
                }
                Err(LeanError::ContextOverflow { needed, available }) => {
                    eprintln!("[context overflow: needed {needed}, available {available}; dropping the oldest turn]");
                    if !trim_oldest_exchange(&mut messages) {
                        eprintln!("[nothing left to drop; try a shorter prompt or a larger --max-ctx]");
                        messages.pop();
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("[error: {e}]");
                    messages.pop();
                    break;
                }
            }
        }
    }
    Ok(())
}
