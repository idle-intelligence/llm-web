//! Reference client for lean's one-turn `generate` API: a multi-turn
//! terminal chat, with the conversation kept here (not in lean - see
//! `chat.rs`'s doc comment) and a simple trim rule on `LeanError::
//! ContextOverflow`: drop the oldest user/assistant pair, keep any system
//! message, retry the same call. This is the same policy
//! `web/lean-chat-worker.js` applies in the browser.
//!
//! ```sh
//! LEAN_GGUF=/path/to/model.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/tokenizer_dir \
//! cargo run -p lean --release --example chat -- [--engine gpu|cpu] [--max-ctx 2048] [--max-new-tokens 256] [--system "..."]
//! ```
//!
//! Reads one prompt per line from stdin, streams the reply to stdout, and
//! repeats until EOF (Ctrl-D) or a line that is exactly `/exit`.

use lean::chat::{cpu_generate, gpu_generate, GenParams, Message, PrefixCache};
use lean::chat_template::read_chat_template;
use lean::cpu::{CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::error::LeanError;
use lean::model::{build_rope_tables, GpuModel, KvCache};
use std::io::Write;
use tokenizers::Tokenizer;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Drops the oldest user/assistant pair (after any leading system message)
/// so the next retry has strictly fewer tokens. Returns `false` (nothing
/// left to drop) once only the system message and the current prompt
/// remain - the caller then has to give up and report the overflow.
fn trim_oldest_pair(messages: &mut Vec<Message>) -> bool {
    let drop_from = if messages.first().is_some_and(|m| m.role == "system") { 1 } else { 0 };
    if messages.len() < drop_from + 2 {
        return false;
    }
    messages.drain(drop_from..drop_from + 2);
    true
}

fn print_token(text: &str) {
    print!("{text}");
    let _ = std::io::stdout().flush();
}

async fn run_gpu(gguf: &str, tokenizer: &Tokenizer, chat_template: &str, max_ctx: u32, max_new_tokens: u32, system: Option<&str>) -> anyhow::Result<()> {
    let engine = Engine::new()?;
    let model = GpuModel::load(&engine, gguf, true)?;
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut prefix = PrefixCache::new();
    let params = GenParams { max_new_tokens, ..GenParams::default() };

    let mut messages: Vec<Message> = Vec::new();
    if let Some(s) = system {
        messages.push(Message::new("system", s));
    }

    for line in std::io::stdin().lines() {
        let line = line?;
        if line == "/exit" {
            break;
        }
        messages.push(Message::new("user", &line));
        loop {
            match gpu_generate(&engine, &model, &mut cache, &cos_buf, &sin_buf, &mut prefix, tokenizer, chat_template, &messages, &params, None, print_token, || false).await {
                Ok(reply) => {
                    println!();
                    messages.push(Message::new("assistant", reply));
                    break;
                }
                Err(LeanError::ContextOverflow { needed, available }) => {
                    eprintln!("[context overflow: needed {needed}, available {available}; dropping the oldest turn]");
                    if !trim_oldest_pair(&mut messages) {
                        eprintln!("[nothing left to drop; try a shorter prompt or a larger --max-ctx]");
                        messages.pop(); // take the unanswered prompt back out
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

async fn run_cpu(gguf: &str, tokenizer: &Tokenizer, chat_template: &str, max_ctx: u32, max_new_tokens: u32, system: Option<&str>) -> anyhow::Result<()> {
    let model = CpuModel::load(gguf)?;
    let mut cache = CpuKvCache::new(&model.config, max_ctx as usize);
    let mut prefix = PrefixCache::new();
    let params = GenParams { max_new_tokens, ..GenParams::default() };

    let mut messages: Vec<Message> = Vec::new();
    if let Some(s) = system {
        messages.push(Message::new("system", s));
    }

    for line in std::io::stdin().lines() {
        let line = line?;
        if line == "/exit" {
            break;
        }
        messages.push(Message::new("user", &line));
        loop {
            match cpu_generate(&model, &mut cache, &mut prefix, tokenizer, chat_template, &messages, &params, print_token, || false, || async {}).await {
                Ok(reply) => {
                    println!();
                    messages.push(Message::new("assistant", reply));
                    break;
                }
                Err(LeanError::ContextOverflow { needed, available }) => {
                    eprintln!("[context overflow: needed {needed}, available {available}; dropping the oldest turn]");
                    if !trim_oldest_pair(&mut messages) {
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gguf = arg(&args, "--gguf").or_else(|| std::env::var("LEAN_GGUF").ok()).ok_or_else(|| anyhow::anyhow!("set LEAN_GGUF or pass --gguf"))?;
    let tokenizer_dir = arg(&args, "--tokenizer-dir").or_else(|| std::env::var("LEAN_TOKENIZER_DIR").ok()).ok_or_else(|| anyhow::anyhow!("set LEAN_TOKENIZER_DIR or pass --tokenizer-dir"))?;
    let max_ctx: u32 = arg(&args, "--max-ctx").map(|s| s.parse()).transpose()?.unwrap_or(2048);
    let max_new_tokens: u32 = arg(&args, "--max-new-tokens").map(|s| s.parse()).transpose()?.unwrap_or(256);
    let system = arg(&args, "--system");
    let engine_kind = arg(&args, "--engine").unwrap_or_else(|| "gpu".to_string());

    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json"))?;

    pollster::block_on(async {
        match engine_kind.as_str() {
            "cpu" => run_cpu(&gguf, &tokenizer, &chat_template, max_ctx, max_new_tokens, system.as_deref()).await,
            _ => run_gpu(&gguf, &tokenizer, &chat_template, max_ctx, max_new_tokens, system.as_deref()).await,
        }
    })
}
