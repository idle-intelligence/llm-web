//! Prefix KV images through `Lean`: the system prompt and the 12 tools of
//! `fixtures/reference/inputs/02_tools_single.json`, rendered with the
//! model's own chat template, is the prefix; the full prompt adds the user
//! turn. Greedy, on one backend:
//!
//! - a fresh engine's reply to the full prompt (one cold prefill) equals
//!   the reply of an engine that restored the prefix image first;
//! - the restored engine's KV cache after the reply equals, byte for byte,
//!   that of the engine that wrote the image (both saved with
//!   `save_prefix` of the full prompt);
//! - an image from another engine build (the header's engine field
//!   changed), for another prefix, or cut short is refused, and the engine
//!   still gives the fresh reply.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! LEAN_GGUF_QWEN3_1_7B=/path/to/Qwen3-1.7B-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_QWEN3_1_7B=/path/to/Qwen3-1.7B \
//! cargo test -p lean --release --features threads --test prefix_image -- --ignored --nocapture --test-threads 1
//! ```

use std::io::Cursor;
use std::time::Instant;

use lean::{Backend, GenParams, GgufSource, Lean, LeanError, LoadOptions, PrefixImageError, TokenizerSource};
use lean_chat::Message;

const MAX_CTX: u32 = 4096;
const MAX_NEW: u32 = 64;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run this test"))
}

fn load(gguf: &str, dir: &str, backend: Backend) -> Lean {
    let opts = LoadOptions { gguf: GgufSource::Path(gguf.into()), tokenizer: TokenizerSource::Path(format!("{dir}/tokenizer.json")), max_ctx: MAX_CTX, backend };
    pollster::block_on(Lean::load(opts)).expect("loading lean")
}

/// (prefix, full prompt): the full prompt rendered with tools and a user
/// turn, and the prefix cut where the user turn starts.
fn prompts(dir: &str) -> (String, String) {
    let input: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/reference/inputs/02_tools_single.json")).unwrap()).unwrap();
    let messages: Vec<Message> = serde_json::from_value(input["messages"].clone()).unwrap();
    let tools = input["tools"].as_array().unwrap().clone();
    let template = lean_chat::read_chat_template(&format!("{dir}/tokenizer_config.json")).unwrap();
    let full = lean_chat::render(&template, &messages, Some(&tools), true).unwrap();
    let user = messages.iter().rev().find(|m| m.role == "user").unwrap();
    let cut = full.rfind(&user.content).unwrap();
    let cut = full[..cut].rfind("<|im_start|>").unwrap();
    (full[..cut].to_string(), full)
}

fn generate(lean: &mut Lean, text: &str) -> (String, f64) {
    let params = GenParams { max_new_tokens: MAX_NEW, temperature: 0.0, ..GenParams::default() };
    let t = Instant::now();
    let mut first = None;
    let reply = pollster::block_on(lean.generate(text, &params, |_| {
        first.get_or_insert_with(|| t.elapsed().as_secs_f64() * 1e3);
    }, None))
    .unwrap();
    (reply, first.unwrap_or(f64::NAN))
}

fn save(lean: &mut Lean, text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let n = pollster::block_on(lean.save_prefix(text, &mut out)).unwrap();
    assert_eq!(n as usize, out.len());
    out
}

fn stale(result: Result<(), LeanError>) -> PrefixImageError {
    match result {
        Err(LeanError::PrefixImage(e)) => e,
        other => panic!("expected a PrefixImage error, got {other:?}"),
    }
}

fn restore_equals_fresh(gguf_var: &str, dir_var: &str, backend: Backend) {
    let (gguf, dir) = (env(gguf_var), env(dir_var));
    let (prefix, full) = prompts(&dir);

    let mut fresh = load(&gguf, &dir, backend);
    let prefix_tokens = fresh.token_count(&prefix).unwrap();
    println!("{backend:?} on {}: prefix {prefix_tokens} tokens, full prompt {} tokens", fresh.backend(), fresh.token_count(&full).unwrap());
    let (want, cold_ttft) = generate(&mut fresh, &full);
    assert!(!want.is_empty());
    drop(fresh);

    let mut writer = load(&gguf, &dir, backend);
    let t = Instant::now();
    let image = save(&mut writer, &prefix);
    let save_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let again = save(&mut writer, &prefix);
    let readback_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(again, image, "a second save of a resident prefix must give the same bytes");
    let (reply_writer, _) = generate(&mut writer, &full);
    let kv_writer = save(&mut writer, &full);
    drop(writer);

    let mut restored = load(&gguf, &dir, backend);
    let key = restored.prefix_key(&prefix).unwrap();
    let t = Instant::now();
    restored.restore_prefix(&prefix, &mut Cursor::new(&image)).unwrap();
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;
    let (reply_restored, warm_ttft) = generate(&mut restored, &full);
    let kv_restored = save(&mut restored, &full);
    println!(
        "  image {} bytes ({:.1} MB), key {key}; save (prefill + readback) {save_ms:.0} ms, readback only {readback_ms:.0} ms, restore {restore_ms:.0} ms; first token: cold {cold_ttft:.0} ms, after restore {warm_ttft:.0} ms",
        image.len(),
        image.len() as f64 / 1e6
    );
    assert_eq!(reply_restored, want, "restore then generate must equal a fresh prefill");
    assert_eq!(reply_writer, want, "save then generate must equal a fresh prefill");
    assert!(kv_restored == kv_writer, "the KV cache after the reply must be bit-identical with and without the image");

    // Refusals leave the engine usable and change nothing in the output.
    let mut other = load(&gguf, &dir, backend);
    let mut old_engine = image.clone();
    old_engine[72] ^= 1;
    assert_eq!(stale(other.restore_prefix(&prefix, &mut Cursor::new(&old_engine))), PrefixImageError::Stale("engine"));
    let changed = format!("{prefix} ");
    assert_ne!(other.prefix_key(&changed).unwrap(), key);
    assert_eq!(stale(other.restore_prefix(&changed, &mut Cursor::new(&image))), PrefixImageError::Stale("prefix"));
    assert_eq!(stale(other.restore_prefix(&prefix, &mut Cursor::new(&image[..image.len() - 4]))), PrefixImageError::Truncated);
    assert_eq!(stale(other.restore_prefix(&prefix, &mut Cursor::new(&image[..64]))), PrefixImageError::NotAnImage);
    assert_eq!(generate(&mut other, &full).0, want, "after refused images the engine must give the fresh reply");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn restore_equals_fresh_gpu_qwen25_0_5b() {
    restore_equals_fresh("LEAN_GGUF", "LEAN_TOKENIZER_DIR", Backend::WebGpu);
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn restore_equals_fresh_cpu_qwen25_0_5b() {
    restore_equals_fresh("LEAN_GGUF", "LEAN_TOKENIZER_DIR", Backend::Cpu);
}

#[test]
#[ignore = "needs LEAN_GGUF_QWEN3_1_7B and LEAN_TOKENIZER_DIR_QWEN3_1_7B on disk; never committed to this repo"]
fn restore_equals_fresh_gpu_qwen3_1_7b() {
    restore_equals_fresh("LEAN_GGUF_QWEN3_1_7B", "LEAN_TOKENIZER_DIR_QWEN3_1_7B", Backend::WebGpu);
}
