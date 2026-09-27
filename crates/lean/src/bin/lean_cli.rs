//! `lean-cli --tokens N [--prompt "..."] [--kernel naive|fast]`: runs
//! prefill + greedy decode on the GPU.
//!
//! With no `--prompt`, runs every case in `--fixture`
//! (`crates/lean/reference/fixture.json`, from `reference/gen_fixture.py`'s
//! HF-transformers-on-the-same-GGUF output) through this crate's own
//! tokenizer + chat-template path (`chat_template.rs` + the `tokenizers`
//! crate) and checks the resulting token ids, prefill top-20 logits, and
//! greedy continuation against each case.
//!
//! With `--prompt`, tokenizes that prompt itself (same path) and just runs
//! generation, with no fixture comparison.
//!
//! Model/tokenizer paths are never hardcoded (this crate is meant to be
//! publishable) — set `LEAN_GGUF` and `LEAN_TOKENIZER_DIR` (a directory
//! with `tokenizer.json` + `tokenizer_config.json`), or pass `--gguf`/
//! `--tokenizer-dir`.

use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use lean::chat_template::{read_chat_template, render_user_prompt};
use lean::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Kernel {
    /// Always the naive reference kernel (linear_q4.wgsl) regardless of M.
    Naive,
    /// Tiled prefill / coalesced-or-subgroup decode kernels (slice 2).
    Fast,
}

#[derive(Parser)]
struct Args {
    /// If set, tokenize and generate for this prompt only (no fixture check).
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    #[arg(long, env = "LEAN_GGUF")]
    gguf: String,
    #[arg(long, env = "LEAN_TOKENIZER_DIR")]
    tokenizer_dir: String,
    #[arg(long, default_value = "crates/lean/reference/fixture.json")]
    fixture: String,
    #[arg(long, value_enum, default_value_t = Kernel::Fast)]
    kernel: Kernel,
}

#[derive(Deserialize)]
struct Top20 {
    ids: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    prompt: String,
    input_ids: Vec<u32>,
    prefill_top20: Top20,
    greedy_continuation: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap());
    idx.into_iter().take(k).map(|i| (i, logits[i as usize])).collect()
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

/// Renders + tokenizes `prompt` the same way `gen_fixture.py`'s
/// `tok.apply_chat_template([...], add_generation_prompt=True)` does:
/// render the template to text, then tokenize with `add_special_tokens =
/// false` (the special tokens like `<|im_start|>` are already literal text
/// in the rendered string, and the tokenizer's added-vocabulary matching
/// picks them out as single tokens — the same mechanism HF's Python
/// tokenizer uses).
fn tokenize_prompt(tokenizer: &Tokenizer, chat_template: &str, prompt: &str) -> Result<Vec<u32>> {
    let rendered = render_user_prompt(chat_template, prompt)?;
    let encoding = tokenizer.encode(rendered, false).map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

fn run_generation(engine: &lean::engine::Engine, model: &GpuModel, token_ids: &[u32], n_tokens: usize) -> (Vec<f32>, Vec<u32>, f64, f64) {
    // Every call here starts an independent generation against a brand-new
    // KvCache — model.pool must be reset first, or decode can silently
    // rebind to a stale, already-dropped KvCache from a previous call (see
    // pool.rs's bug note; this is exactly the bug this line fixes).
    model.pool.reset();
    let seq = token_ids.len() as u32;
    let max_ctx = seq + n_tokens as u32 + 4;
    let mut cache = KvCache::new(engine, &model.config, max_ctx);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    let prefill_start = Instant::now();
    let mut logits = pollster::block_on(forward_prefill(engine, model, &mut cache, token_ids, &cos_buf, &sin_buf));
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_logits = logits.clone();

    let mut got_tokens = Vec::with_capacity(n_tokens);
    let mut decode_ms_total = 0f64;
    for _ in 0..n_tokens {
        let next_id = argmax(&logits);
        got_tokens.push(next_id);
        let decode_start = Instant::now();
        logits = pollster::block_on(forward_decode_step(engine, model, &mut cache, next_id, &cos_buf, &sin_buf));
        decode_ms_total += decode_start.elapsed().as_secs_f64() * 1000.0;
    }

    (prefill_logits, got_tokens, prefill_ms / seq as f64, decode_ms_total / n_tokens as f64)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tokenizer_json = format!("{}/tokenizer.json", args.tokenizer_dir);
    let tokenizer_config_json = format!("{}/tokenizer_config.json", args.tokenizer_dir);
    let tokenizer = Tokenizer::from_file(&tokenizer_json).map_err(|e| anyhow::anyhow!("loading {tokenizer_json}: {e}"))?;
    let chat_template = read_chat_template(&tokenizer_config_json)?;

    let engine = lean::engine::Engine::new()?;
    eprintln!("subgroup support: {}", engine.has_subgroups);
    let load_start = Instant::now();
    let model = GpuModel::load(&engine, &args.gguf, args.kernel == Kernel::Fast)?;
    eprintln!("loaded model in {:?} (kernel={:?})", load_start.elapsed(), args.kernel);
    eprintln!(
        "config: layers={} hidden={} heads={} kv_heads={} head_dim={} intermediate={} vocab={} eps={} theta={}",
        model.config.num_layers, model.config.hidden_size, model.config.num_heads, model.config.num_kv_heads, model.config.head_dim, model.config.intermediate_size, model.config.vocab_size, model.config.rms_norm_eps, model.config.rope_theta
    );

    if let Some(prompt) = &args.prompt {
        let token_ids = tokenize_prompt(&tokenizer, &chat_template, prompt)?;
        println!("tokenized prompt into {} ids: {token_ids:?}", token_ids.len());
        let (_, got_tokens, prefill_ms_per_tok, decode_ms_per_tok) = run_generation(&engine, &model, &token_ids, args.tokens);
        let text = tokenizer.decode(&got_tokens, true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
        println!("continuation: {text}");
        println!("prefill: {prefill_ms_per_tok:.2}ms/token, decode: {decode_ms_per_tok:.2}ms/token");
        return Ok(());
    }

    let fixture_json = std::fs::read_to_string(&args.fixture).with_context(|| format!("reading fixture {}", args.fixture))?;
    let fixture: Fixture = serde_json::from_str(&fixture_json)?;

    println!("kernel={:?}  case            seq  tok_match  top1_match  top20_maxdiff  prefill_ms/tok  decode_ms/tok", args.kernel);
    let mut all_ok = true;
    for case in &fixture.cases {
        let our_ids = tokenize_prompt(&tokenizer, &chat_template, &case.prompt)?;
        let ids_match = our_ids == case.input_ids;
        if !ids_match {
            eprintln!("[{}] TOKENIZATION MISMATCH: ours={our_ids:?} fixture={:?}", case.name, case.input_ids);
        }

        let (prefill_logits, got_tokens, prefill_ms_per_tok, decode_ms_per_tok) = run_generation(&engine, &model, &case.input_ids, args.tokens);

        let got_top20 = top_k(&prefill_logits, 20);
        let top1_match = got_top20[0].0 == case.prefill_top20.ids[0];
        let mut max_abs_diff = 0f32;
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (prefill_logits[id as usize] - case.prefill_top20.values[i]).abs();
            max_abs_diff = max_abs_diff.max(diff);
        }
        let tokens_match = got_tokens == case.greedy_continuation;

        println!(
            "kernel={:?}  {:<12} {:>4}  ids={ids_match:<5}  tok={tokens_match:<5}  top1={top1_match:<5}  top20_maxdiff={max_abs_diff:.6}  prefill={prefill_ms_per_tok:.2}  decode={decode_ms_per_tok:.2}",
            args.kernel, case.name, case.input_ids.len()
        );
        if !ids_match || !tokens_match || max_abs_diff > 1e-3 {
            all_ok = false;
            eprintln!("[{}] got tokens:      {got_tokens:?}", case.name);
            eprintln!("[{}] fixture tokens:  {:?}", case.name, case.greedy_continuation);
        }
    }

    if !all_ok {
        anyhow::bail!("one or more fixture cases failed (see stderr above)");
    }
    println!("all fixture cases passed (kernel={:?})", args.kernel);
    Ok(())
}
