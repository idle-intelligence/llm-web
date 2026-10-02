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
use lean::model::{build_mask_bitset, build_rope_tables, forward_decode_step, forward_decode_step_argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache, KvSnapshot};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Kernel {
    /// Always the naive reference kernel (linear_q4.wgsl) regardless of M.
    Naive,
    /// Tiled prefill / coalesced decode kernels (slice 2).
    Fast,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum EngineKind {
    /// The wgpu forward pass (model.rs) - default, unchanged behavior.
    Gpu,
    /// The single-threaded CPU forward pass (cpu.rs) - see
    /// crates/lean/docs (cpu-fallback plan). Skips the two long-context
    /// agent-tool-calling fixture cases by default (`long_tools_single`/
    /// `long_tools_multiturn`, 2225/2354 prompt tokens) - the CPU rung's
    /// per-token cost makes those minutes-long on this reference kernel;
    /// pass `--long` to include them anyway.
    Cpu,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Check {
    /// Default: the fixture-parity loop this file has always run.
    Fixture,
    /// KV snapshot/restore: prefill(prefix) -> snapshot -> restore ->
    /// prefill(suffix) -> generate must match prefill(prefix+suffix) ->
    /// generate exactly, plus an export/import byte round trip.
    KvSnapshot,
    /// Per-step logit mask: a singleton mask at every step must force an
    /// exact target string; an all-allowed mask must match unmasked output.
    Mask,
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
    /// Which native check to run instead of the default fixture loop.
    #[arg(long, value_enum, default_value_t = Check::Fixture)]
    check: Check,
    /// Which forward-pass rung to run: `gpu` (wgpu, model.rs) or `cpu`
    /// (single-threaded, cpu.rs).
    #[arg(long, value_enum, default_value_t = EngineKind::Gpu)]
    engine: EngineKind,
    /// Include the long-context agent-tool-calling fixture cases
    /// (`long_tools_single`/`long_tools_multiturn`) on the CPU engine. No
    /// effect on `--engine gpu`, which always runs every case.
    #[arg(long, default_value_t = false)]
    long: bool,
}

#[derive(Deserialize)]
struct Top20 {
    ids: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    prompt: Option<String>,
    input_ids: Vec<u32>,
    prefill_top20: Top20,
    greedy_continuation: Vec<u32>,
    /// True for the long-context agent-tool-calling cases (see
    /// `reference/gen_fixture.py`'s `LONG_TOKEN_CASES`, and
    /// `tests/fixture_parity.rs`'s `Case` doc comment): `input_ids` is
    /// already the fully tools-chat-templated prompt, and there is no
    /// `prompt` field to retokenize from (this crate's `chat_template.rs`
    /// has no tools support).
    #[serde(default)]
    no_retokenize: bool,
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

    engine.reset_dispatch_count();
    let prefill_start = Instant::now();
    let logits = pollster::block_on(forward_prefill(engine, model, &mut cache, token_ids, &cos_buf, &sin_buf, None));
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_dispatches = engine.dispatch_count();
    let prefill_logits = logits.clone();

    // Argmax over the prompt's last-position logits happens once here (CPU,
    // off the already-read-back `Vec<f32>`); every subsequent token's argmax
    // runs on the GPU inside `forward_decode_step_argmax` itself, so the
    // decode loop's only readback per step is 4 bytes (one `u32`), not the
    // full `vocab_size * 4`-byte logits vector - see model.rs's doc comment.
    let mut next_id = argmax(&logits);
    let mut got_tokens = Vec::with_capacity(n_tokens);
    let mut decode_ms_total = 0f64;
    engine.reset_dispatch_count();
    for _ in 0..n_tokens {
        got_tokens.push(next_id);
        let decode_start = Instant::now();
        next_id = pollster::block_on(forward_decode_step_argmax(engine, model, &mut cache, next_id, &cos_buf, &sin_buf, None));
        decode_ms_total += decode_start.elapsed().as_secs_f64() * 1000.0;
    }
    let decode_dispatches_per_step = engine.dispatch_count() / n_tokens as u64;
    eprintln!(
        "seq={seq} prefill_dispatches={prefill_dispatches} ({:.1}/token) decode_dispatches_per_step={decode_dispatches_per_step}",
        prefill_dispatches as f64 / seq as f64
    );
    if engine.profiling_enabled() {
        lean::profile_report::report(n_tokens as u64);
    }

    (prefill_logits, got_tokens, prefill_ms / seq as f64, decode_ms_total / n_tokens as f64)
}

/// CPU mirror of `run_generation`: prefill then greedy-decode `n_tokens`
/// against a fresh `CpuKvCache`, timing prefill/decode separately. No
/// `Pool`/dispatch-count bookkeeping (there is no GPU dispatch here) and no
/// `pollster::block_on` (the CPU forward functions are plain synchronous
/// calls, not futures) - otherwise the same contract as `run_generation`.
fn run_generation_cpu(model: &lean::cpu::CpuModel, token_ids: &[u32], n_tokens: usize) -> (Vec<f32>, Vec<u32>, f64, f64) {
    let seq = token_ids.len() as u32;
    let max_ctx = seq + n_tokens as u32 + 4;
    let mut cache = lean::cpu::CpuKvCache::new(&model.config, max_ctx as usize);

    let prefill_start = Instant::now();
    let logits = lean::cpu::forward_prefill(model, &mut cache, token_ids);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_logits = logits.clone();

    let mut next_id = argmax(&logits);
    let mut got_tokens = Vec::with_capacity(n_tokens);
    let mut decode_ms_total = 0f64;
    for _ in 0..n_tokens {
        got_tokens.push(next_id);
        let decode_start = Instant::now();
        next_id = lean::cpu::forward_decode_step_argmax(model, &mut cache, next_id);
        decode_ms_total += decode_start.elapsed().as_secs_f64() * 1000.0;
    }

    (prefill_logits, got_tokens, prefill_ms / seq as f64, decode_ms_total / n_tokens as f64)
}

/// `--engine cpu`'s fixture loop: same checks as the GPU path's inline loop
/// in `main` (tokenization match, prefill top-20 vs fixture, greedy
/// continuation match), driven through `run_generation_cpu` instead of
/// `run_generation`. Kept as its own function (rather than threading an
/// `EngineKind` through `run_generation`/`main`'s loop) since the two
/// engines take different model types (`CpuModel` vs `GpuModel`) with no
/// shared trait yet - see this crate's CPU-fallback plan on that being a
/// possible follow-up once both rungs are proven out.
fn run_fixture_cpu(gguf_path: &str, tokenizer: &Tokenizer, chat_template: &str, fixture: &Fixture, n_tokens: usize, include_long: bool) -> Result<()> {
    let load_start = Instant::now();
    let model = lean::cpu::CpuModel::load(gguf_path)?;
    eprintln!("loaded CPU model in {:?}", load_start.elapsed());

    println!("engine=cpu  case            seq  tok_match  top1_match  top20_maxdiff  prefill_ms/tok  decode_ms/tok");
    let mut all_ok = true;
    for case in &fixture.cases {
        if !include_long && case.name.starts_with("long_tools") {
            println!("engine=cpu  {:<12} skipped (pass --long to include)", case.name);
            continue;
        }
        let ids_match = if case.no_retokenize {
            true // no `prompt` field to retokenize from - see Case's doc comment
        } else {
            let our_ids = tokenize_prompt(tokenizer, chat_template, case.prompt.as_deref().unwrap_or_default())?;
            let m = our_ids == case.input_ids;
            if !m {
                eprintln!("[{}] TOKENIZATION MISMATCH: ours={our_ids:?} fixture={:?}", case.name, case.input_ids);
            }
            m
        };

        let (prefill_logits, got_tokens, prefill_ms_per_tok, decode_ms_per_tok) = run_generation_cpu(&model, &case.input_ids, n_tokens);

        let got_top20 = top_k(&prefill_logits, 20);
        let top1_match = got_top20[0].0 == case.prefill_top20.ids[0];
        let mut max_abs_diff = 0f32;
        for (i, &id) in case.prefill_top20.ids.iter().enumerate() {
            let diff = (prefill_logits[id as usize] - case.prefill_top20.values[i]).abs();
            max_abs_diff = max_abs_diff.max(diff);
        }
        let tokens_match = got_tokens == case.greedy_continuation[..n_tokens.min(case.greedy_continuation.len())];

        println!(
            "engine=cpu  {:<12} {:>4}  ids={ids_match:<5}  tok={tokens_match:<5}  top1={top1_match:<5}  top20_maxdiff={max_abs_diff:.6}  prefill={prefill_ms_per_tok:.2}  decode={decode_ms_per_tok:.2}",
            case.name, case.input_ids.len()
        );
        if !ids_match || !tokens_match {
            all_ok = false;
            eprintln!("[{}] got tokens:      {got_tokens:?}", case.name);
            eprintln!("[{}] fixture tokens:  {:?}", case.name, case.greedy_continuation);
        }
    }

    if !all_ok {
        anyhow::bail!("one or more fixture cases failed (see stderr above)");
    }
    println!("all fixture cases passed (engine=cpu)");
    Ok(())
}

/// `--check kv-snapshot`: prefill(prefix) -> snapshot -> restore into a
/// fresh cache -> prefill(suffix) -> greedy generate must reproduce
/// prefill(prefix+suffix) -> greedy generate exactly, plus an export/import
/// byte round trip. Uses `--prompt` split at its midpoint token.
fn check_kv_snapshot(engine: &lean::engine::Engine, model: &GpuModel, tokenizer: &Tokenizer, chat_template: &str, prompt: &str, n_new: usize) -> Result<()> {
    let all_ids = tokenize_prompt(tokenizer, chat_template, prompt)?;
    anyhow::ensure!(all_ids.len() >= 4, "prompt too short to split into a prefix/suffix ({} tokens)", all_ids.len());
    let split = all_ids.len() / 2;
    let (prefix, suffix) = all_ids.split_at(split);
    let max_ctx = all_ids.len() as u32 + n_new as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // Path A: prefill(prefix+suffix) from scratch, then greedy generate.
    model.pool.reset();
    let mut cache_a = KvCache::new(engine, &model.config, max_ctx);
    let mut logits_a = pollster::block_on(forward_prefill(engine, model, &mut cache_a, &all_ids, &cos_buf, &sin_buf, None));
    let mut tokens_a = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let id = argmax(&logits_a);
        tokens_a.push(id);
        logits_a = pollster::block_on(forward_decode_step(engine, model, &mut cache_a, id, &cos_buf, &sin_buf, None));
    }

    // Path B: prefill(prefix) -> snapshot -> restore -> prefill(suffix) -> generate.
    model.pool.reset();
    let mut cache_prefix = KvCache::new(engine, &model.config, max_ctx);
    let _ = pollster::block_on(forward_prefill(engine, model, &mut cache_prefix, prefix, &cos_buf, &sin_buf, None));
    let snapshot = pollster::block_on(cache_prefix.snapshot(engine));

    let bytes = snapshot.to_bytes();
    let snapshot2 = KvSnapshot::from_bytes(&bytes)?;

    model.pool.reset();
    let mut cache_b = KvCache::new(engine, &model.config, max_ctx);
    cache_b.restore(engine, &snapshot2);
    let mut logits_b = pollster::block_on(forward_prefill_suffix(engine, model, &mut cache_b, suffix, &cos_buf, &sin_buf, None));
    let mut tokens_b = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let id = argmax(&logits_b);
        tokens_b.push(id);
        logits_b = pollster::block_on(forward_decode_step(engine, model, &mut cache_b, id, &cos_buf, &sin_buf, None));
    }

    println!("prefix_tokens={} suffix_tokens={} export_bytes={}", prefix.len(), suffix.len(), bytes.len());
    if tokens_a == tokens_b {
        println!("PASS: restore+prefill(suffix) matches prefill(prefix+suffix) exactly ({tokens_a:?})");
        Ok(())
    } else {
        anyhow::bail!("FAIL: token mismatch\n  full_prefill: {tokens_a:?}\n  restore+suffix: {tokens_b:?}");
    }
}

/// `--check mask`: a per-step singleton mask must force greedy decoding to
/// reproduce a fixed target string; an all-allowed mask must match
/// unmasked output.
fn check_mask(engine: &lean::engine::Engine, model: &GpuModel, tokenizer: &Tokenizer, chat_template: &str, prompt: &str) -> Result<()> {
    let vocab = model.config.vocab_size;
    let prompt_ids = tokenize_prompt(tokenizer, chat_template, prompt)?;
    let target = "{\"ok\":true}";
    let target_ids = tokenizer.encode(target, false).map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?.get_ids().to_vec();

    let max_ctx = prompt_ids.len() as u32 + target_ids.len() as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    model.pool.reset();
    let mut cache = KvCache::new(engine, &model.config, max_ctx);
    let mask0 = engine.buf_u32(&build_mask_bitset(vocab, &[target_ids[0]]), "mask0");
    let logits = pollster::block_on(forward_prefill(engine, model, &mut cache, &prompt_ids, &cos_buf, &sin_buf, Some(&mask0)));
    let mut got = Vec::with_capacity(target_ids.len());
    let mut next = argmax(&logits);
    got.push(next);
    for &want in &target_ids[1..] {
        let mask = engine.buf_u32(&build_mask_bitset(vocab, &[want]), "mask_step");
        next = pollster::block_on(forward_decode_step_argmax(engine, model, &mut cache, next, &cos_buf, &sin_buf, Some(&mask)));
        got.push(next);
    }

    if got != target_ids {
        anyhow::bail!("FAIL: singleton mask did not force the target string\n  got:    {got:?}\n  target: {target_ids:?}");
    }
    let text = tokenizer.decode(&got, true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
    println!("PASS: singleton mask forced target string {text:?}");

    // All-allowed mask must match unmasked output over the same prompt.
    let n_steps = 8usize.min(max_ctx as usize - prompt_ids.len());
    model.pool.reset();
    let mut cache_free = KvCache::new(engine, &model.config, max_ctx);
    let logits_free = pollster::block_on(forward_prefill(engine, model, &mut cache_free, &prompt_ids, &cos_buf, &sin_buf, None));
    let mut unmasked = Vec::with_capacity(n_steps);
    let mut next = argmax(&logits_free);
    for _ in 0..n_steps {
        unmasked.push(next);
        next = pollster::block_on(forward_decode_step_argmax(engine, model, &mut cache_free, next, &cos_buf, &sin_buf, None));
    }

    let all_bits = build_mask_bitset(vocab, &(0..vocab as u32).collect::<Vec<_>>());
    model.pool.reset();
    let mut cache_masked = KvCache::new(engine, &model.config, max_ctx);
    let mask_all0 = engine.buf_u32(&all_bits, "mask_all0");
    let logits_masked = pollster::block_on(forward_prefill(engine, model, &mut cache_masked, &prompt_ids, &cos_buf, &sin_buf, Some(&mask_all0)));
    let mut masked = Vec::with_capacity(n_steps);
    let mut next = argmax(&logits_masked);
    for _ in 0..n_steps {
        masked.push(next);
        let mask_step = engine.buf_u32(&all_bits, "mask_all_step");
        next = pollster::block_on(forward_decode_step_argmax(engine, model, &mut cache_masked, next, &cos_buf, &sin_buf, Some(&mask_step)));
    }

    if unmasked == masked {
        println!("PASS: all-allowed mask matches unmasked output ({} tokens)", n_steps);
        Ok(())
    } else {
        anyhow::bail!("FAIL: all-allowed mask changed output\n  unmasked: {unmasked:?}\n  masked:   {masked:?}");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tokenizer_json = format!("{}/tokenizer.json", args.tokenizer_dir);
    let tokenizer_config_json = format!("{}/tokenizer_config.json", args.tokenizer_dir);
    let tokenizer = Tokenizer::from_file(&tokenizer_json).map_err(|e| anyhow::anyhow!("loading {tokenizer_json}: {e}"))?;
    let chat_template = read_chat_template(&tokenizer_config_json)?;

    if args.engine == EngineKind::Cpu {
        if let Some(prompt) = &args.prompt {
            let token_ids = tokenize_prompt(&tokenizer, &chat_template, prompt)?;
            println!("tokenized prompt into {} ids: {token_ids:?}", token_ids.len());
            let model = lean::cpu::CpuModel::load(&args.gguf)?;
            let (_, got_tokens, prefill_ms_per_tok, decode_ms_per_tok) = run_generation_cpu(&model, &token_ids, args.tokens);
            let text = tokenizer.decode(&got_tokens, true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            println!("continuation: {text}");
            println!("prefill: {prefill_ms_per_tok:.2}ms/token, decode: {decode_ms_per_tok:.2}ms/token");
            return Ok(());
        }
        let fixture_json = std::fs::read_to_string(&args.fixture).with_context(|| format!("reading fixture {}", args.fixture))?;
        let fixture: Fixture = serde_json::from_str(&fixture_json)?;
        return run_fixture_cpu(&args.gguf, &tokenizer, &chat_template, &fixture, args.tokens, args.long);
    }

    let engine = lean::engine::Engine::new()?;
    if std::env::var("LEAN_DEBUG_ADAPTER").as_deref() == Ok("1") {
        eprintln!("adapter: {}", engine.adapter_report);
    }
    let load_start = Instant::now();
    let model = GpuModel::load(&engine, &args.gguf, args.kernel == Kernel::Fast)?;
    eprintln!("loaded model in {:?} (kernel={:?})", load_start.elapsed(), args.kernel);
    eprintln!(
        "config: layers={} hidden={} heads={} kv_heads={} head_dim={} intermediate={} vocab={} eps={} theta={}",
        model.config.num_layers, model.config.hidden_size, model.config.num_heads, model.config.num_kv_heads, model.config.head_dim, model.config.intermediate_size, model.config.vocab_size, model.config.rms_norm_eps, model.config.rope_theta
    );

    let default_check_prompt = "Tell me a short fact about the ocean.".to_string();
    match args.check {
        Check::KvSnapshot => {
            return check_kv_snapshot(&engine, &model, &tokenizer, &chat_template, args.prompt.as_ref().unwrap_or(&default_check_prompt), args.tokens.min(16));
        }
        Check::Mask => {
            return check_mask(&engine, &model, &tokenizer, &chat_template, args.prompt.as_ref().unwrap_or(&default_check_prompt));
        }
        Check::Fixture => {}
    }

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
        let ids_match = if case.no_retokenize {
            true
        } else {
            let our_ids = tokenize_prompt(&tokenizer, &chat_template, case.prompt.as_deref().unwrap_or_default())?;
            let m = our_ids == case.input_ids;
            if !m {
                eprintln!("[{}] TOKENIZATION MISMATCH: ours={our_ids:?} fixture={:?}", case.name, case.input_ids);
            }
            m
        };

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
