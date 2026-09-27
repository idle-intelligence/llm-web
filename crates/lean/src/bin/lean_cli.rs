//! `lean-cli --prompt ... --tokens N`: runs prefill + greedy decode on the
//! GPU and compares against `crates/lean/reference/fixture.json`
//! (`reference/gen_fixture.py`'s HF-transformers-on-the-same-GGUF output).
//!
//! Tokenization note: this slice reads the fixture's own `input_ids`
//! (already chat-templated by the reference script's tokenizer) rather than
//! re-tokenizing `--prompt` itself — this crate doesn't carry a chat
//! template implementation yet (that's llm-wasm's `templates.rs` job, out
//! of scope for a raw-kernel-correctness slice). `--prompt` is accepted and
//! checked against the fixture's own `prompt` field so a mismatch is loud
//! rather than silently comparing against the wrong reference.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};
use serde::Deserialize;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    #[arg(long, default_value = "/Users/tc/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf")]
    gguf: String,
    #[arg(long, default_value = "crates/lean/reference/fixture.json")]
    fixture: String,
}

#[derive(Deserialize)]
struct Top20 {
    ids: Vec<u32>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    prompt: String,
    input_ids: Vec<u32>,
    prefill_top20: Top20,
    greedy_continuation: Vec<u32>,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let fixture_json = std::fs::read_to_string(&args.fixture).with_context(|| format!("reading fixture {}", args.fixture))?;
    let fixture: Fixture = serde_json::from_str(&fixture_json)?;
    ensure!(fixture.prompt == args.prompt, "--prompt ({:?}) does not match fixture's prompt ({:?}) — regenerate the fixture or pass the matching prompt", args.prompt, fixture.prompt);

    let engine = Engine::new()?;
    let load_start = Instant::now();
    let model = GpuModel::load(&engine, &args.gguf)?;
    eprintln!("loaded model in {:?}", load_start.elapsed());
    eprintln!(
        "config: layers={} hidden={} heads={} kv_heads={} head_dim={} intermediate={} vocab={} eps={} theta={}",
        model.config.num_layers, model.config.hidden_size, model.config.num_heads, model.config.num_kv_heads, model.config.head_dim, model.config.intermediate_size, model.config.vocab_size, model.config.rms_norm_eps, model.config.rope_theta
    );

    let seq = fixture.input_ids.len();
    let max_ctx = (seq + args.tokens + 4) as u32;
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    let prefill_start = Instant::now();
    let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &fixture.input_ids, &cos_buf, &sin_buf));
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("prefill: seq={seq} in {prefill_ms:.1}ms ({:.1}ms/token)", prefill_ms / seq as f64);

    let got_top20 = top_k(&logits, 20);
    let got_top1 = got_top20[0].0;
    let expected_top1 = fixture.prefill_top20.ids[0];
    println!("prefill top1: got={got_top1} expected={expected_top1} match={}", got_top1 == expected_top1);

    let mut max_abs_diff = 0f32;
    let mut worst: Option<(u32, f32, f32)> = None;
    for (i, &id) in fixture.prefill_top20.ids.iter().enumerate() {
        let got = logits[id as usize];
        let expected = fixture.prefill_top20.values[i];
        let diff = (got - expected).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst = Some((id, got, expected));
        }
    }
    println!("prefill top20 max abs diff (at fixture's ids): {max_abs_diff:.6} worst={worst:?}");
    println!("prefill top20 (ours):    {:?}", got_top20.iter().take(5).collect::<Vec<_>>());
    println!("prefill top20 (fixture): {:?}", fixture.prefill_top20.ids.iter().zip(fixture.prefill_top20.values.iter()).take(5).collect::<Vec<_>>());

    let mut next_logits = logits;
    let mut decode_ms_total = 0f64;
    let mut got_tokens = Vec::with_capacity(args.tokens);
    let mut first_divergence: Option<usize> = None;
    for step in 0..args.tokens {
        let next_id = argmax(&next_logits);
        got_tokens.push(next_id);
        if first_divergence.is_none() && fixture.greedy_continuation.get(step).copied() != Some(next_id) {
            first_divergence = Some(step);
        }
        let decode_start = Instant::now();
        next_logits = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, next_id, &cos_buf, &sin_buf));
        decode_ms_total += decode_start.elapsed().as_secs_f64() * 1000.0;
    }

    println!("greedy continuation (ours):    {got_tokens:?}");
    println!("greedy continuation (fixture): {:?}", fixture.greedy_continuation);
    println!("greedy tokens identical: {}", got_tokens == fixture.greedy_continuation);
    match first_divergence {
        Some(step) => println!("first divergence at decode step {step}"),
        None => println!("no divergence in {} decode steps", args.tokens),
    }
    println!("decode: {:.2}ms/token ({} steps, native Metal, first number, one GPU job)", decode_ms_total / args.tokens as f64, args.tokens);

    Ok(())
}
