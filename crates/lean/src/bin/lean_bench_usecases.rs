//! Two native benchmarks for the project's own consumer shapes (docs/runs/2026-09-28-lean-use-cases.md),
//! neither of which the plain `lean-cli --prompt` chat-protocol path covers:
//!
//! - `life-cells`: llm-life variant A's real shape — the rules prefix
//!   prefilled once and snapshotted, then many per-cell chunks restored
//!   against that snapshot and forwarded through `forward_chunk_spec` with
//!   the sliced lm-head and a LoRA adapter. Reports cells/second.
//! - `sonos-turns`: the sonos MCP agent's real shape — a long tool prompt
//!   prefilled once and snapshotted, then per turn: restore, append a short
//!   command, greedy-decode up to N tokens. Reports prefill time once and
//!   time per turn.
//!
//! Model/adapter/tokenizer paths are never hardcoded — set env vars, listed
//! per subcommand below.

use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lean::model::{build_rope_tables, forward_chunk_spec, forward_decode_step_argmax, forward_prefill, forward_prefill_suffix, ForwardSpec, GpuModel, KvCache};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// LEAN_GGUF, LEAN_TOKENIZER_DIR, LEAN_LORA_BIN (llm-life's
    /// lora-a-rules-300.bin), --cells (default 256).
    LifeCells {
        #[arg(long, default_value_t = 256)]
        cells: usize,
    },
    /// LEAN_GGUF, LEAN_TOKENIZER_DIR (Qwen3-1.7B), --fixture (path to
    /// fixture_qwen3_1_7b.json, for the long_tools_single input_ids),
    /// --turns (default 5), --max-new (default 48).
    SonosTurns {
        #[arg(long, default_value = "crates/lean/reference/fixture_qwen3_1_7b.json")]
        fixture: String,
        #[arg(long, default_value_t = 5)]
        turns: usize,
        #[arg(long, default_value_t = 48)]
        max_new: usize,
    },
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

/// Reproduces llm-life's `variant_a::rules_prefix`/`cell_prompt` text
/// (crates/llm-life/src/variant_a.rs in the llm-life repo, read-only
/// reference — not a dependency of this crate) so this bench needs no
/// cross-repo dependency. Conway's rule (B3/S23), matching lora-a-rules-300's
/// training rule.
fn rules_prefix() -> String {
    "Cellular automaton, rule B3/S23. Each cell is 0 (dead) or 1 (alive).\n\
     A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n\
     A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n\
     For each cell, answer with one digit: its next state.\n"
        .to_string()
}

fn cell_prompt(neighbors: &[u8; 8], self_state: u8) -> String {
    let mut s = String::from("Neighbors:");
    for &n in neighbors {
        s.push(' ');
        s.push(if n != 0 { '1' } else { '0' });
    }
    s.push_str(" / Self: ");
    s.push(if self_state != 0 { '1' } else { '0' });
    s.push_str(" / Next: ");
    s
}

fn life_cells(cells: usize) -> Result<()> {
    let gguf_path = std::env::var("LEAN_GGUF").context("set LEAN_GGUF")?;
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").context("set LEAN_TOKENIZER_DIR")?;
    let lora_path = std::env::var("LEAN_LORA_BIN").context("set LEAN_LORA_BIN")?;

    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let engine = lean::engine::Engine::new().map_err(|e| anyhow::anyhow!("wgpu engine init: {e}"))?;
    let mut model = GpuModel::load(&engine, &gguf_path, true)?;
    let lora_bytes = std::fs::read(&lora_path)?;
    model.apply_lora(&engine, &lora_bytes)?;

    let dead_id = tokenizer.encode("0", false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids()[0];
    let alive_id = tokenizer.encode("1", false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids()[0];

    let prefix_ids = tokenizer.encode(rules_prefix(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let prefix_len = prefix_ids.len() as u32;

    // Fixed neighbor pattern (a birth case, per docs/runs/2026-09-28-lean-lora.md).
    let cell_ids = tokenizer.encode(cell_prompt(&[1, 1, 1, 0, 0, 0, 0, 0], 0), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let cell_len = cell_ids.len() as u32;

    let max_ctx = prefix_len + cell_len + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    model.pool.reset();
    let mut prefix_cache = KvCache::new(&engine, &model.config, max_ctx);
    let _ = pollster::block_on(forward_prefill(&engine, &model, &mut prefix_cache, &prefix_ids, &cos_buf, &sin_buf, None));
    let snapshot = pollster::block_on(prefix_cache.snapshot(&engine));

    println!("prefix_tokens={prefix_len} cell_tokens={cell_len}");

    // Warmup (shader compile / pipeline cache), not timed.
    for _ in 0..3 {
        model.pool.reset();
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        cache.restore(&engine, &snapshot);
        let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache, &cell_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
        let _ = pollster::block_on(model.lm_head_sliced(&engine, &hidden, cell_len, &[dead_id, alive_id]));
    }

    let start = Instant::now();
    for _ in 0..cells {
        model.pool.reset();
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        cache.restore(&engine, &snapshot);
        let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache, &cell_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
        let _ = pollster::block_on(model.lm_head_sliced(&engine, &hidden, cell_len, &[dead_id, alive_id]));
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("cells={cells} total_s={elapsed:.4} cells_per_s={:.2} ms_per_cell={:.3}", cells as f64 / elapsed, elapsed * 1000.0 / cells as f64);
    Ok(())
}

fn sonos_turns(fixture_path: &str, turns: usize, max_new: usize) -> Result<()> {
    let gguf_path = std::env::var("LEAN_GGUF").context("set LEAN_GGUF")?;

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        input_ids: Vec<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let fixture_json = std::fs::read_to_string(fixture_path).with_context(|| format!("reading {fixture_path}"))?;
    let fixture: Fixture = serde_json::from_str(&fixture_json)?;
    let prompt_ids = fixture.cases.into_iter().find(|c| c.name == "long_tools_single").context("fixture has no long_tools_single case")?.input_ids;
    let prefix_len = prompt_ids.len() as u32;

    // A short follow-up command, tokenized plainly (no chat template; the
    // sonos agent's real per-turn message is short natural-language text,
    // same order of magnitude as this).
    let command_text = "Play jazz in the kitchen and set the volume to 30 percent, then tell me what is now playing there.";

    let engine = lean::engine::Engine::new().map_err(|e| anyhow::anyhow!("wgpu engine init: {e}"))?;
    let model = GpuModel::load(&engine, &gguf_path, true)?;

    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").context("set LEAN_TOKENIZER_DIR")?;
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let command_ids = tokenizer.encode(command_text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let command_len = command_ids.len() as u32;

    let max_ctx = prefix_len + command_len + max_new as u32 + 8;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    model.pool.reset();
    let mut prefix_cache = KvCache::new(&engine, &model.config, max_ctx);
    let prefill_start = Instant::now();
    let _ = pollster::block_on(forward_prefill(&engine, &model, &mut prefix_cache, &prompt_ids, &cos_buf, &sin_buf, None));
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let snapshot = pollster::block_on(prefix_cache.snapshot(&engine));

    println!("prompt_tokens={prefix_len} command_tokens={command_len} prefill_ms={prefill_ms:.1} prefill_ms_per_tok={:.3}", prefill_ms / prefix_len as f64);

    let mut turn_ms = Vec::with_capacity(turns);
    for turn in 0..turns {
        model.pool.reset();
        let mut cache = KvCache::new(&engine, &model.config, max_ctx);
        cache.restore(&engine, &snapshot);

        let turn_start = Instant::now();
        let mut logits = pollster::block_on(forward_prefill_suffix(&engine, &model, &mut cache, &command_ids, &cos_buf, &sin_buf, None));
        let mut next_id = argmax(&logits);
        for _ in 0..max_new {
            next_id = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next_id, &cos_buf, &sin_buf, None));
        }
        let _ = &mut logits;
        let elapsed_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        turn_ms.push(elapsed_ms);
        println!("turn={turn} ms={elapsed_ms:.1} ms_per_new_token={:.3}", elapsed_ms / (max_new as f64 + 1.0));
    }
    let mut sorted = turn_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("median_turn_ms={:.1}", sorted[sorted.len() / 2]);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::LifeCells { cells } => life_cells(cells),
        Cmd::SonosTurns { fixture, turns, max_new } => sonos_turns(&fixture, turns, max_new),
    }
}
