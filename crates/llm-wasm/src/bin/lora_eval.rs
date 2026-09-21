//! Native check: does `crate::lora`'s runtime LoRA path (applied straight
//! onto the base Q4 `LlmModel`, no offline GGUF merge) reproduce a
//! variant-A adapter's own training-time accuracy?
//!
//! llm-wasm cannot depend on llm-life (which owns the prompt text and
//! per-cell packing, `crates/llm-life/src/variant_a.rs`) without a
//! dependency cycle — llm-life depends on llm-wasm, not the other way
//! around — and this task's scope keeps llm-wasm/llm-life edits in
//! separate worktrees. So this binary duplicates only the handful of pure
//! string-formatting/packing helpers it needs (`cell_prompt`, the two
//! prefixes, and a from-scratch `pack_chunk` matching
//! `variant_a::pack_chunk`'s block-diagonal mask), kept intentionally
//! minimal, to drive the same forward `llm-life`'s `LifeEngine::
//! loadAdapter`/`stepChunkA` exercise in the browser. See llm-life's
//! `docs/runs/2026-09-20-runtime-lora.md` for the numbers this produces and
//! `docs/runs/2026-09-20-merge.md` for the offline-merge numbers it's
//! checked against (512/512, 509/512 — the adapters' own best-checkpoint
//! accuracy, `tools/merge/eval_512.py`'s Q8_0 columns).
//!
//! Usage:
//! ```text
//! cargo run --release -p llm-wasm --bin lora-eval --features native -- \
//!   --gguf <base Q4_0 qwen2.5-0.5b-instruct.gguf> \
//!   --tokenizer <tokenizer.json> \
//!   --adapter <lora-a-norules-300.bin> --norules
//! ```

use anyhow::{ensure, Context, Result};
use burn::backend::wgpu::WgpuDevice;
use clap::Parser;
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::lora::LoraAdapter;
use llm_wasm::model::{logits_to_vec, ForwardSpec};
use llm_wasm::tokenizer::Tokenizer;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    /// Must match how the adapter was trained (`lora-a-norules-*.bin` vs
    /// `lora-a-rules-*.bin`).
    #[arg(long)]
    norules: bool,
    #[arg(long, default_value = "64")]
    chunk_cells: usize,
    #[arg(long, default_value = "16")]
    grid_size: usize,
}

/// Copy of `variant_a::cell_prompt` (llm-life).
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

/// Copy of `variant_a::norules_prefix` (llm-life).
fn norules_prefix() -> String {
    "For each cell, answer with one digit: its next state.\n".to_string()
}

/// Copy of `variant_a::rules_prefix` (llm-life), specialized to Conway's
/// Life (`B3/S23`) — the only rule the `-300` adapters were trained on.
fn rules_prefix() -> String {
    "Cellular automaton, rule B3/S23. Each cell is 0 (dead) or 1 (alive).\n\
     A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n\
     A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n\
     For each cell, answer with one digit: its next state.\n"
        .to_string()
}

fn life_next(alive: bool, n: usize) -> bool {
    if alive {
        n == 2 || n == 3
    } else {
        n == 3
    }
}

/// One block-diagonal chunk against a resident prefix of `prefix_len` —
/// matches `variant_a::pack_chunk`'s mask shape exactly (prefix in full,
/// causal within each cell's own block, no cross-block attention).
struct Chunk {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    allowed: Vec<bool>,
    /// Row (within the chunk) whose logits answer each case, in the same
    /// order as the input `cases`.
    answer_rows: Vec<usize>,
}

fn pack_chunk(tok: &Tokenizer, cases: &[([u8; 8], u8)], prefix_len: usize) -> Result<Chunk> {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut starts = Vec::with_capacity(cases.len());
    let mut lens = Vec::with_capacity(cases.len());
    for &(nb, s) in cases {
        let text = format!("\n{}", cell_prompt(&nb, s));
        let ids = tok.encode(&text, false)?;
        starts.push(tokens.len());
        lens.push(ids.len());
        positions.extend((0..ids.len()).map(|p| (prefix_len + p) as u32));
        tokens.extend_from_slice(&ids);
    }
    let t = tokens.len();
    let kv = prefix_len + t;
    let mut allowed = vec![false; t * kv];
    for (&start, &len) in starts.iter().zip(&lens) {
        for i in 0..len {
            let row = (start + i) * kv;
            allowed[row..row + prefix_len].fill(true);
            let base = row + prefix_len + start;
            allowed[base..base + i + 1].fill(true);
        }
    }
    let answer_rows = starts.iter().zip(&lens).map(|(&s, &l)| s + l - 1).collect();
    Ok(Chunk { tokens, positions, allowed, answer_rows })
}

/// p(alive) from a `[T, 2]` (flattened) logits slice's row `row`, column 0
/// = dead, column 1 = alive — matches `variant_a::p_alive_chunk`.
fn p_alive(logits: &[f32], row: usize) -> f32 {
    let (d, a) = (logits[row * 2], logits[row * 2 + 1]);
    let m = d.max(a);
    let (ed, ea) = ((d - m).exp(), (a - m).exp());
    ea / (ed + ea)
}

fn neighbors_of(grid: &[u8], n_side: usize, x: usize, y: usize) -> [u8; 8] {
    let mut nb = [0u8; 8];
    let mut i = 0;
    for dy in [-1i64, 0, 1] {
        for dx in [-1i64, 0, 1] {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < n_side && (ny as usize) < n_side {
                nb[i] = grid[ny as usize * n_side + nx as usize];
            }
            i += 1;
        }
    }
    nb
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = WgpuDevice::default();

    let tok = Tokenizer::from_json(&std::fs::read(&args.tokenizer).context("read tokenizer.json")?)?;
    let prefix_text = if args.norules { norules_prefix() } else { rules_prefix() };
    let prefix = tok.encode(&prefix_text, false)?;
    let one = |s: &str| -> Result<u32> {
        let ids = tok.encode(s, false)?;
        ensure!(ids.len() == 1, "'{s}' is not a single token: {ids:?}");
        Ok(ids[0])
    };
    let dead = one("0")?;
    let alive = one("1")?;

    let file = std::fs::File::open(&args.gguf).context("open gguf")?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;
    let parts = loader.load_deferred(&device)?;
    drop(loader); // free the GGUF reader before finalizing GPU tensors
    let mut model = parts.finalize(&device)?;
    println!("model: {} layers, hidden {}", model.config().num_layers, model.config().hidden_size);

    let adapter_bytes = std::fs::read(&args.adapter).context("read adapter")?;
    let adapter = LoraAdapter::from_bytes(&adapter_bytes, model.config().num_layers, &device).context("parse adapter")?;
    model.apply_lora(adapter).context("apply adapter")?;
    println!("runtime LoRA adapter applied: {} ({} bytes)", args.adapter.display(), adapter_bytes.len());

    let head = model.head_slice(&[dead, alive])?;
    let prefix_len = prefix.len();
    // Generous upper bound for one "\nNeighbors: 0 0 0 0 0 0 0 0 / Self: 0 / Next: " block.
    let max_cell_tokens = 32;
    let mut cache = model.new_cache(prefix_len + args.chunk_cells * max_cell_tokens);
    model.forward_hidden(&prefix, &mut cache)?;

    // --- 1. Exhaustive 512-case (neighbors, self) lookup. ---
    let mut correct = 0usize;
    let mut lookup_secs = 0.0;
    let mut n_chunks = 0usize;
    for first in (0..512).step_by(args.chunk_cells) {
        let ids: Vec<usize> = (first..(first + args.chunk_cells).min(512)).collect();
        let cases: Vec<([u8; 8], u8)> = ids
            .iter()
            .map(|&k| {
                let nb: [u8; 8] = std::array::from_fn(|b| ((k >> b) & 1) as u8);
                (nb, ((k >> 8) & 1) as u8)
            })
            .collect();
        let chunk = pack_chunk(&tok, &cases, prefix_len)?;
        let t = chunk.tokens.len();
        let spec = ForwardSpec::default()
            .with_positions(chunk.positions.clone())
            .with_allowed(&chunk.allowed, t, prefix_len + t, &device);

        let start = Instant::now();
        let resident = cache.snapshot();
        let hidden = model.forward_hidden_spec(&chunk.tokens, &mut cache, &spec)?;
        let logits = model.lm_head_sliced(hidden, &head);
        let logits = logits_to_vec(logits)?;
        cache.restore(resident);
        lookup_secs += start.elapsed().as_secs_f64();
        n_chunks += 1;

        for (&k, &row) in ids.iter().zip(&chunk.answer_rows) {
            let nb: [u8; 8] = std::array::from_fn(|b| ((k >> b) & 1) as u8);
            let self_state = (k >> 8) & 1;
            let n = nb.iter().filter(|&&b| b != 0).count();
            let target = life_next(self_state != 0, n) as u8;
            let pred = (p_alive(&logits, row) > 0.5) as u8;
            if pred == target {
                correct += 1;
            }
        }
    }
    println!(
        "512-case lookup: {correct}/512 = {:.4}  ({:.1} ms/chunk of {}, {n_chunks} chunks, {:.1} ms total)",
        correct as f64 / 512.0,
        lookup_secs * 1000.0 / n_chunks as f64,
        args.chunk_cells,
        lookup_secs * 1000.0,
    );

    // --- 2. One generation on a grid_size^2 glider, scored by IoU. ---
    let n_side = args.grid_size;
    let mut grid = vec![0u8; n_side * n_side];
    let (cx, cy) = (n_side / 2, n_side / 2);
    for (dx, dy) in [(1usize, 0usize), (2, 1), (0, 2), (1, 2), (2, 2)] {
        let (x, y) = (cx + dx, cy + dy);
        if x < n_side && y < n_side {
            grid[y * n_side + x] = 1;
        }
    }

    let n_cells = n_side * n_side;
    let mut p = vec![0.0f32; n_cells];
    let mut gen_secs = 0.0;
    for first in (0..n_cells).step_by(args.chunk_cells) {
        let ids: Vec<usize> = (first..(first + args.chunk_cells).min(n_cells)).collect();
        let cases: Vec<([u8; 8], u8)> = ids
            .iter()
            .map(|&c| {
                let (x, y) = (c % n_side, c / n_side);
                (neighbors_of(&grid, n_side, x, y), grid[c])
            })
            .collect();
        let chunk = pack_chunk(&tok, &cases, prefix_len)?;
        let t = chunk.tokens.len();
        let spec = ForwardSpec::default()
            .with_positions(chunk.positions.clone())
            .with_allowed(&chunk.allowed, t, prefix_len + t, &device);

        let start = Instant::now();
        let resident = cache.snapshot();
        let hidden = model.forward_hidden_spec(&chunk.tokens, &mut cache, &spec)?;
        let logits = model.lm_head_sliced(hidden, &head);
        let logits = logits_to_vec(logits)?;
        cache.restore(resident);
        gen_secs += start.elapsed().as_secs_f64();

        for (&c, &row) in ids.iter().zip(&chunk.answer_rows) {
            p[c] = p_alive(&logits, row);
        }
    }

    let mut intersection = 0usize;
    let mut union = 0usize;
    for c in 0..n_cells {
        let (x, y) = (c % n_side, c / n_side);
        let nb = neighbors_of(&grid, n_side, x, y);
        let n = nb.iter().filter(|&&b| b != 0).count();
        let truth = life_next(grid[c] != 0, n);
        let model_alive = p[c] > 0.5;
        if truth || model_alive {
            union += 1;
        }
        if truth && model_alive {
            intersection += 1;
        }
    }
    let iou = if union == 0 { 1.0 } else { intersection as f64 / union as f64 };
    println!("{n_side}x{n_side} generation: IoU {iou:.4}  ({:.1} ms)", gen_secs * 1000.0);

    Ok(())
}
