//! Runtime LoRA + `forward_chunk_spec` (caller-supplied positions/mask) +
//! sliced lm-head parity, against `reference/gen_fixture_lora.py`'s
//! transformers-GGUF-loaded-base-plus-hand-applied-LoRA-deltas fixture.
//! Covers consumer-survey gap items 4 and 5 together, since llm-life's own
//! call shape (`LifeEngine::step_ids_a` in the Burn engine) always uses
//! them together: a chunk forward against a resident/empty prefix, with
//! LoRA applied to q/k/v/o, read back through a sliced lm-head.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! LEAN_LORA_BIN=/path/to/lora-a-rules-300.bin \
//! cargo test -p lean --release -- --ignored lora_parity
//! ```

use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_chunk_spec, ForwardSpec, GpuModel, KvCache};
use serde::Deserialize;

#[derive(Deserialize)]
struct Logits {
    dead: f32,
    alive: f32,
}

#[derive(Deserialize)]
struct Fixture {
    input_ids: Vec<u32>,
    dead_token_id: u32,
    alive_token_id: u32,
    base_logits: Logits,
    lora_logits: Logits,
}

/// Same tolerance as `fixture_parity.rs`'s top-20 check: this crate's own
/// Q4_0 GPU dequant kernel and transformers' GGUF dequant-on-load are two
/// independent implementations of the same block math, not bit-identical.
const TOL: f32 = 2e-2;

#[test]
#[ignore = "needs LEAN_GGUF, LEAN_TOKENIZER_DIR and a real LLMLIFE2 .bin on disk; never committed to this repo"]
fn lora_and_sliced_head_match_reference() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let lora_path = std::env::var("LEAN_LORA_BIN").expect("set LEAN_LORA_BIN to run this test");

    let fixture_json =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_lora.json")).expect("reading fixture_lora.json (run gen_fixture_lora.py first)");
    let fixture: Fixture = serde_json::from_str(&fixture_json).expect("parsing fixture_lora.json");

    let engine = Engine::new().expect("wgpu engine init");
    let mut model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let max_ctx = fixture.input_ids.len() as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let t = fixture.input_ids.len() as u32;
    let ids = [fixture.dead_token_id, fixture.alive_token_id];

    // Base model (no LoRA), through the generic chunk path with default
    // ForwardSpec (no caller-supplied positions/mask -> plain causal
    // continuation of an empty resident prefix, i.e. an ordinary prefill).
    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache, &fixture.input_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    assert_eq!(cache.kv_len, t, "forward_chunk_spec must advance kv_len by the chunk length");
    let logits = pollster::block_on(model.lm_head_sliced(&engine, &hidden, t, &ids));
    let last = &logits[((t - 1) * 2) as usize..];
    eprintln!("[lora_parity] base: dead={:.6} alive={:.6} (fixture dead={:.6} alive={:.6})", last[0], last[1], fixture.base_logits.dead, fixture.base_logits.alive);
    assert!((last[0] - fixture.base_logits.dead).abs() < TOL, "base dead logit mismatch: got {} want {}", last[0], fixture.base_logits.dead);
    assert!((last[1] - fixture.base_logits.alive).abs() < TOL, "base alive logit mismatch: got {} want {}", last[1], fixture.base_logits.alive);

    // Same chunk, LoRA applied (switchable without a base reload).
    let lora_bytes = std::fs::read(&lora_path).expect("reading LEAN_LORA_BIN");
    model.apply_lora(&engine, &lora_bytes).expect("apply_lora");
    assert!(model.has_lora());

    model.pool.reset();
    let mut cache2 = KvCache::new(&engine, &model.config, max_ctx);
    let hidden2 = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache2, &fixture.input_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    let logits2 = pollster::block_on(model.lm_head_sliced(&engine, &hidden2, t, &ids));
    let last2 = &logits2[((t - 1) * 2) as usize..];
    eprintln!("[lora_parity] +lora: dead={:.6} alive={:.6} (fixture dead={:.6} alive={:.6})", last2[0], last2[1], fixture.lora_logits.dead, fixture.lora_logits.alive);
    assert!((last2[0] - fixture.lora_logits.dead).abs() < TOL, "lora dead logit mismatch: got {} want {}", last2[0], fixture.lora_logits.dead);
    assert!((last2[1] - fixture.lora_logits.alive).abs() < TOL, "lora alive logit mismatch: got {} want {}", last2[1], fixture.lora_logits.alive);

    // clear_lora must return to the base model's numbers with no reload.
    model.clear_lora();
    assert!(!model.has_lora());
    model.pool.reset();
    let mut cache3 = KvCache::new(&engine, &model.config, max_ctx);
    let hidden3 = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache3, &fixture.input_ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    let logits3 = pollster::block_on(model.lm_head_sliced(&engine, &hidden3, t, &ids));
    let last3 = &logits3[((t - 1) * 2) as usize..];
    assert!((last3[0] - fixture.base_logits.dead).abs() < TOL, "clear_lora dead logit mismatch: got {} want {}", last3[0], fixture.base_logits.dead);
    assert!((last3[1] - fixture.base_logits.alive).abs() < TOL, "clear_lora alive logit mismatch: got {} want {}", last3[1], fixture.base_logits.alive);
}

/// llm-life variant A's actual mechanism: several per-cell blocks packed
/// into one sequence against a shared resident prefix with a block-diagonal
/// mask and per-block RoPE restart, must give the *same* answer per cell as
/// forwarding each cell alone against the same prefix (`KvCache::snapshot`/
/// `restore` rewinding between cells, exactly `LifeEngine::step_ids_a`'s
/// pattern in the Burn engine). No PyTorch reference needed here: this is
/// the packing's own numerical-invisibility property (model.rs's
/// `forward_chunk_spec` doc comment), checked directly.
#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn packed_block_diagonal_matches_per_cell_forward() {
    use lean::model::pack_bool_mask;

    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");

    let prefix_text = "Cellular automaton, rule B3/S23. Each cell is 0 (dead) or 1 (alive).\n\
        A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n\
        A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n\
        For each cell, answer with one digit: its next state.\n";
    let prefix_ids = tokenizer.encode(prefix_text, false).unwrap().get_ids().to_vec();

    let cell_text = |neighbors: &[u8; 8], self_state: u8| -> String {
        let mut s = String::from("Neighbors:");
        for &n in neighbors {
            s.push(' ');
            s.push(if n != 0 { '1' } else { '0' });
        }
        s.push_str(" / Self: ");
        s.push(if self_state != 0 { '1' } else { '0' });
        s.push_str(" / Next: ");
        s
    };
    let cell_a = tokenizer.encode(cell_text(&[1, 1, 1, 0, 0, 0, 0, 0], 0), false).unwrap().get_ids().to_vec();
    let cell_b = tokenizer.encode(cell_text(&[0, 0, 0, 0, 0, 0, 0, 0], 1), false).unwrap().get_ids().to_vec();
    let dead_id = tokenizer.encode("0", false).unwrap().get_ids()[0];
    let alive_id = tokenizer.encode("1", false).unwrap().get_ids()[0];
    let ids = [dead_id, alive_id];

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let max_ctx = (prefix_ids.len() + cell_a.len().max(cell_b.len()) * 2 + 8) as u32;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // Prefill the shared prefix once, resident.
    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let _ = pollster::block_on(lean::model::forward_prefill(&engine, &model, &mut cache, &prefix_ids, &cos_buf, &sin_buf, None));
    let prefix_snapshot = pollster::block_on(cache.snapshot(&engine));
    let prefix_len = prefix_snapshot.kv_len;

    // Per-cell forward: rewind to the resident prefix, forward one cell's
    // tokens alone (default causal continuation), read its last-row logits.
    let forward_one = |cache: &mut KvCache, cell: &[u32]| -> (f32, f32) {
        cache.restore(&engine, &prefix_snapshot);
        let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, cache, cell, &cos_buf, &sin_buf, &ForwardSpec::default()));
        let logits = pollster::block_on(model.lm_head_sliced(&engine, &hidden, cell.len() as u32, &ids));
        let last = &logits[(cell.len() - 1) * 2..];
        (last[0], last[1])
    };
    let (a_dead_alone, a_alive_alone) = forward_one(&mut cache, &cell_a);
    let (b_dead_alone, b_alive_alone) = forward_one(&mut cache, &cell_b);

    // Packed forward: both cells in one chunk, block-diagonal mask, each
    // block's positions restarting at prefix_len.
    let t = cell_a.len() + cell_b.len();
    let cols = prefix_len as usize + t;
    let mut allowed = vec![false; t * cols];
    let mut positions = Vec::with_capacity(t);
    let mut tokens = Vec::with_capacity(t);
    let mut block_start = 0usize;
    for block in [&cell_a, &cell_b] {
        for (i, &tok) in block.iter().enumerate() {
            tokens.push(tok);
            positions.push(prefix_len + i as u32);
            let row = block_start + i;
            for j in 0..prefix_len as usize {
                allowed[row * cols + j] = true;
            }
            for j in 0..=i {
                allowed[row * cols + prefix_len as usize + block_start + j] = true;
            }
        }
        block_start += block.len();
    }
    let mask_bits = pack_bool_mask(&allowed, t, cols);
    let spec = ForwardSpec::default().with_positions(positions).with_allowed_bits(mask_bits);

    cache.restore(&engine, &prefix_snapshot);
    let hidden = pollster::block_on(forward_chunk_spec(&engine, &model, &mut cache, &tokens, &cos_buf, &sin_buf, &spec));
    let logits = pollster::block_on(model.lm_head_sliced(&engine, &hidden, t as u32, &ids));
    let a_row = cell_a.len() - 1;
    let b_row = cell_a.len() + cell_b.len() - 1;
    let (a_dead_packed, a_alive_packed) = (logits[a_row * 2], logits[a_row * 2 + 1]);
    let (b_dead_packed, b_alive_packed) = (logits[b_row * 2], logits[b_row * 2 + 1]);

    eprintln!("[packed] cell A alone=({a_dead_alone:.6},{a_alive_alone:.6}) packed=({a_dead_packed:.6},{a_alive_packed:.6})");
    eprintln!("[packed] cell B alone=({b_dead_alone:.6},{b_alive_alone:.6}) packed=({b_dead_packed:.6},{b_alive_packed:.6})");

    assert!((a_dead_alone - a_dead_packed).abs() < 1e-3, "cell A dead logit: packing must be numerically invisible");
    assert!((a_alive_alone - a_alive_packed).abs() < 1e-3, "cell A alive logit: packing must be numerically invisible");
    assert!((b_dead_alone - b_dead_packed).abs() < 1e-3, "cell B dead logit: packing must be numerically invisible");
    assert!((b_alive_alone - b_alive_packed).abs() < 1e-3, "cell B alive logit: packing must be numerically invisible");
}
