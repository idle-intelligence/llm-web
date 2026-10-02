//! `GpuModel::embed_head_sliced` against a CPU reference: the hidden states
//! of one prompt, read back, dotted with the CPU-dequantized
//! `token_embd.weight` rows (`gguf::dequantize_for`). Also checks that
//! swapping one LoRA adapter for another, with no pool reset by the caller,
//! gives the second adapter's numbers (`apply_lora` resets the pool).
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_LORA_BIN=/path/to/lora-a-rules-300.bin \
//! LEAN_LORA_BIN2=/path/to/lora-a-norules-300.bin \
//! cargo test -p lean --release -- --ignored embed_head
//! ```

use lean::engine::Engine;
use lean::gguf::{dequantize_for, GgufReader};
use lean::model::{build_rope_tables, forward_chunk_spec, ForwardSpec, GpuModel, KvCache};

// "Cellular automaton, rule B3/S23. ..." prefix + one cell prompt, the
// token ids of `reference/fixture_lora.json`'s prompt (same tokenizer).
fn fixture_ids() -> Vec<u32> {
    #[derive(serde::Deserialize)]
    struct F {
        input_ids: Vec<u32>,
    }
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture_lora.json")).unwrap();
    serde_json::from_str::<F>(&s).unwrap().input_ids
}

const IDS: [u32; 2] = [15, 16]; // "0", "1" in the Qwen2 tokenizer

fn forward(engine: &Engine, model: &GpuModel, ids: &[u32]) -> (wgpu::Buffer, u32) {
    let t = ids.len() as u32;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, t as usize + 4);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    model.pool.reset();
    let mut cache = KvCache::new(engine, &model.config, t + 4);
    let hidden = pollster::block_on(forward_chunk_spec(engine, model, &mut cache, ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
    (hidden, t)
}

#[test]
#[ignore = "needs LEAN_GGUF on disk; never committed to this repo"]
fn embed_head_matches_cpu_dequant() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let hidden_size = model.config.hidden_size;

    let mut reader = GgufReader::open(std::io::BufReader::new(std::fs::File::open(&gguf_path).unwrap())).unwrap();
    let info = reader.tensor_info("token_embd.weight").unwrap().clone();
    let all = reader.tensor_data("token_embd.weight").unwrap();
    let row_bytes = all.len() / model.config.vocab_size;
    let rows: Vec<Vec<f32>> = IDS
        .iter()
        .map(|&id| dequantize_for(info.dtype(), &all[id as usize * row_bytes..(id as usize + 1) * row_bytes], hidden_size))
        .collect();

    let ids = fixture_ids();
    let (hidden, t) = forward(&engine, &model, &ids);
    let got = pollster::block_on(model.embed_head_sliced(&engine, &hidden, t, &IDS));
    let h = pollster::block_on(engine.read_buffer(&hidden, t as usize * hidden_size));
    let mut max_diff = 0f32;
    for r in 0..t as usize {
        for (k, row) in rows.iter().enumerate() {
            let want: f32 = (0..hidden_size).map(|d| h[r * hidden_size + d] * row[d]).sum();
            max_diff = max_diff.max((got[r * 2 + k] - want).abs());
        }
    }
    eprintln!("[embed_head] {t} rows, max |gpu - cpu| = {max_diff:.3e}");
    assert!(max_diff < 1e-3, "embed_head_sliced diverges from the CPU dequant reference: {max_diff}");
}

#[test]
#[ignore = "needs LEAN_GGUF, LEAN_LORA_BIN and LEAN_LORA_BIN2 on disk; never committed to this repo"]
fn adapter_swap_uses_the_new_adapter() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let a = std::fs::read(std::env::var("LEAN_LORA_BIN").expect("set LEAN_LORA_BIN")).unwrap();
    let b = std::fs::read(std::env::var("LEAN_LORA_BIN2").expect("set LEAN_LORA_BIN2")).unwrap();
    let engine = Engine::new().expect("wgpu engine init");
    let ids = fixture_ids();
    let t = ids.len() as u32;
    let mut model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, t as usize + 4);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // One cache for the whole test, no caller-side pool reset: the only
    // thing that changes between the two forwards is the adapter.
    let run = |model: &GpuModel, cache: &mut KvCache| -> Vec<f32> {
        cache.kv_len = 0;
        let hidden = pollster::block_on(forward_chunk_spec(&engine, model, cache, &ids, &cos_buf, &sin_buf, &ForwardSpec::default()));
        pollster::block_on(model.embed_head_sliced(&engine, &hidden, t, &IDS))
    };

    let mut cache = KvCache::new(&engine, &model.config, t + 4);
    model.apply_lora(&engine, &a).unwrap();
    // Twice: the first call allocates every pool buffer (bumping the pool
    // generation), so only from the second call on are the cached bind
    // groups reused as-is, which is the state an adapter swap meets.
    let _ = run(&model, &mut cache);
    let with_a = run(&model, &mut cache);
    model.apply_lora(&engine, &b).unwrap();
    let swapped_to_b = run(&model, &mut cache);

    let mut fresh = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let mut cache2 = KvCache::new(&engine, &fresh.config, t + 4);
    fresh.apply_lora(&engine, &b).unwrap();
    let only_b = run(&fresh, &mut cache2);

    let last = |v: &[f32]| (v[(t as usize - 1) * 2], v[(t as usize - 1) * 2 + 1]);
    eprintln!("[swap] a={:?} a->b={:?} b alone={:?}", last(&with_a), last(&swapped_to_b), last(&only_b));
    let diff = swapped_to_b.iter().zip(&only_b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(diff < 1e-4, "after apply_lora(b), the model still runs adapter a: max diff {diff}");
}
