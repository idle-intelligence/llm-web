//! KV cache snapshot/restore parity + timing, against the same fixture the
//! other `--ignored` tests use.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release -- --ignored kv_snapshot
//! ```
//!
//! Checks: (1) prefill(prefix) -> snapshot -> restore into a fresh cache ->
//! prefill(suffix) -> greedy generate reproduces the exact token ids of
//! prefill(prefix+suffix) -> greedy generate, run from scratch; (2)
//! export/import (`to_bytes`/`from_bytes`) round-trips a snapshot
//! byte-for-byte; (3) timing of a full ~1000-token prefill vs a
//! snapshot-export + import + restore round trip of the same cache (the
//! time a resident-prefix image saves a consumer that keys a KV snapshot by
//! its own prompt/tool-schema hash - see the engine consumer survey's gap
//! #1). (3) is timing-only, like the wasm harness's own `long_1000` case -
//! `attn_prefill.wgsl`'s `MAX_SEQ` cap means a from-scratch ~1000-token
//! prefill's *logits* are not asserted correct here, only timed.

use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_decode_step, forward_prefill, forward_prefill_suffix, GpuModel, KvCache, KvSnapshot};
use serde::Deserialize;
use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

fn mark(s: &str) {
    eprintln!("[mark] {s}");
    let _ = std::io::stderr().flush();
}

#[derive(Deserialize)]
struct Case {
    #[allow(dead_code)]
    name: String,
    input_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
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

fn greedy_continue(engine: &Engine, model: &GpuModel, cache: &mut KvCache, mut logits: Vec<f32>, cos: &wgpu::Buffer, sin: &wgpu::Buffer, n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = argmax(&logits);
        out.push(id);
        logits = pollster::block_on(forward_decode_step(engine, model, cache, id, cos, sin, None));
    }
    out
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn kv_snapshot_restore_matches_full_prefill_and_round_trips() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");

    let fixture_json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixture.json")).expect("reading fixture.json");
    let fixture: Fixture = serde_json::from_str(&fixture_json).expect("parsing fixture.json");
    // The "long" case (86 tokens) gives both halves a real chunk of context.
    let all_ids = fixture.cases.iter().find(|c| c.input_ids.len() == 86).expect("fixture has an 86-token case").input_ids.clone();
    let split = all_ids.len() / 2;
    let prefix = &all_ids[..split];
    let suffix = &all_ids[split..];
    let n_new = 12;

    mark("loading engine");
    let engine = Engine::new().expect("wgpu engine init");
    mark("loading model");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let max_ctx = all_ids.len() as u32 + n_new as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // Path A: prefill(prefix+suffix) from scratch, then greedy generate.
    mark("path A: prefill(all)");
    model.pool.reset();
    let mut cache_a = KvCache::new(&engine, &model.config, max_ctx);
    let logits_a = pollster::block_on(forward_prefill(&engine, &model, &mut cache_a, &all_ids, &cos_buf, &sin_buf, None));
    mark("path A: greedy_continue");
    let tokens_a = greedy_continue(&engine, &model, &mut cache_a, logits_a, &cos_buf, &sin_buf, n_new);
    mark("path A: done");

    // Path B: prefill(prefix) -> snapshot -> restore into a fresh cache ->
    // prefill(suffix) -> greedy generate.
    model.pool.reset();
    mark("path B: prefill(prefix)");
    let mut cache_prefix = KvCache::new(&engine, &model.config, max_ctx);
    let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache_prefix, prefix, &cos_buf, &sin_buf, None));
    mark("path B: snapshot()");
    let snapshot = pollster::block_on(cache_prefix.snapshot(&engine));
    mark("path B: snapshot done");
    assert_eq!(snapshot.kv_len as usize, prefix.len());

    model.pool.reset(); // new KvCache instance -> pool.rs's bug note applies
    mark("path B: restore()");
    let mut cache_b = KvCache::new(&engine, &model.config, max_ctx);
    cache_b.restore(&engine, &snapshot);
    mark("path B: restore done");
    assert_eq!(cache_b.kv_len as usize, prefix.len());
    mark("path B: forward_prefill_suffix()");
    let logits_b = pollster::block_on(forward_prefill_suffix(&engine, &model, &mut cache_b, suffix, &cos_buf, &sin_buf, None));
    mark("path B: greedy_continue");
    let tokens_b = greedy_continue(&engine, &model, &mut cache_b, logits_b, &cos_buf, &sin_buf, n_new);
    mark("path B: done");

    assert_eq!(tokens_a, tokens_b, "restore+prefill(suffix) must reproduce prefill(prefix+suffix)'s continuation exactly");

    // Export/import round trip.
    let bytes = snapshot.to_bytes();
    let snapshot2 = KvSnapshot::from_bytes(&bytes).expect("parsing exported snapshot bytes");
    assert_eq!(snapshot.kv_len, snapshot2.kv_len);
    assert_eq!(snapshot.num_kv_heads, snapshot2.num_kv_heads);
    assert_eq!(snapshot.head_dim, snapshot2.head_dim);
    assert_eq!(snapshot.num_layers, snapshot2.num_layers);
    assert_eq!(snapshot.k, snapshot2.k, "k round trip must be identical");
    assert_eq!(snapshot.v, snapshot2.v, "v round trip must be identical");

    // Restoring the round-tripped (imported) snapshot must reproduce the
    // same continuation too, not just have identical float contents.
    model.pool.reset();
    let mut cache_c = KvCache::new(&engine, &model.config, max_ctx);
    cache_c.restore(&engine, &snapshot2);
    let logits_c = pollster::block_on(forward_prefill_suffix(&engine, &model, &mut cache_c, suffix, &cos_buf, &sin_buf, None));
    let tokens_c = greedy_continue(&engine, &model, &mut cache_c, logits_c, &cos_buf, &sin_buf, n_new);
    assert_eq!(tokens_a, tokens_c, "generation from an imported snapshot must match too");

    eprintln!("[kv_snapshot] prefix={} suffix={} tokens_match=true export_bytes={}", prefix.len(), suffix.len(), bytes.len());
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn kv_snapshot_timing_1000_token_prefix() {
    let gguf_path = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");

    const SENTENCES: [&str; 10] = [
        "The history of the Roman Empire spans many centuries of political change.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
        "The printing press revolutionized the spread of information across Europe.",
        "Mount Everest is the tallest mountain above sea level on the planet.",
        "The French Revolution reshaped the political landscape of eighteenth century Europe.",
        "DNA carries the genetic instructions used in the growth of living organisms.",
        "The Great Wall of China stretches thousands of kilometers across northern China.",
        "Volcanic eruptions can reshape landscapes and affect climate for years afterward.",
    ];
    let mut text = String::from("Summarize the following notes in one paragraph.\n\n");
    for i in 0..90 {
        text.push_str(SENTENCES[i % SENTENCES.len()]);
        text.push(' ');
    }
    let ids = tokenizer.encode(text, false).expect("tokenizer encode failed").get_ids().to_vec();
    eprintln!("[kv_snapshot_timing] prompt tokenized to {} ids", ids.len());

    let engine = Engine::new().expect("wgpu engine init");
    let model = GpuModel::load(&engine, &gguf_path, true).expect("loading model");
    let max_ctx = ids.len() as u32 + 8;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");

    // Full prefill from scratch (timing only - attn_prefill.wgsl's MAX_SEQ
    // cap means logits at this length are not asserted correct, matching
    // the wasm harness's own long_1000 case).
    model.pool.reset();
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let full_start = Instant::now();
    let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
    let full_prefill_ms = full_start.elapsed().as_secs_f64() * 1000.0;

    // Snapshot the now-populated cache, export to bytes, import, and
    // restore into a fresh cache - the "reuse a resident prefix" path a
    // consumer takes instead of re-running the prefill above.
    let snap_start = Instant::now();
    let snapshot = pollster::block_on(cache.snapshot(&engine));
    let snapshot_ms = snap_start.elapsed().as_secs_f64() * 1000.0;

    let export_start = Instant::now();
    let bytes = snapshot.to_bytes();
    let export_ms = export_start.elapsed().as_secs_f64() * 1000.0;

    let import_start = Instant::now();
    let snapshot2 = KvSnapshot::from_bytes(&bytes).expect("parsing exported snapshot bytes");
    let import_ms = import_start.elapsed().as_secs_f64() * 1000.0;

    model.pool.reset();
    let restore_start = Instant::now();
    let mut cache2 = KvCache::new(&engine, &model.config, max_ctx);
    cache2.restore(&engine, &snapshot2);
    let restore_ms = restore_start.elapsed().as_secs_f64() * 1000.0;

    let round_trip_ms = snapshot_ms + export_ms + import_ms + restore_ms;
    eprintln!(
        "[kv_snapshot_timing] seq={} bytes={} full_prefill_ms={:.2} snapshot_ms={:.2} export_ms={:.2} import_ms={:.2} restore_ms={:.2} round_trip_ms={:.2} speedup={:.1}x",
        ids.len(),
        bytes.len(),
        full_prefill_ms,
        snapshot_ms,
        export_ms,
        import_ms,
        restore_ms,
        round_trip_ms,
        full_prefill_ms / round_trip_ms
    );
}
