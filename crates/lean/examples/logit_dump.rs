//! Writes the logits of a fixed prompt (prefill, then greedy decode steps) to
//! a file, so two builds can be compared on the same GGUF.
//!
//! `LEAN_GGUF=... cargo run --release -p lean --example logit_dump -- <out.bin> [--len 100] [--steps 16] [--repeat 1] [--verify] [--naive]`
//!
//! Prints the greedy token ids and an FNV-1a hash of the raw logits bytes.
//! The file is the concatenation of the prefill logits and each decode
//! step's logits as little-endian f32.

use lean_core::engine::Engine;
use lean_core::model::{build_rope_tables, forward_decode_step, forward_prefill, GpuModel, KvCache};

const PROMPT: [u32; 36] = [
    151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
];

fn arg(args: &[String], name: &str) -> Option<usize> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.parse().unwrap())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).expect("usage: logit_dump <out.bin>");
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let len = arg(&args, "--len").unwrap_or(100);
    let steps = arg(&args, "--steps").unwrap_or(16);
    let fast = !args.iter().any(|a| a == "--naive");
    let engine = Engine::new()?;
    let verify = args.iter().any(|a| a == "--verify");
    engine.set_debug_uploads(verify);
    let model = GpuModel::load(&engine, &gguf, fast)?;
    if verify {
        println!("verify_uploads {}", pollster::block_on(engine.debug_verify_uploads()));
        engine.set_debug_uploads(false);
    }
    let max_ctx = (len + steps + 4) as u32;
    let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let ids: Vec<u32> = (0..len).map(|i| PROMPT[i % PROMPT.len()]).collect();
    let repeat = arg(&args, "--repeat").unwrap_or(1);
    for r in 0..repeat.saturating_sub(1) {
        model.pool.reset();
        cache.kv_len = 0;
        let l = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
        println!("repeat {r}: prefill logits fnv={:016x}", fnv(&l.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()));
    }
    model.pool.reset();
    cache.kv_len = 0;
    let mut all: Vec<f32> = Vec::new();
    let mut logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
    let mut tokens = Vec::new();
    all.extend_from_slice(&logits);
    for _ in 0..steps {
        let t = argmax(&logits);
        tokens.push(t);
        logits = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, t, &cos_buf, &sin_buf, None));
        all.extend_from_slice(&logits);
    }
    let bytes: Vec<u8> = all.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(out_path, &bytes)?;
    let h = fnv(&bytes);
    println!("weight_bytes={} tokens={tokens:?} logits_fnv={h:016x}", model.weight_gpu_bytes());
    Ok(())
}
