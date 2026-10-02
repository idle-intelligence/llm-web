//! Native timing sweep: prefill wall time at several prompt lengths and the
//! greedy decode step time, on one loaded model. Prompt ids are the
//! `short` fixture prompt repeated to length; timing only, no parity check
//! (the fixture tests do that).
//!
//! `LEAN_GGUF=... cargo run --release -p lean --example prefill_sweep -- [--lens 8,16,36] [--reps 5] [--decode 32] [--split]`
//!
//! `--split` adds the per-op-group GPU time of one prefill per length and of
//! the decode steps (pass timestamps, needs `TIMESTAMP_QUERY`).

use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_decode_step_argmax, forward_prefill, GpuModel, KvCache};

const PROMPT: [u32; 36] = [
    151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
];

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn print_split(label: &str, engine: &Engine) {
    if let Some(g) = engine.diag_last_gpu.borrow().as_ref() {
        let segs: Vec<String> = g.segments.iter().map(|(l, ms, _)| format!("{l}={ms:.2}")).collect();
        println!("  {label} gpu span {:.2} ms: {}", g.span_ms, segs.join(" "));
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let lens: Vec<u32> = arg(&args, "--lens").unwrap_or_else(|| "8,16,24,36,48,64,96,128,256".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let reps: usize = arg(&args, "--reps").map(|s| s.parse().unwrap()).unwrap_or(5);
    let n_decode: usize = arg(&args, "--decode").map(|s| s.parse().unwrap()).unwrap_or(32);
    let split = args.iter().any(|a| a == "--split");
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");

    let engine = pollster::block_on(Engine::new_async_with(split))?;
    let model = GpuModel::load(&engine, &gguf, true)?;
    let max_len = *lens.iter().max().unwrap();
    let max_ctx = max_len + n_decode as u32 + 4;
    let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);

    println!("len  prefill_ms_median  min  max");
    for &len in &lens {
        let ids: Vec<u32> = (0..len as usize).map(|i| PROMPT[i % PROMPT.len()]).collect();
        let mut times = Vec::new();
        for r in 0..=reps {
            model.pool.reset();
            cache.kv_len = 0;
            let t = std::time::Instant::now();
            let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
            if r > 0 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (mn, mx) = (times.iter().cloned().fold(f64::MAX, f64::min), times.iter().cloned().fold(0.0, f64::max));
        println!("{len:4} {:8.2} {mn:8.2} {mx:8.2}", median(&mut times));
        if split {
            engine.set_diag(true, true);
            model.pool.reset();
            cache.kv_len = 0;
            let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
            engine.set_diag(false, false);
            print_split("prefill", &engine);
        }
    }

    if n_decode > 0 {
        let ids: Vec<u32> = PROMPT.to_vec();
        model.pool.reset();
        cache.kv_len = 0;
        let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
        let mut next = lean::model::argmax(&logits);
        let mut times = Vec::new();
        let mut encode = Vec::new();
        for i in 0..n_decode {
            let t = std::time::Instant::now();
            next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
            if i > 0 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
                encode.push(engine.diag_encode_ms.get());
            }
        }
        let (mn, mx) = (times.iter().cloned().fold(f64::MAX, f64::min), times.iter().cloned().fold(0.0, f64::max));
        println!("decode ms/step median {:.2} (min {mn:.2} max {mx:.2}), encode+submit median {:.2}", median(&mut times), median(&mut encode));
        if split {
            engine.set_diag(true, true);
            let _ = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
            engine.set_diag(false, false);
            print_split("decode", &engine);
        }
    }
    Ok(())
}
