//! Native per-dispatch GPU profile of decode and prefill at chosen context
//! lengths, for the roofline tables.
//!
//! `LEAN_PROFILE_KERNELS=1 LEAN_GGUF=... cargo run --release -p lean --example profile_roofline -- decode --ctx 512 [--steps 16]`
//! `LEAN_PROFILE_KERNELS=1 LEAN_GGUF=... cargo run --release -p lean --example profile_roofline -- prefill --len 100 [--reps 3]`
//!
//! Stdout carries the model dimensions (`model ...`), wall-clock lines of
//! unsplit runs (`wall ...`) and the GPU time per op group (`seg ...`) from
//! pass-level timestamps (`Engine::set_diag(true, true)`: one compute pass
//! per op group, needs `TIMESTAMP_QUERY` only, which Metal grants). With
//! `LEAN_PROFILE_KERNELS=1` on an adapter that also has
//! `TIMESTAMP_QUERY_INSIDE_PASSES` (Vulkan), the per-dispatch table is
//! printed to stderr as well.

use lean_core::engine::{DiagGpu, Engine};
use lean_core::model::{build_rope_tables, forward_decode_step_argmax, forward_prefill, GpuModel, KvCache};

const PROMPT: [u32; 36] = [
    151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
];

fn arg(args: &[String], name: &str) -> Option<usize> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.parse().unwrap())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("decode");
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");
    let engine = pollster::block_on(Engine::new_async_with(true))?;
    let model = GpuModel::load(&engine, &gguf, true)?;
    let c = &model.config;
    println!(
        "model hidden={} layers={} heads={} kv_heads={} head_dim={} inter={} vocab={} weight_bytes={} profiling={}",
        c.hidden_size,
        c.num_layers,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
        c.intermediate_size,
        c.vocab_size,
        model.weight_gpu_bytes(),
        engine.profiling_enabled()
    );
    let ctx = arg(&args, "--ctx").unwrap_or(128);
    let len = arg(&args, "--len").unwrap_or(100);
    let steps = arg(&args, "--steps").unwrap_or(16);
    let reps = arg(&args, "--reps").unwrap_or(3);
    let max_ctx = (ctx.max(len) + steps + 8) as u32;
    let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);
    let ids = |n: usize| -> Vec<u32> { (0..n).map(|i| PROMPT[i % PROMPT.len()]).collect() };

    if mode == "prefill" {
        let p = ids(len);
        let mut times = Vec::new();
        // One unrecorded warm-up, then `reps` unsplit prefills for wall time.
        for r in 0..=reps {
            model.pool.reset();
            cache.kv_len = 0;
            engine.reset_dispatch_count();
            let t = std::time::Instant::now();
            let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &p, &cos_buf, &sin_buf, None));
            if r == 0 && engine.profiling_enabled() {
                eprintln!("warmup (discarded)");
                lean_core::profile_report::report(1);
            }
            if r > 0 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        if engine.profiling_enabled() {
            eprintln!("prefill len={len}");
            lean_core::profile_report::report(reps as u64);
        }
        println!("wall prefill len={len} ms_median={:.2} dispatches={} per_token={:.3}", median(&mut times), engine.dispatch_count(), engine.dispatch_count() as f64 / len as f64);
        let mut runs: Vec<DiagGpu> = Vec::new();
        engine.set_diag(true, true);
        for _ in 0..reps {
            model.pool.reset();
            cache.kv_len = 0;
            let _ = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &p, &cos_buf, &sin_buf, None));
            runs.push(engine.diag_last_gpu.borrow().clone().expect("pass timestamps"));
        }
        engine.set_diag(false, false);
        print_segments(&format!("prefill len={len}"), &runs);
    } else {
        let p = ids(ctx);
        model.pool.reset();
        cache.kv_len = 0;
        let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &p, &cos_buf, &sin_buf, None));
        let mut next = lean_core::model::argmax(&logits);
        for _ in 0..2 {
            next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
        }
        if engine.profiling_enabled() {
            eprintln!("context prefill and warmup (discarded)");
            lean_core::profile_report::report(1);
        }
        engine.reset_dispatch_count();
        let mut times = Vec::new();
        for _ in 0..steps {
            let t = std::time::Instant::now();
            next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        if engine.profiling_enabled() {
            eprintln!("decode ctx={ctx}");
            lean_core::profile_report::report(steps as u64);
        }
        println!("wall decode ctx={ctx} ms_per_step_median={:.2} dispatches_per_step={}", median(&mut times), engine.dispatch_count() / steps as u64);
        let mut runs: Vec<DiagGpu> = Vec::new();
        engine.set_diag(true, true);
        for _ in 0..steps {
            next = pollster::block_on(forward_decode_step_argmax(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
            runs.push(engine.diag_last_gpu.borrow().clone().expect("pass timestamps"));
        }
        engine.set_diag(false, false);
        print_segments(&format!("decode ctx={ctx}"), &runs);
    }
    Ok(())
}

/// Median over runs of each label's summed pass time, plus the span and the
/// sum of pass times. Labels keep first-seen order.
fn print_segments(tag: &str, runs: &[DiagGpu]) {
    let mut labels: Vec<(String, u32)> = Vec::new();
    for r in runs {
        for (l, _, n) in &r.segments {
            if !labels.iter().any(|(x, _)| x == l) {
                labels.push((l.clone(), *n));
            }
        }
    }
    for (l, n) in &labels {
        let mut v: Vec<f64> = runs.iter().filter_map(|r| r.segments.iter().find(|s| &s.0 == l).and_then(|s| s.1)).collect();
        if !v.is_empty() {
            println!("seg {tag} label={l} passes={n} ms={:.4}", median(&mut v));
        }
    }
    let mut span: Vec<f64> = runs.iter().filter_map(|r| r.span_ms).collect();
    let mut sum: Vec<f64> = runs.iter().map(|r| r.pass_sum_ms).collect();
    println!("seg {tag} label=TOTAL_SPAN passes=0 ms={:.4}", median(&mut span));
    println!("seg {tag} label=TOTAL_PASS_SUM passes=0 ms={:.4}", median(&mut sum));
}
