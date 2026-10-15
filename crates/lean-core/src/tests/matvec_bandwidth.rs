//! Decode matvec read bandwidth per weight format: one compute pass of many
//! matvecs (`rows == 1`) over distinct weights whose total size is far above
//! the device's caches, one submit, one readback. Prints the bytes of
//! weights read per second and the share of `LEAN_READ_CEILING_GBS`
//! (default 91.4, the M2's measured read ceiling). No model file needed.
//!
//! ```sh
//! cargo test -p lean-core --release --lib matvec_bandwidth -- --ignored --nocapture
//! ```

use super::kernel_parity::{synth, Rng};
use crate::engine::Engine;
use crate::gguf::GgmlDtype;
use crate::model::{linear, Pool};
use crate::quant::load_matmul_weight_gguf;

const KEYS: [&str; 16] = ["ta", "tb", "tc", "td", "te", "tf", "tg", "th", "ti", "tj", "tk", "tl", "tm", "tn", "to", "tp"];

fn bench(engine: &Engine, dtype: GgmlDtype, in_dim: usize, out_dim: usize, ceiling: f64) {
    let mut rng = Rng(0x9e37_79b9 ^ out_dim as u32);
    let weights: Vec<_> = (0..KEYS.len()).map(|_| load_matmul_weight_gguf(engine, "w", &[out_dim, in_dim], dtype, &synth(dtype, out_dim * in_dim, &mut rng)).unwrap()).collect();
    let bytes_per_weight = weights[0].gpu_bytes();
    let x: Vec<f32> = (0..in_dim).map(|_| rng.unit()).collect();
    let bias = vec![0f32; out_dim];
    let pool = Pool::new(engine.device.clone(), engine.queue.clone());
    let xb = engine.buf_f32(&x, "x");
    let bb = engine.buf_f32(&bias, "bias");
    let reps = 6;
    let mut times = Vec::new();
    for rep in 0..reps + 1 {
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("matvec_bandwidth") });
        let mut last = None;
        {
            let mut pass = engine.begin_pass(&mut encoder, "matvec_bandwidth");
            for _ in 0..4 {
                for (key, w) in KEYS.iter().zip(&weights) {
                    last = Some(linear(engine, &pool, &mut pass, key, &xb, 1, in_dim as u32, w, &bb, out_dim as u32, true));
                }
            }
        }
        let t = std::time::Instant::now();
        engine.queue.submit(Some(encoder.finish()));
        let _ = pollster::block_on(engine.read_buffer(&last.unwrap(), 1));
        if rep > 0 {
            times.push(t.elapsed().as_secs_f64());
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let gbs = (4 * KEYS.len()) as f64 * bytes_per_weight as f64 / med / 1e9;
    println!("{dtype:?} in {in_dim} out {out_dim}: {:.1} MB per weight, {:.0} us per matvec (min {:.0}), {gbs:.1} GB/s = {:.0}% of {ceiling}", bytes_per_weight as f64 / 1e6, med * 1e6 / (4 * KEYS.len()) as f64, times[0] * 1e6 / (4 * KEYS.len()) as f64, 100.0 * gbs / ceiling);
}

#[test]
#[ignore = "needs a wgpu adapter; timing"]
fn matvec_bandwidth() {
    std::env::set_var("LEAN_F16", "0");
    let engine = Engine::new().expect("wgpu engine init");
    let ceiling: f64 = std::env::var("LEAN_READ_CEILING_GBS").ok().and_then(|s| s.parse().ok()).unwrap_or(91.4);
    for &(in_dim, out_dim) in &[(2048usize, 8192usize), (2048, 2048)] {
        for dtype in [GgmlDtype::Q4_0, GgmlDtype::Q4_K, GgmlDtype::Q5_0, GgmlDtype::Q8_0, GgmlDtype::Q6_K, GgmlDtype::F16] {
            bench(&engine, dtype, in_dim, out_dim, ceiling);
        }
    }
}
