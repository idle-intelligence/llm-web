//! Every GPU matmul kernel of the formats lean keeps quantized, against a CPU
//! reference (`gguf::dequantize_for` followed by a plain f64 matmul): decode
//! matvec (one and four rows per lane), the fused gate/up kernels, short and
//! long prefill, the naive reference kernel, and the f16 prefill tile when
//! the device grants `shader-f16`. Synthetic weights (random block bytes with
//! valid scale fields), so no model files are needed; a wgpu adapter is.

use crate::engine::Engine;
use crate::gguf::{dequantize_for, GgmlDtype};
use crate::model::{gate_up_swiglu_decode, linear, Pool};
use crate::quant::load_matmul_weight_gguf;

pub(super) struct Rng(pub(super) u32);

impl Rng {
    pub(super) fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    pub(super) fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

/// Random bytes for `n_elements` values of `dtype` with sane scale fields.
pub(super) fn synth(dtype: GgmlDtype, n_elements: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::new();
    match dtype {
        GgmlDtype::Q4_0 => {
            for _ in 0..n_elements / 32 {
                out.extend(f16_bytes(0.02 + (rng.next() % 50) as f32 * 0.001));
                out.extend((0..16).map(|_| rng.next() as u8));
            }
        }
        GgmlDtype::Q5_0 => {
            for _ in 0..n_elements / 32 {
                out.extend(f16_bytes(0.01 + (rng.next() % 50) as f32 * 0.001));
                out.extend((0..20).map(|_| rng.next() as u8));
            }
        }
        GgmlDtype::Q4_K => {
            for _ in 0..n_elements / 256 {
                out.extend(f16_bytes(0.002 + (rng.next() % 30) as f32 * 0.0002));
                out.extend(f16_bytes(0.001 + (rng.next() % 30) as f32 * 0.0002));
                out.extend((0..140).map(|_| rng.next() as u8));
            }
        }
        GgmlDtype::Q8_0 => {
            for _ in 0..n_elements / 32 {
                out.extend(f16_bytes(0.01 + (rng.next() % 50) as f32 * 0.001));
                out.extend((0..32).map(|_| rng.next() as u8));
            }
        }
        GgmlDtype::Q6_K => {
            for _ in 0..n_elements / 256 {
                out.extend((0..208).map(|_| rng.next() as u8));
                out.extend(f16_bytes(0.0005 + (rng.next() % 30) as f32 * 0.0001));
            }
        }
        GgmlDtype::F16 => out.extend((0..n_elements).flat_map(|_| f16_bytes(rng.unit() * 0.5))),
        GgmlDtype::BF16 => out.extend((0..n_elements).flat_map(|_| (((rng.unit() * 0.5).to_bits() >> 16) as u16).to_le_bytes())),
        other => panic!("no synthetic generator for {other:?}"),
    }
    out
}

fn reference(w: &[f32], x: &[f32], bias: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    for m in 0..rows {
        for n in 0..out_dim {
            let mut acc = bias[n] as f64;
            for k in 0..in_dim {
                acc += w[n * in_dim + k] as f64 * x[m * in_dim + k] as f64;
            }
            out[m * out_dim + n] = acc as f32;
        }
    }
    out
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32) {
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt().max(1e-3);
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    assert!(worst <= rel * rms.max(1.0), "{name}: worst |diff| {worst} over rms {rms}");
}

#[allow(clippy::too_many_arguments)]
fn run_linear(engine: &Engine, w: &crate::quant::MatMulWeight, x: &[f32], bias: &[f32], rows: u32, in_dim: u32, out_dim: u32, fast: bool) -> Vec<f32> {
    let pool = Pool::new(engine.device.clone(), engine.queue.clone());
    let xb = engine.buf_f32(x, "x");
    let bb = engine.buf_f32(bias, "bias");
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel_parity") });
    let out;
    {
        let mut pass = engine.begin_pass(&mut encoder, "kernel_parity");
        out = linear(engine, &pool, &mut pass, "t", &xb, rows, in_dim, w, &bb, out_dim, fast);
    }
    engine.queue.submit(Some(encoder.finish()));
    pollster::block_on(engine.read_buffer(&out, (rows * out_dim) as usize))
}

fn run_swiglu(engine: &Engine, w: &crate::quant::MatMulWeight, x: &[f32], bias: &[f32], in_dim: u32, inter: u32) -> Option<Vec<f32>> {
    let pool = Pool::new(engine.device.clone(), engine.queue.clone());
    let xb = engine.buf_f32(x, "x");
    let bb = engine.buf_f32(bias, "bias");
    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel_parity_swiglu") });
    let out;
    {
        let mut pass = engine.begin_pass(&mut encoder, "kernel_parity_swiglu");
        out = gate_up_swiglu_decode(engine, &pool, &mut pass, "t", &xb, in_dim, w, &bb, inter, true)?;
    }
    engine.queue.submit(Some(encoder.finish()));
    Some(pollster::block_on(engine.read_buffer(&out, inter as usize)))
}

fn f16_granted(engine: &Engine) -> bool {
    engine.adapter_report.split("\"deviceFeatures\":").nth(1).is_some_and(|r| r.split("\"maxStorage").next().unwrap_or("").contains("SHADER_F16"))
}

fn exercise(engine: &Engine, dtype: GgmlDtype, in_dim: usize, strict: bool) {
    let mut rng = Rng(0x1234_5678 ^ in_dim as u32);
    let f16 = f16_granted(engine);
    // (out_dim, rows): narrow and wide (four rows per lane) decode, short and long prefill.
    for &(out_dim, rows) in &[(80usize, 1usize), (4160, 1), (80, 5), (96, 70), (4160, 3)] {
        let bytes = synth(dtype, out_dim * in_dim, &mut rng);
        let w_ref = dequantize_for(dtype, &bytes, out_dim * in_dim);
        let w = load_matmul_weight_gguf(engine, "w", &[out_dim, in_dim], dtype, &bytes).unwrap();
        let x: Vec<f32> = (0..rows * in_dim).map(|_| rng.unit()).collect();
        let bias: Vec<f32> = (0..out_dim).map(|_| rng.unit()).collect();
        let want = reference(&w_ref, &x, &bias, rows, in_dim, out_dim);
        for fast in [false, true] {
            let got = run_linear(engine, &w, &x, &bias, rows as u32, in_dim as u32, out_dim as u32, fast);
            // The f16 prefill tile rounds weights and activations to f16.
            let rel = if f16 && fast && rows >= 64 && !strict { 4e-2 } else { 1e-3 };
            check(&format!("{dtype:?} in {in_dim} out {out_dim} rows {rows} fast {fast}"), &got, &want, rel);
        }
    }
    // Fused gate/up (decode): a narrow and a wide fused weight.
    for &inter in &[40usize, 2080] {
        let bytes = synth(dtype, 2 * inter * in_dim, &mut rng);
        let w_ref = dequantize_for(dtype, &bytes, 2 * inter * in_dim);
        let w = load_matmul_weight_gguf(engine, "w", &[2 * inter, in_dim], dtype, &bytes).unwrap();
        let x: Vec<f32> = (0..in_dim).map(|_| rng.unit()).collect();
        let bias: Vec<f32> = (0..2 * inter).map(|_| rng.unit()).collect();
        let Some(got) = run_swiglu(engine, &w, &x, &bias, in_dim as u32, inter as u32) else { continue };
        let gu = reference(&w_ref, &x, &bias, 1, in_dim, 2 * inter);
        let want: Vec<f32> = (0..inter).map(|j| gu[j] / (1.0 + (-gu[j]).exp()) * gu[inter + j]).collect();
        check(&format!("{dtype:?} swiglu inter {inter}"), &got, &want, 1e-3);
    }
}

#[test]
#[ignore = "needs a wgpu adapter"]
fn quantized_kernels_match_cpu_reference() {
    std::env::set_var("LEAN_F16", "0");
    let strict = Engine::new().expect("wgpu engine init");
    std::env::remove_var("LEAN_F16");
    let default = Engine::new().expect("wgpu engine init");
    eprintln!("f16 granted on the default engine: {}", f16_granted(&default));
    for dtype in [GgmlDtype::Q4_0, GgmlDtype::Q5_0, GgmlDtype::Q4_K, GgmlDtype::Q6_K, GgmlDtype::F16, GgmlDtype::BF16] {
        // 512 is a multiple of 256 (Q4_K) and 32; 288 is a multiple of 32 only.
        let dims: &[usize] = if matches!(dtype, GgmlDtype::Q4_K | GgmlDtype::Q6_K) { &[512, 1024] } else { &[512, 288] };
        for &in_dim in dims {
            exercise(&strict, dtype, in_dim, true);
            exercise(&default, dtype, in_dim, false);
            eprintln!("{dtype:?} in_dim {in_dim}: ok");
        }
    }
}
