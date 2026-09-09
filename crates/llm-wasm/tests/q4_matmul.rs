//! Q4_0 dequant+matmul kernel vs CPU-dequant reference (C1 checkpoint).
//!
//! Two families of coverage, per the task brief:
//! - Synthetic random Q4_0 weights at the model's actual `[N, K]` shapes
//!   apart from the 151936-wide lm_head (q/o: 2048x2048, gate/up: 11008x2048,
//!   down: 2048x11008), M in {1, 64}, full CPU-dequant-then-matmul reference.
//! - The **real** `token_embd.weight` tensor from the xLAM-2-3b-fc-r GGUF
//!   (Q4_0, [151936, 2048], tied lm_head shape) at M in {1, 64}: the GPU
//!   kernel computes the full 151936-wide output (exercising the real
//!   dispatch-grid/buffer-size path from docs/ENGINE.md §2), but the CPU
//!   reference only dequantizes+dots a handful of sampled output rows
//!   (avoiding a full 1.2GB CPU dequant of the 151936x2048 table — this
//!   crate's whole embedding-lookup design deliberately avoids that, see
//!   gguf.rs's module doc comment).
#![cfg(feature = "wgpu")]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use llm_wasm::gguf::{q4_matmul, Q4Tensor};

fn device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Deterministic xorshift PRNG (no `rand` dependency needed for test data).
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

/// Build a random Q4_0 byte blob of shape `[n, k]` (row-major, 18
/// bytes/32-element block: f16 scale + 16 bytes of paired nibbles) together
/// with its exact CPU-dequantized f32 `[n, k]` reference.
fn random_q4(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    assert!(k.is_multiple_of(32));
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 18;
    let mut bytes = vec![0u8; n * bytes_per_row];
    let mut dequant = vec![0f32; n * k];
    let mut rng = Xorshift::new(seed);

    for row in 0..n {
        for block in 0..blocks_per_row {
            // Small, well-scaled values so f32 accumulation error stays tiny.
            let scale_bits = half_from_f32(0.01 + (rng.next_u32() % 100) as f32 * 0.002);
            let bo = row * bytes_per_row + block * 18;
            bytes[bo] = (scale_bits & 0xFF) as u8;
            bytes[bo + 1] = (scale_bits >> 8) as u8;
            let scale = llm_wasm::gguf::f16_to_f32(scale_bits);

            let mut nibbles = [0u8; 32];
            for n_ in nibbles.iter_mut() {
                *n_ = (rng.next_u32() % 16) as u8;
            }
            for j in 0..16 {
                let lo = nibbles[j];
                let hi = nibbles[j + 16];
                bytes[bo + 2 + j] = lo | (hi << 4);
            }
            let base = row * k + block * 32;
            for j in 0..16 {
                dequant[base + j] = (nibbles[j] as f32 - 8.0) * scale;
                dequant[base + j + 16] = (nibbles[j + 16] as f32 - 8.0) * scale;
            }
        }
    }
    (bytes, dequant)
}

/// Round-trip a small positive f32 through IEEE754 half precision bits.
fn half_from_f32(v: f32) -> u16 {
    // v is always small and positive in this test's usage, so a direct
    // normalized-range encode is sufficient (no subnormal/inf handling).
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3FF;
    ((sign as u16) << 15) | ((exp.clamp(1, 30) as u16) << 10) | (mantissa as u16)
}

fn cpu_matmul(input: &[f32], weights: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        let x = &input[mi * k..(mi + 1) * k];
        for ni in 0..n {
            let w = &weights[ni * k..(ni + 1) * k];
            out[mi * n + ni] = x.iter().zip(w).map(|(a, b)| a * b).sum();
        }
    }
    out
}

fn run_gpu_matmul(input: &[f32], q4_bytes: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let device = device();
    let weights = Q4Tensor::from_q4_bytes(q4_bytes, [n, k], &device).expect("upload Q4 weights");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input, &device).reshape([1, m, k]);
    let out = q4_matmul(input_t, &weights);
    out.into_data().into_vec::<f32>().expect("f32 output")
}

fn random_input(m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift::new(seed);
    (0..m * k)
        .map(|_| (rng.next_u32() % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

/// Synthetic shapes matching the model's actual linear layers (excluding the
/// 151936-wide lm_head, covered separately below against the real GGUF).
#[test]
fn test_q4_matmul_synthetic_shapes() {
    // (K, N) pairs: attn_q/attn_output (2048x2048), ffn_gate/ffn_up
    // (2048->11008), ffn_down (11008->2048).
    let shapes = [(2048usize, 2048usize), (2048, 11008), (11008, 2048)];
    let ms = [1usize, 64usize];

    for &(k, n) in &shapes {
        let (q4_bytes, cpu_weights) = random_q4(n, k, 0xC0FFEE ^ (k as u64) ^ ((n as u64) << 20));
        for &m in &ms {
            let input = random_input(m, k, 0xBEEF + m as u64);
            let cpu_out = cpu_matmul(&input, &cpu_weights, m, k, n);
            let gpu_out = run_gpu_matmul(&input, &q4_bytes, m, k, n);
            assert_eq!(cpu_out.len(), gpu_out.len());

            let mut max_err = 0f32;
            for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
                max_err = max_err.max((c - g).abs());
            }
            // Absolute tolerance: K-length dot products (up to 11008 terms)
            // of values roughly in [-1,1]*[-0.1,0.1]; f32 accumulation order
            // differs between CPU (sequential) and the WGSL kernel
            // (block-vectorized), so allow generous slack scaled by K.
            let tol = 0.05 * (k as f32).sqrt();
            assert!(
                max_err < tol,
                "K={k} N={n} M={m}: max_err={max_err} exceeds tol={tol}"
            );
            println!("synthetic K={k} N={n} M={m}: max_err={max_err} (tol {tol})");
        }
    }
}

/// K1: decode matvec (M=1) at N=151936 (lm_head/embedding width), both K
/// values the model actually uses. `q4_matmul` dispatches the cooperative
/// `shader_q4_matvec.wgsl` kernel whenever `B*M==1` (gguf.rs), so this
/// exercises that kernel at the largest N in the model — the real
/// `token_embd.weight` test below covers K=2048/N=151936 against real
/// weights; this covers the synthetic K=11008/N=151936 combination too, for
/// full N x K coverage at M=1 per the task brief.
#[test]
fn test_q4_matvec_m1_large_n() {
    let shapes = [(2048usize, 151936usize), (11008, 151936)];
    for &(k, n) in &shapes {
        let (q4_bytes, cpu_weights) = random_q4(n, k, 0xFEED ^ (k as u64) ^ ((n as u64) << 20));
        let input = random_input(1, k, 0xC0DE + k as u64);
        let cpu_out = cpu_matmul(&input, &cpu_weights, 1, k, n);
        let gpu_out = run_gpu_matmul(&input, &q4_bytes, 1, k, n);
        assert_eq!(cpu_out.len(), gpu_out.len());

        let mut max_err = 0f32;
        for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((c - g).abs());
        }
        let tol = 0.05 * (k as f32).sqrt();
        assert!(
            max_err < tol,
            "K1 matvec K={k} N={n} M=1: max_err={max_err} exceeds tol={tol}"
        );
        println!("K1 matvec K={k} N={n} M=1: max_err={max_err} (tol {tol})");
    }
}

/// Real `token_embd.weight` (Q4_0, [151936, 2048], tied lm_head shape) from
/// the xLAM-2-3b-fc-r GGUF. Exercises the full 151936-wide dispatch and the
/// ~174MB single-buffer upload (docs/ENGINE.md §2's untested-limit concern).
#[test]
fn test_q4_matmul_real_gguf_token_embd() {
    let model_dir = std::env::var("LLM_MODEL_DIR").unwrap_or_else(|_| {
        "/Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r".to_string()
    });
    let path = format!("{model_dir}/xLAM-2-3b-fc-r-q4_0.gguf");
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }

    let file = std::fs::File::open(&path).expect("open gguf");
    let reader = std::io::BufReader::new(file);
    let mut loader = llm_wasm::gguf::Q4ModelLoader::new(reader).expect("parse gguf");
    let info = loader
        .reader()
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight present")
        .clone();
    let shape: Vec<usize> = info.shape().iter().rev().map(|&d| d as usize).collect();
    let (n, k) = (shape[0], shape[1]);
    assert_eq!((n, k), (151936, 2048));

    // Reading + uploading the full 174MB Q4_0 buffer and successfully
    // launching the kernel at N=151936 below *is* the check for
    // docs/ENGINE.md §2's untested maxStorageBufferBindingSize concern —
    // a limit shortfall would panic on `create_from_slice` or the launch.
    let bytes = loader.tensor_bytes("token_embd.weight").expect("read token_embd bytes");
    drop(loader);

    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 18;
    assert_eq!(bytes.len(), n * bytes_per_row);

    let sample_rows: [usize; 5] = [0, 1234, 75968, 150000, 151935];

    for &m in &[1usize, 64usize] {
        let input = random_input(m, k, 42 + m as u64);
        let gpu_out = run_gpu_matmul(&input, &bytes, m, k, n);
        assert_eq!(gpu_out.len(), m * n);

        let mut max_err = 0f32;
        for row in sample_rows {
            let row_bytes = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let mut w = vec![0f32; k];
            for block in 0..blocks_per_row {
                let bo = block * 18;
                let scale =
                    llm_wasm::gguf::f16_to_f32(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]));
                let base = block * 32;
                for j in 0..16 {
                    let byte = row_bytes[bo + 2 + j];
                    w[base + j] = ((byte & 0x0F) as f32 - 8.0) * scale;
                    w[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * scale;
                }
            }
            for mi in 0..m {
                let x = &input[mi * k..(mi + 1) * k];
                let cpu_val: f32 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
                let gpu_val = gpu_out[mi * n + row];
                max_err = max_err.max((cpu_val - gpu_val).abs());
            }
        }
        // Real weight scales/values are small (typical Q4_0 model weights);
        // absolute tolerance scaled by sqrt(K)=~45 as above.
        let tol = 0.05 * (k as f32).sqrt();
        assert!(
            max_err < tol,
            "real token_embd.weight M={m}: max_err={max_err} exceeds tol={tol}"
        );
        println!("real token_embd.weight N={n} K={k} M={m}: max_err={max_err} (tol {tol})");
    }
}
