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
use llm_wasm::gguf::{
    pad_m_bucket_for_test, q4_dequant_scratch_to_vec, q4_matmul, q4_matmul_naive_forced,
    q4_matmul_scratch_forced, q4_matmul_tiled_forced, Q4ModelLoader, Q4Tensor,
};

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

/// Like `run_gpu_matmul` but with an explicit batch dimension `b` (input
/// shape `[b, m, k]`) — exercises `q4_matmul`'s `B`/`b_valid` handling
/// directly, e.g. B>1 M=1 decode (classifier-free guidance's dual-batch KV
/// caches). Batch rows are numerically independent (same weights, no
/// cross-batch interaction), so `cpu_matmul(..., m = b*m, ...)` is a valid
/// reference regardless of how the b/m split is expressed on the GPU side.
fn run_gpu_matmul_b(input: &[f32], q4_bytes: &[u8], b: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let device = device();
    let weights = Q4Tensor::from_q4_bytes(q4_bytes, [n, k], &device).expect("upload Q4 weights");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input, &device).reshape([b, m, k]);
    let out = q4_matmul(input_t, &weights);
    out.into_data().into_vec::<f32>().expect("f32 output")
}

fn run_gpu_matmul_tiled_forced(input: &[f32], q4_bytes: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let device = device();
    let weights = Q4Tensor::from_q4_bytes(q4_bytes, [n, k], &device).expect("upload Q4 weights");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input, &device).reshape([1, m, k]);
    let out = q4_matmul_tiled_forced(input_t, &weights);
    out.into_data().into_vec::<f32>().expect("f32 output")
}

/// K2's tiled kernel (native, B==1, workgroup-shared dequant/weight reuse)
/// at M in {64, 128, 256} — the task brief's M coverage for the prefill
/// kernel. Not dispatched by default (`q4_matmul`'s doc comment /
/// docs/BENCHMARKS.md: measured 4x slower than the naive kernel), but
/// still correct — exercised directly via `q4_matmul_tiled_forced`.
#[test]
fn test_q4_matmul_tiled_k2_shapes() {
    let shapes = [(2048usize, 2048usize), (2048, 11008), (11008, 2048)];
    let ms = [64usize, 128usize, 256usize];

    for &(k, n) in &shapes {
        let (q4_bytes, cpu_weights) = random_q4(n, k, 0xDEC0DE ^ (k as u64) ^ ((n as u64) << 20));
        for &m in &ms {
            let input = random_input(m, k, 0xF00D + m as u64);
            let cpu_out = cpu_matmul(&input, &cpu_weights, m, k, n);
            let gpu_out = run_gpu_matmul_tiled_forced(&input, &q4_bytes, m, k, n);
            assert_eq!(cpu_out.len(), gpu_out.len());

            let mut max_err = 0f32;
            for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
                max_err = max_err.max((c - g).abs());
            }
            let tol = 0.05 * (k as f32).sqrt();
            assert!(
                max_err < tol,
                "K2 tiled K={k} N={n} M={m}: max_err={max_err} exceeds tol={tol}"
            );
            println!("K2 tiled K={k} N={n} M={m}: max_err={max_err} (tol {tol})");
        }
    }
}

fn random_input(m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift::new(seed);
    (0..m * k)
        .map(|_| (rng.next_u32() % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

/// K2 micro-benchmark: isolates `q4_matmul`'s kernel throughput at a
/// realistic prefill shape (M=2225 matches `02_tools_single`'s prompt
/// length; K=2048/N=11008 matches ffn_gate/up's shape) without any model
/// loading, so tile-parameter iteration takes seconds instead of minutes.
/// `#[ignore]`d: not part of the correctness suite, run explicitly with
/// `cargo test --release --features wgpu --test q4_matmul -- --ignored
/// bench_tiled_matmul_shape --nocapture`.
#[test]
#[ignore]
fn bench_tiled_matmul_shape() {
    let (k, n, m) = (2048usize, 11008usize, 2225usize);
    let (q4_bytes, _cpu_weights) = random_q4(n, k, 0xABCD);
    let input = random_input(m, k, 0x1234);

    let device = device();
    let weights = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("upload");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input.as_slice(), &device).reshape([1, m, k]);

    // Warm-up (pipeline compile).
    let _ = q4_matmul_tiled_forced(input_t.clone(), &weights)
        .into_data()
        .into_vec::<f32>()
        .unwrap();

    let iters = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = q4_matmul_tiled_forced(input_t.clone(), &weights);
        let _ = out.into_data().into_vec::<f32>().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "M={m} K={k} N={n}: {:.1} ms/call, {:.2} GFLOP/s",
        dt * 1000.0,
        flops / dt / 1e9
    );
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

/// Decode matvec (M=1) at N=151936 (lm_head/embedding width), both K
/// values the model actually uses, B in {1, 2} — B>1 exercises the coalesced
/// kernel's `b_valid`/`B` handling directly (Session 6: `q4_matmul` now
/// dispatches the matvec kernel whenever `m==1`, not just `b*m==1` — see
/// gguf.rs's dispatch comment). N=151936 = 37984 * MATVEC_COALESCED_ROWS_PER_WG
/// (exact multiple, but still the largest N in the model and the shape most
/// likely to expose an off-by-one in the dispatch grid math). The real
/// `token_embd.weight` test below covers K=2048/N=151936 against real
/// weights; this covers the synthetic K=11008/N=151936 combination too.
#[test]
fn test_q4_matvec_m1_large_n() {
    let shapes = [(2048usize, 151936usize), (11008, 151936)];
    for &(k, n) in &shapes {
        let (q4_bytes, cpu_weights) = random_q4(n, k, 0xFEED ^ (k as u64) ^ ((n as u64) << 20));
        for &b in &[1usize, 2usize] {
            let input = random_input(b, k, 0xC0DE + k as u64 + b as u64 * 7);
            let cpu_out = cpu_matmul(&input, &cpu_weights, b, k, n);
            let gpu_out = run_gpu_matmul_b(&input, &q4_bytes, b, 1, k, n);
            assert_eq!(cpu_out.len(), gpu_out.len());

            let mut max_err = 0f32;
            for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
                max_err = max_err.max((c - g).abs());
            }
            let tol = 0.05 * (k as f32).sqrt();
            assert!(
                max_err < tol,
                "coalesced matvec K={k} N={n} B={b} M=1: max_err={max_err} exceeds tol={tol}"
            );
            println!("coalesced matvec K={k} N={n} B={b} M=1: max_err={max_err} (tol {tol})");
        }
    }
}

/// Session 6: coalesced matvec (M=1) at the model's small/medium N shapes
/// (2048, 11008), K in {2048, 11008}, B in {1, 2} — the N=151936 shapes are
/// covered separately above (`test_q4_matvec_m1_large_n`) and against real
/// GGUF weights (`test_q4_matmul_real_gguf_token_embd`) to avoid duplicating
/// that test's large CPU-reference-buffer cost here.
#[test]
fn test_q4_matvec_coalesced_shapes() {
    let shapes = [(2048usize, 2048usize), (2048, 11008), (11008, 2048), (11008, 11008)];
    for &(k, n) in &shapes {
        let (q4_bytes, cpu_weights) = random_q4(n, k, 0xC0A1E5CE ^ (k as u64) ^ ((n as u64) << 20));
        for &b in &[1usize, 2usize] {
            let input = random_input(b, k, 0x5A1AD + k as u64 + b as u64 * 13);
            let cpu_out = cpu_matmul(&input, &cpu_weights, b, k, n);
            let gpu_out = run_gpu_matmul_b(&input, &q4_bytes, b, 1, k, n);
            assert_eq!(cpu_out.len(), gpu_out.len());

            let mut max_err = 0f32;
            for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
                max_err = max_err.max((c - g).abs());
            }
            let tol = 0.05 * (k as f32).sqrt();
            assert!(
                max_err < tol,
                "coalesced matvec K={k} N={n} B={b} M=1: max_err={max_err} exceeds tol={tol}"
            );
            println!("coalesced matvec K={k} N={n} B={b} M=1: max_err={max_err} (tol {tol})");
        }
    }
}

/// Real `token_embd.weight` (Q4_0, [151936, 2048], tied lm_head shape) from
/// the xLAM-2-3b-fc-r GGUF. Exercises the full 151936-wide dispatch and the
/// ~174MB single-buffer upload (docs/ENGINE.md §2's untested-limit concern).
#[test]
fn test_q4_matmul_real_gguf_token_embd() {
    let model_dir = std::env::var("LLM_MODEL_DIR").unwrap_or_else(|_| {
        "./models/gguf/xlam-2-3b-fc-r".to_string()
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

// ---------------------------------------------------------------------------
// Session 6 bug hunt: scratch-dequant+matmul vs naive kernel vs CPU
// reference, per-M, on the REAL GGUF's blk.0.attn_q.weight (K=2048) and
// blk.0.ffn_down.weight (K=11008) — isolating whether the split-prefill
// logit divergence (11.14 max-abs at split=1000) traces to the
// scratch-dequant/matmul path at ragged M.
// ---------------------------------------------------------------------------

fn model_dir() -> String {
    std::env::var("LLM_MODEL_DIR")
        .unwrap_or_else(|_| "./models/gguf/xlam-2-3b-fc-r".to_string())
}

/// Full CPU dequant of a raw Q4_0 `[n, k]` tensor's on-disk bytes into a
/// row-major `[n, k]` `Vec<f32>` (same 18-bytes/block layout as
/// `random_q4`'s output, but from real weights instead of synthetic ones).
fn cpu_dequant_full(bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 18;
    assert_eq!(bytes.len(), n * bytes_per_row);
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for block in 0..blocks_per_row {
            let bo = row * bytes_per_row + block * 18;
            let scale = llm_wasm::gguf::f16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
            let base = row * k + block * 32;
            for j in 0..16 {
                let byte = bytes[bo + 2 + j];
                out[base + j] = ((byte & 0x0F) as f32 - 8.0) * scale;
                out[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * scale;
            }
        }
    }
    out
}

fn run_gpu_matmul_naive(input: &[f32], q4_bytes: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let device = device();
    let weights = Q4Tensor::from_q4_bytes(q4_bytes, [n, k], &device).expect("upload Q4 weights");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input, &device).reshape([1, m, k]);
    let out = q4_matmul_naive_forced(input_t, &weights);
    out.into_data().into_vec::<f32>().expect("f32 output")
}

fn run_gpu_matmul_scratch(input: &[f32], q4_bytes: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let device = device();
    let weights = Q4Tensor::from_q4_bytes(q4_bytes, [n, k], &device).expect("upload Q4 weights");
    let input_t: Tensor<Wgpu, 3> =
        Tensor::<Wgpu, 1>::from_floats(input, &device).reshape([1, m, k]);
    let out = q4_matmul_scratch_forced(input_t, &weights);
    out.into_data().into_vec::<f32>().expect("f32 output")
}

/// Like `cpu_matmul` but only computes a sampled subset of output columns
/// `cols` (still over all `m` rows / full `k` dot product) — full N would
/// be prohibitively slow on CPU at ffn_down's K=11008 and M up to 2225
/// (M*N*K ~ 5e10 MACs). Returns `[m, cols.len()]`.
fn cpu_matmul_cols(input: &[f32], weights: &[f32], m: usize, k: usize, cols: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; m * cols.len()];
    for mi in 0..m {
        let x = &input[mi * k..(mi + 1) * k];
        for (ci, &ni) in cols.iter().enumerate() {
            let w = &weights[ni * k..(ni + 1) * k];
            out[mi * cols.len() + ci] = x.iter().zip(w).map(|(a, b)| a * b).sum();
        }
    }
    out
}

fn max_abs_rel(cpu: &[f32], gpu: &[f32]) -> (f32, f32) {
    assert_eq!(cpu.len(), gpu.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        let abs = (c - g).abs();
        max_abs = max_abs.max(abs);
        let rel = abs / c.abs().max(1e-6);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

/// Per-M comparison of the naive kernel and the scratch-dequant+Burn-matmul
/// path against a CPU f32 reference, on real GGUF weights, at ragged M
/// values spanning the `SCRATCH_MATMUL_MIN_M=32` boundary and the actual
/// chunk boundaries this bug report is about (1000, 1225, 2225) plus the
/// real tool-result size (531).
#[test]
fn test_scratch_vs_naive_vs_cpu_per_m_real_gguf() {
    let path = format!("{}/xLAM-2-3b-fc-r-q4_0.gguf", model_dir());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }
    let file = std::fs::File::open(&path).expect("open gguf");
    let reader = std::io::BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader).expect("parse gguf");

    let tensors = [
        ("blk.0.attn_q.weight", 2048usize),
        ("blk.0.ffn_down.weight", 11008usize),
        ("blk.0.ffn_gate.weight", 2048usize),
    ];
    let ms = [31usize, 32, 33, 100, 255, 256, 257, 531, 1000, 1225, 2225];

    for (name, expected_k) in tensors {
        let info = loader.reader().tensor_info(name).expect("tensor present").clone();
        let shape: Vec<usize> = info.shape().iter().rev().map(|&d| d as usize).collect();
        let (n, k) = (shape[0], shape[1]);
        assert_eq!(k, expected_k, "{name}: unexpected K");
        let bytes = loader.tensor_bytes(name).expect("read tensor bytes");
        let cpu_weights = cpu_dequant_full(&bytes, n, k);

        // Sample output columns rather than all N: full M*N*K on CPU is
        // prohibitively slow at ffn_down's K=11008, M up to 2225 (~5e10
        // MACs/M). 40 columns spread across N, including first/last.
        let num_cols = 40usize.min(n);
        let cols: Vec<usize> = (0..num_cols)
            .map(|i| (i * (n - 1)) / (num_cols - 1).max(1))
            .collect();

        println!("--- {name} [N={n}, K={k}] ---");
        for &m in &ms {
            let input = random_input(m, k, 0x51DE + (m as u64) * 7 + k as u64);
            let cpu_out = cpu_matmul_cols(&input, &cpu_weights, m, k, &cols);
            let naive_full = run_gpu_matmul_naive(&input, &bytes, m, k, n);
            let scratch_full = run_gpu_matmul_scratch(&input, &bytes, m, k, n);

            let gather = |full: &[f32]| -> Vec<f32> {
                let mut out = vec![0f32; m * cols.len()];
                for mi in 0..m {
                    for (ci, &ni) in cols.iter().enumerate() {
                        out[mi * cols.len() + ci] = full[mi * n + ni];
                    }
                }
                out
            };
            let naive_out = gather(&naive_full);
            let scratch_out = gather(&scratch_full);

            let (naive_abs, naive_rel) = max_abs_rel(&cpu_out, &naive_out);
            let (scratch_abs, scratch_rel) = max_abs_rel(&cpu_out, &scratch_out);
            println!(
                "M={m:5}: naive  max_abs={naive_abs:.6} max_rel={naive_rel:.6} | scratch max_abs={scratch_abs:.6} max_rel={scratch_rel:.6}"
            );

            // Relative error alone spuriously blows up near cpu_out ~= 0
            // (a dot product crossing zero), so gate on `rel <= 1e-3 OR abs`
            // small in absolute terms (K-length f32 dot products of O(1)
            // values — same convention as this file's other tests' `tol =
            // 0.05 * sqrt(K)`).
            let abs_tol = 0.05 * (k as f32).sqrt();
            assert!(
                naive_rel <= 1e-3 || naive_abs < abs_tol,
                "{name} M={m}: naive max_rel={naive_rel} max_abs={naive_abs} exceeds both 1e-3 rel and {abs_tol} abs"
            );
            assert!(
                scratch_rel <= 1e-3 || scratch_abs < abs_tol,
                "{name} M={m}: scratch max_rel={scratch_rel} max_abs={scratch_abs} exceeds both 1e-3 rel and {abs_tol} abs"
            );
        }
    }
}

/// Isolates the dequant kernel (`shader_q4_dequant.wgsl` via
/// `q4_dequant_scratch_to_vec`) from the matmul that consumes it: dequant
/// `blk.0.ffn_down.weight` and compare every element against the CPU
/// dequant reference. Output layout is transposed `[K, N]`
/// (`out[k*N+n]`) vs the CPU reference's row-major `[N, K]`
/// (`cpu[n*K+k]`).
#[test]
fn test_dequant_scratch_matches_cpu_real_gguf() {
    let path = format!("{}/xLAM-2-3b-fc-r-q4_0.gguf", model_dir());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }
    let file = std::fs::File::open(&path).expect("open gguf");
    let reader = std::io::BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader).expect("parse gguf");

    let name = "blk.0.ffn_down.weight";
    let info = loader.reader().tensor_info(name).expect("tensor present").clone();
    let shape: Vec<usize> = info.shape().iter().rev().map(|&d| d as usize).collect();
    let (n, k) = (shape[0], shape[1]);
    let bytes = loader.tensor_bytes(name).expect("read tensor bytes");
    let cpu_weights = cpu_dequant_full(&bytes, n, k); // [N, K] row-major

    let device = device();
    let weights = Q4Tensor::from_q4_bytes(&bytes, [n, k], &device).expect("upload Q4 weights");
    let gpu_kt_n = q4_dequant_scratch_to_vec(&weights, &device); // [K, N] row-major

    assert_eq!(gpu_kt_n.len(), n * k);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut worst = (0usize, 0usize);
    for row in 0..n {
        for col in 0..k {
            let cpu_v = cpu_weights[row * k + col];
            let gpu_v = gpu_kt_n[col * n + row];
            let abs = (cpu_v - gpu_v).abs();
            let rel = abs / cpu_v.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
                worst = (row, col);
            }
            max_rel = max_rel.max(rel);
        }
    }
    println!(
        "dequant {name} [N={n},K={k}]: max_abs={max_abs} max_rel={max_rel} worst=(row={},col={})",
        worst.0, worst.1
    );
    assert!(max_abs < 1e-3, "dequant max_abs={max_abs} exceeds 1e-3 at {worst:?}");
}

// ---------------------------------------------------------------------------
// Session 10: autotune-bucket padding (docs/BENCHMARKS.md Session 10). The
// scratch-dequant+matmul path now pads M up to `pad_m_bucket`'s fixed
// buckets before calling `Tensor::matmul`, so distinct prefill lengths
// don't each pay a fresh autotune pass in the browser (no persistent
// autotune cache there). This asserts padding is numerically inert: calling
// the public API at M gives the same first-M rows as calling it at the
// bucket boundary `pad_m_bucket(M)` (manually zero-padded here) and slicing
// — i.e. zero rows appended by the internal padding don't perturb the real
// rows' matmul output.
// ---------------------------------------------------------------------------

#[test]
fn test_scratch_matmul_padding_is_numerically_inert() {
    let k = 2048usize;
    let n = 96usize;
    let (bytes, _cpu_weights) = random_q4(n, k, 0x9EC0DE);

    for &m in &[40usize, 100, 531, 1225] {
        let input = random_input(m, k, 0xBADD1E + m as u64);
        let out_m = run_gpu_matmul_scratch(&input, &bytes, m, k, n);

        let padded_m = pad_m_bucket_for_test(m);
        assert!(padded_m >= m, "M={m}: padded_m={padded_m} must be >= m");
        assert_eq!(
            padded_m % if m < 128 { 32 } else { 128 },
            0,
            "M={m}: padded_m={padded_m} not aligned to expected bucket"
        );

        let mut padded_input = input.clone();
        padded_input.resize(padded_m * k, 0.0);
        let out_padded = run_gpu_matmul_scratch(&padded_input, &bytes, padded_m, k, n);

        let mut max_diff = 0f32;
        for row in 0..m {
            for col in 0..n {
                let a = out_m[row * n + col];
                let b = out_padded[row * n + col];
                max_diff = max_diff.max((a - b).abs());
            }
        }
        assert!(
            max_diff < 1e-6,
            "M={m} (padded to {padded_m}): first-M-rows max_diff={max_diff} exceeds 1e-6"
        );
        println!("M={m:5} padded_to={padded_m:5}: max_diff={max_diff:.3e}");
    }
}

/// What the prefill GEMM actually achieves, with no readback per call and
/// no dequant amortization question: one warm-up, then `iters` back-to-back
/// `q4_matmul` calls at llm-life's Qwen2.5-0.5B FFN shapes, with a single
/// readback at the end to sync. This is the number that decides whether the
/// remaining gap to the 5 s / 20 s forward gates is the GEMM kernel or the
/// pipeline around it.
///
/// `cargo test --release --test q4_matmul -- --ignored --nocapture
/// bench_prefill_gemm_shapes`
#[test]
#[ignore]
fn bench_prefill_gemm_shapes() {
    let device = device();
    for (label, m, k, n) in [
        ("ffn_gate/up", 4164usize, 896usize, 4864usize),
        ("ffn_down", 4164, 4864, 896),
        ("attn_q/o", 4164, 896, 896),
        ("square_2048", 2048, 2048, 2048),
    ] {
        let (q4_bytes, _) = random_q4(n, k, 0xABCD ^ (k as u64));
        let weights = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("upload");
        let input = random_input(m, k, 0x1234);
        let input_t: Tensor<Wgpu, 3> =
            Tensor::<Wgpu, 1>::from_floats(input.as_slice(), &device).reshape([1, m, k]);

        let _ = q4_matmul(input_t.clone(), &weights)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        let iters = 10;
        let t0 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(q4_matmul(input_t.clone(), &weights));
        }
        let _ = last.unwrap().into_data().into_vec::<f32>().unwrap();
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        println!(
            "{label:>12} M={m} K={k} N={n}: {:.2} ms/call, {:.0} GFLOP/s",
            dt * 1000.0,
            2.0 * m as f64 * n as f64 * k as f64 / dt / 1e9
        );
    }
}
