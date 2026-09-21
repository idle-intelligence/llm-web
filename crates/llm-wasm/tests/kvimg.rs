//! Tests for the prefix-KV-image format (`kvimg.rs`) and, feature-gated,
//! the `KvCache::export_prefix`/`import_prefix` GPU round trip (`kv.rs`).

use llm_wasm::kvimg::{Dtype, Header, KvImage, KvImageError};

fn tiny_header(n_layers: usize, n_kv_heads: usize, n_tokens: usize, head_dim: usize) -> Header {
    Header {
        model_fingerprint: "deadbeef".to_string(),
        prefix_key: llm_wasm::kvimg::prefix_key("deadbeef", "<system>hi<tools/></system>"),
        tokens: (0..n_tokens as u32).collect(),
        n_layers,
        n_kv_heads,
        head_dim,
        dtype: "f32".to_string(),
        engine: "llm-wasm/0.1.0".to_string(),
        created: "2026-09-10".to_string(),
    }
}

fn synth_layers(
    n_layers: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    head_dim: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..n_layers)
        .map(|l| {
            let len = n_kv_heads * n_tokens * head_dim;
            let k: Vec<f32> = (0..len).map(|i| (l * 1000 + i) as f32).collect();
            let v: Vec<f32> = (0..len).map(|i| -((l * 1000 + i) as f32)).collect();
            (k, v)
        })
        .collect()
}

#[test]
fn roundtrip_write_read_header_and_slices() {
    let n_layers = 3;
    let n_kv_heads = 2;
    let n_tokens = 5;
    let head_dim = 4;

    let header = tiny_header(n_layers, n_kv_heads, n_tokens, head_dim);
    let layers = synth_layers(n_layers, n_kv_heads, n_tokens, head_dim);

    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::F32, layer_refs).unwrap();

    let (read_back_header, data_offset) = KvImage::read_header(&buf).unwrap();
    assert_eq!(read_back_header, header);
    assert_eq!(data_offset % 16, 0, "data offset must be 16-byte aligned");

    let slices = KvImage::layer_slices_at(&buf, &read_back_header, data_offset).unwrap();
    assert_eq!(slices.len(), n_layers);
    for (layer, (expected_k, expected_v)) in layers.iter().enumerate() {
        assert_eq!(slices[layer].k.as_ref(), expected_k.as_slice());
        assert_eq!(slices[layer].v.as_ref(), expected_v.as_slice());
    }
}

#[test]
fn corrupt_magic_is_err() {
    let header = tiny_header(1, 1, 1, 1);
    let layers = synth_layers(1, 1, 1, 1);
    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::F32, layer_refs).unwrap();

    buf[0] = b'X';
    let err = KvImage::read_header(&buf).unwrap_err();
    assert!(matches!(err, KvImageError::BadMagic { .. }));
}

#[test]
fn corrupt_version_is_err() {
    let header = tiny_header(1, 1, 1, 1);
    let layers = synth_layers(1, 1, 1, 1);
    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::F32, layer_refs).unwrap();

    buf[6..10].copy_from_slice(&99u32.to_le_bytes());
    let err = KvImage::read_header(&buf).unwrap_err();
    assert!(matches!(err, KvImageError::UnsupportedVersion(99)));
}

#[test]
fn truncated_buffer_is_err() {
    let header = tiny_header(2, 2, 5, 4);
    let layers = synth_layers(2, 2, 5, 4);
    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::F32, layer_refs).unwrap();

    // Truncate mid-header.
    let short = &buf[..8];
    assert!(matches!(
        KvImage::read_header(short),
        Err(KvImageError::Truncated { .. })
    ));

    // Truncate mid-data: header parses fine, but layer_slices_at must fail.
    let (parsed_header, data_offset) = KvImage::read_header(&buf).unwrap();
    let truncated_data = &buf[..data_offset + 4];
    assert!(matches!(
        KvImage::layer_slices_at(truncated_data, &parsed_header, data_offset),
        Err(KvImageError::Truncated { .. })
    ));
}

#[test]
fn prefix_key_is_deterministic_and_input_sensitive() {
    let a = llm_wasm::kvimg::prefix_key("hash1", "system+tools text");
    let b = llm_wasm::kvimg::prefix_key("hash1", "system+tools text");
    let c = llm_wasm::kvimg::prefix_key("hash1", "different text");
    let d = llm_wasm::kvimg::prefix_key("hash2", "system+tools text");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
    assert_eq!(a.len(), 64, "hex-encoded sha256 is 64 chars");
}

/// Tiny deterministic xorshift64* PRNG — no `rand` crate in this
/// workspace's dependency tree and this crate doesn't own `Cargo.toml`
/// anyway (same reasoning as `kvimg.rs`'s home-rolled SHA-256).
struct XorShift64(u64);
impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 // uniform in [0, 1)
    }
    /// K/V-shaped value: mostly |x| < 10, rare outliers up to 100.
    fn next_kv_value(&mut self) -> f32 {
        let sign = if self.next_u64().is_multiple_of(2) { 1.0 } else { -1.0 };
        let outlier = self.next_u64().is_multiple_of(200); // ~0.5% outlier rate
        let mag = if outlier {
            10.0 + self.next_f32() * 90.0
        } else {
            self.next_f32() * 10.0
        };
        sign * mag
    }
}

fn synth_kv_values(rng: &mut XorShift64, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_kv_value()).collect()
}

#[test]
fn q8_0_quantize_dequantize_roundtrip_error_bounds() {
    use llm_wasm::kvimg::{dequantize_q8_0, quantize_q8_0};

    let mut rng = XorShift64(0x5EED_C0FF_EE15_A5A5);
    let n_blocks = 4096;
    let vals = synth_kv_values(&mut rng, n_blocks * 32);

    let (scales, words) = quantize_q8_0(&vals);
    let recovered = dequantize_q8_0(&scales, &words, vals.len());
    assert_eq!(recovered.len(), vals.len());

    let mut sum_rel_err = 0f64;
    let mut n_rel_samples = 0u64;
    for (block_idx, (orig_block, rec_block)) in vals
        .chunks_exact(32)
        .zip(recovered.chunks_exact(32))
        .enumerate()
    {
        let scale = scales[block_idx]; // == absmax(block) / 127, the per-block max-abs-error bound
        for (&orig, &rec) in orig_block.iter().zip(rec_block.iter()) {
            let err = (orig - rec).abs();
            assert!(
                err <= scale + 1e-6,
                "block {block_idx}: |{orig} - {rec}| = {err} exceeds per-block bound absmax/127={scale}"
            );
            if orig.abs() > 1e-3 {
                sum_rel_err += (err / orig.abs()) as f64;
                n_rel_samples += 1;
            }
        }
    }
    let mean_rel_err = sum_rel_err / n_rel_samples as f64;
    eprintln!("q8_0 roundtrip: {n_blocks} blocks, mean relative error = {mean_rel_err:.6}");
    assert!(
        mean_rel_err < 0.05,
        "mean relative error {mean_rel_err} too high for q8_0"
    );
}

fn q8_0_header(n_layers: usize, n_kv_heads: usize, n_tokens: usize, head_dim: usize) -> Header {
    Header {
        model_fingerprint: "deadbeef".to_string(),
        prefix_key: llm_wasm::kvimg::prefix_key("deadbeef", "<system>hi<tools/></system>"),
        tokens: (0..n_tokens as u32).collect(),
        n_layers,
        n_kv_heads,
        head_dim,
        dtype: "q8_0".to_string(),
        engine: "llm-wasm/0.1.0".to_string(),
        created: "2026-09-10".to_string(),
    }
}

#[test]
fn q8_0_file_roundtrip_write_read() {
    let mut rng = XorShift64(0xA11C_EB0B_1234_5678);
    let n_layers = 3;
    let n_kv_heads = 2;
    let n_tokens = 17;
    let head_dim = 128; // 4 blocks of 32 per row, matches xLAM-2-3b-fc-r

    let header = q8_0_header(n_layers, n_kv_heads, n_tokens, head_dim);
    let len = n_kv_heads * n_tokens * head_dim;
    let layers: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
        .map(|_| {
            (
                synth_kv_values(&mut rng, len),
                synth_kv_values(&mut rng, len),
            )
        })
        .collect();

    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::Q8_0, layer_refs).unwrap();

    let (read_back_header, data_offset) = KvImage::read_header(&buf).unwrap();
    assert_eq!(read_back_header, header);

    for (layer, (expected_k, expected_v)) in layers.iter().enumerate() {
        let (k, v) = KvImage::layer_f32(&buf, &read_back_header, data_offset, layer).unwrap();
        assert_eq!(k.len(), expected_k.len());
        assert_eq!(v.len(), expected_v.len());
        for (a, b) in k.iter().zip(expected_k.iter()) {
            assert!(
                (a - b).abs() <= 10.0 / 127.0 + 90.0 / 127.0 + 1e-3,
                "k mismatch too large: {a} vs {b}"
            );
        }
        for (a, b) in v.iter().zip(expected_v.iter()) {
            assert!(
                (a - b).abs() <= 10.0 / 127.0 + 90.0 / 127.0 + 1e-3,
                "v mismatch too large: {a} vs {b}"
            );
        }
    }

    // Size ratio vs f32 for the same dims: f32 payload is
    // n_layers * 2(k+v) * n_kv_heads * n_tokens * head_dim * 4 bytes;
    // q8_0 payload is the same tensor count at 36 bytes / 32 values
    // (1.125 B/value) instead of 4 B/value, i.e. exactly 4/1.125 = 3.555..x
    // smaller (header/padding overhead is identical between the two and
    // excluded from this ratio).
    let per_tensor = n_kv_heads * n_tokens * head_dim;
    let n_blocks = per_tensor / 32;
    let f32_payload_bytes = n_layers * 2 * per_tensor * 4;
    let q8_0_payload_bytes = n_layers * 2 * n_blocks * 36;
    let ratio = f32_payload_bytes as f64 / q8_0_payload_bytes as f64;
    eprintln!("q8_0 payload size ratio vs f32: {ratio:.3}x (f32={f32_payload_bytes}B, q8_0={q8_0_payload_bytes}B)");
    assert!(
        (ratio - 3.5556).abs() < 0.01,
        "expected ~3.5556x smaller, got {ratio:.4}x"
    );
}

#[test]
fn v1_f32_images_still_read() {
    // Regression: a v1 image written with dtype "f32" (the original,
    // pre-q8_0 format) must still round-trip byte-for-byte through
    // read_header + layer_slices_at, unchanged by the q8_0 addition.
    let n_layers = 2;
    let n_kv_heads = 2;
    let n_tokens = 5;
    let head_dim = 4;
    let header = tiny_header(n_layers, n_kv_heads, n_tokens, head_dim);
    let layers = synth_layers(n_layers, n_kv_heads, n_tokens, head_dim);

    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvImage::write(&mut buf, &header, Dtype::F32, layer_refs).unwrap();

    assert_eq!(
        llm_wasm::kvimg::VERSION,
        1,
        "v1 container format must be unchanged by q8_0 support"
    );
    let (read_back_header, data_offset) = KvImage::read_header(&buf).unwrap();
    assert_eq!(read_back_header.dtype, "f32");
    let slices = KvImage::layer_slices_at(&buf, &read_back_header, data_offset).unwrap();
    for (layer, (expected_k, expected_v)) in layers.iter().enumerate() {
        assert_eq!(slices[layer].k.as_ref(), expected_k.as_slice());
        assert_eq!(slices[layer].v.as_ref(), expected_v.as_slice());
    }
}

#[cfg(feature = "wgpu")]
mod gpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;
    use llm_wasm::kv::{KvCache, KvDtype};

    /// Only run when the GPU is free — this crate's other worker owns
    /// `full_forward`/`llm-agent` GPU usage; poll `pgrep -fl
    /// "full_forward|llm-agent|Chromium"` before running this test
    /// manually. `cargo test` alone will still attempt to acquire a wgpu
    /// device; if none is available in this environment the test is
    /// expected to be skipped/ignored by the runner environment rather
    /// than by code here (no portable "is a GPU available" check exists
    /// in Burn's public API at HEAD).
    #[test]
    fn export_import_prefix_bit_exact_then_append_matches() {
        // Explicit `KvDtype::F32`: this test's `head_dim=4` isn't a
        // multiple of 32, so it can't exercise the (now-default)
        // `KvDtype::Q8_0` mode — see the `q8_*` tests below for that mode's
        // coverage (Session 13, docs/BENCHMARKS.md).
        let device = WgpuDevice::default();
        let num_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let max_ctx = 16;
        let n_tokens = 5;

        let mut src = KvCache::new_with_dtype(num_layers, n_kv_heads, head_dim, max_ctx, &device, KvDtype::F32);
        for pos in 0..n_tokens {
            for layer in 0..num_layers {
                let shape = [1, n_kv_heads, 1, head_dim];
                let base = (layer * 1000 + pos * 10) as f32;
                let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim))
                    .map(|i| base + i as f32)
                    .collect();
                let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim))
                    .map(|i| -(base + i as f32))
                    .collect();
                let k = Tensor::<Wgpu, 4>::from_data(
                    burn::tensor::TensorData::new(k_vals, shape),
                    &device,
                );
                let v = Tensor::<Wgpu, 4>::from_data(
                    burn::tensor::TensorData::new(v_vals, shape),
                    &device,
                );
                src.write(layer, k, v);
            }
            src.advance(1);
        }

        let exported = src.export_prefix(n_tokens);

        let mut dst = KvCache::new_with_dtype(num_layers, n_kv_heads, head_dim, max_ctx, &device, KvDtype::F32);
        dst.import_prefix(&exported, n_tokens);
        assert_eq!(dst.len(), n_tokens);

        let reexported = dst.export_prefix(n_tokens);
        assert_eq!(
            exported, reexported,
            "import_prefix must reproduce export_prefix bit-for-bit"
        );

        // Append one more row on both caches and confirm they still agree.
        for layer in 0..num_layers {
            let shape = [1, n_kv_heads, 1, head_dim];
            let base = 9999.0f32;
            let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim))
                .map(|i| base + i as f32)
                .collect();
            let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim))
                .map(|i| -(base + i as f32))
                .collect();

            let k_src = Tensor::<Wgpu, 4>::from_data(
                burn::tensor::TensorData::new(k_vals.clone(), shape),
                &device,
            );
            let v_src = Tensor::<Wgpu, 4>::from_data(
                burn::tensor::TensorData::new(v_vals.clone(), shape),
                &device,
            );
            src.write(layer, k_src, v_src);

            let k_dst =
                Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(k_vals, shape), &device);
            let v_dst =
                Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(v_vals, shape), &device);
            dst.write(layer, k_dst, v_dst);
        }
        src.advance(1);
        dst.advance(1);

        let src_final = src.export_prefix(n_tokens + 1);
        let dst_final = dst.export_prefix(n_tokens + 1);
        assert_eq!(
            src_final, dst_final,
            "post-import append must match a never-imported cache"
        );
    }

    // -----------------------------------------------------------------
    // Session 13 (docs/BENCHMARKS.md): KvDtype::Q8_0 tests.
    // -----------------------------------------------------------------

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        }
    }

    /// q8 roundtrip on random rows: write via `KvCache` (Q8_0 mode),
    /// dequantize back via `read_or_dequant_f32`, and check every value is
    /// within its block's `absmax/127` quantization step of the original —
    /// the same bound `kvimg.rs`'s `q8_0_quantize_dequantize_roundtrip_error_bounds`
    /// checks for the file-format encoder.
    #[test]
    fn q8_append_roundtrip_error_bound() {
        let device = WgpuDevice::default();
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 64;
        let n_tokens = 17;

        let mut cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        let mut rng = Lcg(42);
        let mut original = vec![0f32; n_kv_heads * n_tokens * head_dim];
        for pos in 0..n_tokens {
            let shape = [1, n_kv_heads, 1, head_dim];
            let mut row = vec![0f32; n_kv_heads * head_dim];
            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    let x = rng.next_f32() * 20.0;
                    row[h * head_dim + d] = x;
                    original[(h * n_tokens + pos) * head_dim + d] = x;
                }
            }
            let k = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(row.clone(), shape), &device);
            let v = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(row, shape), &device);
            cache.write(0, k, v);
            cache.advance(1);
        }

        let (k_deq, _v_deq) = cache.read_or_dequant_f32(0, n_tokens);
        let dequantized = k_deq.into_data().into_vec::<f32>().unwrap();
        assert_eq!(dequantized.len(), original.len());

        for h in 0..n_kv_heads {
            for pos in 0..n_tokens {
                for block in 0..(head_dim / 32) {
                    let base = (h * n_tokens + pos) * head_dim + block * 32;
                    let block_vals = &original[base..base + 32];
                    let absmax = block_vals.iter().fold(0f32, |m, &x| m.max(x.abs()));
                    let bound = absmax / 127.0;
                    for d in 0..32 {
                        let err = (dequantized[base + d] - original[base + d]).abs();
                        assert!(
                            err <= bound + 1e-6,
                            "q8 roundtrip error {err} exceeds bound {bound} at h={h} pos={pos} d={d}"
                        );
                    }
                }
            }
        }
    }

    /// Diagnostic added while chasing a full_forward regression: does a
    /// single bulk `write` of `T=31` rows (prefill's shape) quantize
    /// identically to 31 separate `T=1` `write` calls (decode's shape,
    /// already covered by `q8_append_roundtrip_error_bound`)? If this
    /// fails but the T=1 test above passes, the bug is specifically in
    /// `shader_kv_quantize.wgsl`'s multi-row indexing.
    #[test]
    fn q8_bulk_write_matches_row_by_row() {
        let device = WgpuDevice::default();
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 64;
        let t = 31;

        let mut rng = Lcg(99);
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for _ in 0..t {
            rows.push((0..(n_kv_heads * head_dim)).map(|_| rng.next_f32() * 20.0).collect());
        }

        // Bulk: one write of all T rows at once (prefill shape).
        let mut bulk_flat = vec![0f32; n_kv_heads * t * head_dim];
        for (pos, row) in rows.iter().enumerate() {
            for h in 0..n_kv_heads {
                bulk_flat[(h * t + pos) * head_dim..(h * t + pos) * head_dim + head_dim]
                    .copy_from_slice(&row[h * head_dim..h * head_dim + head_dim]);
            }
        }
        let mut bulk_cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        let bulk_shape = [1, n_kv_heads, t, head_dim];
        let k_bulk = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(bulk_flat, bulk_shape), &device);
        bulk_cache.write(0, k_bulk, Tensor::<Wgpu, 4>::zeros(bulk_shape, &device));
        bulk_cache.advance(t);

        // Row-by-row: T=1 writes (decode shape), same source data.
        let mut row_cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        for row in &rows {
            let shape = [1, n_kv_heads, 1, head_dim];
            let k = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(row.clone(), shape), &device);
            let v = Tensor::<Wgpu, 4>::zeros(shape, &device);
            row_cache.write(0, k, v);
            row_cache.advance(1);
        }

        let (bulk_k, _) = bulk_cache.read_or_dequant_f32(0, t);
        let (row_k, _) = row_cache.read_or_dequant_f32(0, t);
        let bulk_data = bulk_k.into_data().into_vec::<f32>().unwrap();
        let row_data = row_k.into_data().into_vec::<f32>().unwrap();

        let mut max_diff = 0f32;
        for (a, b) in bulk_data.iter().zip(row_data.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("q8_bulk_write_matches_row_by_row: max_diff={max_diff}");
        assert_eq!(bulk_data, row_data, "bulk (T={t}) write must dequantize identically to T=1 writes");
    }

    /// `import_prefix` from a q8_0 image must be bit-identical to `write`ing
    /// the same (already-quantized) rows directly — checked here by round
    /// tripping through `export_prefix_q8`/`import_prefix_q8` (no dequant
    /// step, per Session 13's design) and comparing to a fresh cache built
    /// by `write`ing the same source rows.
    #[test]
    fn import_prefix_q8_bit_identical_to_write() {
        let device = WgpuDevice::default();
        let num_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 64;
        let n_tokens = 9;

        let mut src = KvCache::new_with_dtype(num_layers, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        let mut rng = Lcg(7);
        for _pos in 0..n_tokens {
            for layer in 0..num_layers {
                let shape = [1, n_kv_heads, 1, head_dim];
                let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|_| rng.next_f32() * 15.0).collect();
                let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|_| rng.next_f32() * 15.0).collect();
                let k = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(k_vals, shape), &device);
                let v = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(v_vals, shape), &device);
                src.write(layer, k, v);
            }
            src.advance(1);
        }

        let exported = src.export_prefix_q8(n_tokens);
        let mut dst = KvCache::new_with_dtype(num_layers, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        dst.import_prefix_q8(&exported, n_tokens);
        assert_eq!(dst.len(), n_tokens);

        let reexported = dst.export_prefix_q8(n_tokens);
        for layer in 0..num_layers {
            assert_eq!(exported[layer].k_scales, reexported[layer].k_scales, "layer {layer} k_scales");
            assert_eq!(exported[layer].k_words, reexported[layer].k_words, "layer {layer} k_words");
            assert_eq!(exported[layer].v_scales, reexported[layer].v_scales, "layer {layer} v_scales");
            assert_eq!(exported[layer].v_words, reexported[layer].v_words, "layer {layer} v_words");
        }
    }

    /// Decode-time (M=1) fused q8_0 attention (`gguf::attn_decode_q8_dispatch`)
    /// vs a CPU f64 reference computed on the *same* (already-quantized,
    /// then dequantized-back) K/V — isolates the kernel's own math from
    /// quantization error, per the task brief's "vs the f32 path <= 1e-3
    /// rel" requirement. GQA: `n_heads=4`, `n_kv_heads=2` (kv_head =
    /// h / n_rep, matching `model.rs::repeat_kv`'s head order).
    #[test]
    fn q8_decode_attention_matches_f64_reference() {
        use burn::backend::wgpu::WgpuRuntime;
        use cubecl::Runtime;

        let device = WgpuDevice::default();
        let n_heads = 4;
        let n_kv_heads = 2;
        let n_rep = n_heads / n_kv_heads;
        let head_dim = 128;
        let max_ctx = 6000;

        for &kv_len in &[31usize, 2225, 5000] {
            let mut cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
            let mut rng = Lcg(1000 + kv_len as u64);
            for _pos in 0..kv_len {
                let shape = [1, n_kv_heads, 1, head_dim];
                let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|_| rng.next_f32() * 6.0).collect();
                let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|_| rng.next_f32() * 6.0).collect();
                let k = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(k_vals, shape), &device);
                let v = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(v_vals, shape), &device);
                cache.write(0, k, v);
                cache.advance(1);
            }

            let mut q = vec![0f32; n_heads * head_dim];
            for x in q.iter_mut() {
                *x = rng.next_f32() * 4.0;
            }

            let (k_deq, v_deq) = cache.read_or_dequant_f32(0, kv_len);
            let k_deq = k_deq.into_data().into_vec::<f32>().unwrap();
            let v_deq = v_deq.into_data().into_vec::<f32>().unwrap();

            let scale = (head_dim as f32).powf(-0.5);
            let mut reference = vec![0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let kv_head = h / n_rep;
                let mut scores = vec![0f64; kv_len];
                for j in 0..kv_len {
                    let mut dot = 0f64;
                    for d in 0..head_dim {
                        dot += q[h * head_dim + d] as f64 * k_deq[(kv_head * kv_len + j) * head_dim + d] as f64;
                    }
                    scores[j] = dot * scale as f64;
                }
                let m = scores.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0f64;
                    for (j, &e) in exps.iter().enumerate() {
                        acc += (e / sum) * v_deq[(kv_head * kv_len + j) * head_dim + d] as f64;
                    }
                    reference[h * head_dim + d] = acc as f32;
                }
            }

            let client = WgpuRuntime::client(&device);
            let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
            let q_handle = client.create_from_slice(&q_bytes);
            let scratch = client.empty(n_heads * max_ctx * 4);
            let (ks, kw, vs, vw) = cache.q8_layer(0);
            let out_handle = llm_wasm::gguf::attn_decode_q8_dispatch(
                &client, &q_handle, ks, kw, vs, vw, &scratch, n_heads, n_kv_heads, head_dim, kv_len, max_ctx, scale,
            );
            let out_bytes = client.read_one(out_handle);
            let out: Vec<f32> = out_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let mut max_rel = 0f32;
            for (a, b) in out.iter().zip(reference.iter()) {
                let denom = b.abs().max(1e-6);
                max_rel = max_rel.max((a - b).abs() / denom);
            }
            println!("q8_decode_attention kv_len={kv_len}: max_rel={max_rel}");
            assert!(
                max_rel <= 1e-3,
                "kv_len={kv_len}: fused q8 decode attention diverges from f64 reference by {max_rel} (rel)"
            );
        }
    }
}
