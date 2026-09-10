//! Tests for the prefix-KV-image format (`kvimg.rs`) and, feature-gated,
//! the `KvCache::export_prefix`/`import_prefix` GPU round trip (`kv.rs`).

use llm_wasm::kvimg::{Header, KvImage, KvImageError};

fn tiny_header(n_layers: usize, n_kv_heads: usize, n_tokens: usize, head_dim: usize) -> Header {
    Header {
        model_hash: "deadbeef".to_string(),
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

fn synth_layers(n_layers: usize, n_kv_heads: usize, n_tokens: usize, head_dim: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
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
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, layer_refs).unwrap();

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
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, layer_refs).unwrap();

    buf[0] = b'X';
    let err = KvImage::read_header(&buf).unwrap_err();
    assert!(matches!(err, KvImageError::BadMagic { .. }));
}

#[test]
fn corrupt_version_is_err() {
    let header = tiny_header(1, 1, 1, 1);
    let layers = synth_layers(1, 1, 1, 1);
    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, layer_refs).unwrap();

    buf[6..10].copy_from_slice(&99u32.to_le_bytes());
    let err = KvImage::read_header(&buf).unwrap_err();
    assert!(matches!(err, KvImageError::UnsupportedVersion(99)));
}

#[test]
fn truncated_buffer_is_err() {
    let header = tiny_header(2, 2, 5, 4);
    let layers = synth_layers(2, 2, 5, 4);
    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, layer_refs).unwrap();

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

#[cfg(feature = "wgpu")]
mod gpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;
    use llm_wasm::kv::KvCache;

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
        let device = WgpuDevice::default();
        let num_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let max_ctx = 16;
        let n_tokens = 5;

        let mut src = KvCache::new(num_layers, n_kv_heads, head_dim, max_ctx, &device);
        for pos in 0..n_tokens {
            for layer in 0..num_layers {
                let shape = [1, n_kv_heads, 1, head_dim];
                let base = (layer * 1000 + pos * 10) as f32;
                let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|i| base + i as f32).collect();
                let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|i| -(base + i as f32)).collect();
                let k = Tensor::<Wgpu, 4>::from_data(
                    burn::tensor::TensorData::new(k_vals, shape),
                    &device,
                );
                let v = Tensor::<Wgpu, 4>::from_data(
                    burn::tensor::TensorData::new(v_vals, shape),
                    &device,
                );
                src.append(layer, k, v);
            }
            src.advance(1);
        }

        let exported = src.export_prefix(n_tokens);

        let mut dst = KvCache::new(num_layers, n_kv_heads, head_dim, max_ctx, &device);
        dst.import_prefix(&exported, n_tokens);
        assert_eq!(dst.len(), n_tokens);

        let reexported = dst.export_prefix(n_tokens);
        assert_eq!(exported, reexported, "import_prefix must reproduce export_prefix bit-for-bit");

        // Append one more row on both caches and confirm they still agree.
        for layer in 0..num_layers {
            let shape = [1, n_kv_heads, 1, head_dim];
            let base = 9999.0f32;
            let k_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|i| base + i as f32).collect();
            let v_vals: Vec<f32> = (0..(n_kv_heads * head_dim)).map(|i| -(base + i as f32)).collect();

            let k_src = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(k_vals.clone(), shape), &device);
            let v_src = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(v_vals.clone(), shape), &device);
            src.append(layer, k_src, v_src);

            let k_dst = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(k_vals, shape), &device);
            let v_dst = Tensor::<Wgpu, 4>::from_data(burn::tensor::TensorData::new(v_vals, shape), &device);
            dst.append(layer, k_dst, v_dst);
        }
        src.advance(1);
        dst.advance(1);

        let src_final = src.export_prefix(n_tokens + 1);
        let dst_final = dst.export_prefix(n_tokens + 1);
        assert_eq!(src_final, dst_final, "post-import append must match a never-imported cache");
    }
}
