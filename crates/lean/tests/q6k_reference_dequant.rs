//! Checks `gguf::dequantize_q6_k` against gguf-py's own reference dequant
//! (`gguf.quants.dequantize(..., GGMLQuantizationType.Q6_K)`) on real Q6_K
//! super-blocks pulled straight out of Qwen2.5-3B-Instruct-GGUF's
//! `output.weight` tensor - not synthetic bytes (`quant.rs`'s/
//! `cpu_kernels.rs`'s unit tests already cover the synthetic-block/
//! internal-consistency side). See `reference/gen_q6k_fixture.py` for how
//! `reference/q6k_reference_blocks.json` was produced - committed, unlike
//! the full GGUF fixtures, since it holds only 8 blocks (a few KB), not
//! model weights.

use serde::Deserialize;

#[derive(Deserialize)]
struct Block {
    block_index_in_tensor: u64,
    bytes_hex: String,
    expected: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixture {
    block_bytes: usize,
    qk_k: usize,
    blocks: Vec<Block>,
}

#[test]
fn q6_k_matches_gguf_py_reference_on_real_blocks() {
    let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/q6k_reference_blocks.json")).expect("reading q6k_reference_blocks.json");
    let fixture: Fixture = serde_json::from_str(&json).expect("parsing q6k_reference_blocks.json");
    assert_eq!(fixture.block_bytes, 210);
    assert_eq!(fixture.qk_k, 256);
    assert!(!fixture.blocks.is_empty());

    for block in &fixture.blocks {
        let bytes: Vec<u8> = (0..block.bytes_hex.len() / 2).map(|i| u8::from_str_radix(&block.bytes_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        assert_eq!(bytes.len(), 210);
        let got = lean::gguf::dequantize_q6_k(&bytes, 256);
        assert_eq!(got.len(), block.expected.len());
        for (i, (&g, &e)) in got.iter().zip(&block.expected).enumerate() {
            let diff = (g as f64 - e).abs();
            assert!(diff < 1e-3, "block {}: value {i} mismatch: ours={g} gguf-py={e} diff={diff}", block.block_index_in_tensor);
        }
    }
}
