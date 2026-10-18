//! Checks `gguf::dequantize_{q4_k,q5_0,f16,bf16}` against gguf-py's own
//! reference dequantization on real tensor blocks, bit for bit. See
//! `reference/gen_dequant_blocks.py` for how
//! `reference/dequant_reference_blocks.json` was produced (a few KB, no model
//! weights). The ignored test compares a whole tensor against the reference
//! the same script writes with `--full`.

use crate::gguf::{dequantize_bf16, dequantize_f16, dequantize_q4_k, dequantize_q5_0, GgmlDtype, GgufReader};
use serde::Deserialize;

#[derive(Deserialize)]
struct Block {
    block_index_in_tensor: u64,
    bytes_hex: String,
    expected_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(rename = "type")]
    ty: String,
    block_bytes: usize,
    block_values: usize,
    blocks: Vec<Block>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn dequant(ty: &str, bytes: &[u8], n: usize) -> Vec<f32> {
    match ty {
        "Q4_K" => dequantize_q4_k(bytes, n),
        "Q5_0" => dequantize_q5_0(bytes, n),
        "F16" => dequantize_f16(bytes, n),
        "BF16" => dequantize_bf16(bytes, n),
        other => panic!("unknown type {other}"),
    }
}

#[test]
fn new_types_match_gguf_py_reference_bit_for_bit_on_real_blocks() {
    let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/dequant_reference_blocks.json")).expect("reading dequant_reference_blocks.json");
    let fixture: Fixture = serde_json::from_str(&json).expect("parsing dequant_reference_blocks.json");
    assert_eq!(fixture.cases.len(), 4);
    for case in &fixture.cases {
        assert!(!case.blocks.is_empty());
        for block in &case.blocks {
            let bytes: Vec<u8> = (0..block.bytes_hex.len() / 2).map(|i| u8::from_str_radix(&block.bytes_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
            assert_eq!(bytes.len(), case.block_bytes);
            let got = dequant(&case.ty, &bytes, case.block_values);
            assert_eq!(got.len(), block.expected_bits.len());
            for (i, (&g, &e)) in got.iter().zip(&block.expected_bits).enumerate() {
                assert_eq!(g.to_bits(), e, "{} block {}: value {i}: ours={g} reference={}", case.ty, block.block_index_in_tensor, f32::from_bits(e));
            }
        }
    }
}

/// `LEAN_DEQUANT_GGUF`, `LEAN_DEQUANT_TENSOR` and `LEAN_DEQUANT_REF` (a file of
/// little-endian f32 written by `gen_dequant_blocks.py --full`).
#[test]
#[ignore = "needs a GGUF and the reference dump"]
fn whole_tensor_matches_reference() {
    let gguf = std::env::var("LEAN_DEQUANT_GGUF").expect("set LEAN_DEQUANT_GGUF");
    let tensor = std::env::var("LEAN_DEQUANT_TENSOR").expect("set LEAN_DEQUANT_TENSOR");
    let reference = std::fs::read(std::env::var("LEAN_DEQUANT_REF").expect("set LEAN_DEQUANT_REF")).unwrap();
    let mut reader = GgufReader::open(std::io::BufReader::new(std::fs::File::open(&gguf).unwrap())).unwrap();
    let info = reader.tensor_info(&tensor).unwrap().clone();
    let n = info.shape().iter().product::<usize>();
    let bytes = reader.tensor_data(&tensor).unwrap();
    let got = crate::gguf::dequantize_for(info.dtype(), &bytes, n);
    assert_eq!(reference.len(), n * 4);
    let ty = match info.dtype() {
        GgmlDtype::Q4_K => "Q4_K",
        GgmlDtype::Q5_0 => "Q5_0",
        GgmlDtype::F16 => "F16",
        GgmlDtype::BF16 => "BF16",
        other => panic!("{other:?}"),
    };
    let mut bad = 0usize;
    for (i, g) in got.iter().enumerate() {
        let r = f32::from_le_bytes(reference[4 * i..4 * i + 4].try_into().unwrap());
        if g.to_bits() != r.to_bits() {
            bad += 1;
            if bad < 5 {
                eprintln!("{ty} value {i}: ours={g} reference={r}");
            }
        }
    }
    assert_eq!(bad, 0, "{ty}: {bad} of {n} values differ");
    eprintln!("{ty} {tensor}: {n} values bit-identical");
}
