//! Q8_0 `token_embd.weight` support (llm-life: community base-model GGUFs
//! such as `QuantFactory/Qwen2.5-0.5B-GGUF` keep the token embedding at Q8_0
//! even in a "Q4_0" build, so `Q4ModelLoader` must accept it).
//!
//! No GPU here: `EmbeddingStore` is the CPU-side row dequant, and
//! `quantize_q4_0` is the CPU-side re-quantization `Q4ModelParts::finalize`
//! runs before uploading the tied lm_head.

use llm_wasm::gguf::{f16_to_f32, f32_to_f16, quantize_q4_0, EmbeddingStore, GgmlDtype};

const DIM: usize = 64;
const VOCAB: usize = 3;

/// A synthetic Q8_0 tensor: row `r`, block `b` has scale `0.5 / (r + b + 1)`
/// and quants `q[j] = (j as i8) - 16`.
fn synthetic_q8_0() -> (Vec<u8>, Vec<Vec<f32>>) {
    let blocks_per_row = DIM / 32;
    let mut bytes = Vec::with_capacity(VOCAB * blocks_per_row * 34);
    let mut expected = Vec::with_capacity(VOCAB);
    for r in 0..VOCAB {
        let mut row = Vec::with_capacity(DIM);
        for b in 0..blocks_per_row {
            let d = 0.5f32 / (r + b + 1) as f32;
            let d16 = f32_to_f16(d);
            bytes.extend_from_slice(&d16.to_le_bytes());
            for j in 0..32 {
                let q = (j as i8) - 16;
                bytes.push(q as u8);
                row.push(q as f32 * f16_to_f32(d16));
            }
        }
        expected.push(row);
    }
    (bytes, expected)
}

#[test]
fn q8_0_embedding_rows_dequantize_exactly() {
    let (bytes, expected) = synthetic_q8_0();
    let store = EmbeddingStore::new_with_dtype(bytes, GgmlDtype::Q8_0, VOCAB, DIM);
    assert_eq!(store.dtype(), GgmlDtype::Q8_0);
    for (id, want) in expected.iter().enumerate() {
        let got = store.embed_id(id as u32).unwrap();
        assert_eq!(&got, want, "row {id}");
    }
}

#[test]
fn q8_0_embedding_rejects_out_of_range_id() {
    let (bytes, _) = synthetic_q8_0();
    let store = EmbeddingStore::new_with_dtype(bytes, GgmlDtype::Q8_0, VOCAB, DIM);
    assert!(store.embed_id(VOCAB as u32).is_err());
}

/// The lossy half: what `finalize` uploads for the tied head. Q4_0 has 16
/// levels over the block's range, so the worst case is half a step of
/// `amax / 8` — this pins that bound so a regression in `quantize_q4_0`'s
/// rounding shows up as a test failure rather than as a slightly-wrong model.
#[test]
fn q4_0_requantization_stays_within_half_a_step() {
    let (_bytes, expected) = synthetic_q8_0();
    let flat: Vec<f32> = expected.iter().flatten().copied().collect();
    let q4 = quantize_q4_0(&flat);
    assert_eq!(q4.len(), flat.len() / 32 * 18);

    let store = EmbeddingStore::new_with_dtype(q4, GgmlDtype::Q4_0, VOCAB, DIM);
    for (id, want) in expected.iter().enumerate() {
        let got = store.embed_id(id as u32).unwrap();
        for (b, w) in want.chunks_exact(32).enumerate() {
            let amax = w.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let tol = amax / 8.0 * 0.51;
            for j in 0..32 {
                let g = got[b * 32 + j];
                assert!(
                    (g - w[j]).abs() <= tol,
                    "row {id} block {b} element {j}: {g} vs {} (tol {tol})",
                    w[j]
                );
            }
        }
    }
}

#[test]
fn f16_round_trip() {
    for v in [0.0f32, 1.0, -1.0, 0.5, -0.0625, 1e-5, -3.75, 65504.0] {
        let back = f16_to_f32(f32_to_f16(v));
        // f16 has ~11 bits of mantissa when normal and far fewer when
        // subnormal, hence the absolute term (one subnormal step).
        assert!(
            (back - v).abs() <= v.abs() * 1e-3 + 2f32.powi(-24),
            "{v} -> {back}"
        );
    }
    assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
}

/// Regression: `f16_to_f32`'s denormal path used a bias one too large, so
/// every subnormal f16 (anything under 2^-14 — a legal Q4_0/Q8_0 block
/// scale for a near-zero weight block) dequantized to exactly half its
/// value. The two literals below are the smallest subnormal and a
/// mid-range one, computed by hand from `mantissa * 2^-24`.
#[test]
fn f16_denormals_are_not_halved() {
    assert_eq!(f16_to_f32(1), 2f32.powi(-24));
    assert_eq!(f16_to_f32(167), 167.0 * 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x3FF), 1023.0 * 2f32.powi(-24));
    // Largest subnormal + 1 == smallest normal, and they must be adjacent.
    assert_eq!(f16_to_f32(0x400), 1024.0 * 2f32.powi(-24));
}
