//! Q4_0/Q8_0/Q6_K dot-product kernels for the CPU rung (`cpu.rs`): one row of
//! GGUF block bytes (exactly as they sit on disk - see `quant.rs`'s doc
//! comment on the shared block convention) dotted against an f32 activation
//! vector of the same length. Three implementations behind `cfg`, selected
//! at compile time, not runtime detection:
//!
//! - `neon` (aarch64): NEON is baseline on every aarch64 target (mandated by
//!   ARMv8-A), so no runtime feature probe is needed - this path compiles in
//!   whenever `target_arch = "aarch64"`.
//! - `simd128` (wasm32): only compiles in when the crate itself is built
//!   with `-C target-feature=+simd128` (see the workspace root's
//!   `.cargo/config.toml`, added alongside this file) - `target_feature =
//!   "simd128"` is then a compile-time cfg, not a runtime check, matching
//!   this plan's "no capability-detection branch" call for simd128
//!   specifically (browsers with wasm support all have simd128: Chrome 91+,
//!   Firefox 89+, Safari 16.4+).
//! - scalar: every other target, and the fallback both SIMD paths are
//!   checked against in `tests` below.
//!
//! Kernel shape (block-wise unpack -> widen -> multiply-accumulate ->
//! horizontal sum -> scale) follows llama.cpp's `ggml-cpu` NEON/wasm-simd128
//! dot kernels structurally. the project's own two prior CPU-SIMD efforts are the
//! direct reference points for the target ISA and the general approach, not
//! ported code: the wasm128 half of the author's unmerged candle branch
//! `wasm-simd-opt` (`src/cpu/simd128.rs`'s `vec_dot_f32_4col`, commit
//! `e52128fc`) already prototypes a widen-and-multiply-add wasm SIMD128 dot
//! product against this same GGUF block family, and t0-web's CPU
//! (ndarray) build demonstrated the simd128-vs-scalar win on this class of
//! hardware (`t0-web/docs/runs/2026-09-22-cpu-simd.md`) - but t0 gets its
//! speedup from a `matrixmultiply` rustflag, not a hand-written kernel, so
//! nothing there is reusable code, only the target ISA choice. This file's
//! kernels are written fresh against this crate's own Q4_0/Q8_0 layout.

const QK: usize = 32;
const QK6K: usize = 256;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Scalar reference Q6_K dot: one 210-byte super-block (128 `ql` + 64 `qh` +
/// 16 signed-i8 `scales` + 2-byte f16 `d`) against 256 activations, matching
/// `gguf::dequantize_q6_k`'s convention exactly - see that fn's doc comment
/// for the block-halves/`is`/scale-offset derivation.
#[inline]
#[allow(dead_code)] // always used by tests; used as the fallback impl on non-aarch64/non-simd128 targets
fn dot_q6_k_scalar(bytes: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (bi, block) in bytes.as_chunks::<210>().0.iter().enumerate() {
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let sc_all = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let x_base = bi * QK6K;
        let mut s = 0f32;
        for half in 0..2usize {
            let ql = &ql_all[half * 64..half * 64 + 64];
            let qh = &qh_all[half * 32..half * 32 + 32];
            let sc = &sc_all[half * 8..half * 8 + 8];
            let x_half = &x[x_base + half * 128..x_base + half * 128 + 128];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                s += (sc[is] as i8) as f32 * q1 as f32 * x_half[l];
                s += (sc[is + 2] as i8) as f32 * q2 as f32 * x_half[32 + l];
                s += (sc[is + 4] as i8) as f32 * q3 as f32 * x_half[64 + l];
                s += (sc[is + 6] as i8) as f32 * q4 as f32 * x_half[96 + l];
            }
        }
        acc += s * d;
    }
    acc
}

/// Unpacks one Q6_K half-block's four 32-value signed-weight groups
/// (`q - 32`, same convention as `dot_q6_k_scalar`) into `i8` scratch
/// arrays. The bit-assembly step (nibble | 2-bit field) doesn't vectorize
/// onto a single fixed shift/mask the way Q4_0/Q8_0's do, so both SIMD
/// paths below share this scalar unpack and only vectorize the actual
/// multiply-widen-accumulate against `x` (via each arch's existing 16-lane
/// signed-i8 dot helper), same op split as `dot_q4_0`'s dequant-then-dot
/// shape.
#[inline]
#[allow(dead_code)] // used by the neon/simd128 modules, which don't both compile on every target
fn unpack_q6k_half(ql: &[u8], qh: &[u8]) -> ([i8; 32], [i8; 32], [i8; 32], [i8; 32]) {
    let mut q1 = [0i8; 32];
    let mut q2 = [0i8; 32];
    let mut q3 = [0i8; 32];
    let mut q4 = [0i8; 32];
    for l in 0..32usize {
        q1[l] = (((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
        q2[l] = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
        q3[l] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
        q4[l] = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
    }
    (q1, q2, q3, q4)
}

/// Scalar reference Q4_0 dot: one 18-byte block (2-byte f16 scale + 16
/// packed-nibble bytes) against 32 activations, matching
/// `gguf::dequantize_q4_0`'s convention (`value = (nibble - 8) * scale`,
/// low nibble -> element j, high nibble -> element j+16) exactly, just
/// without materializing the dequantized row.
#[inline]
#[allow(dead_code)] // always used by tests; used as the fallback impl on non-aarch64/non-simd128 targets
fn dot_q4_0_scalar(bytes: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (bi, block) in bytes.as_chunks::<18>().0.iter().enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = bi * QK;
        let mut s = 0f32;
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = (byte >> 4) as f32 - 8.0;
            s += lo * x[base + j] + hi * x[base + 16 + j];
        }
        acc += s * scale;
    }
    acc
}

/// Scalar reference Q8_0 dot: one 34-byte block (2-byte f16 scale + 32
/// signed i8) against 32 activations, matching `gguf::dequantize_q8_0`.
#[inline]
#[allow(dead_code)] // always used by tests; used as the fallback impl on non-aarch64/non-simd128 targets
fn dot_q8_0_scalar(bytes: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (bi, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = bi * QK;
        let mut s = 0f32;
        for j in 0..32 {
            s += (block[2 + j] as i8) as f32 * x[base + j];
        }
        acc += s * scale;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{f16_to_f32, QK};
    use std::arch::aarch64::*;

    /// One block's 16 nibble-pair bytes -> two `int32x4_t` quads (elements
    /// `[0,4)`/`[4,8)` of either the lo or hi half) converted to f32 and
    /// dotted against `x`. `nibble_mask`/`shift` pick low (`0x0F`, 0) or
    /// high (`0xF0`... done via `vshrq_n_u8`, so mask is always applied
    /// after an optional shift) nibbles - see the call sites below.
    // SAFETY: caller must guarantee `x.len() >= 16` - this function reads
    // `x[0..16]` via four unchecked `vld1q_f32` loads (4 lanes each) with no
    // bounds check of its own.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot16_signed(vals_u8: uint8x16_t, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= 16, "dot16_signed: x.len()={} < 16", x.len());
        // vals_u8 lanes are already in [0, 15] (nibble value); subtract 8 to
        // get the signed weight, same as `(nibble as f32) - 8.0` above.
        let signed = vsubq_s8(vreinterpretq_s8_u8(vals_u8), vdupq_n_s8(8));
        let lo16 = vmovl_s8(vget_low_s8(signed)); // elements 0..8
        let hi16 = vmovl_s8(vget_high_s8(signed)); // elements 8..16
        let lo32a = vmovl_s16(vget_low_s16(lo16)); // 0..4
        let lo32b = vmovl_s16(vget_high_s16(lo16)); // 4..8
        let hi32a = vmovl_s16(vget_low_s16(hi16)); // 8..12
        let hi32b = vmovl_s16(vget_high_s16(hi16)); // 12..16
        let fa = vcvtq_f32_s32(lo32a);
        let fb = vcvtq_f32_s32(lo32b);
        let fc = vcvtq_f32_s32(hi32a);
        let fd = vcvtq_f32_s32(hi32b);
        let xa = vld1q_f32(x.as_ptr());
        let xb = vld1q_f32(x.as_ptr().add(4));
        let xc = vld1q_f32(x.as_ptr().add(8));
        let xd = vld1q_f32(x.as_ptr().add(12));
        let mut acc = vmulq_f32(fa, xa);
        acc = vfmaq_f32(acc, fb, xb);
        acc = vfmaq_f32(acc, fc, xc);
        acc = vfmaq_f32(acc, fd, xd);
        vaddvq_f32(acc)
    }

    // SAFETY: caller must guarantee `block.len() >= 18` (2-byte f16 scale +
    // 16 packed nibble bytes) and `x.len() >= 32`; `block[2..18]` is sliced
    // (panics on a short block) but the `vld1q_u8` load itself trusts that
    // slice's length, and `dot16_signed` below trusts `x`'s length.
    #[target_feature(enable = "neon")]
    unsafe fn dot_q4_0_block(block: &[u8], x: &[f32]) -> f32 {
        debug_assert!(block.len() >= 18, "dot_q4_0_block: block.len()={} < 18", block.len());
        debug_assert!(x.len() >= 32, "dot_q4_0_block: x.len()={} < 32", x.len());
        let raw = vld1q_u8(block[2..18].as_ptr());
        let lo = vandq_u8(raw, vdupq_n_u8(0x0F));
        let hi = vshrq_n_u8(raw, 4);
        dot16_signed(lo, &x[0..16]) + dot16_signed(hi, &x[16..32])
    }

    pub(super) fn dot_q4_0(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.as_chunks::<18>().0.iter().enumerate() {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let base = bi * QK;
            let s = unsafe { dot_q4_0_block(block, &x[base..base + QK]) };
            acc += s * scale;
        }
        acc
    }

    // SAFETY: caller must guarantee `qs.len() >= 32` (this reads two
    // unchecked 16-byte NEON loads at offsets 0 and 16) and `x.len() >= 32`
    // (sliced into two 16-element halves passed to `dot16_signed_i8`).
    #[target_feature(enable = "neon")]
    unsafe fn dot_q8_0_block(qs: &[u8], x: &[f32]) -> f32 {
        debug_assert!(qs.len() >= 32, "dot_q8_0_block: qs.len()={} < 32", qs.len());
        debug_assert!(x.len() >= 32, "dot_q8_0_block: x.len()={} < 32", x.len());
        let a = vld1q_s8(qs.as_ptr() as *const i8);
        let b = vld1q_s8(qs.as_ptr().add(16) as *const i8);
        dot16_signed_i8(a, &x[0..16]) + dot16_signed_i8(b, &x[16..32])
    }

    // SAFETY: caller must guarantee `x.len() >= 16`, same contract as
    // `dot16_signed` above (four unchecked 4-lane `vld1q_f32` loads).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot16_signed_i8(vals: int8x16_t, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= 16, "dot16_signed_i8: x.len()={} < 16", x.len());
        let lo16 = vmovl_s8(vget_low_s8(vals));
        let hi16 = vmovl_s8(vget_high_s8(vals));
        let lo32a = vmovl_s16(vget_low_s16(lo16));
        let lo32b = vmovl_s16(vget_high_s16(lo16));
        let hi32a = vmovl_s16(vget_low_s16(hi16));
        let hi32b = vmovl_s16(vget_high_s16(hi16));
        let fa = vcvtq_f32_s32(lo32a);
        let fb = vcvtq_f32_s32(lo32b);
        let fc = vcvtq_f32_s32(hi32a);
        let fd = vcvtq_f32_s32(hi32b);
        let xa = vld1q_f32(x.as_ptr());
        let xb = vld1q_f32(x.as_ptr().add(4));
        let xc = vld1q_f32(x.as_ptr().add(8));
        let xd = vld1q_f32(x.as_ptr().add(12));
        let mut acc = vmulq_f32(fa, xa);
        acc = vfmaq_f32(acc, fb, xb);
        acc = vfmaq_f32(acc, fc, xc);
        acc = vfmaq_f32(acc, fd, xd);
        vaddvq_f32(acc)
    }

    pub(super) fn dot_q8_0(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.as_chunks::<34>().0.iter().enumerate() {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let base = bi * QK;
            let s = unsafe { dot_q8_0_block(&block[2..34], &x[base..base + QK]) };
            acc += s * scale;
        }
        acc
    }

    /// One Q6_K super-block dotted against 256 activations: scalar bit
    /// unpack (`super::unpack_q6k_half`) into four 32-value `i8` groups per
    /// half-block, each group's two 16-value sub-ranges (`is = l/16`, its own
    /// `scales` entry) dotted via the existing 16-lane `dot16_signed_i8`
    /// widen helper - see `super::unpack_q6k_half`'s doc comment on why the
    /// unpack itself stays scalar.
    pub(super) fn dot_q6_k(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.as_chunks::<210>().0.iter().enumerate() {
            let ql_all = &block[0..128];
            let qh_all = &block[128..192];
            let sc_all = &block[192..208];
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let x_base = bi * super::QK6K;
            let mut s = 0f32;
            for half in 0..2usize {
                let ql = &ql_all[half * 64..half * 64 + 64];
                let qh = &qh_all[half * 32..half * 32 + 32];
                let sc = &sc_all[half * 8..half * 8 + 8];
                let x_half = &x[x_base + half * 128..x_base + half * 128 + 128];
                let (q1, q2, q3, q4) = super::unpack_q6k_half(ql, qh);
                unsafe {
                    s += (sc[0] as i8) as f32 * dot16_signed_i8(vld1q_s8(q1[0..16].as_ptr()), &x_half[0..16]);
                    s += (sc[1] as i8) as f32 * dot16_signed_i8(vld1q_s8(q1[16..32].as_ptr()), &x_half[16..32]);
                    s += (sc[2] as i8) as f32 * dot16_signed_i8(vld1q_s8(q2[0..16].as_ptr()), &x_half[32..48]);
                    s += (sc[3] as i8) as f32 * dot16_signed_i8(vld1q_s8(q2[16..32].as_ptr()), &x_half[48..64]);
                    s += (sc[4] as i8) as f32 * dot16_signed_i8(vld1q_s8(q3[0..16].as_ptr()), &x_half[64..80]);
                    s += (sc[5] as i8) as f32 * dot16_signed_i8(vld1q_s8(q3[16..32].as_ptr()), &x_half[80..96]);
                    s += (sc[6] as i8) as f32 * dot16_signed_i8(vld1q_s8(q4[0..16].as_ptr()), &x_half[96..112]);
                    s += (sc[7] as i8) as f32 * dot16_signed_i8(vld1q_s8(q4[16..32].as_ptr()), &x_half[112..128]);
                }
            }
            acc += s * d;
        }
        acc
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use super::{f16_to_f32, QK};
    use std::arch::wasm32::*;

    /// `vals`' 16 lanes, already-signed i8 (Q8_0's raw bytes), converted to
    /// f32 and dotted against `x`. No offset applied - see
    /// `dot16_unsigned_nibbles` for the Q4_0 case, which needs one.
    // SAFETY: caller must guarantee `x.len() >= 16` - this reads `x[0..16]`
    // via four unchecked `v128_load`s (4 lanes each) with no bounds check.
    #[inline]
    unsafe fn dot16_signed_i8(vals: v128, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= 16, "dot16_signed_i8: x.len()={} < 16", x.len());
        let lo16 = i16x8_extend_low_i8x16(vals);
        let hi16 = i16x8_extend_high_i8x16(vals);
        let lo32a = i32x4_extend_low_i16x8(lo16);
        let lo32b = i32x4_extend_high_i16x8(lo16);
        let hi32a = i32x4_extend_low_i16x8(hi16);
        let hi32b = i32x4_extend_high_i16x8(hi16);
        let fa = f32x4_convert_i32x4(lo32a);
        let fb = f32x4_convert_i32x4(lo32b);
        let fc = f32x4_convert_i32x4(hi32a);
        let fd = f32x4_convert_i32x4(hi32b);
        let xa = v128_load(x.as_ptr() as *const v128);
        let xb = v128_load(x.as_ptr().add(4) as *const v128);
        let xc = v128_load(x.as_ptr().add(8) as *const v128);
        let xd = v128_load(x.as_ptr().add(12) as *const v128);
        let mut acc = f32x4_mul(fa, xa);
        acc = f32x4_add(acc, f32x4_mul(fb, xb));
        acc = f32x4_add(acc, f32x4_mul(fc, xc));
        acc = f32x4_add(acc, f32x4_mul(fd, xd));
        f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc)
    }

    /// `vals`' 16 lanes, each an unsigned nibble value `[0, 15]` (Q4_0's
    /// packed weights), offset by `-8` to the signed weight range before
    /// converting to f32 - the bug this comment guards against: reusing
    /// `dot16_signed_i8`'s already-signed path here (or its own path for
    /// Q8_0) silently corrupts one or the other, since only Q4_0's nibbles
    /// need the offset (caught by `tests::q8_dispatch_matches_scalar_reference`
    /// on a real wasm32+simd128 run under wasmtime - this file's dispatch
    /// wrapper looked identical for both dtypes in an earlier version and
    /// wasn't).
    // SAFETY: delegates to `dot16_signed_i8`, same `x.len() >= 16` contract.
    #[inline]
    unsafe fn dot16_unsigned_nibbles(vals: v128, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= 16, "dot16_unsigned_nibbles: x.len()={} < 16", x.len());
        dot16_signed_i8(i8x16_sub(vals, i8x16_splat(8)), x)
    }

    // SAFETY: `block.len() >= 18` (2-byte f16 scale + 16 packed nibble
    // bytes; `block[2..18]` is sliced first and panics if too short, but the
    // `v128_load` itself trusts the slice) and `x.len() >= 32`.
    fn dot_q4_0_block(block: &[u8], x: &[f32]) -> f32 {
        debug_assert!(block.len() >= 18, "dot_q4_0_block: block.len()={} < 18", block.len());
        debug_assert!(x.len() >= 32, "dot_q4_0_block: x.len()={} < 32", x.len());
        unsafe {
            let raw = v128_load(block[2..18].as_ptr() as *const v128);
            let lo = v128_and(raw, u8x16_splat(0x0F));
            let hi = u8x16_shr(raw, 4);
            dot16_unsigned_nibbles(lo, &x[0..16]) + dot16_unsigned_nibbles(hi, &x[16..32])
        }
    }

    pub(super) fn dot_q4_0(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.chunks_exact(18).enumerate() {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let base = bi * QK;
            acc += dot_q4_0_block(block, &x[base..base + QK]) * scale;
        }
        acc
    }

    // SAFETY: `qs.len() >= 32` (two unchecked 16-byte `v128_load`s at
    // offsets 0 and 16) and `x.len() >= 32`.
    fn dot_q8_0_block(qs: &[u8], x: &[f32]) -> f32 {
        debug_assert!(qs.len() >= 32, "dot_q8_0_block: qs.len()={} < 32", qs.len());
        debug_assert!(x.len() >= 32, "dot_q8_0_block: x.len()={} < 32", x.len());
        unsafe {
            let a = v128_load(qs.as_ptr() as *const v128);
            let b = v128_load(qs.as_ptr().add(16) as *const v128);
            dot16_signed_i8(a, &x[0..16]) + dot16_signed_i8(b, &x[16..32])
        }
    }

    pub(super) fn dot_q8_0(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.chunks_exact(34).enumerate() {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let base = bi * QK;
            acc += dot_q8_0_block(&block[2..34], &x[base..base + QK]) * scale;
        }
        acc
    }

    /// wasm simd128 mirror of `neon::dot_q6_k`: scalar bit unpack
    /// (`super::unpack_q6k_half`), then `dot16_signed_i8` (already-signed,
    /// no nibble offset needed since the unpack already applied `- 32`) for
    /// each 16-value sub-range's widen-and-accumulate.
    pub(super) fn dot_q6_k(bytes: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0f32;
        for (bi, block) in bytes.chunks_exact(210).enumerate() {
            let ql_all = &block[0..128];
            let qh_all = &block[128..192];
            let sc_all = &block[192..208];
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let x_base = bi * super::QK6K;
            let mut s = 0f32;
            for half in 0..2usize {
                let ql = &ql_all[half * 64..half * 64 + 64];
                let qh = &qh_all[half * 32..half * 32 + 32];
                let sc = &sc_all[half * 8..half * 8 + 8];
                let x_half = &x[x_base + half * 128..x_base + half * 128 + 128];
                let (q1, q2, q3, q4) = super::unpack_q6k_half(ql, qh);
                unsafe {
                    s += (sc[0] as i8) as f32 * dot16_signed_i8(v128_load(q1[0..16].as_ptr() as *const v128), &x_half[0..16]);
                    s += (sc[1] as i8) as f32 * dot16_signed_i8(v128_load(q1[16..32].as_ptr() as *const v128), &x_half[16..32]);
                    s += (sc[2] as i8) as f32 * dot16_signed_i8(v128_load(q2[0..16].as_ptr() as *const v128), &x_half[32..48]);
                    s += (sc[3] as i8) as f32 * dot16_signed_i8(v128_load(q2[16..32].as_ptr() as *const v128), &x_half[48..64]);
                    s += (sc[4] as i8) as f32 * dot16_signed_i8(v128_load(q3[0..16].as_ptr() as *const v128), &x_half[64..80]);
                    s += (sc[5] as i8) as f32 * dot16_signed_i8(v128_load(q3[16..32].as_ptr() as *const v128), &x_half[80..96]);
                    s += (sc[6] as i8) as f32 * dot16_signed_i8(v128_load(q4[0..16].as_ptr() as *const v128), &x_half[96..112]);
                    s += (sc[7] as i8) as f32 * dot16_signed_i8(v128_load(q4[16..32].as_ptr() as *const v128), &x_half[112..128]);
                }
            }
            acc += s * d;
        }
        acc
    }
}

/// Four f32 lanes for the CPU backend's multi-row kernels (`cpu.rs`'s
/// `dot_tile`/`dot_f32`): NEON on aarch64, SIMD128 on a `+simd128` wasm32
/// build, a plain array elsewhere - chosen at compile time like the dot
/// kernels above. Only lane-wise IEEE add and mul (no FMA), so every
/// implementation gives the same bits.
#[derive(Clone, Copy)]
pub struct F4(
    #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))] std::arch::aarch64::float32x4_t,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] std::arch::wasm32::v128,
    #[cfg(not(any(all(target_arch = "aarch64", not(feature = "force_scalar")), all(target_arch = "wasm32", target_feature = "simd128"))))] [f32; 4],
);

impl F4 {
    #[inline(always)]
    pub fn zero() -> F4 {
        #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
        return F4(unsafe { std::arch::aarch64::vdupq_n_f32(0.0) });
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return F4(std::arch::wasm32::f32x4_splat(0.0));
        #[cfg(not(any(all(target_arch = "aarch64", not(feature = "force_scalar")), all(target_arch = "wasm32", target_feature = "simd128"))))]
        return F4([0.0; 4]);
    }

    /// `s[0..4]`.
    #[inline(always)]
    pub fn load(s: &[f32]) -> F4 {
        assert!(s.len() >= 4);
        #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
        return F4(unsafe { std::arch::aarch64::vld1q_f32(s.as_ptr()) });
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return F4(unsafe { std::arch::wasm32::v128_load(s.as_ptr() as *const std::arch::wasm32::v128) });
        #[cfg(not(any(all(target_arch = "aarch64", not(feature = "force_scalar")), all(target_arch = "wasm32", target_feature = "simd128"))))]
        return F4([s[0], s[1], s[2], s[3]]);
    }

    /// `self + a * b`, as a separate multiply and add.
    #[inline(always)]
    pub fn add_mul(self, a: F4, b: F4) -> F4 {
        #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
        return F4(unsafe { std::arch::aarch64::vaddq_f32(self.0, std::arch::aarch64::vmulq_f32(a.0, b.0)) });
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return F4(std::arch::wasm32::f32x4_add(self.0, std::arch::wasm32::f32x4_mul(a.0, b.0)));
        #[cfg(not(any(all(target_arch = "aarch64", not(feature = "force_scalar")), all(target_arch = "wasm32", target_feature = "simd128"))))]
        return F4([self.0[0] + a.0[0] * b.0[0], self.0[1] + a.0[1] * b.0[1], self.0[2] + a.0[2] * b.0[2], self.0[3] + a.0[3] * b.0[3]]);
    }

    /// `(l0 + l1) + (l2 + l3)`.
    #[inline(always)]
    pub fn sum(self) -> f32 {
        #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
        // SAFETY: `float32x4_t` is 16 bytes holding four f32 lanes in lane
        // order, the same size and layout as `[f32; 4]`, and every bit
        // pattern is a valid f32.
        let l: [f32; 4] = unsafe { std::mem::transmute(self.0) };
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        // SAFETY: `v128` is 16 bytes, and read as f32x4 its lanes are four
        // f32 in lane order, the same size and layout as `[f32; 4]`; every
        // bit pattern is a valid f32.
        let l: [f32; 4] = unsafe { std::mem::transmute(self.0) };
        #[cfg(not(any(all(target_arch = "aarch64", not(feature = "force_scalar")), all(target_arch = "wasm32", target_feature = "simd128"))))]
        let l = self.0;
        (l[0] + l[1]) + (l[2] + l[3])
    }
}

/// One row's Q4_0 block bytes dotted against `x` (`x.len()` must equal
/// `blocks_per_row * 32`). Dispatches to the NEON path on aarch64, simd128
/// on a wasm32 build compiled with `+simd128`, scalar everywhere else -
/// selected at compile time (see this module's doc comment), never at
/// runtime.
#[inline]
pub fn dot_q4_0(bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
    {
        neon::dot_q4_0(bytes, x)
    }
    #[cfg(all(target_arch = "aarch64", feature = "force_scalar"))]
    {
        dot_q4_0_scalar(bytes, x)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        simd128::dot_q4_0(bytes, x)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        dot_q4_0_scalar(bytes, x)
    }
}

/// One row's Q8_0 block bytes dotted against `x` (`x.len()` must equal
/// `blocks_per_row * 32`). Same dispatch rule as `dot_q4_0`.
#[inline]
pub fn dot_q8_0(bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
    {
        neon::dot_q8_0(bytes, x)
    }
    #[cfg(all(target_arch = "aarch64", feature = "force_scalar"))]
    {
        dot_q8_0_scalar(bytes, x)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        simd128::dot_q8_0(bytes, x)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        dot_q8_0_scalar(bytes, x)
    }
}

/// One row's Q6_K block bytes dotted against `x` (`x.len()` must equal
/// `blocks_per_row * 256`). Same dispatch rule as `dot_q4_0`/`dot_q8_0`.
#[inline]
pub fn dot_q6_k(bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", not(feature = "force_scalar")))]
    {
        neon::dot_q6_k(bytes, x)
    }
    #[cfg(all(target_arch = "aarch64", feature = "force_scalar"))]
    {
        dot_q6_k_scalar(bytes, x)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        simd128::dot_q6_k(bytes, x)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        dot_q6_k_scalar(bytes, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_q4_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 18];
        let scale = half::f16::from_f32(0.05 + (seed % 7) as f32 * 0.03);
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        for (i, b) in block[2..18].iter_mut().enumerate() {
            *b = ((i as u32 * 11 + seed * 5) % 256) as u8;
        }
        block
    }

    fn synth_q8_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 34];
        let scale = half::f16::from_f32(0.02 + (seed % 5) as f32 * 0.01);
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        for (i, b) in block[2..34].iter_mut().enumerate() {
            *b = ((i as i32 * 7 + seed as i32 * 3) % 256 - 128) as i8 as u8;
        }
        block
    }

    #[test]
    fn q4_dispatch_matches_scalar_reference() {
        let n_blocks = 5;
        let mut bytes = Vec::new();
        for bi in 0..n_blocks {
            bytes.extend(synth_q4_block(bi as u32));
        }
        let x: Vec<f32> = (0..n_blocks * 32).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let scalar = dot_q4_0_scalar(&bytes, &x);
        let dispatched = dot_q4_0(&bytes, &x);
        assert!((scalar - dispatched).abs() < 1e-3, "scalar={scalar} dispatched={dispatched}");
    }

    #[test]
    fn q8_dispatch_matches_scalar_reference() {
        let n_blocks = 5;
        let mut bytes = Vec::new();
        for bi in 0..n_blocks {
            bytes.extend(synth_q8_block(bi as u32));
        }
        let x: Vec<f32> = (0..n_blocks * 32).map(|i| (i as f32) * 0.02 - 1.0).collect();
        let scalar = dot_q8_0_scalar(&bytes, &x);
        let dispatched = dot_q8_0(&bytes, &x);
        assert!((scalar - dispatched).abs() < 1e-3, "scalar={scalar} dispatched={dispatched}");
    }

    fn synth_q6k_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        for (i, b) in block[0..192].iter_mut().enumerate() {
            *b = ((i as u32 * 13 + seed * 37) % 256) as u8;
        }
        for (i, b) in block[192..208].iter_mut().enumerate() {
            *b = ((i as i32 * 17 + seed as i32 * 11) % 256 - 128) as i8 as u8;
        }
        let d = half::f16::from_f32(0.02 + (seed % 5) as f32 * 0.01);
        block[208..210].copy_from_slice(&d.to_le_bytes());
        block
    }

    #[test]
    fn q6k_dispatch_matches_scalar_reference() {
        let n_blocks = 3;
        let mut bytes = Vec::new();
        for bi in 0..n_blocks {
            bytes.extend(synth_q6k_block(bi as u32));
        }
        let x: Vec<f32> = (0..n_blocks * 256).map(|i| (i as f32) * 0.005 - 1.0).collect();
        let scalar = dot_q6_k_scalar(&bytes, &x);
        let dispatched = dot_q6_k(&bytes, &x);
        assert!((scalar - dispatched).abs() < 1e-2, "scalar={scalar} dispatched={dispatched}");
    }
}
