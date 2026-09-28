// Integer-dot-product counterpart of linear_q4_decode.wgsl (this crate's own
// design - see quantize_act_q8.wgsl's header for the technique this project
// is drawing on from llama.cpp's `mul_mat_vecq.comp`, and
// docs/runs/2026-09-28-lean-perf-2.md session 4's step 4 survey). Same
// dispatch shape (4 rows/workgroup, 32 lanes/row, shared-memory reduction)
// as the f32 kernel; the difference is the inner per-word contribution:
// each Q4_0 word (4 bytes = 8 nibbles) is repacked here into two 4-packed
// signed-int8 words - `packed_lo` (the low nibbles, values k..k+3) and
// `packed_hi` (the high nibbles, values k+16..k+19 - Q4_0's on-disk layout
// stores a block's second half in the high nibbles of the same 16 bytes,
// see linear_q4_decode.wgsl's own `accumulate_word`) - each nibble's `-8`
// zero-point folded directly into the repacked byte (`(nibble - 8u) & 0xFFu`
// is the nibble's value in 8-bit two's complement, exploiting u32
// subtraction's defined wraparound), so no separate activation-sum
// correction term is needed (unlike llama.cpp's Q8_1, which keeps the raw
// unsigned nibble and corrects for the offset via a precomputed activation
// block sum - this crate folds the offset in at repack time instead, which
// this crate's own fixture-parity gate is the check that this is
// numerically equivalent within tolerance).
//
// Selected only when `Engine::has_dp4` (the adapter's own
// `packed_4x8_integer_dot_product` WGSL language feature, queried once at
// startup - see engine.rs) reports support *and* the caller has explicitly
// opted in (`Model`'s `dp4_decode` flag) - see model.rs::linear. Never
// selected by matching a device name/vendor string.
requires packed_4x8_integer_dot_product;

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    n_offset: u32,
    n_total: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> act_q: array<u32>;
@group(0) @binding(1) var<storage, read> act_scale: array<f32>;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn accumulate_word(word_idx: u32, weights_row_base: u32) -> f32 {
    let blk = word_idx / 4u;
    let wj = word_idx % 4u;
    let scale = scales[weights_row_base / 4u + blk];
    let packed = qs[weights_row_base + word_idx];

    let b0 = packed & 0xFFu;
    let b1 = (packed >> 8u) & 0xFFu;
    let b2 = (packed >> 16u) & 0xFFu;
    let b3 = (packed >> 24u) & 0xFFu;

    let lo0 = (b0 & 0xFu) - 8u;
    let lo1 = (b1 & 0xFu) - 8u;
    let lo2 = (b2 & 0xFu) - 8u;
    let lo3 = (b3 & 0xFu) - 8u;
    let packed_lo = (lo0 & 0xFFu) | ((lo1 & 0xFFu) << 8u) | ((lo2 & 0xFFu) << 16u) | ((lo3 & 0xFFu) << 24u);

    let hi0 = ((b0 >> 4u) & 0xFu) - 8u;
    let hi1 = ((b1 >> 4u) & 0xFu) - 8u;
    let hi2 = ((b2 >> 4u) & 0xFu) - 8u;
    let hi3 = ((b3 >> 4u) & 0xFu) - 8u;
    let packed_hi = (hi0 & 0xFFu) | ((hi1 & 0xFFu) << 8u) | ((hi2 & 0xFFu) << 16u) | ((hi3 & 0xFFu) << 24u);

    // act_q's word layout is 8 words/32-value block (quantize_act_q8.wgsl);
    // this weight word's low nibbles cover k = blk*32 + wj*4 .. +3, whose
    // act word index is blk*8 + wj; the high nibbles cover k+16, act word
    // index blk*8 + wj + 4.
    let a_lo = act_q[blk * 8u + wj];
    let a_hi = act_q[blk * 8u + wj + 4u];

    let dp = dot4I8Packed(packed_lo, a_lo) + dot4I8Packed(packed_hi, a_hi);
    return f32(dp) * scale * act_scale[blk];
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let N = dims.n;
    let blocks_per_row = dims.blocks_per_row;

    let tid = local_id.x;
    let row_in_wg = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * ROWS_PER_WG + row_in_wg;
    let row_has_output = n < N;

    let weights_row_base = n * blocks_per_row * 4u;
    let total_words = blocks_per_row * 4u;

    var acc: f32 = 0.0;
    if (row_has_output) {
        var w0: u32 = lane;
        loop {
            if (w0 >= total_words) {
                break;
            }
            acc += accumulate_word(w0, weights_row_base);
            let w1 = w0 + 32u;
            if (w1 < total_words) {
                acc += accumulate_word(w1, weights_row_base);
            }
            let w2 = w0 + 64u;
            if (w2 < total_words) {
                acc += accumulate_word(w2, weights_row_base);
            }
            let w3 = w0 + 96u;
            if (w3 < total_words) {
                acc += accumulate_word(w3, weights_row_base);
            }
            w0 = w0 + 128u;
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = THREADS_PER_ROW / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u && row_has_output) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
