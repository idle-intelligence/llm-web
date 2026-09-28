// Integer-dot-product counterpart of linear_q8_decode.wgsl (this crate's own
// design - see quantize_act_q8.wgsl's header for the technique this project
// is drawing on from llama.cpp's `mul_mat_vecq.comp`, and
// docs/runs/2026-09-28-lean-perf-2.md session 4's step 4 survey). Same
// dispatch shape (4 rows/workgroup, 32 lanes/row, shared-memory reduction)
// as the f32 kernel; the only change is the inner per-word contribution:
// Q8_0 weight words are already 4 packed signed int8 values (this crate's
// own on-disk layout, see linear_q8_decode.wgsl's `accumulate_word`), which
// is exactly the bit layout `dot4I8Packed` expects, so the weight word needs
// no repacking - it is used directly against the matching quantized
// activation word from `quantize_act_q8.wgsl` (same block-of-32, same
// 8-words-per-block layout, so word index `w` in this loop indexes both
// arrays identically).
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
    let blk = word_idx / 8u;
    let w_word = qs[weights_row_base + word_idx];
    let a_word = act_q[blk * 8u + (word_idx % 8u)];
    let dp = dot4I8Packed(w_word, a_word);
    return f32(dp) * scales[weights_row_base / 8u + blk] * act_scale[blk];
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

    let weights_row_base = n * blocks_per_row * 8u;
    let total_words = blocks_per_row * 8u;

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
