// Adapted from llm-wasm/src/wgsl/shader_q4_matvec_coalesced.wgsl (Q4_0
// Coalesced Matvec, M=1/decode — see that file's header for the full
// word-interleaved coalescing rationale). Differences from the llm-wasm
// original: bias folded in (`b[n]`, this crate's linear-layer contract),
// `info` replaced by a `Dims` uniform struct, and `B` (batch) dropped since
// this crate never batches decode (always exactly one token/step) — so the
// wg_id.y/`b` axis and its `b_valid` guard are gone entirely.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn accumulate_word(word_idx: u32, weights_row_base: u32, scale_row_base: u32) -> f32 {
    let blk = word_idx / 4u;
    let wj = word_idx % 4u;
    let scale = scales[scale_row_base + blk];
    let packed = qs[weights_row_base + word_idx];

    let b0 = packed & 0xFFu;
    let b1 = (packed >> 8u) & 0xFFu;
    let b2 = (packed >> 16u) & 0xFFu;
    let b3 = (packed >> 24u) & 0xFFu;

    let k_lo = blk * 32u + wj * 4u;
    let k_hi = k_lo + 16u;

    let w_lo = (vec4<f32>(
        f32(b0 & 0xFu), f32(b1 & 0xFu),
        f32(b2 & 0xFu), f32(b3 & 0xFu)
    ) - vec4<f32>(8.0)) * scale;
    let in_lo = vec4<f32>(x[k_lo], x[k_lo + 1u], x[k_lo + 2u], x[k_lo + 3u]);

    let w_hi = (vec4<f32>(
        f32((b0 >> 4u) & 0xFu), f32((b1 >> 4u) & 0xFu),
        f32((b2 >> 4u) & 0xFu), f32((b3 >> 4u) & 0xFu)
    ) - vec4<f32>(8.0)) * scale;
    let in_hi = vec4<f32>(x[k_hi], x[k_hi + 1u], x[k_hi + 2u], x[k_hi + 3u]);

    return dot(w_lo, in_lo) + dot(w_hi, in_hi);
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
    let scale_row_base = n * blocks_per_row;
    let total_words = blocks_per_row * 4u;

    var acc: f32 = 0.0;
    if (row_has_output) {
        var w0: u32 = lane;
        loop {
            if (w0 >= total_words) {
                break;
            }
            acc += accumulate_word(w0, weights_row_base, scale_row_base);
            let w1 = w0 + 32u;
            if (w1 < total_words) {
                acc += accumulate_word(w1, weights_row_base, scale_row_base);
            }
            let w2 = w0 + 64u;
            if (w2 < total_words) {
                acc += accumulate_word(w2, weights_row_base, scale_row_base);
            }
            let w3 = w0 + 96u;
            if (w3 < total_words) {
                acc += accumulate_word(w3, weights_row_base, scale_row_base);
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
        var v = partial_sums[tid] + b[n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[n] = v;
    }
}
