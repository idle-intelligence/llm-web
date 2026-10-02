// Small-M Q4_0 matmul for short prefills (2 <= M < SMALL_M_MAX_ROWS in
// model.rs). Started from this project's batched-decode kernel
// (linear_q4_decode_batched.wgsl on the lean-spec branch, written for
// speculative decoding's 2-8 row verify step): the matvec layout of
// linear_q4_decode.wgsl (lanes of one output column striding over the
// row's Q4_0 words, shared-memory tree reduction), with every weight word
// read and dequantised once and dotted against up to ROWS query rows held
// in registers.
//
// Changes from the lean-spec kernel:
// - the M axis is spread over workgroup_id.y in groups of ROWS (8), so one
//   dispatch covers any M instead of capping M at 8;
// - the word is dequantised once, outside the row loop (the lean-spec
//   version redid the load and dequant per row);
// - `x` is bound as vec4s (K is a multiple of 32, so a block's 4-value runs
//   are 16-byte aligned);
// - 4 lanes per output column (one Q4_0 block's 4 words per step, one
//   shared scale) and 32 columns per workgroup, instead of 32 lanes and 4
//   columns: with 8 rows of work per word, 32 lanes left most of each
//   workgroup's time in the reduction on short rows (K = 896), and the M2
//   sweep in docs/runs/2026-10-02-lean-mobile.md ran 36-row prefill in
//   124 ms with 4 or 8 lanes vs 204 ms with 32;
// - the 8 row accumulators are two vec4s with fixed indices, no
//   dynamically indexed private array (which some compilers spill to
//   scratch memory). Rows past M in the last group read row m0 again and
//   are never written.
//
// Q8 (pipeline override, `linear_q8_small_m` in engine.rs): the same
// kernel over Q8_0 blocks (8 words of 4 signed bytes per block, one word =
// 4 consecutive k) instead of Q4_0 (4 words of 8 nibbles).
//
// Bindings and Dims are the same as linear_q4.wgsl/linear_q4_decode.wgsl.
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

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

override Q8: bool = false;

const ROWS: u32 = 8u;
const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 4u;
const COLS_PER_WG: u32 = 32u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_a: array<vec4<f32>, WG_SIZE>;
var<workgroup> partial_b: array<vec4<f32>, WG_SIZE>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let col_in_wg = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * COLS_PER_WG + col_in_wg;
    let col_ok = n < dims.n;
    let m0 = wg_id.y * ROWS;
    let mloc = min(ROWS, dims.m - m0);

    // Query row r of this group, clamped to a valid row (results of clamped
    // rows are discarded below), as a vec4 index base.
    let k4 = dims.k / 4u;
    let last = mloc - 1u;
    let r0 = (m0 + min(0u, last)) * k4;
    let r1 = (m0 + min(1u, last)) * k4;
    let r2 = (m0 + min(2u, last)) * k4;
    let r3 = (m0 + min(3u, last)) * k4;
    let r4 = (m0 + min(4u, last)) * k4;
    let r5 = (m0 + min(5u, last)) * k4;
    let r6 = (m0 + min(6u, last)) * k4;
    let r7 = (m0 + min(7u, last)) * k4;

    let words_per_block = select(4u, 8u, Q8);
    let total_words = dims.blocks_per_row * words_per_block;
    let w_base = n * total_words;
    let s_base = n * dims.blocks_per_row;

    var acc_a = vec4<f32>(0.0);
    var acc_b = vec4<f32>(0.0);
    if (col_ok) {
        for (var w: u32 = lane; w < total_words; w = w + THREADS_PER_ROW) {
            let blk = w / words_per_block;
            let scale = scales[s_base + blk];
            let packed = qs[w_base + w];
            if (Q8) {
                // Word w%8 of a Q8_0 block is k = blk*32 + (w%8)*4 .. +3.
                let wv = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(packed << 24u, packed << 16u, packed << 8u, packed)) >> vec4<u32>(24u)) * scale;
                let kq = blk * 8u + (w % 8u);
                acc_a += vec4<f32>(dot(wv, x[r0 + kq]), dot(wv, x[r1 + kq]), dot(wv, x[r2 + kq]), dot(wv, x[r3 + kq]));
                acc_b += vec4<f32>(dot(wv, x[r4 + kq]), dot(wv, x[r5 + kq]), dot(wv, x[r6 + kq]), dot(wv, x[r7 + kq]));
                continue;
            }
            let lo = vec4<f32>(vec4<u32>(packed, packed >> 8u, packed >> 16u, packed >> 24u) & vec4<u32>(0xFu));
            let hi = vec4<f32>(vec4<u32>(packed >> 4u, packed >> 12u, packed >> 20u, packed >> 28u) & vec4<u32>(0xFu));
            let w_lo = (lo - vec4<f32>(8.0)) * scale;
            let w_hi = (hi - vec4<f32>(8.0)) * scale;
            // vec4 index of k = blk*32 + (w%4)*4 within a row; +4 is k+16.
            let kq = blk * 8u + (w % 4u);
            acc_a += vec4<f32>(
                dot(w_lo, x[r0 + kq]) + dot(w_hi, x[r0 + kq + 4u]),
                dot(w_lo, x[r1 + kq]) + dot(w_hi, x[r1 + kq + 4u]),
                dot(w_lo, x[r2 + kq]) + dot(w_hi, x[r2 + kq + 4u]),
                dot(w_lo, x[r3 + kq]) + dot(w_hi, x[r3 + kq + 4u]),
            );
            acc_b += vec4<f32>(
                dot(w_lo, x[r4 + kq]) + dot(w_hi, x[r4 + kq + 4u]),
                dot(w_lo, x[r5 + kq]) + dot(w_hi, x[r5 + kq + 4u]),
                dot(w_lo, x[r6 + kq]) + dot(w_hi, x[r6 + kq + 4u]),
                dot(w_lo, x[r7 + kq]) + dot(w_hi, x[r7 + kq + 4u]),
            );
        }
    }

    partial_a[tid] = acc_a;
    partial_b[tid] = acc_b;
    workgroupBarrier();
    for (var stride: u32 = THREADS_PER_ROW / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_a[tid] += partial_a[tid + stride];
            partial_b[tid] += partial_b[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && col_ok) {
        let bias = b[dims.n_offset + n];
        let sa = partial_a[tid] + vec4<f32>(bias);
        let sb = partial_b[tid] + vec4<f32>(bias);
        let vals = array<f32, 8>(sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w);
        for (var r: u32 = 0u; r < mloc; r = r + 1u) {
            var v = vals[r];
            if (dims.act == 1u) {
                v = max(v, 0.0);
            }
            out[(m0 + r) * dims.n_total + dims.n_offset + n] = v;
        }
    }
}
