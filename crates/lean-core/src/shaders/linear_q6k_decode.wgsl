// Decode-shaped (M=1) Q6_K matvec, following the same coalesced-matvec
// structure as linear_q4_decode.wgsl/linear_q8_decode.wgsl (128 threads/
// workgroup, ROWS_PER_WG=4 rows, THREADS_PER_ROW=32 lanes, shared-memory
// tree reduction) rather than linear_q6k.wgsl's one-thread-per-row-and-column
// naive loop. This was the naive kernel's documented gap: the official
// Qwen2.5-3B GGUF's `output.weight` ships Q6_K and hit the naive kernel at
// decode (7.6% of peak bandwidth per
// docs/runs/2026-09-29-lean-vs-llamacpp-profile.md), the only remaining
// naive-kernel matvec once rmsnorm was fixed.
//
// Per-block math (block_q6_K, 256-value super-block: `ql` low 4 bits/weight,
// `qh` high 2 bits/weight, 16 signed-i8 sub-block scales, one f32 `d`) is
// copied verbatim from linear_q6k.wgsl's inner loop - see that file's header
// for the full bit-layout derivation and its cross-check against
// `gguf.rs::dequantize_q6_k`/`quant.rs::split_q6k_blocks`. The only change
// here is *who* does which part of a block: the naive kernel's single thread
// walks all 64 `l`-in-`{half}` sub-iterations of a block serially; this
// kernel splits those same 64 sub-iterations across 32 lanes (2 each,
// `iter_idx = lane` and `iter_idx = lane + 32`), then reduces the 32 lanes'
// partial sums the same way linear_q4_decode.wgsl already does. Same values,
// same per-element formula, different accumulation order (parallel tree vs.
// serial) - the small float non-associativity this introduces matches what
// linear_q4_decode/linear_q8_decode already accept relative to their own
// naive counterparts, covered by the same fixture_parity_qwen25_3b gate this
// kernel change is gated on.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    // See linear_q4.wgsl's Dims doc comment: row-chunk offset/total for
    // weights split across bindings.
    n_offset: u32,
    n_total: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> ql: array<u32>;
@group(0) @binding(2) var<storage, read> qh: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> dscale: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn byte_of(word: u32, i: u32) -> u32 {
    return (word >> (i * 8u)) & 0xFFu;
}

fn sc_i8(word: u32, i: u32) -> f32 {
    let ub = byte_of(word, i);
    return f32((i32(ub) << 24u) >> 24u);
}

// One of a block's 64 (half, l) sub-iterations - see linear_q6k.wgsl's inner
// loop for what each part computes.
fn accumulate_subiter(iter_idx: u32, ql_base: u32, qh_base: u32, sc_base: u32, k_base: u32, dval: f32) -> f32 {
    let half = iter_idx / 32u;
    let l = iter_idx % 32u;
    let is = l / 16u;
    let ql_off = ql_base + half * 16u;
    let qh_off = qh_base + half * 8u;
    let y_base = k_base + half * 128u;

    let ql_w0 = ql[ql_off + l / 4u];
    let ql_b0 = byte_of(ql_w0, l % 4u);
    let idx1 = l + 32u;
    let ql_w1 = ql[ql_off + idx1 / 4u];
    let ql_b1 = byte_of(ql_w1, idx1 % 4u);
    let qh_w = qh[qh_off + l / 4u];
    let qh_b = byte_of(qh_w, l % 4u);

    let e0 = half * 8u + is;
    let e2 = e0 + 2u;
    let e4 = e0 + 4u;
    let e6 = e0 + 6u;
    let sc0 = sc_i8(scales[sc_base + e0 / 4u], e0 % 4u);
    let sc2 = sc_i8(scales[sc_base + e2 / 4u], e2 % 4u);
    let sc4 = sc_i8(scales[sc_base + e4 / 4u], e4 % 4u);
    let sc6 = sc_i8(scales[sc_base + e6 / 4u], e6 % 4u);

    let q1 = f32(i32((ql_b0 & 0xFu) | ((qh_b & 3u) << 4u)) - 32);
    let q2 = f32(i32((ql_b1 & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - 32);
    let q3 = f32(i32((ql_b0 >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - 32);
    let q4 = f32(i32((ql_b1 >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - 32);

    var s: f32 = 0.0;
    s = s + x[y_base + l] * (dval * sc0 * q1);
    s = s + x[y_base + 32u + l] * (dval * sc2 * q2);
    s = s + x[y_base + 64u + l] * (dval * sc4 * q3);
    s = s + x[y_base + 96u + l] * (dval * sc6 * q4);
    return s;
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

    let ql_row_base = n * blocks_per_row * 32u;
    let qh_row_base = n * blocks_per_row * 16u;
    let sc_row_base = n * blocks_per_row * 4u;
    let d_row_base = n * blocks_per_row;
    let x_base: u32 = 0u; // M is always 1 at decode (single row 0); no row offset into x.

    var acc: f32 = 0.0;
    if (row_has_output) {
        for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
            let dval = dscale[d_row_base + blk];
            let ql_base = ql_row_base + blk * 32u;
            let qh_base = qh_row_base + blk * 16u;
            let sc_base = sc_row_base + blk * 4u;
            let k_base = x_base + blk * 256u;
            acc += accumulate_subiter(lane, ql_base, qh_base, sc_base, k_base, dval);
            acc += accumulate_subiter(lane + 32u, ql_base, qh_base, sc_base, k_base, dval);
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
