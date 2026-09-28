// Register-blocked tiled Q8_0 GEMM: this project's own port of the same
// 32x32/TK=16/2x2-register-blocking scheme as `linear_q4_tiled_rb.wgsl`
// (see that file's header for the size-gating rationale), applied to
// Q8_0-resident weights instead of Q4_0. Q8_0 prefill previously only had
// the naive per-element kernel (`linear_q8.wgsl`) - this is the first
// tiled/shared-memory Q8_0 prefill kernel in this crate, added alongside
// the Q4_0 one so both quant formats get the same size-gated choice in
// `model.rs::linear`.
//
// Adapted from this project's own `linear_q4_tiled_rb.wgsl` (dequant swapped
// for Q8_0's signed-byte-per-value block layout, see `linear_q8.wgsl`'s
// doc comment for that layout) and originally informed by t0-web's
// `crates/t0-fast/src/shaders/linear_tiled_q8.wgsl` tile scheme (same
// TM/TN/TK/2x2 accumulator structure). `Dims`/output addressing differ
// from t0-web's original the same way `linear_q4_tiled_rb.wgsl`'s does:
// `n_offset`/`n_total` for this crate's row-chunked weight bindings.
const TM: u32 = 32u;
const TN: u32 = 32u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;

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

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

var<workgroup> a_tile: array<array<f32, TK>, TM>;
var<workgroup> b_tile: array<array<f32, TK>, TN>;

fn dequant_q8(row: u32, col: u32) -> f32 {
    let blk = col / 32u;
    let scale = scales[row * dims.blocks_per_row + blk];
    let within = col % 32u;
    let word = qs[row * (dims.k / 4u) + blk * 8u + within / 4u];
    let shift = (within % 4u) * 8u;
    let byteval = (word >> shift) & 0xFFu;
    let sv = (i32(byteval) << 24u) >> 24u;
    return f32(sv) * scale;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    var kb: u32 = 0u;
    loop {
        if (kb >= dims.k) {
            break;
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm = idx / TK;
            let kk = idx % TK;
            let m = m0 + mm;
            let ka = kb + kk;
            a_tile[mm][kk] = select(0.0, x[m * dims.k + ka], m < dims.m && ka < dims.k);
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let nn = idx / TK;
            let kk = idx % TK;
            let n = n0 + nn;
            let bk = kb + kk;
            let valid = n < dims.n && bk < dims.k;
            let safe_n = select(0u, n, valid);
            let safe_bk = select(0u, bk, valid);
            b_tile[nn][kk] = select(0.0, dequant_q8(safe_n, safe_bk), valid);
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
            let a0 = a_tile[ty][kk];
            let a1 = a_tile[ty + 16u][kk];
            let b0 = b_tile[tx][kk];
            let b1 = b_tile[tx + 16u][kk];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
        kb = kb + TK;
    }

    let m0_ = m0 + ty;
    let m1_ = m0 + ty + 16u;
    let n0_ = n0 + tx;
    let n1_ = n0 + tx + 16u;
    let relu = dims.act == 1u;
    let n_total = dims.n_total;
    let n_off = dims.n_offset;

    if (m0_ < dims.m && n0_ < dims.n) {
        var v = acc00 + b[n_off + n0_];
        if (relu) { v = max(v, 0.0); }
        out[m0_ * n_total + n_off + n0_] = v;
    }
    if (m0_ < dims.m && n1_ < dims.n) {
        var v = acc01 + b[n_off + n1_];
        if (relu) { v = max(v, 0.0); }
        out[m0_ * n_total + n_off + n1_] = v;
    }
    if (m1_ < dims.m && n0_ < dims.n) {
        var v = acc10 + b[n_off + n0_];
        if (relu) { v = max(v, 0.0); }
        out[m1_ * n_total + n_off + n0_] = v;
    }
    if (m1_ < dims.m && n1_ < dims.n) {
        var v = acc11 + b[n_off + n1_];
        if (relu) { v = max(v, 0.0); }
        out[m1_ * n_total + n_off + n1_] = v;
    }
}
