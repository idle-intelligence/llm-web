// Register-blocked tiled Q4_0 GEMM for prefill: out[m, n] = x[m, :] . W[n, :]
// + b[n]. 64x64 output tile per workgroup, 16x8 = 128 threads, each thread
// an 8x4 register block (rows 8ty..8ty+7, columns tx + 16j), one Q4_0 block
// (K step 32) per shared-memory stage.
//
// Compared with a 16x16 thread layout with 4x4 blocks, a thread here
// produces 32 outputs from 12 shared vec4 loads per 4 k (8 x rows shared by
// the 16 threads of a row, 4 weight columns), against 16 outputs from 8
// loads, and the workgroup has half the threads to synchronise at the two
// barriers per stage. The accumulators stay at 32 floats per thread, the
// same register footprint as linear_q4_small_m.wgsl.
//
// Shared tiles are [64][8] vec4s (a row's 32 k values), with the vec4 slot
// XOR-swizzled by the row's low 3 bits, so the 16 threads along x read 16
// different columns at the same k from 8 different bank groups. Staging
// loops over the tile's vec4s with the workgroup as stride; every shared
// store is a whole vec4 written by one thread. Each stage dequantises a
// weight word (8 values) with one scale load.
//
// Q41 (build constant, `linear_q41_tiled_rb` in engine.rs): Q4_1 blocks,
// value = nibble * d + m, with `scales` holding [d, m] f32 pairs.
//
// Q8 (build constant, `linear_q8_tiled_rb` in engine.rs): the same
// kernel over Q8_0 blocks; a thread dequantises words pp and pp + 4 of its
// column's block (4 signed bytes each, k4 slots pp and pp + 4).
//
// Sizes are fixed so the kernel is valid on every WebGPU device: 128
// invocations and 16 KiB of workgroup storage are within the spec's
// guaranteed minimums (maxComputeInvocationsPerWorkgroup = 256,
// maxComputeWorkgroupStorageSize = 16384). Rows and columns past M/N are
// zero-filled in the tiles and never written.
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

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

const Q8: bool = false;
// Q41 (build constant, `linear_q41_tiled_rb` in engine.rs): Q4_1 blocks,
// value = nibble * d + m; `scales` holds [d, m] f32 pairs.
const Q41: bool = false;
// Q50 and K4 (build constants, `linear_q50_tiled_rb`, `linear_k4_tiled_rb`
// in engine.rs): Q5_0 and Q4_K blocks; see aux_common.wgsl. K4 runs the Q4_1
// arithmetic with the block's (d * sc, -dmin * m) in place of its (d, m).
const Q50: bool = false;
const K4: bool = false;

//@AUX

const TM: u32 = 64u;
const TN: u32 = 64u;

// [TM rows][8 vec4] each: x rows of the tile, dequantised weight columns.
var<workgroup> xs: array<vec4<f32>, 512>;
var<workgroup> ws: array<vec4<f32>, 512>;

fn sw(r: u32, k4: u32) -> u32 {
    return r * 8u + (k4 ^ (r & 7u));
}

@compute @workgroup_size(TN / 4u, TM / 8u, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let nt = (TN / 4u) * (TM / 8u);
    let tid = ty * (TN / 4u) + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;
    let k4 = dims.k / 4u;
    let bpr = dims.blocks_per_row;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);

    for (var blk: u32 = 0u; blk < bpr; blk = blk + 1u) {
        // Weights: TN columns x 4 words (word pp = k4 slots pp and pp + 4).
        for (var i: u32 = tid; i < TN * 4u; i = i + nt) {
            let c = i / 4u;
            let pp = i % 4u;
            let n_g = n0 + c;
            let n_ok = n_g < dims.n;
            let n_safe = select(0u, n_g, n_ok);
            var scale = 0.0;
            var mn = 0.0;
            if (K4) {
                let dm = k4_dm((n_safe * bpr + blk) >> 3u, blk & 7u);
                scale = select(0.0, dm.x, n_ok);
                mn = select(0.0, dm.y, n_ok);
            } else if (Q41) {
                scale = select(0.0, scales[2u * (n_safe * bpr + blk)], n_ok);
                mn = select(0.0, scales[2u * (n_safe * bpr + blk) + 1u], n_ok);
            } else {
                scale = select(0.0, scales[n_safe * bpr + blk], n_ok);
            }
            if (Q8) {
                let w0 = qs[(n_safe * bpr + blk) * 8u + pp];
                let w1 = qs[(n_safe * bpr + blk) * 8u + pp + 4u];
                ws[sw(c, pp)] = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w0 << 24u, w0 << 16u, w0 << 8u, w0)) >> vec4<u32>(24u)) * scale;
                ws[sw(c, pp + 4u)] = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w1 << 24u, w1 << 16u, w1 << 8u, w1)) >> vec4<u32>(24u)) * scale;
            } else if (Q50) {
                let word = qs[(n_safe * bpr + blk) * 4u + pp];
                let qh = aux_at(n_safe * bpr + blk);
                ws[sw(c, pp)] = q5_vals(word, false, pp, qh) * scale;
                ws[sw(c, pp + 4u)] = q5_vals(word, true, pp, qh) * scale;
            } else if (Q41 || K4) {
                let word = qs[(n_safe * bpr + blk) * 4u + pp];
                ws[sw(c, pp)] = vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFu)) * scale + vec4<f32>(mn);
                ws[sw(c, pp + 4u)] = vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(0xFu)) * scale + vec4<f32>(mn);
            } else {
                let word = qs[(n_safe * bpr + blk) * 4u + pp];
                ws[sw(c, pp)] = (vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale;
                ws[sw(c, pp + 4u)] = (vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale;
            }
        }
        // x: TM rows x 8 vec4.
        for (var i: u32 = tid; i < TM * 8u; i = i + nt) {
            let r = i / 8u;
            let s4 = i % 8u;
            let m_g = m0 + r;
            let m_ok = m_g < dims.m;
            let x_row = select(0u, m_g, m_ok) * k4;
            xs[sw(r, s4)] = select(vec4<f32>(0.0), x[x_row + blk * 8u + s4], m_ok);
        }
        workgroupBarrier();

        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let a0 = xs[sw(ty * 8u + 0u, q)];
            let a1 = xs[sw(ty * 8u + 1u, q)];
            let a2 = xs[sw(ty * 8u + 2u, q)];
            let a3 = xs[sw(ty * 8u + 3u, q)];
            let a4 = xs[sw(ty * 8u + 4u, q)];
            let a5 = xs[sw(ty * 8u + 5u, q)];
            let a6 = xs[sw(ty * 8u + 6u, q)];
            let a7 = xs[sw(ty * 8u + 7u, q)];
            let w0 = ws[sw(tx, q)];
            let w1 = ws[sw(tx + TN / 4u, q)];
            let w2 = ws[sw(tx + TN / 2u, q)];
            let w3 = ws[sw(tx + 3u * (TN / 4u), q)];
            acc0 += vec4<f32>(dot(a0, w0), dot(a0, w1), dot(a0, w2), dot(a0, w3));
            acc1 += vec4<f32>(dot(a1, w0), dot(a1, w1), dot(a1, w2), dot(a1, w3));
            acc2 += vec4<f32>(dot(a2, w0), dot(a2, w1), dot(a2, w2), dot(a2, w3));
            acc3 += vec4<f32>(dot(a3, w0), dot(a3, w1), dot(a3, w2), dot(a3, w3));
            acc4 += vec4<f32>(dot(a4, w0), dot(a4, w1), dot(a4, w2), dot(a4, w3));
            acc5 += vec4<f32>(dot(a5, w0), dot(a5, w1), dot(a5, w2), dot(a5, w3));
            acc6 += vec4<f32>(dot(a6, w0), dot(a6, w1), dot(a6, w2), dot(a6, w3));
            acc7 += vec4<f32>(dot(a7, w0), dot(a7, w1), dot(a7, w2), dot(a7, w3));
        }
        workgroupBarrier();
    }

    let n_out = n0 + tx;
    let m_out = m0 + ty * 8u;
    let accs = array<vec4<f32>, 8>(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let m = m_out + i;
        if (m >= dims.m) {
            break;
        }
        let v4 = accs[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let n = n_out + j * (TN / 4u);
            if (n < dims.n) {
                var v = v4[j] + b[dims.n_offset + n];
                if (dims.act == 1u) {
                    v = max(v, 0.0);
                }
                out[m * dims.n_total + dims.n_offset + n] = v;
            }
        }
    }
}
