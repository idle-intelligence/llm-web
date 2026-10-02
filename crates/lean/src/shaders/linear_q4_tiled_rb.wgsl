// Register-blocked tiled Q4_0 GEMM for prefill: out[m, n] = x[m, :] . W[n, :]
// + b[n]. 64x64 output tile per workgroup, 16x16 threads, each thread a
// 4x4 register block (rows 4ty..4ty+3, columns tx + 16j), one Q4_0 block
// (K step 32) per shared-memory stage.
//
// Rewritten from this project's earlier 32x32/TK=16 version of this file
// (itself from t0-web's linear_tiled_q4.wgsl) after it measured
// pathological on a mobile GPU. What changed and why:
// - the old weight tile was [TN][TK] floats read as b_tile[tx][kk]: a
//   stride of TK floats across threads, so all 16 reads hit one bank. Here
//   both tiles are [64][8] vec4s (a row's 32 k values), with the vec4 slot
//   XOR-swizzled by the row's low 3 bits, so the 16 threads along x read
//   16 different columns at the same k from 8 different bank groups;
// - one stage per 32-value Q4_0 block (two barriers per 32 K instead of
//   per 16), and each thread dequantises one whole u32 word (8 values) per
//   stage with one scale load, instead of one value per word/scale load;
// - every shared store is a whole vec4 written by one thread (filling a
//   vec4 tile with per-component stores from 4 threads races where a
//   component store becomes a whole-vector read-modify-write);
// - each thread produces 16 outputs from 8 shared vec4 loads per 4 k
//   (the old kernel produced 4 outputs from 4 scalar loads per k).
//
// Sizes are fixed so the kernel is valid on every WebGPU device: 256
// invocations and 16 KiB of workgroup storage are exactly the spec's
// guaranteed minimums (maxComputeInvocationsPerWorkgroup = 256,
// maxComputeWorkgroupStorageSize = 16384), and Engine::new_async checks
// both limits before creating the pipeline. Rows and columns past M/N are
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

const TM: u32 = 64u;
const TN: u32 = 64u;

// [64 rows][8 vec4] each: x rows of the tile, dequantised weight columns.
var<workgroup> xs: array<vec4<f32>, 512>;
var<workgroup> ws: array<vec4<f32>, 512>;

fn sw(r: u32, k4: u32) -> u32 {
    return r * 8u + (k4 ^ (r & 7u));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;
    let k4 = dims.k / 4u;
    let bpr = dims.blocks_per_row;

    // Staging: thread -> (tile row/column l = tid % 64, part p = tid / 64).
    // Weights: word p of column l's block (low nibbles are k4 slot p, high
    // nibbles slot p + 4). x: vec4 slots p and p + 4 of row l.
    let l = tid % 64u;
    let p = tid / 64u;
    let n_g = n0 + l;
    let n_ok = n_g < dims.n;
    let n_safe = select(0u, n_g, n_ok);
    let m_g = m0 + l;
    let m_ok = m_g < dims.m;
    let x_row = select(0u, m_g, m_ok) * k4;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var blk: u32 = 0u; blk < bpr; blk = blk + 1u) {
        let scale = select(0.0, scales[n_safe * bpr + blk], n_ok);
        let word = qs[(n_safe * bpr + blk) * 4u + p];
        ws[sw(l, p)] = (vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale;
        ws[sw(l, p + 4u)] = (vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale;
        xs[sw(l, p)] = select(vec4<f32>(0.0), x[x_row + blk * 8u + p], m_ok);
        xs[sw(l, p + 4u)] = select(vec4<f32>(0.0), x[x_row + blk * 8u + p + 4u], m_ok);
        workgroupBarrier();

        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let a0 = xs[sw(ty * 4u, q)];
            let a1 = xs[sw(ty * 4u + 1u, q)];
            let a2 = xs[sw(ty * 4u + 2u, q)];
            let a3 = xs[sw(ty * 4u + 3u, q)];
            let w0 = ws[sw(tx, q)];
            let w1 = ws[sw(tx + 16u, q)];
            let w2 = ws[sw(tx + 32u, q)];
            let w3 = ws[sw(tx + 48u, q)];
            acc0 += vec4<f32>(dot(a0, w0), dot(a0, w1), dot(a0, w2), dot(a0, w3));
            acc1 += vec4<f32>(dot(a1, w0), dot(a1, w1), dot(a1, w2), dot(a1, w3));
            acc2 += vec4<f32>(dot(a2, w0), dot(a2, w1), dot(a2, w2), dot(a2, w3));
            acc3 += vec4<f32>(dot(a3, w0), dot(a3, w1), dot(a3, w2), dot(a3, w3));
        }
        workgroupBarrier();
    }

    let n_out = n0 + tx;
    let m_out = m0 + ty * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let m = m_out + i;
        if (m >= dims.m) {
            break;
        }
        let v4 = accs[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let n = n_out + j * 16u;
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
