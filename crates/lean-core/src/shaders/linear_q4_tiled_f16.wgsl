// f16 variant of linear_q4_tiled_rb.wgsl, used when the device grants
// `shader-f16`. Same 64x64 tile, 16x16 threads, 4x4 register block, K step
// of one 32-value block, same swizzle and the same bindings; what changes:
// - both shared tiles hold vec4<f16> (8 KiB of workgroup storage instead of
//   16 KiB): the dequantised weights (q - 8) * scale rounded once to f16,
//   and the activations rounded to f16 when staged;
// - the inner product of each 4-value slot is an f16 dot, and WINDOW slots
//   (4 * WINDOW k values; 8 is one whole Q4_0 block) are summed in f16
//   before they are widened and added to the f32 accumulators, so the
//   accumulation across blocks, and thus across the K dimension, stays in
//   f32. Window 4 changed the logit error by about 15% and was 4% slower.
// Output, bias, activation and dispatch shape are identical to the f32
// kernel.

enable f16;

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

override Q8: bool = false;
override WINDOW: u32 = 8u;
// Q50 and K4 (pipeline overrides, `linear_q50_tiled_f16`, `linear_k4_tiled_f16`
// in engine.rs): Q5_0 and Q4_K blocks; see aux_common.wgsl. K4 dequantises as
// nibble * (d * sc) - dmin * m, the weight rounded once to f16 like the other
// formats.
override Q50: bool = false;
override K4: bool = false;

//@AUX

const TM: u32 = 64u;
const TN: u32 = 64u;

// [64 rows][8 vec4] each: x rows of the tile, dequantised weight columns.
var<workgroup> xs: array<vec4<f16>, 512>;
var<workgroup> ws: array<vec4<f16>, 512>;

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
        var scale = 0.0;
        var mn = 0.0;
        if (K4) {
            let dm = k4_dm((n_safe * bpr + blk) >> 3u, blk & 7u);
            scale = select(0.0, dm.x, n_ok);
            mn = select(0.0, dm.y, n_ok);
        } else {
            scale = select(0.0, scales[n_safe * bpr + blk], n_ok);
        }
        if (Q8) {
            let w0 = qs[(n_safe * bpr + blk) * 8u + p];
            let w1 = qs[(n_safe * bpr + blk) * 8u + p + 4u];
            ws[sw(l, p)] = vec4<f16>(vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w0 << 24u, w0 << 16u, w0 << 8u, w0)) >> vec4<u32>(24u)) * scale);
            ws[sw(l, p + 4u)] = vec4<f16>(vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w1 << 24u, w1 << 16u, w1 << 8u, w1)) >> vec4<u32>(24u)) * scale);
        } else if (Q50) {
            let word = qs[(n_safe * bpr + blk) * 4u + p];
            let qh = aux_at(n_safe * bpr + blk);
            ws[sw(l, p)] = vec4<f16>(q5_vals(word, false, p, qh) * scale);
            ws[sw(l, p + 4u)] = vec4<f16>(q5_vals(word, true, p, qh) * scale);
        } else if (K4) {
            let word = qs[(n_safe * bpr + blk) * 4u + p];
            ws[sw(l, p)] = vec4<f16>(vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFu)) * scale + vec4<f32>(mn));
            ws[sw(l, p + 4u)] = vec4<f16>(vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(0xFu)) * scale + vec4<f32>(mn));
        } else {
            let word = qs[(n_safe * bpr + blk) * 4u + p];
            ws[sw(l, p)] = vec4<f16>((vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale);
            ws[sw(l, p + 4u)] = vec4<f16>((vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * scale);
        }
        xs[sw(l, p)] = vec4<f16>(select(vec4<f32>(0.0), x[x_row + blk * 8u + p], m_ok));
        xs[sw(l, p + 4u)] = vec4<f16>(select(vec4<f32>(0.0), x[x_row + blk * 8u + p + 4u], m_ok));
        workgroupBarrier();

        for (var q0: u32 = 0u; q0 < 8u; q0 = q0 + WINDOW) {
            var t0 = vec4<f16>(0.0);
            var t1 = vec4<f16>(0.0);
            var t2 = vec4<f16>(0.0);
            var t3 = vec4<f16>(0.0);
            for (var q: u32 = q0; q < q0 + WINDOW; q = q + 1u) {
                let a0 = xs[sw(ty * 4u, q)];
                let a1 = xs[sw(ty * 4u + 1u, q)];
                let a2 = xs[sw(ty * 4u + 2u, q)];
                let a3 = xs[sw(ty * 4u + 3u, q)];
                let w0 = ws[sw(tx, q)];
                let w1 = ws[sw(tx + 16u, q)];
                let w2 = ws[sw(tx + 32u, q)];
                let w3 = ws[sw(tx + 48u, q)];
                t0 += vec4<f16>(dot(a0, w0), dot(a0, w1), dot(a0, w2), dot(a0, w3));
                t1 += vec4<f16>(dot(a1, w0), dot(a1, w1), dot(a1, w2), dot(a1, w3));
                t2 += vec4<f16>(dot(a2, w0), dot(a2, w1), dot(a2, w2), dot(a2, w3));
                t3 += vec4<f16>(dot(a3, w0), dot(a3, w1), dot(a3, w2), dot(a3, w3));
            }
            acc0 += vec4<f32>(t0);
            acc1 += vec4<f32>(t1);
            acc2 += vec4<f32>(t2);
            acc3 += vec4<f32>(t3);
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
