// Register-blocked tiled GEMM for prefill over Q6_K weights: out[m, n] =
// x[m, :] . W[n, :] + b[n]. The tile, swizzle and register block are those of
// linear_q4_tiled_rb.wgsl (64x64 output tile per workgroup, 16x16 threads, a
// 4x4 block each, K stepped 32 at a time); the weight tile is dequantised from
// the Q6_K buffers of linear_q6k.wgsl (`quant.rs::split_q6k_blocks`).
//
// A stage of 32 values is one group of one half of a 256-value super-block
// (stage s: half = (s / 4) % 2, group g = s % 4): its values l = 0..31 come
// from ql byte (g % 2) * 32 + l (low nibble for g < 2, high for g >= 2) and qh
// byte l (bits 2g..2g+1) of that half, with scales entries half * 8 + l / 16 +
// 2g. Thread (l, p) takes words c = p and c = p + 4 of that group (four
// weights each, one scale per word), assembles the four 6-bit values of a word
// with masks and shifts and converts them with the exponent trick of
// linear_q6k_decode.wgsl.
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
@group(0) @binding(1) var<storage, read> ql: array<u32>;
@group(0) @binding(2) var<storage, read> qh: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> dscale: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

const TM: u32 = 64u;
const TN: u32 = 64u;

var<workgroup> xs: array<vec4<f32>, 512>;
var<workgroup> ws: array<vec4<f32>, 512>;

fn sw(r: u32, k4: u32) -> u32 {
    return r * 8u + (k4 ^ (r & 7u));
}

// Four bytes (each 0..63) of `q` as `byte - 32` floats.
fn q4(q: u32) -> vec4<f32> {
    let v = vec4<u32>(q & 0xFFu, (q >> 8u) & 0xFFu, (q >> 16u) & 0xFFu, q >> 24u) | vec4<u32>(0x4B000000u);
    return bitcast<vec4<f32>>(v) - vec4<f32>(8388640.0);
}

// Signed byte `k` (0..3) of `w`.
fn sc_i8(w: u32, k: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - 8u * k)) >> 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;
    let k4 = dims.k / 4u;
    let bpr = dims.blocks_per_row * 8u; // 32-value stages

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
        let sb = n_safe * dims.blocks_per_row + blk / 8u;
        let half = (blk / 4u) & 1u;
        let g = blk & 3u;
        let d = dscale[sb];
        let wl_base = sb * 32u + half * 16u + (g & 1u) * 8u;
        let wh_base = sb * 16u + half * 8u;
        let lo_shift = 4u * (g >> 1u);
        var wv = array<vec4<f32>, 2>(vec4<f32>(0.0), vec4<f32>(0.0));
        for (var h: u32 = 0u; h < 2u; h = h + 1u) {
            let c = p + 4u * h;
            let wl = ql[wl_base + c];
            let wq = qh[wh_base + c];
            let q = ((wl >> lo_shift) & 0x0F0F0F0Fu) | (((wq >> (2u * g)) & 0x03030303u) << 4u);
            let e = half * 8u + (c >> 2u) + 2u * g;
            let sc = sc_i8(scales[sb * 4u + (e >> 2u)], e & 3u);
            wv[h] = select(vec4<f32>(0.0), q4(q) * (d * sc), n_ok);
        }
        ws[sw(l, p)] = wv[0];
        ws[sw(l, p + 4u)] = wv[1];
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
