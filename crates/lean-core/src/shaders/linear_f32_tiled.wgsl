// Register-blocked tiled F32 GEMM for prefill: out[m, n] = x[m, :] . W[n, :]
// + b[n], W as [n, k] row-major F32. The same tile as
// linear_q4_tiled_rb.wgsl (64x64 output tile per workgroup, 16x16 threads,
// each a 4x4 register block, K stepped 32 at a time through swizzled
// [64][8] vec4 tiles in workgroup memory), with the weight tile read as
// plain F32 instead of dequantised from Q4_0/Q8_0 blocks. It serves the
// tensors a quantised GGUF keeps at F32 (SmolLM2's Q4_0 files hold three
// ffn_down weights as Q4_1, which load as F32) and F16/F32 GGUFs.
// Requires k % 32 == 0 (model.rs falls back to linear.wgsl otherwise).
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const TM: u32 = 64u;
const TN: u32 = 64u;

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
    let bpr = dims.k / 32u;

    // Staging: thread -> (tile row/column l = tid % 64, part p = tid / 64):
    // vec4 slots p and p + 4 of the 32-value block, for the weight column and
    // the x row.
    let l = tid % 64u;
    let p = tid / 64u;
    let n_g = n0 + l;
    let n_ok = n_g < dims.n;
    let w_row = select(0u, n_g, n_ok) * k4;
    let m_g = m0 + l;
    let m_ok = m_g < dims.m;
    let x_row = select(0u, m_g, m_ok) * k4;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var blk: u32 = 0u; blk < bpr; blk = blk + 1u) {
        ws[sw(l, p)] = select(vec4<f32>(0.0), w[w_row + blk * 8u + p], n_ok);
        ws[sw(l, p + 4u)] = select(vec4<f32>(0.0), w[w_row + blk * 8u + p + 4u], n_ok);
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
                var v = v4[j] + b[n];
                if (dims.act == 1u) {
                    v = max(v, 0.0);
                }
                out[m * dims.n + n] = v;
            }
        }
    }
}
