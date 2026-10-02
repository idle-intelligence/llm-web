// Q8_0 counterpart of linear_q4_decode.wgsl, same layout (8 lanes per
// output row, 16 rows per 128-thread workgroup, one whole block per lane
// per step, shared-memory reduction) adapted to Q8_0 blocks: 32 signed
// int8 values packed 4 per u32 (8 words, two vec4<u32> loads per block),
// one scale per block. Same bindings/Dims contract as linear_q8.wgsl.
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
@group(0) @binding(1) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const LANES: u32 = 8u;
const ROWS_PER_WG: u32 = 16u; // WG_SIZE / LANES

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn i8x4(w: u32) -> vec4<f32> {
    let v = bitcast<vec4<i32>>(vec4<u32>(w << 24u, w << 16u, w << 8u, w)) >> vec4<u32>(24u);
    return vec4<f32>(v);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let lane = tid % LANES;
    let n = wg_id.x * ROWS_PER_WG + tid / LANES;
    let bpr = dims.blocks_per_row;

    var acc: f32 = 0.0;
    if (n < dims.n) {
        let row = n * bpr;
        for (var blk: u32 = lane; blk < bpr; blk = blk + LANES) {
            let q0 = qs[(row + blk) * 2u];
            let q1 = qs[(row + blk) * 2u + 1u];
            let s = scales[row + blk];
            let xb = blk * 8u;
            let d = dot(i8x4(q0.x), x[xb]) + dot(i8x4(q0.y), x[xb + 1u])
                + dot(i8x4(q0.z), x[xb + 2u]) + dot(i8x4(q0.w), x[xb + 3u])
                + dot(i8x4(q1.x), x[xb + 4u]) + dot(i8x4(q1.y), x[xb + 5u])
                + dot(i8x4(q1.z), x[xb + 6u]) + dot(i8x4(q1.w), x[xb + 7u]);
            acc += d * s;
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && n < dims.n) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
