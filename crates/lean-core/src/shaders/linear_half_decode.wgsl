// Matvec for decode (M = 1) over F16 or BF16 weights kept at two bytes per
// value, packed two per u32 (value 2i in the low half). F16 widens with the
// core WGSL unpack2x16float, so no browser needs shader-f16 for it; BF16
// (build constant BF16) shifts the 16 bits into the top of an f32.
//
// Lane layout as linear_q4_decode.wgsl: 8 lanes share an output row and take
// units of 8 weights (one 16-byte vec4<u32> load, 8 consecutive units = 128
// contiguous bytes per row and step), 128 threads cover 16 rows, the lanes'
// partial sums are added through workgroup memory. x is read as two vec4s per
// unit. `dims.k` is a multiple of 8.
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
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const LANES: u32 = 8u;
const ROWS_PER_WG: u32 = 16u; // WG_SIZE / LANES

const BF16: bool = false;

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn pair(v: u32) -> vec2<f32> {
    if (BF16) {
        return vec2<f32>(bitcast<f32>(v << 16u), bitcast<f32>(v & 0xFFFF0000u));
    }
    return unpack2x16float(v);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let lane = tid % LANES;
    let n = wg_id.x * ROWS_PER_WG + tid / LANES;
    let row_ok = n < dims.n;
    let units = dims.k / 8u;
    let row = min(n, dims.n - 1u) * units;

    var acc = 0.0;
    for (var u: u32 = lane; u < units; u = u + LANES) {
        let q = w[row + u];
        acc += dot(vec4<f32>(pair(q.x), pair(q.y)), x[2u * u]) + dot(vec4<f32>(pair(q.z), pair(q.w)), x[2u * u + 1u]);
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u && row_ok) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
