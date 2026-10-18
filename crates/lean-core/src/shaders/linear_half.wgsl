// Naive F16/BF16 matmul (one thread per output element), the reference
// kernel for weights kept at two bytes per value (see linear_half_decode.wgsl
// for the packing). Same bindings and Dims contract as linear_half_decode.wgsl,
// except x is read as plain f32. `dims.k` is even.
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
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const BF16: bool = false;

fn pair(v: u32) -> vec2<f32> {
    if (BF16) {
        return vec2<f32>(bitcast<f32>(v << 16u), bitcast<f32>(v & 0xFFFF0000u));
    }
    return unpack2x16float(v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= dims.m || n >= dims.n) {
        return;
    }
    let row = n * (dims.k / 2u);
    var acc = 0.0;
    for (var i: u32 = 0u; i < dims.k / 2u; i = i + 1u) {
        let v = pair(w[row + i]);
        acc += x[m * dims.k + 2u * i] * v.x + x[m * dims.k + 2u * i + 1u] * v.y;
    }
    acc += b[dims.n_offset + n];
    if (dims.act == 1u) {
        acc = max(acc, 0.0);
    }
    out[m * dims.n_total + dims.n_offset + n] = acc;
}
