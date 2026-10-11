// Ported verbatim from t0-web/crates/t0-fast/src/shaders/linear.wgsl
// (generic F32 naive matvec/matmul; naive only in this slice, no tiled
// variant yet — see engine.rs's doc comment on what's deferred).
// out[m,n] = dot(x[m,:], w[n,:]) + b[n], optionally relu'd.
// w is PyTorch/GGUF [out,in] layout (row n = output channel n), so no transpose.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32, // 0 = none, 1 = relu
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= dims.m || n >= dims.n) {
        return;
    }
    var acc: f32 = 0.0;
    let x_base = m * dims.k;
    let w_base = n * dims.k;
    for (var k: u32 = 0u; k < dims.k; k = k + 1u) {
        acc = acc + x[x_base + k] * w[w_base + k];
    }
    acc = acc + b[n];
    if (dims.act == 1u) {
        acc = max(acc, 0.0);
    }
    out[m * dims.n + n] = acc;
}
