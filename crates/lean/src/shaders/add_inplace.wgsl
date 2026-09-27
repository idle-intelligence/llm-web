// Ported verbatim from t0-web/crates/t0-fast/src/shaders/add_inplace.wgsl
// (generic elementwise op, no model-specific content).
// a[i] += b[i], length `len`.
struct Dims { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.len) {
        return;
    }
    a[i] = a[i] + b[i];
}
