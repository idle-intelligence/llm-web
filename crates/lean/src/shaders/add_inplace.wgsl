// Ported verbatim from t0-web/crates/t0-fast/src/shaders/add_inplace.wgsl
// (generic elementwise op, no model-specific content).
// a[i] += b[i], length `len`.
//
// Dispatched over a 2-D grid when `len` needs more than
// `max_compute_workgroups_per_dimension` (65535) groups at this shader's
// workgroup_size(256) - see `model.rs::grid1d`'s doc comment. `stride_x` is
// that dispatch's x-dimension group count times 256, letting this shader
// recover the flat index from `gid.y`/`gid.x` instead of assuming `gid.x`
// alone spans `len` (true only when `y`=1, the common case).
struct Dims { len: u32, stride_x: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * dims.stride_x + gid.x;
    if (i >= dims.len) {
        return;
    }
    a[i] = a[i] + b[i];
}
