// Adapted from t0-web/crates/t0-fast/src/shaders/silu_mul.wgsl: t0-fast's
// version reads gate/up out of one concatenated [rows, 2*hidden] buffer
// (t0's mlp.0 produces both halves in one matmul). Qwen2's GGUF keeps
// ffn_gate.weight and ffn_up.weight as two separate tensors, so this
// version takes gate and up as two separate buffers instead of splitting
// one — same SwiGLU math (`silu(gate) * up`).
struct Dims { rows: u32, hidden: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.rows * dims.hidden;
    if (id >= total) {
        return;
    }
    let g = gate[id];
    let u = up[id];
    let silu = g / (1.0 + exp(-g));
    out[id] = silu * u;
}
