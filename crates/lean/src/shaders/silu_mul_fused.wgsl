// Step 3 of this session's brief (gate/up fusion): `src` is one
// `[rows, 2*hidden]` buffer (`ffn_gate.weight`/`ffn_up.weight` concatenated
// at load time into one `MatMulWeight`, see `model.rs::gguf_matmul_concat2`
// - one `linear()` dispatch produces both halves per row instead of two).
// Replaces the old two-buffer `silu_mul.wgsl` (gate/up as separate
// buffers, from separate matmul dispatches) now that both halves come out
// of the same fused matmul. Ported from t0-web's own
// `crates/t0-fast/src/shaders/silu_mul.wgsl`, which already assumed this
// fused-matmul layout (t0's `mlp.0` produces both halves in one matmul,
// same as this crate's fused `gate_up_w` now does) - this crate previously
// had gate/up as two separate GGUF tensors, hence its own prior two-buffer
// variant's doc comment explaining the difference, now obsolete.
struct Dims { rows: u32, hidden: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.rows * dims.hidden;
    if (id >= total) {
        return;
    }
    let row = id / dims.hidden;
    let c = id % dims.hidden;
    let base = row * 2u * dims.hidden;
    let gate = src[base + c];
    let up = src[base + dims.hidden + c];
    let silu = gate / (1.0 + exp(-gate));
    out[id] = silu * up;
}
