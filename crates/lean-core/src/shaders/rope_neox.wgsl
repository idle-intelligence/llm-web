// New kernel: split-half ("NeoX-style") RoPE, the convention Qwen2/Llama use
// in HF transformers (`rotate_half`: pairs are (d, d+head_dim/2), NOT
// adjacent (2j, 2j+1)). This differs from t0-fast's shaders/rope.wgsl,
// which implements t0's interleaved-pair xpos convention — that kernel is
// NOT reusable here; llama.cpp's GGUF conversion for Qwen2 uses
// LLAMA_ROPE_TYPE_NEOX (no Q/K weight permutation), so this kernel's pairing
// must match rotate_half directly. Structurally modeled on llm-wasm's
// shader_rope.wgsl (same cos/sin-table-by-position convention, same
// read-both-before-write-either safety argument for in-place q/k mutation),
// but operating on one buffer (q or k) per dispatch instead of llm-wasm's
// fused q+k kernel, and with an explicit `pos_base` field instead of q/k
// each in their own dispatch anyway.
//
// buf: [rows, heads, head_dim] contiguous, row_stride = heads*head_dim.
// cos_t/sin_t: [max_positions, half] with half = head_dim/2, position p's
// row read at `p * half + j`. Absolute position of buffer row r is
// `pos_base + r` (prefill: pos_base=0, r is the sequence index; decode:
// pos_base = kv_len, r is always 0 since decode processes one row).
struct Dims { rows: u32, heads: u32, head_dim: u32, pos_base: u32 };

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = dims.head_dim / 2u;
    let total = dims.rows * dims.heads * half;
    let id = gid.x;
    if (id >= total) {
        return;
    }
    let j = id % half;
    let vh = id / half;
    let head = vh % dims.heads;
    let row = vh / dims.heads;
    let pos = dims.pos_base + row;

    let row_stride = dims.heads * dims.head_dim;
    let base = row * row_stride + head * dims.head_dim;
    let x0 = buf[base + j];
    let x1 = buf[base + half + j];
    let c = cos_t[pos * half + j];
    let s = sin_t[pos * half + j];

    buf[base + j] = x0 * c - x1 * s;
    buf[base + half + j] = x1 * c + x0 * s;
}
