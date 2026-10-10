// Split-half ("NeoX-style") RoPE, same convention as rope_neox.wgsl, but
// reading each row's absolute position from a caller-supplied buffer
// instead of assuming a contiguous `pos_base + row` run. This is what lets
// a chunk restart RoPE per block (llm-life variant A packs several
// per-cell prompts into one sequence, each block's positions starting back
// at the shared prefix length rather than continuing the previous block's
// count - see model.rs's `ForwardSpec::with_positions`).
//
// buf: [rows, heads, head_dim] contiguous, row_stride = heads*head_dim.
// cos_t/sin_t: [max_positions, half] with half = head_dim/2, position p's
// row read at `p * half + j`. positions: [rows] absolute position per row.
struct Dims { rows: u32, heads: u32, head_dim: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<uniform> dims: Dims;

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
    let pos = positions[row];

    let row_stride = dims.heads * dims.head_dim;
    let base = row * row_stride + head * dims.head_dim;
    let x0 = buf[base + j];
    let x1 = buf[base + half + j];
    let c = cos_t[pos * half + j];
    let s = sin_t[pos * half + j];

    buf[base + j] = x0 * c - x1 * s;
    buf[base + half + j] = x1 * c + x0 * s;
}
