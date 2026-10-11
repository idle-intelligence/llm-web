// Writes `rows` freshly computed K or V rows (`[rows, kv_heads, head_dim]`,
// row-major, starting at element `src_offset` of `src`) into the KV cache's
// `[kv_head, position, head_dim]` layout at positions `kv_base + row`. One
// dispatch per buffer per layer, replacing one `copy_buffer_to_buffer` per
// (row, kv_head): at a 1,800-token chunk that was 170k copy commands per
// forward and dominated its time. Flat index over a 2-D grid, same
// `stride_x` convention as add_inplace.wgsl (`model.rs::grid1d`).
struct Dims {
    rows: u32,
    kv_heads: u32,
    head_dim: u32,
    kv_base: u32,
    max_ctx: u32,
    src_offset: u32,
    stride_x: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * dims.stride_x + gid.x;
    if (i >= dims.rows * dims.kv_heads * dims.head_dim) {
        return;
    }
    let d = i % dims.head_dim;
    let h = (i / dims.head_dim) % dims.kv_heads;
    let row = i / (dims.head_dim * dims.kv_heads);
    cache[(h * dims.max_ctx + dims.kv_base + row) * dims.head_dim + d] = src[dims.src_offset + i];
}
