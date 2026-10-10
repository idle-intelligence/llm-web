// New kernel (no direct t0-fast/llm-wasm equivalent — t0-fast has no
// embedding table, and llm-wasm's `EmbeddingStore` dequantizes rows on the
// CPU, not the GPU). Gathers `rows` embedding rows straight out of the
// Q4_0-resident `token_embd.weight` ([vocab, hidden]) without ever
// dequantizing the full table — same block layout/convention as
// linear_q4.wgsl (`quant.rs::split_q4_blocks`): nibble j < 16 -> low nibble,
// j >= 16 -> high nibble of byte (j % 16), `value = (nibble - 8) * scale`.
// Dispatched over a 2-D grid when `rows*hidden` needs more than
// `max_compute_workgroups_per_dimension` (65535) groups at this shader's
// workgroup_size(256) - see `model.rs::grid1d`'s doc comment. `stride_x` is
// that dispatch's x-dimension group count times 256.
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, stride_x: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.y * dims.stride_x + gid.x;
    let total = dims.rows * dims.hidden;
    if (id >= total) {
        return;
    }
    let row = id / dims.hidden;
    let d = id % dims.hidden;
    let token = token_ids[row];

    let blk = d / 32u;
    let local = d % 32u;
    let is_hi = local >= 16u;
    let jj = select(local, local - 16u, is_hi);
    let wi = jj / 4u;
    let byte_i = jj % 4u;

    let scale = scales[token * dims.blocks_per_row + blk];
    let word = qs[(token * dims.blocks_per_row + blk) * 4u + wi];
    let byteval = (word >> (byte_i * 8u)) & 0xFFu;
    let nibble = select(f32(byteval & 0xFu), f32((byteval >> 4u) & 0xFu), is_hi);
    out[id] = (nibble - 8.0) * scale;
}
