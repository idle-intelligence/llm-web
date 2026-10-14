// Gathers `rows` embedding rows straight out of a Q5_0-resident
// `token_embd.weight` ([vocab, hidden]) without dequantizing the table: same
// layout as embed_gather_q4.wgsl plus `aux[block]`, the block's 32 fifth bits
// (bit j belongs to value j), `value = ((nibble | bit << 4) - 16) * d`
// (`quant.rs::split_q5_0_blocks`, `gguf.rs::dequantize_q5_0`). The dispatch
// grid is folded as in embed_gather_q4.wgsl.
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, stride_x: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.y * dims.stride_x + gid.x;
    if (id >= dims.rows * dims.hidden) {
        return;
    }
    let row = id / dims.hidden;
    let d = id % dims.hidden;
    let token = token_ids[row];

    let blk = token * dims.blocks_per_row + d / 32u;
    let local = d % 32u;
    let is_hi = local >= 16u;
    let jj = select(local, local - 16u, is_hi);
    let word = qs[blk * 4u + jj / 4u];
    let byteval = (word >> ((jj % 4u) * 8u)) & 0xFFu;
    let nibble = select(byteval & 0xFu, byteval >> 4u, is_hi);
    let q = nibble | (((aux[blk] >> local) & 1u) << 4u);
    out[id] = (f32(q) - 16.0) * scales[blk];
}
