// Gathers `rows` embedding rows out of a Q4_K-resident `token_embd.weight`
// ([vocab, hidden]) without dequantizing the table. Layout as in
// `quant.rs::split_q4_k_blocks`: 32-value blocks with Q4_0's nibble layout,
// `aux[5 * sb ..]` = d and dmin (two f16), 6-bit scales and mins as bytes of
// the 256-value super-block `sb` (see aux_common.wgsl);
// `value = d * sc * nibble - dmin * m` (`gguf.rs::dequantize_q4_k`).
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, stride_x: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

fn aux_at(i: u32) -> u32 {
    return aux[i];
}

//@K4_DM

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
    let dm = k4_dm(blk >> 3u, blk & 7u);
    out[id] = f32(nibble) * dm.x + dm.y;
}
