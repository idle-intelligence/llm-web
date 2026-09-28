// Q8_0 counterpart of embed_gather_q4.wgsl - same gather-without-full-
// dequant shape, but the table is Q8_0-resident (Qwen3's official GGUFs
// ship no Q4_0 quant at all, only Q8_0 - see the qwen3 survey). Same
// per-block layout as linear_q8.wgsl/gather_dequant_q8_rows.wgsl: `qs`
// packs 4 signed int8 values per u32, row-major, block-contiguous;
// `scales[block]` is the already-f32-decoded per-32-value-block scale.
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.rows * dims.hidden;
    if (id >= total) {
        return;
    }
    let row = id / dims.hidden;
    let d = id % dims.hidden;
    let token = token_ids[row];

    let blk = d / 32u;
    let j = d % 32u;
    let wi = j / 4u;
    let byte_i = j % 4u;

    let scale = scales[token * dims.blocks_per_row + blk];
    let word = qs[(token * dims.blocks_per_row + blk) * 8u + wi];
    let byteval = (word >> (byte_i * 8u)) & 0xFFu;
    let sv = (i32(byteval) << 24u) >> 24u;
    out[id] = f32(sv) * scale;
}
