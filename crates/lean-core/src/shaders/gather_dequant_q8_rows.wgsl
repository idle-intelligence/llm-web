// Gathers and dequantizes selected rows of a Q8_0 matmul weight (the lm
// head, `output.weight`) into a small contiguous F32 buffer - the GPU side
// of a sliced lm-head: a caller wanting logits for only a handful of vocab
// ids (llm-life's variant A/B read exactly 2 - "0"/"1") never needs to
// materialize a `[rows, vocab]` logits buffer. `model::linear()`'s
// existing `MatMulWeight::F32` path then does the actual matmul against
// this gathered weight - no new matmul kernel needed, only the gather.
//
// Same per-block dequant math as linear_q8.wgsl (`qs` packs 4 signed int8
// values per u32, row-major, block-contiguous; `scales[block]` is the
// already-f32-decoded per-32-value-block scale), read for `row_ids[i]`
// (relative to whichever chunk's `qs`/`scales` buffers are bound - the
// caller maps an absolute vocab id to its owning chunk and local row
// before calling this, see model.rs's `gather_dequant_head_rows`) and
// written to output row `dims.out_row_offset + i`, letting one call fill in
// one row of a larger multi-call gather (model.rs dispatches once per
// selected id today, since llm-life's sliced sets are tiny).
struct Dims {
    n_selected: u32,
    k: u32,
    blocks_per_row: u32,
    out_row_offset: u32,
};

@group(0) @binding(0) var<storage, read> row_ids: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = dims.n_selected * dims.k;
    if (id >= total) {
        return;
    }
    let i = id / dims.k;
    let d = id % dims.k;
    let row = row_ids[i];
    let blk = d / 32u;
    let j = d % 32u;
    let wi = j / 4u;
    let byte_i = j % 4u;
    let scale = scales[row * dims.blocks_per_row + blk];
    let word = qs[row * (dims.k / 4u) + blk * 8u + wi];
    let shift = byte_i * 8u;
    let byteval = (word >> shift) & 0xFFu;
    let sv = (i32(byteval) << 24u) >> 24u;
    out[(dims.out_row_offset + i) * dims.k + d] = f32(sv) * scale;
}
