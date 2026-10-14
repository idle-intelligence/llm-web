// Gathers `rows` embedding rows out of an F16- or BF16-resident
// `token_embd.weight` ([vocab, hidden], two values per u32, value 2i in the
// low half). F16 widens with unpack2x16float, BF16 shifts the 16 bits into
// the top of an f32. The dispatch grid is folded as in embed_gather_q4.wgsl.
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, stride_x: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

override BF16: bool = false;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.y * dims.stride_x + gid.x;
    if (id >= dims.rows * dims.hidden) {
        return;
    }
    let row = id / dims.hidden;
    let d = id % dims.hidden;
    let i = token_ids[row] * dims.hidden + d;
    let word = w[i / 2u];
    let hi = (i & 1u) == 1u;
    if (BF16) {
        out[id] = bitcast<f32>(select(word << 16u, word & 0xFFFF0000u, hi));
    } else {
        let v = unpack2x16float(word);
        out[id] = select(v.x, v.y, hi);
    }
}
