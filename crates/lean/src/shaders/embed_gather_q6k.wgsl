// Q6_K counterpart of embed_gather_q4.wgsl/embed_gather_q8.wgsl: gathers
// `rows` embedding rows straight out of a Q6_K-resident `token_embd.weight`
// without dequantizing the full table. Same block layout/index math as
// `linear_q6k.wgsl` (see that file's doc comment) and
// `quant.rs::split_q6k_blocks` - one output element per thread, so the
// per-(row,half,l)-group loop that kernel runs sequentially is instead
// picked by a single `select`/`if` chain on which of the 4 groups
// (`q1`/`q2`/`q3`/`q4`) this thread's `d` (hidden index) falls into.
struct Dims { rows: u32, hidden: u32, blocks_per_row: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> ql: array<u32>;
@group(0) @binding(2) var<storage, read> qh: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> dscale: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

fn byte_of(word: u32, i: u32) -> u32 {
    return (word >> (i * 8u)) & 0xFFu;
}

fn sc_i8(word: u32, i: u32) -> f32 {
    let ub = byte_of(word, i);
    return f32((i32(ub) << 24u) >> 24u);
}

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

    let blk = d / 256u;
    let local = d % 256u;
    let half = local / 128u;
    let p = local % 128u;
    let g = p / 32u;
    let l = p % 32u;
    let is = l / 16u;

    let ql_base = token * dims.blocks_per_row * 32u + blk * 32u + half * 16u;
    let qh_base = token * dims.blocks_per_row * 16u + blk * 16u + half * 8u;
    let sc_base = token * dims.blocks_per_row * 4u + blk * 4u;
    let d_base = token * dims.blocks_per_row + blk;

    // g==0/2 read ql[l]'s low/high nibble, g==1/3 read ql[l+32]'s.
    let ql_idx = select(l, l + 32u, g == 1u || g == 3u);
    let ql_byte = byte_of(ql[ql_base + ql_idx / 4u], ql_idx % 4u);
    let nibble = select(ql_byte & 0xFu, ql_byte >> 4u, g == 2u || g == 3u);

    let qh_byte = byte_of(qh[qh_base + l / 4u], l % 4u);
    let qh_shift = g * 2u;
    let qh_bits = (qh_byte >> qh_shift) & 3u;

    let sc_off = g * 2u; // group g's scale offset is {0,2,4,6} = g*2
    let entry = half * 8u + is + sc_off;
    let sc = sc_i8(scales[sc_base + entry / 4u], entry % 4u);

    let q = i32((nibble & 0xFu) | (qh_bits << 4u)) - 32;
    out[id] = dscale[d_base] * sc * f32(q);
}
