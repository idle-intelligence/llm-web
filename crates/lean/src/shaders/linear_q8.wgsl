// Ported verbatim from t0-web/crates/t0-fast/src/shaders/linear_q8.wgsl.
// Same contract as linear.wgsl but `w` is Q8_0-resident: `qs` packs 4
// signed int8 weight values per u32 (row-major, block-contiguous), one
// `scales[block]` f32 per 32-value block (already decoded from the
// on-disk f16 scale at load time). `blocks_per_row = k / 32`.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    // See linear_q4.wgsl's Dims doc comment: row-chunk offset/total for
    // weights split across bindings.
    n_offset: u32,
    n_total: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= dims.m || n >= dims.n) {
        return;
    }
    let row_u32_base = n * (dims.k / 4u);
    let row_blk_base = n * dims.blocks_per_row;
    let x_base = m * dims.k;

    var acc: f32 = 0.0;
    var k: u32 = 0u;
    loop {
        if (k >= dims.k) {
            break;
        }
        let blk = k / 32u;
        let scale = scales[row_blk_base + blk];
        let word_base = row_u32_base + blk * 8u;
        for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
            let word = qs[word_base + wi];
            for (var byte_i: u32 = 0u; byte_i < 4u; byte_i = byte_i + 1u) {
                let shift = byte_i * 8u;
                let byteval = (word >> shift) & 0xFFu;
                let sv = (i32(byteval) << 24u) >> 24u;
                let wv = f32(sv) * scale;
                acc = acc + x[x_base + k + wi * 4u + byte_i] * wv;
            }
        }
        k = k + 32u;
    }
    acc = acc + b[dims.n_offset + n];
    if (dims.act == 1u) {
        acc = max(acc, 0.0);
    }
    out[m * dims.n_total + dims.n_offset + n] = acc;
}
