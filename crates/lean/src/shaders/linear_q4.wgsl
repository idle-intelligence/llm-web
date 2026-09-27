// Ported verbatim from t0-web/crates/t0-fast/src/shaders/linear_q4.wgsl.
// `w` is Q4_0-resident: 4 bits/value, `qs` packs 4 bytes (16 values: byte's
// low nibble -> value j, high nibble -> value j+16) per u32, 4 u32/32-value
// block, matching llama.cpp's `quantize_row_q4_0_ref` packing (see
// gguf.rs::dequantize_q4_0 / quant.rs::split_q4_blocks). `scales[block]` is
// f32 (decoded from f16 at load time). `blocks_per_row = k / 32`.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    _p0: u32,
    _p1: u32,
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
    let row_u32_base = n * dims.blocks_per_row * 4u;
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
        let word_base = row_u32_base + blk * 4u;
        for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
            let word = qs[word_base + wi];
            for (var byte_i: u32 = 0u; byte_i < 4u; byte_i = byte_i + 1u) {
                let byteval = (word >> (byte_i * 8u)) & 0xFFu;
                let lo = f32(byteval & 0xFu) - 8.0;
                let hi = f32((byteval >> 4u) & 0xFu) - 8.0;
                let j = wi * 4u + byte_i;
                acc = acc + x[x_base + k + j] * (lo * scale);
                acc = acc + x[x_base + k + 16u + j] * (hi * scale);
            }
        }
        k = k + 32u;
    }
    acc = acc + b[n];
    if (dims.act == 1u) {
        acc = max(acc, 0.0);
    }
    out[m * dims.n + n] = acc;
}
