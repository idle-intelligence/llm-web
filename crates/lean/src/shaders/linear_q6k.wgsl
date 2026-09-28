// Naive per-output-element Q6_K matmul/matvec - the portable-by-construction
// kernel (runs on every WebGPU adapter, WASM included, no cooperative-group
// or shared-memory tiling), same shape as linear_q4.wgsl/linear_q8.wgsl:
// one thread computes one (row, col) output, looping the whole K dimension.
// Used for both prefill (M>1) and decode (M=1) - Q6_K only ever shows up on
// `token_embd.weight`/`output.weight` in this crate's models (one matmul
// per forward, not a hot loop the way q/k/v/o/gate/up/down are), so a
// tiled/coalesced variant is not worth porting yet - see model.rs's `linear()`
// doc comment on Q8_0 for the same reasoning.
//
// `w` is Q6_K-resident (llama.cpp's `block_q6_K`, 256-value super-block):
// `ql` packs the low 4 bits of every 6-bit weight (128 bytes/block -> 32
// u32), `qh` packs the high 2 bits (64 bytes/block -> 16 u32, two 2-bit
// fields per byte), `scales` packs 16 signed-i8 per-16-value sub-block
// scales (4 u32/block, sign-extended the same way linear_q8.wgsl already
// does), `d` is the super-block's own f32 scale (decoded from f16 at load
// time, one per block like every other quant kind's `scales`/`d` buffer).
// Index math and the `q - 32` unsigned-to-signed offset match
// `gguf.rs::dequantize_q6_k` / `quant.rs::split_q6k_blocks` exactly (see
// those doc comments for the block-halves/`is`/scale-offset derivation) -
// this kernel is verified against them by
// `quant.rs::tests::q6k_split_matches_reference_dequant`.
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
@group(0) @binding(1) var<storage, read> ql: array<u32>;
@group(0) @binding(2) var<storage, read> qh: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> dscale: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

fn byte_of(word: u32, i: u32) -> u32 {
    return (word >> (i * 8u)) & 0xFFu;
}

fn sc_i8(word: u32, i: u32) -> f32 {
    let ub = byte_of(word, i);
    return f32((i32(ub) << 24u) >> 24u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= dims.m || n >= dims.n) {
        return;
    }
    let ql_row_base = n * dims.blocks_per_row * 32u;
    let qh_row_base = n * dims.blocks_per_row * 16u;
    let sc_row_base = n * dims.blocks_per_row * 4u;
    let d_row_base = n * dims.blocks_per_row;
    let x_base = m * dims.k;

    var acc: f32 = 0.0;
    for (var blk: u32 = 0u; blk < dims.blocks_per_row; blk = blk + 1u) {
        let dval = dscale[d_row_base + blk];
        let ql_base = ql_row_base + blk * 32u;
        let qh_base = qh_row_base + blk * 16u;
        let sc_base = sc_row_base + blk * 4u;
        let k_base = x_base + blk * 256u;

        for (var half: u32 = 0u; half < 2u; half = half + 1u) {
            let ql_off = ql_base + half * 16u;
            let qh_off = qh_base + half * 8u;
            let y_base = k_base + half * 128u;

            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let is = l / 16u;
                let ql_w0 = ql[ql_off + l / 4u];
                let ql_b0 = byte_of(ql_w0, l % 4u);
                let idx1 = l + 32u;
                let ql_w1 = ql[ql_off + idx1 / 4u];
                let ql_b1 = byte_of(ql_w1, idx1 % 4u);
                let qh_w = qh[qh_off + l / 4u];
                let qh_b = byte_of(qh_w, l % 4u);

                let e0 = half * 8u + is;
                let e2 = e0 + 2u;
                let e4 = e0 + 4u;
                let e6 = e0 + 6u;
                let sc0 = sc_i8(scales[sc_base + e0 / 4u], e0 % 4u);
                let sc2 = sc_i8(scales[sc_base + e2 / 4u], e2 % 4u);
                let sc4 = sc_i8(scales[sc_base + e4 / 4u], e4 % 4u);
                let sc6 = sc_i8(scales[sc_base + e6 / 4u], e6 % 4u);

                let q1 = f32(i32((ql_b0 & 0xFu) | ((qh_b & 3u) << 4u)) - 32);
                let q2 = f32(i32((ql_b1 & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - 32);
                let q3 = f32(i32((ql_b0 >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - 32);
                let q4 = f32(i32((ql_b1 >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - 32);

                acc = acc + x[y_base + l] * (dval * sc0 * q1);
                acc = acc + x[y_base + 32u + l] * (dval * sc2 * q2);
                acc = acc + x[y_base + 64u + l] * (dval * sc4 * q3);
                acc = acc + x[y_base + 96u + l] * (dval * sc6 * q4);
            }
        }
    }
    acc = acc + b[dims.n_offset + n];
    if (dims.act == 1u) {
        acc = max(acc, 0.0);
    }
    out[m * dims.n_total + dims.n_offset + n] = acc;
}
