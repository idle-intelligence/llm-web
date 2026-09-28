// Q8_0 counterpart of linear_q4_decode.wgsl, same structure (4 rows/
// workgroup, 32 lanes/row, one shared-memory reduction per row) adapted to
// Q8_0 blocks: 32 signed int8 values per block, packed 4/u32 (8 words per
// block, vs Q4_0's 4 words per block since Q4_0 packs 2 values/byte). One
// scale per 32-value block. Same bindings/Dims contract as linear_q8.wgsl.
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

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn accumulate_word(word_idx: u32, weights_row_base: u32, scale_row_base: u32) -> f32 {
    let blk = word_idx / 8u;
    let wj = word_idx % 8u;
    let scale = scales[scale_row_base + blk];
    let packed = qs[weights_row_base + word_idx];

    let b0 = packed & 0xFFu;
    let b1 = (packed >> 8u) & 0xFFu;
    let b2 = (packed >> 16u) & 0xFFu;
    let b3 = (packed >> 24u) & 0xFFu;

    let sv0 = (i32(b0) << 24u) >> 24u;
    let sv1 = (i32(b1) << 24u) >> 24u;
    let sv2 = (i32(b2) << 24u) >> 24u;
    let sv3 = (i32(b3) << 24u) >> 24u;

    let k0 = blk * 32u + wj * 4u;

    let w = vec4<f32>(f32(sv0), f32(sv1), f32(sv2), f32(sv3)) * scale;
    let in_v = vec4<f32>(x[k0], x[k0 + 1u], x[k0 + 2u], x[k0 + 3u]);

    return dot(w, in_v);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let N = dims.n;
    let blocks_per_row = dims.blocks_per_row;

    let tid = local_id.x;
    let row_in_wg = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * ROWS_PER_WG + row_in_wg;
    let row_has_output = n < N;

    let weights_row_base = n * blocks_per_row * 8u;
    let scale_row_base = n * blocks_per_row;
    let total_words = blocks_per_row * 8u;

    var acc: f32 = 0.0;
    if (row_has_output) {
        var w0: u32 = lane;
        loop {
            if (w0 >= total_words) {
                break;
            }
            acc += accumulate_word(w0, weights_row_base, scale_row_base);
            let w1 = w0 + 32u;
            if (w1 < total_words) {
                acc += accumulate_word(w1, weights_row_base, scale_row_base);
            }
            let w2 = w0 + 64u;
            if (w2 < total_words) {
                acc += accumulate_word(w2, weights_row_base, scale_row_base);
            }
            let w3 = w0 + 96u;
            if (w3 < total_words) {
                acc += accumulate_word(w3, weights_row_base, scale_row_base);
            }
            w0 = w0 + 128u;
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = THREADS_PER_ROW / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u && row_has_output) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
