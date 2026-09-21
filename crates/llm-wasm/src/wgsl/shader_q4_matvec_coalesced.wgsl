// Q4_0 Coalesced Matvec — M=1 (decode) — Session 6 (docs/BENCHMARKS.md).
//
// Replaces K1's stride-4-words-per-lane access pattern (each lane owns one
// whole Q4_0 block = 4 consecutive words, so lane l reads words 4l..4l+3
// while lane l+1 reads 4l+4..4l+7 — a stride-4-words gap between lanes,
// uncoalesced) with a word-interleaved mapping: a row's nibble data is
// `blocks_per_row * 4` consecutive u32 words (see gguf.rs's `Q4Tensor`
// module doc / `from_q4_bytes` — the repacked `weights` buffer stores one
// row's blocks back-to-back, 4 aligned words/block, scale stripped into the
// parallel `scales` buffer). Lane `l` (0..31) reads word `l`, then word
// `l+32`, `l+64`, ... — consecutive lanes read consecutive words, so each
// 32-lane row-group's read of one 128-byte-aligned chunk is a single
// coalesced transaction instead of 8 scattered ones.
//
// Byte/word layout within a block (unchanged from K1, verified against
// `Q4Tensor::from_q4_bytes`): byte `b` (0..15) of a block holds element `b`
// in its low nibble and element `b+16` in its high nibble (GGUF's Q4_0
// pairing, copied verbatim by `from_q4_bytes` — only the 2-byte f16 scale is
// stripped out). Word `w` (0..3) of a block holds bytes `4w..4w+3`, i.e.
// elements `4w..4w+3` (low nibbles) and `4w+16..4w+19` (high nibbles) — 8
// elements per word. So for global word index `word_idx` within a row:
//   blk = word_idx / 4          (which Q4_0 block, 32 elements)
//   wj  = word_idx % 4          (which word within that block)
//   k_lo = blk*32 + wj*4        (base of the 4 low-nibble elements)
//   k_hi = k_lo + 16            (base of the 4 high-nibble elements)
//
// Thread mapping (WG_SIZE=128): ROWS_PER_WG=4 output rows per workgroup,
// THREADS_PER_ROW=32 lanes cooperatively summing one row's K elements.
// `x` is read directly from the global `input` buffer (not staged into
// workgroup-shared memory): x is tiny relative to the weight traffic this
// kernel is bandwidth-bound on (K floats = 8-44KB vs. blocks_per_row*20
// bytes/row of weight+scale traffic per *row*, read by 4 rows/workgroup
// independently) and stays resident in the GPU's read cache across the
// row-groups in a workgroup and across workgroups scheduled close in time;
// measured isolated GB/s (docs/BENCHMARKS.md Session 6) confirms this is
// not the bottleneck once weight reads are coalesced. Word loop unrolled
// x4 (4 words = 128 bytes = one coalesced transaction's worth per lane
// group per outer iteration) to amortize loop overhead.
//
// Tint uniformity: unlike K1, this kernel's only `workgroupBarrier()` calls
// are in the final tree reduction, which every thread reaches
// unconditionally (row_has_output/b_valid only gate the accumulation loads
// and the final store, never the barriers or the reduction loop's bounds —
// `stride` is derived from the compile-time `THREADS_PER_ROW` constant, not
// a storage load) — no `workgroupUniformLoad` staging needed here (contrast
// K1's tile-loop `break` bound on `K`, which does gate barriers and does
// need it).

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> info: array<u32>;

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

// Accumulate one word's 8 elements' contribution to the row's dot product.
fn accumulate_word(
    word_idx: u32,
    weights_row_base: u32,
    scale_row_base: u32,
    input_base: u32,
) -> f32 {
    let blk = word_idx / 4u;
    let wj = word_idx % 4u;
    let scale = scales[scale_row_base + blk];
    let packed = weights[weights_row_base + word_idx];

    let b0 = packed & 0xFFu;
    let b1 = (packed >> 8u) & 0xFFu;
    let b2 = (packed >> 16u) & 0xFFu;
    let b3 = (packed >> 24u) & 0xFFu;

    let k_lo = blk * 32u + wj * 4u;
    let k_hi = k_lo + 16u;

    let w_lo = (vec4<f32>(
        f32(b0 & 0xFu), f32(b1 & 0xFu),
        f32(b2 & 0xFu), f32(b3 & 0xFu)
    ) - vec4<f32>(8.0)) * scale;
    let in_lo = vec4<f32>(
        input[input_base + k_lo], input[input_base + k_lo + 1u],
        input[input_base + k_lo + 2u], input[input_base + k_lo + 3u]
    );

    let w_hi = (vec4<f32>(
        f32((b0 >> 4u) & 0xFu), f32((b1 >> 4u) & 0xFu),
        f32((b2 >> 4u) & 0xFu), f32((b3 >> 4u) & 0xFu)
    ) - vec4<f32>(8.0)) * scale;
    let in_hi = vec4<f32>(
        input[input_base + k_hi], input[input_base + k_hi + 1u],
        input[input_base + k_hi + 2u], input[input_base + k_hi + 3u]
    );

    return dot(w_lo, in_lo) + dot(w_hi, in_hi);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let B = info[0];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4];

    let tid = local_id.x;
    let b = wg_id.y;
    let b_valid = b < B;

    let row_in_wg = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * ROWS_PER_WG + row_in_wg;
    let row_has_output = n < N && b_valid;

    let input_base = b * K;
    let weights_row_base = n * blocks_per_row * 4u;
    let scale_row_base = n * blocks_per_row;
    let total_words = blocks_per_row * 4u;

    var acc: f32 = 0.0;
    if (row_has_output) {
        var w0: u32 = lane;
        loop {
            if (w0 >= total_words) {
                break;
            }
            acc += accumulate_word(w0, weights_row_base, scale_row_base, input_base);
            let w1 = w0 + 32u;
            if (w1 < total_words) {
                acc += accumulate_word(w1, weights_row_base, scale_row_base, input_base);
            }
            let w2 = w0 + 64u;
            if (w2 < total_words) {
                acc += accumulate_word(w2, weights_row_base, scale_row_base, input_base);
            }
            let w3 = w0 + 96u;
            if (w3 < total_words) {
                acc += accumulate_word(w3, weights_row_base, scale_row_base, input_base);
            }
            w0 = w0 + 128u;
        }
    }

    // Tree-reduce within each row's 32-thread group. Unconditional/uniform
    // control flow (see header comment) — safe under Tint's barrier rule.
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
        output[b * N + n] = partial_sums[tid];
    }
}
