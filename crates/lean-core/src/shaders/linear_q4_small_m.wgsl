// Small-M Q4_0 matmul for short prefills (2 <= M < SMALL_M_MAX_ROWS in
// model.rs): out[m, n] = x[m, :] . W[n, :] + b[n].
//
// This project's earlier version of this file (row groups of 8, 4 lanes
// per output column, x read straight from the storage buffer) loaded 16
// x vec4s from global memory for every weight word it dequantised: one
// global load per multiply-add. On a GPU whose storage-buffer loads are
// not cached close to the ALUs (Adreno 6xx), a 36-row prefill ran its MLP
// matmuls at about 3% of the device's FP32 rate (docs/runs/
// 2026-10-03-lean-prefill.md). This version keeps the same row grouping
// and the same once-per-word dequantisation, and moves x into workgroup
// memory with a register block per thread, as linear_q4_tiled_rb.wgsl
// does for long prompts:
// - a workgroup is 8 query rows x 64 output columns, 64 threads: 16 column
//   lanes (`cl`) x 4 K-split lanes (`ks`);
// - each step stages 4 Q4_0 blocks (128 k) of the 8 x rows in workgroup
//   memory, and thread (ks, cl) takes block 4*step + ks of its 4 columns
//   (cl + 16j): one vec4<u32> load per column per block (a whole block's
//   4 words), so the 4 ks lanes of a column read 64 contiguous bytes;
// - the thread's 8 rows x 4 columns are 8 vec4 accumulators with fixed
//   names (no dynamically indexed private array); every x vec4 it reads
//   from workgroup memory feeds 4 columns, and the 16 column lanes read the
//   same x address (a broadcast);
// - the 4 K-split partial sums are added through workgroup memory at the
//   end, each thread writing 2 rows x 4 columns.
// The row group is workgroup_id.x and the column tile workgroup_id.y, so the
// workgroups that read the same weight columns are dispatched next to each
// other and the repeated weight reads (one per row group) can hit in cache.
// Rows past M read zeros and are never written.
//
// Q8 (pipeline override, `linear_q8_small_m` in engine.rs): the same
// kernel over Q8_0 blocks, two vec4<u32> per column per block (8 words of
// 4 signed bytes, word s = k4 slot s) instead of one (4 words of 8
// nibbles, low nibbles of word s = slot s, high nibbles = slot s + 4).
//
// Sizes are fixed (64 invocations, 8 KiB of workgroup storage), within
// WebGPU's guaranteed minimums on every device.
//
// Bindings and Dims are those of linear_q4.wgsl, except qs is read as
// vec4<u32> (one Q4_0 block, or half a Q8_0 block, per element).
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    n_offset: u32,
    n_total: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

override Q8: bool = false;

const ROWS: u32 = 8u;
const KS: u32 = 4u;
const CL: u32 = 16u;
const COLS: u32 = 64u; // CL * 4 columns per thread
// x tile: [8 rows][4 ks][8 vec4 slots], each ks run padded to 9 vec4 so the
// 4 ks lanes reading the same row and slot hit different banks.
const KS_STRIDE: u32 = 9u;
const ROW_STRIDE: u32 = 36u; // KS * KS_STRIDE

// One workgroup array for both phases: the x tile (ROWS * ROW_STRIDE = 288
// vec4s) during the K loop, then the K-split partial sums (64 threads x 8
// rows). 8 KiB in all, so that more workgroups fit on a compute unit.
var<workgroup> xs: array<vec4<f32>, 512>;

fn nib4(w: u32, s: f32) -> vec4<f32> {
    return (vec4<f32>(vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0)) * s;
}

fn i8x4(w: u32, s: f32) -> vec4<f32> {
    return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w << 24u, w << 16u, w << 8u, w)) >> vec4<u32>(24u)) * s;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let cl = tid % CL;
    let ks = tid / CL;
    let m0 = wg_id.x * ROWS;
    let n0 = wg_id.y * COLS;
    let k4 = dims.k / 4u;
    let bpr = dims.blocks_per_row;

    // Column j of this thread is n0 + cl + 16j; columns past N read column
    // 0's weights and are never written.
    var nb = vec4<u32>(n0 + cl) + vec4<u32>(0u, 16u, 32u, 48u);
    let n_ok = nb < vec4<u32>(dims.n);
    nb = select(vec4<u32>(0u), nb, n_ok);
    let sb = nb * bpr;

    // x staging: thread t loads 4 of the tile's 256 vec4s, row t / 8,
    // vec4 slot (t % 8) * 4 .. + 3 of the step's 32 (block (t % 8) / 2).
    let st_row = tid / 8u;
    let st_slot = (tid % 8u) * 4u;
    let st_m = m0 + st_row;
    let st_ok = st_m < dims.m;
    let st_x = select(0u, st_m, st_ok) * k4;
    let st_dst = st_row * ROW_STRIDE + (st_slot / 8u) * KS_STRIDE + st_slot % 8u;
    let xbase = ks * KS_STRIDE;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);

    for (var blk0: u32 = 0u; blk0 < bpr; blk0 = blk0 + KS) {
        let st_blk = blk0 + st_slot / 8u;
        let ld = st_ok && st_blk < bpr;
        let src = st_x + blk0 * 8u + st_slot;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            xs[st_dst + i] = select(vec4<f32>(0.0), x[select(0u, src + i, ld)], ld);
        }
        workgroupBarrier();

        let blk = blk0 + ks;
        if (blk < bpr) {
            let s = vec4<f32>(scales[sb.x + blk], scales[sb.y + blk], scales[sb.z + blk], scales[sb.w + blk]);
            let qi = select(sb + vec4<u32>(blk), (sb + vec4<u32>(blk)) * 2u, Q8);
            // 8 k4 slots; Q4_0: slot p < 4 is the low nibbles of word p, slot
            // p + 4 the high nibbles; Q8_0: slot p is word p (second vec4 from 4).
            var qa0 = qs[qi.x];
            var qa1 = qs[qi.y];
            var qa2 = qs[qi.z];
            var qa3 = qs[qi.w];
            for (var p: u32 = 0u; p < 8u; p = p + 1u) {
                let pw = p % 4u;
                if (Q8 && p == 4u) {
                    qa0 = qs[qi.x + 1u];
                    qa1 = qs[qi.y + 1u];
                    qa2 = qs[qi.z + 1u];
                    qa3 = qs[qi.w + 1u];
                }
                let wd = vec4<u32>(qa0[pw], qa1[pw], qa2[pw], qa3[pw]);
                var w0: vec4<f32>;
                var w1: vec4<f32>;
                var w2: vec4<f32>;
                var w3: vec4<f32>;
                if (Q8) {
                    w0 = i8x4(wd.x, s.x);
                    w1 = i8x4(wd.y, s.y);
                    w2 = i8x4(wd.z, s.z);
                    w3 = i8x4(wd.w, s.w);
                } else {
                    let sh = select(0u, 4u, p >= 4u);
                    w0 = nib4(wd.x >> sh, s.x);
                    w1 = nib4(wd.y >> sh, s.y);
                    w2 = nib4(wd.z >> sh, s.z);
                    w3 = nib4(wd.w >> sh, s.w);
                }
                let xb = xbase + p;
                var a = xs[xb];
                acc0 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 1u * ROW_STRIDE];
                acc1 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 2u * ROW_STRIDE];
                acc2 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 3u * ROW_STRIDE];
                acc3 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 4u * ROW_STRIDE];
                acc4 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 5u * ROW_STRIDE];
                acc5 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 6u * ROW_STRIDE];
                acc6 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
                a = xs[xb + 7u * ROW_STRIDE];
                acc7 += vec4<f32>(dot(a, w0), dot(a, w1), dot(a, w2), dot(a, w3));
            }
        }
        workgroupBarrier();
    }

    // K-split reduction (the loop's last barrier has retired the x tile):
    // xs[(ks * 16 + cl) * 8 + r] holds row r, columns j of thread (ks, cl); thread (g = tid / 16, cl) sums rows 2g, 2g + 1.
    let rb = tid * 8u;
    xs[rb] = acc0;
    xs[rb + 1u] = acc1;
    xs[rb + 2u] = acc2;
    xs[rb + 3u] = acc3;
    xs[rb + 4u] = acc4;
    xs[rb + 5u] = acc5;
    xs[rb + 6u] = acc6;
    xs[rb + 7u] = acc7;
    workgroupBarrier();

    let g = tid / CL;
    let bias = vec4<f32>(b[dims.n_offset + nb.x], b[dims.n_offset + nb.y], b[dims.n_offset + nb.z], b[dims.n_offset + nb.w]);
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let r = g * 2u + h;
        let m = m0 + r;
        if (m >= dims.m) {
            break;
        }
        var v = bias;
        for (var k: u32 = 0u; k < KS; k = k + 1u) {
            v += xs[(k * CL + cl) * 8u + r];
        }
        if (dims.act == 1u) {
            v = max(v, vec4<f32>(0.0));
        }
        let o = m * dims.n_total + dims.n_offset;
        if (n_ok.x) { out[o + nb.x] = v.x; }
        if (n_ok.y) { out[o + nb.y] = v.y; }
        if (n_ok.z) { out[o + nb.z] = v.z; }
        if (n_ok.w) { out[o + nb.w] = v.w; }
    }
}
