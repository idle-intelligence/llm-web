// Q4_0 Tiled Matmul — M>1 (prefill), native-only.
//
// Naive-kernel weight reuse problem: shader_naive.wgsl re-dequantizes every
// weight row from scratch for every one of the M input rows (M-fold
// redundant dequant work during prefill, where M is the prompt length).
// This kernel instead dequantizes a `[TN, TK]` tile of `weights` once into
// workgroup shared memory and reuses it across all `TM` rows of `input` in
// the same tile (and vice versa: the `input` tile is reused across all `TN`
// output columns).
//
// TK=32 is exactly one Q4_0 block, so each K-tile step needs only one scale
// per weight row (no partial-block handling). TM=TN=128, 16x16=256 threads
// per workgroup, each thread computes an 8x8 micro-tile of the output
// (128/16=8 per axis).
//
// v1 of this kernel used TM=TN=64/MICRO=4 (16KB shared mem) and *regressed*
// prefill 4.4x vs the naive kernel (368s vs 84s on the 2225-token
// `02_tools_single` fixture) despite passing correctness tests — with only
// 512 FMAs of compute between each pair of workgroupBarrier() calls (one
// Q4_0 block's worth, TK=32), barrier/sync overhead dominated over the
// weight-reuse win. Doubling TM/TN to 128 (MICRO=8) quadruples the compute
// (2048 FMAs) done per barrier pair for the same TK=32 dequant/load cost,
// at the price of doubling shared memory to 32KB/workgroup (still under
// Metal's default per-threadgroup limit) and more registers/thread (64
// accumulators instead of 16).
//
// Dispatch: ceil(N / TN) workgroups in X, ceil(M / TM) in Y. B is assumed 1
// (this crate's single-session assumption — Q4Attention::forward asserts
// it elsewhere; this kernel doesn't handle B>1).

// K5 (docs/BENCHMARKS.md): see shader_q4_matvec.wgsl's binding comment —
// `weights` is nibbles-only (4 aligned u32/block), scale lives in `scales`.
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> info: array<u32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u; // one Q4_0 block
const WG_DIM: u32 = 16u; // 16x16 threads
const MICRO: u32 = 4u;   // each thread computes a 4x4 micro-tile

var<workgroup> w_tile: array<f32, 2048>; // [TN=64][TK=32]
var<workgroup> x_tile: array<f32, 2048>; // [TM=64][TK=32]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let M = info[1];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4];

    let tid = local_id.y * WG_DIM + local_id.x; // 0..255
    let n_base = wg_id.x * TN;
    let m_base = wg_id.y * TM;

    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < MICRO; i = i + 1u) {
        for (var j = 0u; j < MICRO; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let k_base = blk * TK;

        // Cooperative dequant of the weight tile, vectorized: TN=64 rows x
        // 4 u32 words/row (16 data bytes per Q4_0 block) == 256 == WG_SIZE,
        // so each thread handles exactly one (row, word) pair, reading one
        // packed u32 (4 bytes = 8 nibbles = 8 output values, matching
        // shader_naive.wgsl's vec4 dequant) instead of 8 independent scalar
        // `read_u8` calls per row — the original per-element scalar version
        // of this loop measured 30-40 GFLOP/s (vs the naive kernel's ~159
        // GFLOP/s effective at full-model granularity); this vectorized
        // form is the fix.
        let n_local = tid / 4u;
        let wi = tid % 4u;
        let n_global = n_base + n_local;
        let row_off = n_local * TK;
        let base_i = wi * 4u;
        if (n_global < N) {
            let global_block = n_global * blocks_per_row + blk;
            let scale = scales[global_block];
            let packed = weights[global_block * 4u + wi];
            let b0 = packed & 0xFFu;
            let b1 = (packed >> 8u) & 0xFFu;
            let b2 = (packed >> 16u) & 0xFFu;
            let b3 = (packed >> 24u) & 0xFFu;
            w_tile[row_off + base_i] = (f32(b0 & 0xFu) - 8.0) * scale;
            w_tile[row_off + base_i + 1u] = (f32(b1 & 0xFu) - 8.0) * scale;
            w_tile[row_off + base_i + 2u] = (f32(b2 & 0xFu) - 8.0) * scale;
            w_tile[row_off + base_i + 3u] = (f32(b3 & 0xFu) - 8.0) * scale;
            w_tile[row_off + 16u + base_i] = (f32((b0 >> 4u) & 0xFu) - 8.0) * scale;
            w_tile[row_off + 16u + base_i + 1u] = (f32((b1 >> 4u) & 0xFu) - 8.0) * scale;
            w_tile[row_off + 16u + base_i + 2u] = (f32((b2 >> 4u) & 0xFu) - 8.0) * scale;
            w_tile[row_off + 16u + base_i + 3u] = (f32((b3 >> 4u) & 0xFu) - 8.0) * scale;
        } else {
            w_tile[row_off + base_i] = 0.0;
            w_tile[row_off + base_i + 1u] = 0.0;
            w_tile[row_off + base_i + 2u] = 0.0;
            w_tile[row_off + base_i + 3u] = 0.0;
            w_tile[row_off + 16u + base_i] = 0.0;
            w_tile[row_off + 16u + base_i + 1u] = 0.0;
            w_tile[row_off + 16u + base_i + 2u] = 0.0;
            w_tile[row_off + 16u + base_i + 3u] = 0.0;
        }

        // Cooperative load of the input tile.
        for (var idx: u32 = tid; idx < TM * TK; idx = idx + 256u) {
            let m_local = idx / TK;
            let k_local = idx % TK;
            let m_global = m_base + m_local;
            var val: f32 = 0.0;
            if (m_global < M && (k_base + k_local) < K) {
                val = input[m_global * K + k_base + k_local];
            }
            x_tile[idx] = val;
        }

        workgroupBarrier();

        let m_local0 = local_id.y * MICRO;
        let n_local0 = local_id.x * MICRO;
        for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
            var xv: array<f32, 4>;
            var wv: array<f32, 4>;
            for (var i = 0u; i < MICRO; i = i + 1u) {
                xv[i] = x_tile[(m_local0 + i) * TK + kk];
            }
            for (var j = 0u; j < MICRO; j = j + 1u) {
                wv[j] = w_tile[(n_local0 + j) * TK + kk];
            }
            for (var i = 0u; i < MICRO; i = i + 1u) {
                for (var j = 0u; j < MICRO; j = j + 1u) {
                    acc[i][j] += xv[i] * wv[j];
                }
            }
        }

        workgroupBarrier();
    }

    let m_local0 = local_id.y * MICRO;
    let n_local0 = local_id.x * MICRO;
    for (var i = 0u; i < MICRO; i = i + 1u) {
        let m_global = m_base + m_local0 + i;
        if (m_global >= M) {
            continue;
        }
        for (var j = 0u; j < MICRO; j = j + 1u) {
            let n_global = n_base + n_local0 + j;
            if (n_global >= N) {
                continue;
            }
            output[m_global * N + n_global] = acc[i][j];
        }
    }
}
