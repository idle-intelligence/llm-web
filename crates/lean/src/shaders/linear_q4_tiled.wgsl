// Adapted from llm-wasm/src/wgsl/shader_q4_tiled.wgsl (Q4_0 Tiled Matmul,
// M>1/prefill, native-only — see that file's header for the full
// weight-reuse rationale and the TM=TN=64/MICRO=4/"v3" tuning history).
// Two differences from the llm-wasm original: (1) bias is folded in here
// (`b[n]`) so this kernel satisfies the same linear-layer contract as
// linear_q4.wgsl/linear.wgsl, rather than being bias-free the way
// llm-wasm's `q4_matmul` is (bias handled separately there); (2) `info` is
// a `Dims` uniform struct (this crate's convention) instead of a raw
// `array<u32>` storage buffer. Block/word layout, tile sizes, and the
// vectorized cooperative-dequant loop are otherwise unchanged.
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
    let M = dims.m;
    let K = dims.k;
    let N = dims.n;
    let blocks_per_row = dims.blocks_per_row;

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

        let n_local = tid / 4u;
        let wi = tid % 4u;
        let n_global = n_base + n_local;
        let row_off = n_local * TK;
        let base_i = wi * 4u;
        if (n_global < N) {
            let global_block = n_global * blocks_per_row + blk;
            let scale = scales[global_block];
            let packed = qs[global_block * 4u + wi];
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

        for (var idx: u32 = tid; idx < TM * TK; idx = idx + 256u) {
            let m_local = idx / TK;
            let k_local = idx % TK;
            let m_global = m_base + m_local;
            var val: f32 = 0.0;
            if (m_global < M && (k_base + k_local) < K) {
                val = x[m_global * K + k_base + k_local];
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
            var v = acc[i][j] + b[n_global];
            if (dims.act == 1u) {
                v = max(v, 0.0);
            }
            out[m_global * N + n_global] = v;
        }
    }
}
