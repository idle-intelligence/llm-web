// Adapted from llm-wasm/src/wgsl/shader_q4_tiled.wgsl (Q4_0 Tiled Matmul,
// M>1/prefill, native-only — see that file's header for the full
// weight-reuse rationale and the TM=TN=64/MICRO=4/"v3" tuning history).
// Two differences from the llm-wasm original: (1) bias is folded in here
// (`b[n]`) so this kernel satisfies the same linear-layer contract as
// linear_q4.wgsl/linear.wgsl, rather than being bias-free the way
// llm-wasm's `q4_matmul` is (bias handled separately there); (2) `info` is
// a `Dims` uniform struct (this crate's convention) instead of a raw
// `array<u32>` storage buffer. Block/tile sizes and the MICRO=4 register
// tiling are otherwise unchanged from that port.
//
// vec4 rewrite (this session, see docs/runs/2026-09-28-lean-perf.md's
// ~4.7%-of-peak finding): the original port read `x` and `qs` element-by-
// element (one scalar f32/u32 load per thread per iteration) and ran the
// TK=32 inner dot-product loop scalar. `x` and `qs` are now bound as
// `array<vec4<f32>>`/`array<vec4<u32>>` (both are always 16-byte-aligned:
// `K` and Q4_0's 32-element block are always multiples of 4), so each load
// pulls 4 elements/one full 16-byte Q4_0 dequant-source word in a single
// instruction, `w_tile`/`x_tile` are `vec4<f32>` arrays at the same total
// byte size as before (2048 f32 == 512 vec4, no shared-memory increase),
// and the inner product loop uses `dot()` over 8 vec4 groups instead of 32
// scalar multiply-adds. The Q4_0 dequant itself is now done by one thread
// per output row (`tid < 64u`, one `vec4<u32>` load = the whole 16-byte
// block) instead of 4 threads each loading one of its 4 words — fewer,
// larger transactions instead of many small ones — while every thread
// still participates in the `x_tile` vec4 load in the same iteration (no
// added barrier for keeping both groups busy). Dequant math/mapping
// (nibble -> k_local index) is byte-for-byte the same as before, just
// grouped by word instead of split across 4 threads.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    // See linear_q4.wgsl's Dims doc comment: row-chunk offset/total for
    // weights split across bindings. `n`/`N` here stay the chunk's local
    // row count (qs/scales are per-chunk buffers, addressed from 0); only
    // the bias read and output write need the global column.
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

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u; // one Q4_0 block
const TK4: u32 = 8u; // TK / 4 (vec4 groups per row per block)
const WG_DIM: u32 = 16u; // 16x16 threads
const MICRO: u32 = 4u;   // each thread computes a 4x4 micro-tile

var<workgroup> w_tile: array<vec4<f32>, 512>; // [TN=64][TK4=8]
var<workgroup> x_tile: array<vec4<f32>, 512>; // [TM=64][TK4=8]

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

        // Dequant: one thread per output row (n_local = tid, only tid < 64
        // active), one vec4<u32> load = the block's full 16 bytes.
        if (tid < TN) {
            let n_local = tid;
            let n_global = n_base + n_local;
            if (n_global < N) {
                let global_block = n_global * blocks_per_row + blk;
                let scale = scales[global_block];
                let words4 = qs[global_block];
                let words = array<u32, 4>(words4.x, words4.y, words4.z, words4.w);
                for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
                    let w = words[wi];
                    let b0 = w & 0xFFu;
                    let b1 = (w >> 8u) & 0xFFu;
                    let b2 = (w >> 16u) & 0xFFu;
                    let b3 = (w >> 24u) & 0xFFu;
                    // low nibbles -> k_local = wi*4..wi*4+3, group = wi
                    w_tile[n_local * TK4 + wi] = vec4<f32>(
                        (f32(b0 & 0xFu) - 8.0) * scale,
                        (f32(b1 & 0xFu) - 8.0) * scale,
                        (f32(b2 & 0xFu) - 8.0) * scale,
                        (f32(b3 & 0xFu) - 8.0) * scale,
                    );
                    // high nibbles -> k_local = 16+wi*4..+3, group = 4+wi
                    w_tile[n_local * TK4 + 4u + wi] = vec4<f32>(
                        (f32((b0 >> 4u) & 0xFu) - 8.0) * scale,
                        (f32((b1 >> 4u) & 0xFu) - 8.0) * scale,
                        (f32((b2 >> 4u) & 0xFu) - 8.0) * scale,
                        (f32((b3 >> 4u) & 0xFu) - 8.0) * scale,
                    );
                }
            } else {
                for (var g: u32 = 0u; g < TK4; g = g + 1u) {
                    w_tile[n_local * TK4 + g] = vec4<f32>(0.0);
                }
            }
        }

        // x_tile: all 256 threads load one vec4 group each iteration
        // (TM*TK4 = 512, so 2 iterations at stride 256).
        for (var idx4: u32 = tid; idx4 < TM * TK4; idx4 = idx4 + 256u) {
            let m_local = idx4 / TK4;
            let g = idx4 % TK4;
            let m_global = m_base + m_local;
            var val: vec4<f32> = vec4<f32>(0.0);
            if (m_global < M) {
                val = x[(m_global * K + k_base + g * 4u) / 4u];
            }
            x_tile[idx4] = val;
        }

        workgroupBarrier();

        let m_local0 = local_id.y * MICRO;
        let n_local0 = local_id.x * MICRO;
        for (var g: u32 = 0u; g < TK4; g = g + 1u) {
            var xv: array<vec4<f32>, 4>;
            var wv: array<vec4<f32>, 4>;
            for (var i = 0u; i < MICRO; i = i + 1u) {
                xv[i] = x_tile[(m_local0 + i) * TK4 + g];
            }
            for (var j = 0u; j < MICRO; j = j + 1u) {
                wv[j] = w_tile[(n_local0 + j) * TK4 + g];
            }
            for (var i = 0u; i < MICRO; i = i + 1u) {
                for (var j = 0u; j < MICRO; j = j + 1u) {
                    acc[i][j] += dot(xv[i], wv[j]);
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
            var v = acc[i][j] + b[dims.n_offset + n_global];
            if (dims.act == 1u) {
                v = max(v, 0.0);
            }
            out[m_global * dims.n_total + dims.n_offset + n_global] = v;
        }
    }
}
