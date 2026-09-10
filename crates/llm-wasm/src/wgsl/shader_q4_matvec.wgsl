// Q4_0 Cooperative Matvec — M=1 (decode) — shared-memory reduction variant.
//
// Portable default: no subgroup ops, safe for WASM/WebGPU where
// `Features::SUBGROUP` may be unavailable (see shader_q4_matvec_subgroup.wgsl
// and gguf.rs's `has_subgroup_support`). Modeled on sts-web's
// shader_q4k_matvec_coop.wgsl cooperative-K-split structure, adapted from
// Q4_K's 256-element/144-byte block to Q4_0's 32-element/18-byte block (see
// shader_naive.wgsl for the byte layout this shares: f16 scale + 16 bytes of
// paired nibbles, low nibble -> element j, high nibble -> element j+16).
//
// Thread mapping (WG_SIZE=256 threads):
//   row_in_wg = tid / THREADS_PER_ROW   (0..ROWS_PER_WG-1: which output row)
//   k_lane    = tid % THREADS_PER_ROW   (0..THREADS_PER_ROW-1: which K-slice)
// ROWS_PER_WG=8, THREADS_PER_ROW=32 output rows/threads-per-row.
//
// The input vector x (up to K=11008 f32 = 43KB, too large for a single
// workgroup-shared array under the browser's ~16KB minimum guarantee) is
// staged into shared memory TILE_K=1024 elements (4KB) at a time: all 256
// threads cooperatively load one tile, barrier, then every one of the 8
// row-groups in the workgroup reads that tile from shared memory instead of
// re-issuing 8x redundant global-memory reads of the same x values. TILE_K
// is exactly THREADS_PER_ROW(32) * block_size(32), so each of the 32
// k_lane threads in a row-group handles exactly one Q4_0 block per tile.
//
// After accumulating across all tiles, each row-group's 32 threads
// tree-reduce their partial sums in shared memory; k_lane==0 writes the row.
//
// Dispatch: ceil(N / ROWS_PER_WG) workgroups in X, B in Y. Requires M==1
// (caller in gguf.rs only dispatches this kernel for decode's M=1 step).
//
// Tint uniformity fix (docs/ENGINE.md "Known issues / fixed"): this kernel
// used to `return` early when `b >= B`, before the workgroupBarrier() calls
// in the tile loop below. `b` (== wg_id.y) is uniform per workgroup, but
// `B` is loaded from a storage buffer and Tint conservatively taints every
// storage-buffer load as non-uniform, so a branch that skips a later
// barrier is rejected. Fix: never branch around a barrier — guard only the
// loads/accumulation/store with `b_valid`/`row_has_output`, keep every
// workgroupBarrier() at the top level of the tile loop so it is always
// reached by the whole workgroup regardless of B or N.

// K5 (docs/BENCHMARKS.md): `weights` holds only the 16-byte nibble portion
// of each Q4_0 block (4 u32/block, always aligned) — `gguf.rs::Q4Tensor`
// strips the interleaved f16 scale into the separate `scales` buffer at
// load time. This removes the unaligned two-word load every other block
// used to require.
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> info: array<u32>;

const WG_SIZE: u32 = 256u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 8u; // WG_SIZE / THREADS_PER_ROW
const TILE_K: u32 = 1024u;   // THREADS_PER_ROW * 32 (Q4_0 block size)

var<workgroup> x_shared: array<f32, 1024>;
var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
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
    let k_lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * ROWS_PER_WG + row_in_wg;
    let row_has_output = n < N && b_valid;

    let input_base = b * K;
    var acc: f32 = 0.0;

    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= K) {
            break;
        }
        let tile_len = min(TILE_K, K - tile_start);

        // Cooperative load: all 256 threads stage this tile of x into
        // shared memory once.
        var i: u32 = tid;
        loop {
            if (i >= tile_len) {
                break;
            }
            if (b_valid) {
                x_shared[i] = input[input_base + tile_start + i];
            } else {
                x_shared[i] = 0.0;
            }
            i = i + WG_SIZE;
        }
        workgroupBarrier();

        let tile_blocks = tile_len / 32u;
        if (row_has_output && k_lane < tile_blocks) {
            let blk = tile_start / 32u + k_lane;
            let global_block = n * blocks_per_row + blk;
            let scale = scales[global_block];
            let nibble_base = global_block * 4u; // 4 aligned u32/block
            let local_k = k_lane * 32u;

            for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
                let packed = weights[nibble_base + wi];
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                let base_i = wi * 4u;

                let w_lo = (vec4<f32>(
                    f32(b0 & 0xFu), f32(b1 & 0xFu),
                    f32(b2 & 0xFu), f32(b3 & 0xFu)
                ) - vec4<f32>(8.0)) * scale;
                let in_lo = vec4<f32>(
                    x_shared[local_k + base_i],
                    x_shared[local_k + base_i + 1u],
                    x_shared[local_k + base_i + 2u],
                    x_shared[local_k + base_i + 3u]
                );
                acc += dot(w_lo, in_lo);

                let w_hi = (vec4<f32>(
                    f32((b0 >> 4u) & 0xFu), f32((b1 >> 4u) & 0xFu),
                    f32((b2 >> 4u) & 0xFu), f32((b3 >> 4u) & 0xFu)
                ) - vec4<f32>(8.0)) * scale;
                let in_hi = vec4<f32>(
                    x_shared[local_k + 16u + base_i],
                    x_shared[local_k + 16u + base_i + 1u],
                    x_shared[local_k + 16u + base_i + 2u],
                    x_shared[local_k + 16u + base_i + 3u]
                );
                acc += dot(w_hi, in_hi);
            }
        }
        workgroupBarrier(); // wait for all reads of this tile before next load overwrites it
        tile_start = tile_start + TILE_K;
    }

    // Tree-reduce within each row's 32-thread group.
    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = THREADS_PER_ROW / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (k_lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (k_lane == 0u && row_has_output) {
        output[b * N + n] = partial_sums[tid];
    }
}
