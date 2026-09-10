// Q4_0 Cooperative Matvec — M=1 (decode) — subgroupAdd() reduction variant.
//
// Same tiled-shared-memory staging of x as shader_q4_matvec.wgsl, but each
// row's THREADS_PER_ROW=32 threads are a single hardware subgroup (32 lanes
// on Apple Silicon/Metal) and the final per-row reduction uses subgroupAdd()
// instead of a shared-memory tree reduction. Requires `Features::SUBGROUP`
// and WGSL's `enable subgroups;` — gated behind gguf.rs's
// `has_subgroup_support()` runtime check (mirrors sts-web's
// shader_q4k_matvec_subgroup.wgsl / gguf.rs:33-46 pattern). Nothing in this
// crate currently calls `set_subgroup_support(true)` (that requires probing
// `wgpu::Features::SUBGROUP` at device-init time in web.rs, outside this
// crate's owned files) — this kernel is therefore dead code on the default
// path until that wiring exists, same as sts-web's native default.
//
// Dispatch: identical grid to shader_q4_matvec.wgsl — ceil(N / ROWS_PER_WG)
// workgroups in X, B in Y, ROWS_PER_WG=8 (WG_SIZE=256 / SUBGROUP_SIZE=32).
//
// Tint uniformity fix (docs/ENGINE.md "Known issues / fixed"): same bug and
// same fix as shader_q4_matvec.wgsl — an early `return` on `b >= B` (b ==
// wg_id.y, uniform per workgroup, but B is a storage-buffer load Tint
// taints as non-uniform) used to precede both workgroupBarrier() and
// subgroupAdd() calls below, both of which require uniform control flow.
// Guard loads/accumulation/store with `b_valid`/`row_has_output` instead;
// keep every barrier and subgroupAdd() at the top level of `main`. The
// tile loop's `tile_start >= K` break also gates a barrier, so `K` (and
// every other control value read from `info`) is staged through
// `var<workgroup>` + `workgroupUniformLoad` instead of read directly —
// see shader_q4_matvec.wgsl's header comment for why.

enable subgroups;

// K5 (docs/BENCHMARKS.md): see shader_q4_matvec.wgsl's binding comment —
// `weights` is nibbles-only (4 aligned u32/block), scale lives in `scales`.
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> info: array<u32>;

const WG_SIZE: u32 = 256u;
const SUBGROUP_SIZE: u32 = 32u;
const ROWS_PER_WG: u32 = 8u; // WG_SIZE / SUBGROUP_SIZE
const TILE_K: u32 = 1024u;   // SUBGROUP_SIZE * 32 (Q4_0 block size)

var<workgroup> x_shared: array<f32, 1024>;
var<workgroup> wg_info: array<u32, 5>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    if (local_id.x == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
        wg_info[3] = info[3];
        wg_info[4] = info[4];
    }
    let loaded_info = workgroupUniformLoad(&wg_info);
    let B = loaded_info[0];
    let K = loaded_info[2];
    let N = loaded_info[3];
    let blocks_per_row = loaded_info[4];

    let tid = local_id.x;
    let b = wg_id.y;
    let b_valid = b < B;

    let sg_idx = tid / SUBGROUP_SIZE;
    let n = wg_id.x * ROWS_PER_WG + sg_idx;
    let row_has_output = n < N && b_valid;

    let input_base = b * K;
    var acc: f32 = 0.0;

    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= K) {
            break;
        }
        let tile_len = min(TILE_K, K - tile_start);

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
        if (row_has_output && sg_id < tile_blocks) {
            let blk = tile_start / 32u + sg_id;
            let global_block = n * blocks_per_row + blk;
            let scale = scales[global_block];
            let nibble_base = global_block * 4u; // 4 aligned u32/block
            let local_k = sg_id * 32u;

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
        workgroupBarrier();
        tile_start = tile_start + TILE_K;
    }

    let row_sum = subgroupAdd(acc);
    if (sg_id == 0u && row_has_output) {
        output[b * N + n] = row_sum;
    }
}
