// Adapted from llm-wasm/src/wgsl/shader_q4_matvec_subgroup.wgsl (Q4_0
// Cooperative Matvec, M=1/decode, subgroupAdd() reduction — see that file's
// header for the tiled-shared-memory-x + subgroup-reduction rationale).
// Requires `wgpu::Features::SUBGROUP`; only compiled/dispatched when
// `Engine::has_subgroups` is true (probed at device-init time from the
// adapter's supported features — see engine.rs). Falls back to
// linear_q4_decode.wgsl (the coalesced, no-subgroup kernel) otherwise, same
// pattern llm-wasm documents but never wired up on its own default path.
// Differences from the llm-wasm original: bias folded in, `info` replaced
// by a `Dims` uniform struct, no batch axis (decode is always one token).
enable subgroups;

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

const WG_SIZE: u32 = 256u;
const SUBGROUP_SIZE: u32 = 32u;
const ROWS_PER_WG: u32 = 8u; // WG_SIZE / SUBGROUP_SIZE
const TILE_K: u32 = 1024u;   // SUBGROUP_SIZE * 32 (Q4_0 block size)

var<workgroup> x_shared: array<f32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let K = dims.k;
    let N = dims.n;
    let blocks_per_row = dims.blocks_per_row;

    let tid = local_id.x;
    let sg_idx = tid / SUBGROUP_SIZE;
    let n = wg_id.x * ROWS_PER_WG + sg_idx;
    let row_has_output = n < N;

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
            x_shared[i] = x[tile_start + i];
            i = i + WG_SIZE;
        }
        workgroupBarrier();

        let tile_blocks = tile_len / 32u;
        if (row_has_output && sg_id < tile_blocks) {
            let blk = tile_start / 32u + sg_id;
            let global_block = n * blocks_per_row + blk;
            let scale = scales[global_block];
            let nibble_base = global_block * 4u;
            let local_k = sg_id * 32u;

            for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
                let packed = qs[nibble_base + wi];
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
        var v = row_sum + b[n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[n] = v;
    }
}
