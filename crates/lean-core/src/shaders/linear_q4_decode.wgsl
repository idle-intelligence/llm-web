// Q4_0 matvec for decode (M = 1). Started from llm-wasm's
// shader_q4_matvec_coalesced.wgsl (lanes of one output row striding over
// that row's weights, shared-memory tree reduction), with this crate's
// bias and Dims contract and no batch axis.
//
// Each lane now reads a whole Q4_0 block per step: one 16-byte vec4<u32>
// load for its 32 nibbles, one scale load, and the 32 matching x values as
// 8 vec4 loads. 8 lanes share an output row (8 consecutive blocks, 128
// contiguous bytes per step) and a 128-thread workgroup covers 16 rows.
// The earlier layout (32 lanes per row, one u32 word and one scale load
// per lane per step, x read as scalars) issued four times the load
// instructions per byte and, at K = 896, left most lanes idle after one
// step while the 5-step reduction ran.
//
// GATE_UP (pipeline override, `linear_q4_decode_swiglu` in engine.rs): the
// weight is the fused [gate; up] matrix (dims.n = 2 * inter) and a
// workgroup computes gate rows j..j+7 and the matching up rows inter+j..,
// then writes silu(gate) * up for those 8 j into `out` (inter long), the
// same arithmetic as silu_mul_fused.wgsl. This saves decode one dispatch
// and one round trip of the 2 * inter activations per layer.
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

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const LANES: u32 = 8u;
const ROWS_PER_WG: u32 = 16u; // WG_SIZE / LANES

override GATE_UP: bool = false;

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn nib_lo(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
}

fn nib_hi(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w >> 4u, w >> 12u, w >> 20u, w >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let lane = tid % LANES;
    let r = tid / LANES;
    let inter = dims.n / 2u;
    let j = wg_id.x * (ROWS_PER_WG / 2u) + r % (ROWS_PER_WG / 2u);
    var n = wg_id.x * ROWS_PER_WG + r;
    var row_ok = n < dims.n;
    if (GATE_UP) {
        n = select(j, inter + j, r >= ROWS_PER_WG / 2u);
        row_ok = j < inter;
    }
    let bpr = dims.blocks_per_row;

    var acc: f32 = 0.0;
    if (row_ok) {
        let row = n * bpr;
        for (var blk: u32 = lane; blk < bpr; blk = blk + LANES) {
            let q = qs[row + blk];
            let s = scales[row + blk];
            // Word j holds k = 4j..4j+3 (low nibbles) and 16+4j.. (high).
            let xb = blk * 8u;
            let d = dot(nib_lo(q.x), x[xb]) + dot(nib_lo(q.y), x[xb + 1u])
                + dot(nib_lo(q.z), x[xb + 2u]) + dot(nib_lo(q.w), x[xb + 3u])
                + dot(nib_hi(q.x), x[xb + 4u]) + dot(nib_hi(q.y), x[xb + 5u])
                + dot(nib_hi(q.z), x[xb + 6u]) + dot(nib_hi(q.w), x[xb + 7u]);
            acc += d * s;
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (GATE_UP) {
        if (lane == 0u && r < ROWS_PER_WG / 2u && row_ok) {
            let gate = partial_sums[tid] + b[j];
            let up = partial_sums[tid + (ROWS_PER_WG / 2u) * LANES] + b[inter + j];
            out[j] = gate / (1.0 + exp(-gate)) * up;
        }
        return;
    }
    if (lane == 0u && row_ok) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
