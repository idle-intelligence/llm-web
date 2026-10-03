// Q4_0 matvec for decode (M = 1). Started from llm-wasm's
// shader_q4_matvec_coalesced.wgsl (lanes of one output row striding over
// that row's weights, shared-memory tree reduction), with this crate's
// bias and Dims contract and no batch axis.
//
// Each lane reads a whole Q4_0 block per step and row: one 16-byte
// vec4<u32> load for its 32 nibbles and one scale load. 8 lanes share a
// group of 4 output rows (8 consecutive blocks of each row, 128 contiguous
// bytes per row and step). A lane loads the block's 32 x values (8 vec4
// loads) once and dots them against the 4 rows' blocks, so it makes 2 x
// loads per weight word instead of 8 when one row per lane: per byte of
// weights it moves 2x the x bytes, not 8x. 64 threads = 8 row groups = 32
// rows per workgroup.
//
// GATE_UP (pipeline override, `linear_q4_decode_swiglu` in engine.rs): the
// weight is the fused [gate; up] matrix (dims.n = 2 * inter). A row group
// takes gate rows j, j+1 and up rows inter+j, inter+j+1 and writes
// silu(gate) * up for those 2 j into `out` (inter long), the same
// arithmetic as silu_mul_fused.wgsl: 16 j per workgroup. This saves decode
// one dispatch and one round trip of the 2 * inter activations per layer.
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

const WG_SIZE: u32 = 64u;
const LANES: u32 = 8u;
const GROUPS: u32 = 8u; // WG_SIZE / LANES, 4 rows each

override GATE_UP: bool = false;

var<workgroup> partial_sums: array<vec4<f32>, WG_SIZE>;

fn nib_lo(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
}

fn nib_hi(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w >> 4u, w >> 12u, w >> 20u, w >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
}

// Word j of a block holds k = 4j..4j+3 (low nibbles) and 16+4j.. (high).
fn block_dot(q: vec4<u32>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return dot(nib_lo(q.x), x0) + dot(nib_lo(q.y), x1) + dot(nib_lo(q.z), x2) + dot(nib_lo(q.w), x3)
        + dot(nib_hi(q.x), x4) + dot(nib_hi(q.y), x5) + dot(nib_hi(q.z), x6) + dot(nib_hi(q.w), x7);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let lane = tid % LANES;
    let g = tid / LANES;
    let inter = dims.n / 2u;
    // The group's 4 rows, clamped into range (out-of-range rows compute a
    // duplicate and are never written).
    var n0 = (wg_id.x * GROUPS + g) * 4u;
    var r0 = min(n0, dims.n - 1u);
    var r1 = min(n0 + 1u, dims.n - 1u);
    var r2 = min(n0 + 2u, dims.n - 1u);
    var r3 = min(n0 + 3u, dims.n - 1u);
    let j = (wg_id.x * GROUPS + g) * 2u;
    if (GATE_UP) {
        r0 = min(j, inter - 1u);
        r1 = min(j + 1u, inter - 1u);
        r2 = inter + r0;
        r3 = inter + r1;
    }
    let bpr = dims.blocks_per_row;
    r0 = r0 * bpr;
    r1 = r1 * bpr;
    r2 = r2 * bpr;
    r3 = r3 * bpr;

    var acc = vec4<f32>(0.0);
    for (var blk: u32 = lane; blk < bpr; blk = blk + LANES) {
        let xb = blk * 8u;
        let x0 = x[xb];
        let x1 = x[xb + 1u];
        let x2 = x[xb + 2u];
        let x3 = x[xb + 3u];
        let x4 = x[xb + 4u];
        let x5 = x[xb + 5u];
        let x6 = x[xb + 6u];
        let x7 = x[xb + 7u];
        let s = vec4<f32>(scales[r0 + blk], scales[r1 + blk], scales[r2 + blk], scales[r3 + blk]);
        let d = vec4<f32>(
            block_dot(qs[r0 + blk], x0, x1, x2, x3, x4, x5, x6, x7),
            block_dot(qs[r1 + blk], x0, x1, x2, x3, x4, x5, x6, x7),
            block_dot(qs[r2 + blk], x0, x1, x2, x3, x4, x5, x6, x7),
            block_dot(qs[r3 + blk], x0, x1, x2, x3, x4, x5, x6, x7),
        );
        acc += d * s;
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }
    if (lane != 0u) {
        return;
    }
    let v = partial_sums[tid];

    if (GATE_UP) {
        if (j < inter) {
            let gate = v.x + b[j];
            let up = v.z + b[inter + j];
            out[j] = gate / (1.0 + exp(-gate)) * up;
        }
        if (j + 1u < inter) {
            let gate = v.y + b[j + 1u];
            let up = v.w + b[inter + j + 1u];
            out[j + 1u] = gate / (1.0 + exp(-gate)) * up;
        }
        return;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let n = n0 + i;
        if (n < dims.n) {
            var o = v[i] + b[dims.n_offset + n];
            if (dims.act == 1u) {
                o = max(o, 0.0);
            }
            out[dims.n_offset + n] = o;
        }
    }
}
