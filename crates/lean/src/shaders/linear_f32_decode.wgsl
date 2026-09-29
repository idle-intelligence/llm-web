// Decode-shaped matvec for MatMulWeight::F32 weights (rows == 1), same
// 4-rows/workgroup, 32-lanes/row, shared-memory tree-reduction structure as
// linear_q4_decode.wgsl/linear_q8_decode.wgsl, adapted to a plain f32
// weight row (no dequant - just a coalesced, cooperative dot product).
// Some GGUFs keep a handful of tensors at F32 residency even in an
// otherwise-quantized file (SmolLM2-360M-Instruct-Q4_0.gguf ships 4 of 32
// ffn_down tensors this way, llama.cpp's own mixed-precision heuristic) -
// those previously fell through to linear.wgsl's naive one-thread-per-
// output-element kernel at decode, ~20x slower per call than the Q4_0 path
// for a smaller matrix (docs/runs/2026-09-29-lean-kernels.md, Observations).
// Same bindings/Dims contract as linear.wgsl. Assumes dims.k is a multiple
// of 4 (every hidden/intermediate dim in this crate's supported models is).
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const THREADS_PER_ROW: u32 = 32u;
const ROWS_PER_WG: u32 = 4u; // WG_SIZE / THREADS_PER_ROW

var<workgroup> partial_sums: array<f32, WG_SIZE>;

fn accumulate_word(word_idx: u32, weights_row_base: u32) -> f32 {
    let k0 = word_idx * 4u;
    let wv = vec4<f32>(w[weights_row_base + k0], w[weights_row_base + k0 + 1u], w[weights_row_base + k0 + 2u], w[weights_row_base + k0 + 3u]);
    let xv = vec4<f32>(x[k0], x[k0 + 1u], x[k0 + 2u], x[k0 + 3u]);
    return dot(wv, xv);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let N = dims.n;
    let total_words = dims.k / 4u;

    let tid = local_id.x;
    let row_in_wg = tid / THREADS_PER_ROW;
    let lane = tid % THREADS_PER_ROW;
    let n = wg_id.x * ROWS_PER_WG + row_in_wg;
    let row_has_output = n < N;

    let weights_row_base = n * dims.k;

    var acc: f32 = 0.0;
    if (row_has_output) {
        var w0: u32 = lane;
        loop {
            if (w0 >= total_words) {
                break;
            }
            acc += accumulate_word(w0, weights_row_base);
            let w1 = w0 + 32u;
            if (w1 < total_words) {
                acc += accumulate_word(w1, weights_row_base);
            }
            let w2 = w0 + 64u;
            if (w2 < total_words) {
                acc += accumulate_word(w2, weights_row_base);
            }
            let w3 = w0 + 96u;
            if (w3 < total_words) {
                acc += accumulate_word(w3, weights_row_base);
            }
            w0 = w0 + 128u;
        }
    }

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
        var v = partial_sums[tid] + b[n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[n] = v;
    }
}
