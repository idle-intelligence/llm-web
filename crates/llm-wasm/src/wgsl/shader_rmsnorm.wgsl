// D2 (docs/BENCHMARKS.md Session 3): fused RMSNorm — one dispatch per call
// instead of Burn's unfused cast/square/mean_dim/add/sqrt/div/mul chain
// (~8 ops, per `print_dispatch_estimate` in bin/llm-agent.rs). Matches
// burn-nn 0.20's `RmsNorm::forward` exactly (burn-nn-0.20.1/src/modules/
// norm/rms.rs): `Y = X / sqrt(mean(X^2) + eps) * gamma`. Model runs entirely
// in f32 (no mixed-precision cast needed here — see kv.rs's module doc
// comment on why the whole pipeline stays f32).
//
// One workgroup per row (B*T rows total), 256 threads/workgroup,
// shared-memory tree reduction for the sum-of-squares — same structure as
// shader_q4_matvec.wgsl's row-group reduction.
//
// info: [rows, hidden, eps] packed as f32 (rows/hidden are exact integers
// well within f32's 24-bit mantissa, cast back with u32()).
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let rows = u32(info[0]);
    let hidden = u32(info[1]);
    let eps = info[2];

    let row = wg_id.x;
    if (row >= rows) {
        return;
    }
    let tid = local_id.x;
    let row_base = row * hidden;

    var acc: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= hidden) {
            break;
        }
        let v = input[row_base + i];
        acc += v * v;
        i = i + WG_SIZE;
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let rms = sqrt(partial_sums[0] / f32(hidden) + eps);

    var j: u32 = tid;
    loop {
        if (j >= hidden) {
            break;
        }
        output[row_base + j] = (input[row_base + j] / rms) * weight[j];
        j = j + WG_SIZE;
    }
}
