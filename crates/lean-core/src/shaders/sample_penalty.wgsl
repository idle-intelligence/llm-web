// Repetition penalty on the GPU: for every vocabulary id that occurs in the
// history (nonzero count), applies the penalty once to its logit, in place,
// whatever the count, as transformers' RepetitionPenaltyLogitsProcessor does.
// Multiplications only (the host passes 1 / penalty), so the result is the
// same on every device. Positive logits shrink, negative ones grow (kept
// finite: a masked logit is -3.4e38).
struct Dims {
    n: u32,
    penalty: f32,
    inv_penalty: f32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    if (counts[i] == 0u) {
        return;
    }
    let v = logits[i];
    if (v > 0.0) {
        logits[i] = v * dims.inv_penalty;
    } else {
        logits[i] = max(v * dims.penalty, -3.4e38);
    }
}
