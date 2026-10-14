// Takes the token ids of the repetition-penalty history out of a logits
// vector before the top-k: their raw values are copied to `res` (the host
// applies the penalty and merges them with the top-k of the rest) and the
// logits are overwritten with the lowest f32 so they cannot be selected.
// `ids` holds each distinct id once, so no two threads touch one logit.
struct Dims {
    n: u32,
    res_base: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> res: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let id = ids[i];
    res[dims.res_base + i] = bitcast<u32>(logits[id]);
    logits[id] = bitcast<f32>(0xff7fffffu);
}
