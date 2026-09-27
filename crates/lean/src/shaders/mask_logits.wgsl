// New kernel: applies an allowed-token bitset to one row of a logits
// buffer, in place, before argmax/readback - the GPU side of constrained
// decoding (model.rs's `mask_logits_gpu`). `mask_bits` is a packed bitset,
// 32 vocab ids per u32 (bit i%32 of word i/32 - see
// `model::build_mask_bitset`); a disallowed id's logit is set to -3.4e38
// (the same "effectively -inf" sentinel `argmax.wgsl` and softmax already
// tolerate) so it can never win argmax or dominate a softmax-based sampler.
// `dims.offset` lets one call target a single row inside a larger
// `[rows, vocab]` buffer (prefill masks only the last row; decode's buffer
// is already just one row, offset 0).
struct Dims {
    n: u32,
    offset: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> mask_bits: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let word = mask_bits[i / 32u];
    let bit = (word >> (i % 32u)) & 1u;
    if (bit == 0u) {
        logits[dims.offset + i] = -3.4e38;
    }
}
