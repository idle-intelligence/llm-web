// Single-workgroup argmax over a logits vector -> one u32 (the index of the
// max value). New kernel, not ported from t0/llm-wasm (t0 has no
// vocabulary/argmax step). Decode's hot loop only needs this one index per
// step (`model.rs::forward_decode_step`'s `argmax_readback = true` path) -
// running the reduction on the GPU turns a `vocab_size * 4`-byte readback
// (~608KB for this model) into a 4-byte one. Ties broken toward the lower
// index (matches `lean_cli.rs`/`web.rs`'s scalar `argmax()` CPU reference,
// which keeps the first strictly-greater value).
struct Dims {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

const WG_SIZE: u32 = 256u;

var<workgroup> best_val: array<f32, WG_SIZE>;
var<workgroup> best_idx: array<u32, WG_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    var v: f32 = -3.4e38;
    var idx: u32 = 0u;
    var i: u32 = tid;
    loop {
        if (i >= dims.n) {
            break;
        }
        let cand = logits[i];
        if (cand > v) {
            v = cand;
            idx = i;
        }
        i = i + WG_SIZE;
    }
    best_val[tid] = v;
    best_idx[tid] = idx;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            let other_val = best_val[tid + stride];
            let other_idx = best_idx[tid + stride];
            // Tie-break toward the lower index, regardless of reduction order.
            if (other_val > best_val[tid] || (other_val == best_val[tid] && other_idx < best_idx[tid])) {
                best_val[tid] = other_val;
                best_idx[tid] = other_idx;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        out_idx[0] = best_idx[0];
    }
}
