// Split-K decode attention pass 2: combines `num_splits` partial
// online-softmax results from `attn_decode_split.wgsl` into the final
// per-head output, using the same running-max/sum merge rule flash
// attention uses to combine any two partial softmax states. One workgroup
// per head, `workgroup_size(HEAD_DIM)` (thread `tid` owns output dim `tid`,
// matching `attn_decode_split.wgsl`'s partial-accumulator layout).
//
// `MAX_SPLITS` is the partial buffers' fixed per-head stride (must match
// `attn_decode_split.wgsl` and `model.rs::MAX_SPLITS` exactly); `num_splits`
// (how many of those `MAX_SPLITS` slots this decode step actually wrote) is
// a per-dispatch uniform, since `kv_len` - and so the split count - grows
// every decode step.
override HEAD_DIM: u32 = 64u;
const MAX_SPLITS: u32 = 32u;

struct Dims { n_heads: u32, head_dim: u32, num_splits: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> partial_m: array<f32>;
@group(0) @binding(1) var<storage, read> partial_l: array<f32>;
@group(0) @binding(2) var<storage, read> partial_acc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(HEAD_DIM)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let h = wg_id.x;
    let tid = local_id.x;

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: f32 = 0.0;

    var s: u32 = 0u;
    loop {
        if (s >= dims.num_splits) {
            break;
        }
        let idx = h * MAX_SPLITS + s;
        let pm = partial_m[idx];
        let pl = partial_l[idx];
        let new_m = max(m, pm);
        let f_old = exp(m - new_m);
        let f_new = exp(pm - new_m);
        l = l * f_old + pl * f_new;
        acc = acc * f_old + partial_acc[idx * HEAD_DIM + tid] * f_new;
        m = new_m;
        s = s + 1u;
    }

    out[h * dims.head_dim + tid] = acc / l;
}
