// Decode-time (M=1) fused attention over an F32 KV cache: QK^T -> stable
// softmax -> PV, entirely in one kernel per layer, reading K/V directly out
// of the cache's f32 tensors (no dequant at all — this is the F32-cache
// counterpart of `shader_attn_decode_q8.wgsl`; see that file's header for
// the shared algorithm/GQA/dispatch rationale, reproduced here only where
// the layout differs). Accumulates in f32 throughout. One workgroup per
// query head (`wg_id.x = h`, `workgroup_size = head_dim` — this model's
// head_dim is 128); GQA maps query head `h` to kv head `h / (n_heads /
// n_kv_heads)`, matching `model.rs`'s `repeat_kv` head-repetition order.
//
// No causal mask needed: decode's single query is always the newest
// position (kv_len - 1), so it may attend to every key in [0, kv_len).
//
// K/V layout: `kv.rs`'s `KvDtype::F32` cache is one contiguous Burn tensor
// per layer, `[1, n_kv_heads, max_ctx, head_dim]` — squeezing the batch
// axis, that's `k[kv_head, pos, d]` at flat offset
// `kv_head * max_ctx * head_dim + pos * head_dim + d`. No block/word
// unpacking needed; each thread `tid` (= output dim `d` in phase C, or a
// strided key index in phases A/B) reads `head_dim`-contiguous rows
// directly.
//
// Three phases per workgroup (identical structure to the q8_0 kernel, see
// its header for the full uniformity/barrier rationale):
//   A. each thread computes raw (pre-softmax) scores for its strided keys
//      (thread `tid` owns keys `tid, tid+128, ...`), writing them to a
//      per-head scratch row and tracking a running max.
//   B. workgroup-reduce the max, then each thread re-visits its keys,
//      overwrites its scratch entries with `exp(score - max)`, and tracks a
//      running sum.
//   C. workgroup-reduce the sum; then thread `tid` owns *output* dimension
//      `d = tid` and loops over every key accumulating
//      `sum_key(p[key]/sum * V[key][d])`.
//
// Dispatch: `CubeCount::new_1d(n_heads)`, `CubeDim::new_1d(head_dim)`.
// `scratch` must be >= n_heads * max_ctx f32 (reused across layers/steps —
// same scratch-handle cache pattern as the q8_0 path, see `model.rs`).

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read_write> info: array<u32>;

const WG: u32 = 128u;
var<workgroup> q_shared: array<f32, 128>;
var<workgroup> reduce_buf: array<f32, 128>;
var<workgroup> wg_info: array<u32, 6>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (local_id.x == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
        wg_info[3] = info[3];
        wg_info[4] = info[4];
        wg_info[5] = info[5];
    }
    let li = workgroupUniformLoad(&wg_info);
    let n_heads = li[0];
    let n_kv_heads = li[1];
    let head_dim = li[2];
    let kv_len = li[3];
    let max_ctx = li[4];
    let attn_scale = bitcast<f32>(li[5]);

    let h = wg_id.x;
    let n_rep = n_heads / n_kv_heads;
    let kv_head = h / n_rep;
    let tid = local_id.x;
    let kv_head_stride = max_ctx * head_dim;
    let kv_head_base = kv_head * kv_head_stride;

    q_shared[tid] = q[h * head_dim + tid];
    workgroupBarrier();

    // Phase A: raw scores for this thread's strided keys, tracking a
    // running max.
    var local_max: f32 = -3.4028235e38;
    var key: u32 = tid;
    loop {
        if (key >= kv_len) {
            break;
        }
        var dot: f32 = 0.0;
        let row_base = kv_head_base + key * head_dim;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            dot += k_cache[row_base + d] * q_shared[d];
        }
        let raw = dot * attn_scale;
        scratch[h * max_ctx + key] = raw;
        local_max = max(local_max, raw);
        key = key + WG;
    }
    reduce_buf[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = reduce_buf[0];
    workgroupBarrier();

    // Phase B: exp(score - max) in place, tracking a running sum.
    var local_sum: f32 = 0.0;
    key = tid;
    loop {
        if (key >= kv_len) {
            break;
        }
        let p = exp(scratch[h * max_ctx + key] - row_max);
        scratch[h * max_ctx + key] = p;
        local_sum += p;
        key = key + WG;
    }
    reduce_buf[tid] = local_sum;
    workgroupBarrier();
    stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = reduce_buf[0];
    workgroupBarrier();

    // Phase C: thread `tid` owns output dim d=tid, loops over every key.
    let d = tid;
    var acc: f32 = 0.0;
    var k2: u32 = 0u;
    loop {
        if (k2 >= kv_len) {
            break;
        }
        let p = scratch[h * max_ctx + k2] / row_sum;
        acc += p * v_cache[kv_head_base + k2 * head_dim + d];
        k2 = k2 + 1u;
    }
    output[h * head_dim + d] = acc;
}
