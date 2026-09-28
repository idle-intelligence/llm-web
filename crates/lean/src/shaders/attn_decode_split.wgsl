// Split-K ("flash-decoding") pass 1 of decode attention: each workgroup
// computes a partial online-softmax result over one contiguous chunk of
// keys `[kv_start, kv_end)` for one head, instead of the whole `[0, kv_len)`
// range in one workgroup (`attn_decode.wgsl`). At long context, that single
// workgroup's O(kv_len) sequential tile loop is the entire decode-step
// latency for its head regardless of how many other heads/workgroups the
// GPU could run concurrently (n_heads workgroups is far below what an M2 GPU
// can have in flight at once). Splitting one head's KV range across
// `num_splits` workgroups lets the GPU actually parallelize that O(kv_len)
// work; `attn_decode_reduce.wgsl` (pass 2) combines the partial results with
// the same online-softmax merge rule.
//
// Split count is chosen from `kv_len` alone (a workload-size fact known at
// dispatch time), never from a timing measurement: `model.rs::attn_decode`
// picks `num_splits = min(MAX_SPLITS, ceil(kv_len / SPLIT_CHUNK))` and
// `chunk = ceil(kv_len / num_splits)`, both fixed formulas independent of
// which device is running. `MAX_SPLITS` bounds the partial-result buffers'
// per-head stride so they never need to regrow as `kv_len` grows one token
// per decode step; must match `model.rs`'s `MAX_SPLITS` exactly.
//
// Why this generalizes beyond one GPU: the model shapes this crate targets
// have few attention heads (14-16), so the un-split kernel only ever
// occupies 14-16 concurrent workgroups regardless of vendor - well under
// the workgroup-in-flight capacity of essentially any WebGPU-class GPU
// (integrated or discrete, Apple/AMD/Intel/NVIDIA), not an M2-specific
// headroom. Splitting trades "one long serial loop per head" for "more,
// shorter, independent loops the driver can schedule across whatever
// concurrency that device actually has" - it can only add workgroups for
// the scheduler to use, never assume a specific occupancy number, SIMD
// width or vendor-specific extension. `workgroup_size(HEAD_DIM)` stays in
// {64, 128}, inside WebGPU's guaranteed 64-256 range on every implementation;
// no subgroup/wave ops, no vendor-specific tile size.
//
// Dispatch: `(n_heads, num_splits, 1)`, `workgroup_size(HEAD_DIM)` (same
// per-thread-owns-one-output-dim design and `HEAD_DIM` override constant as
// `attn_decode.wgsl` - see that file's doc comment).
override HEAD_DIM: u32 = 64u;
const SHARED_CAP: u32 = 256u;
const MAX_SPLITS: u32 = 32u;
var<workgroup> q_shared: array<f32, SHARED_CAP>;
var<workgroup> tile_scores: array<f32, SHARED_CAP>;
var<workgroup> reduce_buf: array<f32, SHARED_CAP>;

struct Dims { n_heads: u32, n_kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32, scale: f32, chunk: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial_m: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_l: array<f32>;
@group(0) @binding(5) var<storage, read_write> partial_acc: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

@compute @workgroup_size(HEAD_DIM)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let h = wg_id.x;
    let split = wg_id.y;
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = h / n_rep;
    let tid = local_id.x;
    let hd = dims.head_dim;

    let kv_start = split * dims.chunk;
    let kv_end = min(kv_start + dims.chunk, dims.kv_len);

    q_shared[tid] = q[h * hd + tid];
    workgroupBarrier();

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: f32 = 0.0; // this thread owns output dimension `tid`

    // Defensive: ceil-division split boundaries should never put kv_start
    // past kv_len for a dispatched split index, but writing a neutral
    // partial here (instead of trusting the arithmetic) makes an off-by-one
    // in the split/chunk formula a no-op in the reduce merge rather than a
    // NaN from an empty softmax.
    if (kv_start >= kv_end) {
        let idx0 = h * MAX_SPLITS + split;
        if (tid == 0u) {
            partial_m[idx0] = -1e30;
            partial_l[idx0] = 0.0;
        }
        partial_acc[idx0 * HEAD_DIM + tid] = 0.0;
        return;
    }

    var tile_start: u32 = kv_start;
    loop {
        if (tile_start >= kv_end) {
            break;
        }
        let key = tile_start + tid;
        var raw: f32 = -1e30;
        if (key < kv_end) {
            var dot: f32 = 0.0;
            let k_base = (kv_head * dims.max_ctx + key) * hd;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + k_cache[k_base + d] * q_shared[d];
            }
            raw = dot * dims.scale;
        }
        tile_scores[tid] = raw;
        workgroupBarrier();

        reduce_buf[tid] = raw;
        workgroupBarrier();
        var stride: u32 = HEAD_DIM / 2u;
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
        let tile_max = reduce_buf[0];
        workgroupBarrier();

        let new_m = max(m, tile_max);
        let factor = exp(m - new_m);
        l = l * factor;
        acc = acc * factor;

        let this_p = select(0.0, exp(raw - new_m), key < kv_end);
        tile_scores[tid] = this_p;
        reduce_buf[tid] = this_p;
        workgroupBarrier();
        stride = HEAD_DIM / 2u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (tid < stride) {
                reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        l = l + reduce_buf[0];
        workgroupBarrier();

        var j: u32 = 0u;
        loop {
            if (j >= HEAD_DIM) {
                break;
            }
            let kk = tile_start + j;
            if (kk >= kv_end) {
                break;
            }
            acc = acc + tile_scores[j] * v_cache[(kv_head * dims.max_ctx + kk) * hd + tid];
            j = j + 1u;
        }

        m = new_m;
        tile_start = tile_start + HEAD_DIM;
        workgroupBarrier();
    }

    let idx = h * MAX_SPLITS + split;
    if (tid == 0u) {
        partial_m[idx] = m;
        partial_l[idx] = l;
    }
    partial_acc[idx * HEAD_DIM + tid] = acc;
}
