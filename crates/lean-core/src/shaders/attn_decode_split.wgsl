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
const MAX_SPLITS: u32 = 32u;
// Each key's dot product is shared by LANES adjacent threads, each reading a
// contiguous slice of the key row as vec4 loads (adjacent threads read
// adjacent memory), instead of one thread walking a whole row with a
// HEAD_DIM-float stride between neighbours. The LANES partial sums go through
// workgroup memory and the key's owner thread adds them.
const LANES: u32 = 8u;
// Workgroup arrays are sized for the largest head_dim (128).
var<workgroup> q_shared: array<vec4<f32>, 32>;
var<workgroup> partial: array<f32, 1024>;
var<workgroup> tile_scores: array<f32, 128>;

struct Dims { n_heads: u32, n_kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32, scale: f32, chunk: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<vec4<f32>>;
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
    let lane = tid % LANES;
    let key_in_pass = tid / LANES;
    let slice4 = HEAD_DIM / (LANES * 4u); // vec4 loads per lane per key
    let kpp = HEAD_DIM / LANES;           // keys covered per pass

    let kv_start = split * dims.chunk;
    let kv_end = min(kv_start + dims.chunk, dims.kv_len);

    if (tid < HEAD_DIM / 4u) {
        let b = h * hd + tid * 4u;
        q_shared[tid] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]);
    }
    workgroupBarrier();

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: f32 = 0.0; // this thread owns output dimension `tid`

    // An empty chunk writes a neutral partial so the reduce merge ignores it.
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
        // Scores: pass p covers keys tile_start + p*kpp + key_in_pass.
        for (var p: u32 = 0u; p < LANES; p = p + 1u) {
            let key = tile_start + p * kpp + key_in_pass;
            var sum: f32 = 0.0;
            if (key < kv_end) {
                let base4 = ((kv_head * dims.max_ctx + key) * hd) / 4u + lane * slice4;
                for (var i: u32 = 0u; i < slice4; i = i + 1u) {
                    sum = sum + dot(k_cache[base4 + i], q_shared[lane * slice4 + i]);
                }
            }
            partial[p * HEAD_DIM + tid] = sum;
        }
        workgroupBarrier();

        // Thread `tid` owns key tile_start + tid: partials of key j live at
        // pass j / kpp, threads (j % kpp) * LANES .. + LANES.
        let pass_of = tid / kpp;
        let pbase = pass_of * HEAD_DIM + (tid % kpp) * LANES;
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < LANES; i = i + 1u) {
            s = s + partial[pbase + i];
        }
        let key_o = tile_start + tid;
        tile_scores[tid] = select(-1e30, s * dims.scale, key_o < kv_end);
        workgroupBarrier();

        // Tile max: every thread reads the same shared values (broadcast).
        var tile_max: f32 = -1e30;
        for (var j: u32 = 0u; j < HEAD_DIM; j = j + 1u) {
            tile_max = max(tile_max, tile_scores[j]);
        }
        let new_m = max(m, tile_max);
        let factor = exp(m - new_m);
        l = l * factor;
        acc = acc * factor;
        workgroupBarrier();
        tile_scores[tid] = select(0.0, exp(tile_scores[tid] - new_m), key_o < kv_end);
        workgroupBarrier();

        let n_valid = min(HEAD_DIM, kv_end - tile_start);
        let v_base = (kv_head * dims.max_ctx + tile_start) * hd + tid;
        var j: u32 = 0u;
        for (; j + 4u <= n_valid; j = j + 4u) {
            let p0 = tile_scores[j];
            let p1 = tile_scores[j + 1u];
            let p2 = tile_scores[j + 2u];
            let p3 = tile_scores[j + 3u];
            l = l + (p0 + p1 + p2 + p3);
            acc = acc + p0 * v_cache[v_base + j * hd] + p1 * v_cache[v_base + (j + 1u) * hd] + p2 * v_cache[v_base + (j + 2u) * hd] + p3 * v_cache[v_base + (j + 3u) * hd];
        }
        for (; j < n_valid; j = j + 1u) {
            let pj = tile_scores[j];
            l = l + pj;
            acc = acc + pj * v_cache[v_base + j * hd];
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
