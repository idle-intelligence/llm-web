// Single-token GQA decode attention over an F32 KV cache, length-independent
// in kv_len and generic in head_dim via a pipeline-overridable constant.
//
// Previous version stored every key's raw score in a `scratch: array<f32,
// 2048>` workgroup array, capping `kv_len` at 2048 (silently wrong past
// that, same failure class as attn_prefill.wgsl's old MAX_SEQ=256: an
// out-of-bounds `scratch[key]` write for any key >= 2048). The Sonos MCP
// agent's real prompts run 1,000-2,000 tokens before decode even starts, so
// this was reachable.
//
// Fix: flash-attention-style tiled online softmax. Keys are processed in
// tiles of `HEAD_DIM` at a time; each tile's raw scores live in a
// `HEAD_DIM`-sized shared array (reused every tile, not one slot per key), so
// shared-memory use no longer depends on `kv_len` at all. Per-thread state
// across tiles is just a scalar running max/sum (`m`/`l`) plus one scalar
// accumulator (`acc`, since thread `tid` owns output dimension `tid` and
// the tile width equals `HEAD_DIM`): same running-softmax update flash
// attention uses, applied one `HEAD_DIM`-wide tile of keys at a time instead
// of one key at a time (attn_prefill.wgsl's per-thread version, which has no
// shared memory to batch into, updates one key at a time instead).
//
// `HEAD_DIM` is a pipeline-overridable constant (WGSL `override`), not a
// compile-time literal: engine.rs creates one pipeline per model head_dim
// (64 for Qwen2.5, 128 for Qwen3) from this single shader source, setting
// `HEAD_DIM` via `PipelineCompilationOptions::constants` at pipeline
// creation, and `@workgroup_size(HEAD_DIM)` picks up the same override. The
// per-thread-owns-one-output-dim design requires workgroup width to equal
// head_dim exactly (one thread per dim, one tile-of-keys per workgroup pass
// wide), which is why this can't just be a runtime uniform.
//
// Shared arrays are still sized at compile time (WGSL array lengths must be
// const, not override, expressions) at `SHARED_CAP`, a fixed upper bound
// independent of which model is loaded; only the *active* `0..HEAD_DIM`
// prefix of each is ever touched. `SHARED_CAP == 256` matches WebGPU's
// browser-mandated max workgroup invocation count, so this kernel already
// covers every head_dim a workgroup could dispatch with one thread per dim.
// 3 arrays * 256 * 4 bytes = 3KB, far under WebGPU's guaranteed minimum
// 16KB workgroup storage limit - no adapter-limit check needed.
override HEAD_DIM: u32 = 64u;
const SHARED_CAP: u32 = 256u;
var<workgroup> q_shared: array<f32, SHARED_CAP>;
var<workgroup> tile_scores: array<f32, SHARED_CAP>; // this tile's raw/exp'd scores, reused per tile
var<workgroup> reduce_buf: array<f32, SHARED_CAP>;

// KV cache layout (load-bearing, documented here since this is the
// kernel that defines it): [n_kv_heads, max_ctx, head_dim], head-major,
// contiguous per head, written once per decode step at `[kv_head, kv_len,
// :]`: see model.rs's `KvCache`. No causal mask needed: decode's single
// query is always the newest position, so it attends to every key in
// [0, kv_len).
struct Dims { n_heads: u32, n_kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32, scale: f32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(HEAD_DIM)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let h = wg_id.x;
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = h / n_rep;
    let tid = local_id.x;
    let hd = dims.head_dim; // must equal HEAD_DIM at dispatch time - the override picks the pipeline, this is the runtime cross-check value used for indexing

    q_shared[tid] = q[h * hd + tid];
    workgroupBarrier();

    var m: f32 = -1e30; // effectively -infinity for this kernel's score range; see attn_prefill.wgsl's comment on why not -f32::MAX's literal
    var l: f32 = 0.0;
    var acc: f32 = 0.0; // this thread owns output dimension `tid`

    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= dims.kv_len) {
            break;
        }
        let key = tile_start + tid;
        var raw: f32 = -1e30;
        if (key < dims.kv_len) {
            var dot: f32 = 0.0;
            let k_base = (kv_head * dims.max_ctx + key) * hd;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + k_cache[k_base + d] * q_shared[d];
            }
            raw = dot * dims.scale;
        }
        tile_scores[tid] = raw;
        workgroupBarrier();

        // Tile max (tree reduction into reduce_buf). HEAD_DIM is a power of
        // two (64/128/...), so halving the stride to 0 covers the whole tile.
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

        // exp(score - new_m) per key in this tile, and this tile's sum.
        let this_p = select(0.0, exp(raw - new_m), key < dims.kv_len);
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

        // Weighted V accumulation for this tile: every thread (owns dim
        // `tid`) walks all HEAD_DIM keys in the tile.
        var j: u32 = 0u;
        loop {
            if (j >= HEAD_DIM) {
                break;
            }
            let kk = tile_start + j;
            if (kk >= dims.kv_len) {
                break;
            }
            acc = acc + tile_scores[j] * v_cache[(kv_head * dims.max_ctx + kk) * hd + tid];
            j = j + 1u;
        }

        m = new_m;
        tile_start = tile_start + HEAD_DIM;
        workgroupBarrier();
    }

    out[h * hd + tid] = acc / l;
}
