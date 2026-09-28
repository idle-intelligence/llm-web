// Single-token GQA decode attention over an F32 KV cache, length-independent.
//
// Previous version stored every key's raw score in a `scratch: array<f32,
// 2048>` workgroup array, capping `kv_len` at 2048 (silently wrong past
// that, same failure class as attn_prefill.wgsl's old MAX_SEQ=256: an
// out-of-bounds `scratch[key]` write for any key >= 2048). The Sonos MCP
// agent's real prompts run 1,000-2,000 tokens before decode even starts, so
// this was reachable.
//
// Fix: flash-attention-style tiled online softmax. Keys are processed in
// tiles of `WG` (64) at a time; each tile's raw scores live in a
// `WG`-sized shared array (reused every tile, not one slot per key), so
// shared-memory use no longer depends on `kv_len` at all. Per-thread state
// across tiles is just a scalar running max/sum (`m`/`l`) plus one scalar
// accumulator (`acc`, since thread `tid` owns output dimension `tid` and
// `WG == head_dim`) — same running-softmax update flash attention uses,
// applied one `WG`-wide tile of keys at a time instead of one key at a
// time (attn_prefill.wgsl's per-thread version, which has no shared memory
// to batch into, updates one key at a time instead).
//
// KV cache layout (load-bearing, documented here since this is the
// kernel that defines it): [n_kv_heads, max_ctx, head_dim], head-major,
// contiguous per head, written once per decode step at `[kv_head, kv_len,
// :]` — see model.rs's `KvCache`. No causal mask needed: decode's single
// query is always the newest position, so it attends to every key in
// [0, kv_len).
const WG: u32 = 64u; // head_dim for Qwen2.5-0.5B; must equal dims.head_dim at dispatch.
var<workgroup> q_shared: array<f32, 64>;
var<workgroup> tile_scores: array<f32, 64>; // this tile's WG raw/exp'd scores, reused per tile
var<workgroup> reduce_buf: array<f32, 64>;

struct Dims { n_heads: u32, n_kv_heads: u32, head_dim: u32, kv_len: u32, max_ctx: u32, scale: f32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let h = wg_id.x;
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = h / n_rep;
    let tid = local_id.x;
    let hd = dims.head_dim;

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

        // Tile max (tree reduction into reduce_buf).
        reduce_buf[tid] = raw;
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
        stride = WG / 2u;
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
        // `tid`) walks all WG keys in the tile.
        var j: u32 = 0u;
        loop {
            if (j >= WG) {
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
        tile_start = tile_start + WG;
        workgroupBarrier();
    }

    out[h * hd + tid] = acc / l;
}
