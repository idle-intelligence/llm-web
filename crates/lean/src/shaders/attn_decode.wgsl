// New kernel: single-token GQA decode attention over an F32 KV cache.
// Structurally modeled on llm-wasm's shader_attn_decode_q8.wgsl (same
// three-phase workgroup-reduction shape: A. raw scores + running max, B.
// exp(score-max) + running sum, C. weighted V sum, one workgroup per query
// head, WG size = head_dim) but reading K/V directly as F32 (no per-key
// dequant) — quantized-KV-cache decode is a later slice, not this one.
//
// KV cache layout (load-bearing, documented here since this is the
// kernel that defines it): [n_kv_heads, max_ctx, head_dim], head-major,
// contiguous per head, written once per decode step at `[kv_head, kv_len,
// :]` — see model.rs's `KvCache`. No causal mask needed: decode's single
// query is always the newest position, so it attends to every key in
// [0, kv_len).
const WG: u32 = 64u; // head_dim for Qwen2.5-0.5B; must equal dims.head_dim at dispatch.
var<workgroup> q_shared: array<f32, 64>;
var<workgroup> reduce_buf: array<f32, 64>;
// 2048 f32 = 8KB; plus q_shared/reduce_buf (256B each) stays under the
// browser's 16KB minimum-guaranteed workgroup memory — native-only in this
// slice, but sized with wasm headroom in mind already. Bump alongside
// max_ctx in model.rs if a longer fixed context is needed.
var<workgroup> scratch: array<f32, 2048>;

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

    var local_max: f32 = -1e30;
    var key: u32 = tid;
    loop {
        if (key >= dims.kv_len) {
            break;
        }
        var dot: f32 = 0.0;
        let k_base = (kv_head * dims.max_ctx + key) * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dot = dot + k_cache[k_base + d] * q_shared[d];
        }
        let raw = dot * dims.scale;
        scratch[key] = raw;
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

    var local_sum: f32 = 0.0;
    key = tid;
    loop {
        if (key >= dims.kv_len) {
            break;
        }
        let p = exp(scratch[key] - row_max);
        scratch[key] = p;
        local_sum = local_sum + p;
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
            reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = reduce_buf[0];
    workgroupBarrier();

    let d = tid;
    var acc: f32 = 0.0;
    var k2: u32 = 0u;
    loop {
        if (k2 >= dims.kv_len) {
            break;
        }
        let p = scratch[k2] / row_sum;
        acc = acc + p * v_cache[(kv_head * dims.max_ctx + k2) * hd + d];
        k2 = k2 + 1u;
    }
    out[h * hd + d] = acc;
}
