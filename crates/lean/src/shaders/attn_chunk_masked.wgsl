// Pluggable-attention-mask GQA over a resident-prefix KV cache: `t` query
// rows (this call's chunk, freshly computed q, not yet in the cache)
// attend over `kv_total = prefix_len + t` keys already scattered into the
// cache (the caller scatters this chunk's own k/v into the cache *before*
// dispatching this kernel, at cache positions `[prefix_len, prefix_len+t)`
// - see model.rs's `forward_chunk_spec`), gated by an explicit per-(query,
// key) bitset instead of an implicit causal rule.
//
// This is the generic mechanism behind llm-life's block-diagonal packed
// mask (variant A: each cell's block attends the shared prefix plus only
// its own block, causally within the block) and its sparse 9-key stencil
// (variant B) - the bitset is caller-built (`model::pack_bool_mask`), this
// kernel has no opinion about what shape it encodes.
//
// Same online-softmax structure as attn_prefill.wgsl (one thread per query
// row, per-thread O(head_dim) accumulator, no per-row score storage), with
// the causal `j <= i` bound replaced by a full `j < kv_total` loop gated by
// `mask_bits`. Naive/reference shape (no flash-attention tiling): correct
// first, fast later - llm-life's chunks are hundreds, not thousands, of
// keys, and this is also the WASM-safe shape (one workgroup's threads only
// ever read, never share, key data).
const HEAD_DIM_MAX: u32 = 128u;

struct Dims {
    t: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_total: u32,
    max_ctx: u32,
    scale: f32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> mask_bits: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg.x;
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = head / n_rep;
    let hd = dims.head_dim;

    let i = wg.y * 256u + lid.x;
    if (i >= dims.t) {
        return;
    }

    let q_base = (i * dims.n_heads + head) * hd;
    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var acc: array<f32, HEAD_DIM_MAX>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        acc[d] = 0.0;
    }

    let row_bit_base = i * dims.kv_total;
    for (var j: u32 = 0u; j < dims.kv_total; j = j + 1u) {
        let bitidx = row_bit_base + j;
        let word = mask_bits[bitidx / 32u];
        let bit = (word >> (bitidx % 32u)) & 1u;
        if (bit == 0u) {
            continue;
        }

        let k_base = (kv_head * dims.max_ctx + j) * hd;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dot = dot + q[q_base + d] * k_cache[k_base + d];
        }
        let score = dot * dims.scale;

        let new_m = max(m, score);
        let factor = exp(m - new_m);
        let p = exp(score - new_m);
        l = l * factor + p;

        let v_base = (kv_head * dims.max_ctx + j) * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            acc[d] = acc[d] * factor + p * v_cache[v_base + d];
        }
        m = new_m;
    }

    let out_base = (i * dims.n_heads + head) * hd;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        out[out_base + d] = acc[d] / l;
    }
}
