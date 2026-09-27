// New kernel: causal GQA attention over a full prefill sequence. Structure
// (one workgroup per (head), one thread per query position, MAX_SEQ cap)
// is modeled on t0-fast's shaders/attention.wgsl, but t0-fast has no GQA
// (its q/k/v share one head count) and no causal mask (t0 is
// bidirectional/patch-masked, not autoregressive) — both are new here.
// Reads q/k/v out of three separate buffers (Qwen2's GGUF keeps attn_q/k/v
// as separate tensors, unlike t0's fused wQKV), each [seq, heads, head_dim]
// contiguous (heads = n_heads for q, n_kv_heads for k/v). kv_head for query
// head h is `h / (n_heads / n_kv_heads)` (llama.cpp's repeat_kv order:
// kv_head k serves query heads [k*n_rep, (k+1)*n_rep)).
const MAX_SEQ: u32 = 256u;

struct Dims { seq: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg.x;
    let i = lid.x;
    if (i >= dims.seq) {
        return;
    }
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = head / n_rep;
    let hd = dims.head_dim;

    let q_base = (i * dims.n_heads + head) * hd;
    var scores: array<f32, MAX_SEQ>;

    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let k_base = (j * dims.n_kv_heads + kv_head) * hd;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            acc = acc + q[q_base + d] * k[k_base + d];
        }
        scores[j] = acc * dims.scale;
    }

    var maxv: f32 = scores[0];
    for (var j: u32 = 1u; j <= i; j = j + 1u) {
        maxv = max(maxv, scores[j]);
    }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let e = exp(scores[j] - maxv);
        scores[j] = e;
        sum = sum + e;
    }

    let out_base = (i * dims.n_heads + head) * hd;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j <= i; j = j + 1u) {
            let v_base = (j * dims.n_kv_heads + kv_head) * hd;
            acc = acc + (scores[j] / sum) * v[v_base + d];
        }
        out[out_base + d] = acc;
    }
}
