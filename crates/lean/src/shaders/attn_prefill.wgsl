// Causal GQA attention over a full prefill sequence, length-independent.
//
// Previous version capped `seq` at a fixed `MAX_SEQ = 256` private `scores`
// array (one workgroup per head, one thread per query position, so only
// query rows 0..255 ever got dispatched at all — rows >= 256 silently
// produced no attention output). The Sonos MCP agent's real prompts run
// 1,000-2,000 tokens (see fixtures/reference/rendered/02_tools_single.txt
// at 2225 tokens), so that cap made every long-prompt prefill wrong, not
// just slow.
//
// Fix: flash-attention-style online (running) softmax. Each thread still
// owns query rows via workgroup_size(256)'s `lid.x`, but now loops
// (grid-stride) over every row `i = lid.x, lid.x+256, ...` up to `seq`, and
// for each row keeps only a scalar running max/sum plus a `head_dim`-sized
// accumulator in registers — no per-row storage of all `j <= i` scores, so
// memory use is O(head_dim) per thread regardless of sequence length. Same
// online-softmax update as attn_decode.wgsl's per-tile version, applied
// per-key instead of per-tile since there's no shared memory here to batch
// keys into.
//
// Reads q/k/v out of three separate buffers (Qwen2's GGUF keeps attn_q/k/v
// as separate tensors), each [seq, heads, head_dim] contiguous (heads =
// n_heads for q, n_kv_heads for k/v). kv_head for query head h is
// `h / (n_heads / n_kv_heads)` (llama.cpp's repeat_kv order: kv_head k
// serves query heads [k*n_rep, (k+1)*n_rep)).
//
// `HEAD_DIM_MAX` is a compile-time bound on the private accumulator array
// (WGSL arrays need a const length); 64 covers every head_dim this crate
// targets today (Qwen2.5-0.5B's head_dim=64, same assumption
// attn_decode.wgsl's `WG: u32 = 64u` already hardcodes). Bump alongside
// that constant if a model with a larger head_dim is ever added.
const HEAD_DIM_MAX: u32 = 64u;

struct Dims { seq: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg.x;
    let n_rep = dims.n_heads / dims.n_kv_heads;
    let kv_head = head / n_rep;
    let hd = dims.head_dim;

    var i: u32 = lid.x;
    loop {
        if (i >= dims.seq) {
            break;
        }

        let q_base = (i * dims.n_heads + head) * hd;
        var m: f32 = -3.4028235e38; // -f32::MAX; running max
        var l: f32 = 0.0; // running softmax denominator
        var acc: array<f32, HEAD_DIM_MAX>;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            acc[d] = 0.0;
        }

        for (var j: u32 = 0u; j <= i; j = j + 1u) {
            let k_base = (j * dims.n_kv_heads + kv_head) * hd;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + q[q_base + d] * k[k_base + d];
            }
            let score = dot * dims.scale;

            let new_m = max(m, score);
            let factor = exp(m - new_m);
            let p = exp(score - new_m);
            l = l * factor + p;

            let v_base = (j * dims.n_kv_heads + kv_head) * hd;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                acc[d] = acc[d] * factor + p * v[v_base + d];
            }
            m = new_m;
        }

        let out_base = (i * dims.n_heads + head) * hd;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            out[out_base + d] = acc[d] / l;
        }

        i = i + 256u;
    }
}
