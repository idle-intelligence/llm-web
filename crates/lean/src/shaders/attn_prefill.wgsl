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
// Fix: flash-attention-style online (running) softmax, dispatched over a
// 2D grid `(n_heads, ceil(seq/256))` instead of `(n_heads, 1)` — each
// workgroup owns one 256-row query tile of one head, one thread per row
// (`i = wg.y * 256 + lid.x`), so the number of GPU threads in flight scales
// with `seq` instead of staying fixed at `n_heads * 256`. An earlier
// version kept the `(n_heads, 1)` dispatch and had each thread grid-stride
// over every row of a long sequence (~9 rows/thread at 2225 tokens, the
// longest queued behind the single largest row's O(seq) inner loop) - that
// serialized far too much scalar per-thread work onto too few threads and
// triggered a native Metal `device poll failed: Timeout` on real
// ~2,200-2,400 token Sonos-agent prompts (fixture cases
// `long_tools_single`/`long_tools_multiturn`); this dispatch shape is the
// fix. Each thread keeps only a scalar running max/sum plus a
// `head_dim`-sized accumulator in registers for its one row — no per-row
// storage of all `j <= i` scores, so per-thread memory is O(head_dim)
// regardless of sequence length. Same online-softmax update as
// attn_decode.wgsl's per-tile version, applied per-key instead of per-tile
// since there's no shared memory here to batch keys into.
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

    let i = wg.y * 256u + lid.x;
    if (i >= dims.seq) {
        return;
    }

    let q_base = (i * dims.n_heads + head) * hd;
    var m: f32 = -1e30; // effectively -infinity for this kernel's score range; -f32::MAX's own literal (-3.4028235e38) fails Tint's WGSL f32 range check (rounds just past f32::MAX)
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
}
