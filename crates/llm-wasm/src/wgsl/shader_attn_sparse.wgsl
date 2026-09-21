// Prefill-time sparse attention: QK^T -> stable softmax -> PV where each
// query attends to a caller-supplied key *list* instead of to all of
// kv_len. Never materializes a [T, kv_len] score matrix.
//
// Why: `llm-life` (CONCEPT.md §2) drives this model as a cellular-automaton
// update rule — one token per cell, and a cell attends to the rules prefix,
// to itself and to its 8 grid neighbours, i.e. ~77 of 4164 keys. Run through
// the dense `ForwardSpec::mask_out` path that is a [T, kv_len] score tensor
// plus a bool mask plus `mask_fill` plus softmax, all O(T^2) in both flops
// and bandwidth, and it measured 68% of a 64x64 forward
// (llm-life docs/runs/2026-09-18-forward-profile.md). Here the cost is
// O(T * n_keys) instead.
//
// Mask representation (`model.rs`'s `SparseMask`): query `i` attends to the
// contiguous key range `[0, prefix_len[i])` plus `n_keys[i]` explicit key
// indices from row `i` of `keys` (stride `max_keys`). The prefix range
// covers both the shared text prefix every cell reads and the causal
// self-attention of the prefix rows themselves (`prefix_len[i] = i + 1`);
// the explicit list covers the stencil neighbourhood (<= 9) and variant A's
// own-cell block. Order within a row does not matter — softmax is
// permutation invariant over keys — but duplicate keys would be counted
// twice, so the builder must not emit any.
//
// Layouts:
//   q, output: [n_heads, T, head_dim], contiguous (Burn's [1,H,T,Dh] after
//              `into_contiguous`) — same in/out shape as the dense path's
//              `attention_with_mask`, so the caller's permute/reshape is
//              unchanged.
//   k_cache, v_cache: `kv.rs`'s `KvDtype::F32` layout,
//              [n_kv_heads, max_ctx, head_dim], sequence-major.
//   qmeta: [T, 2] = (prefix_len, n_explicit_keys) per query.
//   keys: [T, max_keys] u32 key indices; entries past `n_explicit` unread.
//   info: [n_heads, n_kv_heads, head_dim, T, max_ctx, max_keys, scale_bits].
//
// GQA: query head `h` reads kv head `h / (n_heads / n_kv_heads)` — same
// mapping as `model.rs`'s `repeat_kv`, so no K/V head materialization.
//
// Dispatch: `CubeCount::new_2d(T, n_heads)` (one workgroup per (query,
// head); the 2D grid is needed because T * n_heads exceeds WebGPU's 65535
// per-dimension cap well before T does), `CubeDim::new_1d(64)`.
//
// Limits, all asserted host-side in `gguf::attn_sparse_dispatch`:
// `prefix_len + n_keys <= MAXK` (the per-query score/key scratch lives in
// workgroup memory — that is what keeps this kernel from ever touching a
// T x kv_len buffer) and `head_dim <= MAXD`. A caller whose mask does not
// fit falls back to the dense path.
//
// Uniformity: every loop bound reaches the code through
// `workgroupUniformLoad`, including the two per-query values read out of
// `meta`. Tint treats a raw storage load as non-uniform and then rejects
// the `workgroupBarrier`s that follow a loop gated on it (see
// docs/ENGINE.md, 2026-09-10 Tint uniformity note); routing them through a
// workgroup variable is what makes the barriers legal.

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> qmeta: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read_write> info: array<u32>;

const WG: u32 = 64u;
const MAXK: u32 = 512u;

var<workgroup> q_sh: array<f32, 256>;
var<workgroup> sc: array<f32, 512>;
var<workgroup> kidx: array<u32, 512>;
var<workgroup> red: array<f32, 64>;
var<workgroup> wg_info: array<u32, 7>;
var<workgroup> wg_qmeta: array<u32, 2>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    if (tid == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
        wg_info[3] = info[3];
        wg_info[4] = info[4];
        wg_info[5] = info[5];
        wg_info[6] = info[6];
    }
    let li = workgroupUniformLoad(&wg_info);
    let n_heads = li[0];
    let n_kv_heads = li[1];
    let head_dim = li[2];
    let t = li[3];
    let max_ctx = li[4];
    let max_keys = li[5];
    let attn_scale = bitcast<f32>(li[6]);

    let qi = wg_id.x;
    let h = wg_id.y;
    let n_rep = n_heads / n_kv_heads;
    let kv_base = (h / n_rep) * max_ctx * head_dim;

    if (tid == 0u) {
        wg_qmeta[0] = qmeta[qi * 2u];
        wg_qmeta[1] = qmeta[qi * 2u + 1u];
    }
    let qm = workgroupUniformLoad(&wg_qmeta);
    let prefix_len = qm[0];
    let n_expl = qm[1];
    let n_keys = min(prefix_len + n_expl, MAXK);

    // Gather this query's key indices into workgroup memory once: phases A
    // and C both need them, and the prefix/explicit split is a branch worth
    // paying once rather than twice per key.
    var j = tid;
    loop {
        if (j >= n_keys) {
            break;
        }
        if (j < prefix_len) {
            kidx[j] = j;
        } else {
            kidx[j] = keys[qi * max_keys + (j - prefix_len)];
        }
        j = j + WG;
    }

    var d = tid;
    loop {
        if (d >= head_dim) {
            break;
        }
        q_sh[d] = q[(h * t + qi) * head_dim + d];
        d = d + WG;
    }
    workgroupBarrier();

    // Phase A: raw scores for this thread's strided keys, running max.
    // -1e30: a sentinel far below any attention score. Do NOT use
    // -3.4028235e38 — Tint rejects it as not representable in f32 while
    // native Naga accepts it (docs/ENGINE.md).
    var local_max: f32 = -1e30;
    j = tid;
    loop {
        if (j >= n_keys) {
            break;
        }
        let row = kv_base + kidx[j] * head_dim;
        var dot: f32 = 0.0;
        for (var dd = 0u; dd < head_dim; dd = dd + 1u) {
            dot += k_cache[row + dd] * q_sh[dd];
        }
        let raw = dot * attn_scale;
        sc[j] = raw;
        local_max = max(local_max, raw);
        j = j + WG;
    }
    red[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            red[tid] = max(red[tid], red[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = red[0];
    workgroupBarrier();

    // Phase B: exp(score - max) in place, running sum.
    var local_sum: f32 = 0.0;
    j = tid;
    loop {
        if (j >= n_keys) {
            break;
        }
        let p = exp(sc[j] - row_max);
        sc[j] = p;
        local_sum += p;
        j = j + WG;
    }
    red[tid] = local_sum;
    workgroupBarrier();
    stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = red[0];
    workgroupBarrier();

    // Phase C: thread `tid` owns output dims d = tid, tid+WG, ... and walks
    // every key.
    let inv = 1.0 / row_sum;
    d = tid;
    loop {
        if (d >= head_dim) {
            break;
        }
        var acc: f32 = 0.0;
        var j2: u32 = 0u;
        loop {
            if (j2 >= n_keys) {
                break;
            }
            acc += sc[j2] * v_cache[kv_base + kidx[j2] * head_dim + d];
            j2 = j2 + 1u;
        }
        output[(h * t + qi) * head_dim + d] = acc * inv;
        d = d + WG;
    }
}
