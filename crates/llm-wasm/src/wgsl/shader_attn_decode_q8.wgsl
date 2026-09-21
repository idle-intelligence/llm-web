// Decode-time (M=1) fused attention over a q8_0 KV cache: QK^T -> stable
// softmax -> PV, entirely in one kernel per layer, reading K/V directly out
// of the cache's q8_0 buffers (no dequant-to-f32 round trip). Accumulates
// in f32 throughout. One workgroup per query head (`wg_id.x = h`,
// `workgroup_size = head_dim` — this model's head_dim is 128); GQA maps
// query head `h` to kv head `h / (n_heads / n_kv_heads)`, matching
// `model.rs`'s `repeat_kv` head-repetition order (kv_head `k` repeats as
// query heads `[k*n_rep, (k+1)*n_rep)`).
//
// No causal mask needed: decode's single query is always the newest
// position (kv_len - 1), so it may attend to every key in [0, kv_len).
//
// Three phases per workgroup, each a full loop over `kv_len` (up to
// max_ctx=12288), synchronized by workgroup-wide reductions so a
// thread-divergent loop trip count around a barrier never happens (barriers
// always sit at the top level after a loop, guarded per the
// `workgroupUniformLoad` convention shared with `shader_q4_matvec.wgsl` —
// see that file's header for the Tint-uniformity rationale):
//   A. each thread computes raw (pre-softmax) scores for its strided keys
//      (thread `tid` owns keys `tid, tid+128, ...`), writing them to a
//      per-head scratch row and tracking a running max.
//   B. workgroup-reduce the max, then each thread re-visits its keys,
//      overwrites its scratch entries with `exp(score - max)`, and tracks a
//      running sum.
//   C. workgroup-reduce the sum; then thread `tid` owns *output* dimension
//      `d = tid` and loops over every key accumulating
//      `sum_key(p[key]/sum * V[key][d])` — no cross-thread reduction needed
//      here since each thread's accumulator is already the final value for
//      its own output element.
//
// Dispatch: `CubeCount::new_1d(n_heads)`, `CubeDim::new_1d(head_dim)`.
// `scratch` must be >= n_heads * max_ctx f32 (reused across layers/steps —
// see `model.rs`'s scratch-handle cache, same pattern as gguf.rs's
// `DEQUANT_SCRATCH`).

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_words: array<u32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> info: array<u32>;

const WG: u32 = 128u;
var<workgroup> q_shared: array<f32, 128>;
var<workgroup> reduce_buf: array<f32, 128>;
var<workgroup> wg_info: array<u32, 6>;

fn dequant_byte(byte: u32) -> f32 {
    let sval = bitcast<i32>(byte << 24u) >> 24u;
    return f32(sval);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (local_id.x == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
        wg_info[3] = info[3];
        wg_info[4] = info[4];
        wg_info[5] = info[5];
    }
    let li = workgroupUniformLoad(&wg_info);
    let n_heads = li[0];
    let n_kv_heads = li[1];
    let head_dim = li[2];
    let kv_len = li[3];
    let max_ctx = li[4];
    let attn_scale = bitcast<f32>(li[5]);

    let h = wg_id.x;
    let n_rep = n_heads / n_kv_heads;
    let kv_head = h / n_rep;
    let tid = local_id.x;
    let blocks_per_row = head_dim / 32u;
    let words_per_row = blocks_per_row * 8u;
    let krow_stride = max_ctx * blocks_per_row;
    let kword_stride = max_ctx * words_per_row;

    q_shared[tid] = q[h * head_dim + tid];
    workgroupBarrier();

    // Phase A: raw scores for this thread's strided keys, tracking a
    // running max.
    // -1e30: a sentinel far below any attention score. Do NOT use
    // -3.4028235e38 (f32::MIN as Rust prints it): Tint rejects it as
    // not representable in f32, while native Naga accepts it.
    var local_max: f32 = -1e30;
    var key: u32 = tid;
    loop {
        if (key >= kv_len) {
            break;
        }
        var dot: f32 = 0.0;
        let scale_base = kv_head * krow_stride + key * blocks_per_row;
        let word_base = kv_head * kword_stride + key * words_per_row;
        for (var bi = 0u; bi < blocks_per_row; bi = bi + 1u) {
            let s = k_scales[scale_base + bi];
            let d0 = bi * 32u;
            for (var wi = 0u; wi < 8u; wi = wi + 1u) {
                let word = k_words[word_base + bi * 8u + wi];
                let d = d0 + wi * 4u;
                dot += dequant_byte(word & 0xFFu) * s * q_shared[d];
                dot += dequant_byte((word >> 8u) & 0xFFu) * s * q_shared[d + 1u];
                dot += dequant_byte((word >> 16u) & 0xFFu) * s * q_shared[d + 2u];
                dot += dequant_byte((word >> 24u) & 0xFFu) * s * q_shared[d + 3u];
            }
        }
        let raw = dot * attn_scale;
        scratch[h * max_ctx + key] = raw;
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

    // Phase B: exp(score - max) in place, tracking a running sum.
    var local_sum: f32 = 0.0;
    key = tid;
    loop {
        if (key >= kv_len) {
            break;
        }
        let p = exp(scratch[h * max_ctx + key] - row_max);
        scratch[h * max_ctx + key] = p;
        local_sum += p;
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
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = reduce_buf[0];
    workgroupBarrier();

    // Phase C: thread `tid` owns output dim d=tid, loops over every key.
    let d = tid;
    let block = d / 32u;
    let byte_in_block = d % 32u;
    let word_idx = block * 8u + byte_in_block / 4u;
    let byte_shift = (byte_in_block % 4u) * 8u;
    let v_scale_base = kv_head * krow_stride + block;
    let v_word_base = kv_head * kword_stride + word_idx;

    var acc: f32 = 0.0;
    var k2: u32 = 0u;
    loop {
        if (k2 >= kv_len) {
            break;
        }
        let p = scratch[h * max_ctx + k2] / row_sum;
        let vs = v_scales[v_scale_base + k2 * blocks_per_row];
        let vw = v_words[v_word_base + k2 * words_per_row];
        let vb = (vw >> byte_shift) & 0xFFu;
        acc += p * (dequant_byte(vb) * vs);
        k2 = k2 + 1u;
    }
    output[h * head_dim + d] = acc;
}
