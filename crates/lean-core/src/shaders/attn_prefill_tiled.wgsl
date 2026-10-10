// Causal GQA prefill attention in 8-query x 8-key tiles, for head_dim 64
// or 128 (`HEAD_DIM` override, the same two values and the same
// shape-based choice as attn_decode_split.wgsl; model.rs falls back to
// attn_prefill.wgsl for any other head_dim).
//
// attn_prefill.wgsl gives each query row one thread with a dynamically
// indexed head_dim-long private accumulator and reads q, k and v as
// scalars from storage buffers, one key at a time; for a 36-token prompt
// only 36 of each workgroup's 256 threads have a row. Here a workgroup
// of 64 threads owns 8 query rows of one head, 8 lanes per row:
// - per key tile, K and V rows (8 keys) are staged in workgroup memory as
//   vec4s next to the tile's q rows;
// - lane t of a row computes that row's score against key t of the tile,
//   then every lane of the row takes the tile's 8 scores and does the
//   online-softmax update (running max and sum per tile of keys, as in
//   attn_decode.wgsl) on its own 8 or 16 of the head_dim outputs, held in
//   fixed vec4 registers;
// - rows are padded by one vec4 in workgroup memory so the 8 rows (and 8
//   keys) read in the same step land in different banks.
// Keys past a row's position (and rows past `seq`) are masked to a score of
// -1e30, which gives p = 0 once the row has seen key 0 (key 0 is in its
// first tile). Results are written as vec4s.
//
// Layouts as attn_prefill.wgsl: q [seq, n_heads, head_dim], k/v [seq,
// n_kv_heads, head_dim], out [seq, n_heads, head_dim]; kv_head = head /
// (n_heads / n_kv_heads). Dispatch: (n_heads, ceil(seq / 8), 1).
override HEAD_DIM: u32 = 64u;

const QT: u32 = 8u;
const KT: u32 = 8u;
override DV: u32 = HEAD_DIM / 4u; // vec4s per row
override RS: u32 = HEAD_DIM / 4u + 1u; // padded row stride in vec4s
override UPL: u32 = HEAD_DIM / 32u; // output vec4s per lane (2 or 4)

struct Dims { seq: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> qs: array<vec4<f32>, 264>; // QT * RS, RS <= 33
var<workgroup> ks: array<vec4<f32>, 264>;
var<workgroup> vs: array<vec4<f32>, 264>;
var<workgroup> sc: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let head = wg.x;
    let kv_head = head / (dims.n_heads / dims.n_kv_heads);
    let lane = tid % 8u;
    let r = tid / 8u;
    let i0 = wg.y * QT;
    let i = i0 + r;
    let last = min(i0 + QT, dims.seq) - 1u;

    // Stage the tile's q rows (rows past seq read row 0 and are never written).
    for (var e: u32 = tid; e < QT * DV; e = e + 64u) {
        let rr = e / DV;
        let d = e % DV;
        let qi = select(0u, i0 + rr, i0 + rr < dims.seq);
        qs[rr * RS + d] = q[(qi * dims.n_heads + head) * DV + d];
    }

    var m: f32 = -1e30;
    var l: f32 = 0.0;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    for (var j0: u32 = 0u; j0 <= last; j0 = j0 + KT) {
        for (var e: u32 = tid; e < KT * DV; e = e + 64u) {
            let kk = e / DV;
            let d = e % DV;
            let j = select(0u, j0 + kk, j0 + kk < dims.seq);
            let src = (j * dims.n_kv_heads + kv_head) * DV + d;
            ks[kk * RS + d] = k[src];
            vs[kk * RS + d] = v[src];
        }
        workgroupBarrier();

        // Score of row r against key j0 + lane.
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < DV; d = d + 1u) {
            s = s + dot(qs[r * RS + d], ks[lane * RS + d]);
        }
        let j = j0 + lane;
        sc[tid] = select(-1e30, s * dims.scale, j <= i && i < dims.seq);
        workgroupBarrier();

        let sb = r * 8u;
        var mt = sc[sb];
        for (var t: u32 = 1u; t < KT; t = t + 1u) {
            mt = max(mt, sc[sb + t]);
        }
        let new_m = max(m, mt);
        let factor = exp(m - new_m);
        a0 = a0 * factor;
        a1 = a1 * factor;
        a2 = a2 * factor;
        a3 = a3 * factor;
        var psum: f32 = 0.0;
        for (var t: u32 = 0u; t < KT; t = t + 1u) {
            let p = exp(sc[sb + t] - new_m);
            psum = psum + p;
            let vb = t * RS + lane;
            a0 += p * vs[vb];
            a1 += p * vs[vb + 8u];
            if (UPL == 4u) {
                a2 += p * vs[vb + 16u];
                a3 += p * vs[vb + 24u];
            }
        }
        l = l * factor + psum;
        m = new_m;
        workgroupBarrier();
    }

    if (i < dims.seq) {
        let ob = (i * dims.n_heads + head) * DV + lane;
        out[ob] = a0 / l;
        out[ob + 8u] = a1 / l;
        if (UPL == 4u) {
            out[ob + 16u] = a2 / l;
            out[ob + 24u] = a3 / l;
        }
    }
}
