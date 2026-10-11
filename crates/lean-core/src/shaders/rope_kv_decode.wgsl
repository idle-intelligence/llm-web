// One decode step's RoPE and K/V cache write in a single dispatch: rotates
// q into `q_out`, rotates k and writes it into the K cache at `pos`, and
// copies v into the V cache at `pos`. Replaces two rope_neox.wgsl dispatches
// plus four copy_buffer_to_buffer commands (and the compute-pass break they
// needed) per layer. Same split-half (NeoX) rotation and the same
// arithmetic as rope_neox.wgsl; cache layout is kv_scatter.wgsl's
// `[kv_head, position, head_dim]`.
//
// q, k and v are read-only views (they may be ranges of one fused qkv
// buffer, which can only be bound read-only more than once in a dispatch),
// hence q goes to its own output buffer instead of being rotated in place.
struct Dims {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    max_ctx: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> cos_t: array<f32>;
@group(0) @binding(4) var<storage, read> sin_t: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(8) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = dims.head_dim;
    let half = hd / 2u;
    let nq = dims.n_heads * half;
    let nk = dims.n_kv_heads * half;
    let nv = dims.n_kv_heads * hd;
    var id = gid.x;
    if (id < nq + nk) {
        let is_k = id >= nq;
        let r = select(id, id - nq, is_k);
        let j = r % half;
        let head = r / half;
        let base = head * hd;
        let c = cos_t[dims.pos * half + j];
        let s = sin_t[dims.pos * half + j];
        if (is_k) {
            let x0 = k[base + j];
            let x1 = k[base + half + j];
            let dst = (head * dims.max_ctx + dims.pos) * hd;
            k_cache[dst + j] = x0 * c - x1 * s;
            k_cache[dst + half + j] = x1 * c + x0 * s;
        } else {
            let x0 = q[base + j];
            let x1 = q[base + half + j];
            q_out[base + j] = x0 * c - x1 * s;
            q_out[base + half + j] = x1 * c + x0 * s;
        }
        return;
    }
    id = id - nq - nk;
    if (id < nv) {
        let d = id % hd;
        let head = id / hd;
        v_cache[(head * dims.max_ctx + dims.pos) * hd + d] = v[id];
    }
}
