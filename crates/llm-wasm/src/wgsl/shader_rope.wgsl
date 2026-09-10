// F1 (docs/BENCHMARKS.md Session 9): fused rotate-half RoPE for q and k in
// one dispatch, replacing model.rs's old apply_rope Burn-op chain (narrow
// x2 + mul_scalar + cat + mul + mul + add, x2 for q and k — ~10 dispatches,
// see bin/llm-agent.rs's print_dispatch_estimate). q/k are mutated
// in-place: this kernel assigns exactly one thread to each (row, head, j)
// pair with j in [0, half_dim); that thread owns both x[...,j] and
// x[...,half+j] for its (row,head) — it reads both before writing either —
// so no other thread ever observes a partially-rotated value, and
// read_write aliasing of input==output on the same buffer is safe. No
// workgroup barriers are used at all (every thread is fully independent),
// so none of the Tint barrier-under-branch hazard described in
// shader_rmsnorm.wgsl's header comment applies here — the early `return`
// below guards work only, never a barrier.
//
// q: [1, T, H, Dh] contiguous, row-major (T, then H, then Dh) — the
// natural reshape() layout *before* model.rs permutes to [1,H,T,Dh]; fusing
// here avoids paying for permute's extra into_contiguous. k: [1, T, Hkv,
// Dh], same layout. cos/sin: RoPE::new's existing [max_seq_len, Dh] table
// (`emb = cat([freqs, freqs])`, HF Qwen2RotaryEmbedding convention) — both
// halves hold identical values, so only the first half_dim columns are
// read, indexed with the full Dh stride. info: [T, H, Hkv, half_dim,
// offset, cos_sin_stride] packed as f32 — exact integers well within f32's
// 24-bit mantissa.
//
// x_rope = x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1]:
//   out[j]      = x1*c - x2*s
//   out[half+j] = x2*c + x1*s
// Must match model.rs's `apply_rope`/`rotate_half` bit-for-bit (within
// float-op reordering tolerance) — see model.rs's
// `rope_fused_matches_apply_rope` unit test.
//
// Dispatch: WebGPU caps every dispatch dimension at 65535 workgroups (a
// separate limit from the 256-invocation-per-workgroup cap) — at prefill
// this kernel's flat `T*(H+Hkv)*half_dim` work exceeds that in one 1D
// dispatch, so the launch is 2D (`gguf.rs::workgroups_2d`) and the flat
// element index is recovered as `gid.y * row_width + gid.x`, `row_width`
// (== `wg_x * 256`) passed via `info[6]`.
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> info: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = u32(info[0]);
    let h = u32(info[1]);
    let hkv = u32(info[2]);
    let half = u32(info[3]);
    let offset = u32(info[4]);
    let stride = u32(info[5]);
    let row_width = u32(info[6]);

    let idx = gid.y * row_width + gid.x;
    let per_row = (h + hkv) * half;
    let total = t * per_row;
    if (idx >= total) {
        return;
    }

    let row = idx / per_row;
    let rem = idx % per_row;
    let q_len = h * half;
    let dh = half * 2u;
    let pos = offset + row;

    if (rem < q_len) {
        let head = rem / half;
        let j = rem % half;
        let c = cos_table[pos * stride + j];
        let s = sin_table[pos * stride + j];
        let base = row * h * dh + head * dh;
        let x1 = q[base + j];
        let x2 = q[base + half + j];
        q[base + j] = x1 * c - x2 * s;
        q[base + half + j] = x2 * c + x1 * s;
    } else {
        let rem2 = rem - q_len;
        let head = rem2 / half;
        let j = rem2 % half;
        let c = cos_table[pos * stride + j];
        let s = sin_table[pos * stride + j];
        let base = row * hkv * dh + head * dh;
        let x1 = k[base + j];
        let x2 = k[base + half + j];
        k[base + j] = x1 * c - x2 * s;
        k[base + half + j] = x2 * c + x1 * s;
    }
}
