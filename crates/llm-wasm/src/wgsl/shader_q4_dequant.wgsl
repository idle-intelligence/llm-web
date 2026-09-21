// P1 (docs/BENCHMARKS.md Session 4, Approach A): dequantize a Q4_0 weight
// tensor [N, K] into a transposed f32 scratch buffer [K, N], so the caller
// can run `x[B,M,K] . W_f32[1,K,N]` through Burn's `Tensor::matmul` (cubecl's
// tiled/cmma kernels) instead of the naive per-element-redundant-dequant
// matmul kernel (`shader_naive.wgsl`) used for M>=32 prefill matmuls.
//
// One thread per output scalar `output[k*N + n]`. Threads are dispatched so
// that `n` is the fastest-varying index (idx = k*N + n, n = idx % N), which
// makes writes to `output` fully coalesced across a warp/subgroup — the
// larger of the two data movements (N*K f32 out vs N*K/2 bytes of nibbles
// in), so coalescing the write side matters more than the read side here.
// Reads of `weights`/`scales` are not coalesced across threads (adjacent
// threads read different rows `n`), which is the accepted tradeoff for a
// first cut — see docs/BENCHMARKS.md Session 4 P1 for the measured effect.
//
// No barriers, no loops bounded by a storage-buffer value — the Tint
// uniformity rule (docs/ENGINE.md "Known issues / fixed") doesn't apply to
// this kernel; the only branch is a per-thread bounds-check `return`, which
// is safe because there's no barrier/subgroup op anywhere in this shader.

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

// Dispatched as a 2D workgroup grid (`info[3]` = workgroups-per-row on the
// x axis) because a 1D dispatch's total workgroup count can exceed
// WebGPU's per-dimension dispatch limit (65535) for this model's larger
// layers (e.g. 11008x2048 needs 88064 workgroups of 256 threads) — see
// docs/BENCHMARKS.md Session 4 P1.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_dim = info[0];
    let k_dim = info[1];
    let blocks_per_row = info[2];
    let threads_per_row = info[3];

    let idx = gid.y * threads_per_row + gid.x;
    let total = n_dim * k_dim;
    if (idx >= total) {
        return;
    }

    let k = idx / n_dim;
    let n = idx % n_dim;

    let blk = k / 32u;
    let e = k % 32u;

    let global_block = n * blocks_per_row + blk;
    let scale = scales[global_block];
    let nibble_base = global_block * 4u;

    var wi: u32;
    var byte_idx: u32;
    var use_hi: bool;
    if (e < 16u) {
        wi = e / 4u;
        byte_idx = e % 4u;
        use_hi = false;
    } else {
        let e2 = e - 16u;
        wi = e2 / 4u;
        byte_idx = e2 % 4u;
        use_hi = true;
    }

    let packed = weights[nibble_base + wi];
    let byte = (packed >> (byte_idx * 8u)) & 0xFFu;
    var nib: u32;
    if (use_hi) {
        nib = (byte >> 4u) & 0xFu;
    } else {
        nib = byte & 0xFu;
    }

    output[k * n_dim + n] = (f32(nib) - 8.0) * scale;
}
