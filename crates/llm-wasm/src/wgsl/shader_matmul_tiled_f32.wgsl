// Prefill f32 GEMM: C[M,N] = A[M,K] . B[K,N], all row-major, all f32.
//
// Why a hand-written GEMM when `cubek_matmul` is already wired up: on this
// M2/Metal-via-wgpu adapter every CMMA/MMA candidate fails kernel selection
// (docs/ENGINE.md, Session 11), so cubek falls back to a non-tensor-core
// `DoubleUnit` kernel. Measured end to end on llm-life's Qwen2.5-0.5B
// prefill that is ~430 GFLOP/s against the M2's ~3.6 TFLOP/s f32 peak —
// about 12% — and it is the single largest term in a variant-B forward once
// the dense attention is gone (75% of it, llm-life
// docs/runs/2026-09-18-forward-profile.md). An `llm-agent prefill-sweep`
// across every non-dead cubek `Strategy` at this model's three shapes
// (896x896, 896x4864, 4864x896) found no better candidate, so the kernel is
// the thing to replace.
//
// Standard register-tiled GEMM, sized to WebGPU's limits rather than to
// Metal's: 128x128 output tile per workgroup, 16x16 = 256 invocations (the
// browser's per-workgroup cap), 8x8 outputs per thread, K stepped 8 at a
// time through two 4 KB workgroup staging buffers (8 KB total, inside the
// 16 KB floor). Arithmetic intensity in the inner loop is 64 FMAs per 16
// workgroup-memory loads (10 of them: 8 broadcast scalars and 2 vec4s).
//
// The 16 accumulators and the 8 staged A values are named variables, not an
// `array<vec4<f32>, 16>` indexed by the unroll counter: the array form
// measured 4x *slower* end to end, the signature of the Metal backend
// spilling it out of registers.
//
// The tile is 128 wide and not 64 because this GEMM is global-memory bound
// before it is compute bound: each workgroup re-reads the whole 128-row
// strip of A and 128-column strip of B, so total traffic is
// `(M/BM)*K*N + (N/BN)*M*K` floats — 160 GB per forward at BM=BN=64 on
// llm-life's 64x64 grid, half that at 128.
//
// A, B and C are bound as `array<vec4<f32>>`, so each staging load is one
// 16-byte load per invocation instead of four 4-byte ones. That is why the
// host requires `K % 8 == 0` and `N % 4 == 0` (both hold for every
// projection of every model this crate loads — K and N are multiples of 64)
// and routes anything else back to `cubek_matmul`. M is unconstrained: it is
// the token count, and its edge is handled by guarded loads and stores.
//
// B is `gguf::q4_dequant_scratch`'s output, which is already [K, N] — the
// transposed layout that dequant emits precisely so no separate transpose
// dispatch is needed.
//
// Dispatch: `CubeCount::new_2d(ceil(N/128), ceil(M/128))`,
// `CubeDim::new_2d(16, 16)`.

@group(0) @binding(0) var<storage, read_write> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;

// B's staging tile is a `vec4` array so the inner loop reads it as 2 vec4s
// per K-step instead of 8 scalars for the same 16 vec4 FMAs: on this GPU
// the scalar form is bound by threadgroup-memory issue rate, not ALU. A's
// tile stays scalar because each invocation stages 4 consecutive K of one
// row, which would land in one *component* of four different vec4 slots —
// and a component-wise write to a workgroup `vec4` is a read-modify-write
// of the whole vector, so four invocations sharing a slot race and corrupt
// each other (observed: tests/q4_matmul.rs's CPU-reference checks fail).
// Reading A as a broadcast scalar per row costs little: all 16 invocations
// with the same `ty` read the same address.
var<workgroup> as_tile: array<f32, 1024>;
var<workgroup> bs_tile: array<vec4<f32>, 256>;
var<workgroup> wg_info: array<u32, 3>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let tid = ty * 16u + tx;
    if (tid == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
    }
    // Routing the loop bounds through a uniform load is what keeps the
    // barriers below legal under Tint's uniformity analysis — a raw storage
    // read is non-uniform (docs/ENGINE.md, 2026-09-10).
    let li = workgroupUniformLoad(&wg_info);
    let m = li[0];
    let n = li[1];
    let k = li[2];
    let k4 = k / 4u;
    let n4 = n / 4u;

    let row0 = wg_id.y * BM;
    let col0 = wg_id.x * BN;

    // A staging: invocation `tid` owns one vec4, row `mi` of the tile,
    // K-quarter `kq` — so lane-consecutive invocations read address
    // consecutive vec4s along K.
    let a_mi = tid / 2u;
    let a_kq = tid % 2u;
    let a_row = row0 + a_mi;
    let a_live = a_row < m;
    let a_base = a_row * k4 + a_kq;
    // B staging: invocation `tid` owns one vec4, K-row `kk` of the tile,
    // column group `nq` — lane-consecutive along N, matching B's row-major
    // [K, N].
    let b_kk = tid / 32u;
    let b_nq = tid % 32u;
    let b_col = col0 + b_nq * 4u;
    let b_live = b_col < n;
    let b_base = col0 / 4u + b_nq;

    var acc0l = vec4<f32>(0.0);
    var acc0r = vec4<f32>(0.0);
    var acc1l = vec4<f32>(0.0);
    var acc1r = vec4<f32>(0.0);
    var acc2l = vec4<f32>(0.0);
    var acc2r = vec4<f32>(0.0);
    var acc3l = vec4<f32>(0.0);
    var acc3r = vec4<f32>(0.0);
    var acc4l = vec4<f32>(0.0);
    var acc4r = vec4<f32>(0.0);
    var acc5l = vec4<f32>(0.0);
    var acc5r = vec4<f32>(0.0);
    var acc6l = vec4<f32>(0.0);
    var acc6r = vec4<f32>(0.0);
    var acc7l = vec4<f32>(0.0);
    var acc7r = vec4<f32>(0.0);

    var k0: u32 = 0u;
    loop {
        if (k0 >= k) {
            break;
        }
        var av4 = vec4<f32>(0.0);
        if (a_live) {
            av4 = a[a_base + k0 / 4u];
        }
        // This invocation's vec4 is 4 consecutive K for one row, which lands
        // in one component of 4 different tile slots.
        let a_slot = a_kq * 4u * BM + a_mi;
        as_tile[a_slot] = av4.x;
        as_tile[a_slot + BM] = av4.y;
        as_tile[a_slot + 2u * BM] = av4.z;
        as_tile[a_slot + 3u * BM] = av4.w;

        var bv4 = vec4<f32>(0.0);
        if (b_live) {
            bv4 = b[(k0 + b_kk) * n4 + b_base];
        }
        bs_tile[b_kk * 32u + b_nq] = bv4;
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let bo = kk * 32u + tx * 2u;
            let b0 = bs_tile[bo];
            let b1 = bs_tile[bo + 1u];
            let ao = kk * BM + ty * 8u;
            let a0 = vec4<f32>(as_tile[ao]);
            acc0l = fma(a0, b0, acc0l);
            acc0r = fma(a0, b1, acc0r);
            let a1 = vec4<f32>(as_tile[ao + 1u]);
            acc1l = fma(a1, b0, acc1l);
            acc1r = fma(a1, b1, acc1r);
            let a2 = vec4<f32>(as_tile[ao + 2u]);
            acc2l = fma(a2, b0, acc2l);
            acc2r = fma(a2, b1, acc2r);
            let a3 = vec4<f32>(as_tile[ao + 3u]);
            acc3l = fma(a3, b0, acc3l);
            acc3r = fma(a3, b1, acc3r);
            let a4 = vec4<f32>(as_tile[ao + 4u]);
            acc4l = fma(a4, b0, acc4l);
            acc4r = fma(a4, b1, acc4r);
            let a5 = vec4<f32>(as_tile[ao + 5u]);
            acc5l = fma(a5, b0, acc5l);
            acc5r = fma(a5, b1, acc5r);
            let a6 = vec4<f32>(as_tile[ao + 6u]);
            acc6l = fma(a6, b0, acc6l);
            acc6r = fma(a6, b1, acc6r);
            let a7 = vec4<f32>(as_tile[ao + 7u]);
            acc7l = fma(a7, b0, acc7l);
            acc7r = fma(a7, b1, acc7r);
        }
        workgroupBarrier();
        k0 = k0 + BK;
    }

    let cr = row0 + ty * 8u;
    let cc = col0 + tx * 8u;
    if (cc >= n) {
        return;
    }
    let cbase = cc / 4u;
    if (cr + 0u < m) { c[(cr + 0u) * n4 + cbase] = acc0l; c[(cr + 0u) * n4 + cbase + 1u] = acc0r; }
    if (cr + 1u < m) { c[(cr + 1u) * n4 + cbase] = acc1l; c[(cr + 1u) * n4 + cbase + 1u] = acc1r; }
    if (cr + 2u < m) { c[(cr + 2u) * n4 + cbase] = acc2l; c[(cr + 2u) * n4 + cbase + 1u] = acc2r; }
    if (cr + 3u < m) { c[(cr + 3u) * n4 + cbase] = acc3l; c[(cr + 3u) * n4 + cbase + 1u] = acc3r; }
    if (cr + 4u < m) { c[(cr + 4u) * n4 + cbase] = acc4l; c[(cr + 4u) * n4 + cbase + 1u] = acc4r; }
    if (cr + 5u < m) { c[(cr + 5u) * n4 + cbase] = acc5l; c[(cr + 5u) * n4 + cbase + 1u] = acc5r; }
    if (cr + 6u < m) { c[(cr + 6u) * n4 + cbase] = acc6l; c[(cr + 6u) * n4 + cbase + 1u] = acc6r; }
    if (cr + 7u < m) { c[(cr + 7u) * n4 + cbase] = acc7l; c[(cr + 7u) * n4 + cbase + 1u] = acc7r; }
}
