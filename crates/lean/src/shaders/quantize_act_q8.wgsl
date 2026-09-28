// Online per-32-block int8 activation quantization (this crate's own
// design, not ported from another project's shader - see llama.cpp's
// `mul_mat_vecq.comp`/`quantize_row_q8_1` in
// docs/runs/2026-09-28-lean-perf-2.md session 4's step 4 for the *technique*
// this implements, symmetric int8 rather than llama.cpp's Q8_1 sum-carrying
// variant since this crate's Q4_0 nibbles are repacked with the -8 offset
// already folded in at dequant time - see linear_q4_decode_dp4.wgsl - so no
// activation-sum correction term is needed).
//
// One workgroup per 32-value block: reduces the block's absmax across its
// 32 lanes (shared-memory tree, no subgroup ops - this crate's portability
// rule), picks `scale = absmax / 127`, quantizes each lane's value to a
// clamped `i8` (stored in its 8-bit two's complement pattern inside a u32),
// and packs 4 consecutive lanes' quantized values into one `u32` (`act_q`)
// laid out identically to `linear_q8_decode.wgsl`'s own qs/word convention
// (8 words per 32-value block) so the two dp4 decode kernels can index
// `act_q` and their own weight `qs` word-for-word.
struct Dims {
    k: u32,
    blocks: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> act_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> act_scale: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> reduce_buf: array<f32, 32>;
var<workgroup> qi: array<i32, 32>;
var<workgroup> block_scale: f32;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let blk = wg_id.x;
    let lane = local_id.x;
    let k0 = blk * 32u + lane;

    var v: f32 = 0.0;
    if (k0 < dims.k) {
        v = x[k0];
    }
    reduce_buf[lane] = abs(v);
    workgroupBarrier();

    var stride: u32 = 16u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduce_buf[lane] = max(reduce_buf[lane], reduce_buf[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        let absmax = reduce_buf[0];
        let scale = select(absmax / 127.0, 1.0, absmax == 0.0);
        block_scale = scale;
        act_scale[blk] = scale;
    }
    workgroupBarrier();

    let scale = block_scale;
    let q = clamp(i32(round(v / scale)), -127, 127);
    qi[lane] = q;
    workgroupBarrier();

    // Lanes 0..7 each pack 4 consecutive quantized values (their own
    // `lane*4 .. lane*4+3`) into one u32, matching linear_q8_decode.wgsl's
    // word layout (8 words / 32-value block).
    if (lane < 8u) {
        let base = lane * 4u;
        let b0 = u32(qi[base]) & 0xFFu;
        let b1 = u32(qi[base + 1u]) & 0xFFu;
        let b2 = u32(qi[base + 2u]) & 0xFFu;
        let b3 = u32(qi[base + 3u]) & 0xFFu;
        act_q[blk * 8u + lane] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    }
}
