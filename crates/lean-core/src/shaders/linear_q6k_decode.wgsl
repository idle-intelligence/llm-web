// Q6_K matvec for decode (M = 1). Same per-weight formula and block layout
// as linear_q6k.wgsl (see that file's header and `gguf.rs::dequantize_q6_k`),
// reorganised so that each lane works on whole vec4 words:
//
// - A row is 4 units per 256-weight super-block: (half, is) with half the
//   128-weight half and `is` the 16-wide slice of `l` it covers, 64 weights
//   per unit. A unit reads its weights as vec4<u32> (4 words, 16 consecutive
//   `l`) and x as vec4<f32>: two ql vec4 (the words of `l` and of `l + 32`),
//   one qh vec4, the block's scale vec4, and every word component yields 4
//   weights of each of the four groups (l, l + 32, l + 64, l + 96).
// - 8 lanes share an output row and a 128-thread workgroup covers 16 rows.
//   Lane `i` takes units `i`, `i + 8`, `i + 16`, `i + 24` of the row, so at
//   any one step the 8 lanes read 8 consecutive units (consecutive vec4s of
//   ql, qh and x, which coalesces). A lane that instead owned a whole block
//   was 1.7x slower on an RTX 3080 and no faster on an M2.
// - The 6-bit values of 4 weights are assembled in one u32 (four bytes,
//   each 0..63) by masks and shifts on the whole word, then turned into
//   floats with the exponent trick: a byte OR 0x4B000000 is the float
//   8388608 + byte, so one subtraction of 8388608 + 32 gives `q - 32`
//   exactly. That is 5 or 6 integer/float ops per weight against ~30 for the
//   scalar byte-by-byte version this replaces, which was bound by ALU rather
//   than by memory (the head ran at about a third of the read ceiling on an
//   M2).
// - The per-16 sub-block scale multiplies a partial dot product per group,
//   and `d` multiplies the unit's sum, so x is never scaled per weight.
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    // See linear_q4.wgsl's Dims doc comment: row-chunk offset/total for
    // weights split across bindings.
    n_offset: u32,
    n_total: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ql: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> qh: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> scales: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> dscale: array<f32>;
@group(0) @binding(5) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

const WG_SIZE: u32 = 128u;
const LANES: u32 = 8u;
const ROWS_PER_WG: u32 = 16u; // WG_SIZE / LANES
const UNITS_PER_LANE: u32 = 4u;

var<workgroup> partial_sums: array<f32, WG_SIZE>;

// Four bytes (each 0..63) of `q` as `byte - 32` floats.
fn q4(q: u32) -> vec4<f32> {
    let v = vec4<u32>(q & 0xFFu, (q >> 8u) & 0xFFu, (q >> 16u) & 0xFFu, q >> 24u) | vec4<u32>(0x4B000000u);
    return bitcast<vec4<f32>>(v) - vec4<f32>(8388640.0);
}

// Signed byte `k` (0..3) of `w`.
fn sc_i8(w: u32, k: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - 8u * k)) >> 24u);
}

// The four groups' 6-bit words for the 4 weights `l4 .. l4 + 3` of each:
// wa holds the ql bytes of `l` (low nibble: group 0, high: group 2), wb the
// ql bytes of `l + 32` (groups 1 and 3), wh the qh bytes (2 bits per group).
fn group_dot(c: u32, wa: vec4<u32>, wb: vec4<u32>, wh: vec4<u32>, xb: u32, s: ptr<function, vec4<f32>>) {
    let a = wa[c];
    let bb = wb[c];
    let h = wh[c];
    let n = 0x0F0F0F0Fu;
    let q0 = (a & n) | ((h & 0x03030303u) << 4u);
    let q1 = (bb & n) | ((h & 0x0C0C0C0Cu) << 2u);
    let q2 = ((a >> 4u) & n) | (h & 0x30303030u);
    let q3 = ((bb >> 4u) & n) | ((h >> 2u) & 0x30303030u);
    var t = *s;
    t.x += dot(x[xb + c], q4(q0));
    t.y += dot(x[xb + 8u + c], q4(q1));
    t.z += dot(x[xb + 16u + c], q4(q2));
    t.w += dot(x[xb + 24u + c], q4(q3));
    *s = t;
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let lane = tid % LANES;
    let r = tid / LANES;
    let n = wg_id.x * ROWS_PER_WG + r;
    let row_ok = n < dims.n;
    let row = min(n, dims.n - 1u) * dims.blocks_per_row;
    let total = dims.blocks_per_row * 4u;

    var acc: f32 = 0.0;
    for (var base: u32 = 0u; base < total; base = base + LANES * UNITS_PER_LANE) {
        for (var i: u32 = 0u; i < UNITS_PER_LANE; i = i + 1u) {
            let u = base + lane + i * LANES;
            if (u >= total) {
                continue;
            }
            let blk = u / 4u;
            let is = u & 1u;
            let half = (u >> 1u) & 1u;
            let bi = row + blk;
            let sc = scales[bi];
            let sw0 = select(sc.x, sc.z, half == 1u);
            let sw1 = select(sc.y, sc.w, half == 1u);
            let wa = ql[bi * 8u + half * 4u + is];
            let wb = ql[bi * 8u + half * 4u + 2u + is];
            let wh = qh[bi * 4u + half * 2u + is];
            let xb = blk * 64u + half * 32u + is * 4u;
            var s = vec4<f32>(0.0);
            group_dot(0u, wa, wb, wh, xb, &s);
            group_dot(1u, wa, wb, wh, xb, &s);
            group_dot(2u, wa, wb, wh, xb, &s);
            group_dot(3u, wa, wb, wh, xb, &s);
            let scl = vec4<f32>(sc_i8(sw0, is), sc_i8(sw0, is + 2u), sc_i8(sw1, is), sc_i8(sw1, is + 2u));
            acc += dscale[bi] * dot(s, scl);
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && row_ok) {
        var v = partial_sums[tid] + b[dims.n_offset + n];
        if (dims.act == 1u) {
            v = max(v, 0.0);
        }
        out[dims.n_offset + n] = v;
    }
}
