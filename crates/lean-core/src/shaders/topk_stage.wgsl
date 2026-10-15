// One stage of a GPU top-k over a logits vector. Each workgroup sorts 1024
// (value, id) items in shared memory with a bitonic network and writes its
// best `out_per_group` (64) items. Stage 0 reads the logits (id = index),
// later stages read the previous stage's output; the last stage has one
// workgroup and writes the final 64. Order is value descending, ties toward
// the lower id, so the result does not depend on the reduction order.
// Padding carries the lowest f32 and sorts last.
// Integer-valued output: [value bits, id] pairs, 2 u32 per item.
struct Dims {
    n_in: u32,
    mode: u32,
    out_per_group: u32,
    out_base: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> cand_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> cand_out: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const N: u32 = 1024u;
const LOWEST_BITS: u32 = 0xff7fffffu;

var<workgroup> sv: array<f32, 1024>;
var<workgroup> si: array<u32, 1024>;

fn before(va: f32, ia: u32, vb: f32, ib: u32) -> bool {
    return va > vb || (va == vb && ia < ib);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let base = wid.x * N;
    for (var r = 0u; r < 4u; r = r + 1u) {
        let s = t + r * 256u;
        let g = base + s;
        var v = bitcast<f32>(LOWEST_BITS);
        var id = 0xffffffffu;
        if (g < dims.n_in) {
            if (dims.mode == 0u) {
                v = logits[g];
                id = g;
            } else {
                v = bitcast<f32>(cand_in[2u * g]);
                id = cand_in[2u * g + 1u];
            }
        }
        sv[s] = v;
        si[s] = id;
    }
    workgroupBarrier();

    for (var size = 2u; size <= N; size = size << 1u) {
        for (var stride = size >> 1u; stride > 0u; stride = stride >> 1u) {
            for (var q = 0u; q < 2u; q = q + 1u) {
                let p = t + q * 256u;
                let lo = p & (stride - 1u);
                let i = ((p - lo) << 1u) | lo;
                let j = i + stride;
                let asc = (i & size) == 0u;
                let a_first = before(sv[i], si[i], sv[j], si[j]);
                let b_first = before(sv[j], si[j], sv[i], si[i]);
                if (select(a_first, b_first, asc)) {
                    let tv = sv[i];
                    let ti = si[i];
                    sv[i] = sv[j];
                    si[i] = si[j];
                    sv[j] = tv;
                    si[j] = ti;
                }
            }
            workgroupBarrier();
        }
    }

    if (t < dims.out_per_group) {
        let pos = wid.x * dims.out_per_group + t;
        cand_out[dims.out_base + 2u * pos] = bitcast<u32>(sv[t]);
        cand_out[dims.out_base + 2u * pos + 1u] = si[t];
    }
}
