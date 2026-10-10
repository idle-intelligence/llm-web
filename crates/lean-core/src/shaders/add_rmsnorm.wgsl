// Fuses `add_inplace.wgsl` (residual add) with the immediately-following
// `rmsnorm.wgsl` read of that same residual into one dispatch - every
// residual add in this crate's decoder layer (`add1`/`add2`) is followed by
// exactly one rmsnorm read of the updated residual (the next norm, or
// `dec_out_norm`/`out_norm` after the last layer), so the two dispatches
// always pair up. Same one-workgroup-per-row, 256-thread shared-memory
// tree-reduction structure as rmsnorm.wgsl (see that file's header) with the
// residual add folded into its first pass: each thread computes
// `v = a[i] + delta[i]`, writes `v` back into `a` in place (so `a` still
// holds the updated residual for whatever reads it later, exactly as
// `add_inplace` left it), and accumulates `v*v` for the sum-of-squares in
// the same loop - saving the separate `add_inplace` dispatch entirely
// instead of running it before this one. Same per-element math and
// operation order as `add_inplace` followed by `rmsnorm` (elementwise add,
// then the existing cooperative-reduction rmsnorm), so this is a
// dispatch-count reduction only, no numerical change.
struct Dims { rows: u32, dim: u32, eps: f32, _pad0: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG_SIZE: u32 = 256u;

var<workgroup> partial_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = wg_id.x;
    let valid = row < dims.rows;
    let tid = local_id.x;
    let row_base = row * dims.dim;

    var acc: f32 = 0.0;
    if (valid) {
        var i: u32 = tid;
        loop {
            if (i >= dims.dim) {
                break;
            }
            let v = a[row_base + i] + delta[row_base + i];
            a[row_base + i] = v;
            acc += v * v;
            i = i + WG_SIZE;
        }
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (valid) {
        let denom = sqrt(partial_sums[0] / f32(dims.dim) + dims.eps);
        var j: u32 = tid;
        loop {
            if (j >= dims.dim) {
                break;
            }
            out[row_base + j] = (a[row_base + j] / denom) * scale[j];
            j = j + WG_SIZE;
        }
    }
}
