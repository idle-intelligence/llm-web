// One workgroup per row, 256-thread shared-memory tree reduction for the
// sum-of-squares. Reused from this project's own crates/llm-wasm/src/wgsl/
// shader_rmsnorm.wgsl (same "one workgroup per row" design already proven
// there), adapted to lean's binding layout: read-only x/scale storage
// buffers and a `Dims` uniform struct instead of llm-wasm's four
// read_write storage buffers with a flat `info: array<f32>`.
// out[r, :] = x[r, :] / sqrt(mean(x[r,:]^2) + eps) * scale[:], dim = `dim`.
//
// Replaces the previous one-thread-per-row design (a verbatim port of
// t0-web/crates/t0-fast/src/shaders/rmsnorm_full.wgsl), which left every
// GPU thread but one idle at decode's `rows == 1` case, serially looping
// `2 * dim` times alone (see docs/runs/2026-09-29-lean-vs-llamacpp-profile.md:
// 40-63% of decode time on this kernel before this change).
//
// `dims` is a `uniform` buffer here (unlike llm-wasm's storage-buffer
// `info`), so `dims.rows`/`dims.dim` are already uniform values per WGSL's
// uniformity analysis - no `workgroupUniformLoad` staging step is needed to
// safely branch around the `workgroupBarrier()` calls below (that step in
// llm-wasm's version works around Tint tainting *storage*-buffer loads as
// non-uniform; it does not apply to `uniform` address space loads).
struct Dims { rows: u32, dim: u32, eps: f32, _pad0: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

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
            let v = x[row_base + i];
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
            out[row_base + j] = (x[row_base + j] / denom) * scale[j];
            j = j + WG_SIZE;
        }
    }
}
