// F2 (docs/BENCHMARKS.md Session 9): fused `silu(gate) * up`, one dispatch
// over the flattened [M, ffn_dim] elementwise op instead of Burn's separate
// `silu` (sigmoid + mul, ~2 dispatches) then `mul` (1 dispatch) chain — see
// model.rs's `Q4FeedForward::forward`. Purely elementwise (thread `idx`
// only ever reads/writes its own element), so output aliases the `gate`
// buffer directly (in-place, no separate allocation) with no
// read-before-write hazard and no barriers — the early `return` guards
// work only. info: [n, row_width] (n = M * ffn_dim) packed as f32.
//
// Dispatch: WebGPU caps every dispatch dimension at 65535 workgroups (a
// separate limit from the 256-invocation-per-workgroup cap) — at prefill
// this kernel's flat `M*ffn_dim` work exceeds that in one 1D dispatch, so
// the launch is 2D (`gguf.rs::workgroups_2d`) and the flat element index is
// recovered as `gid.y * row_width + gid.x`, `row_width` (== `wg_x * 256`)
// passed via `info[1]`.
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> info: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(info[0]);
    let row_width = u32(info[1]);
    let idx = gid.y * row_width + gid.x;
    if (idx >= n) {
        return;
    }
    let g = gate[idx];
    let silu_g = g / (1.0 + exp(-g));
    gate[idx] = silu_g * up[idx];
}
