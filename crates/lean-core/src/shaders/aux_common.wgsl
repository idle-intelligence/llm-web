// Block metadata shared by the Q4_0-family kernels' Q5_0 and Q4_K modes.
// The Rust side (`engine.rs`, `compose_aux`) puts either `aux_with.wgsl`
// (binding 6 declared) or `aux_none.wgsl` (no binding, so the Q4_0, Q4_1 and
// Q8_0 pipelines keep their original bind group layout) at the marker comment
// of each kernel, followed by this file. It follows the declaration of
// `scales`.
//
// Q5_0: `aux[block]` is the block's 4 high-bit bytes as one u32 (bit j is
// the fifth bit of value j); `scales[block]` is d.
// Q4_K: the qs nibbles are repacked into 32-value blocks laid out like Q4_0's
// (low nibble j -> value j, high nibble j -> value j + 16), so a block is
// 4 u32 of qs as for Q4_0. Per 256-value super-block `sb` (8 blocks),
// `aux[5 * sb ..]` holds d and dmin as two f16 in one u32 (d low), then the
// eight 6-bit scales as bytes in two u32 and the eight 6-bit mins as bytes in
// two u32 (`quant.rs::split_q4_k_blocks` unpacks ggml's `get_scale_min_k4`
// once at load); `scales` is not used.

// (d * sc, -(dmin * m)) of sub-block `s` (0..8) of super-block `sb`: value =
// nibble * x - y as in `dequantize_row_q4_K`, written as nibble * x + y.
fn k4_dm(sb: u32, s: u32) -> vec2<f32> {
    let dm = unpack2x16float(aux_at(5u * sb));
    let sh = 8u * (s & 3u);
    let sc = (aux_at(5u * sb + 1u + (s >> 2u)) >> sh) & 255u;
    let mn = (aux_at(5u * sb + 3u + (s >> 2u)) >> sh) & 255u;
    return vec2<f32>(dm.x * f32(sc), -(dm.y * f32(mn)));
}

// The 4 values (q - 16) of Q5_0 word slot `p` (0..3) of a block: the low
// nibbles of `w` (values 4p..4p+3) or the high nibbles (values 16+4p..), with
// their fifth bits from `qh`.
fn q5_vals(w: u32, hi: bool, p: u32, qh: u32) -> vec4<f32> {
    let sh = select(0u, 4u, hi);
    let nib = vec4<u32>(w >> sh, w >> (sh + 8u), w >> (sh + 16u), w >> (sh + 24u)) & vec4<u32>(15u);
    let h = qh >> (4u * p + select(0u, 16u, hi));
    let bit = vec4<u32>(h, h >> 1u, h >> 2u, h >> 3u) & vec4<u32>(1u);
    return vec4<f32>(nib | (bit << vec4<u32>(4u))) - vec4<f32>(16.0);
}
