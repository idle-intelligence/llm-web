// KV cache q8_0 append kernel: quantizes `t` new K or V rows (f32, shape
// [n_kv_heads, t, head_dim] contiguous, batch axis dropped since B=1) into
// the cache's per-layer q8_0 buffers at row offset `dest_offset` (see
// `kv.rs`'s module docs for the on-GPU layout: `scales`
// [n_kv_heads, max_ctx, head_dim/32] f32, `words`
// [n_kv_heads, max_ctx, head_dim/4] u32, one absmax/127 scale + 32 packed
// i8 values (4-per-u32, little-endian) per 32-element block along
// head_dim). One thread per (kv_head, row, block) triple — dispatched as a
// flat 1D grid, `total_blocks = n_kv_heads * t * (head_dim / 32)`.
//
// Session 13 (docs/BENCHMARKS.md): companion to `shader_kv_dequant_range.wgsl`
// (prefill/export path) and `shader_attn_decode_q8.wgsl` (decode attention
// reading these buffers directly, no dequant-to-f32 round trip).

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> words: array<u32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

const WG_SIZE: u32 = 64u;
var<workgroup> wg_info: array<u32, 5>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (local_id.x == 0u) {
        wg_info[0] = info[0];
        wg_info[1] = info[1];
        wg_info[2] = info[2];
        wg_info[3] = info[3];
        wg_info[4] = info[4];
    }
    let li = workgroupUniformLoad(&wg_info);
    let n_kv_heads = li[0];
    let t = li[1];
    let head_dim = li[2];
    let max_ctx = li[3];
    let dest_offset = li[4];

    let blocks_per_row = head_dim / 32u;
    let words_per_row = blocks_per_row * 8u;
    let total_blocks = n_kv_heads * t * blocks_per_row;

    let block_id = gid.x;
    if (block_id >= total_blocks) {
        return;
    }

    let kv_head = block_id / (t * blocks_per_row);
    let rem = block_id % (t * blocks_per_row);
    let row = rem / blocks_per_row;
    let block = rem % blocks_per_row;
    let dest_row = dest_offset + row;

    let input_base = (kv_head * t + row) * head_dim + block * 32u;

    var vals: array<f32, 32>;
    var absmax: f32 = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let x = input[input_base + i];
        vals[i] = x;
        absmax = max(absmax, abs(x));
    }
    let scale = absmax / 127.0;
    let inv_scale = select(0.0, 1.0 / scale, scale != 0.0);

    let scale_idx = kv_head * max_ctx * blocks_per_row + dest_row * blocks_per_row + block;
    scales[scale_idx] = scale;

    let word_base = kv_head * max_ctx * words_per_row + dest_row * words_per_row + block * 8u;
    for (var wi = 0u; wi < 8u; wi = wi + 1u) {
        var word: u32 = 0u;
        for (var bi = 0u; bi < 4u; bi = bi + 1u) {
            let x = vals[wi * 4u + bi];
            let q = i32(round(clamp(x * inv_scale, -127.0, 127.0)));
            let byte = bitcast<u32>(q) & 0xFFu;
            word = word | (byte << (bi * 8u));
        }
        words[word_base + wi] = word;
    }
}
