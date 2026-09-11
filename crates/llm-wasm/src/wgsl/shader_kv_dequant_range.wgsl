// KV cache q8_0 -> f32 range dequant kernel. Reads `kv_len` rows (from
// position 0) of one layer's q8_0 K or V buffers (see
// `shader_kv_quantize.wgsl`'s header for the layout) and writes a
// contiguous f32 `[n_kv_heads, kv_len, head_dim]` tensor (batch axis
// dropped, matches the pre-permute cache tensor layout `model.rs` expects).
// Used by the prefill fallback path (`Q4Attention::forward`'s M>1 branch,
// which still runs the existing Burn-matmul attention on dequantized f32
// K/V) and by `KvCache::export_prefix` in Q8_0 mode. One thread per
// (kv_head, row, block) triple, same grid shape as the quantize kernel.

@group(0) @binding(0) var<storage, read_write> scales: array<f32>;
@group(0) @binding(1) var<storage, read_write> words: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

var<workgroup> wg_info: array<u32, 4>;

fn dequant_byte(byte: u32) -> f32 {
    let sval = bitcast<i32>(byte << 24u) >> 24u;
    return f32(sval);
}

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
    }
    let li = workgroupUniformLoad(&wg_info);
    let n_kv_heads = li[0];
    let kv_len = li[1];
    let head_dim = li[2];
    let max_ctx = li[3];

    let blocks_per_row = head_dim / 32u;
    let words_per_row = blocks_per_row * 8u;
    let total_blocks = n_kv_heads * kv_len * blocks_per_row;

    let block_id = gid.x;
    if (block_id >= total_blocks) {
        return;
    }

    let kv_head = block_id / (kv_len * blocks_per_row);
    let rem = block_id % (kv_len * blocks_per_row);
    let row = rem / blocks_per_row;
    let block = rem % blocks_per_row;

    let scale = scales[kv_head * max_ctx * blocks_per_row + row * blocks_per_row + block];
    let word_base = kv_head * max_ctx * words_per_row + row * words_per_row + block * 8u;
    let out_base = (kv_head * kv_len + row) * head_dim + block * 32u;

    for (var wi = 0u; wi < 8u; wi = wi + 1u) {
        let word = words[word_base + wi];
        output[out_base + wi * 4u + 0u] = dequant_byte(word & 0xFFu) * scale;
        output[out_base + wi * 4u + 1u] = dequant_byte((word >> 8u) & 0xFFu) * scale;
        output[out_base + wi * 4u + 2u] = dequant_byte((word >> 16u) & 0xFFu) * scale;
        output[out_base + wi * 4u + 3u] = dequant_byte((word >> 24u) & 0xFFu) * scale;
    }
}
