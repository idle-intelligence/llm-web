//! GPU residency for Q4_0/Q8_0/F32 matmul weights, ported from
//! `t0-web/crates/t0-fast/src/quant.rs`: the block's f16 scale is decoded to
//! f32 once host-side into its own buffer, and the quantized nibbles/bytes
//! are packed 4/u32 so WGSL reads them as `array<u32>` without a
//! byte-addressed storage buffer. Block math matches `gguf.rs`'s
//! `dequantize_q4_0`/`dequantize_q8_0` exactly - this module only repacks
//! the same on-disk bytes into that GPU-friendly layout, no
//! dequantize-then-requantize round trip.

use crate::engine::Engine;
use crate::gguf::GgmlDtype;

const QK: usize = 32;

/// One row-range of a matmul weight, GPU-resident. `row_start`/`rows` are in
/// units of the weight's `out_dim` (GGUF/PyTorch `shape[0]`) - the same axis
/// `linear()`'s output columns index. A weight that fits under the device's
/// `max_storage_buffer_binding_size` in one binding gets exactly one chunk
/// covering `0..out_dim`.
pub struct QChunk {
    pub qs: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub row_start: u32,
    pub rows: u32,
}

pub enum MatMulWeight {
    F32 { w: wgpu::Buffer },
    Q8_0 { chunks: Vec<QChunk>, blocks_per_row: u32, out_dim: u32 },
    Q4_0 { chunks: Vec<QChunk>, blocks_per_row: u32, out_dim: u32 },
}

impl MatMulWeight {
    pub fn gpu_bytes(&self) -> u64 {
        match self {
            MatMulWeight::F32 { w } => w.size(),
            MatMulWeight::Q8_0 { chunks, .. } | MatMulWeight::Q4_0 { chunks, .. } => {
                chunks.iter().map(|c| c.qs.size() + c.scales.size()).sum()
            }
        }
    }
}

fn split_q8_blocks(bytes: &[u8], n_elements: usize) -> (Vec<u32>, Vec<f32>) {
    let n_blocks = n_elements / QK;
    let mut qs = vec![0u32; n_elements / 4];
    let mut scales = vec![0f32; n_blocks];
    for (bi, block) in bytes.chunks_exact(34).enumerate() {
        scales[bi] = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..QK {
            let byte = block[2 + j] as u32;
            let word_idx = bi * (QK / 4) + j / 4;
            let shift = (j % 4) * 8;
            qs[word_idx] |= byte << shift;
        }
    }
    (qs, scales)
}

fn split_q4_blocks(bytes: &[u8], n_elements: usize) -> (Vec<u32>, Vec<f32>) {
    let n_blocks = n_elements / QK;
    let mut qs = vec![0u32; n_blocks * 4]; // 16 bytes/block = 4 u32/block
    let mut scales = vec![0f32; n_blocks];
    for (bi, block) in bytes.chunks_exact(18).enumerate() {
        scales[bi] = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for byte_i in 0..QK / 2 {
            let byte = block[2 + byte_i] as u32;
            let word_idx = bi * 4 + byte_i / 4;
            let shift = (byte_i % 4) * 8;
            qs[word_idx] |= byte << shift;
        }
    }
    (qs, scales)
}

/// Splits `bytes` (row-major, block-contiguous Q4_0/Q8_0 data for a
/// `[out_dim, in_dim]` weight) into row-aligned chunks no larger than the
/// device's `max_storage_buffer_binding_size`, uploading each chunk's `qs`/
/// `scales` as its own pair of buffers. General mechanism (not special-cased
/// to any one tensor) - a weight that already fits in one binding gets
/// exactly one chunk, identical to the pre-chunking layout.
fn chunk_rows(engine: &Engine, label: &str, bytes: &[u8], out_dim: usize, blocks_per_row: usize, bytes_per_row: usize, is_q8: bool) -> Vec<QChunk> {
    let limit = engine.max_storage_buffer_binding_size() as usize;
    let rows_per_chunk = (limit / bytes_per_row).clamp(1, out_dim);
    let mut chunks = Vec::with_capacity(out_dim.div_ceil(rows_per_chunk));
    let mut row_start = 0usize;
    while row_start < out_dim {
        let rows = rows_per_chunk.min(out_dim - row_start);
        let byte_start = row_start * bytes_per_row;
        let byte_end = byte_start + rows * bytes_per_row;
        let n_elements = rows * blocks_per_row * QK;
        let (qs, scales) = if is_q8 {
            split_q8_blocks(&bytes[byte_start..byte_end], n_elements)
        } else {
            split_q4_blocks(&bytes[byte_start..byte_end], n_elements)
        };
        chunks.push(QChunk {
            qs: engine.buf_u32(&qs, &format!("{label}.qs[{row_start}]")),
            scales: engine.buf_f32(&scales, &format!("{label}.scales[{row_start}]")),
            row_start: row_start as u32,
            rows: rows as u32,
        });
        row_start += rows;
    }
    chunks
}

/// Loads one matmul weight (`shape = [out_dim, in_dim]`, GGUF/PyTorch
/// convention) straight from GGUF bytes at whatever residency it already
/// has on disk: Q4_0/Q8_0 blocks go straight into the split_*_blocks repack
/// (no requantize), F32/F16 dequantizes to f32. `bytes` is consumed and can
/// be dropped by the caller immediately after this call returns. Q4_0/Q8_0
/// weights are split into row chunks by `chunk_rows` when they'd otherwise
/// exceed the device's single-binding size limit (see that fn's doc
/// comment); `linear()` in `model.rs` dispatches once per chunk.
pub fn load_matmul_weight_gguf(engine: &Engine, label: &str, shape: &[usize], dtype: GgmlDtype, bytes: &[u8]) -> MatMulWeight {
    let out_dim = shape[0];
    let in_dim = shape[1];
    let n_elements: usize = shape.iter().product();
    match dtype {
        GgmlDtype::Q8_0 => {
            let blocks_per_row = in_dim / QK;
            let chunks = chunk_rows(engine, label, bytes, out_dim, blocks_per_row, blocks_per_row * 34, true);
            MatMulWeight::Q8_0 { chunks, blocks_per_row: blocks_per_row as u32, out_dim: out_dim as u32 }
        }
        GgmlDtype::Q4_0 => {
            let blocks_per_row = in_dim / QK;
            let chunks = chunk_rows(engine, label, bytes, out_dim, blocks_per_row, blocks_per_row * 18, false);
            MatMulWeight::Q4_0 { chunks, blocks_per_row: blocks_per_row as u32, out_dim: out_dim as u32 }
        }
        GgmlDtype::F32 | GgmlDtype::F16 => {
            let data = crate::gguf::dequantize_for(dtype, bytes, n_elements);
            MatMulWeight::F32 { w: engine.buf_f32(&data, label) }
        }
    }
}

/// Same Q4_0-resident buffers, but exposed for the embedding-gather kernel
/// (`token_embd.weight`, tied to the lm head), which needs `qs`/`scales`
/// directly rather than through `MatMulWeight`'s linear-kernel bind group.
pub struct Q4EmbeddingTable {
    pub qs: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub blocks_per_row: u32,
    pub hidden: u32,
    pub vocab: u32,
}

/// Not chunked like `load_matmul_weight_gguf`'s Q4_0/Q8_0 weights: this
/// model's `token_embd.weight` (~68MB packed) fits in one binding under
/// every WebGPU adapter this crate targets. If a larger vocab/hidden size
/// ever pushes it over a device's `max_storage_buffer_binding_size`,
/// `embed_gather_q4.wgsl` would need the same chunk-dispatch treatment
/// `linear()` gets for `MatMulWeight`.
pub fn load_q4_embedding_gguf(engine: &Engine, label: &str, shape: &[usize], bytes: &[u8]) -> Q4EmbeddingTable {
    let vocab = shape[0];
    let hidden = shape[1];
    let n_elements = vocab * hidden;
    let (qs, scales) = split_q4_blocks(bytes, n_elements);
    Q4EmbeddingTable {
        qs: engine.buf_u32(&qs, &format!("{label}.qs")),
        scales: engine.buf_f32(&scales, &format!("{label}.scales")),
        blocks_per_row: (hidden / QK) as u32,
        hidden: hidden as u32,
        vocab: vocab as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::dequantize_q4_0;

    #[test]
    fn q4_split_matches_reference_dequant() {
        // Build a synthetic Q4_0 block stream (2 blocks = 64 elements) and
        // check the GPU-layout repack decodes to the same values as
        // `gguf::dequantize_q4_0`.
        let mut bytes = vec![0u8; 36];
        for (bi, block) in bytes.chunks_exact_mut(18).enumerate() {
            let scale = half::f16::from_f32(0.1 + bi as f32 * 0.05);
            block[0..2].copy_from_slice(&scale.to_le_bytes());
            for (i, b) in block[2..18].iter_mut().enumerate() {
                *b = ((i * 7 + bi * 3) % 256) as u8;
            }
        }
        let deq = dequantize_q4_0(&bytes, 64);
        let (qs, scales) = split_q4_blocks(&bytes, 64);

        let mut out = vec![0f32; 64];
        for (col, o) in out.iter_mut().enumerate() {
            let blk = col / 32;
            let j = col % 32;
            let word_idx = blk * 4 + (j % 16) / 4;
            let shift = ((j % 16) % 4) * 8;
            let word = qs[word_idx];
            let byte = (word >> shift) & 0xFF;
            let nibble = if j < 16 { byte & 0xF } else { byte >> 4 };
            *o = (nibble as f32 - 8.0) * scales[blk];
        }
        for i in 0..64 {
            assert!((deq[i] - out[i]).abs() < 1e-6, "i={i}: deq={} out={}", deq[i], out[i]);
        }
    }
}
