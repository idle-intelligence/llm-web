//! GPU residency for Q4_0/Q8_0/Q6_K/F32 matmul weights, ported from
//! `t0-web/crates/t0-fast/src/quant.rs`: the block's f16 scale is decoded to
//! f32 once host-side into its own buffer, and the quantized nibbles/bytes
//! are packed 4/u32 so WGSL reads them as `array<u32>` without a
//! byte-addressed storage buffer. Block math matches `gguf.rs`'s
//! `dequantize_q4_0`/`dequantize_q8_0`/`dequantize_q6_k` exactly - this
//! module only repacks the same on-disk bytes into that GPU-friendly
//! layout, no dequantize-then-requantize round trip.

use crate::engine::Engine;
use crate::gguf::GgmlDtype;

const QK: usize = 32;
/// Q6_K's super-block width (`QK_K` in llama.cpp) - distinct from `QK`
/// (Q4_0/Q8_0's 32-value block), see `gguf.rs::dequantize_q6_k`'s doc
/// comment for the block layout this repacks.
const QK6K: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

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

/// Q6_K's row-chunk (see `QChunk`'s doc comment for the chunking rationale):
/// four buffers instead of two, since a Q6_K block splits into `ql`/`qh`/
/// per-sub-block `scales`/super-block `d` (see `gguf.rs::dequantize_q6_k`),
/// none of which packs into the other three.
pub struct QChunk6K {
    pub ql: wgpu::Buffer,
    pub qh: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub d: wgpu::Buffer,
    pub row_start: u32,
    pub rows: u32,
}

#[allow(non_camel_case_types)]
pub enum MatMulWeight {
    F32 { w: wgpu::Buffer },
    Q8_0 { chunks: Vec<QChunk>, blocks_per_row: u32, out_dim: u32 },
    Q4_0 { chunks: Vec<QChunk>, blocks_per_row: u32, out_dim: u32 },
    Q6_K { chunks: Vec<QChunk6K>, blocks_per_row: u32, out_dim: u32 },
}

impl MatMulWeight {
    pub fn gpu_bytes(&self) -> u64 {
        match self {
            MatMulWeight::F32 { w } => w.size(),
            MatMulWeight::Q8_0 { chunks, .. } | MatMulWeight::Q4_0 { chunks, .. } => {
                chunks.iter().map(|c| c.qs.size() + c.scales.size()).sum()
            }
            MatMulWeight::Q6_K { chunks, .. } => chunks.iter().map(|c| c.ql.size() + c.qh.size() + c.scales.size() + c.d.size()).sum(),
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

/// Splits a Q6_K block stream into GPU-friendly `u32`-packed arrays: `ql`
/// (128 bytes/block -> 32 u32), `qh` (64 bytes/block -> 16 u32) and `scales`
/// (16 signed-i8 bytes/block -> 4 u32, sign-extended in WGSL the same way
/// `linear_q8.wgsl` already does) are each a straight little-endian
/// byte-to-u32 repack (every sub-array is already a multiple of 4 bytes, no
/// nibble/bit reshuffle needed - unlike `split_q4_blocks`, which has to
/// interleave). `d` (the super-block's f16 scale) is decoded to f32 once
/// here, one per block, matching every other quant kind's `scales` buffer.
fn split_q6k_blocks(bytes: &[u8], n_elements: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let n_blocks = n_elements / QK6K;
    let mut ql = vec![0u32; n_blocks * 32];
    let mut qh = vec![0u32; n_blocks * 16];
    let mut scales = vec![0u32; n_blocks * 4];
    let mut d = vec![0f32; n_blocks];
    for (bi, block) in bytes.chunks_exact(Q6K_BLOCK_BYTES).enumerate() {
        let ql_bytes = &block[0..128];
        let qh_bytes = &block[128..192];
        let sc_bytes = &block[192..208];
        d[bi] = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        for i in 0..32 {
            ql[bi * 32 + i] = u32::from_le_bytes([ql_bytes[4 * i], ql_bytes[4 * i + 1], ql_bytes[4 * i + 2], ql_bytes[4 * i + 3]]);
        }
        for i in 0..16 {
            qh[bi * 16 + i] = u32::from_le_bytes([qh_bytes[4 * i], qh_bytes[4 * i + 1], qh_bytes[4 * i + 2], qh_bytes[4 * i + 3]]);
        }
        for i in 0..4 {
            scales[bi * 4 + i] = u32::from_le_bytes([sc_bytes[4 * i], sc_bytes[4 * i + 1], sc_bytes[4 * i + 2], sc_bytes[4 * i + 3]]);
        }
    }
    (ql, qh, scales, d)
}

/// Q6_K counterpart of `chunk_rows`: same row-aligned chunking under the
/// device's `max_storage_buffer_binding_size`, but uploads the four
/// `split_q6k_blocks` arrays per chunk instead of one `qs`/`scales` pair.
fn chunk_rows_q6k(engine: &Engine, label: &str, bytes: &[u8], out_dim: usize, blocks_per_row: usize, bytes_per_row: usize) -> Vec<QChunk6K> {
    let limit = engine.max_storage_buffer_binding_size() as usize;
    let rows_per_chunk = (limit / bytes_per_row).clamp(1, out_dim);
    let mut chunks = Vec::with_capacity(out_dim.div_ceil(rows_per_chunk));
    let mut row_start = 0usize;
    while row_start < out_dim {
        let rows = rows_per_chunk.min(out_dim - row_start);
        let byte_start = row_start * bytes_per_row;
        let byte_end = byte_start + rows * bytes_per_row;
        let n_elements = rows * blocks_per_row * QK6K;
        let (ql, qh, scales, d) = split_q6k_blocks(&bytes[byte_start..byte_end], n_elements);
        chunks.push(QChunk6K {
            ql: engine.buf_u32(&ql, &format!("{label}.ql[{row_start}]")),
            qh: engine.buf_u32(&qh, &format!("{label}.qh[{row_start}]")),
            scales: engine.buf_u32(&scales, &format!("{label}.scales[{row_start}]")),
            d: engine.buf_f32(&d, &format!("{label}.d[{row_start}]")),
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
        GgmlDtype::Q6_K => {
            let blocks_per_row = in_dim / QK6K;
            let chunks = chunk_rows_q6k(engine, label, bytes, out_dim, blocks_per_row, blocks_per_row * Q6K_BLOCK_BYTES);
            MatMulWeight::Q6_K { chunks, blocks_per_row: blocks_per_row as u32, out_dim: out_dim as u32 }
        }
        // Q4_1 is dequantized host-side straight to F32 rather than given
        // its own GPU-resident block kernel (see `gguf.rs::dequantize_q4_1`'s
        // doc comment): only a handful of tensors (some of SmolLM2's
        // `ffn_down.weight`s) are ever Q4_1, not worth a third quantized
        // matmul kernel for.
        GgmlDtype::F32 | GgmlDtype::F16 | GgmlDtype::Q4_1 => {
            let data = crate::gguf::dequantize_for(dtype, bytes, n_elements);
            MatMulWeight::F32 { w: engine.buf_f32(&data, label) }
        }
    }
}

/// Q4_0/Q8_0-resident buffers for the embedding-gather kernel
/// (`token_embd.weight`), which needs `qs`/`scales` directly rather than
/// through `MatMulWeight`'s linear-kernel bind group. One struct shared by
/// both quant kinds - `EmbeddingTable` picks the WGSL pipeline
/// (`embed_gather_q4`/`embed_gather_q8`) at the call site in `model.rs`.
pub struct QEmbeddingTable {
    pub qs: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub blocks_per_row: u32,
    pub hidden: u32,
    pub vocab: u32,
}

/// Q6_K counterpart of `QEmbeddingTable` - four buffers instead of two, same
/// reason as `QChunk6K`.
pub struct QEmbeddingTable6K {
    pub ql: wgpu::Buffer,
    pub qh: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub d: wgpu::Buffer,
    pub blocks_per_row: u32,
    pub hidden: u32,
    pub vocab: u32,
}

#[allow(non_camel_case_types)]
pub enum EmbeddingTable {
    Q4_0(QEmbeddingTable),
    Q8_0(QEmbeddingTable),
    Q6_K(QEmbeddingTable6K),
}

impl EmbeddingTable {
    /// Only meaningful for the Q4_0/Q8_0 variants, which share one struct
    /// shape; Q6_K's four-buffer table has no equivalent accessor (its one
    /// call site in `model.rs::embed_gather` matches on `EmbeddingTable`
    /// directly).
    pub fn table(&self) -> &QEmbeddingTable {
        match self {
            EmbeddingTable::Q4_0(t) | EmbeddingTable::Q8_0(t) => t,
            EmbeddingTable::Q6_K(_) => panic!("EmbeddingTable::table: Q6_K has no QEmbeddingTable - match on EmbeddingTable directly"),
        }
    }
}

/// Not chunked like `load_matmul_weight_gguf`'s Q4_0/Q8_0 weights: every
/// model this crate targets so far has a `token_embd.weight` that fits in
/// one binding under every WebGPU adapter this crate targets. If a larger
/// vocab/hidden size ever pushes it over a device's
/// `max_storage_buffer_binding_size`, `embed_gather_q4.wgsl`/
/// `embed_gather_q8.wgsl` would need the same chunk-dispatch treatment
/// `linear()` gets for `MatMulWeight`.
pub fn load_embedding_table_gguf(engine: &Engine, label: &str, shape: &[usize], dtype: GgmlDtype, bytes: &[u8]) -> EmbeddingTable {
    let vocab = shape[0];
    let hidden = shape[1];
    let n_elements = vocab * hidden;
    match dtype {
        GgmlDtype::Q4_0 => {
            let (qs, scales) = split_q4_blocks(bytes, n_elements);
            EmbeddingTable::Q4_0(QEmbeddingTable {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (hidden / QK) as u32,
                hidden: hidden as u32,
                vocab: vocab as u32,
            })
        }
        GgmlDtype::Q8_0 => {
            let (qs, scales) = split_q8_blocks(bytes, n_elements);
            EmbeddingTable::Q8_0(QEmbeddingTable {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (hidden / QK) as u32,
                hidden: hidden as u32,
                vocab: vocab as u32,
            })
        }
        GgmlDtype::Q6_K => {
            let (ql, qh, scales, d) = split_q6k_blocks(bytes, n_elements);
            EmbeddingTable::Q6_K(QEmbeddingTable6K {
                ql: engine.buf_u32(&ql, &format!("{label}.ql")),
                qh: engine.buf_u32(&qh, &format!("{label}.qh")),
                scales: engine.buf_u32(&scales, &format!("{label}.scales")),
                d: engine.buf_f32(&d, &format!("{label}.d")),
                blocks_per_row: (hidden / QK6K) as u32,
                hidden: hidden as u32,
                vocab: vocab as u32,
            })
        }
        other => panic!("load_embedding_table_gguf: unsupported embedding dtype {other:?} (only Q4_0/Q8_0/Q6_K embedding tables are implemented)"),
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

    #[test]
    #[allow(clippy::needless_range_loop)] // `blk` indexes several parallel arrays with different offsets, not just one
    fn q6k_split_matches_reference_dequant() {
        // Two synthetic Q6_K super-blocks (2 * 256 = 512 elements): check the
        // GPU-layout repack (`split_q6k_blocks`) decodes, via the same index
        // math `shaders/linear_q6k.wgsl` uses, to the same values as
        // `gguf::dequantize_q6_k`.
        use crate::gguf::dequantize_q6_k;
        let mut bytes = vec![0u8; 420];
        for (bi, block) in bytes.chunks_exact_mut(210).enumerate() {
            for (i, b) in block[0..192].iter_mut().enumerate() {
                *b = ((i * 13 + bi * 37) % 256) as u8;
            }
            for (i, b) in block[192..208].iter_mut().enumerate() {
                *b = ((i as i32 * 17 + bi as i32 * 11) % 256 - 128) as i8 as u8;
            }
            let d = half::f16::from_f32(0.02 + bi as f32 * 0.01);
            block[208..210].copy_from_slice(&d.to_le_bytes());
        }
        let deq = dequantize_q6_k(&bytes, 512);
        let (ql, qh, scales, dscale) = split_q6k_blocks(&bytes, 512);

        let sc_i8 = |word: u32, i: u32| -> f32 {
            let b = (word >> (i * 8)) & 0xFF;
            ((b as i32) << 24 >> 24) as f32
        };
        let byte_of = |word: u32, i: u32| (word >> (i * 8)) & 0xFF;

        let mut out = vec![0f32; 512];
        for blk in 0..2usize {
            let dval = dscale[blk];
            let ql_base = blk * 32;
            let qh_base = blk * 16;
            let sc_base = blk * 4;
            let y_base = blk * 256;
            for half in 0..2u32 {
                for l in 0..32u32 {
                    let is = l / 16;
                    let ql_w0 = ql[ql_base + (half * 16 + l / 4) as usize];
                    let ql_b0 = byte_of(ql_w0, l % 4);
                    let idx1 = l + 32;
                    let ql_w1 = ql[ql_base + (half * 16 + idx1 / 4) as usize];
                    let ql_b1 = byte_of(ql_w1, idx1 % 4);
                    let qh_w = qh[qh_base + (half * 8 + l / 4) as usize];
                    let qh_b = byte_of(qh_w, l % 4);

                    let entry = |off: u32| -> f32 {
                        let e = half * 8 + is + off;
                        sc_i8(scales[sc_base + (e / 4) as usize], e % 4)
                    };
                    let sc0 = entry(0);
                    let sc2 = entry(2);
                    let sc4 = entry(4);
                    let sc6 = entry(6);

                    let q1 = ((ql_b0 & 0xF) | ((qh_b & 3) << 4)) as i32 - 32;
                    let q2 = ((ql_b1 & 0xF) | (((qh_b >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql_b0 >> 4) | (((qh_b >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql_b1 >> 4) | (((qh_b >> 6) & 3) << 4)) as i32 - 32;

                    let yb = y_base + (half * 128 + l) as usize;
                    out[yb] = dval * sc0 * q1 as f32;
                    out[yb + 32] = dval * sc2 * q2 as f32;
                    out[yb + 64] = dval * sc4 * q3 as f32;
                    out[yb + 96] = dval * sc6 * q4 as f32;
                }
            }
        }
        for i in 0..512 {
            assert!((deq[i] - out[i]).abs() < 1e-4, "i={i}: deq={} out={}", deq[i], out[i]);
        }
    }
}
