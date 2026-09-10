// Rewritten from the stt-web copy (see git history) for Qwen2/xLAM-2-3b-fc-r's
// GGUF layout. GGUF metadata KV pairs are now parsed generically (not
// skipped) so `qwen2.*` hyperparameters can be read from the file instead of
// hand-copied; tensor names follow llama.cpp's `blk.N.attn_{q,k,v,output}`
// convention (docs/MODELS.md §2). See docs/ENGINE.md §1 for the audit this
// replaces.

//! Q4 GGUF weight loader and WGSL dequantization shaders.
//!
//! Pipeline: GGUF file → parse header/metadata/tensors → store Q4 blocks as
//! raw bytes on GPU → dequantize via WGSL compute shader → matmul.
//!
//! - `ShardedCursor`: Read+Seek over Vec<Vec<u8>> for multi-shard GGUF (stays under 2GB allocation limit)
//! - Two-phase loading: parse GGUF, drop reader, finalize tensors (stays under 4GB address space)
//! - Naive WGSL kernel for WASM (tiled kernel is native-only, not yet written)
//!
//! Embedding-lookup strategy (see docs/ENGINE.md §1's 1.16GB-dequant warning):
//! `token_embd.weight` ([151936, 2048], Q4_0, tied to the lm_head — no
//! `output.weight` tensor exists) is kept as raw Q4_0 bytes on **both** the
//! CPU (`EmbeddingStore`, for cheap per-token row dequant at input time —
//! 2048 values, ~64 blocks, negligible cost even at M=8000+ prefill) and the
//! GPU (`Q4Tensor`, reused directly as the lm_head's matmul weight through
//! the same kernel that serves every other linear layer). This avoids ever
//! materializing the full 151936×2048 table as F32 (which would cost 1.2GB),
//! unlike stt-wasm's `dequant_embedding_to_gpu` (appropriate there only
//! because its vocab was 8001, ~62MB) — deliberately *not* reusing that
//! pattern here.

use anyhow::{bail, ensure, Context, Result};
use burn::backend::wgpu::{
    into_contiguous, AutoCompiler, CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate,
    WgpuDevice, WgpuRuntime,
};
use burn::backend::Wgpu;
use burn::module::{Param, ParamId};
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use byteorder::{LittleEndian, ReadBytesExt};
use cubecl::prelude::KernelId;
use cubecl::server::{Bindings, CubeCount, Handle};
use cubecl::{CubeTask, Runtime};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{
    LlmModel, Q4Attention, Q4FeedForward, Q4TransformerBlock, RmsNormLayer, RoPE,
};
use crate::LlmConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const ALIGNMENT: u64 = 32;

// Naive kernel workgroup sizes (16×16 = 256, the WebGPU limit)
const NAIVE_WG_X: u32 = 16;
const NAIVE_WG_Y: u32 = 16;

/// WebGPU caps every `CubeCount`/dispatch dimension at 65535 workgroups
/// (not just the 256-invocation-per-workgroup cap — a *separate* limit on
/// dispatch group *count*). F1/F2's elementwise kernels (`rope_fused`,
/// `silu_mul_fused`) flatten their whole `[rows, cols]` work into one 1D
/// grid of `wg_size`-wide workgroups; at prefill sequence lengths in the
/// low thousands that 1D count blows past 65535 (e.g. SiLU*up at T=2225,
/// ffn_dim=11008: `2225*11008/256 ≈ 95674` workgroups — the bug this
/// helper fixes, caught by `full_forward`'s prefill tests). Splits the flat
/// 1D workgroup count into a 2D `(x, y)` grid with `x <= 65535`; the
/// shader recovers the flat element index as
/// `gid.y * (wg_x * wg_size) + gid.x` (see `shader_rope.wgsl`/
/// `shader_silu_mul.wgsl`'s `row_width` info field). Returns
/// `(wg_x, wg_y, row_width_elements)`.
fn workgroups_2d(elements: usize, wg_size: u32) -> (u32, u32, u32) {
    const MAX_WG_DIM: u32 = 65535;
    let total_wg = (elements as u32).div_ceil(wg_size);
    let wg_x = total_wg.clamp(1, MAX_WG_DIM);
    let wg_y = total_wg.div_ceil(wg_x).max(1);
    (wg_x, wg_y, wg_x * wg_size)
}

// Q4_0 matvec (M=1 decode) cooperative kernel: WG_SIZE=256, ROWS_PER_WG=8 —
// see wgsl/shader_q4_matvec.wgsl's header comment. K1, unused by default
// dispatch since Session 6 (see MATVEC_COALESCED_ROWS_PER_WG below).
const MATVEC_ROWS_PER_WG: usize = 8;

// Session 6 coalesced matvec: WG_SIZE=128, ROWS_PER_WG=4 — see
// wgsl/shader_q4_matvec_coalesced.wgsl's header comment.
const MATVEC_COALESCED_ROWS_PER_WG: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (f16) bits to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let mut e = 1u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = 127u32.wrapping_sub(15 + e - 1);
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        // Normalized
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

/// Reverse GGUF dimension order to get PyTorch convention.
///
/// GGUF stores dimensions `ne[]` with `ne[0]` fastest-varying (innermost),
/// while PyTorch/`Q4Tensor` use `[out_features, in_features]` convention.
fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

// ---------------------------------------------------------------------------
// GGUF String / Value helpers
// ---------------------------------------------------------------------------

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

/// A parsed GGUF metadata value. Arrays are not materialized (we never need
/// array *contents* for config extraction — vocab size is instead read off
/// `token_embd.weight`'s tensor shape) but their length/element-type are kept
/// so `llm-agent gguf-info` can report them.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array { elem_type: u32, len: u64 },
}

fn read_gguf_scalar<R: Read>(reader: &mut R, value_type: u32) -> Result<GgufValue> {
    Ok(match value_type {
        0 => GgufValue::U8(reader.read_u8()?),
        1 => GgufValue::I8(reader.read_i8()?),
        2 => GgufValue::U16(reader.read_u16::<LittleEndian>()?),
        3 => GgufValue::I16(reader.read_i16::<LittleEndian>()?),
        4 => GgufValue::U32(reader.read_u32::<LittleEndian>()?),
        5 => GgufValue::I32(reader.read_i32::<LittleEndian>()?),
        6 => GgufValue::F32(reader.read_f32::<LittleEndian>()?),
        7 => GgufValue::Bool(reader.read_u8()? != 0),
        8 => GgufValue::String(read_gguf_string(reader)?),
        10 => GgufValue::U64(reader.read_u64::<LittleEndian>()?),
        11 => GgufValue::I64(reader.read_i64::<LittleEndian>()?),
        12 => GgufValue::F64(reader.read_f64::<LittleEndian>()?),
        other => bail!("Unexpected scalar GGUF value type: {other}"),
    })
}

/// Skip (without allocating) the contents of a GGUF value; used for array
/// elements, whose contents we don't need (tokenizer.ggml.tokens etc.).
fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<()> {
    match value_type {
        0 | 1 | 7 => {
            reader.seek(SeekFrom::Current(1))?;
        }
        2 | 3 => {
            reader.seek(SeekFrom::Current(2))?;
        }
        4..=6 => {
            reader.seek(SeekFrom::Current(4))?;
        }
        8 => {
            let _ = read_gguf_string(reader)?;
        }
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_gguf_value(reader, elem_type)?;
            }
        }
        10..=12 => {
            reader.seek(SeekFrom::Current(8))?;
        }
        other => bail!("Unknown GGUF metadata value type: {other}"),
    }
    Ok(())
}

/// Read a top-level GGUF metadata value (scalar or array).
fn read_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<GgufValue> {
    if value_type == 9 {
        let elem_type = reader.read_u32::<LittleEndian>()?;
        let count = reader.read_u64::<LittleEndian>()?;
        for _ in 0..count {
            skip_gguf_value(reader, elem_type)?;
        }
        Ok(GgufValue::Array {
            elem_type,
            len: count,
        })
    } else {
        read_gguf_scalar(reader, value_type)
    }
}

// ---------------------------------------------------------------------------
// GgmlDtype
// ---------------------------------------------------------------------------

/// GGML data type codes used in GGUF tensor descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    F32,
    F16,
    Q4_0,
}

impl GgmlDtype {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            other => bail!("Unsupported GGML dtype code: {other}"),
        }
    }

    pub fn byte_size(&self, num_elements: u64) -> Result<u64> {
        match self {
            Self::F32 => num_elements.checked_mul(4),
            Self::F16 => num_elements.checked_mul(2),
            Self::Q4_0 => {
                let num_blocks = num_elements / 32;
                num_blocks.checked_mul(18)
            }
        }
        .context("tensor size overflow")
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    dimensions: Vec<u64>,
    dtype: GgmlDtype,
    offset: u64,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    pub fn num_elements(&self) -> Result<u64> {
        self.dimensions.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d)
                .with_context(|| format!("Tensor '{}': element count overflow", self.name))
        })
    }

    pub fn byte_size(&self) -> Result<u64> {
        self.dtype.byte_size(self.num_elements()?)
    }
}

// ---------------------------------------------------------------------------
// GgufReader
// ---------------------------------------------------------------------------

/// A reader for GGUF v2/v3 files.
pub struct GgufReader<R: Read + Seek> {
    reader: R,
    version: u32,
    tensor_count: u64,
    tensors: HashMap<String, GgufTensorInfo>,
    /// Ordered tensor names, in on-disk order (for `gguf-info` printing).
    tensor_order: Vec<String>,
    metadata: HashMap<String, GgufValue>,
    data_section_offset: u64,
}

/// Cap on `with_capacity` hints derived from untrusted GGUF header counts
/// (`metadata_kv_count`, `tensor_count`, per-tensor `ndims`) — a malformed
/// or adversarial file must not be able to force a huge up-front
/// allocation before any of the counted items are actually read.
const MAX_CAPACITY_HINT: usize = 1 << 16;

impl<R: Read + Seek> GgufReader<R> {
    /// Parse a GGUF file from the given reader.
    pub fn open(mut reader: R) -> Result<Self> {
        let file_len = reader
            .seek(SeekFrom::End(0))
            .context("Failed to seek to end of GGUF")?;
        reader
            .seek(SeekFrom::Start(0))
            .context("Failed to seek to start of GGUF")?;

        let magic = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF magic")?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})");
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF version")?;
        if version != 2 && version != 3 {
            bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
        }

        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read tensor count")?;
        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read metadata KV count")?;

        // Parse metadata key-value pairs generically (not skipped — Qwen2
        // hyperparameters come from here rather than being hand-copied).
        let mut metadata =
            HashMap::with_capacity((metadata_kv_count as usize).min(MAX_CAPACITY_HINT));
        for i in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read metadata key {i}"))?;
            let value_type = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read metadata value type {i}"))?;
            let value = read_gguf_value(&mut reader, value_type)
                .with_context(|| format!("Failed to read metadata value {i} ('{key}')"))?;
            metadata.insert(key, value);
        }

        // Parse tensor index
        let mut tensors =
            HashMap::with_capacity((tensor_count as usize).min(MAX_CAPACITY_HINT));
        let mut tensor_order = Vec::with_capacity((tensor_count as usize).min(MAX_CAPACITY_HINT));
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read tensor name {i}"))?;
            let ndims = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read ndims for tensor {i}"))?;
            ensure!(
                ndims <= 4,
                "Tensor {i} ('{name}'): ndims={ndims} exceeds supported maximum of 4"
            );
            let mut dimensions = Vec::with_capacity(ndims as usize);
            for d in 0..ndims {
                dimensions.push(
                    reader
                        .read_u64::<LittleEndian>()
                        .with_context(|| format!("Failed to read dim {d} for tensor {i}"))?,
                );
            }
            let dtype = GgmlDtype::from_u32(
                reader
                    .read_u32::<LittleEndian>()
                    .with_context(|| format!("Failed to read dtype for tensor {i}"))?,
            )?;
            let offset = reader
                .read_u64::<LittleEndian>()
                .with_context(|| format!("Failed to read offset for tensor {i}"))?;
            ensure!(
                offset <= file_len,
                "Tensor {i} ('{name}'): offset {offset} exceeds file length {file_len}"
            );

            tensor_order.push(name.clone());
            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    dtype,
                    offset,
                },
            );
        }

        let current_pos = reader.stream_position()?;
        let data_section_offset = align_up(current_pos, ALIGNMENT);

        // Validate every tensor's data range lies within the file before any
        // caller can slice it (`tensor_data`/`load_q4_linear` etc.).
        for info in tensors.values() {
            let byte_size = info.byte_size()?;
            let abs_offset = data_section_offset
                .checked_add(info.offset)
                .with_context(|| format!("Tensor '{}': offset overflow", info.name))?;
            let end = abs_offset
                .checked_add(byte_size)
                .with_context(|| format!("Tensor '{}': data range overflow", info.name))?;
            ensure!(
                end <= file_len,
                "Tensor '{}': data range [{abs_offset}, {end}) exceeds file length {file_len}",
                info.name
            );
        }

        Ok(Self {
            reader,
            version,
            tensor_count,
            tensors,
            tensor_order,
            metadata,
            data_section_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_order
    }

    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    /// Read a metadata value as u32, widening from any integer variant.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U8(v) => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I8(v) => Some(*v as u32),
            GgufValue::I16(v) => Some(*v as u32),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Read a metadata value as f32, widening from f32/f64.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn meta_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Read raw tensor data bytes from the file.
    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("Tensor '{name}' not found in GGUF"))?
            .clone();
        let byte_size = usize::try_from(info.byte_size()?)
            .with_context(|| format!("Tensor '{name}': byte size does not fit in usize"))?;
        let abs_offset = self.data_section_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; byte_size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Config extraction — qwen2.* metadata -> LlmConfig
// ---------------------------------------------------------------------------

/// Build an [`LlmConfig`] from a GGUF's `qwen2.*` metadata (docs/MODELS.md §2).
///
/// Vocab size is read from `token_embd.weight`'s tensor shape rather than any
/// metadata array length, since the tokenizer's own vocab (151665) is smaller
/// than the padded embedding matrix (151936) — see docs/MODELS.md §1.
pub fn config_from_gguf<R: Read + Seek>(reader: &GgufReader<R>) -> Result<LlmConfig> {
    let arch = reader.meta_string("general.architecture").unwrap_or("");
    ensure!(
        arch == "qwen2",
        "Expected general.architecture = 'qwen2', got '{arch}'"
    );

    let num_layers = reader
        .meta_u32("qwen2.block_count")
        .context("Missing qwen2.block_count")? as usize;
    let hidden_size = reader
        .meta_u32("qwen2.embedding_length")
        .context("Missing qwen2.embedding_length")? as usize;
    let intermediate_size = reader
        .meta_u32("qwen2.feed_forward_length")
        .context("Missing qwen2.feed_forward_length")? as usize;
    let num_heads = reader
        .meta_u32("qwen2.attention.head_count")
        .context("Missing qwen2.attention.head_count")? as usize;
    let num_kv_heads = reader
        .meta_u32("qwen2.attention.head_count_kv")
        .context("Missing qwen2.attention.head_count_kv")? as usize;
    let rms_norm_eps = reader
        .meta_f32("qwen2.attention.layer_norm_rms_epsilon")
        .context("Missing qwen2.attention.layer_norm_rms_epsilon")? as f64;
    let rope_theta = reader
        .meta_f32("qwen2.rope.freq_base")
        .context("Missing qwen2.rope.freq_base")? as f64;
    let max_seq_len = reader
        .meta_u32("qwen2.context_length")
        .context("Missing qwen2.context_length")? as usize;

    let bos_token_id = reader.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(151643);
    let eos_token_id = reader.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(151645);
    let pad_token_id = reader
        .meta_u32("tokenizer.ggml.padding_token_id")
        .unwrap_or(151643);
    // generation_config.json's eos list is [151645, 151643] (im_end, then
    // endoftext/pad as fallback) — see docs/MODELS.md §1. Dedup in case the
    // GGUF's own eos/pad happen to coincide.
    let mut eos_token_ids = vec![eos_token_id];
    if pad_token_id != eos_token_id {
        eos_token_ids.push(pad_token_id);
    }

    let embd_info = reader
        .tensor_info("token_embd.weight")
        .context("Missing tensor 'token_embd.weight'")?;
    let embd_shape = reverse_gguf_dims(embd_info.shape());
    ensure!(
        embd_shape.len() == 2,
        "Expected 2D token_embd.weight, got {embd_shape:?}"
    );
    let vocab_size = embd_shape[0];
    ensure!(
        embd_shape[1] == hidden_size,
        "token_embd.weight hidden dim {} != qwen2.embedding_length {hidden_size}",
        embd_shape[1]
    );

    Ok(LlmConfig {
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        vocab_size,
        rope_theta,
        max_seq_len,
        rms_norm_eps,
        bos_token_id,
        eos_token_ids,
    })
}

// ---------------------------------------------------------------------------
// ShardedCursor — Read + Seek over multiple buffers
// ---------------------------------------------------------------------------

/// A cursor that provides `Read + Seek` over multiple contiguous byte buffers.
///
/// Each shard is kept as a separate `Vec<u8>` to stay under the WASM32
/// `isize::MAX` (~2 GB) per-allocation limit while supporting total sizes > 2 GB.
pub struct ShardedCursor {
    shards: Vec<Vec<u8>>,
    ends: Vec<u64>,
    pos: u64,
    total_len: u64,
}

impl ShardedCursor {
    pub fn new(shards: Vec<Vec<u8>>) -> Self {
        let mut ends = Vec::with_capacity(shards.len());
        let mut total: u64 = 0;
        for s in &shards {
            total += s.len() as u64;
            ends.push(total);
        }
        Self {
            shards,
            ends,
            pos: 0,
            total_len: total,
        }
    }

    fn shard_for_offset(&self, offset: u64) -> Option<(usize, usize)> {
        if offset >= self.total_len {
            return None;
        }
        let shard_idx = self.ends.partition_point(|&end| end <= offset);
        let shard_start = if shard_idx > 0 {
            self.ends[shard_idx - 1]
        } else {
            0
        };
        Some((shard_idx, (offset - shard_start) as usize))
    }
}

impl Read for ShardedCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.total_len {
            return Ok(0);
        }
        let (shard_idx, local_offset) = self.shard_for_offset(self.pos).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "no shard found for offset {} (total_len={})",
                    self.pos, self.total_len
                ),
            )
        })?;
        let shard = &self.shards[shard_idx];
        let available = shard.len() - local_offset;
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&shard[local_offset..local_offset + to_read]);
        self.pos += to_read as u64;
        Ok(to_read)
    }
}

impl Seek for ShardedCursor {
    fn seek(&mut self, style: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match style {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.total_len as i64 + offset,
            SeekFrom::Current(offset) => self.pos as i64 + offset,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

// ---------------------------------------------------------------------------
// Q4Tensor — GPU buffer of Q4_0 blocks
// ---------------------------------------------------------------------------

/// A Q4_0 quantized weight tensor living on GPU.
///
/// GGUF's on-disk Q4_0 block is 18 bytes (2-byte f16 scale + 16 bytes of
/// paired nibbles) interleaved per 32-element block — 18 isn't a multiple of
/// 4, so every other block's nibble data starts at a byte offset that isn't
/// u32-aligned, forcing every WGSL nibble read through a two-word
/// unaligned-load path (`read_u32_unaligned`) with no coalescing across
/// lanes reading adjacent blocks. `from_q4_bytes` repacks on load into two
/// separate, individually-aligned GPU buffers: `scales` (one f32 per block)
/// and `nibbles` (16 bytes = exactly 4 u32 per block, scale stripped out) —
/// see docs/BENCHMARKS.md's K5 section for the measured effect. All four
/// WGSL kernels (naive, matvec, matvec_subgroup, tiled) read this layout.
/// `Clone` is a cheap handle clone (ref-counted GPU buffer, not a copy) —
/// used to share `token_embd.weight`'s buffers between the embedding table
/// and the tied lm_head matmul.
#[derive(Clone)]
pub struct Q4Tensor {
    pub(crate) nibbles: Handle,
    pub(crate) scales: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q4Tensor {
    /// Upload raw Q4_0 bytes (GGUF's interleaved 18-bytes/block on-disk
    /// layout) to GPU, repacked into the two-buffer layout described on
    /// [`Q4Tensor`].
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`.
    /// `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes.
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        // Both output buffers are naturally 4-byte aligned (16 and 4 bytes
        // per block respectively) — no padding needed, unlike the old
        // single-buffer 18-bytes/block layout.
        let mut nibbles_bytes = vec![0u8; num_blocks * 16];
        let mut scales_bytes = vec![0u8; num_blocks * 4];
        for blk in 0..num_blocks {
            let bo = blk * 18;
            let scale_bits = u16::from_le_bytes([raw_bytes[bo], raw_bytes[bo + 1]]);
            let scale = f16_to_f32(scale_bits);
            scales_bytes[blk * 4..blk * 4 + 4].copy_from_slice(&scale.to_le_bytes());
            nibbles_bytes[blk * 16..blk * 16 + 16].copy_from_slice(&raw_bytes[bo + 2..bo + 18]);
        }

        let client = WgpuRuntime::client(device);
        let nibbles = client.create_from_slice(&nibbles_bytes);
        let scales = client.create_from_slice(&scales_bytes);

        Ok(Self {
            nibbles,
            scales,
            shape,
            num_blocks,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ---------------------------------------------------------------------------
// Q4Linear
// ---------------------------------------------------------------------------

/// A linear layer with Q4_0 quantized weights.
///
/// Stores weights as `[out_features, in_features]` in Q4_0 format and an
/// optional f32 bias (Qwen2's q/k/v projections carry bias; attn_output and
/// all ffn_* projections don't — docs/MODELS.md §2). Forward: `x @ weights^T
/// + bias` via fused dequant+matmul.
pub struct Q4Linear {
    weights: Q4Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4Linear {
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    pub fn out_features(&self) -> usize {
        self.weights.shape()[0]
    }

    pub fn in_features(&self) -> usize {
        self.weights.shape()[1]
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q4_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// Q4 matmul kernel dispatch
// ---------------------------------------------------------------------------

/// Set (from device-init code, e.g. after probing `wgpu::Features::SUBGROUP`)
/// to route the M=1 decode matvec through `shader_q4_matvec_subgroup.wgsl`
/// instead of the portable shared-memory-reduction variant — mirrors
/// sts-web's `gguf.rs:33-46` / `web/bindings.rs:162-163` pattern. Nothing in
/// this crate currently calls `set_subgroup_support(true)` (that wiring
/// lives in web.rs, outside this crate's owned files for this task), so this
/// defaults to `false` — every measured number in docs/BENCHMARKS.md uses
/// the shared-memory variant.
static HAS_SUBGROUPS: AtomicBool = AtomicBool::new(false);

pub fn set_subgroup_support(supported: bool) {
    HAS_SUBGROUPS.store(supported, Ordering::Relaxed);
}

pub fn has_subgroup_support() -> bool {
    HAS_SUBGROUPS.load(Ordering::Relaxed)
}

/// Bench-only instrumentation (docs/BENCHMARKS.md P1c): when set, `q4_matmul`
/// skips its GPU kernel launch entirely and returns an uninitialized output
/// buffer of the correct shape/dtype. Used by `llm-agent bench` to measure
/// "everything else" (RMSNorm, RoPE, attention, SwiGLU elementwise, residual
/// adds, cache writes) in isolation from the Q4 matvec/matmul cost. Defaults
/// to `false`; never set outside the bench binary — output is garbage
/// whenever this is `true`, so it must never be enabled on a path whose
/// numerics are checked (`full_forward`'s greedy-match tests never touch
/// this).
static SKIP_MATVEC_FOR_BENCH: AtomicBool = AtomicBool::new(false);

pub fn set_skip_matvec_for_bench(skip: bool) {
    SKIP_MATVEC_FOR_BENCH.store(skip, Ordering::Relaxed);
}

pub fn skip_matvec_for_bench() -> bool {
    SKIP_MATVEC_FOR_BENCH.load(Ordering::Relaxed)
}

struct Q4MatvecKernel;

impl KernelSource for Q4MatvecKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4_matvec.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Session 6 (docs/BENCHMARKS.md): word-interleaved coalesced matvec —
/// replaces `Q4MatvecKernel` (K1) as the default portable (non-subgroup)
/// M=1 kernel. K1 is kept in-tree (unused by default dispatch) as a
/// reference/rollback point — see `wgsl/shader_q4_matvec_coalesced.wgsl`'s
/// header comment for the access-pattern fix.
struct Q4MatvecCoalescedKernel;

impl KernelSource for Q4MatvecCoalescedKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4_matvec_coalesced.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

struct Q4MatvecSubgroupKernel;

impl KernelSource for Q4MatvecSubgroupKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4_matvec_subgroup.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// K2 — tiled prefill (M>1) matmul with workgroup-shared dequant/weight
/// reuse (wgsl/shader_q4_tiled.wgsl). Native-only: the naive kernel stays
/// the WASM/WebGPU default for M>1 (docs/ENGINE.md §2 / this module's doc
/// comment — "tiled kernel is native-only").
#[cfg(not(target_arch = "wasm32"))]
struct Q4MatmulTiledKernel;

#[cfg(not(target_arch = "wasm32"))]
impl KernelSource for Q4MatmulTiledKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4_tiled.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

// Must match wgsl/shader_q4_tiled.wgsl's TM/TN constants.
#[cfg(not(target_arch = "wasm32"))]
const TILED_TM: usize = 64;
#[cfg(not(target_arch = "wasm32"))]
const TILED_TN: usize = 64;

struct Q4MatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q4MatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_naive.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q4_0 dequant+matmul on GPU.
///
/// Computes `output[B, M, N] = input[B, M, K] × weights[N, K]^T`. General
/// over M — serves both prefill (M = prompt length) and decode (M = 1).
/// M==1 dispatches K1's cooperative matvec kernel; M>1 dispatches the naive
/// kernel. K2 (wgsl/shader_q4_tiled.wgsl, tiled prefill matmul with
/// workgroup-shared dequant/weight reuse) exists and passes correctness
/// tests (`q4_matmul_tiled_forced` below, exercised by
/// tests/q4_matmul.rs's `test_q4_matmul_synthetic_shapes`/`bench_tiled_*`)
/// but is **not** dispatched here: measured at 30-40 GFLOP/s in isolation
/// (docs/BENCHMARKS.md) vs the naive kernel's ~159 GFLOP/s effective at
/// full-model granularity — a ~4x regression traced to two rounds of fixes
/// (barrier:compute ratio, vectorized dequant reads) that moved the number
/// by <35% total, not closed the gap; root cause not found within this
/// session's budget. Left as tested-but-unused for future work rather than
/// shipped as a regression — see that doc's K2 section.
pub fn q4_matmul(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    q4_matmul_dispatch(input, weights, ForceKernel::Auto)
}

/// Test-only routing override for `q4_matmul_dispatch`, isolating a
/// specific kernel path regardless of the M/N thresholds `q4_matmul` would
/// otherwise apply — see tests/q4_matmul.rs's per-M scratch-vs-naive
/// comparison (Session 6 bug hunt for the split-prefill logit divergence).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ForceKernel {
    /// Production routing: M==1 -> matvec, M>=32 && N<100000 -> scratch
    /// dequant + Burn matmul, else naive.
    Auto,
    /// K2's tiled kernel (native only, B==1) — see `q4_matmul`'s doc
    /// comment for why it's not the default.
    Tiled,
    /// Force the naive one-thread-per-output kernel even when M/N would
    /// otherwise route through scratch-dequant+matmul.
    Naive,
    /// Force the scratch-dequant + Burn `Tensor::matmul` path even when M
    /// is below `SCRATCH_MATMUL_MIN_M`.
    Scratch,
    /// Force K1's superseded stride-4-words matvec kernel (requires M==1) —
    /// kept selectable for Session 6's before/after A/B bench comparison in
    /// docs/BENCHMARKS.md; not used by production `ForceKernel::Auto`
    /// routing, which now always takes the coalesced kernel for M==1.
    MatvecK1,
}

/// Forces K2's tiled kernel regardless of M (native only, requires B==1) —
/// used only by tests/q4_matmul.rs to keep correctness coverage on the
/// kernel while it's unselected in production. See `q4_matmul`'s doc
/// comment for why it's not the default.
#[cfg(not(target_arch = "wasm32"))]
pub fn q4_matmul_tiled_forced(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    q4_matmul_dispatch(input, weights, ForceKernel::Tiled)
}

/// Forces the naive per-element kernel regardless of M — test-only, see
/// `ForceKernel::Naive`.
pub fn q4_matmul_naive_forced(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    q4_matmul_dispatch(input, weights, ForceKernel::Naive)
}

/// Forces the scratch-dequant + Burn matmul path regardless of M —
/// test-only, see `ForceKernel::Scratch`.
pub fn q4_matmul_scratch_forced(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    q4_matmul_dispatch(input, weights, ForceKernel::Scratch)
}

/// Forces K1's superseded matvec kernel (requires M==1) — test/bench-only,
/// see `ForceKernel::MatvecK1`.
pub fn q4_matmul_matvec_k1_forced(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    q4_matmul_dispatch(input, weights, ForceKernel::MatvecK1)
}

struct Q4DequantKernel;

impl KernelSource for Q4DequantKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q4_dequant.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

const DEQUANT_WG: u32 = 256;

/// P1 Approach A threshold (docs/BENCHMARKS.md Session 4): below this M,
/// the naive per-element kernel's redundant dequant isn't amortized enough
/// across output rows to beat the naive kernel's simplicity/dispatch-count
/// tradeoff, so decode-shaped calls (M=1) keep using the matvec kernel via
/// the `b * m == 1` branch above this one, and small M falls through to the
/// naive kernel.
const SCRATCH_MATMUL_MIN_M: usize = 32;

/// The tied lm_head (`token_embd.weight`, [151936, 2048]) is never routed
/// through the scratch-dequant+matmul path: its dequantized scratch buffer
/// would be 151936 * 2048 * 4 = ~1.16GB, and it's only ever run on a
/// handful of rows at the end of prefill — stays on the naive/matvec path.
const SCRATCH_MATMUL_MAX_N: usize = 100_000;

thread_local! {
    /// Reused across calls, grows to fit the largest layer matrix seen
    /// (11008x2048x4 = ~90MB for this model's FFN up/gate/down
    /// projections). Not shared across threads/devices by design — see
    /// `q4_dequant_scratch`'s doc comment.
    static DEQUANT_SCRATCH: RefCell<Option<(Handle, usize)>> = const { RefCell::new(None) };
}

/// P1 Approach A (docs/BENCHMARKS.md Session 4): dequantize `weights`
/// ([N, K] Q4_0) into a transposed f32 scratch buffer of shape `[K, N]`
/// via `shader_q4_dequant.wgsl`, reusing one thread-local scratch
/// allocation across calls instead of allocating fresh each time. The
/// transposed layout means the caller can run `x[B,M,K] . W[1,K,N]`
/// directly through Burn's `Tensor::matmul` (cubecl's tiled/cmma kernels)
/// with no separate transpose dispatch.
fn q4_dequant_scratch(
    client: &cubecl::client::ComputeClient<WgpuRuntime>,
    weights: &Q4Tensor,
) -> Handle {
    let [n, k] = weights.shape();
    let needed_bytes = n * k * 4;
    let blocks_per_row = k / 32;

    let handle = DEQUANT_SCRATCH.with(|cell| {
        let mut cell = cell.borrow_mut();
        let reuse = match &*cell {
            Some((_, size)) => *size >= needed_bytes,
            None => false,
        };
        if !reuse {
            *cell = Some((client.empty(needed_bytes), needed_bytes));
        }
        cell.as_ref().unwrap().0.clone()
    });

    // 2D workgroup grid: WebGPU caps a single dispatch dimension at 65535
    // workgroups, which a 1D dispatch can exceed for this model's larger
    // layers (e.g. 11008x2048 needs 88064 workgroups of 256 threads).
    let total = (n * k) as u32;
    let wg_total = total.div_ceil(DEQUANT_WG);
    const MAX_WG_DIM: u32 = 65535;
    let wg_x = wg_total.min(MAX_WG_DIM);
    let wg_y = wg_total.div_ceil(wg_x);
    let threads_per_row = wg_x * DEQUANT_WG;

    let info: [u32; 4] = [n as u32, k as u32, blocks_per_row as u32, threads_per_row];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.nibbles.clone().binding())
        .with_buffer(weights.scales.clone().binding())
        .with_buffer(handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(Q4DequantKernel, CubeDim::new_1d(DEQUANT_WG));
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("Q4 dequant kernel launch failed");

    handle
}

/// Test-only: run `shader_q4_dequant.wgsl` on `weights` and read back the
/// full `[K, N]` transposed scratch buffer as a flat row-major `Vec<f32>`
/// (`out[k * n + n_idx]`) — isolates the dequant kernel itself from
/// `scratch_matmul_chunked`/Burn's `Tensor::matmul`. See
/// tests/q4_matmul.rs's dequant-only coverage (Session 6 bug hunt).
pub fn q4_dequant_scratch_to_vec(weights: &Q4Tensor, device: &WgpuDevice) -> Vec<f32> {
    let [n, k] = weights.shape();
    let client = WgpuRuntime::client(device);
    let handle = q4_dequant_scratch(&client, weights);
    let tensor = CubeTensor::new_contiguous(
        client,
        device.clone(),
        burn::prelude::Shape::from(vec![k, n]),
        handle,
        DType::F32,
    );
    let out = Tensor::<Wgpu, 2>::from_primitive(TensorPrimitive::Float(tensor));
    out.into_data().into_vec::<f32>().expect("f32 readback")
}

/// Largest M chunk for `scratch_matmul_chunked`'s calls into Burn's
/// `Tensor::matmul`. Session 4: this build had no `autotune` cubecl feature,
/// so `Tensor::matmul` fell back to a fixed `Strategy::Auto` matmul kernel
/// that panicked with "shared memory ... hardware limit" ("needs 40960
/// shared memory bytes but hardware limit is 32768") once M got into the low
/// thousands on this M2/Metal adapter, forcing a 256-wide chunk (matching
/// `model.rs::ATTN_QUERY_CHUNK`'s same-root-cause workaround).
///
/// Session 5: enabled `burn/autotune` (see `Cargo.toml`'s `wgpu` feature) —
/// autotune picks a kernel strategy that fits the 32768-byte shared-memory
/// limit at this shape, so chunking is no longer required for correctness.
/// Raised to 2225 (this benchmark's full prompt length, i.e.
/// `scratch_matmul_chunked` never actually chunks at that prompt length) for
/// throughput: 73.3 tok/s warm vs 43.4 tok/s at the old 256 chunk on the
/// `02_tools_single` prefill (docs/BENCHMARKS.md Session 5) — fewer, larger
/// `Tensor::matmul` calls beat more numerous smaller ones once autotune keeps
/// them from panicking. Verified via `full_forward`'s greedy-match tests
/// (01/02/03 exact) at both 256 and 2225; not verified past this prompt
/// length's shape — if a much longer prefill panics again, the fix is to
/// lower this back down (chunking still works correctly for M > this value,
/// it's purely a throughput/safety-margin tradeoff, not a correctness one).
const SCRATCH_MATMUL_CHUNK_M: usize = 2225;

/// Chunks `x[B,M,K] . w[1,K,N]` over the M dimension — see
/// `SCRATCH_MATMUL_CHUNK_M`'s doc comment for why.
fn scratch_matmul_chunked(x: Tensor<Wgpu, 3>, w: Tensor<Wgpu, 3>, m: usize) -> Tensor<Wgpu, 3> {
    if m <= SCRATCH_MATMUL_CHUNK_M {
        return x.matmul(w);
    }
    let mut chunks = Vec::with_capacity(m.div_ceil(SCRATCH_MATMUL_CHUNK_M));
    let mut start = 0usize;
    while start < m {
        let len = SCRATCH_MATMUL_CHUNK_M.min(m - start);
        let x_chunk = x.clone().narrow(1, start, len);
        chunks.push(x_chunk.matmul(w.clone()));
        start += len;
    }
    Tensor::cat(chunks, 1)
}

fn q4_matmul_dispatch(input: Tensor<Wgpu, 3>, weights: &Q4Tensor, force: ForceKernel) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    if skip_matvec_for_bench() {
        // Bench-only bypass: skip the kernel launch, return the
        // uninitialized buffer as-is. See `set_skip_matvec_for_bench`'s doc
        // comment — output is garbage, never enabled on a numerics-checked
        // path.
        let output_handle = client.empty(b * m * n * 4);
        let output_tensor = CubeTensor::new_contiguous(
            client,
            device,
            burn::prelude::Shape::from(vec![b, m, n]),
            output_handle,
            DType::F32,
        );
        return Tensor::from_primitive(TensorPrimitive::Float(output_tensor));
    }

    // P1 Approach A (docs/BENCHMARKS.md Session 4): for prefill-shaped
    // calls (M >= 32) on layers small enough to dequant into a scratch f32
    // buffer (excludes the 151936-wide lm_head), dequant once via
    // `shader_q4_dequant.wgsl` and run the actual matmul through Burn's
    // `Tensor::matmul` (cubecl's tiled/cmma kernels) instead of the naive
    // per-element-redundant-dequant kernel. `ForceKernel::Tiled`/`Naive`
    // (test-only, K2/naive-vs-scratch comparison) bypass this to keep
    // exercising those kernels directly; `ForceKernel::Scratch` forces this
    // path even below `SCRATCH_MATMUL_MIN_M`.
    let take_scratch = match force {
        ForceKernel::Auto => m >= SCRATCH_MATMUL_MIN_M && n < SCRATCH_MATMUL_MAX_N,
        ForceKernel::Scratch => true,
        ForceKernel::Tiled | ForceKernel::Naive | ForceKernel::MatvecK1 => false,
    };
    if take_scratch {
        let w_handle = q4_dequant_scratch(&client, weights);
        let w_tensor = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            burn::prelude::Shape::from(vec![1, k, n]),
            w_handle,
            DType::F32,
        );
        let x = Tensor::from_primitive(TensorPrimitive::Float(cube_input.clone()));
        let w = Tensor::<Wgpu, 3>::from_primitive(TensorPrimitive::Float(w_tensor));
        return scratch_matmul_chunked(x, w, m);
    }

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.nibbles.clone().binding())
        .with_buffer(weights.scales.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    // M==1 (decode): dispatch the cooperative matvec kernel (Session 6's
    // coalesced kernel — wgsl/shader_q4_matvec_coalesced.wgsl, or K1's
    // subgroup variant when available) instead of the naive
    // one-thread-per-output kernel. Condition is `m == 1`, not `b * m == 1`
    // — the matvec kernel's `b`/`B` handling (wg_id.y, `b_valid` guard) was
    // already batch-general, so B>1 M=1 decode (e.g. classifier-free
    // guidance's dual-batch KV caches) now also takes this path instead of
    // falling through to the naive kernel. M>1 (prefill) keeps the naive
    // kernel unless a test forces K2's tiled kernel (see `q4_matmul`'s doc
    // comment — not the production default).
    if force == ForceKernel::Auto && m == 1 {
        let (kernel, rows_per_wg): (Box<dyn CubeTask<AutoCompiler>>, usize) = if has_subgroup_support() {
            (
                Box::new(SourceKernel::new(Q4MatvecSubgroupKernel, CubeDim::new_1d(256))),
                MATVEC_ROWS_PER_WG,
            )
        } else {
            (
                Box::new(SourceKernel::new(Q4MatvecCoalescedKernel, CubeDim::new_1d(128))),
                MATVEC_COALESCED_ROWS_PER_WG,
            )
        };
        let wg_x = n.div_ceil(rows_per_wg) as u32;
        let wg_y = b as u32;
        client
            .launch(kernel, CubeCount::new_2d(wg_x, wg_y), bindings)
            .expect("Q4 matvec kernel launch failed");
    } else if force == ForceKernel::MatvecK1 {
        assert_eq!(m, 1, "ForceKernel::MatvecK1 requires M==1");
        let kernel: Box<dyn CubeTask<AutoCompiler>> =
            Box::new(SourceKernel::new(Q4MatvecKernel, CubeDim::new_1d(256)));
        let wg_x = n.div_ceil(MATVEC_ROWS_PER_WG) as u32;
        let wg_y = b as u32;
        client
            .launch(kernel, CubeCount::new_2d(wg_x, wg_y), bindings)
            .expect("Q4 matvec (K1) kernel launch failed");
    } else if cfg!(not(target_arch = "wasm32")) && b == 1 && force == ForceKernel::Tiled {
        // K2 (native only, B==1, test-forced only): tiled matmul with
        // workgroup-shared dequant/weight reuse — see
        // wgsl/shader_q4_tiled.wgsl.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let kernel = SourceKernel::new(Q4MatmulTiledKernel, CubeDim::new_2d(16, 16));
            let wg_x = n.div_ceil(TILED_TN) as u32;
            let wg_y = m.div_ceil(TILED_TM) as u32;
            client
                .launch(
                    Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
                    CubeCount::new_2d(wg_x, wg_y),
                    bindings,
                )
                .expect("Q4 tiled matmul kernel launch failed");
        }
    } else {
        let kernel = SourceKernel::new(
            Q4MatmulNaiveKernel {
                workgroup_size_x: NAIVE_WG_X,
                workgroup_size_y: NAIVE_WG_Y,
            },
            CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
        );
        let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
        let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
        client
            .launch(
                Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
                CubeCount::new_2d(wg_x, wg_y),
                bindings,
            )
            .expect("Q4 naive matmul kernel launch failed");
    }

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// D2 — fused RMSNorm kernel
// ---------------------------------------------------------------------------

struct RmsNormKernel;

impl KernelSource for RmsNormKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_rmsnorm.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused RMSNorm: `Y = X / sqrt(mean(X^2) + eps) * gamma`, one dispatch for
/// the whole `[B, T, hidden]` input instead of burn-nn's unfused
/// cast/square/mean_dim/add/sqrt/div/mul chain. See
/// `wgsl/shader_rmsnorm.wgsl` for the kernel and `RmsNormLayer::forward`
/// (model.rs) for the call site.
pub fn rmsnorm_fused(x: Tensor<Wgpu, 3>, weight: Tensor<Wgpu, 1>, eps: f32) -> Tensor<Wgpu, 3> {
    let cube_x: CubeTensor<WgpuRuntime> = x.into_primitive().tensor();
    let cube_x = into_contiguous(cube_x);
    let cube_w: CubeTensor<WgpuRuntime> = weight.into_primitive().tensor();
    let cube_w = into_contiguous(cube_w);

    let [b, t, hidden] = [cube_x.shape.dims[0], cube_x.shape.dims[1], cube_x.shape.dims[2]];
    let rows = b * t;
    assert_eq!(cube_w.shape.dims[0], hidden, "RMSNorm weight/hidden size mismatch");

    let client = cube_x.client.clone();
    let device = cube_x.device.clone();
    let output_handle = client.empty(rows * hidden * 4);

    let info: [f32; 3] = [rows as f32, hidden as f32, eps];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_x.handle.clone().binding())
        .with_buffer(cube_w.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(RmsNormKernel, CubeDim::new_1d(256));
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_1d(rows as u32),
            bindings,
        )
        .expect("RMSNorm kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, t, hidden]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// F1 — fused RoPE kernel
// ---------------------------------------------------------------------------

struct RopeKernel;

impl KernelSource for RopeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_rope.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused rotate-half RoPE applied in place to `q`/`k`. `q`: `[1, T, H, Dh]`,
/// `k`: `[1, T, Hkv, Dh]` (the natural `reshape()` layout, *before*
/// `model.rs` permutes to `[1, H, T, Dh]`). `cos`/`sin`: `RoPE`'s
/// `[max_seq_len, Dh]` precomputed tables. `offset`: absolute position of
/// row 0. One dispatch total (both q and k) instead of the ~10-dispatch
/// Burn op chain — see `wgsl/shader_rope.wgsl`'s doc comment for the layout
/// and safety argument, and `model.rs::apply_rope`/`rotate_half` for the
/// reference formula this replaces.
pub fn rope_fused(
    q: Tensor<Wgpu, 4>,
    k: Tensor<Wgpu, 4>,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
    offset: usize,
) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
    let cube_q: CubeTensor<WgpuRuntime> = q.into_primitive().tensor();
    let cube_q = into_contiguous(cube_q);
    let cube_k: CubeTensor<WgpuRuntime> = k.into_primitive().tensor();
    let cube_k = into_contiguous(cube_k);
    let cube_cos: CubeTensor<WgpuRuntime> = cos.clone().into_primitive().tensor();
    let cube_cos = into_contiguous(cube_cos);
    let cube_sin: CubeTensor<WgpuRuntime> = sin.clone().into_primitive().tensor();
    let cube_sin = into_contiguous(cube_sin);

    let t = cube_q.shape.dims[1];
    let h = cube_q.shape.dims[2];
    let dh = cube_q.shape.dims[3];
    let hkv = cube_k.shape.dims[2];
    let half = dh / 2;
    let cos_stride = cube_cos.shape.dims[1];
    assert_eq!(cube_k.shape.dims[3], dh, "RoPE q/k head_dim mismatch");
    assert_eq!(cube_k.shape.dims[1], t, "RoPE q/k row-count mismatch");
    assert_eq!(cos_stride, dh, "RoPE cos table stride/head_dim mismatch");

    let client = cube_q.client.clone();
    let device = cube_q.device.clone();

    let total = t * (h + hkv) * half;
    let (wg_x, wg_y, row_width) = workgroups_2d(total, 256);

    let info: [f32; 7] = [
        t as f32,
        h as f32,
        hkv as f32,
        half as f32,
        offset as f32,
        cos_stride as f32,
        row_width as f32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_q.handle.clone().binding())
        .with_buffer(cube_k.handle.clone().binding())
        .with_buffer(cube_cos.handle.binding())
        .with_buffer(cube_sin.handle.binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(RopeKernel, CubeDim::new_1d(256));
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("RoPE kernel launch failed");

    let q_shape = burn::prelude::Shape::from(vec![1, t, h, dh]);
    let k_shape = burn::prelude::Shape::from(vec![1, t, hkv, dh]);
    let q_out = CubeTensor::new_contiguous(client.clone(), device.clone(), q_shape, cube_q.handle, DType::F32);
    let k_out = CubeTensor::new_contiguous(client, device, k_shape, cube_k.handle, DType::F32);
    (
        Tensor::from_primitive(TensorPrimitive::Float(q_out)),
        Tensor::from_primitive(TensorPrimitive::Float(k_out)),
    )
}

// ---------------------------------------------------------------------------
// F2 — fused SiLU*up kernel
// ---------------------------------------------------------------------------

struct SiluMulKernel;

impl KernelSource for SiluMulKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_silu_mul.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Fused `silu(gate) * up`, in place into `gate`'s buffer. `gate`/`up`:
/// same shape (`[1, T, ffn_dim]`). One dispatch instead of Burn's separate
/// `silu` + `mul` chain — see `wgsl/shader_silu_mul.wgsl`'s doc comment.
pub fn silu_mul_fused(gate: Tensor<Wgpu, 3>, up: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
    let cube_gate: CubeTensor<WgpuRuntime> = gate.into_primitive().tensor();
    let cube_gate = into_contiguous(cube_gate);
    let cube_up: CubeTensor<WgpuRuntime> = up.into_primitive().tensor();
    let cube_up = into_contiguous(cube_up);
    assert_eq!(cube_gate.shape.dims, cube_up.shape.dims, "silu_mul_fused shape mismatch");

    let n: usize = cube_gate.shape.dims.iter().product();
    let client = cube_gate.client.clone();
    let device = cube_gate.device.clone();

    let (wg_x, wg_y, row_width) = workgroups_2d(n, 256);

    let info: [f32; 2] = [n as f32, row_width as f32];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(cube_gate.handle.clone().binding())
        .with_buffer(cube_up.handle.binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(SiluMulKernel, CubeDim::new_1d(256));
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("SiLU*up kernel launch failed");

    let shape = burn::prelude::Shape::from(cube_gate.shape.dims.clone());
    let out = CubeTensor::new_contiguous(client, device, shape, cube_gate.handle, DType::F32);
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------------------------------------------------------------------------
// EmbeddingStore — Q4 embeddings for token lookups
// ---------------------------------------------------------------------------

/// Q4 embedding table stored as CPU bytes for efficient row lookups.
///
/// Dequantizes individual rows on-the-fly (avoids materializing the full
/// embedding table as f32, which would cost 1.2GB at this model's 151936-row
/// vocab — see module doc comment).
pub struct EmbeddingStore {
    cpu_bytes: Vec<u8>,
    vocab_size: usize,
    dim: usize,
}

impl EmbeddingStore {
    /// Create from raw Q4 bytes.
    pub fn new(cpu_bytes: Vec<u8>, vocab_size: usize, dim: usize) -> Self {
        Self {
            cpu_bytes,
            vocab_size,
            dim,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dequantize a single row, returning `dim` f32s.
    pub fn embed_id(&self, id: u32) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.dim];
        self.embed_id_add_cpu(id, &mut out)?;
        Ok(out)
    }

    /// Dequantize a single row into an existing CPU buffer (for accumulation).
    ///
    /// Adds the dequantized embedding to `out_buf` (which must be `dim` f32s).
    /// Errors (rather than panics/indexes out of bounds) when `id` is not a
    /// valid row in this table — `id` can come from tokenizer output on
    /// untrusted input.
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) -> Result<()> {
        ensure!(
            (id as usize) < self.vocab_size,
            "embed_id_add_cpu: id {id} out of range (vocab_size={})",
            self.vocab_size
        );
        assert_eq!(out_buf.len(), self.dim);
        let blocks_per_row = self.dim / 32;
        let bytes_per_row = blocks_per_row * 18;
        let row_offset = (id as usize) * bytes_per_row;
        let row_bytes = &self.cpu_bytes[row_offset..row_offset + bytes_per_row];

        for block in 0..blocks_per_row {
            let bo = block * 18;
            let d = f16_to_f32(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]));
            let base = block * 32;
            for j in 0..16 {
                let byte = row_bytes[bo + 2 + j];
                out_buf[base + j] += ((byte & 0x0F) as f32 - 8.0) * d;
                out_buf[base + j + 16] += (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Q4ModelParts — deferred loading intermediate
// ---------------------------------------------------------------------------

/// All Q4 model components with `token_embd.weight` still in raw Q4 form.
///
/// Used by [`Q4ModelLoader::load_deferred`] to allow freeing the GGUF
/// reader's memory before creating the GPU embedding/lm_head buffer.
pub struct Q4ModelParts {
    pub layers: Vec<Q4TransformerBlock>,
    pub rope: RoPE,
    pub out_norm: RmsNormLayer,
    pub token_embd_bytes: Vec<u8>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub config: LlmConfig,
}

impl Q4ModelParts {
    /// Assemble the final model from deferred parts.
    ///
    /// Uploads `token_embd.weight`'s Q4_0 bytes to GPU once (~174MB at this
    /// model's 151936×2048 shape) and reuses that same buffer as both the
    /// embedding table (via CPU-side `EmbeddingStore`, for input lookups)
    /// and the lm_head weight (via `Q4Linear`, tied — no separate
    /// `output.weight` tensor exists in this GGUF, docs/MODELS.md §2).
    pub fn finalize(self, device: &WgpuDevice) -> Result<LlmModel> {
        let embed_gpu = Q4Tensor::from_q4_bytes(
            &self.token_embd_bytes,
            [self.vocab_size, self.hidden_size],
            device,
        )?;
        let lm_head = Q4Linear::new(embed_gpu, None);
        let embed_store = EmbeddingStore::new(self.token_embd_bytes, self.vocab_size, self.hidden_size);

        Ok(LlmModel::new(
            embed_store,
            self.layers,
            self.rope,
            self.out_norm,
            lm_head,
            self.config,
            device.clone(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Q4ModelLoader — GGUF → LlmModel
// ---------------------------------------------------------------------------

/// Loads a Q4-quantized Qwen2 model from a GGUF file.
pub struct Q4ModelLoader<R: Read + Seek> {
    reader: GgufReader<R>,
}

impl Q4ModelLoader<ShardedCursor> {
    /// Open a GGUF from multiple shards (for WASM where >2GB allocs fail).
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Result<Self> {
        let reader = GgufReader::open(ShardedCursor::new(shards))?;
        Ok(Self { reader })
    }
}

impl<R: Read + Seek> Q4ModelLoader<R> {
    /// Open a GGUF from a single reader (native: a `BufReader<File>`).
    pub fn new(reader: R) -> Result<Self> {
        let reader = GgufReader::open(reader)?;
        Ok(Self { reader })
    }

    pub fn reader(&self) -> &GgufReader<R> {
        &self.reader
    }

    /// Read a tensor's raw bytes directly (for tests / `gguf-info`; real
    /// model loading goes through `load_deferred`).
    pub fn tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        self.reader.tensor_data(name)
    }

    /// Load model components without materializing `token_embd.weight` on GPU.
    ///
    /// Returns [`Q4ModelParts`] — the caller should drop the loader to free
    /// GGUF memory (the underlying file / shard buffers), then call
    /// [`Q4ModelParts::finalize`].
    pub fn load_deferred(&mut self, device: &WgpuDevice) -> Result<Q4ModelParts> {
        let config = config_from_gguf(&self.reader)?;
        tracing::info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            layers = config.num_layers,
            hidden = config.hidden_size,
            vocab = config.vocab_size,
            "Loading Qwen2 Q4 model from GGUF (deferred)"
        );

        let head_dim = config.hidden_size / config.num_heads;
        let rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta, device);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = self
                .load_transformer_layer(i, &config, device)
                .with_context(|| format!("Failed to load transformer layer {i}"))?;
            layers.push(layer);
        }

        let out_norm = self.load_rms_norm("output_norm.weight", config.rms_norm_eps, device)?;

        let embd_info = self
            .reader
            .tensor_info("token_embd.weight")
            .context("Tensor 'token_embd.weight' not found")?
            .clone();
        ensure!(
            embd_info.dtype() == GgmlDtype::Q4_0,
            "Expected Q4_0 for 'token_embd.weight', got {:?}",
            embd_info.dtype()
        );
        let embd_shape = reverse_gguf_dims(embd_info.shape());
        let token_embd_bytes = self.reader.tensor_data("token_embd.weight")?;

        tracing::info!("Qwen2 Q4 model loaded (token_embd deferred)");

        Ok(Q4ModelParts {
            layers,
            rope,
            out_norm,
            token_embd_bytes,
            vocab_size: embd_shape[0],
            hidden_size: embd_shape[1],
            config,
        })
    }

    /// Load a single transformer layer from GGUF (llama.cpp `blk.N.*` naming,
    /// docs/MODELS.md §2).
    fn load_transformer_layer(
        &mut self,
        layer_idx: usize,
        config: &LlmConfig,
        device: &WgpuDevice,
    ) -> Result<Q4TransformerBlock> {
        let p = format!("blk.{layer_idx}");

        let attention_norm =
            self.load_rms_norm(&format!("{p}.attn_norm.weight"), config.rms_norm_eps, device)?;

        let q_proj = self.load_q4_linear_with_bias(
            &format!("{p}.attn_q.weight"),
            &format!("{p}.attn_q.bias"),
            device,
        )?;
        let k_proj = self.load_q4_linear_with_bias(
            &format!("{p}.attn_k.weight"),
            &format!("{p}.attn_k.bias"),
            device,
        )?;
        let v_proj = self.load_q4_linear_with_bias(
            &format!("{p}.attn_v.weight"),
            &format!("{p}.attn_v.bias"),
            device,
        )?;
        let o_proj = self.load_q4_linear(&format!("{p}.attn_output.weight"), device)?;

        let head_dim = config.hidden_size / config.num_heads;
        let attention = Q4Attention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config.num_heads,
            config.num_kv_heads,
            head_dim,
        );

        let ffn_norm =
            self.load_rms_norm(&format!("{p}.ffn_norm.weight"), config.rms_norm_eps, device)?;

        let gate_proj = self.load_q4_linear(&format!("{p}.ffn_gate.weight"), device)?;
        let up_proj = self.load_q4_linear(&format!("{p}.ffn_up.weight"), device)?;
        let down_proj = self.load_q4_linear(&format!("{p}.ffn_down.weight"), device)?;

        let ffn = Q4FeedForward::new(gate_proj, up_proj, down_proj);

        Ok(Q4TransformerBlock::new(
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        ))
    }

    // -----------------------------------------------------------------------
    // Primitive loading helpers
    // -----------------------------------------------------------------------

    fn load_q4_linear(&mut self, name: &str, device: &WgpuDevice) -> Result<Q4Linear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        ensure!(
            shape.len() == 2,
            "Tensor '{name}': expected 2D shape for Q4 linear, got {shape:?}"
        );
        let bytes = self.reader.tensor_data(name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(Q4Linear::new(q4, None))
    }

    fn load_q4_linear_with_bias(
        &mut self,
        weight_name: &str,
        bias_name: &str,
        device: &WgpuDevice,
    ) -> Result<Q4Linear> {
        let info = self
            .reader
            .tensor_info(weight_name)
            .with_context(|| format!("Tensor '{weight_name}' not found"))?
            .clone();
        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{weight_name}', got {:?}", info.dtype());
        }
        let shape = reverse_gguf_dims(info.shape());
        ensure!(
            shape.len() == 2,
            "Tensor '{weight_name}': expected 2D shape for Q4 linear, got {shape:?}"
        );
        let bytes = self.reader.tensor_data(weight_name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
        let bias = self.load_f32_vector(bias_name, device)?;
        Ok(Q4Linear::new(q4, Some(bias)))
    }

    fn load_f32_vector(&mut self, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 1>> {
        let data = self.read_f32_data(name)?;
        let n = data.len();
        Ok(Tensor::from_data(TensorData::new(data, [n]), device))
    }

    fn read_f32_data(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();
        let bytes = self.reader.tensor_data(name)?;
        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            GgmlDtype::Q4_0 => bail!("Cannot load Q4_0 tensor '{name}' as a dense f32 vector"),
        };
        Ok(data)
    }

    fn load_rms_norm(
        &mut self,
        name: &str,
        eps: f64,
        device: &WgpuDevice,
    ) -> Result<RmsNormLayer> {
        let data = self.read_f32_data(name)?;
        let n = data.len();
        let weight: Tensor<Wgpu, 1> = Tensor::from_data(TensorData::new(data, [n]), device);
        Ok(RmsNormLayer {
            inner: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), weight),
                epsilon: eps,
            },
        })
    }
}
