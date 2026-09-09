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
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

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

    pub fn byte_size(&self, num_elements: u64) -> u64 {
        match self {
            Self::F32 => num_elements * 4,
            Self::F16 => num_elements * 2,
            Self::Q4_0 => {
                let num_blocks = num_elements / 32;
                num_blocks * 18
            }
        }
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

    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn byte_size(&self) -> u64 {
        self.dtype.byte_size(self.num_elements())
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

impl<R: Read + Seek> GgufReader<R> {
    /// Parse a GGUF file from the given reader.
    pub fn open(mut reader: R) -> Result<Self> {
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
        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
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
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        let mut tensor_order = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read tensor name {i}"))?;
            let ndims = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read ndims for tensor {i}"))?;
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
        let byte_size = info.byte_size() as usize;
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
/// The buffer contains raw Q4_0 blocks (18 bytes per block of 32 elements).
/// The WGSL shader interprets the buffer as `array<u32>`. `Clone` is a cheap
/// handle clone (ref-counted GPU buffer, not a copy) — used to share
/// `token_embd.weight`'s buffer between the embedding table and the tied
/// lm_head matmul.
#[derive(Clone)]
pub struct Q4Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q4Tensor {
    /// Upload raw Q4_0 bytes to a GPU storage buffer.
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

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
        let padded = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            buf
        } else {
            raw_bytes.to_vec()
        };
        let handle = client.create_from_slice(&padded);

        Ok(Self {
            handle,
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
/// over M — serves both prefill (M = prompt length) and decode (M = 1)
/// through the same naive kernel (docs/ENGINE.md §2).
pub fn q4_matmul(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
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
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

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
    pub fn embed_id(&self, id: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.embed_id_add_cpu(id, &mut out);
        out
    }

    /// Dequantize a single row into an existing CPU buffer (for accumulation).
    ///
    /// Adds the dequantized embedding to `out_buf` (which must be `dim` f32s).
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) {
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
