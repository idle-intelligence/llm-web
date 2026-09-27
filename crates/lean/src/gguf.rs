//! GGUF header/metadata/tensor-index parsing and Q4_0/Q8_0/F16/F32 tensor
//! reading, ported from `crates/llm-wasm/src/gguf.rs`'s pure-parsing half
//! (`read_gguf_string`, `GgufValue`, `GgufReader::open`, `config_from_gguf`'s
//! metadata-key convention) — everything Burn/CubeCL-specific in that file
//! (`Q4Tensor`, the WGSL kernel dispatch, `Q4ModelLoader`) is left behind;
//! this crate writes its own model loading in `model.rs`.
//!
//! Difference from llm-wasm's `GgmlDtype`: this one adds `Q8_0` (code 8),
//! needed for tensors this GGUF may carry at that residency, and reads
//! straight off a `File` via `Read + Seek` rather than a `ShardedCursor` —
//! single-shard only for this slice.

use anyhow::{bail, ensure, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const ALIGNMENT: u64 = 32;
const MAX_CAPACITY_HINT: usize = 1 << 16;

/// IEEE-754 half -> f32. Copied from llm-wasm's `gguf.rs::f16_to_f32`
/// (kept local rather than pulling in the `half` crate's own conversion so
/// this file matches its source line-for-line where it can).
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

/// GGUF stores dims `ne[]` with `ne[0]` fastest-varying (innermost); PyTorch
/// convention is `[out_features, in_features]`. Same as llm-wasm's
/// `reverse_gguf_dims`.
fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("invalid UTF-8 in GGUF string")
}

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
        other => bail!("unexpected scalar GGUF value type: {other}"),
    })
}

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
        other => bail!("unknown GGUF metadata value type: {other}"),
    }
    Ok(())
}

fn read_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<GgufValue> {
    if value_type == 9 {
        let elem_type = reader.read_u32::<LittleEndian>()?;
        let count = reader.read_u64::<LittleEndian>()?;
        for _ in 0..count {
            skip_gguf_value(reader, elem_type)?;
        }
        Ok(GgufValue::Array { elem_type, len: count })
    } else {
        read_gguf_scalar(reader, value_type)
    }
}

/// GGML tensor dtype codes actually needed for a Qwen2 Q4_0 GGUF: F32 (0),
/// F16 (1), Q4_0 (2), Q8_0 (8). Anything else is rejected (this slice never
/// needs Q5/Q6/K-quants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

impl GgmlDtype {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            other => bail!("unsupported GGML dtype code: {other} (this crate only reads F32/F16/Q4_0/Q8_0)"),
        }
    }

    pub fn byte_size(&self, num_elements: u64) -> Result<u64> {
        match self {
            Self::F32 => num_elements.checked_mul(4),
            Self::F16 => num_elements.checked_mul(2),
            Self::Q4_0 => {
                let num_blocks = num_elements / 32;
                num_blocks.checked_mul(18) // 2 (f16 scale) + 16 (nibbles)
            }
            Self::Q8_0 => {
                let num_blocks = num_elements / 32;
                num_blocks.checked_mul(34) // 2 (f16 scale) + 32 (i8)
            }
        }
        .context("tensor size overflow")
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    dimensions: Vec<u64>,
    dtype: GgmlDtype,
    offset: u64,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> Vec<usize> {
        reverse_gguf_dims(&self.dimensions)
    }

    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    pub fn num_elements(&self) -> Result<u64> {
        self.dimensions
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d).with_context(|| format!("tensor '{}': element count overflow", self.name)))
    }

    pub fn byte_size(&self) -> Result<u64> {
        self.dtype.byte_size(self.num_elements()?)
    }
}

/// Parses a GGUF v2/v3 file: header, metadata KV pairs, tensor index. Tensor
/// data itself is read lazily via `tensor_data` (two-phase loading: this
/// struct plus its underlying `File` can be dropped right after `model.rs`
/// finishes converting every tensor into a GPU-resident buffer, so the raw
/// GGUF bytes never sit in memory alongside the GPU copy).
pub struct GgufReader<R: Read + Seek> {
    reader: R,
    tensors: HashMap<String, GgufTensorInfo>,
    tensor_order: Vec<String>,
    metadata: HashMap<String, GgufValue>,
    data_section_offset: u64,
    file_len: u64,
}

impl<R: Read + Seek> GgufReader<R> {
    pub fn open(mut reader: R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0)).context("seeking to end of GGUF")?;
        reader.seek(SeekFrom::Start(0)).context("seeking to start of GGUF")?;

        let magic = reader.read_u32::<LittleEndian>().context("reading GGUF magic")?;
        if magic != GGUF_MAGIC {
            bail!("invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})");
        }
        let version = reader.read_u32::<LittleEndian>().context("reading GGUF version")?;
        if version != 2 && version != 3 {
            bail!("unsupported GGUF version: {version} (expected 2 or 3)");
        }
        let tensor_count = reader.read_u64::<LittleEndian>().context("reading tensor count")?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>().context("reading metadata KV count")?;

        let mut metadata = HashMap::with_capacity((metadata_kv_count as usize).min(MAX_CAPACITY_HINT));
        for i in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader).with_context(|| format!("reading metadata key {i}"))?;
            let value_type = reader.read_u32::<LittleEndian>().with_context(|| format!("reading metadata value type {i}"))?;
            let value = read_gguf_value(&mut reader, value_type).with_context(|| format!("reading metadata value {i} ('{key}')"))?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity((tensor_count as usize).min(MAX_CAPACITY_HINT));
        let mut tensor_order = Vec::with_capacity((tensor_count as usize).min(MAX_CAPACITY_HINT));
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader).with_context(|| format!("reading tensor name {i}"))?;
            let ndims = reader.read_u32::<LittleEndian>().with_context(|| format!("reading ndims for tensor {i}"))?;
            ensure!(ndims <= 4, "tensor {i} ('{name}'): ndims={ndims} exceeds supported maximum of 4");
            let mut dimensions = Vec::with_capacity(ndims as usize);
            for d in 0..ndims {
                dimensions.push(reader.read_u64::<LittleEndian>().with_context(|| format!("reading dim {d} for tensor {i}"))?);
            }
            let dtype = GgmlDtype::from_u32(reader.read_u32::<LittleEndian>().with_context(|| format!("reading dtype for tensor {i}"))?)?;
            let offset = reader.read_u64::<LittleEndian>().with_context(|| format!("reading offset for tensor {i}"))?;
            ensure!(offset <= file_len, "tensor {i} ('{name}'): offset {offset} exceeds file length {file_len}");

            tensor_order.push(name.clone());
            tensors.insert(name.clone(), GgufTensorInfo { name, dimensions, dtype, offset });
        }

        let current_pos = reader.stream_position()?;
        let data_section_offset = align_up(current_pos, ALIGNMENT);

        for info in tensors.values() {
            let byte_size = info.byte_size()?;
            let abs_offset = data_section_offset.checked_add(info.offset).with_context(|| format!("tensor '{}': offset overflow", info.name))?;
            let end = abs_offset.checked_add(byte_size).with_context(|| format!("tensor '{}': data range overflow", info.name))?;
            ensure!(end <= file_len, "tensor '{}': data range [{abs_offset}, {end}) exceeds file length {file_len}", info.name);
        }

        Ok(Self { reader, tensors, tensor_order, metadata, data_section_offset, file_len })
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_order
    }

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

    /// Read one tensor's raw on-disk bytes (Q4_0/Q8_0 block bytes, or raw
    /// F32/F16). Caller converts to a GPU buffer and drops this `Vec<u8>`
    /// immediately (see `model.rs::load`) — never held alongside the GPU
    /// copy, the two-phase-loading point.
    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self.tensors.get(name).with_context(|| format!("tensor '{name}' not found in GGUF"))?.clone();
        let byte_size = usize::try_from(info.byte_size()?).with_context(|| format!("tensor '{name}': byte size does not fit in usize"))?;
        let abs_offset = self.data_section_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; byte_size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Dequantize a Q4_0 block stream to f32. Block: 2 bytes f16 scale + 16
/// bytes of paired nibbles (low nibble -> element j, high nibble -> element
/// j+16), `value = (nibble - 8) * scale` — llama.cpp's `quantize_row_q4_0_ref`
/// convention, same as `t0-fast`'s `quant.rs` and `llm-wasm`'s WGSL kernels.
pub fn dequantize_q4_0(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_elements];
    for (bi, block) in bytes.chunks_exact(18).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = (byte >> 4) as f32 - 8.0;
            let base = bi * 32;
            out[base + j] = lo * scale;
            out[base + 16 + j] = hi * scale;
        }
    }
    out
}

/// Dequantize a Q8_0 block stream to f32. Block: 2 bytes f16 scale + 32
/// signed i8 values, `value = qs[j] * scale`.
pub fn dequantize_q8_0(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_elements];
    for (bi, block) in bytes.chunks_exact(34).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..32 {
            out[bi * 32 + j] = (block[2 + j] as i8) as f32 * scale;
        }
    }
    out
}

pub fn dequantize_f16(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    (0..n_elements).map(|i| f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))).collect()
}

pub fn dequantize_f32(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    (0..n_elements).map(|i| f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]])).collect()
}

pub fn dequantize_for(dtype: GgmlDtype, bytes: &[u8], n_elements: usize) -> Vec<f32> {
    match dtype {
        GgmlDtype::F32 => dequantize_f32(bytes, n_elements),
        GgmlDtype::F16 => dequantize_f16(bytes, n_elements),
        GgmlDtype::Q4_0 => dequantize_q4_0(bytes, n_elements),
        GgmlDtype::Q8_0 => dequantize_q8_0(bytes, n_elements),
    }
}
