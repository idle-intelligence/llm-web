//! Prefix KV images: a build-time-produced file that stores a `KvCache`
//! prefix (K/V for the constant system+tools preamble, see
//! `docs/ENGINE.md`'s "Prefix cache" section and `kv.rs`'s module docs) so
//! any engine instance can load it in one shot instead of running prefill
//! token-by-token.
//!
//! ## Format v1 (little-endian)
//!
//! ```text
//! magic:      6 bytes   "KVIMG\0"
//! version:    u32        1
//! header_len: u32        byte length of the JSON header below
//! header:     header_len bytes, UTF-8 JSON (see `Header`)
//! padding:    0..=15 bytes of zero, so tensor data starts 16-byte aligned
//!             from the start of the file
//! data:       for layer in 0..n_layers:
//!               K: [n_kv_heads, n_tokens, head_dim] f32, row-major
//!               V: [n_kv_heads, n_tokens, head_dim] f32, row-major
//! ```
//!
//! Padding the data offset to 16 bytes (rather than just 4) means the data
//! region is aligned for any reasonable SIMD/vector width, not just
//! `f32`'s own 4-byte alignment; `layer_slices` still falls back to a copy
//! if a caller hands it a buffer that isn't aligned at all (e.g. read into
//! an unaligned `Vec<u8>` sub-slice).
//!
//! ## `dtype: "q8_0"`
//!
//! Selected by `Header::dtype`; the container version stays 1 (the header
//! already carries the layout selector, so no version bump is needed) and
//! v1 f32 images remain readable unchanged. Per tensor (`K` or `V`,
//! flattened `[n_kv_heads, n_tokens, head_dim]` row-major, same as f32):
//! values are grouped into blocks of 32 consecutive values along
//! `head_dim` (so `head_dim` must be a multiple of 32 — `head_dim=128` is
//! 4 blocks/row), each block quantized to one `f32` scale
//! (`absmax(block)/127`) plus 32 `i8` values packed 4-per-`u32`
//! little-endian (so a WGSL shader can later read a block as 8 `u32`
//! words directly) = 4 + 32 = 36 bytes per 32 values (1.125 B/value vs 4
//! B/value for f32, ~3.55x smaller once the negligible header overhead is
//! counted). Within one tensor, all block scales are written first as a
//! contiguous `f32` array, then all blocks' packed words follow as a
//! contiguous `u32` array — this mirrors the Q4 repack convention of two
//! separately-bindable buffers (scales, packed data) for a future kernel.
//! Layer order (K then V) and layer count are unchanged from f32.
//!
//! Attention still accumulates in f32 regardless of KV storage dtype;
//! quantizing K/V does not touch the accumulator. See
//! `docs/ENGINE.md`'s "Prefix KV images" section and
//! `trucs.ai/.claude/worktrees/sonos-mcp/docs/kv-cache-images.md`'s "KV
//! quantisation" section for the rationale (no `shader-f16` dependency,
//! llama.cpp-reported q8_0 KV perplexity delta of 0.002-0.05).
//!
//! ## Hashing
//!
//! `model_fingerprint` (named `model_hash` in an earlier version of this
//! module — renamed because it is deliberately **not** a hash of the whole
//! GGUF file) identifies the model without ever reading the full 1.7GB+
//! file: `gguf_header_fingerprint` hashes only the GGUF header region
//! (magic through the end of the tensor-info table — a few KB, contains
//! every tensor's name/shape/dtype, so two different quantizations or
//! checkpoints of "the same" model almost certainly differ here) plus the
//! file size. Cheap enough to recompute on every load, natively (`llm-agent
//! kv-export`, via `gguf::GgufReader::header_bytes`) and in the browser
//! (`web.rs`, from the already-fetched shard bytes — no `crypto.subtle`
//! pass over the whole model needed). `content_fingerprint` (size + first/
//! last 1 MiB) is a coarser alternative kept for callers that don't have a
//! parsed GGUF header handy; either is just an opaque string as far as this
//! module and `prefix_key` are concerned. No `sha2` crate is in this
//! workspace's dependency tree (`cargo tree -i sha2` prints nothing), so
//! this module implements SHA-256 itself (`sha256_reader`) rather than
//! adding a new dependency — this crate is not `Cargo.toml`-owned by this
//! worker in any case.
//!
//! `prefix_key` is `sha256(model_fingerprint || rendered_prefix_text)` —
//! hashing in the *rendered* prompt text (not just the system prompt / tool list
//! separately) means the key is sensitive to chat-template version, system
//! prompt wording, and tool ordering all at once, matching how `web.rs`'s
//! `compute_prefix_len`/`resident_tokens` already treat the rendered
//! prompt as the unit of comparison.

use std::io::{self, Read, Write};

pub const MAGIC: [u8; 6] = *b"KVIMG\0";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Header {
    pub model_fingerprint: String,
    pub prefix_key: String,
    pub tokens: Vec<u32>,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: String,
    pub engine: String,
    pub created: String,
}

#[derive(Debug, thiserror::Error)]
pub enum KvImageError {
    #[error("truncated: need at least {need} bytes, got {got}")]
    Truncated { need: usize, got: usize },
    #[error("bad magic: expected {expected:?}, got {actual:?}")]
    BadMagic { expected: [u8; 6], actual: [u8; 6] },
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(String),
    #[error("invalid header JSON: {0}")]
    InvalidHeader(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
}

/// `n_tokens` implied by `Header::tokens.len()`, used to size each layer's
/// K/V slice.
fn n_tokens(header: &Header) -> usize {
    header.tokens.len()
}

/// Tensor storage dtype for [`KvImage::write`]. `Header::dtype` (a plain
/// string, for forward-compat with dtypes this enum doesn't know about
/// yet) must agree with the variant passed to `write` — see
/// [`Dtype::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Q8_0,
}

impl Dtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::Q8_0 => "q8_0",
        }
    }
}

pub struct KvImage;

impl KvImage {
    /// Write a full image: magic, version, header (padded so tensor data
    /// starts 16-byte aligned from the start of the stream), then each
    /// layer's `(k, v)` slices in order, encoded per `dtype` (see the
    /// module docs' `dtype: "q8_0"` section for the on-disk layout).
    /// `layers_iter` yields `(&[f32] k, &[f32] v)` per layer, each already
    /// flattened `[n_kv_heads, n_tokens, head_dim]`, regardless of
    /// `dtype` — quantization (if any) happens here.
    pub fn write<'a, W: Write, I>(
        w: &mut W,
        header: &Header,
        dtype: Dtype,
        layers_iter: I,
    ) -> Result<(), KvImageError>
    where
        I: IntoIterator<Item = (&'a [f32], &'a [f32])>,
    {
        assert_eq!(
            header.dtype,
            dtype.as_str(),
            "header.dtype={:?} does not match dtype arg={:?}",
            header.dtype,
            dtype.as_str()
        );
        if dtype == Dtype::Q8_0 {
            assert_eq!(
                header.head_dim % 32,
                0,
                "q8_0 requires head_dim % 32 == 0, got {}",
                header.head_dim
            );
        }

        let header_json = serde_json::to_vec(header)?;
        let prefix_len = MAGIC.len() + 4 + 4 + header_json.len();
        let padded_len = prefix_len.div_ceil(16) * 16;
        let pad = padded_len - prefix_len;

        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(header_json.len() as u32).to_le_bytes())?;
        w.write_all(&header_json)?;
        w.write_all(&vec![0u8; pad])?;

        let expect_len = header.n_kv_heads * n_tokens(header) * header.head_dim;
        let mut layers_written = 0;
        for (k, v) in layers_iter {
            assert_eq!(
                k.len(),
                expect_len,
                "layer {layers_written} k length mismatch"
            );
            assert_eq!(
                v.len(),
                expect_len,
                "layer {layers_written} v length mismatch"
            );
            match dtype {
                Dtype::F32 => {
                    w.write_all(bytemuck_f32(k))?;
                    w.write_all(bytemuck_f32(v))?;
                }
                Dtype::Q8_0 => {
                    Self::write_q8_0_tensor(w, k)?;
                    Self::write_q8_0_tensor(w, v)?;
                }
            }
            layers_written += 1;
        }
        assert_eq!(
            layers_written, header.n_layers,
            "layers_iter yielded {layers_written} layers, header says n_layers={}",
            header.n_layers
        );
        Ok(())
    }

    /// Write one tensor's q8_0 encoding: all block scales (contiguous
    /// `f32`) followed by all blocks' packed words (contiguous `u32`).
    fn write_q8_0_tensor<W: Write>(w: &mut W, vals: &[f32]) -> Result<(), KvImageError> {
        let (scales, words) = quantize_q8_0(vals);
        for s in &scales {
            w.write_all(&s.to_le_bytes())?;
        }
        for word in &words {
            w.write_all(&word.to_le_bytes())?;
        }
        Ok(())
    }

    /// Parse magic/version/header from the start of `buf`, returning the
    /// parsed `Header` and the byte offset (into `buf`) where tensor data
    /// begins.
    pub fn read_header(buf: &[u8]) -> Result<(Header, usize), KvImageError> {
        if buf.len() < MAGIC.len() + 8 {
            return Err(KvImageError::Truncated {
                need: MAGIC.len() + 8,
                got: buf.len(),
            });
        }
        let mut magic = [0u8; 6];
        magic.copy_from_slice(&buf[0..6]);
        if magic != MAGIC {
            return Err(KvImageError::BadMagic {
                expected: MAGIC,
                actual: magic,
            });
        }
        let version = u32::from_le_bytes(buf[6..10].try_into().unwrap());
        if version != VERSION {
            return Err(KvImageError::UnsupportedVersion(version));
        }
        let header_len = u32::from_le_bytes(buf[10..14].try_into().unwrap()) as usize;
        let header_start = 14;
        let header_end = header_start + header_len;
        if buf.len() < header_end {
            return Err(KvImageError::Truncated {
                need: header_end,
                got: buf.len(),
            });
        }
        let header: Header = serde_json::from_slice(&buf[header_start..header_end])?;
        let padded_len = header_end.div_ceil(16) * 16;
        if buf.len() < padded_len {
            return Err(KvImageError::Truncated {
                need: padded_len,
                got: buf.len(),
            });
        }
        Ok((header, padded_len))
    }

    /// Zero-copy `(k, v)` views per layer starting at `data_offset` (as
    /// returned by `read_header`), unless `buf`'s start (and thus
    /// `data_offset`) isn't 4-byte aligned in memory, in which case each
    /// layer's slice is copied into an owned, aligned buffer instead.
    pub fn layer_slices<'a>(
        buf: &'a [u8],
        header: &Header,
    ) -> Result<Vec<LayerSlice<'a>>, KvImageError> {
        let data_offset = {
            let header_json_len = serde_json::to_vec(header)?.len();
            // Recompute the same padding rule `write` used, from the known
            // header length recorded in the file itself — callers pass the
            // offset back from `read_header` in the normal path; this path
            // is only used when a caller has `buf` + `Header` without the
            // offset. Prefer the `read_header`-returned offset when
            // available (see `layer_slices_at`).
            let prefix_len = MAGIC.len() + 4 + 4 + header_json_len;
            prefix_len.div_ceil(16) * 16
        };
        Self::layer_slices_at(buf, header, data_offset)
    }

    /// Same as `layer_slices` but takes an explicit `data_offset` (as
    /// returned by `read_header`) instead of recomputing it.
    pub fn layer_slices_at<'a>(
        buf: &'a [u8],
        header: &Header,
        data_offset: usize,
    ) -> Result<Vec<LayerSlice<'a>>, KvImageError> {
        let per_tensor = header.n_kv_heads * n_tokens(header) * header.head_dim;
        let per_tensor_bytes = per_tensor * 4;
        let per_layer_bytes = per_tensor_bytes * 2;
        let need = data_offset + per_layer_bytes * header.n_layers;
        if buf.len() < need {
            return Err(KvImageError::Truncated {
                need,
                got: buf.len(),
            });
        }

        let base_ptr_aligned = (buf.as_ptr() as usize + data_offset).is_multiple_of(4);

        let mut out = Vec::with_capacity(header.n_layers);
        let mut off = data_offset;
        for _ in 0..header.n_layers {
            let k_bytes = &buf[off..off + per_tensor_bytes];
            off += per_tensor_bytes;
            let v_bytes = &buf[off..off + per_tensor_bytes];
            off += per_tensor_bytes;

            let slice = if base_ptr_aligned {
                LayerSlice {
                    k: Cow::Borrowed(bytes_to_f32(k_bytes)),
                    v: Cow::Borrowed(bytes_to_f32(v_bytes)),
                }
            } else {
                LayerSlice {
                    k: Cow::Owned(copy_to_f32(k_bytes)),
                    v: Cow::Owned(copy_to_f32(v_bytes)),
                }
            };
            out.push(slice);
        }
        Ok(out)
    }

    /// dtype-agnostic per-layer accessor: returns `(k, v)` as owned `Vec<f32>`,
    /// each flattened `[n_kv_heads, n_tokens, head_dim]`. For `dtype:
    /// "f32"` this just copies the raw bytes (see [`Self::layer_slices_at`]
    /// for a zero-copy alternative in that case); for `dtype: "q8_0"` it
    /// dequantizes each block. This is the path an importer (e.g.
    /// `KvCache::import_prefix`, which already takes `Vec<f32>`) should use
    /// when it doesn't want to special-case dtype itself.
    pub fn layer_f32(
        buf: &[u8],
        header: &Header,
        data_offset: usize,
        layer: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), KvImageError> {
        let per_tensor = header.n_kv_heads * n_tokens(header) * header.head_dim;
        let per_tensor_bytes = Self::tensor_bytes(header, per_tensor);
        let per_layer_bytes = per_tensor_bytes * 2;
        let off = data_offset + per_layer_bytes * layer;
        let need = off + per_layer_bytes;
        if buf.len() < need {
            return Err(KvImageError::Truncated {
                need,
                got: buf.len(),
            });
        }
        let k_bytes = &buf[off..off + per_tensor_bytes];
        let v_bytes = &buf[off + per_tensor_bytes..off + per_layer_bytes];
        let (k, v) = match header.dtype.as_str() {
            "f32" => (copy_to_f32(k_bytes), copy_to_f32(v_bytes)),
            "q8_0" => (
                Self::decode_q8_0_tensor(k_bytes, per_tensor),
                Self::decode_q8_0_tensor(v_bytes, per_tensor),
            ),
            other => return Err(KvImageError::UnsupportedDtype(other.to_string())),
        };
        Ok((k, v))
    }

    /// Byte size of one tensor (`K` or `V`) under `header.dtype`.
    fn tensor_bytes(header: &Header, per_tensor_values: usize) -> usize {
        match header.dtype.as_str() {
            "f32" => per_tensor_values * 4,
            "q8_0" => {
                assert_eq!(
                    per_tensor_values % 32,
                    0,
                    "q8_0 tensor length {per_tensor_values} not a multiple of block size 32"
                );
                let n_blocks = per_tensor_values / 32;
                n_blocks * 36 // 4 bytes scale + 32 bytes (8 u32 words) per block
            }
            other => panic!("unsupported dtype {other:?}"),
        }
    }

    fn decode_q8_0_tensor(bytes: &[u8], per_tensor: usize) -> Vec<f32> {
        let n_blocks = per_tensor / 32;
        let scales_bytes = &bytes[..n_blocks * 4];
        let words_bytes = &bytes[n_blocks * 4..n_blocks * 4 + n_blocks * 32];
        let scales: Vec<f32> = scales_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let words: Vec<u32> = words_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        dequantize_q8_0(&scales, &words, per_tensor)
    }
}

/// Quantize `vals` (length a multiple of 32) into per-32-block `(scales,
/// words)`: one `f32` scale (`absmax/127`) per block, and the block's 32
/// values as `i8` packed 4-per-`u32` little-endian (`words.len() ==
/// vals.len() / 4`). A zero-valued block gets scale `0.0` and all-zero
/// words (dequantizing back to exact zero, not division by zero).
pub fn quantize_q8_0(vals: &[f32]) -> (Vec<f32>, Vec<u32>) {
    assert_eq!(
        vals.len() % 32,
        0,
        "quantize_q8_0: length {} not a multiple of 32",
        vals.len()
    );
    let n_blocks = vals.len() / 32;
    let mut scales = Vec::with_capacity(n_blocks);
    let mut words = Vec::with_capacity(n_blocks * 8);

    for block in vals.chunks_exact(32) {
        let absmax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let scale = absmax / 127.0;
        scales.push(scale);

        let mut q = [0i8; 32];
        if scale != 0.0 {
            for (qi, &x) in q.iter_mut().zip(block.iter()) {
                *qi = (x / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        for word_bytes in q.chunks_exact(4) {
            let word = u32::from_le_bytes([
                word_bytes[0] as u8,
                word_bytes[1] as u8,
                word_bytes[2] as u8,
                word_bytes[3] as u8,
            ]);
            words.push(word);
        }
    }
    (scales, words)
}

/// Inverse of [`quantize_q8_0`]: `scales.len() * 32 == len`, `words.len()
/// == scales.len() * 8`.
pub fn dequantize_q8_0(scales: &[f32], words: &[u32], len: usize) -> Vec<f32> {
    assert_eq!(
        len % 32,
        0,
        "dequantize_q8_0: length {len} not a multiple of 32"
    );
    let n_blocks = len / 32;
    assert_eq!(scales.len(), n_blocks);
    assert_eq!(words.len(), n_blocks * 8);

    let mut out = Vec::with_capacity(len);
    for b in 0..n_blocks {
        let scale = scales[b];
        for &word in &words[b * 8..b * 8 + 8] {
            let bytes = word.to_le_bytes();
            for byte in bytes {
                out.push((byte as i8) as f32 * scale);
            }
        }
    }
    out
}

use std::borrow::Cow;

pub struct LayerSlice<'a> {
    pub k: Cow<'a, [f32]>,
    pub v: Cow<'a, [f32]>,
}

fn bytemuck_f32(x: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns and any alignment
    // requirement is <= the byte view we produce (we only read `u8`s from
    // it), so a plain reinterpret is sound.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn bytes_to_f32(buf: &[u8]) -> &[f32] {
    // SAFETY: caller (`layer_slices_at`) only takes this branch after
    // checking `buf.as_ptr()` (offset into the original buffer) is
    // 4-aligned, and `buf.len()` here is always a multiple of 4 (it's
    // `per_tensor_bytes = per_tensor * 4`).
    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) }
}

fn copy_to_f32(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `sha256(model_fingerprint || rendered_prefix_text)`, hex-encoded.
pub fn prefix_key(model_fingerprint: &str, rendered_prefix_text: &str) -> String {
    let mut h = Sha256::new();
    h.update(model_fingerprint.as_bytes());
    h.update(rendered_prefix_text.as_bytes());
    h.hex_digest()
}

/// Cheap model-identity fingerprint: `sha256(file_size_le_u64 ||
/// header_bytes)`, where `header_bytes` is the GGUF header region (magic
/// through the end of the tensor-info table — a few KB, from
/// `gguf::GgufReader::header_bytes`/`data_section_offset` — contains every
/// tensor's name/shape/dtype). Deliberately **not** a hash of the full
/// (1.7GB+) file: cheap enough to recompute on every native `kv-export`
/// run and every browser `load()`, unlike `sha256_reader`/
/// `content_fingerprint` over the whole GGUF.
pub fn gguf_header_fingerprint(file_size: u64, header_bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(&file_size.to_le_bytes());
    h.update(header_bytes);
    h.hex_digest()
}

/// sha256 over the full contents read from `r`, hex-encoded. Streams input
/// in 64KiB chunks rather than requiring the whole file in memory.
pub fn sha256_reader<R: Read>(r: &mut R) -> io::Result<String> {
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.hex_digest())
}

/// Cheap fingerprint for large files where a full SHA-256 pass is too slow:
/// `sha256(file_size_le_u64 || first up to 1MiB || last up to 1MiB)`. Not a
/// content hash of the whole file — two different files of the same size
/// sharing the same first/last MiB collide. Good enough as a
/// change-detection key for a GGUF file that is otherwise identified by
/// its path/URL.
pub fn content_fingerprint<R: Read + std::io::Seek>(r: &mut R) -> io::Result<String> {
    use std::io::SeekFrom;
    const CHUNK: u64 = 1024 * 1024;
    let size = r.seek(SeekFrom::End(0))?;

    let mut h = Sha256::new();
    h.update(&size.to_le_bytes());

    r.seek(SeekFrom::Start(0))?;
    let head_len = CHUNK.min(size) as usize;
    let mut head = vec![0u8; head_len];
    r.read_exact(&mut head)?;
    h.update(&head);

    if size > CHUNK {
        let tail_start = size.saturating_sub(CHUNK);
        r.seek(SeekFrom::Start(tail_start))?;
        let tail_len = (size - tail_start) as usize;
        let mut tail = vec![0u8; tail_len];
        r.read_exact(&mut tail)?;
        h.update(&tail);
    }

    Ok(h.hex_digest())
}

/// Minimal streaming SHA-256 (FIPS 180-4). No external dependency: `sha2`
/// is not in this workspace's tree (`cargo tree -i sha2` — checked before
/// writing this), and this crate doesn't own `Cargo.toml` in this task's
/// split anyway.
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len += data.len() as u64;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.process_block(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: [u8; 64] = data[..64].try_into().unwrap();
            self.process_block(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len * 8;
        let mut pad = vec![0x80u8];
        let rem = (self.total_len + 1) % 64;
        let zeros = if rem <= 56 { 56 - rem } else { 120 - rem };
        pad.extend(std::iter::repeat_n(0u8, zeros as usize));
        pad.extend_from_slice(&bit_len.to_be_bytes());
        self.update_no_len_track(&pad);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn update_no_len_track(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.process_block(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: [u8; 64] = data[..64].try_into().unwrap();
            self.process_block(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn hex_digest(self) -> String {
        let digest = self.finalize();
        let mut s = String::with_capacity(64);
        for b in digest {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(test)]
mod hash_tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        // NIST/RFC test vectors.
        assert_eq!(
            sha256_reader(&mut &b""[..]).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_reader(&mut &b"abc"[..]).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_reader(&mut &b"The quick brown fox jumps over the lazy dog"[..]).unwrap(),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }
}
