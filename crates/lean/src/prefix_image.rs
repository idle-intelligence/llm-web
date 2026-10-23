//! Prefix KV images: the KV cache of a prompt prefix as bytes, written by
//! [`crate::Lean::save_prefix`] and read back by
//! [`crate::Lean::restore_prefix`], so a long fixed prefix (a system prompt,
//! a tools block) is prefilled once and restored in later sessions. Where
//! the bytes are kept is the caller's choice.
//!
//! Layout, little-endian:
//!
//! ```text
//! offset size
//!   0     8   magic "LEANKVIM"
//!   8     4   format version (FORMAT_VERSION)
//!  12     4   dtype (0 = f32)
//!  16     4   num_layers
//!  20     4   num_kv_heads
//!  24     4   head_dim
//!  28     4   positions (the prefix's token count)
//!  32     4   backend (0 = WebGPU, 1 = CPU)
//!  36     4   reserved, 0
//!  40    16   model fingerprint (GGUF header bytes and file length)
//!  56    16   tokenizer fingerprint (tokenizer.json bytes)
//!  72    16   engine fingerprint (this build's LEAN_ENGINE_HASH)
//!  88    16   prefix fingerprint (the prefix's token ids)
//! 104    24   reserved, 0
//! 128         per layer: K then V, each [num_kv_heads, positions, head_dim] f32
//! ```
//!
//! Everything in the header is fixed by the engine, the loaded files and
//! the prefix text, before any forward pass: the key is a hash of bytes
//! 8..104, and a restore refuses an image whose header differs in any field.
//! The hashes are not cryptographic: they guard against stale images, not
//! against a forged one.

use std::fmt;

pub(crate) const HEADER_LEN: usize = 128;
const MAGIC: &[u8; 8] = b"LEANKVIM";
pub(crate) const FORMAT_VERSION: u32 = 1;
const DTYPE_F32: u32 = 0;

/// Why [`crate::Lean::restore_prefix`] refused an image. Every case means
/// the same thing to a caller: prefill the prefix again
/// ([`crate::Lean::save_prefix`]) and overwrite the stored image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixImageError {
    /// The bytes do not start with a prefix image header.
    NotAnImage,
    /// The image was written for something else than this engine, these
    /// files and this prefix; the field names the first difference:
    /// `"format"`, `"dtype"`, `"engine"`, `"model"`, `"tokenizer"`,
    /// `"backend"`, `"shape"` or `"prefix"`.
    Stale(&'static str),
    /// The source ended or failed before the last tensor. The engine has
    /// dropped the partly written cache and stays usable.
    Truncated,
}

impl fmt::Display for PrefixImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrefixImageError::NotAnImage => write!(f, "not a prefix KV image"),
            PrefixImageError::Stale(field) => write!(f, "stale prefix KV image: {field} differs"),
            PrefixImageError::Truncated => write!(f, "prefix KV image is truncated"),
        }
    }
}

/// 128-bit non-cryptographic hash: two 64-bit multiply-rotate lanes over
/// 8-byte words, fed the input length at the end.
#[derive(Clone)]
pub(crate) struct Hash128 {
    a: u64,
    b: u64,
    len: u64,
    tail: Vec<u8>,
}

impl Hash128 {
    pub(crate) fn new() -> Self {
        Hash128 { a: 0x243f_6a88_85a3_08d3, b: 0x1319_8a2e_0370_7344, len: 0, tail: Vec::with_capacity(8) }
    }

    fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        self.b = (self.b ^ w.rotate_left(32)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f).rotate_left(31) ^ self.a;
    }

    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        if !self.tail.is_empty() {
            let take = (8 - self.tail.len()).min(bytes.len());
            self.tail.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.tail.len() < 8 {
                return;
            }
            let w = u64::from_le_bytes(self.tail[..8].try_into().unwrap());
            self.tail.clear();
            self.word(w);
        }
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.word(u64::from_le_bytes(*w));
        }
        self.tail.extend_from_slice(rest);
    }

    pub(crate) fn finish(mut self) -> [u8; 16] {
        let mut last = [0u8; 8];
        last[..self.tail.len()].copy_from_slice(&self.tail);
        let n = self.tail.len() as u64;
        self.word(u64::from_le_bytes(last) ^ (n << 56));
        self.word(self.len);
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&fmix(self.a ^ self.b.rotate_left(17)).to_le_bytes());
        out[8..].copy_from_slice(&fmix(self.b ^ self.a.rotate_left(41)).to_le_bytes());
        out
    }
}

fn fmix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

pub(crate) fn hash(bytes: &[u8]) -> [u8; 16] {
    let mut h = Hash128::new();
    h.update(bytes);
    h.finish()
}

/// What an engine needs to know to write or check an image header.
#[derive(Clone)]
pub(crate) struct ImageEnv {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub backend: u32,
    pub model: [u8; 16],
    pub tokenizer: [u8; 16],
}

pub(crate) fn engine_fingerprint() -> [u8; 16] {
    hash(env!("LEAN_ENGINE_HASH").as_bytes())
}

pub(crate) fn ids_fingerprint(ids: &[u32]) -> [u8; 16] {
    let mut h = Hash128::new();
    for id in ids {
        h.update(&id.to_le_bytes());
    }
    h.finish()
}

pub(crate) fn header(env: &ImageEnv, ids: &[u32]) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..8].copy_from_slice(MAGIC);
    for (off, v) in [(8, FORMAT_VERSION), (12, DTYPE_F32), (16, env.num_layers), (20, env.num_kv_heads), (24, env.head_dim), (28, ids.len() as u32), (32, env.backend)] {
        h[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    h[40..56].copy_from_slice(&env.model);
    h[56..72].copy_from_slice(&env.tokenizer);
    h[72..88].copy_from_slice(&engine_fingerprint());
    h[88..104].copy_from_slice(&ids_fingerprint(ids));
    h
}

/// The storage key of the image `header` describes: 32 hex characters.
pub(crate) fn key(header: &[u8; HEADER_LEN]) -> String {
    hash(&header[8..104]).iter().map(|b| format!("{b:02x}")).collect()
}

/// Compares a stored header with the one this engine would write for the
/// same prefix, field by field, so the error names what changed.
pub(crate) fn check(found: &[u8; HEADER_LEN], expected: &[u8; HEADER_LEN]) -> Result<(), PrefixImageError> {
    if &found[0..8] != MAGIC {
        return Err(PrefixImageError::NotAnImage);
    }
    let fields: [(&'static str, std::ops::Range<usize>); 8] = [("format", 8..12), ("dtype", 12..16), ("engine", 72..88), ("model", 40..56), ("tokenizer", 56..72), ("backend", 32..36), ("shape", 16..28), ("prefix", 28..32)];
    for (name, r) in fields {
        if found[r.clone()] != expected[r] {
            return Err(PrefixImageError::Stale(name));
        }
    }
    if found[88..104] != expected[88..104] {
        return Err(PrefixImageError::Stale("prefix"));
    }
    if found[..] != expected[..] {
        return Err(PrefixImageError::Stale("format"));
    }
    Ok(())
}

/// Bytes of one K or V tensor of the image.
pub(crate) fn tensor_bytes(env: &ImageEnv, positions: usize) -> usize {
    env.num_kv_heads as usize * positions * env.head_dim as usize * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_independent_of_chunking() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let whole = hash(&data);
        let mut h = Hash128::new();
        for c in data.chunks(7) {
            h.update(c);
        }
        assert_eq!(h.finish(), whole);
        assert_ne!(hash(&data[..999]), whole);
    }

    fn env() -> ImageEnv {
        ImageEnv { num_layers: 2, num_kv_heads: 2, head_dim: 64, backend: 0, model: [1; 16], tokenizer: [2; 16] }
    }

    #[test]
    fn any_changed_input_changes_key_and_fails_check() {
        let base = header(&env(), &[1, 2, 3]);
        assert!(check(&base, &base).is_ok());
        let mut other_model = env();
        other_model.model[0] = 9;
        let mut other_backend = env();
        other_backend.backend = 1;
        let cases = [(header(&env(), &[1, 2, 4]), "prefix"), (header(&env(), &[1, 2, 3, 4]), "prefix"), (header(&other_model, &[1, 2, 3]), "model"), (header(&other_backend, &[1, 2, 3]), "backend")];
        for (h, field) in cases {
            assert_ne!(key(&h), key(&base));
            assert_eq!(check(&h, &base), Err(PrefixImageError::Stale(field)));
        }
        let mut old_engine = base;
        old_engine[72] ^= 1;
        assert_ne!(key(&old_engine), key(&base));
        assert_eq!(check(&old_engine, &base), Err(PrefixImageError::Stale("engine")));
        let mut old_format = base;
        old_format[8] = 0;
        assert_eq!(check(&old_format, &base), Err(PrefixImageError::Stale("format")));
        assert_eq!(check(&[0u8; HEADER_LEN], &base), Err(PrefixImageError::NotAnImage));
    }
}
