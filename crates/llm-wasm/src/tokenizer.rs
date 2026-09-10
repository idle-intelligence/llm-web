//! BPE tokenizer wrapper (`tokenizers` crate, fancy-regex backend) for the
//! Qwen2 vocab used by xLAM-2-3b-fc-r. Owned by phase 1a.
//!
//! Requires the `native` or `web` cargo feature (both pull in the optional
//! `tokenizers` dependency — see `crates/llm-wasm/Cargo.toml`); both are on
//! by default for this crate (`default = ["wgpu", "native"]`), so this file
//! doesn't need its own `cfg` gate for the build commands `docs/OVERVIEW.md`
//! documents.

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer as HfTokenizer;

/// The two turn-ending tokens xLAM-2 stops generation on — see
/// `docs/MODELS.md` §1 (`generation_config.json`'s `eos_token_id: [151645,
/// 151643]`).
const EOS_TOKEN_STRS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

pub struct Tokenizer {
    inner: HfTokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load from the raw bytes of a `tokenizer.json` file.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let inner = HfTokenizer::from_bytes(bytes)
            .map_err(|e| anyhow!("failed to load tokenizer.json: {e}"))?;
        let eos_ids = EOS_TOKEN_STRS
            .iter()
            .filter_map(|tok| inner.token_to_id(tok))
            .collect();
        Ok(Self { inner, eos_ids })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("decode failed: {e}"))
    }

    /// Ids of `<|im_end|>` and `<|endoftext|>`, i.e. the ids generation
    /// should stop on — see `docs/MODELS.md` §1.
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Total vocab size (including added/special tokens), i.e. the id
    /// space a `grammar::TokenMask` must cover.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The exact byte string a single token id decodes to, keeping special
    /// tokens (`decode(.., skip_special=false)`) so callers can special-case
    /// them (e.g. EOS) rather than have them silently vanish.
    ///
    /// Used by `grammar::Grammar` to precompute, once per generation, the
    /// byte string of every vocab entry so token-level acceptance can be
    /// checked by walking those bytes through the character matcher.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.inner
            .decode(&[id], false)
            .unwrap_or_default()
            .into_bytes()
    }
}
