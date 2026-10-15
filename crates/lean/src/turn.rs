//! One inference turn over already-tokenized text, on either backend, plus
//! the two pieces of state [`crate::Lean`] keeps between turns:
//!
//! - [`PrefixCache`]: the token ids the KV cache currently holds, left over
//!   from the previous turn. The next turn reuses the longest prefix its
//!   own ids share with them and prefills only the remainder; with nothing
//!   in common it prefills from position 0. The cache serves one client at
//!   a time: two clients sharing one engine evict each other's prefix
//!   (nothing breaks, each call just costs a full prefill).
//! - [`TextStream`]: incremental detokenization. Decoding one id at a time
//!   is wrong for byte-level BPE (a multi-byte character spans tokens) and
//!   for SentencePiece-style decoders (a leading space is dropped at the
//!   start of a decode); this decodes a sliding window of the accumulated
//!   ids and emits only the new text, holding back a trailing incomplete
//!   character. The concatenated deltas equal `decode(all ids)`.

use std::future::Future;

use tokenizers::Tokenizer;

use lean_core::cpu::{forward_decode_step as cpu_forward_decode_step, forward_prefill as cpu_forward_prefill, CpuKvCache, CpuModel};
use lean_core::engine::Engine;
use crate::error::LeanError;
use crate::generate::decode_loop;
use lean_core::model::{argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache};
use crate::sampling::{id_counts, sample, Rng64, SamplingParams};
use lean_core::resident_sampling::sample_reference;

/// Everything one `generate` call needs beyond its text: how many tokens
/// to produce and how to pick them. `temperature <= 0` is greedy.
/// `mask_bits` is this crate's mask-bitset convention
/// (`model::build_mask_bitset`); `None` or empty means unmasked.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub mask_bits: Option<Vec<u32>>,
}

impl Default for GenParams {
    fn default() -> Self {
        let s = SamplingParams::default();
        GenParams { max_new_tokens: 256, temperature: s.temperature, top_k: s.top_k, top_p: s.top_p, repetition_penalty: s.repetition_penalty, seed: s.seed, mask_bits: None }
    }
}

impl GenParams {
    pub fn sampling(&self) -> SamplingParams {
        SamplingParams { temperature: self.temperature, top_k: self.top_k, top_p: self.top_p, repetition_penalty: self.repetition_penalty, seed: self.seed }
    }

    /// The mask, if any, after checking it covers the vocabulary.
    pub(crate) fn mask(&self, vocab_size: usize) -> Result<Option<&[u32]>, LeanError> {
        match self.mask_bits.as_deref() {
            None | Some([]) => Ok(None),
            Some(bits) if bits.len() * 32 < vocab_size => Err(LeanError::Other(anyhow::anyhow!("mask_bits is shorter than vocab_size / 32"))),
            Some(bits) => Ok(Some(bits)),
        }
    }
}

/// The token ids the KV cache currently holds - see the module doc.
#[derive(Default)]
pub(crate) struct PrefixCache {
    cached_ids: Vec<u32>,
}

impl PrefixCache {
    /// How many of `ids`' leading tokens are already resident at `kv_len`:
    /// the longest common prefix of the cached ids and `ids`, capped so at
    /// least one id is left to prefill (a turn always needs the logits of
    /// its last prompt position).
    fn reuse(&self, ids: &[u32], kv_len: usize) -> usize {
        let usable = self.cached_ids.len().min(kv_len);
        let lcp = self.cached_ids[..usable].iter().zip(ids).take_while(|(a, b)| a == b).count();
        lcp.min(ids.len().saturating_sub(1))
    }

    /// Records what a turn left resident: `ids` plus `generated`, truncated
    /// to `kv_len` (the last generated id is never fed back into the cache).
    fn commit(&mut self, ids: Vec<u32>, generated: &[u32], kv_len: usize) {
        let mut cached = ids;
        cached.extend_from_slice(generated);
        cached.truncate(kv_len);
        self.cached_ids = cached;
    }
}

/// Tokenizes `text` as-is: special tokens written in the text (a chat
/// template's `<|im_start|>`) are matched by the tokenizer's added
/// vocabulary, and nothing is added around it.
pub(crate) fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, LeanError> {
    let enc = tokenizer.encode(text, false).map_err(|e| LeanError::Other(anyhow::anyhow!("tokenizer encode failed: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

/// `ids` plus `max_new_tokens` must fit `max_ctx`; checked before anything
/// is mutated, so an error leaves the engine exactly as it was.
pub(crate) fn check_fits(ids: &[u32], max_new_tokens: u32, max_ctx: usize) -> Result<(), LeanError> {
    if ids.is_empty() {
        return Err(LeanError::Other(anyhow::anyhow!("the prompt tokenizes to nothing")));
    }
    let needed = ids.len() + max_new_tokens as usize;
    if needed > max_ctx {
        return Err(LeanError::ContextOverflow { needed, available: max_ctx });
    }
    Ok(())
}

/// See the module doc. `prev..cur` is the window decoded at the previous
/// emission; the next decode covers `prev..`, so a decoder that drops the
/// first token's leading space drops it from both and the difference is
/// exact.
#[derive(Default)]
pub struct TextStream {
    ids: Vec<u32>,
    prev: usize,
    cur: usize,
}

impl TextStream {
    pub fn new() -> Self {
        Self::default()
    }

    fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, LeanError> {
        if ids.is_empty() {
            return Ok(String::new());
        }
        tokenizer.decode(ids, true).map_err(|e| LeanError::Other(anyhow::anyhow!("tokenizer decode failed: {e}")))
    }

    fn delta(&self, tokenizer: &Tokenizer, flush: bool) -> Result<Option<String>, LeanError> {
        let prev_text = Self::decode(tokenizer, &self.ids[self.prev..self.cur])?;
        let text = Self::decode(tokenizer, &self.ids[self.prev..])?;
        if text.len() > prev_text.len() && text.starts_with(&prev_text) && (flush || !text.ends_with('\u{FFFD}')) {
            return Ok(Some(text[prev_text.len()..].to_string()));
        }
        Ok(None)
    }

    /// Adds one generated id; returns the text it completes (often "", e.g.
    /// for a special token or the first half of a multi-byte character).
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String, LeanError> {
        self.ids.push(id);
        match self.delta(tokenizer, false)? {
            Some(d) => {
                self.prev = self.cur;
                self.cur = self.ids.len();
                Ok(d)
            }
            None => Ok(String::new()),
        }
    }

    /// Text still held back at the end of generation (an incomplete
    /// character cut off by `max_new_tokens` or an abort); usually "".
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String, LeanError> {
        let d = self.delta(tokenizer, true)?.unwrap_or_default();
        self.prev = self.cur;
        self.cur = self.ids.len();
        Ok(d)
    }
}

/// Logits of disallowed ids set to the value `mask_logits.wgsl` writes, so
/// the CPU picks exactly what the GPU picks under the same bitset.
fn apply_mask_cpu(logits: &mut [f32], mask_bits: Option<&[u32]>) {
    if let Some(bits) = mask_bits {
        for (i, l) in logits.iter_mut().enumerate() {
            if (bits[i / 32] >> (i % 32)) & 1 == 0 {
                *l = -3.4e38;
            }
        }
    }
}

/// CPU mirror of `generate::decode_loop` (same greedy/sampling choices,
/// same stop rules), with a `yield_now` future awaited after every token so
/// a browser worker can run its message handlers (the abort path) between
/// tokens.
#[allow(clippy::too_many_arguments)]
async fn cpu_decode_loop<F: Future<Output = ()>>(
    model: &CpuModel,
    cache: &mut CpuKvCache,
    mut first_logits: Vec<f32>,
    mask_bits: Option<&[u32]>,
    max_new_tokens: u32,
    params: &SamplingParams,
    history: &mut Vec<u32>,
    mut on_token: impl FnMut(u32),
    mut should_stop: impl FnMut() -> bool,
    mut yield_now: impl FnMut() -> F,
) -> Vec<u32> {
    let greedy = params.is_greedy();
    let mut rng = Rng64::new(params.seed);
    let vocab = model.config.vocab_size;
    // The same sampler as the GPU's for the settings it serves, so both
    // backends give the same reply for the same seed.
    let resident = params.resident(vocab);
    let mut counts = if resident.is_some() { id_counts(vocab, history) } else { Vec::new() };
    apply_mask_cpu(&mut first_logits, mask_bits);
    let mut next_id = match &resident {
        Some(s) => sample_reference(&first_logits, &counts, s, 0),
        None if greedy => argmax(&first_logits),
        None => sample(&mut first_logits, params, history, &mut rng),
    };

    let mut generated = Vec::with_capacity(max_new_tokens as usize);
    for step in 0..max_new_tokens {
        if model.config.eos_token_ids.contains(&next_id) || should_stop() {
            break;
        }
        generated.push(next_id);
        history.push(next_id);
        if let Some(c) = counts.get_mut(next_id as usize) {
            *c += 1;
        }
        on_token(next_id);
        yield_now().await;
        if should_stop() {
            break;
        }
        let mut logits = cpu_forward_decode_step(model, cache, next_id);
        apply_mask_cpu(&mut logits, mask_bits);
        next_id = match &resident {
            Some(s) => sample_reference(&logits, &counts, s, step + 1),
            None if greedy => argmax(&logits),
            None => sample(&mut logits, params, history, &mut rng),
        };
    }
    generated
}

/// Streams generated ids as text through `on_token`, then flushes the tail
/// and returns the whole reply.
struct Streamer<'a, T: FnMut(&str)> {
    tokenizer: &'a Tokenizer,
    stream: TextStream,
    on_token: T,
    err: Option<LeanError>,
}

impl<'a, T: FnMut(&str)> Streamer<'a, T> {
    fn new(tokenizer: &'a Tokenizer, on_token: T) -> Self {
        Streamer { tokenizer, stream: TextStream::new(), on_token, err: None }
    }

    fn push(&mut self, id: u32) {
        match self.stream.push(self.tokenizer, id) {
            Ok(delta) if !delta.is_empty() => (self.on_token)(&delta),
            Ok(_) => {}
            Err(e) => self.err = Some(e),
        }
    }

    fn finish(mut self, generated: &[u32]) -> Result<String, LeanError> {
        if let Some(e) = self.err {
            return Err(e);
        }
        let tail = self.stream.finish(self.tokenizer)?;
        if !tail.is_empty() {
            (self.on_token)(&tail);
        }
        TextStream::decode(self.tokenizer, generated)
    }
}

/// One turn on the GPU over `ids` (already checked with [`check_fits`]):
/// prefill what is not cached (from position 0, or after rewinding to the
/// reused prefix), decode, stream the reply, record what the turn left
/// resident in `prefix`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn gpu_turn(
    engine: &Engine,
    model: &GpuModel,
    cache: &mut KvCache,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    prefix: &mut PrefixCache,
    tokenizer: &Tokenizer,
    ids: Vec<u32>,
    params: &GenParams,
    mask: Option<&wgpu::Buffer>,
    on_token: impl FnMut(&str),
    should_stop: impl FnMut() -> bool,
) -> Result<String, LeanError> {
    let reuse = prefix.reuse(&ids, cache.kv_len as usize);
    let logits = if reuse == 0 {
        forward_prefill(engine, model, cache, &ids, cos, sin, mask).await
    } else {
        cache.kv_len = reuse as u32;
        forward_prefill_suffix(engine, model, cache, &ids[reuse..], cos, sin, mask).await
    };
    let mut history = ids.clone();
    let mut streamer = Streamer::new(tokenizer, on_token);
    let generated = decode_loop(engine, model, cache, logits, cos, sin, mask, params.max_new_tokens, &params.sampling(), &mut history, |id| streamer.push(id), should_stop).await;
    let reply = streamer.finish(&generated);
    prefix.commit(ids, &generated, cache.kv_len as usize);
    reply
}

/// One turn on the CPU, same steps as [`gpu_turn`]; the CPU prefill
/// continues from the cache's resident prefix directly.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn cpu_turn<F: Future<Output = ()>>(
    model: &CpuModel,
    cache: &mut CpuKvCache,
    prefix: &mut PrefixCache,
    tokenizer: &Tokenizer,
    ids: Vec<u32>,
    params: &GenParams,
    mask_bits: Option<&[u32]>,
    on_token: impl FnMut(&str),
    should_stop: impl FnMut() -> bool,
    yield_now: impl FnMut() -> F,
) -> Result<String, LeanError> {
    let reuse = prefix.reuse(&ids, cache.kv_len);
    cache.kv_len = reuse;
    let logits = cpu_forward_prefill(model, cache, &ids[reuse..]);
    let mut history = ids.clone();
    let mut streamer = Streamer::new(tokenizer, on_token);
    let generated = cpu_decode_loop(model, cache, logits, mask_bits, params.max_new_tokens, &params.sampling(), &mut history, |id| streamer.push(id), should_stop, yield_now).await;
    let reply = streamer.finish(&generated);
    prefix.commit(ids, &generated, cache.kv_len);
    reply
}
