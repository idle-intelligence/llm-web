//! One inference turn, shared by both browser engines (`web.rs`'s
//! `LeanEngine` on WebGPU and `LeanEngineCpu` on CPU threads or one
//! thread), the native CLI example (`examples/chat.rs`), and the native
//! tests, so the wasm surface and the CLI are both thin wrappers over the
//! code the tests exercise.
//!
//! Conversation state - which messages to send on the next call, when to
//! drop old ones - is the caller's job, not this engine's: a call here
//! takes the whole conversation (`messages`) every time and returns one
//! reply. What this module keeps between calls is purely an optimization,
//! invisible to the caller:
//!
//! - [`PrefixCache`]: the token ids this engine's KV cache currently holds,
//!   left over from the previous call. The next call renders its own
//!   `messages` through the chat template, tokenizes it, and reuses the
//!   longest prefix the two id sequences agree on; only the remainder is
//!   prefilled. If nothing matches (the caller dropped an early turn, or
//!   this is the first call), the whole conversation is prefilled from
//!   scratch - this is `forward_prefill`'s default behavior regardless of
//!   the cache's prior contents, so no explicit reset is needed. The cache
//!   serves one caller at a time: two callers sharing one engine evict
//!   each other's prefix, since each call's reuse check only ever sees the
//!   previous call's own result.
//! - [`TextStream`]: incremental detokenization, unchanged in shape from
//!   before this module dropped conversation history. Decoding one id at a
//!   time is wrong for byte-level BPE (a multi-byte character spans
//!   tokens) and for SentencePiece-style decoders (a leading space is
//!   dropped at the start of a decode); this decodes a sliding window of
//!   the accumulated ids and emits only the new text, holding back a
//!   trailing incomplete character. The concatenated deltas equal
//!   `decode(all ids)`.
//! - [`gpu_generate`] / [`cpu_generate`]: one turn on each backend. Takes
//!   `messages`, streams the reply as text (not token ids - see
//!   `GenParams`'s doc comment on why the token-level primitives this
//!   builds on, `forward_prefill`/`forward_prefill_suffix`/`decode_loop`,
//!   are the right level for a caller that wants raw ids instead).

use std::future::Future;

use tokenizers::Tokenizer;

use crate::chat_template::render_conversation;
use crate::cpu::{forward_decode_step as cpu_forward_decode_step, forward_prefill as cpu_forward_prefill, CpuKvCache, CpuModel};
use crate::engine::Engine;
use crate::error::LeanError;
use crate::generate::decode_loop;
use crate::model::{argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache};
use crate::sampling::{sample, Rng64, SamplingParams};

/// One turn's input: `role` + `content`. A client renders its own chat
/// template context for any system prompt/history policy before building
/// this slice; this crate only renders it through the model's
/// `chat_template` string.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message { role: role.into(), content: content.into() }
    }
}

/// Everything one `generate` call needs beyond `messages`: how many tokens
/// to produce, and the sampling choices `sampling::SamplingParams` already
/// carries (wrapped here rather than duplicated, so a caller builds one
/// params type instead of two). `mask_bits` is this crate's mask-bitset
/// convention (`model::build_mask_bitset`); `None` means unmasked - the
/// token-level primitives (`forward_prefill`, `forward_decode_step_argmax`,
/// `decode_loop`) take the same convention directly if a caller wants raw
/// token ids instead of this message-level wrapper.
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
}

/// The token ids this engine's KV cache currently holds, left over from
/// whichever call (or low-level write) last touched the cache - see the
/// module doc comment on single-client reuse.
#[derive(Default)]
pub struct PrefixCache {
    cached_ids: Vec<u32>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop trusting the cached ids: the next call prefills its whole
    /// rendered conversation from scratch. Call this whenever something
    /// else writes the KV cache directly (a low-level `prefillTokens`/
    /// `decodeStepArgmax`/`restoreKv` call, or any direct `cache.kv_len`
    /// change) so the next `generate` doesn't trust a prefix that is no
    /// longer actually resident.
    pub fn invalidate(&mut self) {
        self.cached_ids.clear();
    }

    /// How many of `ids`' leading tokens are already resident at `kv_len`:
    /// the longest common prefix of the previously cached ids and `ids`,
    /// capped so at least one id is always left to prefill (a call always
    /// produces at least one position's logits).
    fn reuse(&self, ids: &[u32], kv_len: usize) -> usize {
        let usable = self.cached_ids.len().min(kv_len);
        let lcp = self.cached_ids[..usable].iter().zip(ids).take_while(|(a, b)| a == b).count();
        lcp.min(ids.len().saturating_sub(1))
    }

    /// Records what a call left resident: `ids`' first `kv_len` tokens,
    /// plus `generated` appended, truncated to `kv_len` (an abort can skip
    /// feeding the last generated id back into the cache).
    fn commit(&mut self, ids: Vec<u32>, generated: &[u32], kv_len: usize) {
        let mut cached = ids;
        cached.extend_from_slice(generated);
        cached.truncate(kv_len);
        self.cached_ids = cached;
    }
}

/// Renders `messages` through `chat_template` (`add_generation_prompt =
/// true`) - the exact text a `generate` call would tokenize and run,
/// without running it.
pub fn render_prompt(chat_template: &str, messages: &[Message]) -> Result<String, LeanError> {
    let turns: Vec<(&str, &str)> = messages.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
    render_conversation(chat_template, &turns, true).map_err(LeanError::from)
}

/// [`render_prompt`] + tokenize, with no forward-pass work - how many
/// tokens a `generate` call over `messages` would need, so a caller can
/// decide whether to trim before calling.
pub fn token_count(tokenizer: &Tokenizer, chat_template: &str, messages: &[Message]) -> Result<usize, LeanError> {
    let rendered = render_prompt(chat_template, messages)?;
    let ids = tokenizer.encode(rendered, false).map_err(|e| LeanError::Other(anyhow::anyhow!("tokenizer encode failed: {e}")))?;
    Ok(ids.get_ids().len())
}

/// Renders + tokenizes `messages` and checks the result plus
/// `max_new_tokens` fits `max_ctx`, returning the typed overflow error
/// ([`LeanError::ContextOverflow`]) rather than running anything if not -
/// shared by [`gpu_generate`]/[`cpu_generate`]. Nothing is mutated before
/// this returns, so an `Err` here leaves the engine's cache state exactly
/// as it was.
fn plan(tokenizer: &Tokenizer, chat_template: &str, messages: &[Message], max_new_tokens: u32, max_ctx: usize) -> Result<Vec<u32>, LeanError> {
    let rendered = render_prompt(chat_template, messages)?;
    let ids = tokenizer.encode(rendered, false).map_err(|e| LeanError::Other(anyhow::anyhow!("tokenizer encode failed: {e}")))?.get_ids().to_vec();
    if ids.is_empty() {
        return Err(LeanError::Other(anyhow::anyhow!("the rendered conversation tokenizes to nothing")));
    }
    let needed = ids.len() + max_new_tokens as usize;
    if needed > max_ctx {
        return Err(LeanError::ContextOverflow { needed, available: max_ctx });
    }
    Ok(ids)
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
pub fn apply_mask_cpu(logits: &mut [f32], mask_bits: Option<&[u32]>) {
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
/// tokens. Natively `yield_now` is `|| async {}`.
#[allow(clippy::too_many_arguments)]
pub async fn cpu_decode_loop<F: Future<Output = ()>>(
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
    apply_mask_cpu(&mut first_logits, mask_bits);
    let mut next_id = if greedy { argmax(&first_logits) } else { sample(&mut first_logits, params, history, &mut rng) };

    let mut generated = Vec::with_capacity(max_new_tokens as usize);
    for _ in 0..max_new_tokens {
        if model.config.eos_token_ids.contains(&next_id) || should_stop() {
            break;
        }
        generated.push(next_id);
        history.push(next_id);
        on_token(next_id);
        yield_now().await;
        if should_stop() {
            break;
        }
        let mut logits = cpu_forward_decode_step(model, cache, next_id);
        apply_mask_cpu(&mut logits, mask_bits);
        next_id = if greedy { argmax(&logits) } else { sample(&mut logits, params, history, &mut rng) };
    }
    generated
}

/// One inference turn on the GPU: plan, prefill what is not cached
/// (`forward_prefill` from scratch, else `forward_prefill_suffix` after
/// rewinding to the reused prefix), stream the reply as text through
/// `on_token`, record what the call left resident in `prefix`. Returns the
/// reply text.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_generate(
    engine: &Engine,
    model: &GpuModel,
    cache: &mut KvCache,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    prefix: &mut PrefixCache,
    tokenizer: &Tokenizer,
    chat_template: &str,
    messages: &[Message],
    params: &GenParams,
    mask: Option<&wgpu::Buffer>,
    mut on_token: impl FnMut(&str),
    should_stop: impl FnMut() -> bool,
) -> Result<String, LeanError> {
    let ids = plan(tokenizer, chat_template, messages, params.max_new_tokens, cache.max_ctx as usize)?;
    let reuse = prefix.reuse(&ids, cache.kv_len as usize);
    let logits = if reuse == 0 {
        forward_prefill(engine, model, cache, &ids, cos, sin, mask).await
    } else {
        cache.kv_len = reuse as u32;
        forward_prefill_suffix(engine, model, cache, &ids[reuse..], cos, sin, mask).await
    };
    let mut history = ids.clone();
    let sampling = params.sampling();
    let mut stream = TextStream::new();
    let mut stream_err: Option<LeanError> = None;
    let generated = decode_loop(
        engine,
        model,
        cache,
        logits,
        cos,
        sin,
        mask,
        params.max_new_tokens,
        &sampling,
        &mut history,
        |id| match stream.push(tokenizer, id) {
            Ok(delta) if !delta.is_empty() => on_token(&delta),
            Ok(_) => {}
            Err(e) => stream_err = Some(e),
        },
        should_stop,
    )
    .await;
    if let Some(e) = stream_err {
        return Err(e);
    }
    let tail = stream.finish(tokenizer)?;
    if !tail.is_empty() {
        on_token(&tail);
    }
    let reply = TextStream::decode(tokenizer, &generated)?;
    prefix.commit(ids, &generated, cache.kv_len as usize);
    Ok(reply)
}

/// One inference turn on the CPU, same steps as [`gpu_generate`]; the CPU
/// prefill continues from the cache's resident prefix directly.
#[allow(clippy::too_many_arguments)]
pub async fn cpu_generate<F: Future<Output = ()>>(
    model: &CpuModel,
    cache: &mut CpuKvCache,
    prefix: &mut PrefixCache,
    tokenizer: &Tokenizer,
    chat_template: &str,
    messages: &[Message],
    params: &GenParams,
    mut on_token: impl FnMut(&str),
    should_stop: impl FnMut() -> bool,
    yield_now: impl FnMut() -> F,
) -> Result<String, LeanError> {
    let ids = plan(tokenizer, chat_template, messages, params.max_new_tokens, cache.max_ctx())?;
    let reuse = prefix.reuse(&ids, cache.kv_len);
    cache.kv_len = reuse;
    let logits = cpu_forward_prefill(model, cache, &ids[reuse..]);
    let mut history = ids.clone();
    let sampling = params.sampling();
    let mask_bits = params.mask_bits.as_deref();
    let mut stream = TextStream::new();
    let mut stream_err: Option<LeanError> = None;
    let generated = cpu_decode_loop(
        model,
        cache,
        logits,
        mask_bits,
        params.max_new_tokens,
        &sampling,
        &mut history,
        |id| match stream.push(tokenizer, id) {
            Ok(delta) if !delta.is_empty() => on_token(&delta),
            Ok(_) => {}
            Err(e) => stream_err = Some(e),
        },
        should_stop,
        yield_now,
    )
    .await;
    if let Some(e) = stream_err {
        return Err(e);
    }
    let tail = stream.finish(tokenizer)?;
    if !tail.is_empty() {
        on_token(&tail);
    }
    let reply = TextStream::decode(tokenizer, &generated)?;
    prefix.commit(ids, &generated, cache.kv_len);
    Ok(reply)
}
