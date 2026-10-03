//! Multi-turn chat shared by both browser engines (`web.rs`'s `LeanEngine`
//! on WebGPU and `LeanEngineCpu` on CPU threads or one thread) and by the
//! native tests, so the wasm surface is a thin wrapper over the code the
//! tests exercise.
//!
//! - [`ChatSession`]: the conversation (`(role, content)` turns) and the
//!   exact token ids resident in the KV cache. Each turn re-renders the whole
//!   conversation through the model's own chat template, tokenizes it, and
//!   reuses the longest common prefix with the cached ids; only the rest is
//!   run through the model. The KV state after N turns is therefore the
//!   state a from-scratch prefill of the rendered conversation would build,
//!   including when the template rewrites an earlier turn or when the reply
//!   text re-tokenizes differently from the generated ids.
//! - [`TextStream`]: incremental detokenization. Decoding one id at a time
//!   is wrong for byte-level BPE (a multi-byte character spans tokens) and
//!   for SentencePiece-style decoders (a leading space is dropped at the
//!   start of a decode); this decodes a sliding window of the accumulated
//!   ids and emits only the new text, holding back a trailing incomplete
//!   character. The concatenated deltas equal `decode(all ids)`.
//! - [`gpu_chat_turn`] / [`cpu_chat_turn`]: one turn on each backend.

use std::future::Future;

use anyhow::{bail, Context, Result};
use tokenizers::Tokenizer;

use crate::chat_template::render_conversation;
use crate::cpu::{forward_decode_step as cpu_forward_decode_step, forward_prefill as cpu_forward_prefill, CpuKvCache, CpuModel};
use crate::engine::Engine;
use crate::generate::decode_loop;
use crate::model::{argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache};
use crate::sampling::{sample, Rng64, SamplingParams};

#[derive(Default)]
pub struct ChatSession {
    history: Vec<(String, String)>,
    /// The token ids at KV positions `[0, cached_ids.len())`, as this session
    /// wrote them. Cleared by [`ChatSession::invalidate_cache`] whenever
    /// anything else writes the cache.
    cached_ids: Vec<u32>,
}

/// What a turn has to run: the whole rendered conversation's ids, and how
/// many leading ids are already in the KV cache (always fewer than
/// `ids.len()`, so at least one id is prefilled and yields logits).
pub struct TurnPlan {
    pub ids: Vec<u32>,
    pub reuse: usize,
}

impl ChatSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the conversation and the cached ids (the caller resets the KV
    /// length to 0).
    pub fn reset(&mut self) {
        self.history.clear();
        self.cached_ids.clear();
    }

    /// Keep the conversation but stop trusting the KV cache: the next turn
    /// prefills the whole rendered conversation again.
    pub fn invalidate_cache(&mut self) {
        self.cached_ids.clear();
    }

    pub fn history(&self) -> &[(String, String)] {
        &self.history
    }

    fn render(&self, chat_template: &str, add_generation_prompt: bool) -> Result<String> {
        let turns: Vec<(&str, &str)> = self.history.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
        render_conversation(chat_template, &turns, add_generation_prompt)
    }

    /// Appends `prompt` as a user turn and plans the prefill. `kv_len` is
    /// the cache's current length: cached ids beyond it are not reused. On
    /// error the user turn is taken back out.
    pub fn begin_turn(&mut self, tokenizer: &Tokenizer, chat_template: &str, prompt: &str, kv_len: usize, max_new_tokens: usize, max_ctx: usize) -> Result<TurnPlan> {
        self.history.push(("user".to_string(), prompt.to_string()));
        let plan = self.plan(tokenizer, chat_template, kv_len, max_new_tokens, max_ctx);
        if plan.is_err() {
            self.history.pop();
        }
        plan
    }

    fn plan(&self, tokenizer: &Tokenizer, chat_template: &str, kv_len: usize, max_new_tokens: usize, max_ctx: usize) -> Result<TurnPlan> {
        let rendered = self.render(chat_template, true).context("rendering the conversation")?;
        let ids = tokenizer.encode(rendered, false).map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?.get_ids().to_vec();
        if ids.is_empty() {
            bail!("the rendered conversation tokenizes to nothing");
        }
        if ids.len() + max_new_tokens > max_ctx {
            bail!("conversation ({} tokens) + max_new_tokens ({max_new_tokens}) exceeds max_ctx ({max_ctx})", ids.len());
        }
        let usable = self.cached_ids.len().min(kv_len);
        let lcp = self.cached_ids[..usable].iter().zip(&ids).take_while(|(a, b)| a == b).count();
        let reuse = lcp.min(ids.len() - 1);
        Ok(TurnPlan { ids, reuse })
    }

    /// Records the reply: the cache now holds the first `kv_len` ids of
    /// `plan_ids` + `generated` (every generated id is fed back except one
    /// whose step an abort skipped), and `reply` becomes the assistant turn.
    pub fn end_turn(&mut self, plan_ids: Vec<u32>, generated: &[u32], kv_len: usize, reply: String) {
        let mut cached = plan_ids;
        cached.extend_from_slice(generated);
        cached.truncate(kv_len);
        self.cached_ids = cached;
        self.history.push(("assistant".to_string(), reply));
    }
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

    fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
        if ids.is_empty() {
            return Ok(String::new());
        }
        tokenizer.decode(ids, true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
    }

    fn delta(&self, tokenizer: &Tokenizer, flush: bool) -> Result<Option<String>> {
        let prev_text = Self::decode(tokenizer, &self.ids[self.prev..self.cur])?;
        let text = Self::decode(tokenizer, &self.ids[self.prev..])?;
        if text.len() > prev_text.len() && text.starts_with(&prev_text) && (flush || !text.ends_with('\u{FFFD}')) {
            return Ok(Some(text[prev_text.len()..].to_string()));
        }
        Ok(None)
    }

    /// Adds one generated id; returns the text it completes (often "", e.g.
    /// for a special token or the first half of a multi-byte character).
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String> {
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
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String> {
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

/// One chat turn on the GPU: plan, prefill what is not cached
/// (`forward_prefill` from position 0, else `forward_prefill_suffix` after
/// rewinding to the reused prefix), stream the reply, record it. Returns
/// the generated ids and the reply text.
#[allow(clippy::too_many_arguments)]
pub async fn gpu_chat_turn(
    engine: &Engine,
    model: &GpuModel,
    cache: &mut KvCache,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    session: &mut ChatSession,
    tokenizer: &Tokenizer,
    chat_template: &str,
    prompt: &str,
    max_new_tokens: u32,
    params: &SamplingParams,
    mask: Option<&wgpu::Buffer>,
    on_token: impl FnMut(u32),
    should_stop: impl FnMut() -> bool,
) -> Result<(Vec<u32>, String)> {
    let plan = session.begin_turn(tokenizer, chat_template, prompt, cache.kv_len as usize, max_new_tokens as usize, cache.max_ctx as usize)?;
    let logits = if plan.reuse == 0 {
        forward_prefill(engine, model, cache, &plan.ids, cos, sin, mask).await
    } else {
        cache.kv_len = plan.reuse as u32;
        forward_prefill_suffix(engine, model, cache, &plan.ids[plan.reuse..], cos, sin, mask).await
    };
    let mut history = plan.ids.clone();
    let generated = decode_loop(engine, model, cache, logits, cos, sin, mask, max_new_tokens, params, &mut history, on_token, should_stop).await;
    let reply = TextStream::decode(tokenizer, &generated)?;
    session.end_turn(plan.ids, &generated, cache.kv_len as usize, reply.clone());
    Ok((generated, reply))
}

/// One chat turn on the CPU, same steps as [`gpu_chat_turn`]; the CPU
/// prefill continues from the cache's resident prefix directly.
#[allow(clippy::too_many_arguments)]
pub async fn cpu_chat_turn<F: Future<Output = ()>>(
    model: &CpuModel,
    cache: &mut CpuKvCache,
    session: &mut ChatSession,
    tokenizer: &Tokenizer,
    chat_template: &str,
    prompt: &str,
    max_new_tokens: u32,
    params: &SamplingParams,
    mask_bits: Option<&[u32]>,
    on_token: impl FnMut(u32),
    should_stop: impl FnMut() -> bool,
    yield_now: impl FnMut() -> F,
) -> Result<(Vec<u32>, String)> {
    let plan = session.begin_turn(tokenizer, chat_template, prompt, cache.kv_len, max_new_tokens as usize, cache.max_ctx())?;
    cache.kv_len = plan.reuse;
    let logits = cpu_forward_prefill(model, cache, &plan.ids[plan.reuse..]);
    let mut history = plan.ids.clone();
    let generated = cpu_decode_loop(model, cache, logits, mask_bits, max_new_tokens, params, &mut history, on_token, should_stop, yield_now).await;
    let reply = TextStream::decode(tokenizer, &generated)?;
    session.end_turn(plan.ids, &generated, cache.kv_len, reply.clone());
    Ok((generated, reply))
}
