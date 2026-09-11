//! Agent loop: render -> encode -> generate -> parse -> [tool results ->
//! repeat] -> text. Owned by phase 1a.
//!
//! Two APIs are exposed:
//!
//! - **Step-wise** (`Agent::start` / `Agent::provide_tool_results`): the
//!   browser can't call tools synchronously (an MCP tool call is itself an
//!   async round trip through the page), so each call does exactly one
//!   render -> generate -> parse round and returns a [`StepOutcome`]
//!   describing what happened. If the model asked for tool calls, the
//!   caller executes them (however it likes — synchronously, over a
//!   worker's postMessage, whatever) and feeds the results back via
//!   `provide_tool_results`. Generation itself is still synchronous inside
//!   a step (the model forward pass blocks the calling thread); that's the
//!   right tradeoff in a Web Worker, which has no other work to interleave
//!   with anyway.
//! - **`run`**: a synchronous convenience built on the step-wise API, for
//!   native callers (tests, `llm-agent` CLI) that have a synchronous
//!   [`ToolCaller`] and just want the whole transcript.
//!
//! ## Prefix caching
//!
//! The rendered prompt for every turn in a conversation with the same
//! `tools` set shares an identical *token* prefix: system message + tool
//! schema preamble, which the xLAM-2 chat template renders before the user
//! turn (see `template.rs`). Re-running the full prefill for that prefix on
//! every step wastes most of prefill's cost once a conversation has more
//! than a couple of tools. `Agent` computes this prefix's token length once
//! per distinct `tools` set (comparing two renders — the actual utterance
//! plus a throwaway probe utterance under the same tools — and taking their
//! common leading run of token ids) and passes it to
//! [`Generator::generate_with_cached_prefix`] on every step, so a concrete
//! `Generator` backed by a real model can `KvCache::restore()` to that
//! length and prefill only the new suffix instead of the whole prompt (see
//! `kv.rs`'s `snapshot`/`restore`, which this trait method is designed to
//! be implemented against). `FixtureGenerator`'s default implementation
//! just ignores the hint and regenerates from the full prompt, since it
//! has no cache to restore.

use crate::grammar::{self, Constraint, Grammar, GrammarConstraint, IdValues, TokenVocab};
use crate::schemadiet::{diet_tools, DietLevel};
use crate::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use crate::tokenizer::Tokenizer;
use crate::tools::{format_tool_error, format_tool_result, parse_output, tool_error_message, ParsedOutput, ToolCall};
use anyhow::{bail, Result};
use serde_json::Value;
use std::time::{Duration, Instant};

/// A generic, non-committal nudge appended as a `user` message when a
/// model output is malformed/empty/unknown-tool twice in a row (see
/// `Agent::step_inner`'s retry policy) — the second and last retry, after
/// "try again with the grammar constraint enabled" (the first retry) has
/// also failed to produce a valid call.
const RETRY_NOTE: &str = "Respond with a tool call from the list or a final answer.";

/// Executes one tool call and returns its JSON result.
pub trait ToolCaller {
    fn call(&mut self, name: &str, args: &Value) -> Result<Value>;
}

/// Runs the model forward from `prompt_ids`, sampling `max_new_tokens`
/// tokens (or fewer, stopping early on any id in `stop_ids`). The real
/// model (Burn+wgpu) implements this in phase 1b; `FixtureGenerator` below
/// replays a scripted sequence for tests.
pub trait Generator {
    fn generate(&mut self, prompt_ids: &[u32], max_new_tokens: usize, stop_ids: &[u32]) -> Result<Vec<u32>>;

    /// Like [`Generator::generate`], but tells the generator that the first
    /// `prefix_len` tokens of `prompt_ids` are identical to a prefix this
    /// generator has already prefilled in an earlier call (from a previous
    /// step or turn under the same `tools` set) — a concrete implementation
    /// backed by a real KV cache can restore to that snapshot and prefill
    /// only `prompt_ids[prefix_len..]`. The default implementation ignores
    /// the hint and behaves exactly like `generate`, which is always
    /// correct (just not faster) — implementing this is an optimization,
    /// not a correctness requirement.
    fn generate_with_cached_prefix(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
        stop_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let _ = prefix_len;
        self.generate(prompt_ids, max_new_tokens, stop_ids)
    }

    /// Prefill/decode timing breakdown for the most recently completed
    /// `generate`/`generate_with_cached_prefix` call, for eval metrics
    /// (`eval.rs`'s `mean_prefill_s`/`mean_decode_tok_s`). The default
    /// returns `(Duration::ZERO, Duration::ZERO)` — "no breakdown
    /// available" — in which case `Agent::step_inner` attributes its own
    /// wall-clock measurement of the whole call to decode time instead. A
    /// concrete implementation backed by a real KV cache (which knows when
    /// prefill ends and decode begins) should override this.
    fn last_call_timing(&self) -> (Duration, Duration) {
        (Duration::ZERO, Duration::ZERO)
    }

    /// Like [`Generator::generate_with_cached_prefix`], but drives an
    /// optional schema [`Constraint`] (`grammar.rs`) through the decode
    /// loop — see `docs/ENGINE.md` "Schema-constrained decoding" for the
    /// jump-forward semantics a concrete implementation backed by a real
    /// model (`model.rs::LlmModel::generate_with_constraint`) should give
    /// this. The default implementation ignores `constraint` entirely and
    /// falls back to `generate_with_cached_prefix`, reporting every
    /// generated token as one model step and zero forced tokens — correct
    /// (unconstrained) but not what `Agent`'s `constrained: true` mode is
    /// for; `FixtureGenerator` relies on this default since it has no
    /// model to drive a real constraint through.
    fn generate_constrained(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
        stop_ids: &[u32],
        constraint: Option<&mut dyn Constraint>,
    ) -> Result<GenerateOutput> {
        let _ = constraint;
        let ids = self.generate_with_cached_prefix(prompt_ids, prefix_len, max_new_tokens, stop_ids)?;
        let model_steps = ids.len();
        Ok(GenerateOutput {
            ids,
            model_steps,
            forced_tokens: 0,
        })
    }
}

/// Result of [`Generator::generate_constrained`]: the generated ids plus a
/// model-step / forced-token breakdown (jump-forward runs count as one
/// model step covering multiple tokens — see `model.rs::GenerateStats`,
/// which a real implementation's `model_steps`/`forced_tokens` should
/// mirror).
pub struct GenerateOutput {
    pub ids: Vec<u32>,
    pub model_steps: usize,
    pub forced_tokens: usize,
}

/// One render -> generate -> parse round of the agent loop.
#[derive(Debug)]
pub struct Step {
    pub prompt_tokens: Vec<u32>,
    pub generated_text: String,
    pub parsed: ParsedOutput,
    /// Set when this step's parsed output was tool call(s) *and* they've
    /// already been executed (only true for steps produced by `run`, which
    /// owns a synchronous `ToolCaller`): the result of the *last* call made
    /// this step. Steps produced by the step-wise API (`start` /
    /// `provide_tool_results`) leave this `None` — the caller hasn't
    /// necessarily run the tools yet when the `Step` is handed back.
    pub tool_result: Option<Value>,
    pub timings: Duration,
    /// Number of tokens sampled this step (`out_ids.len()`).
    pub tokens_generated: usize,
    /// Prefill time for this step's `generate_with_cached_prefix` call, or
    /// `Duration::ZERO` when the `Generator` doesn't report a breakdown
    /// (see `Generator::last_call_timing`).
    pub prefill_time: Duration,
    /// Decode time for this step's `generate_with_cached_prefix` call. When
    /// the `Generator` doesn't report a breakdown, this is the whole
    /// step's wall-clock time (same value as `timings`), i.e. prefill cost
    /// is folded into it rather than lost.
    pub decode_time: Duration,
    /// Model forward-pass steps this step's generation took (a
    /// jump-forward run of several forced tokens is 1 step — see
    /// `GenerateOutput`). Equals `tokens_generated` for an unconstrained
    /// step (every token costs its own step).
    pub model_steps: usize,
    /// Of `tokens_generated`, how many were jump-forwarded (schema-forced,
    /// not individually sampled). 0 for an unconstrained step.
    pub forced_tokens: usize,
    /// Number of malformed-output retries (`Agent`'s empty-array /
    /// unparsable-JSON / unknown-tool retry policy — see `step_inner`)
    /// spent before this step's output was accepted. 0 when the first
    /// attempt was already valid. Does not count toward `max_steps`.
    pub retries: usize,
    /// Running count of *consecutive* tool-error results fed back into
    /// the conversation as of this step (see `Agent::provide_tool_results`
    /// / `tool_error_message`); 0 unless the immediately preceding
    /// `provide_tool_results` call carried an error result.
    pub tool_errors: usize,
}

pub struct Transcript {
    pub steps: Vec<Step>,
    pub final_text: String,
}

/// One `generate_attempt` round's output, before `step_inner` decides
/// whether it's valid or needs a retry — `parsed` is `Err` for a parse
/// failure (retryable) rather than short-circuiting like the other
/// render/encode/generate/decode failures `generate_attempt` returns as
/// `Err(String)` directly.
struct AttemptOutput {
    prompt_tokens: Vec<u32>,
    generated_text: String,
    parsed: Result<ParsedOutput, String>,
    timings: Duration,
    tokens_generated: usize,
    prefill_time: Duration,
    decode_time: Duration,
    model_steps: usize,
    forced_tokens: usize,
}

/// One tool call the model asked for, awaiting a result via
/// [`Agent::provide_tool_results`]. `call_id` is scoped to the step that
/// produced it (stable identifiers like `"call_0"`, `"call_1"`, ...,
/// matching the `id` on the corresponding `tool_calls` entry appended to
/// history) — pass it back unchanged in `provide_tool_results` so the
/// result is attached to the right call.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

/// Outcome of one step ([`Agent::start`] or [`Agent::provide_tool_results`]).
#[derive(Debug)]
pub enum StepOutcome {
    /// The model asked for one or more tool calls; execute them (by
    /// whatever means) and pass `(call_id, result)` pairs to
    /// `provide_tool_results`.
    NeedTools { calls: Vec<PendingToolCall>, step: Step },
    /// The model produced a final text answer; the turn is done.
    Final { text: String, step: Step },
    /// Rendering, generation, or output parsing failed, or the agent hit
    /// its step budget without a final answer. No further steps should be
    /// taken on this `Agent` for the current turn without calling `start`
    /// again. `step` is populated when a `Step` was actually produced
    /// before the failure (e.g. unparseable model output); it's `None` for
    /// failures before or during generation (render/encode/generate
    /// errors, or the max-steps guard).
    Error { message: String, step: Option<Step> },
}

/// A `ToolCaller` that serves canned JSON results from
/// `fixtures/sonos/results/<tool>.json` and records every call it saw, in
/// order — for tests.
pub struct FixtureCaller {
    results_dir: std::path::PathBuf,
    pub calls_seen: Vec<(String, Value)>,
}

impl FixtureCaller {
    pub fn new(results_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            results_dir: results_dir.into(),
            calls_seen: Vec::new(),
        }
    }
}

impl ToolCaller for FixtureCaller {
    fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
        self.calls_seen.push((name.to_string(), args.clone()));
        let path = self.results_dir.join(format!("{name}.json"));
        let data = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("no fixture result for tool `{name}` at {path:?}: {e}"))?;
        let parsed: Value = serde_json::from_str(&data)?;
        Ok(as_mcp_result(&parsed))
    }
}

/// Wrap a canned fixture value as the shape a real MCP `tools/call` result
/// actually has — `{"content":[{"type":"text","text":"<json>"}]}`, the
/// payload pretty-printed the way Sonos's (and MCP servers generally)
/// server does, *not* returned as a parsed value directly. Making the
/// fixture path faithful to this is what surfaces bugs (e.g. id
/// harvesting from the embedded text) that only showed up against a real
/// server and not against the old parsed-value fixture shape.
///
/// A fixture file carrying a top-level `error` field (none currently do,
/// but the shape is supported) becomes an MCP error result instead:
/// `{"content":[{"type":"text","text":"<message>"}],"isError":true}`, per
/// MCP's `CallToolResult` error convention (see `tools::tool_error_message`).
fn as_mcp_result(parsed: &Value) -> Value {
    if let Some(error) = parsed.get("error") {
        let message = match error {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        return serde_json::json!({
            "content": [{"type": "text", "text": message}],
            "isError": true,
        });
    }
    let text = serde_json::to_string_pretty(parsed).unwrap_or_else(|_| parsed.to_string());
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
    })
}

/// A `Generator` that replays a scripted sequence of token-id vectors, one
/// per call to `generate`, ignoring the prompt — for tests.
pub struct FixtureGenerator {
    script: Vec<Vec<u32>>,
    next: usize,
}

impl FixtureGenerator {
    pub fn new(script: Vec<Vec<u32>>) -> Self {
        Self { script, next: 0 }
    }
}

impl Generator for FixtureGenerator {
    fn generate(&mut self, _prompt_ids: &[u32], _max_new_tokens: usize, _stop_ids: &[u32]) -> Result<Vec<u32>> {
        let out = self
            .script
            .get(self.next)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("FixtureGenerator script exhausted after {} call(s)", self.next))?;
        self.next += 1;
        Ok(out)
    }
}

/// Default number of steps `Agent::new` allows before the step-wise API's
/// max-steps guard fires; override with [`Agent::set_max_steps`]. `run`
/// overrides this per-call with its own `max_steps` argument instead.
const DEFAULT_MAX_STEPS: usize = 8;

/// Default per-step cap on malformed-output retries (empty tool-call
/// array, unparsable JSON starting with `[`, or a call naming a tool not
/// in the current `tools` set) — override with [`Agent::set_max_retries`].
/// See `docs/ENGINE.md` "Agent loop" for the retry policy.
const DEFAULT_MAX_RETRIES: usize = 2;

/// Number of consecutive tool-error results (see `tool_error_message`)
/// `Agent` will feed back to the model before giving up with
/// `StepOutcome::Error` instead of trying a third time.
const MAX_CONSECUTIVE_TOOL_ERRORS: usize = 2;

/// Diet level applied to raw MCP tool lists by [`Agent::start_from_mcp`]
/// when [`Agent`]'s diet flag is on (the default) — see `schemadiet.rs`.
const AGENT_DIET_LEVEL: DietLevel = DietLevel::Level1;

/// Drives one conversation turn: render the prompt, generate, parse the
/// output, and either finish with plain text or execute the requested tool
/// call(s) and loop.
pub struct Agent<G: Generator, C: ToolCaller> {
    template: ChatTemplate,
    tokenizer: Tokenizer,
    generator: G,
    caller: C,
    system_prompt: String,
    max_new_tokens: usize,
    max_steps: usize,
    max_retries: usize,
    /// Whether [`Agent::start_from_mcp`] runs raw MCP tool lists through
    /// `schemadiet::diet_tools` before `Tool::from_mcp`. On by default;
    /// disable for byte-fidelity comparisons against undieted fixtures.
    diet: bool,

    // Step-wise conversation state (see `start`/`provide_tool_results`).
    messages: Vec<Message>,
    tools: Vec<Tool>,
    step_index: usize,
    pending_calls: Vec<PendingToolCall>,
    /// Consecutive tool-error results fed back so far this turn (reset by
    /// `start`/`reset`, and whenever a non-error tool result arrives) —
    /// see `provide_tool_results`.
    consecutive_tool_errors: usize,
    /// Cached (tools set, common prefix token length), recomputed only when
    /// `tools` changes between `start` calls — see module docs.
    prefix_cache: Option<(Vec<Tool>, usize)>,

    /// Whether to constrain generation to `grammar.rs`'s tool-call schema
    /// (see [`Agent::set_constrained`]). Off by default so existing
    /// unconstrained callers/tests are unaffected.
    constrained: bool,
    /// Id-shaped strings harvested from every tool result seen so far this
    /// conversation (`IdValues::collect_from_result`, called from
    /// `provide_tool_results`) — the only values a `*_id`/`*_ids` schema
    /// property may take (`docs/ENGINE.md` "The id rule"). Reset in
    /// `start`/`reset`.
    id_values: IdValues,
    /// Built lazily on first constrained step and cached — `TokenVocab`
    /// precomputes every vocab id's byte string once, not once per step.
    token_vocab: Option<TokenVocab>,
}

impl<G: Generator, C: ToolCaller> Agent<G, C> {
    pub fn new(
        template: ChatTemplate,
        tokenizer: Tokenizer,
        generator: G,
        caller: C,
        system_prompt: impl Into<String>,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            template,
            tokenizer,
            generator,
            caller,
            system_prompt: system_prompt.into(),
            max_new_tokens,
            max_steps: DEFAULT_MAX_STEPS,
            max_retries: DEFAULT_MAX_RETRIES,
            diet: true,
            messages: Vec::new(),
            tools: Vec::new(),
            step_index: 0,
            pending_calls: Vec::new(),
            consecutive_tool_errors: 0,
            prefix_cache: None,
            constrained: false,
            id_values: IdValues::new(),
            token_vocab: None,
        }
    }

    pub fn caller(&self) -> &C {
        &self.caller
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Turn schema-constrained decoding on/off (`grammar.rs`,
    /// `docs/ENGINE.md` "Schema-constrained decoding"). Off by default. A
    /// `Generator` that doesn't override `generate_constrained` runs
    /// unconstrained regardless of this flag (its default impl ignores the
    /// constraint) — see that method's docs.
    pub fn set_constrained(&mut self, constrained: bool) {
        self.constrained = constrained;
    }

    /// Override the step budget the step-wise API (`start` /
    /// `provide_tool_results`) errors out after. `run` ignores this — it
    /// takes its own `max_steps` argument.
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    /// Override the per-step cap on malformed-output retries (default
    /// [`DEFAULT_MAX_RETRIES`]) — see `step_inner`'s retry policy.
    pub fn set_max_retries(&mut self, max_retries: usize) {
        self.max_retries = max_retries;
    }

    /// Turn the MCP tool-schema token diet [`Agent::start_from_mcp`]
    /// applies on/off. On by default; turn off for byte-fidelity tests
    /// that must see the raw (undieted) schema.
    pub fn set_diet(&mut self, diet: bool) {
        self.diet = diet;
    }

    /// Drop any in-progress conversation state (e.g. after an `Error`
    /// outcome, or to abandon a turn). Does not touch the KV-cache prefix
    /// cache, which is keyed by `tools` and safe to keep across turns.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.tools.clear();
        self.step_index = 0;
        self.pending_calls.clear();
        self.consecutive_tool_errors = 0;
        self.id_values = IdValues::new();
    }

    /// Begin a new turn: render `utterance` against `tools`, generate, and
    /// parse — one step. Returns `NeedTools` if the model wants to call
    /// tools (pass results to `provide_tool_results`), `Final` if it
    /// answered directly, or `Error` on failure.
    pub fn start(&mut self, utterance: &str, tools: &[Tool]) -> StepOutcome {
        self.tools = tools.to_vec();
        self.messages = vec![Message::system(&self.system_prompt), Message::user(utterance)];
        self.step_index = 0;
        self.pending_calls.clear();
        self.consecutive_tool_errors = 0;
        self.id_values = IdValues::new();

        match self.prefix_len_for(&self.tools, utterance) {
            Ok(len) => self.prefix_cache = Some((self.tools.clone(), len)),
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("failed to compute prefix cache length: {e}"),
                    step: None,
                }
            }
        }

        self.step_inner()
    }

    /// Like [`Agent::start`], but takes raw MCP `tools/list` entries
    /// (`{"name", "description", "inputSchema"}`) instead of already-built
    /// `Tool`s — runs them through `schemadiet::diet_tools` (when
    /// [`Agent::set_diet`] is on, the default) before `Tool::from_mcp`,
    /// per `docs/ENGINE.md`'s "Tool-schema token diet" hook-in point.
    pub fn start_from_mcp(&mut self, utterance: &str, raw_tools: &[Value]) -> StepOutcome {
        let tools = tools_from_mcp(raw_tools, self.diet);
        self.start(utterance, &tools)
    }

    /// Continue the current turn with the results of the tool calls from
    /// the most recent `NeedTools` outcome, keyed by `call_id`. Unmatched
    /// or missing `call_id`s are silently skipped (their tool message is
    /// never appended) — callers should supply exactly the `call_id`s from
    /// that `NeedTools`.
    pub fn provide_tool_results(&mut self, results: Vec<(String, Value)>) -> StepOutcome {
        for (call_id, result) in results {
            if let Some(pending) = self.pending_calls.iter().find(|c| c.call_id == call_id) {
                if let Some(error_message) = tool_error_message(&result) {
                    self.consecutive_tool_errors += 1;
                    if self.consecutive_tool_errors > MAX_CONSECUTIVE_TOOL_ERRORS {
                        self.pending_calls.clear();
                        return StepOutcome::Error {
                            message: format!(
                                "{} consecutive tool errors (giving up after {}): {error_message}",
                                self.consecutive_tool_errors, MAX_CONSECUTIVE_TOOL_ERRORS
                            ),
                            step: None,
                        };
                    }
                    self.messages.push(format_tool_error(&pending.name, &error_message));
                } else {
                    self.consecutive_tool_errors = 0;
                    self.id_values.collect_from_result(&result);
                    self.messages.push(format_tool_result(&pending.name, &result));
                }
            }
        }
        self.pending_calls.clear();
        self.step_inner()
    }

    /// One render -> generate -> parse round using `self.messages`/`self.tools`.
    fn step_inner(&mut self) -> StepOutcome {
        if self.step_index >= self.max_steps {
            return StepOutcome::Error {
                message: format!(
                    "agent exceeded max_steps ({}) without a final text response",
                    self.max_steps
                ),
                step: None,
            };
        }
        self.step_index += 1;

        // Malformed-output retry loop (`docs/ENGINE.md` "Agent loop"): an
        // empty tool-call array, unparsable `[...]` JSON, or a call naming
        // a tool outside `self.tools` does *not* get appended to history
        // as an assistant turn (that would just teach the model its own
        // junk is valid conversation) — instead we regenerate, changing
        // something each time so a deterministic sampler doesn't just
        // reproduce the same output: attempt 1 turns schema-constrained
        // decoding on (if not already on and tools are available), attempt
        // 2 appends a short generic nudge as a `user` message. Exhausting
        // `self.max_retries` gives up with `StepOutcome::Error`. Retries
        // don't consume the `max_steps` budget (already charged above).
        let mut retries = 0usize;
        loop {
            let force_constrained = retries == 1 && !self.constrained && !self.tools.is_empty();
            let attempt = match self.generate_attempt(force_constrained) {
                Ok(a) => a,
                Err(message) => return StepOutcome::Error { message, step: None },
            };

            let invalid = match &attempt.parsed {
                Err(_) => true,
                Ok(ParsedOutput::ToolCalls(calls)) => {
                    calls.is_empty()
                        || calls
                            .iter()
                            .any(|c| !self.tools.iter().any(|t| t.function.name == c.name))
                }
                Ok(ParsedOutput::Text(_)) => false,
            };

            if invalid {
                if retries < self.max_retries {
                    retries += 1;
                    if retries == 2 {
                        self.messages.push(Message::user(RETRY_NOTE));
                    }
                    continue;
                }
                return StepOutcome::Error {
                    message: "model produced no valid call".to_string(),
                    step: None,
                };
            }

            let AttemptOutput {
                prompt_tokens,
                generated_text,
                parsed,
                timings,
                tokens_generated,
                prefill_time,
                decode_time,
                model_steps,
                forced_tokens,
            } = attempt;
            let parsed = parsed.expect("checked valid above");
            let tool_errors = self.consecutive_tool_errors;

            return match parsed {
                ParsedOutput::ToolCalls(calls) => {
                    let pending = pending_calls_from(&calls);
                    let entries = tool_call_entries(&pending);
                    self.messages.push(Message::assistant_tool_calls(entries));
                    self.pending_calls = pending.clone();

                    let step = Step {
                        prompt_tokens,
                        generated_text,
                        parsed: ParsedOutput::ToolCalls(calls),
                        tool_result: None,
                        timings,
                        tokens_generated,
                        prefill_time,
                        decode_time,
                        model_steps,
                        forced_tokens,
                        retries,
                        tool_errors,
                    };
                    StepOutcome::NeedTools { calls: pending, step }
                }
                ParsedOutput::Text(text) => {
                    let step = Step {
                        prompt_tokens,
                        generated_text,
                        parsed: ParsedOutput::Text(text.clone()),
                        tool_result: None,
                        timings,
                        tokens_generated,
                        prefill_time,
                        decode_time,
                        model_steps,
                        forced_tokens,
                        retries,
                        tool_errors,
                    };
                    StepOutcome::Final { text, step }
                }
            };
        }
    }

    /// One render -> encode -> generate -> decode -> parse round, with an
    /// override to force schema-constrained decoding on for this attempt
    /// regardless of `self.constrained` (used by `step_inner`'s retry
    /// policy). Returns `Err(message)` only for failures that aren't part
    /// of the retry policy (render/encode/generate/decode) — a parse
    /// failure is returned as `Ok` with `parsed: Err(_)` so the caller can
    /// decide whether to retry.
    fn generate_attempt(&mut self, force_constrained: bool) -> Result<AttemptOutput, String> {
        let prompt = self
            .template
            .render_prompt(&self.messages, &self.tools, true)
            .map_err(|e| format!("failed to render prompt: {e}"))?;
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(|e| format!("failed to encode prompt: {e}"))?;

        let prefix_len = self
            .prefix_cache
            .as_ref()
            .filter(|(cached_tools, _)| cached_tools == &self.tools)
            .map(|(_, len)| (*len).min(prompt_tokens.len()))
            .unwrap_or(0);

        // Build this step's schema constraint (if `constrained`, or this
        // attempt forces it on) from the current tool set + every id
        // harvested from tool results so far — see `docs/ENGINE.md`
        // "Schema-constrained decoding".
        let use_constrained = self.constrained || force_constrained;
        let grammar_tools: Vec<grammar::Tool> = if use_constrained {
            self.tools
                .iter()
                .map(|t| grammar::Tool::from_schema(&t.function.name, &t.function.parameters))
                .collect()
        } else {
            Vec::new()
        };
        let grammar_for_step = use_constrained.then(|| Grammar::for_tools(&grammar_tools, &self.id_values));
        if use_constrained && self.token_vocab.is_none() {
            self.token_vocab = Some(TokenVocab::from_tokenizer(&self.tokenizer));
        }
        let mut constraint_impl: Option<GrammarConstraint> = grammar_for_step
            .as_ref()
            .map(|g| GrammarConstraint::new(g, &self.tokenizer, self.token_vocab.as_ref().expect("built above")));
        let constraint: Option<&mut dyn Constraint> =
            constraint_impl.as_mut().map(|c| c as &mut dyn Constraint);

        let start = Instant::now();
        let out = self
            .generator
            .generate_constrained(
                &prompt_tokens,
                prefix_len,
                self.max_new_tokens,
                self.tokenizer.eos_ids(),
                constraint,
            )
            .map_err(|e| format!("generation failed: {e}"))?;
        let timings = start.elapsed();
        let out_ids = out.ids;
        let tokens_generated = out_ids.len();
        let model_steps = out.model_steps;
        let forced_tokens = out.forced_tokens;
        let (prefill_time, decode_time) = match self.generator.last_call_timing() {
            (p, d) if p.is_zero() && d.is_zero() => (Duration::ZERO, timings),
            breakdown => breakdown,
        };

        let generated_text = self
            .tokenizer
            .decode(&out_ids, false)
            .map_err(|e| format!("failed to decode generated tokens: {e}"))?;
        let parsed = parse_output(&generated_text).map_err(|e| e.to_string());

        Ok(AttemptOutput {
            prompt_tokens,
            generated_text,
            parsed,
            timings,
            tokens_generated,
            prefill_time,
            decode_time,
            model_steps,
            forced_tokens,
        })
    }

    /// Common leading token-id run between rendering `utterance` and a
    /// throwaway probe utterance under the same `tools` — see module docs.
    fn prefix_len_for(&self, tools: &[Tool], utterance: &str) -> Result<usize> {
        let a = self.render_and_encode(tools, utterance)?;
        // A probe utterance chosen to diverge from any real first word;
        // only its token-level divergence point from `a` matters.
        let b = self.render_and_encode(tools, "\u{0}prefix-cache-probe\u{0}")?;
        Ok(a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count())
    }

    fn render_and_encode(&self, tools: &[Tool], utterance: &str) -> Result<Vec<u32>> {
        let messages = vec![Message::system(&self.system_prompt), Message::user(utterance)];
        let prompt = self.template.render_prompt(&messages, tools, true)?;
        self.tokenizer.encode(&prompt, false)
    }

    /// Run the agent loop for one user `utterance`, executing tool calls
    /// against `self.caller` until the model responds with plain text, or
    /// `max_steps` generations have happened without one (an error, not an
    /// infinite loop). A synchronous convenience built on `start` /
    /// `provide_tool_results`.
    pub fn run(&mut self, utterance: &str, tools: &[Tool], max_steps: usize) -> Result<Transcript> {
        self.max_steps = max_steps;
        let mut steps = Vec::with_capacity(max_steps);
        let mut outcome = self.start(utterance, tools);

        loop {
            match outcome {
                StepOutcome::Final { text, step } => {
                    steps.push(step);
                    return Ok(Transcript { steps, final_text: text });
                }
                StepOutcome::NeedTools { calls, step } => {
                    steps.push(step);
                    let mut results = Vec::with_capacity(calls.len());
                    for call in &calls {
                        let result = self.caller.call(&call.name, &call.arguments)?;
                        results.push((call.call_id.clone(), result));
                    }
                    outcome = self.provide_tool_results(results);
                }
                StepOutcome::Error { message, step } => {
                    if let Some(step) = step {
                        steps.push(step);
                    }
                    bail!("{message}");
                }
            }
        }
    }
}

/// Build `Tool`s from raw MCP `tools/list` entries, optionally running
/// them through `schemadiet::diet_tools` first — see `Agent::start_from_mcp`
/// and `docs/ENGINE.md`'s "Tool-schema token diet" hook-in point. Entries
/// missing `name`/`description`/`inputSchema` are skipped rather than
/// panicking (same "don't trust the wire" posture as `FixtureCaller`).
fn tools_from_mcp(raw_tools: &[Value], diet: bool) -> Vec<Tool> {
    let dieted;
    let raw_tools = if diet {
        dieted = diet_tools(raw_tools, AGENT_DIET_LEVEL);
        &dieted
    } else {
        raw_tools
    };
    raw_tools
        .iter()
        .filter_map(|t| {
            Some(Tool::from_mcp(
                t.get("name")?.as_str()?,
                t.get("description")?.as_str()?,
                t.get("inputSchema")?.clone(),
            ))
        })
        .collect()
}

fn pending_calls_from(calls: &[ToolCall]) -> Vec<PendingToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| PendingToolCall {
            call_id: format!("call_{i}"),
            name: c.name.clone(),
            arguments: c.arguments.clone(),
        })
        .collect()
}

fn tool_call_entries(pending: &[PendingToolCall]) -> Vec<ToolCallEntry> {
    pending
        .iter()
        .map(|c| ToolCallEntry {
            id: Some(c.call_id.clone()),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            },
        })
        .collect()
}
