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
//! Every step's rendered prompt is almost entirely a prefix of the
//! previous step's: not just the constant system + tool-schema preamble
//! (see `template.rs`), but the whole conversation tail up to that point —
//! prior tool calls, tool results, everything. `Agent` tracks
//! `resident_tokens`, its belief about what the `Generator`'s KV cache
//! physically holds right now (the last rendered prompt plus whatever of
//! its generation was actually forwarded — see that field's doc comment),
//! and on every `generate_attempt` call computes the longest run of
//! leading tokens the new prompt shares with it (`common_prefix_len`) as
//! the `prefix_len` hint passed to
//! [`Generator::generate_with_cached_prefix`]. A concrete `Generator`
//! backed by a real model can then `KvCache::restore()` to that length and
//! prefill only the new suffix instead of the whole prompt (see `kv.rs`'s
//! `snapshot`/`restore`, which this trait method is designed to be
//! implemented against) — genuinely new tokens only, not the whole
//! conversation tail re-prefilled from the constant prefix every step (see
//! `tests/resident_reuse.rs`, which quantifies the difference).
//! `FixtureGenerator`'s default implementation just ignores the hint and
//! regenerates from the full prompt, since it has no cache to restore.

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
    /// Set when this step's grammar was rebuilt via
    /// `Grammar::for_tools_unrestricted_ids` instead of `Grammar::for_tools`
    /// because every still-callable tool was a read tool already called
    /// this turn (fail-open — see `docs/ENGINE.md` "Agent loop"). `false`
    /// for an unconstrained step.
    pub id_rule_relaxed: bool,
    /// Set when reaching this step required detecting and nudging past at
    /// least one repeated identical tool call this turn (see `step_inner`'s
    /// generic repeated-call loop guard, `docs/ENGINE.md` "Agent loop") —
    /// regardless of whether the step that follows the nudge is a normal
    /// model answer or a `forced_text_answer`. `false` when no repeat was
    /// detected reaching this step.
    pub repeat_guard: bool,
    /// Set when this step's text answer wasn't produced by the model's own
    /// choice but forced by regenerating under `Grammar::text_only` after
    /// the malformed/repeated-call retry budget was exhausted (see
    /// `step_inner`) — a read-only question the model kept re-querying
    /// instead of answering still gets a `Final` outcome instead of
    /// `Error`. `false` for every other step.
    pub forced_text_answer: bool,
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
    id_rule_relaxed: bool,
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
    /// Token ids the `Generator`'s KV cache actually holds, in order — the
    /// full sequence of the last rendered prompt plus whatever of its
    /// generation was forwarded into the cache (every generated token
    /// except a trailing stop id, which ends the decode loop before it's
    /// ever fed through a forward pass — see module docs and
    /// `generate_attempt`). Not cleared by `reset`/`start`: it's a belief
    /// about the *Generator's* physical cache state, which outlives an
    /// `Agent`-level conversation reset — see `reset`'s doc comment.
    resident_tokens: Vec<u32>,

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
    /// Every tool call made so far this turn whose result was *not* an
    /// error, in order — see `step_inner`'s generic repeated-call loop
    /// guard: a call identical (same name+args) to one of these is never
    /// re-executed, no matter how many steps back it was made. A call
    /// identical to one whose *earlier* result was an error is deliberately
    /// excluded from this list — retrying a failed call is legitimate.
    /// Reset in `start`/`reset`.
    successful_calls_this_turn: Vec<ToolCall>,
    /// Every tool call the model has been given (i.e. every call in a
    /// `NeedTools` step) so far this turn, in order — used only by the
    /// fail-open "is the model stuck" check in `generate_attempt` (see
    /// `docs/ENGINE.md` "Agent loop" — "fail-open"). Reset in
    /// `start`/`reset`.
    calls_made_this_turn: Vec<ToolCall>,
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
            resident_tokens: Vec::new(),
            constrained: false,
            id_values: IdValues::new(),
            token_vocab: None,
            successful_calls_this_turn: Vec::new(),
            calls_made_this_turn: Vec::new(),
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
    /// outcome, or to abandon a turn). Does not touch `resident_tokens`:
    /// it tracks the `Generator`'s physical KV cache, not `Agent`'s own
    /// conversation state, and is still a valid (if now-stale) prefix to
    /// diff the next turn's render against — see `generate_attempt`.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.tools.clear();
        self.step_index = 0;
        self.pending_calls.clear();
        self.consecutive_tool_errors = 0;
        self.id_values = IdValues::new();
        self.successful_calls_this_turn.clear();
        self.calls_made_this_turn.clear();
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
        self.successful_calls_this_turn.clear();
        self.calls_made_this_turn.clear();

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
                    self.successful_calls_this_turn.push(ToolCall {
                        name: pending.name.clone(),
                        arguments: pending.arguments.clone(),
                    });
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
        let mut repeat_guard = false;
        loop {
            let force_constrained = retries == 1 && !self.constrained && !self.tools.is_empty();
            let attempt = match self.generate_attempt(force_constrained, false) {
                Ok(a) => a,
                Err(message) => return StepOutcome::Error { message, step: None },
            };

            let empty_calls = matches!(&attempt.parsed, Ok(ParsedOutput::ToolCalls(calls)) if calls.is_empty());
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

            // Generic repeated-call loop guard (`docs/ENGINE.md` "Agent
            // loop"): a well-formed, valid call identical (same name+args)
            // to one already made *and answered without an error* earlier
            // this run — not only the immediately preceding step — is a
            // wasted step, not a genuine retry of anything — seen live as
            // six consecutive `get_households_and_groups_and_players({})`
            // steps once the model had nothing new to ask about. A repeat
            // of a call whose earlier result *was* an error is excluded
            // (not in `successful_calls_this_turn`) — retrying a failed
            // call is legitimate.
            let is_repeat = !invalid
                && matches!(&attempt.parsed, Ok(ParsedOutput::ToolCalls(calls))
                    if calls.iter().any(|c| self.successful_calls_this_turn.contains(c)));
            if is_repeat {
                repeat_guard = true;
            }

            if invalid || is_repeat {
                if retries < self.max_retries {
                    retries += 1;
                    // A repeat's problem isn't output *shape* (the
                    // constraint, if any, was already satisfied) — forcing
                    // it on again buys nothing, so always go straight to
                    // the nudge for a repeat; a genuinely invalid first
                    // attempt still gets one constrained-only retry first.
                    if is_repeat || retries == 2 {
                        self.messages.push(Message::user(RETRY_NOTE));
                    }
                    continue;
                }
                // Retries exhausted. A repeated or empty call means the
                // model has (or believes it has) everything it needs but
                // won't say so — force a text-only answer instead of
                // erroring out with no answer at all. Any other kind of
                // invalid output (unparsable JSON, an unknown tool name)
                // still gives up with `Error`, since there's no evidence
                // the model has anything useful to say.
                if is_repeat || empty_calls {
                    return self.force_final_answer(repeat_guard);
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
                id_rule_relaxed,
            } = attempt;
            let parsed = parsed.expect("checked valid above");
            let tool_errors = self.consecutive_tool_errors;

            return match parsed {
                ParsedOutput::ToolCalls(calls) => {
                    let pending = pending_calls_from(&calls);
                    let entries = tool_call_entries(&pending);
                    self.messages.push(Message::assistant_tool_calls(entries));
                    self.pending_calls = pending.clone();
                    self.calls_made_this_turn.extend(calls.iter().cloned());

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
                        id_rule_relaxed,
                        repeat_guard,
                        forced_text_answer: false,
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
                        id_rule_relaxed,
                        repeat_guard,
                        forced_text_answer: false,
                    };
                    StepOutcome::Final { text, step }
                }
            };
        }
    }

    /// Last resort when `step_inner`'s malformed/repeated-call retry budget
    /// is exhausted on a repeated or empty tool call: regenerate this step
    /// once more under `Grammar::text_only` (no tool-call array allowed at
    /// all — see that constructor's doc comment) and return whatever text
    /// comes back as `StepOutcome::Final`, so a read-only question the
    /// model kept re-querying instead of answering still ends with an
    /// answer instead of `StepOutcome::Error`. Only a genuine generation
    /// failure (render/encode/generate/decode) still returns `Error` here.
    fn force_final_answer(&mut self, repeat_guard: bool) -> StepOutcome {
        let attempt = match self.generate_attempt(false, true) {
            Ok(a) => a,
            Err(message) => return StepOutcome::Error { message, step: None },
        };
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
            id_rule_relaxed: _,
        } = attempt;
        // The text-only grammar forbids a leading `[`, so `parsed` should
        // always come back `Ok(ParsedOutput::Text(_))` — but fall back to
        // the raw generated text rather than erroring on the off chance it
        // doesn't (an empty/odd forced answer is still better than no
        // answer at all).
        let text = match &parsed {
            Ok(ParsedOutput::Text(t)) => t.clone(),
            _ => generated_text.clone(),
        };
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
            retries: self.max_retries,
            tool_errors: self.consecutive_tool_errors,
            id_rule_relaxed: false,
            repeat_guard,
            forced_text_answer: true,
        };
        StepOutcome::Final { text, step }
    }

    /// One render -> encode -> generate -> decode -> parse round, with two
    /// independent overrides used by `step_inner`'s retry policy:
    /// `force_constrained` turns schema-constrained decoding on for this
    /// attempt regardless of `self.constrained`; `force_text_only` (used
    /// only by `force_final_answer`) replaces the whole tool-call grammar
    /// with `Grammar::text_only`, so the model cannot emit a tool-call
    /// array at all this attempt no matter what `self.tools`/`self.constrained`
    /// say. The two are mutually exclusive in practice (`force_final_answer`
    /// always passes `force_constrained: false`). Returns `Err(message)`
    /// only for failures that aren't part of the retry policy (render/
    /// encode/generate/decode) — a parse failure is returned as `Ok` with
    /// `parsed: Err(_)` so the caller can decide whether to retry.
    fn generate_attempt(&mut self, force_constrained: bool, force_text_only: bool) -> Result<AttemptOutput, String> {
        let prompt = self
            .template
            .render_prompt(&self.messages, &self.tools, true)
            .map_err(|e| format!("failed to render prompt: {e}"))?;
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(|e| format!("failed to encode prompt: {e}"))?;

        // Longest run of leading tokens `prompt_tokens` shares with what's
        // actually resident in the `Generator`'s KV cache right now (see
        // `resident_tokens`'s doc comment) — a prefix of *both* sequences
        // by construction, and never longer than `resident_tokens`, so
        // it's always safe to pass as `prefix_len`'s "already resident"
        // hint below.
        let mut prefix_len = common_prefix_len(&self.resident_tokens, &prompt_tokens);
        // The repeat-guard / forced-text-answer retry paths re-render the
        // exact same prompt a previous step already prefilled, so the
        // common prefix can cover the *whole* prompt. Step back one token
        // so `Generator::generate_constrained` always has at least the
        // final token to prefill — generation needs a last-position logit
        // to sample from regardless of whether anything is actually "new"
        // (mirrors `web.rs::generate_attempt`'s `effective_prefix` clamp).
        if prompt_tokens.is_empty() {
            return Err("empty prompt: nothing to prefill".to_string());
        }
        if prefix_len == prompt_tokens.len() {
            prefix_len -= 1;
        }

        // Build this step's schema constraint (if `constrained`, or this
        // attempt forces it on) from the current tool set + every id
        // harvested from tool results so far — see `docs/ENGINE.md`
        // "Schema-constrained decoding". `force_text_only` bypasses all of
        // this: the constraint is `Grammar::text_only` outright, and the
        // id rule / fail-open logic (which only makes sense for a tool-call
        // grammar) never runs.
        let use_constrained = self.constrained || force_constrained || force_text_only;
        let grammar_tools: Vec<grammar::Tool> = if use_constrained && !force_text_only {
            self.tools
                .iter()
                .map(|t| grammar::Tool::from_schema(&t.function.name, &t.function.parameters))
                .collect()
        } else {
            Vec::new()
        };
        let mut grammar_for_step = if force_text_only {
            Some(Grammar::text_only())
        } else {
            use_constrained.then(|| Grammar::for_tools(&grammar_tools, &self.id_values))
        };
        // Fail-open (`docs/ENGINE.md` "Agent loop" — "fail-open"): if the
        // id rule left the model nothing new to call — every still-
        // callable tool is a read tool it's already called this turn —
        // rebuild without the id restriction instead of leaving it boxed
        // into repeating itself (that repeat is what `is_repeat` above
        // would otherwise have to catch one wasted step later).
        let id_rule_relaxed = !force_text_only
            && grammar_for_step.as_ref().is_some_and(|g| {
                let callable = g.callable_tool_names();
                !callable.is_empty()
                    && callable
                        .iter()
                        .all(|name| is_read_tool(name) && self.calls_made_this_turn.iter().any(|c| &c.name == name))
            });
        if id_rule_relaxed {
            grammar_for_step = Some(Grammar::for_tools_unrestricted_ids(&grammar_tools));
        }
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

        // Record what's now actually resident in the `Generator`'s cache:
        // this attempt's prompt, plus every generated token that was fed
        // through a forward pass — i.e. all of `out_ids` except a trailing
        // stop id, which ends the decode loop before it's ever forwarded
        // (see `resident_tokens`'s doc comment). Updated unconditionally,
        // even for an attempt `step_inner` goes on to discard as invalid —
        // the physical cache holds these tokens regardless of whether the
        // output parsed.
        self.resident_tokens = prompt_tokens.clone();
        let forwarded = match out_ids.last() {
            Some(last) if self.tokenizer.eos_ids().contains(last) => &out_ids[..out_ids.len() - 1],
            _ => &out_ids[..],
        };
        self.resident_tokens.extend_from_slice(forwarded);

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
            id_rule_relaxed,
        })
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
/// Naming-convention heuristic for a read (non-mutating) tool — no MCP
/// annotation to consult generically (`readOnlyHint` is stripped by the
/// schema diet's `annotations` drop, and isn't guaranteed present even
/// undieted), so this mirrors the one Sonos-adjacent-but-not-Sonos-specific
/// convention already used elsewhere in this codebase (`tool_order`'s
/// "listing-first"): a `get_`/`list_` prefix. Used only by the fail-open
/// check in `generate_attempt` — see `docs/ENGINE.md` "Agent loop".
fn is_read_tool(name: &str) -> bool {
    name.starts_with("get_") || name.starts_with("list_")
}

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

/// Longest run of leading token ids shared by `a` and `b` — a prefix of
/// both sequences by construction. Used to find how much of `resident_tokens`
/// still applies to a freshly rendered prompt (see `generate_attempt`).
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
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
