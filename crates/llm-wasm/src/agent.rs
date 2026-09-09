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

use crate::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use crate::tokenizer::Tokenizer;
use crate::tools::{format_tool_result, parse_output, ParsedOutput, ToolCall};
use anyhow::{bail, Result};
use serde_json::Value;
use std::time::{Duration, Instant};

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
}

/// One render -> generate -> parse round of the agent loop.
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
}

pub struct Transcript {
    pub steps: Vec<Step>,
    pub final_text: String,
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
        Ok(serde_json::from_str(&data)?)
    }
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

    // Step-wise conversation state (see `start`/`provide_tool_results`).
    messages: Vec<Message>,
    tools: Vec<Tool>,
    step_index: usize,
    pending_calls: Vec<PendingToolCall>,
    /// Cached (tools set, common prefix token length), recomputed only when
    /// `tools` changes between `start` calls — see module docs.
    prefix_cache: Option<(Vec<Tool>, usize)>,
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
            messages: Vec::new(),
            tools: Vec::new(),
            step_index: 0,
            pending_calls: Vec::new(),
            prefix_cache: None,
        }
    }

    pub fn caller(&self) -> &C {
        &self.caller
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Override the step budget the step-wise API (`start` /
    /// `provide_tool_results`) errors out after. `run` ignores this — it
    /// takes its own `max_steps` argument.
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    /// Drop any in-progress conversation state (e.g. after an `Error`
    /// outcome, or to abandon a turn). Does not touch the KV-cache prefix
    /// cache, which is keyed by `tools` and safe to keep across turns.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.tools.clear();
        self.step_index = 0;
        self.pending_calls.clear();
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

    /// Continue the current turn with the results of the tool calls from
    /// the most recent `NeedTools` outcome, keyed by `call_id`. Unmatched
    /// or missing `call_id`s are silently skipped (their tool message is
    /// never appended) — callers should supply exactly the `call_id`s from
    /// that `NeedTools`.
    pub fn provide_tool_results(&mut self, results: Vec<(String, Value)>) -> StepOutcome {
        for (call_id, result) in results {
            if let Some(pending) = self.pending_calls.iter().find(|c| c.call_id == call_id) {
                self.messages.push(format_tool_result(&pending.name, &result));
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

        let prompt = match self.template.render_prompt(&self.messages, &self.tools, true) {
            Ok(p) => p,
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("failed to render prompt: {e}"),
                    step: None,
                }
            }
        };
        let prompt_tokens = match self.tokenizer.encode(&prompt, false) {
            Ok(t) => t,
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("failed to encode prompt: {e}"),
                    step: None,
                }
            }
        };

        let prefix_len = self
            .prefix_cache
            .as_ref()
            .filter(|(cached_tools, _)| cached_tools == &self.tools)
            .map(|(_, len)| (*len).min(prompt_tokens.len()))
            .unwrap_or(0);

        let start = Instant::now();
        let out_ids = match self.generator.generate_with_cached_prefix(
            &prompt_tokens,
            prefix_len,
            self.max_new_tokens,
            self.tokenizer.eos_ids(),
        ) {
            Ok(ids) => ids,
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("generation failed: {e}"),
                    step: None,
                }
            }
        };
        let timings = start.elapsed();

        let generated_text = match self.tokenizer.decode(&out_ids, false) {
            Ok(t) => t,
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("failed to decode generated tokens: {e}"),
                    step: None,
                }
            }
        };
        let parsed = match parse_output(&generated_text) {
            Ok(p) => p,
            Err(e) => {
                return StepOutcome::Error {
                    message: format!("failed to parse model output: {e}"),
                    step: None,
                }
            }
        };

        match parsed {
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
                };
                StepOutcome::Final { text, step }
            }
        }
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
