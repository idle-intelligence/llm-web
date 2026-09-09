//! Agent loop: render -> encode -> generate -> parse -> [tool results ->
//! repeat] -> text. Owned by phase 1a.

use crate::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use crate::tokenizer::Tokenizer;
use crate::tools::{format_tool_result, parse_output, ParsedOutput};
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
}

/// One render -> generate -> parse round of the agent loop.
pub struct Step {
    pub prompt_tokens: Vec<u32>,
    pub generated_text: String,
    pub parsed: ParsedOutput,
    /// Set when this step's parsed output was tool call(s): the result of
    /// the *last* call made this step (each call's own result is also fed
    /// back into the conversation as a `tool` message regardless).
    pub tool_result: Option<Value>,
    pub timings: Duration,
}

pub struct Transcript {
    pub steps: Vec<Step>,
    pub final_text: String,
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
        }
    }

    /// Run the agent loop for one user `utterance`, executing tool calls
    /// against `self.caller` until the model responds with plain text, or
    /// `max_steps` generations have happened without one (an error, not an
    /// infinite loop).
    pub fn caller(&self) -> &C {
        &self.caller
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn run(&mut self, utterance: &str, tools: &[Tool], max_steps: usize) -> Result<Transcript> {
        let mut messages = vec![Message::system(&self.system_prompt), Message::user(utterance)];
        let mut steps = Vec::with_capacity(max_steps);

        for _ in 0..max_steps {
            let prompt = self.template.render_prompt(&messages, tools, true)?;
            let prompt_tokens = self.tokenizer.encode(&prompt, false)?;

            let start = Instant::now();
            let out_ids =
                self.generator
                    .generate(&prompt_tokens, self.max_new_tokens, self.tokenizer.eos_ids())?;
            let timings = start.elapsed();

            let generated_text = self.tokenizer.decode(&out_ids, false)?;
            let parsed = parse_output(&generated_text)
                .map_err(|e| anyhow::anyhow!("failed to parse model output: {e}"))?;

            match &parsed {
                ParsedOutput::ToolCalls(calls) => {
                    let entries: Vec<ToolCallEntry> = calls
                        .iter()
                        .enumerate()
                        .map(|(i, c)| ToolCallEntry {
                            id: Some(format!("call_{i}")),
                            kind: "function".to_string(),
                            function: ToolCallFunction {
                                name: c.name.clone(),
                                arguments: c.arguments.clone(),
                            },
                        })
                        .collect();
                    messages.push(Message::assistant_tool_calls(entries));

                    let mut last_result = None;
                    for call in calls {
                        let result = self.caller.call(&call.name, &call.arguments)?;
                        messages.push(format_tool_result(&call.name, &result));
                        last_result = Some(result);
                    }

                    steps.push(Step {
                        prompt_tokens,
                        generated_text,
                        parsed,
                        tool_result: last_result,
                        timings,
                    });
                }
                ParsedOutput::Text(text) => {
                    let final_text = text.clone();
                    steps.push(Step {
                        prompt_tokens,
                        generated_text,
                        parsed,
                        tool_result: None,
                        timings,
                    });
                    return Ok(Transcript { steps, final_text });
                }
            }
        }

        bail!("agent exceeded max_steps ({max_steps}) without a final text response");
    }
}
