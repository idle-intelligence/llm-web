//! Offline Sonos MCP tool-call eval (`eval/README.md`, phase 3 of
//! `sonos/PLAN.md`). No live Sonos speakers involved: the agent runs
//! against `eval/utterances.json` with a [`FixtureCaller`](crate::agent::FixtureCaller)
//! answering every tool call from `fixtures/sonos/results/`, and each run is
//! scored offline against the utterance's expected tool-call sequence.
//!
//! See `eval/README.md` for the full scoring rules this module implements;
//! [`score`] is the literal encoding of that section.

use crate::agent::{Agent, FixtureCaller, Generator, StepOutcome, ToolCaller};
use crate::template::Tool;
use crate::tools::ToolCall;
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

/// How an [`EvalCase`] is scored against the calls the agent actually made
/// — see `eval/README.md`'s "Scoring rules".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accept {
    Exact,
    Subset,
    /// `expected` is `[]`; correct iff no mutating tool call was made
    /// (reads are free) and the run ended in a final text answer, not
    /// max-steps — see `eval/README.md`'s `m21`.
    #[serde(rename = "no_mutation")]
    NoMutation,
    /// `expected` is `[]`; correct iff at least one mutating `play_*`
    /// call was made with a `group_id` that exists in the fixture
    /// household — see `eval/README.md`'s `m22`.
    #[serde(rename = "any_play")]
    AnyPlay,
}

/// One expected tool call in an [`EvalCase`]'s `expected` list.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ExpectedCall {
    pub name: String,
    pub args: Value,
}

/// One scored utterance from `eval/utterances.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub utterance: String,
    pub kind: String,
    pub expected: Vec<ExpectedCall>,
    pub accept: Accept,
    pub tools12_ok: bool,
    #[serde(default)]
    pub notes: String,
    /// `"fr"` etc. for a non-English utterance; scored identically to an
    /// equivalent English item (`eval/README.md`). Unknown/absent fields
    /// like this and `pending_harness` deserialise fine regardless —
    /// serde ignores JSON keys with no matching struct field.
    #[serde(default)]
    pub lang: Option<String>,
}

/// Load the scored utterances from `eval/utterances.json`.
pub fn load_cases(path: impl AsRef<Path>) -> Result<Vec<EvalCase>> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading eval cases from {path:?}"))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing eval cases from {path:?}"))
}

/// Which tool set an eval run exercises — the full 34-tool schema, or the
/// 12-tool subset (`fixtures/sonos/tools-12.json`) some utterances aren't
/// solvable under (see [`EvalCase::tools12_ok`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSet {
    All,
    Twelve,
}

impl ToolSet {
    /// Number of tools in this set, for the results table header — fixed
    /// by the two committed fixture files (34 and 12 tools respectively).
    pub fn tool_count(&self) -> usize {
        match self {
            ToolSet::All => 34,
            ToolSet::Twelve => 12,
        }
    }
}

impl std::fmt::Display for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolSet::All => write!(f, "34 tools"),
            ToolSet::Twelve => write!(f, "12 tools"),
        }
    }
}

/// Parse a `fixtures/sonos/tools.json`/`tools-12.json`-shaped array of MCP
/// `tools/list` entries into [`Tool`]s.
fn load_mcp_tools(path: impl AsRef<Path>) -> Result<Vec<Tool>> {
    let path = path.as_ref();
    let raw: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading tools from {path:?}"))?,
    )?;
    raw.into_iter()
        .map(|t| {
            let name = t["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool entry missing string `name`: {t}"))?;
            let description = t["description"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool entry missing string `description`: {t}"))?;
            Ok(Tool::from_mcp(name, description, t["inputSchema"].clone()))
        })
        .collect()
}

/// Load `fixtures/sonos/tools.json` (the full 34-tool schema) from
/// `fixtures_dir`.
pub fn load_all_tools(fixtures_dir: impl AsRef<Path>) -> Result<Vec<Tool>> {
    load_mcp_tools(fixtures_dir.as_ref().join("tools.json"))
}

/// Load `fixtures/sonos/tools.json` as raw MCP `tools/list` entries
/// (`Value`s), kept opaque rather than converted to [`Tool`] — used only
/// to look up a tool's `annotations.readOnlyHint` when deciding whether a
/// call has a real side effect (see [`is_mutating`]).
pub fn load_raw_tools(fixtures_dir: impl AsRef<Path>) -> Result<Vec<Value>> {
    let path = fixtures_dir.as_ref().join("tools.json");
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading tools from {path:?}"))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing tools from {path:?}"))
}

/// Whether a tool call named `name` has a real side effect. Prefers the
/// tool's own `annotations.readOnlyHint` from `raw_tools` (MCP's own
/// signal: `true` means read-only) when present; falls back to the
/// `get_`-prefix heuristic used elsewhere in this module otherwise — so
/// this works for any MCP tool set, not just Sonos's, once servers start
/// annotating.
fn is_mutating(name: &str, raw_tools: &[Value]) -> bool {
    if let Some(hint) = raw_tools
        .iter()
        .find(|t| t["name"].as_str() == Some(name))
        .and_then(|t| t["annotations"]["readOnlyHint"].as_bool())
    {
        return !hint;
    }
    !name.starts_with("get_")
}

/// The set of `groupId`s present in a `get_households_and_groups_and_players`
/// result — used by `accept: "any_play"` to check a `play_*` call's
/// `group_id` against a real group. `households` is the parsed tool result
/// captured mid-run (see [`run_case`]) — `None` (the discovery tool was
/// never called this run) or an unparseable value both yield an empty set,
/// since this is scoring-time best effort, not a hard dependency.
fn valid_group_ids(households: Option<&Value>) -> std::collections::HashSet<String> {
    let Some(parsed) = households else {
        return Default::default();
    };
    parsed["households"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|h| h["groups"].as_array().cloned().unwrap_or_default())
        .filter_map(|g| g["groupId"].as_str().map(|s| s.to_string()))
        .collect()
}

/// Select the [`Tool`]s for a given [`ToolSet`]: `all` unchanged for
/// [`ToolSet::All`], or `all`'s schema entries for the names in
/// `fixtures/sonos/tools-12.json`, **in `tools-12.json`'s own order**, for
/// [`ToolSet::Twelve`] (so the returned tools carry whatever schema `all`
/// was built from, but the model sees the curated 12-tool ordering — e.g.
/// the listing tool first — not `tools.json`'s alphabetical order. Getting
/// this wrong silently reorders the tool preamble the model sees, which
/// changes its greedy output independently of any KV-cache/prefill
/// behavior — see docs/ENGINE.md "Known issues / fixed").
pub fn select_tools(all: &[Tool], subset: ToolSet, fixtures_dir: impl AsRef<Path>) -> Result<Vec<Tool>> {
    match subset {
        ToolSet::All => Ok(all.to_vec()),
        ToolSet::Twelve => {
            let twelve = load_mcp_tools(fixtures_dir.as_ref().join("tools-12.json"))?;
            twelve
                .iter()
                .map(|t| {
                    all.iter()
                        .find(|a| a.function.name == t.function.name)
                        .cloned()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "tools-12.json tool `{}` not found in the full tool set",
                                t.function.name
                            )
                        })
                })
                .collect()
        }
    }
}

/// Which order the tool preamble is rendered in — independent of
/// [`ToolSet`] (see docs/ENGINE.md "Known issues / fixed": tool order
/// changes the model's first call). `ListingFirst` puts
/// `get_households_and_groups_and_players` first; `Alphabetical` uses
/// `tools.json`'s own (alphabetical) order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolOrder {
    Alphabetical,
    ListingFirst,
}

impl std::fmt::Display for ToolOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolOrder::Alphabetical => write!(f, "alphabetical"),
            ToolOrder::ListingFirst => write!(f, "listing-first"),
        }
    }
}

/// [`select_tools`] plus a [`ToolOrder`] choice, for both [`ToolSet`]s:
///
/// - `Twelve` + `ListingFirst`: `tools-12.json`'s own order (identical to
///   [`select_tools`] — the fixed behavior).
/// - `Twelve` + `Alphabetical`: the pre-fix behavior — `all`'s
///   (alphabetical) order filtered down to `tools-12.json`'s names,
///   discarding `tools-12.json`'s curated order.
/// - `All` + `Alphabetical`: `all` unchanged (identical to
///   [`select_tools`] — `tools.json` is already alphabetical).
/// - `All` + `ListingFirst`: `fixtures/sonos/tools-listing-first.json`'s
///   order (the listing tool first, then the remaining 33 in `tools.json`
///   order), looked up against `all`'s schemas.
pub fn select_tools_ordered(
    all: &[Tool],
    subset: ToolSet,
    order: ToolOrder,
    fixtures_dir: impl AsRef<Path>,
) -> Result<Vec<Tool>> {
    let fixtures_dir = fixtures_dir.as_ref();
    match (subset, order) {
        (ToolSet::Twelve, ToolOrder::ListingFirst) => select_tools(all, subset, fixtures_dir),
        (ToolSet::All, ToolOrder::Alphabetical) => Ok(all.to_vec()),
        (ToolSet::Twelve, ToolOrder::Alphabetical) => {
            let twelve = load_mcp_tools(fixtures_dir.join("tools-12.json"))?;
            let names: std::collections::HashSet<&str> =
                twelve.iter().map(|t| t.function.name.as_str()).collect();
            Ok(all
                .iter()
                .filter(|t| names.contains(t.function.name.as_str()))
                .cloned()
                .collect())
        }
        (ToolSet::All, ToolOrder::ListingFirst) => {
            let ordered = load_mcp_tools(fixtures_dir.join("tools-listing-first.json"))?;
            ordered
                .iter()
                .map(|t| {
                    all.iter()
                        .find(|a| a.function.name == t.function.name)
                        .cloned()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "tools-listing-first.json tool `{}` not found in the full tool set",
                                t.function.name
                            )
                        })
                })
                .collect()
        }
    }
}

/// Outcome of running one [`EvalCase`] against an [`Agent`].
#[derive(Debug, Clone)]
pub struct CaseResult {
    pub id: String,
    /// `false` when `skipped` is `true` — a skipped case is neither correct
    /// nor incorrect, it's excluded from the percentage entirely.
    pub correct: bool,
    /// `true` when this case's `tools12_ok` is `false` under
    /// [`ToolSet::Twelve`] — not run, excluded from `correct_pct`.
    pub skipped: bool,
    pub reason: String,
    pub steps: usize,
    pub calls_made: Vec<ToolCall>,
    pub prefill_ms_total: f64,
    pub decode_ms_total: f64,
    pub tokens_generated: usize,
    pub total_ms: f64,
}

/// Max tool calls tolerated per utterance before it's scored incorrect
/// regardless of whether it would eventually reach the expected sequence
/// (`eval/README.md`, "Max 6 steps per utterance").
const MAX_TOOL_CALLS: usize = 6;

/// Whether `made` satisfies `expected` per `eval/README.md`'s "Args match
/// by subset" rule: every key in `expected` (when it's a JSON object) must
/// be present in `made` with the exact same value; keys in `made` beyond
/// `expected` are ignored (e.g. a schema-required param the utterance
/// doesn't determine, like `play_artist`'s `music_service`).
fn args_match(expected: &Value, made: &Value) -> bool {
    match (expected, made) {
        (Value::Object(exp), Value::Object(got)) => {
            exp.iter().all(|(k, v)| got.get(k) == Some(v))
        }
        _ => expected == made,
    }
}

/// Score `calls_made` (in order) against `case.expected` per
/// `eval/README.md`'s "Scoring rules". Returns `(correct, reason)`.
///
/// `raw_tools` and `households` are only consulted by `no_mutation` /
/// `any_play` cases (mutation detection and `group_id` validation
/// respectively — `households` is the parsed
/// `get_households_and_groups_and_players` result captured during the run,
/// if any, see [`run_case`]); `ended_in_final` is whether the run
/// terminated with a [`crate::agent::StepOutcome::Final`] rather than
/// hitting max-steps or erroring — `no_mutation` requires it.
pub fn score(
    case: &EvalCase,
    calls_made: &[ToolCall],
    raw_tools: &[Value],
    households: Option<&Value>,
    ended_in_final: bool,
) -> (bool, String) {
    if calls_made.len() > MAX_TOOL_CALLS {
        return (
            false,
            format!("exceeded max steps: {} tool calls made (limit {MAX_TOOL_CALLS})", calls_made.len()),
        );
    }

    match case.accept {
        Accept::Exact => {
            if calls_made.len() != case.expected.len() {
                return (
                    false,
                    format!(
                        "exact mode: expected {} call(s), got {}",
                        case.expected.len(),
                        calls_made.len()
                    ),
                );
            }
            for (i, expected) in case.expected.iter().enumerate() {
                let made = &calls_made[i];
                if made.name != expected.name || made.arguments != expected.args {
                    return (
                        false,
                        format!(
                            "exact mode: call {i} mismatch: expected {}({}), got {}({})",
                            expected.name, expected.args, made.name, made.arguments
                        ),
                    );
                }
            }
            (true, "matched expected sequence exactly".to_string())
        }
        Accept::Subset => {
            let required: Vec<&ExpectedCall> = case
                .expected
                .iter()
                .filter(|c| !c.name.starts_with("get_"))
                .collect();

            if !required.is_empty() {
                // Every mutating expected call must appear, in order, with
                // exact args; reads (expected or not) are free.
                let mut cursor = 0usize;
                for expected in &required {
                    match calls_made[cursor..]
                        .iter()
                        .position(|c| c.name == expected.name && args_match(&expected.args, &c.arguments))
                    {
                        Some(offset) => cursor += offset + 1,
                        None => {
                            return (
                                false,
                                format!(
                                    "missing required call {}({}) in order after step {cursor}",
                                    expected.name, expected.args
                                )
                            )
                        }
                    }
                }
                (true, "all required (mutating) calls made in order with correct args".to_string())
            } else {
                // Read-only case: no mutating call to check, so correctness
                // hinges on the final expected read happening with the
                // right args (eval/README.md's "Metrics per row" note).
                let last = case
                    .expected
                    .last()
                    .expect("EvalCase.expected is never empty");
                if calls_made.iter().any(|c| c.name == last.name && args_match(&last.args, &c.arguments)) {
                    (true, "final read made with correct args".to_string())
                } else {
                    (
                        false,
                        format!("final read {}({}) never made with correct args", last.name, last.args),
                    )
                }
            }
        }
        Accept::NoMutation => {
            if !ended_in_final {
                return (
                    false,
                    "no_mutation mode: run did not end in a final text answer".to_string(),
                );
            }
            match calls_made.iter().find(|c| is_mutating(&c.name, raw_tools)) {
                None => (true, "no mutating call made; final text answer given".to_string()),
                Some(c) => (
                    false,
                    format!("no_mutation mode: mutating call {}({}) was made", c.name, c.arguments),
                ),
            }
        }
        Accept::AnyPlay => {
            let group_ids = valid_group_ids(households);
            match calls_made.iter().find(|c| {
                c.name.starts_with("play_")
                    && is_mutating(&c.name, raw_tools)
                    && c.arguments
                        .get("group_id")
                        .and_then(|v| v.as_str())
                        .is_some_and(|id| group_ids.contains(id))
            }) {
                Some(c) => (
                    true,
                    format!("any_play mode: {}({}) made with a valid group_id", c.name, c.arguments),
                ),
                None => (
                    false,
                    "any_play mode: no play_* call with a valid group_id was made".to_string(),
                ),
            }
        }
    }
}

/// Run one [`EvalCase`] against `agent`, driving the step-wise API directly
/// (rather than [`Agent::run`]) so scoring can enforce
/// `eval/README.md`'s max-6-tool-calls rule independently of `Agent`'s own
/// step budget, and so `caller`'s `calls_seen` reflects exactly this case
/// (call with a fresh `FixtureCaller` per case).
///
/// Along the way this captures the result of any
/// `get_households_and_groups_and_players` call the run makes, so `score`
/// can validate an `accept: "any_play"` case's `group_id` against it
/// without this function needing a `results_dir` of its own — the fixture
/// tools set has no `annotations.readOnlyHint` entries today, so mutation
/// detection here falls back to the `get_`-prefix heuristic (see
/// [`is_mutating`]); `score` also accepts real tool annotations directly
/// for callers (e.g. [`run_all`]) that have them.
pub fn run_case<G: Generator>(
    agent: &mut Agent<G, FixtureCaller>,
    tools: &[Tool],
    case: &EvalCase,
    caller: &mut FixtureCaller,
) -> CaseResult {
    let wall_start = std::time::Instant::now();
    let mut steps = 0usize;
    let mut prefill_ms_total = 0.0f64;
    let mut decode_ms_total = 0.0f64;
    let mut tokens_generated = 0usize;
    let mut households: Option<Value> = None;

    let mut accumulate = |step: &crate::agent::Step| {
        prefill_ms_total += step.prefill_time.as_secs_f64() * 1000.0;
        decode_ms_total += step.decode_time.as_secs_f64() * 1000.0;
        tokens_generated += step.tokens_generated;
    };

    let mut outcome = agent.start(&case.utterance, tools);
    let (correct, reason) = loop {
        steps += 1;
        match outcome {
            StepOutcome::Final { step, .. } => {
                accumulate(&step);
                let calls_made = to_tool_calls(&caller.calls_seen);
                break score(case, &calls_made, &[], households.as_ref(), true);
            }
            StepOutcome::NeedTools { calls, step } => {
                accumulate(&step);
                let mut results = Vec::with_capacity(calls.len());
                for call in &calls {
                    let result = caller.call(&call.name, &call.arguments).unwrap_or(Value::Null);
                    if call.name == "get_households_and_groups_and_players" {
                        households = Some(result.clone());
                    }
                    results.push((call.call_id.clone(), result));
                }
                if caller.calls_seen.len() > MAX_TOOL_CALLS {
                    let calls_made = to_tool_calls(&caller.calls_seen);
                    break score(case, &calls_made, &[], households.as_ref(), false);
                }
                outcome = agent.provide_tool_results(results);
            }
            StepOutcome::Error { message, step } => {
                if let Some(step) = step {
                    accumulate(&step);
                }
                break (false, format!("agent error: {message}"));
            }
        }
    };

    let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    CaseResult {
        id: case.id.clone(),
        correct,
        skipped: false,
        reason,
        steps,
        calls_made: to_tool_calls(&caller.calls_seen),
        prefill_ms_total,
        decode_ms_total,
        tokens_generated,
        total_ms,
    }
}

fn to_tool_calls(calls_seen: &[(String, Value)]) -> Vec<ToolCall> {
    calls_seen
        .iter()
        .map(|(name, arguments)| ToolCall {
            name: name.clone(),
            arguments: arguments.clone(),
        })
        .collect()
}

/// A skipped case's result: not run, excluded from `correct_pct`
/// (`eval/README.md`: "not `tools12_ok` under the 12-tool set... record it
/// as `skipped: true`... exclude from the percentage").
fn skipped_result(case: &EvalCase) -> CaseResult {
    CaseResult {
        id: case.id.clone(),
        correct: false,
        skipped: true,
        reason: "tools12_ok=false: not solvable under the 12-tool subset".to_string(),
        steps: 0,
        calls_made: Vec::new(),
        prefill_ms_total: 0.0,
        decode_ms_total: 0.0,
        tokens_generated: 0,
        total_ms: 0.0,
    }
}

/// A completed eval run: every case's result plus the aggregate metrics
/// `render_markdown` reports.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub model: String,
    pub tool_set: ToolSet,
    pub backend: String,
    pub date: String,
    pub results: Vec<CaseResult>,
    pub correct_pct: f64,
    pub mean_steps: f64,
    pub mean_prefill_s: f64,
    pub mean_decode_tok_s: f64,
    pub mean_total_s: f64,
    /// `--label` this run was invoked with (`eval/README.md`'s run
    /// naming), verbatim into the parameters block.
    pub label: String,
    /// System prompt verbatim (`--system`, or the built-in default).
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub max_steps: usize,
    /// Token length of the constant system+tools prefix this run's cases
    /// shared (see `agent.rs` module docs on prefix caching).
    pub prefix_tokens: usize,
}

/// Run every case in `cases` against `agent` under `tool_set`, skipping
/// cases with `tools12_ok: false` when `tool_set` is [`ToolSet::Twelve`],
/// and aggregate the results into an [`EvalReport`]. `model` and `date` are
/// recorded verbatim into the report (this crate has no notion of a GGUF
/// filename or wall-clock date — the caller, e.g. the `llm-agent` CLI,
/// supplies them).
pub fn run_all<G: Generator>(
    agent: &mut Agent<G, FixtureCaller>,
    tools: &[Tool],
    cases: &[EvalCase],
    tool_set: ToolSet,
    results_dir: impl AsRef<Path>,
    model: impl Into<String>,
    date: impl Into<String>,
) -> EvalReport {
    let results_dir = results_dir.as_ref();
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        if tool_set == ToolSet::Twelve && !case.tools12_ok {
            results.push(skipped_result(case));
            continue;
        }
        let mut caller = FixtureCaller::new(results_dir);
        results.push(run_case(agent, tools, case, &mut caller));
    }

    let scored: Vec<&CaseResult> = results.iter().filter(|r| !r.skipped).collect();
    let n = scored.len().max(1) as f64;
    let correct_pct = scored.iter().filter(|r| r.correct).count() as f64 / n * 100.0;
    let mean_steps = scored.iter().map(|r| r.steps as f64).sum::<f64>() / n;
    let mean_prefill_s = scored.iter().map(|r| r.prefill_ms_total / 1000.0).sum::<f64>() / n;
    let mean_total_s = scored.iter().map(|r| r.total_ms / 1000.0).sum::<f64>() / n;
    let decode_tok_s_values: Vec<f64> = scored
        .iter()
        .filter_map(|r| {
            let decode_s = r.decode_ms_total / 1000.0;
            (decode_s > 0.0).then(|| r.tokens_generated as f64 / decode_s)
        })
        .collect();
    let mean_decode_tok_s = if decode_tok_s_values.is_empty() {
        0.0
    } else {
        decode_tok_s_values.iter().sum::<f64>() / decode_tok_s_values.len() as f64
    };

    EvalReport {
        model: model.into(),
        tool_set,
        backend: "native".to_string(),
        date: date.into(),
        results,
        correct_pct,
        mean_steps,
        mean_prefill_s,
        mean_decode_tok_s,
        mean_total_s,
        label: String::new(),
        system_prompt: String::new(),
        max_new_tokens: 0,
        max_steps: 0,
        prefix_tokens: 0,
    }
}

/// Render an [`EvalReport`] in the research-log layout `eval/README.md`
/// specifies: a parameters block, then a per-case table, then a summary
/// line. Equivalent to [`render_markdown_with_cases`] with no cases (so
/// never shows a `lang` column) — `CaseResult` itself carries no `lang`,
/// since it's the [`EvalCase`] a result came from that knows its language;
/// pass the cases the report was built from to `render_markdown_with_cases`
/// to get that column when any of them are non-English.
pub fn render_markdown(report: &EvalReport) -> String {
    render_markdown_with_cases(report, &[])
}

/// [`render_markdown`], plus a `lang` column (shown only if at least one of
/// `cases` has a `lang`) — cases are matched to `report.results` by `id`.
pub fn render_markdown_with_cases(report: &EvalReport, cases: &[EvalCase]) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let lang_of = |id: &str| -> Option<&str> {
        cases.iter().find(|c| c.id == id).and_then(|c| c.lang.as_deref())
    };

    let _ = writeln!(out, "# Sonos MCP agent eval — {}", report.date);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "## Run: {}, {}, {}, label={}",
        report.model, report.tool_set, report.backend, report.label
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "- system: {:?}", report.system_prompt);
    let _ = writeln!(out, "- tool_count: {}", report.tool_set.tool_count());
    let _ = writeln!(out, "- max_new_tokens: {}", report.max_new_tokens);
    let _ = writeln!(out, "- max_steps: {}", report.max_steps);
    let _ = writeln!(out, "- prefix_tokens: {}", report.prefix_tokens);
    let _ = writeln!(out);
    let show_lang = report.results.iter().any(|r| lang_of(&r.id).is_some());
    if show_lang {
        let _ = writeln!(out, "| id | correct | steps | prefill_s | decode_tok_s | total_s | lang | reason |");
        let _ = writeln!(out, "|----|---------|-------|-----------|---------------|---------|------|--------|");
    } else {
        let _ = writeln!(out, "| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |");
        let _ = writeln!(out, "|----|---------|-------|-----------|---------------|---------|--------|");
    }
    for r in &report.results {
        let lang = lang_of(&r.id).unwrap_or("-");
        if r.skipped {
            if show_lang {
                let _ = writeln!(out, "| {} | skipped | - | - | - | - | {} | {} |", r.id, lang, r.reason);
            } else {
                let _ = writeln!(out, "| {} | skipped | - | - | - | - | {} |", r.id, r.reason);
            }
            continue;
        }
        let decode_s = r.decode_ms_total / 1000.0;
        let decode_tok_s = if decode_s > 0.0 { r.tokens_generated as f64 / decode_s } else { 0.0 };
        if show_lang {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.3} | {:.2} | {:.3} | {} | {} |",
                r.id,
                r.correct,
                r.steps,
                r.prefill_ms_total / 1000.0,
                decode_tok_s,
                r.total_ms / 1000.0,
                lang,
                r.reason,
            );
        } else {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.3} | {:.2} | {:.3} | {} |",
                r.id,
                r.correct,
                r.steps,
                r.prefill_ms_total / 1000.0,
                decode_tok_s,
                r.total_ms / 1000.0,
                r.reason,
            );
        }
    }
    let _ = writeln!(out);
    let scored = report.results.iter().filter(|r| !r.skipped).count();
    let skipped = report.results.len() - scored;
    let _ = writeln!(
        out,
        "**Summary**: correct {:.1}% ({} scored, {} skipped) — mean_steps {:.2}, mean_prefill_s {:.3}, mean_decode_tok_s {:.2}, mean_total_s {:.3}",
        report.correct_pct, scored, skipped, report.mean_steps, report.mean_prefill_s, report.mean_decode_tok_s, report.mean_total_s
    );

    out
}
