//! MCP tool-call schema (de)serialization and the xLAM tool-call output
//! parser. Owned by phase 1a. See `docs/MODELS.md` §3 ("Tool-call OUTPUT
//! format the model emits"): xLAM-2 emits a bare JSON array
//! `[{"name": ..., "arguments": {...}}, ...]` at the start of the turn,
//! with no `<tool_call>` wrapper tags — detection is "does the (stripped)
//! output start with `[`".

use crate::template::{to_json_compact_py, Message};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// One parsed tool call: `{"name": ..., "arguments": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// The result of parsing one turn of model output.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOutput {
    ToolCalls(Vec<ToolCall>),
    Text(String),
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("output starts with '[' but is not a valid tool-call array: {0}")]
    InvalidJson(String),
}

/// Parse one turn of raw (decoded) model output.
///
/// - Leading whitespace before the array is tolerated.
/// - A trailing `<|im_end|>` (and any whitespace before it) is stripped
///   before the `[`-prefix check, since the model emits the array
///   immediately followed by its turn-end token (see
///   `fixtures/reference/logits/*.json`'s `greedy_first_32_tokens_text`).
/// - Output starting with `[` that isn't a valid `Vec<ToolCall>` (truncated
///   generation, malformed JSON, …) is an error, not a panic.
/// - Anything else is plain-text content.
pub fn parse_output(raw: &str) -> Result<ParsedOutput, ParseError> {
    let trimmed = raw.trim_start();
    let without_end = trimmed
        .strip_suffix("<|im_end|>")
        .unwrap_or(trimmed)
        .trim_end();

    if without_end.starts_with('[') {
        serde_json::from_str::<Vec<ToolCall>>(without_end)
            .map(ParsedOutput::ToolCalls)
            .map_err(|e| ParseError::InvalidJson(e.to_string()))
    } else {
        Ok(ParsedOutput::Text(without_end.to_string()))
    }
}

/// Above this many characters, a compacted tool result is logged (not
/// truncated — truncating a result silently could hide data the model
/// needs) — see `format_tool_result`.
pub const MAX_RESULT_CHARS: usize = 8000;

/// If `value` is a string that itself parses as JSON, replace it with a
/// compact re-serialization of that parsed JSON (still a string); anything
/// else (non-JSON text, numbers, bools, null) is left verbatim. Recurses
/// into arrays/objects, so this also compacts JSON-typed text nested
/// inside a result at any depth — e.g. an MCP `content: [{"type": "text",
/// "text": "<json>"}]` item's `text` field.
///
/// This is what fixes the actual bug: many MCP servers (Sonos's included)
/// return a tool result whose payload is pretty-printed JSON text (real
/// newlines + indentation) embedded as a *string*, not a nested JSON
/// value. `to_json_compact_py` alone can't touch that — serializing a
/// `Value::String` just JSON-escapes its bytes (`\n`, literal spaces and
/// all), so the whitespace survives as extra prompt tokens. Re-parsing and
/// re-compacting the string's contents here removes it before the result
/// ever reaches `to_json_compact_py`.
fn compact_embedded_json(value: &Value) -> Value {
    match value {
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => {
                let compacted = compact_embedded_json(&parsed);
                Value::String(to_json_compact_py(&compacted).unwrap_or_else(|_| s.clone()))
            }
            Err(_) => value.clone(),
        },
        Value::Array(items) => Value::Array(items.iter().map(compact_embedded_json).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), compact_embedded_json(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Build a `tool`-role `Message` carrying a tool's result, matching how
/// `fixtures/reference/inputs/03_tools_multiturn.json` feeds results back:
/// `content` is a JSON-encoded *string* (not a nested object), formatted
/// the way Python's `json.dumps(result, ensure_ascii=False)` would (see
/// `template::to_json_compact_py`) — after first compacting any
/// pretty-printed JSON embedded in `result` (see `compact_embedded_json`).
///
/// This is the one place both agent loops (`agent.rs`'s step-wise/`run`
/// API and `web.rs`'s `provide_tool_results`) build the `tool` message
/// from a raw `ToolCaller`/MCP result, so it's also the agent loop's
/// `MAX_RESULT_CHARS` safety net: a compacted result over that size is
/// kept in full (never silently truncated) but logged as a warning.
pub fn format_tool_result(name: &str, result: &Value) -> Message {
    let compacted = compact_embedded_json(result);
    let content = to_json_compact_py(&compacted).unwrap_or_else(|_| result.to_string());
    if content.len() > MAX_RESULT_CHARS {
        tracing::warn!(
            tool = name,
            chars = content.len(),
            max = MAX_RESULT_CHARS,
            "tool result exceeds max_result_chars after compaction; keeping full result uncut"
        );
    }
    Message {
        role: "tool".to_string(),
        content: Some(Value::String(content)),
        tool_calls: None,
        name: Some(name.to_string()),
        tool_call_id: None,
    }
}
