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

/// Build a `tool`-role `Message` carrying a tool's result, matching how
/// `fixtures/reference/inputs/03_tools_multiturn.json` feeds results back:
/// `content` is a JSON-encoded *string* (not a nested object), formatted
/// the way Python's `json.dumps(result, ensure_ascii=False)` would (see
/// `template::to_json_compact_py`).
pub fn format_tool_result(name: &str, result: &Value) -> Message {
    let content = to_json_compact_py(result).unwrap_or_else(|_| result.to_string());
    Message {
        role: "tool".to_string(),
        content: Some(Value::String(content)),
        tool_calls: None,
        name: Some(name.to_string()),
        tool_call_id: None,
    }
}
