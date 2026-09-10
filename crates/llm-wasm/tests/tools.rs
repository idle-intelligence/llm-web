//! MCP tool-call parsing tests (phase 1a).

use llm_wasm::tools::{format_tool_result, parse_output, ParsedOutput, ToolCall};
use serde_json::json;
use std::path::PathBuf;

fn fixture_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference")
}

/// The recorded greedy first-32-token strings from every logits fixture
/// (`greedy_first_32_tokens_text`) are real xLAM-2 outputs — parse each and
/// check it comes back as the expected tool call(s).
#[test]
fn parses_recorded_greedy_outputs() {
    let logits_dir = fixture_root().join("logits");
    let Ok(entries) = std::fs::read_dir(&logits_dir) else {
        println!("skipping: no {logits_dir:?}");
        return;
    };

    let mut checked = 0;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let Some(text) = data.get("greedy_first_32_tokens_text").and_then(|v| v.as_str()) else {
            continue;
        };

        let parsed = parse_output(text).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"));
        match parsed {
            ParsedOutput::ToolCalls(calls) => {
                assert!(!calls.is_empty(), "{path:?} parsed to an empty tool-call list");
            }
            ParsedOutput::Text(_) => panic!("{path:?}'s greedy text should have parsed as tool call(s): {text}"),
        }
        checked += 1;
    }
    assert!(checked > 0, "no logits fixtures found under {logits_dir:?}");
    println!("checked {checked} recorded greedy output(s)");
}

#[test]
fn parses_pause_call_with_arguments() {
    let raw = r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#;
    let parsed = parse_output(raw).unwrap();
    assert_eq!(
        parsed,
        ParsedOutput::ToolCalls(vec![ToolCall {
            name: "pause".to_string(),
            arguments: json!({"group_id": "RINCON_KITCHEN01:1"}),
        }])
    );
}

#[test]
fn parses_parallel_tool_calls() {
    let raw = r#"[{"name": "pause", "arguments": {"group_id": "a"}}, {"name": "get_now_playing", "arguments": {"group_id": "b"}}]"#;
    let parsed = parse_output(raw).unwrap();
    match parsed {
        ParsedOutput::ToolCalls(calls) => assert_eq!(calls.len(), 2),
        other => panic!("expected tool calls, got {other:?}"),
    }
}

#[test]
fn tolerates_leading_whitespace() {
    let raw = "   \n[{\"name\": \"resume\", \"arguments\": {}}]";
    let parsed = parse_output(raw).unwrap();
    assert!(matches!(parsed, ParsedOutput::ToolCalls(ref c) if c.len() == 1));
}

#[test]
fn plain_text_is_not_a_tool_call() {
    let raw = "Sure, turning on the kitchen speaker now.";
    let parsed = parse_output(raw).unwrap();
    assert_eq!(parsed, ParsedOutput::Text(raw.to_string()));
}

#[test]
fn truncated_array_is_an_error_not_a_panic() {
    let raw = r#"[{"name": "pause", "arguments": {"group_id": "RINCON_K"#;
    let err = parse_output(raw).expect_err("truncated JSON should error");
    let msg = err.to_string();
    assert!(msg.contains("not a valid tool-call array"), "unexpected error: {msg}");
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    let raw = "[this is not json]";
    assert!(parse_output(raw).is_err());
}

#[test]
fn format_tool_result_builds_tool_message() {
    let msg = format_tool_result(
        "get_households_and_groups_and_players",
        &json!({"households": []}),
    );
    assert_eq!(msg.role, "tool");
    assert_eq!(msg.name.as_deref(), Some("get_households_and_groups_and_players"));
    let content = msg.content.expect("content should be set");
    assert_eq!(content, serde_json::Value::String(r#"{"households": []}"#.to_string()));
}

/// The bug this fixes: an MCP tool result whose payload is pretty-printed
/// JSON text (real newlines + indentation) embedded as a *string* — e.g.
/// `content: [{"type": "text", "text": "<pretty json>"}]` — must come out
/// of `format_tool_result` with that embedded JSON compacted (no
/// newlines, no indentation), not carrying its whitespace through as
/// literal bytes inside the JSON string.
#[test]
fn compacts_pretty_printed_json_embedded_in_content_text() {
    let pretty_payload = "{\n  \"households\": [\n    {\n      \"householdId\": \"Sonos_ABC123XYZ\"\n    }\n  ]\n}";
    let result = json!({
        "content": [
            {"type": "text", "text": pretty_payload}
        ]
    });

    let msg = format_tool_result("get_households_and_groups_and_players", &result);
    let content = msg.content.expect("content should be set");
    let content_str = content.as_str().expect("content should be a JSON string");

    assert!(!content_str.contains('\n'), "embedded JSON should be compacted, got: {content_str}");
    assert!(
        !content_str.contains("    "),
        "embedded JSON should have no indentation left, got: {content_str}"
    );

    // Decoding the outer content string then the inner `text` string
    // should round-trip to the same JSON value the pretty payload encodes.
    let outer: serde_json::Value = serde_json::from_str(content_str).unwrap();
    let inner_text = outer["content"][0]["text"].as_str().unwrap();
    let inner: serde_json::Value = serde_json::from_str(inner_text).unwrap();
    assert_eq!(inner, serde_json::from_str::<serde_json::Value>(pretty_payload).unwrap());
}

/// A top-level string result (not wrapped in a `content` array) that
/// itself parses as pretty-printed JSON gets the same treatment.
#[test]
fn compacts_pretty_printed_json_top_level_string_result() {
    let pretty = "{\n  \"groupId\": \"RINCON_KITCHEN01:1\",\n  \"playbackState\": \"PAUSED\"\n}";
    let msg = format_tool_result("pause", &json!(pretty));
    let content = msg.content.unwrap();
    let content_str = content.as_str().unwrap();
    assert!(!content_str.contains('\n'));
    // content_str is itself a JSON-encoded string of a JSON-encoded string.
    let outer: String = serde_json::from_str(content_str).unwrap();
    assert!(!outer.contains('\n'), "inner string should also be compacted, got: {outer}");
}

/// Plain, non-JSON text (e.g. an error message or free text tool result)
/// is left completely verbatim — no attempt to "fix" it.
#[test]
fn non_json_text_is_left_verbatim() {
    let msg = format_tool_result("pause", &json!("Kitchen is now paused."));
    let content = msg.content.unwrap();
    assert_eq!(content, serde_json::Value::String("\"Kitchen is now paused.\"".to_string()));
}
