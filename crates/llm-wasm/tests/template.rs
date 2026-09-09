//! Chat template tests (phase 1a).

use llm_wasm::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r")
        })
}

fn load_template() -> Option<ChatTemplate> {
    let dir = model_dir();
    let cfg_path = dir.join("tokenizer_config.json");
    if !cfg_path.exists() {
        println!(
            "skipping: no tokenizer_config.json at {cfg_path:?} (set LLM_MODEL_DIR)"
        );
        return None;
    }
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    Some(ChatTemplate::from_tokenizer_config(&cfg).expect("template should compile"))
}

#[derive(serde::Deserialize)]
struct ReferenceInput {
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference")
}

/// Byte-for-byte fidelity against every `fixtures/reference/inputs/*.json`.
#[test]
fn renders_byte_identical_to_reference_for_every_fixture() {
    let Some(template) = load_template() else {
        return;
    };

    let inputs_dir = fixture_root().join("inputs");
    let rendered_dir = fixture_root().join("rendered");

    let mut cases: Vec<PathBuf> = std::fs::read_dir(&inputs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no fixtures found in {inputs_dir:?}");

    let mut checked = 0;
    for input_path in cases {
        let name = input_path.file_stem().unwrap().to_str().unwrap().to_string();
        let expected_path = rendered_dir.join(format!("{name}.txt"));
        if !expected_path.exists() {
            continue;
        }

        let input: ReferenceInput =
            serde_json::from_str(&std::fs::read_to_string(&input_path).unwrap())
                .unwrap_or_else(|e| panic!("failed to parse {input_path:?}: {e}"));
        let expected = std::fs::read_to_string(&expected_path).unwrap();

        let actual = template
            .render_prompt(&input.messages, &input.tools, true)
            .unwrap_or_else(|e| panic!("render failed for {name}: {e}"));

        assert_eq!(actual, expected, "mismatch rendering fixture `{name}`");
        checked += 1;
    }
    assert!(checked > 0, "no fixture had a matching rendered/*.txt");
    println!("checked {checked} reference fixture(s) byte-for-byte");
}

/// The reference `03_tools_multiturn` fixture happens to carry
/// OpenAI-shaped `tool_calls[].function.arguments` as a JSON-*encoded
/// string* (`"{}"`) rather than a real object. The agent loop in
/// `agent.rs` instead appends `arguments` as a real JSON object (what the
/// model itself emits) when it replays a tool call into history — the
/// template must render that unquoted, not re-stringify it.
#[test]
fn tool_call_arguments_render_as_object_when_given_as_object() {
    let Some(template) = load_template() else {
        return;
    };

    let messages = vec![
        Message::system("You are a helpful home assistant with access to Sonos speaker controls."),
        Message::user("pause the kitchen"),
        Message::assistant_tool_calls(vec![ToolCallEntry {
            id: Some("call_0".to_string()),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "pause".to_string(),
                arguments: serde_json::json!({"group_id": "RINCON_KITCHEN01:1"}),
            },
        }]),
    ];

    let rendered = template
        .render_prompt(&messages, &[], false)
        .expect("render should succeed");

    assert!(
        rendered.contains(r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]"#),
        "expected unquoted object arguments in rendered output, got:\n{rendered}"
    );
    // And make sure it's *not* rendered as a JSON-encoded string.
    assert!(!rendered.contains(r#""arguments": "{\"group_id\""#));
}

#[test]
fn tool_call_arguments_render_as_string_when_given_as_string() {
    let Some(template) = load_template() else {
        return;
    };

    let messages = vec![
        Message::system("sys"),
        Message::user("hi"),
        Message::assistant_tool_calls(vec![ToolCallEntry {
            id: Some("call_0".to_string()),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: "get_households_and_groups_and_players".to_string(),
                arguments: Value::String("{}".to_string()),
            },
        }]),
    ];

    let rendered = template
        .render_prompt(&messages, &[], false)
        .expect("render should succeed");

    assert!(rendered.contains(r#"[{"name": "get_households_and_groups_and_players", "arguments": "{}"}]"#));
}

#[test]
fn empty_tools_list_renders_without_tool_preamble() {
    let Some(template) = load_template() else {
        return;
    };
    let messages = vec![
        Message::system("sys"),
        Message::user("hi"),
    ];
    let rendered = template.render_prompt(&messages, &[], true).unwrap();
    assert!(!rendered.contains("You have access to a set of tools"));
}
