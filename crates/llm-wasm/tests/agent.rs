//! Agent loop tests (phase 1a).

use llm_wasm::agent::{Agent, FixtureCaller, FixtureGenerator};
use llm_wasm::template::{ChatTemplate, Tool};
use llm_wasm::tokenizer::Tokenizer;
use llm_wasm::tools::ParsedOutput;
use serde_json::Value;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r")
        })
}

fn results_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sonos/results")
}

fn tools_12() -> Vec<Tool> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sonos/tools-12.json");
    let raw: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    raw.into_iter()
        .map(|t| {
            Tool::from_mcp(
                t["name"].as_str().unwrap(),
                t["description"].as_str().unwrap(),
                t["inputSchema"].clone(),
            )
        })
        .collect()
}

/// Loads the real tokenizer + chat template, or prints a skip message and
/// returns `None` if the model files aren't available locally.
fn load_agent_parts() -> Option<(ChatTemplate, Tokenizer)> {
    let dir = model_dir();
    let cfg_path = dir.join("tokenizer_config.json");
    let tok_path = dir.join("tokenizer.json");
    if !cfg_path.exists() || !tok_path.exists() {
        println!("skipping: model files not found under {dir:?} (set LLM_MODEL_DIR)");
        return None;
    }
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let template = ChatTemplate::from_tokenizer_config(&cfg).unwrap();
    let tokenizer = Tokenizer::from_json(&std::fs::read(&tok_path).unwrap()).unwrap();
    Some((template, tokenizer))
}

/// Encode a model turn's raw text (as the model would emit it, i.e. ending
/// in `<|im_end|>`) into the token ids a `FixtureGenerator` should replay.
fn script_tokens(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).expect("encode should succeed")
}

#[test]
fn pause_the_kitchen_end_to_end() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let discover_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "Paused the Kitchen.<|im_end|>");

    let generator = FixtureGenerator::new(vec![discover_turn, pause_turn, answer_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(
        template,
        tokenizer,
        generator,
        caller,
        "You are a helpful home assistant with access to Sonos speaker controls.",
        64,
    );

    let transcript = agent
        .run("pause the kitchen", &tools_12(), 6)
        .expect("agent should finish within max_steps");

    assert_eq!(transcript.steps.len(), 3, "expected 3 steps (discover, pause, text)");
    assert_eq!(transcript.final_text, "Paused the Kitchen.");

    match &transcript.steps[0].parsed {
        ParsedOutput::ToolCalls(calls) => assert_eq!(calls[0].name, "get_households_and_groups_and_players"),
        other => panic!("expected tool call, got {other:?}"),
    }
    match &transcript.steps[1].parsed {
        ParsedOutput::ToolCalls(calls) => {
            assert_eq!(calls[0].name, "pause");
            assert_eq!(calls[0].arguments, serde_json::json!({"group_id": "RINCON_KITCHEN01:1"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    assert!(matches!(&transcript.steps[2].parsed, ParsedOutput::Text(t) if t == "Paused the Kitchen."));

    // Caller saw exactly these two calls, in order.
    let caller = agent.caller();
    assert_eq!(caller.calls_seen.len(), 2);
    assert_eq!(caller.calls_seen[0].0, "get_households_and_groups_and_players");
    assert_eq!(caller.calls_seen[1].0, "pause");
    assert_eq!(
        caller.calls_seen[1].1,
        serde_json::json!({"group_id": "RINCON_KITCHEN01:1"})
    );

    // The prompt re-rendered before generating step 2 (the `pause` call)
    // must already contain the first tool call's result text.
    let prompt_at_step_2 = agent
        .tokenizer()
        .decode(&transcript.steps[1].prompt_tokens, false)
        .unwrap();
    assert!(
        prompt_at_step_2.contains("RINCON_KITCHEN01"),
        "expected step 2's prompt to include the discover call's result:\n{prompt_at_step_2}"
    );
}

#[test]
fn max_steps_guard_errors_instead_of_looping_forever() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let discover_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    // Always returns a tool call, never a text answer.
    let generator = FixtureGenerator::new(vec![
        discover_turn.clone(),
        discover_turn.clone(),
        discover_turn,
    ]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(
        template,
        tokenizer,
        generator,
        caller,
        "sys",
        64,
    );

    let result = agent.run("pause the kitchen", &tools_12(), 2);
    assert!(result.is_err(), "expected max_steps guard to error");
}
