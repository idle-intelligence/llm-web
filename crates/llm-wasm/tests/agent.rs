//! Agent loop tests (phase 1a).

use llm_wasm::agent::{Agent, FixtureCaller, FixtureGenerator, StepOutcome};
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

/// Same "pause the kitchen" transcript as `pause_the_kitchen_end_to_end`,
/// but driven through the step-wise `start`/`provide_tool_results` API the
/// browser uses (tool results supplied out-of-band, not via a synchronous
/// `ToolCaller`).
#[test]
fn step_wise_pause_the_kitchen() {
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
    // No calls will actually be routed through this synchronous caller in
    // the step-wise path — the fixture results are read directly below —
    // but `Agent` still requires a `ToolCaller` type parameter.
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(
        template,
        tokenizer,
        generator,
        caller,
        "You are a helpful home assistant with access to Sonos speaker controls.",
        64,
    );

    let load_fixture = |name: &str| -> Value {
        let path = results_dir().join(format!("{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };

    let outcome = agent.start("pause the kitchen", &tools_12());
    let calls = match outcome {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_households_and_groups_and_players");
            assert!(matches!(step.parsed, ParsedOutput::ToolCalls(_)));
            calls
        }
        _ => panic!("expected NeedTools for step 1"),
    };
    let results = vec![(calls[0].call_id.clone(), load_fixture(&calls[0].name))];

    let outcome = agent.provide_tool_results(results);
    let calls = match outcome {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "pause");
            assert_eq!(calls[0].arguments, serde_json::json!({"group_id": "RINCON_KITCHEN01:1"}));
            assert!(
                step.prompt_tokens.len()
                    > agent
                        .tokenizer()
                        .encode("pause the kitchen", false)
                        .unwrap()
                        .len(),
                "step 2's prompt should include step 1's tool result"
            );
            calls
        }
        _ => panic!("expected NeedTools for step 2"),
    };
    let results = vec![(calls[0].call_id.clone(), load_fixture(&calls[0].name))];

    let outcome = agent.provide_tool_results(results);
    match outcome {
        StepOutcome::Final { text, .. } => assert_eq!(text, "Paused the Kitchen."),
        StepOutcome::NeedTools { .. } => panic!("expected Final for step 3"),
        StepOutcome::Error { message, .. } => panic!("unexpected error: {message}"),
    }
}

/// The rendered prompt's prefix (system + all 12 tool schemas) is identical
/// across two different utterances under the same tools set — the
/// assumption prefix caching relies on. Also reports the prefix's token
/// count for the 12-tool Sonos prompt.
#[test]
fn prefix_is_stable_across_utterances_for_same_tools() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };
    let tools = tools_12();
    let system = "You are a helpful home assistant with access to Sonos speaker controls.";

    let render = |utterance: &str| -> Vec<u32> {
        let messages = vec![
            llm_wasm::template::Message::system(system),
            llm_wasm::template::Message::user(utterance),
        ];
        let prompt = template.render_prompt(&messages, &tools, true).unwrap();
        tokenizer.encode(&prompt, false).unwrap()
    };

    let a = render("pause the kitchen");
    let b = render("what's playing in the living room");

    let common = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    println!("12-tool prefix common token length: {common} (a={}, b={})", a.len(), b.len());

    // The two utterances diverge immediately after the constant
    // system+tools preamble, so the common prefix should cover the large
    // majority of both prompts, not just a handful of tokens.
    assert!(common > 100, "expected a substantial shared prefix, got {common} tokens");
    assert!(common < a.len() && common < b.len());
}

// --- Retry / tool-error / schema-diet policy (see `docs/ENGINE.md` "Agent loop") ---

/// (a) An empty tool-call array `[]` followed by a valid call: 1 retry
/// recorded on step 1, 2 steps total, and the tool actually gets called
/// (no bogus assistant turn was appended for the `[]`).
#[test]
fn retries_once_on_empty_array_then_calls_tool() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let empty_turn = script_tokens(&tokenizer, "[]<|im_end|>");
    let pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "Paused.<|im_end|>");

    let generator = FixtureGenerator::new(vec![empty_turn, pause_turn, answer_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let transcript = agent
        .run("pause the kitchen", &tools_12(), 6)
        .expect("agent should recover from the empty array via retry");

    assert_eq!(transcript.steps.len(), 2, "retry should not count as its own step");
    assert_eq!(transcript.steps[0].retries, 1);
    match &transcript.steps[0].parsed {
        ParsedOutput::ToolCalls(calls) => assert_eq!(calls[0].name, "pause"),
        other => panic!("expected tool call after retry, got {other:?}"),
    }
    assert_eq!(agent.caller().calls_seen.len(), 1);
    assert_eq!(agent.caller().calls_seen[0].0, "pause");
}

/// (b) Two consecutive `[]` outputs, then a note-retry that produces a
/// valid call: 2 retries recorded on the one step.
#[test]
fn retries_twice_with_note_then_calls_tool() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let empty_turn = script_tokens(&tokenizer, "[]<|im_end|>");
    let pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );

    let generator = FixtureGenerator::new(vec![empty_turn.clone(), empty_turn, pause_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("pause the kitchen", &tools_12());
    match outcome {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(step.retries, 2);
            assert_eq!(calls[0].name, "pause");
        }
        StepOutcome::Final { .. } => panic!("expected NeedTools"),
        StepOutcome::Error { message, .. } => panic!("unexpected error: {message}"),
    }
}

/// (c) Three invalid outputs in a row (more than `max_retries` allows)
/// gives up with `StepOutcome::Error` instead of looping forever.
#[test]
fn three_invalid_outputs_give_up_with_error() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let empty_turn = script_tokens(&tokenizer, "[]<|im_end|>");
    let generator = FixtureGenerator::new(vec![empty_turn.clone(), empty_turn.clone(), empty_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("pause the kitchen", &tools_12());
    match outcome {
        StepOutcome::Error { message, step } => {
            assert!(step.is_none());
            assert!(message.contains("no valid call"), "unexpected message: {message}");
        }
        other => panic!("expected Error after exhausting retries, got: {other:?}"),
    }
}

/// (d) A tool-error result is fed back as `{"error": ...}` and the model's
/// next call is still accepted (one consecutive error is well under the
/// give-up threshold).
#[test]
fn tool_error_result_is_fed_back_and_next_call_accepted() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );
    let resume_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "resume", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );

    let generator = FixtureGenerator::new(vec![pause_turn, resume_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("pause the kitchen", &tools_12());
    let call_id = match outcome {
        StepOutcome::NeedTools { calls, .. } => calls[0].call_id.clone(),
        other => panic!("expected NeedTools for step 1, got {other:?}"),
    };

    let outcome = agent.provide_tool_results(vec![(call_id, serde_json::json!({"error": "group not found"}))]);
    match outcome {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(calls[0].name, "resume");
            assert_eq!(step.tool_errors, 1);
            let prompt = agent.tokenizer().decode(&step.prompt_tokens, false).unwrap();
            assert!(prompt.contains("{\"error\":"), "expected error feedback in prompt:\n{prompt}");
        }
        other => panic!("expected NeedTools (model's next call accepted), got {other:?}"),
    }
}

/// (e) A call naming a tool outside the current `tools` set counts as a
/// retry, not an accepted call.
#[test]
fn unknown_tool_name_retries_instead_of_calling() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let unknown_turn = script_tokens(&tokenizer, r#"[{"name": "not_a_real_tool", "arguments": {}}]<|im_end|>"#);
    let pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]<|im_end|>"#,
    );

    let generator = FixtureGenerator::new(vec![unknown_turn, pause_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("pause the kitchen", &tools_12());
    match outcome {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(step.retries, 1);
            assert_eq!(calls[0].name, "pause");
        }
        other => panic!("expected NeedTools after retrying past the unknown tool, got {other:?}"),
    }
}

/// (f) With the schema diet on (the default), a raw MCP tool carrying an
/// `annotations` field renders into a prompt that no longer contains it.
#[test]
fn diet_on_strips_annotations_from_rendered_prompt() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let raw_tools = vec![serde_json::json!({
        "name": "pause",
        "description": "Pause playback on a group.",
        "inputSchema": {
            "type": "object",
            "properties": {"group_id": {"type": "string"}},
            "required": ["group_id"],
        },
        "annotations": {"title": "Pause", "readOnlyHint": false},
    })];

    let answer_turn = script_tokens(&tokenizer, "OK.<|im_end|>");
    let generator = FixtureGenerator::new(vec![answer_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start_from_mcp("pause the kitchen", &raw_tools);
    let step = match outcome {
        StepOutcome::Final { step, .. } => step,
        other => panic!("expected Final, got {other:?}"),
    };
    let prompt = agent.tokenizer().decode(&step.prompt_tokens, false).unwrap();
    assert!(!prompt.contains("annotations"), "diet should have stripped annotations:\n{prompt}");
}
