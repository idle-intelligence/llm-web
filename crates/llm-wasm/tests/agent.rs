//! Agent loop tests (phase 1a).

use llm_wasm::agent::{Agent, FixtureCaller, FixtureGenerator, Generator, StepOutcome, ToolCaller};
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

/// Like `FixtureGenerator`, but actually enforces the contract a real
/// KV-cache-backed `Generator` does around `prefix_len`: erroring if handed
/// a `prefix_len` covering the *whole* prompt (no new tokens to prefill),
/// exactly as `web.rs::generate_attempt`'s `cache.restore`/prefill would
/// before its fix. Used to give `agent.rs::generate_attempt`'s `prefix_len`
/// clamp (mirroring `web.rs`'s `effective_prefix` clamp) real regression
/// coverage — `FixtureGenerator`'s default `generate_with_cached_prefix`
/// ignores the hint entirely, so it can't catch this.
struct FullyCachedGuardGenerator {
    script: Vec<Vec<u32>>,
    next: usize,
}

impl FullyCachedGuardGenerator {
    fn new(script: Vec<Vec<u32>>) -> Self {
        Self { script, next: 0 }
    }
}

impl Generator for FullyCachedGuardGenerator {
    fn generate(&mut self, prompt_ids: &[u32], max_new_tokens: usize, stop_ids: &[u32]) -> anyhow::Result<Vec<u32>> {
        self.generate_with_cached_prefix(prompt_ids, 0, max_new_tokens, stop_ids)
    }

    fn generate_with_cached_prefix(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        _max_new_tokens: usize,
        _stop_ids: &[u32],
    ) -> anyhow::Result<Vec<u32>> {
        if prefix_len >= prompt_ids.len() {
            anyhow::bail!("prompt fully cached with no new tokens to prefill");
        }
        let out = self
            .script
            .get(self.next)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("FullyCachedGuardGenerator script exhausted after {} call(s)", self.next))?;
        self.next += 1;
        Ok(out)
    }
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

/// (c) Three invalid outputs in a row (more than `max_retries` allows),
/// none of them empty or a repeat, gives up with `StepOutcome::Error`
/// instead of looping forever — unlike an empty or repeated call (see
/// `repeats_exhausting_retries_force_a_text_only_answer`), there's no
/// evidence here the model has anything useful to say, so forcing a text
/// answer wouldn't help.
#[test]
fn three_invalid_outputs_give_up_with_error() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let unknown_turn = script_tokens(&tokenizer, r#"[{"name": "not_a_real_tool", "arguments": {}}]<|im_end|>"#);
    let generator = FixtureGenerator::new(vec![unknown_turn.clone(), unknown_turn.clone(), unknown_turn]);
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

/// `FixtureCaller::call` must return the same shape a real MCP `tools/call`
/// result has — `{"content":[{"type":"text","text":"<json>"}]}`, the
/// payload JSON-encoded as pretty-printed *text*, not returned as a parsed
/// value directly (that mismatch is what let the browser loop pass on
/// native fixture-driven evals while failing against the real server: ids
/// never got harvested from a value that was already parsed). This is the
/// documented `CallToolResult` shape (see `agent::FixtureCaller`'s doc
/// comment / `docs/ENGINE.md`); no live-run capture of the exact bytes was
/// available to diff against, so this asserts the documented shape.
#[test]
fn fixture_caller_result_is_mcp_content_shaped() {
    let mut caller = FixtureCaller::new(results_dir());
    let result = caller.call("get_households_and_groups_and_players", &serde_json::json!({})).unwrap();

    let content = result["content"].as_array().expect("result.content should be an array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let text = content[0]["text"].as_str().expect("content[0].text should be a string");

    // The text is the fixture's JSON, pretty-printed (real newlines +
    // 2-space indentation), not compacted onto one line.
    assert!(text.contains('\n'), "expected pretty-printed JSON text:\n{text}");
    assert!(text.contains("  \"households\""), "expected 2-space indentation:\n{text}");

    // And it round-trips back to exactly the canned fixture value.
    let fixture_path = results_dir().join("get_households_and_groups_and_players.json");
    let expected: Value = serde_json::from_str(&std::fs::read_to_string(fixture_path).unwrap()).unwrap();
    let embedded: Value = serde_json::from_str(text).unwrap();
    assert_eq!(embedded, expected);

    // No top-level `error` in this fixture, so no `isError`.
    assert!(result.get("isError").is_none());
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

/// Generic repeated-call loop guard (`docs/ENGINE.md` "Agent loop"): the
/// model calling the exact same tool with the exact same arguments as one
/// it already successfully called earlier this run is never re-executed —
/// instead the step retries with a nudge telling the model it already has
/// that information. If the very next attempt answers, the turn ends
/// `Final` with `repeat_guard` recording that a repeat was caught along the
/// way. Regression test for a live bug (owner's browser session,
/// 2026-09-11): `get_households_and_groups_and_players({})` called, then
/// called again for "List all my speakers.", which used to hard-error
/// (`Error: model is repeating a call`) instead of ever answering.
#[test]
fn repeated_call_is_not_reexecuted_and_nudges_to_an_answer() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let listing_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "You have the Kitchen and Living Room speakers.<|im_end|>");

    let generator = FixtureGenerator::new(vec![listing_turn.clone(), listing_turn, answer_turn]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("what's playing?", &tools_12());
    let StepOutcome::NeedTools { calls, step } = outcome else {
        panic!("expected first call to succeed, got {outcome:?}");
    };
    assert_eq!(step.retries, 0);
    assert!(!step.repeat_guard);
    assert_eq!(calls[0].name, "get_households_and_groups_and_players");

    let result: Value = serde_json::from_str(
        &std::fs::read_to_string(results_dir().join("get_households_and_groups_and_players.json")).unwrap(),
    )
    .unwrap();
    let outcome2 = agent.provide_tool_results(vec![(calls[0].call_id.clone(), result)]);
    match outcome2 {
        StepOutcome::Final { text, step } => {
            assert_eq!(text, "You have the Kitchen and Living Room speakers.");
            assert!(step.repeat_guard, "expected the repeat to have been caught and nudged past");
            assert!(!step.forced_text_answer, "the model answered on its own after the nudge, not forced");
        }
        other => panic!("expected Final after the nudge, got: {other:?}"),
    }
    // The repeated call was never handed back as a second `NeedTools` for
    // the harness to execute — `provide_tool_results` went straight from
    // the (already-answered) listing call to a `Final` answer.
}

/// When the malformed/repeated-call retry budget is fully exhausted on
/// nothing but repeats, `step_inner` forces a final answer by regenerating
/// under `Grammar::text_only` rather than giving up with `StepOutcome::Error`
/// — a read-only question must still end with an answer.
#[test]
fn repeats_exhausting_retries_force_a_text_only_answer() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let listing_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let forced_answer = script_tokens(&tokenizer, "You already have that information.<|im_end|>");

    // start() (1) + provide_tool_results()'s retry loop: attempt (2),
    // retry 1 (3), retry 2 (4, exhausts `max_retries`) — all identical
    // repeats — then the forced text-only generation (5).
    let generator = FixtureGenerator::new(vec![
        listing_turn.clone(),
        listing_turn.clone(),
        listing_turn.clone(),
        listing_turn,
        forced_answer,
    ]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("what's playing?", &tools_12());
    let StepOutcome::NeedTools { calls, .. } = outcome else {
        panic!("expected first call to succeed, got {outcome:?}");
    };

    let result: Value = serde_json::from_str(
        &std::fs::read_to_string(results_dir().join("get_households_and_groups_and_players.json")).unwrap(),
    )
    .unwrap();
    let outcome2 = agent.provide_tool_results(vec![(calls[0].call_id.clone(), result)]);
    match outcome2 {
        StepOutcome::Final { text, step } => {
            assert_eq!(text, "You already have that information.");
            assert!(step.repeat_guard);
            assert!(step.forced_text_answer);
        }
        other => panic!("expected Final via the forced text-only answer, got: {other:?}"),
    }
}

/// Regression test for the browser-gate bug (case s02 "List all my
/// speakers.", 2026-09-11, `Error: prompt fully cached with no new tokens
/// to prefill`, introduced by `07e439b`'s longest-common-prefix restore):
/// the repeat-guard's first retry re-renders the exact same prompt the
/// previous attempt just generated from, so the common prefix with what's
/// now resident (that prompt plus the repeat it produced) covers the
/// *whole* prompt. `generate_attempt` must clamp its `prefix_len` hint
/// back one token instead of handing a real `Generator` a hint with no new
/// tokens to prefill — verified here with `FullyCachedGuardGenerator`,
/// which errors exactly as the real KV-cache-backed one used to. The turn
/// still completes `Final`, exhausting retries into a forced text-only
/// answer, with a non-empty final text.
#[test]
fn fully_cached_repeat_guard_step_does_not_error() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let listing_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let forced_answer = script_tokens(&tokenizer, "You already have that information.<|im_end|>");

    let generator = FullyCachedGuardGenerator::new(vec![
        listing_turn.clone(),
        listing_turn.clone(),
        listing_turn.clone(),
        listing_turn,
        forced_answer,
    ]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("what's playing?", &tools_12());
    let StepOutcome::NeedTools { calls, .. } = outcome else {
        panic!("expected first call to succeed, got {outcome:?}");
    };

    let result: Value = serde_json::from_str(
        &std::fs::read_to_string(results_dir().join("get_households_and_groups_and_players.json")).unwrap(),
    )
    .unwrap();
    let outcome2 = agent.provide_tool_results(vec![(calls[0].call_id.clone(), result)]);
    match outcome2 {
        StepOutcome::Final { text, step } => {
            assert!(!text.is_empty());
            assert!(step.forced_text_answer || step.repeat_guard);
        }
        other => panic!("expected Final, got: {other:?}"),
    }
}

/// A repeat whose *earlier* result was an error is allowed to execute
/// again — retrying a failed call is legitimate, unlike retrying a call
/// that already succeeded.
#[test]
fn repeat_after_an_error_result_is_reexecuted() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let get_status = Tool::from_mcp(
        "get_status",
        "Read current status.",
        serde_json::json!({"type": "object", "properties": {}, "required": []}),
    );
    let tools = vec![get_status];

    let status_call = script_tokens(&tokenizer, r#"[{"name": "get_status", "arguments": {}}]<|im_end|>"#);

    let generator = FixtureGenerator::new(vec![status_call.clone(), status_call]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);

    let outcome = agent.start("what's the status?", &tools);
    let StepOutcome::NeedTools { calls, .. } = outcome else {
        panic!("expected first call to succeed, got {outcome:?}");
    };

    // Feed back an error result — the call is never added to
    // `successful_calls_this_turn`.
    let error_result = serde_json::json!({"error": "timeout"});
    let outcome2 = agent.provide_tool_results(vec![(calls[0].call_id.clone(), error_result)]);
    match outcome2 {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(calls[0].name, "get_status");
            assert_eq!(step.retries, 0, "the identical retry should execute immediately, not be nudged away");
            assert!(!step.repeat_guard, "a retry after an error is not the repeat guard's business");
        }
        other => panic!("expected the identical call to be allowed again after an error, got: {other:?}"),
    }
}

/// Fail-open (`docs/ENGINE.md` "Agent loop" — "fail-open"): when the only
/// tools still callable under the id-restricted grammar are read tools
/// already called this turn (the id rule left nothing new — e.g. a
/// listing tool's result carried no id-shaped values, and the only
/// mutating tool needs an id that's still unknown), the next step's
/// grammar is rebuilt via `Grammar::for_tools_unrestricted_ids` instead of
/// leaving the model boxed into repeating itself, and `Step.id_rule_relaxed`
/// records it. `FixtureGenerator` ignores the actual mask (see its doc
/// comment), so this only exercises the trigger condition itself, not
/// generation under the relaxed grammar.
#[test]
fn fail_open_relaxes_id_rule_when_only_option_is_a_repeat() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };

    let get_status = Tool::from_mcp(
        "get_status",
        "Read current status.",
        serde_json::json!({"type": "object", "properties": {}, "required": []}),
    );
    let pause = Tool::from_mcp(
        "pause",
        "Pause a group.",
        serde_json::json!({
            "type": "object",
            "properties": {"group_id": {"type": "string", "description": "The group ID."}},
            "required": ["group_id"],
        }),
    );
    let tools = vec![get_status, pause];

    let status_call = script_tokens(&tokenizer, r#"[{"name": "get_status", "arguments": {}}]<|im_end|>"#);
    // What the model would plausibly emit once the *relaxed* grammar makes
    // `pause` typable at all (`FixtureGenerator` ignores the actual mask —
    // see its doc comment — so this script stands in for "the model took
    // the newly-opened option" rather than proving the mask forced it).
    let pause_call = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "group-123"}}]<|im_end|>"#,
    );

    let generator = FixtureGenerator::new(vec![status_call, pause_call]);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(template, tokenizer, generator, caller, "sys", 64);
    agent.set_constrained(true);

    let outcome = agent.start("what's the status?", &tools);
    let StepOutcome::NeedTools { calls, step } = outcome else {
        panic!("expected first call to succeed, got {outcome:?}");
    };
    assert_eq!(calls[0].name, "get_status");
    assert!(!step.id_rule_relaxed, "nothing to relax yet on the first call");

    // Feed back a result with no id-shaped values at all — `pause` stays
    // uncallable under the id-restricted grammar afterward.
    let result = serde_json::json!({"status": "idle"});
    let outcome2 = agent.provide_tool_results(vec![(calls[0].call_id.clone(), result)]);
    match outcome2 {
        StepOutcome::NeedTools { calls, step } => {
            assert_eq!(calls[0].name, "pause");
            assert!(
                step.id_rule_relaxed,
                "expected fail-open to trigger: get_status is a read tool already called with no ids harvested"
            );
        }
        other => panic!("expected NeedTools, got: {other:?}"),
    }
}
