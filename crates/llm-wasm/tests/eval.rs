//! Eval harness tests (phase 3): drive `eval::run_case`/`run_all` with a
//! scripted `FixtureGenerator` over real `eval/utterances.json` cases and
//! check the scoring rules in `eval/README.md`.

use llm_wasm::agent::{Agent, FixtureCaller, FixtureGenerator};
use llm_wasm::eval::{self, EvalCase, ToolSet};
use llm_wasm::template::ChatTemplate;
use llm_wasm::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r")
        })
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sonos")
}

fn results_dir() -> PathBuf {
    fixtures_dir().join("results")
}

fn cases() -> Vec<EvalCase> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/utterances.json");
    eval::load_cases(path).expect("eval/utterances.json should load")
}

fn case(id: &str) -> EvalCase {
    cases().into_iter().find(|c| c.id == id).unwrap_or_else(|| panic!("no such case: {id}"))
}

/// Loads the real tokenizer + chat template, or prints a skip message and
/// returns `None` if the model files aren't available locally (mirrors
/// `tests/agent.rs`).
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

fn script_tokens(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).expect("encode should succeed")
}

fn new_agent(
    template: ChatTemplate,
    tokenizer: Tokenizer,
    script: Vec<Vec<u32>>,
) -> Agent<FixtureGenerator, FixtureCaller> {
    let generator = FixtureGenerator::new(script);
    let caller = FixtureCaller::new(results_dir());
    let mut agent = Agent::new(
        template,
        tokenizer,
        generator,
        caller,
        "You are a helpful home assistant with access to Sonos speaker controls.",
        64,
    );
    agent.set_max_steps(8);
    agent
}

#[test]
fn single_step_correct() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };
    let now_playing_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_now_playing", "arguments": {"group_id": "RINCON_LIVING01:2"}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "A jazz playlist is playing in the Living Room.<|im_end|>");
    let mut agent = new_agent(template, tokenizer, vec![now_playing_turn, answer_turn]);

    let tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let s01 = case("s01");
    let mut caller = FixtureCaller::new(results_dir());
    let result = eval::run_case(&mut agent, &tools, &s01, &mut caller);

    assert!(result.correct, "expected s01 to score correct: {}", result.reason);
    assert!(!result.skipped);
    assert_eq!(result.calls_made.len(), 1);
    assert_eq!(result.calls_made[0].name, "get_now_playing");
}

#[test]
fn multi_step_correct_pause_the_kitchen() {
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
    let mut agent = new_agent(template, tokenizer, vec![discover_turn, pause_turn, answer_turn]);

    let tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let m01 = case("m01");
    let mut caller = FixtureCaller::new(results_dir());
    let result = eval::run_case(&mut agent, &tools, &m01, &mut caller);

    assert!(result.correct, "expected m01 to score correct: {}", result.reason);
    assert_eq!(result.calls_made.len(), 2);
    assert_eq!(result.calls_made[1].name, "pause");
    assert_eq!(
        result.calls_made[1].arguments,
        serde_json::json!({"group_id": "RINCON_KITCHEN01:1"})
    );
}

#[test]
fn wrong_room_is_incorrect() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };
    let discover_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    // Model pauses the bedroom instead of the requested kitchen.
    let wrong_pause_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "pause", "arguments": {"group_id": "RINCON_BEDROOM01:3"}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "Paused the Bedroom.<|im_end|>");
    let mut agent = new_agent(template, tokenizer, vec![discover_turn, wrong_pause_turn, answer_turn]);

    let tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let m01 = case("m01");
    let mut caller = FixtureCaller::new(results_dir());
    let result = eval::run_case(&mut agent, &tools, &m01, &mut caller);

    assert!(!result.correct, "expected m01 with the wrong room to score incorrect");
    assert!(
        result.reason.contains("missing required call"),
        "expected reason to explain the missing required call, got: {}",
        result.reason
    );
}

#[test]
fn extra_read_under_subset_is_still_correct() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };
    // s01 ("what's playing right now?") is a read-only, `accept: subset`
    // case whose expected list is just `get_now_playing`. A model that
    // first re-runs discovery to resolve the room, then calls
    // `get_now_playing`, should still score correct — that extra read is
    // free under `subset` (eval/README.md).
    let discover_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let now_playing_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_now_playing", "arguments": {"group_id": "RINCON_LIVING01:2"}}]<|im_end|>"#,
    );
    let answer_turn = script_tokens(&tokenizer, "A jazz playlist is playing in the Living Room.<|im_end|>");
    let mut agent = new_agent(template, tokenizer, vec![discover_turn, now_playing_turn, answer_turn]);

    let tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let s01 = case("s01");
    let mut caller = FixtureCaller::new(results_dir());
    let result = eval::run_case(&mut agent, &tools, &s01, &mut caller);

    assert!(result.correct, "expected the extra discovery read to be free: {}", result.reason);
    assert_eq!(result.calls_made.len(), 2);
}

#[test]
fn tools12_case_is_skipped_under_the_12_tool_set() {
    let Some((template, tokenizer)) = load_agent_parts() else {
        return;
    };
    // m06 ("group the bedroom with the kitchen") has `tools12_ok: false` —
    // `add_players_to_group` isn't in the 12-tool subset, so under
    // `ToolSet::Twelve` it must be skipped, not scored, regardless of what
    // the model would have done. The generator is never asked to produce
    // anything for it (script is empty / unrelated).
    let discover_turn = script_tokens(
        &tokenizer,
        r#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]<|im_end|>"#,
    );
    let mut agent = new_agent(template, tokenizer, vec![discover_turn]);

    let all_tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let tools = eval::select_tools(&all_tools, ToolSet::Twelve, fixtures_dir()).unwrap();
    assert_eq!(tools.len(), 13);

    let report = eval::run_all(
        &mut agent,
        &tools,
        &[case("m06")],
        ToolSet::Twelve,
        results_dir(),
        "test-model",
        "2026-09-10",
    );

    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert!(result.skipped, "expected m06 to be skipped under the 12-tool set");
    assert!(!result.correct);
    assert_eq!(result.steps, 0);
    // A skipped case is excluded from the percentage entirely.
    assert_eq!(report.correct_pct, 0.0);
}

#[test]
fn render_markdown_contains_summary_and_one_row_per_case() {
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
    // Reused for every case in this small run; each case gets its own
    // FixtureCaller inside `run_all`, but the generator's script is shared
    // and short, so only run one live case plus one skipped case.
    let mut agent = new_agent(template, tokenizer, vec![discover_turn, pause_turn, answer_turn]);

    let all_tools = eval::load_all_tools(fixtures_dir()).unwrap();
    let tools = eval::select_tools(&all_tools, ToolSet::Twelve, fixtures_dir()).unwrap();

    let report = eval::run_all(
        &mut agent,
        &tools,
        &[case("m01"), case("m06")],
        ToolSet::Twelve,
        results_dir(),
        "test-model",
        "2026-09-10",
    );

    let markdown = eval::render_markdown(&report);
    assert!(markdown.contains("**Summary**"), "missing summary line:\n{markdown}");
    assert!(markdown.contains("| m01 "), "missing row for m01:\n{markdown}");
    assert!(markdown.contains("| m06 "), "missing row for m06:\n{markdown}");
    assert!(markdown.contains("skipped"), "expected m06's row to say skipped:\n{markdown}");
}
