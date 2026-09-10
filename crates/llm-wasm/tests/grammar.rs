//! Tests for `grammar.rs` (schema-constrained decoding). Grammar-shape
//! tests ((b)-(f)) drive `GrammarState::feed_bytes` directly against
//! literal strings so they're independent of how the tokenizer happens to
//! chunk text into BPE tokens. Tests that need real vocab ids ((a), (g))
//! load the real tokenizer via `LLM_MODEL_DIR` and skip if it's absent.

use llm_wasm::grammar::{tools_from_json, Grammar, GrammarState, IdValues, TokenVocab};
use llm_wasm::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r")
        })
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_tokenizer() -> Option<Tokenizer> {
    let path = model_dir().join("tokenizer.json");
    if !path.exists() {
        println!("skipping: no tokenizer.json at {path:?} (set LLM_MODEL_DIR)");
        return None;
    }
    Some(Tokenizer::from_json(&std::fs::read(&path).unwrap()).expect("tokenizer should load"))
}

fn sonos_tools() -> Vec<llm_wasm::grammar::Tool> {
    let path = fixture_root().join("sonos/tools.json");
    let raw: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    tools_from_json(&raw)
}

fn ids(values: &[&str]) -> IdValues {
    let mut iv = IdValues::new();
    for v in values {
        iv.insert(*v);
    }
    iv
}

// (a) The two recorded greedy outputs are accepted token by token given
// the right id set.
#[test]
fn recorded_greedy_outputs_accepted_token_by_token() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();

    let cases: [(&str, IdValues); 2] = [
        (
            "02_tools_single.json",
            IdValues::new(), // get_households_and_groups_and_players takes no args
        ),
        (
            "03_tools_multiturn.json",
            ids(&["RINCON_KITCHEN01:1"]), // pause's group_id
        ),
    ];

    for (fname, id_values) in cases {
        let path = fixture_root().join("reference/logits").join(fname);
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let token_ids: Vec<u32> = rec["greedy_first_32_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let grammar = Grammar::for_tools(&tools, &id_values);
        let mut state = GrammarState::new(&grammar);

        for &tok in &token_ids {
            let mask = state.allowed(&vocab);
            assert!(
                mask.is_allowed(tok as usize),
                "{fname}: token {tok} rejected by grammar mask"
            );
            state.advance(tok, &vocab);
        }
        assert!(
            state.is_complete(),
            "{fname}: state not complete after recorded output"
        );
    }
}

// (b) `[{"name": "pause", "arguments": {}}]` is rejected at the point
// where `group_id` is missing — the `}` closing `arguments` is not
// allowed.
#[test]
fn pause_without_group_id_rejected_at_close() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &ids(&["RINCON_KITCHEN01:1"]));
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(br#"[{"name": "pause", "arguments": {"#));
    assert!(
        !state.feed_bytes(b"}"),
        "closing `arguments` with group_id missing should be rejected"
    );
}

// (c) With no known ids, `pause` cannot even be started, but
// `get_households_and_groups_and_players` can.
#[test]
fn uncallable_tool_excluded_from_name_choice() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &IdValues::new());
    assert!(!grammar.can_call("pause"));
    assert!(grammar.can_call("get_households_and_groups_and_players"));

    let mut state = GrammarState::new(&grammar);
    assert!(
        !state.feed_bytes(br#"[{"name": "pa"#),
        "\"pause\" should be unreachable with no known ids"
    );

    let mut state = GrammarState::new(&grammar);
    assert!(state.feed_bytes(br#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]"#));
}

// (d) `play_artist` without `shuffle` is rejected at `}`.
#[test]
fn play_artist_without_shuffle_rejected_at_close() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &ids(&["RINCON_KITCHEN01:1"]));
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(
        br#"[{"name": "play_artist", "arguments": {"group_id": "RINCON_KITCHEN01:1", "artist": "Radiohead""#
    ));
    assert!(
        !state.feed_bytes(b"}"),
        "closing `arguments` without required `shuffle` should be rejected"
    );
}

// (e) An enum violation on `music_service` is rejected.
#[test]
fn music_service_enum_violation_rejected() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &ids(&["RINCON_KITCHEN01:1"]));
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(
        br#"[{"name": "play_artist", "arguments": {"group_id": "RINCON_KITCHEN01:1", "artist": "Radiohead", "music_service": ""#
    ));
    assert!(
        !state.feed_bytes(b"Bogus"),
        "a music_service value outside the enum should be rejected"
    );
}

// (f) Free text is accepted.
#[test]
fn free_text_accepted() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &IdValues::new());
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(b"The kitchen is paused."));
    assert!(state.is_complete());
}

// (g) Mask time per step, printed.
#[test]
fn mask_time_per_step() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &ids(&["RINCON_KITCHEN01:1"]));
    let mut state = GrammarState::new(&grammar);
    assert!(state.feed_bytes(br#"[{"name": "pause", "arguments": {"#));

    // Warm up (page faults, branch predictor) before timing.
    let _ = state.allowed(&vocab);

    let n = 20;
    let start = Instant::now();
    for _ in 0..n {
        let mask = state.allowed(&vocab);
        assert!(mask.count() > 0);
    }
    let elapsed = start.elapsed();
    let per_step = elapsed / n;
    println!(
        "grammar mask: {:?}/step over {} vocab entries ({} steps averaged)",
        per_step,
        vocab.len(),
        n
    );
}
