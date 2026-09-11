//! Tests for `grammar.rs` (schema-constrained decoding). Grammar-shape
//! tests ((b)-(f)) drive `GrammarState::feed_bytes` directly against
//! literal strings so they're independent of how the tokenizer happens to
//! chunk text into BPE tokens. Tests that need real vocab ids ((a), (g))
//! load the real tokenizer via `LLM_MODEL_DIR` and skip if it's absent.

use llm_wasm::grammar::{tools_from_json, Constraint, Grammar, GrammarConstraint, GrammarState, IdValues, TokenVocab};
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

// `Grammar::text_only` (agent loops' forced-final-answer fallback,
// `docs/ENGINE.md` "Agent loop"): a leading `[` is rejected outright — the
// tool-call-array branch doesn't exist under this grammar — while prose is
// accepted exactly like `Grammar::for_tools`'s free-text branch.
#[test]
fn text_only_rejects_leading_array_bracket() {
    let grammar = Grammar::text_only();
    let mut state = GrammarState::new(&grammar);

    assert!(
        !state.feed_bytes(b"["),
        "text_only grammar should reject a leading '[' outright"
    );
}

#[test]
fn text_only_accepts_prose() {
    let grammar = Grammar::text_only();
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(b"I already checked and nothing has changed."));
    assert!(state.is_complete());
}

// `Grammar::tools_only` (both agent loops' `require_tool_call_first_step`,
// `docs/ENGINE.md` "Agent loop"): the free-text branch is rejected at
// `Pos::Start` — a leading non-`[` byte is invalid — while the tool-call
// array branch behaves exactly like `Grammar::for_tools`'s.
#[test]
fn tools_only_rejects_leading_prose() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &IdValues::new()).tools_only();
    let mut state = GrammarState::new(&grammar);

    assert!(
        !state.feed_bytes(b"I can't do that."),
        "tools_only grammar should reject a leading non-'[' byte outright"
    );
}

#[test]
fn tools_only_accepts_tool_call_array() {
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &IdValues::new()).tools_only();
    let mut state = GrammarState::new(&grammar);

    assert!(state.feed_bytes(br#"[{"name": "get_households_and_groups_and_players", "arguments": {}}]"#));
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

// (h) Session 12 fix: `forced_run` re-encodes whatever `forced_bytes`
// walked in isolation (`Tokenizer::encode` on just that substring), which
// can land on a token boundary the model's own left-to-right generation
// never produces mid-string (BPE merges aren't prefix-invariant). Forcing
// the three ids' shared `RINCON_` prefix that way is exactly what made
// constrained decoding pick `RINCON_KITCHEN01:1` over the intended
// `RINCON_LIVING01:2` in `constrained43`'s s08/m02/m09/s11 (see
// `docs/BENCHMARKS.md` session 12 addendum). With >1 live id, nothing past
// the opening quote should ever be forced — masked per-token decoding
// (test below) picks the id instead.
#[test]
fn forced_run_does_not_force_past_quote_for_multi_candidate_id() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(
        &tools,
        &ids(&["RINCON_KITCHEN01:1", "RINCON_LIVING01:2", "RINCON_BEDROOM01:3"]),
    );
    let mut constraint = GrammarConstraint::new(&grammar, &tokenizer, &vocab);

    let prefix = tokenizer
        .encode("[{\"name\": \"pause\", \"arguments\": {\"group_id\": \"", false)
        .expect("encode prefix up to the id value's opening quote");
    for &t in &prefix {
        constraint.advance(t);
    }

    assert!(
        constraint.forced_run().is_none(),
        "with 3 live ids sharing the `RINCON_` prefix, nothing should be forced right after the opening quote"
    );
}

// (i) Token healing: every live id must still be reachable one *natural*
// BPE token at a time via masked decoding alone (no forcing) — i.e. the
// mask admits, at every position along the tokenizer's own segmentation of
// the *full* call (not a re-encoding of an isolated shared-prefix
// substring), the token the model would actually produce.
#[test]
fn multi_candidate_id_reachable_token_by_token_for_every_alternative() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();
    let id_values = ids(&["RINCON_KITCHEN01:1", "RINCON_LIVING01:2", "RINCON_BEDROOM01:3"]);

    for candidate in ["RINCON_KITCHEN01:1", "RINCON_LIVING01:2", "RINCON_BEDROOM01:3"] {
        let grammar = Grammar::for_tools(&tools, &id_values);
        let mut state = GrammarState::new(&grammar);

        let text = format!("[{{\"name\": \"pause\", \"arguments\": {{\"group_id\": \"{candidate}\"}}}}]");
        let token_ids = tokenizer.encode(&text, false).expect("encode full call");

        for &tok in &token_ids {
            let mask = state.allowed(&vocab);
            assert!(
                mask.is_allowed(tok as usize),
                "{candidate}: token {tok} rejected mid-generation of its own natural tokenization"
            );
            state.advance(tok, &vocab);
        }
        assert!(
            state.is_complete(),
            "{candidate}: state should be complete after the full call"
        );
    }
}

/// Typed ids (`docs/ENGINE.md` "Typed ids" — the `bloupblip` regression,
/// owner's browser session 2026-09-11): a player id must never satisfy a
/// `group_id`-typed property, and vice versa, even though both are
/// `RINCON_`-shaped strings the old flat `IdValues` set couldn't tell
/// apart. Harvests real fixture data (households/groups/players) and
/// checks three things: (1) `get_now_playing`'s `group_id` accepts the
/// fixture's actual group id token-by-token and rejects the fixture's
/// player id (a byte-for-byte prefix of that group id) at the first
/// divergent byte; (2) `add_players_to_group`'s array-of-ids property
/// (`player_ids`) accepts a player id element and rejects a group id
/// element; (3) the untyped fallback bucket still works when a result
/// carries no id-shaped keys at all.
#[test]
fn typed_ids_keep_group_and_player_pools_separate() {
    let tools = sonos_tools();
    let fixture_path = fixture_root().join("sonos/results/get_households_and_groups_and_players.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();

    let mut id_values = IdValues::new();
    id_values.collect_from_result(&fixture);

    // (1) group_id: the real group id is accepted...
    let grammar = Grammar::for_tools(&tools, &id_values);
    let mut state = GrammarState::new(&grammar);
    assert!(state.feed_bytes(
        br#"[{"name": "get_now_playing", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]"#
    ));
    assert!(state.is_complete(), "the fixture's real group id should be accepted for group_id");

    // ...but the player id sharing its `RINCON_KITCHEN01` prefix is
    // rejected: the bytes up to and including that shared prefix are still
    // a valid partial match against the group-id candidates, so acceptance
    // holds there, but the very next byte — the closing quote, since the
    // player id ends where the group id still expects `:1` — has no live
    // candidate left and is rejected.
    let mut state = GrammarState::new(&grammar);
    assert!(
        state.feed_bytes(br#"[{"name": "get_now_playing", "arguments": {"group_id": "RINCON_KITCHEN01"#),
        "the shared prefix with the group id should still be a valid partial match"
    );
    assert!(
        !state.feed_bytes(b"\""),
        "closing the quote on the bare player id (not a real group id) should be rejected"
    );

    // (2) player_ids (array-of-ids): a real player id is accepted as an
    // element...
    let mut state = GrammarState::new(&grammar);
    assert!(state.feed_bytes(
        br#"[{"name": "add_players_to_group", "arguments": {"group_id": "RINCON_KITCHEN01:1", "player_ids": ["RINCON_KITCHEN01"]}}]"#
    ));
    assert!(state.is_complete());

    // ...but a group id is not, even though it's also an id-shaped string
    // known to `id_values` — it's in the wrong type bucket.
    let mut state = GrammarState::new(&grammar);
    assert!(!state.feed_bytes(
        br#"[{"name": "add_players_to_group", "arguments": {"group_id": "RINCON_KITCHEN01:1", "player_ids": ["RINCON_KITCHEN01:1"]}}]"#
    ));

    // (3) Untyped fallback: a result with no id-shaped keys at all still
    // populates the untyped ("") bucket via the generic `looks_like_id`
    // shape heuristic, and a property whose typed bucket is empty (no
    // `favorite`-typed value was ever harvested) still falls back to it —
    // `get_sonos_favorites`'s `favorite_id` isn't in `tools.json`, so use
    // `pause`'s `group_id` against a set of untyped-only ids instead.
    let mut untyped = IdValues::new();
    untyped.collect_from_result(&serde_json::json!({"lastKnownDevice": "RINCON_KITCHEN01:1"}));
    let grammar2 = Grammar::for_tools(&tools, &untyped);
    let mut state = GrammarState::new(&grammar2);
    assert!(state.feed_bytes(
        br#"[{"name": "pause", "arguments": {"group_id": "RINCON_KITCHEN01:1"}}]"#
    ));
    assert!(state.is_complete(), "the untyped fallback bucket should still constrain group_id");
}

/// Regression test for the real-MCP-shape bug (owner's browser session,
/// 2026-09-11): MCP tool results carry their payload as
/// `{"content":[{"type":"text","text":"<pretty-printed JSON string>"}]}`,
/// not as a bare JSON object/array — the households/groups/players listing
/// is a JSON string *inside* `content[0].text`, not a nested value. Native
/// evals never caught this because their fixtures hand the parsed JSON
/// directly (`fixtures/sonos/results/*.json`), bypassing the `content`/
/// `text` wrapper real MCP servers use. Before the fix, `IdValues` stayed
/// empty against this shape, which made every `*_id`-taking tool
/// (`pause`, `set_group_volume`, ...) uncallable per `Grammar::for_tools`'s
/// "drop a tool with no known id" rule — the model could only ever repeat
/// `get_households_and_groups_and_players`.
#[test]
fn collect_from_result_parses_ids_from_mcp_content_text_wrapper() {
    let mcp_result = serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&serde_json::json!([{
                "householdId": "Sonos_abc123XYZ.household",
                "groups": [{
                    "groupId": "RINCON_KITCHEN01:1",
                    "players": [{"playerId": "RINCON_KITCHEN01:0", "name": "Kitchen"}],
                }, {
                    "groupId": "RINCON_LIVING01:2",
                    "players": [{"playerId": "RINCON_LIVING01:0", "name": "Living Room"}],
                }],
            }]))
            .unwrap(),
        }],
    });

    let mut id_values = IdValues::new();
    id_values.collect_from_result(&mcp_result);

    let got = id_values.sorted_vec();
    for expected in [
        "Sonos_abc123XYZ.household",
        "RINCON_KITCHEN01:1",
        "RINCON_LIVING01:2",
        "RINCON_KITCHEN01:0",
        "RINCON_LIVING01:0",
    ] {
        assert!(
            got.iter().any(|s| s == expected),
            "expected {expected:?} to be harvested from the content/text-wrapped MCP result, got {got:?}"
        );
    }
}
