//! Tokenizer fidelity tests (phase 1a). Not in the task's literal test
//! file list (`tests/{template,tools,agent}.rs`) but `tokenizer.rs` is
//! explicitly owned and specced with its own fidelity requirement, so it
//! gets its own test file rather than being smuggled into `template.rs`'s.

use llm_wasm::tokenizer::Tokenizer;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("./models/hf/xLAM-2-3b-fc-r")
        })
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference")
}

fn load_tokenizer() -> Option<Tokenizer> {
    let path = model_dir().join("tokenizer.json");
    if !path.exists() {
        println!("skipping: no tokenizer.json at {path:?} (set LLM_MODEL_DIR)");
        return None;
    }
    Some(Tokenizer::from_json(&std::fs::read(&path).unwrap()).expect("tokenizer should load"))
}

#[test]
fn encode_matches_recorded_tokens_for_every_fixture() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };

    let rendered_dir = fixture_root().join("rendered");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&rendered_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no rendered fixtures found in {rendered_dir:?}");

    let mut checked = 0;
    for txt_path in cases {
        let name = txt_path.file_stem().unwrap().to_str().unwrap().to_string();
        let tokens_path = rendered_dir.join(format!("{name}.tokens.json"));
        if !tokens_path.exists() {
            continue;
        }

        let text = std::fs::read_to_string(&txt_path).unwrap();
        let expected: Vec<u32> =
            serde_json::from_str(&std::fs::read_to_string(&tokens_path).unwrap()).unwrap();

        let actual = tokenizer
            .encode(&text, false)
            .unwrap_or_else(|e| panic!("encode failed for {name}: {e}"));
        assert_eq!(actual, expected, "token id mismatch for fixture `{name}`");

        let decoded = tokenizer.decode(&actual, false).unwrap();
        assert_eq!(decoded, text, "decode round-trip mismatch for fixture `{name}`");

        checked += 1;
    }
    assert!(checked > 0, "no fixture had a matching .tokens.json");
    println!("checked {checked} fixture(s) for encode/decode fidelity");
}

/// Token-count proxy for the MCP-result-compaction fix (`tools.rs`'s
/// `compact_embedded_json`/`format_tool_result`): pretty-printing the
/// households/groups/players fixture the way an MCP server's
/// `json.dumps(result, indent=2)`-style output would costs real prompt
/// tokens over the compact form the fix now produces. Prints both counts
/// with `-- --nocapture`.
#[test]
fn compacting_households_fixture_reduces_token_count() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };

    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/sonos/results/get_households_and_groups_and_players.json");
    let raw = std::fs::read_to_string(&fixture_path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let pretty = serde_json::to_string_pretty(&value).unwrap();
    let compact = serde_json::to_string(&value).unwrap();

    let pretty_tokens = tokenizer.encode(&pretty, false).unwrap().len();
    let compact_tokens = tokenizer.encode(&compact, false).unwrap().len();

    println!(
        "households/groups/players fixture: pretty = {pretty_tokens} tokens, compact = {compact_tokens} tokens"
    );

    assert!(
        compact_tokens < pretty_tokens,
        "expected compact ({compact_tokens}) < pretty ({pretty_tokens})"
    );
}

#[test]
fn eos_ids_are_im_end_and_endoftext() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let mut ids = tokenizer.eos_ids().to_vec();
    ids.sort_unstable();
    assert_eq!(ids, vec![151643, 151645]);
}
