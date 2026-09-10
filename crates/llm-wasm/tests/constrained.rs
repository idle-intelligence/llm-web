//! GPU tests for schema-constrained decoding: `grammar.rs`'s `Constraint`/
//! `GrammarConstraint` driven through `model.rs::LlmModel::generate_with_constraint`
//! (see `docs/ENGINE.md` "Schema-constrained decoding").
//!
//! Env vars, skip behavior, and helper shapes mirror `tests/full_forward.rs`
//! (same model/reference directories, same "print + return instead of
//! failing when files are absent" convention). Run serially
//! (`--test-threads=1`) alongside `full_forward.rs` — both load the model
//! onto GPU.

#![cfg(feature = "wgpu")]

use std::path::Path;

use burn::backend::wgpu::WgpuDevice;
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::grammar::{tools_from_json, Constraint, Grammar, GrammarConstraint, IdValues, TokenVocab};
use llm_wasm::model::LlmModel;
use llm_wasm::tokenizer::Tokenizer;
use serde_json::Value;

fn model_dir() -> String {
    std::env::var("LLM_MODEL_DIR")
        .unwrap_or_else(|_| "/Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r".to_string())
}

fn tokenizer_model_dir() -> String {
    std::env::var("LLM_TOKENIZER_DIR")
        .unwrap_or_else(|_| "/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r".to_string())
}

fn fixtures_dir() -> String {
    format!("{}/../../fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn load_model(device: &WgpuDevice) -> Option<LlmModel> {
    let path = format!("{}/xLAM-2-3b-fc-r-q4_0.gguf", model_dir());
    if !Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let file = std::fs::File::open(&path).expect("open gguf");
    let reader = std::io::BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader).expect("parse gguf header");
    let parts = loader.load_deferred(device).expect("load_deferred");
    drop(loader); // free the GGUF reader/file before finalizing GPU tensors
    Some(parts.finalize(device).expect("finalize"))
}

fn load_tokenizer() -> Option<Tokenizer> {
    let path = format!("{}/tokenizer.json", tokenizer_model_dir());
    if !Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    Some(Tokenizer::from_json(&std::fs::read(&path).unwrap()).expect("tokenizer should load"))
}

fn load_tokens(name: &str) -> Option<Vec<u32>> {
    let path = format!("{}/reference/rendered/{name}.tokens.json", fixtures_dir());
    if !Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let data = std::fs::read_to_string(&path).expect("read tokens json");
    Some(serde_json::from_str(&data).expect("parse tokens json"))
}

#[derive(serde::Deserialize)]
struct RefLogitsSummary {
    greedy_first_32_token_ids: Vec<u32>,
}

fn load_ref_summary(name: &str) -> Option<RefLogitsSummary> {
    let path = format!("{}/reference/logits/{name}.json", fixtures_dir());
    if !Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let data = std::fs::read_to_string(&path).expect("read ref summary json");
    Some(serde_json::from_str(&data).expect("parse ref summary json"))
}

fn sonos_tools() -> Vec<llm_wasm::grammar::Tool> {
    let path = format!("{}/sonos/tools.json", fixtures_dir());
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

/// Runs one fixture's prompt through `generate_with_constraint`, asserting
/// the output matches the recorded unconstrained greedy reference exactly
/// (the grammar should never reject the model's own natural choice — these
/// fixtures were picked because their reference output is already
/// grammar-legal, see `tests/grammar.rs`'s test (a)) and reports the
/// model-step / forced-token breakdown.
fn run_constrained_case(fixture: &str, id_values: IdValues, max_ctx: usize) {
    let Some(tokens) = load_tokens(fixture) else {
        return;
    };
    let Some(summary) = load_ref_summary(fixture) else {
        return;
    };
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };

    let device = WgpuDevice::default();
    let Some(model) = load_model(&device) else {
        return;
    };

    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &id_values);
    let mut constraint = GrammarConstraint::new(&grammar, &tokenizer, &vocab);

    let mut cache = model.new_cache(max_ctx);
    let stop_ids = model.config().eos_token_ids.clone();

    let t0 = std::time::Instant::now();
    let (generated, stats) = model
        .generate_with_constraint(&tokens, 32, &stop_ids, &mut cache, Some(&mut constraint))
        .unwrap();
    let dt = t0.elapsed().as_secs_f32();

    let ref_greedy = &summary.greedy_first_32_token_ids;
    let n = generated.len().min(ref_greedy.len());
    eprintln!(
        "{fixture}: constrained={:?} ref_greedy(first {n})={:?}",
        generated,
        &ref_greedy[..n]
    );
    eprintln!(
        "{fixture}: model_steps={} forced_tokens={} total_tokens={} ({:.1}% forced) in {dt:.3}s",
        stats.model_steps,
        stats.forced_tokens,
        stats.total_tokens,
        100.0 * stats.forced_tokens as f32 / stats.total_tokens.max(1) as f32,
    );

    assert_eq!(
        &generated[..n],
        &ref_greedy[..n],
        "{fixture}: constrained decode diverged from the reference greedy output"
    );
    assert!(
        stats.forced_tokens * 2 >= stats.total_tokens,
        "{fixture}: expected forced tokens to be >= 50% of total ({}/{})",
        stats.forced_tokens,
        stats.total_tokens
    );
}

/// Fixture 02 (12-tool prompt, `get_households_and_groups_and_players`
/// takes no arguments): with no ids known, only the listing tool is
/// callable at all, so the generated call must be exactly that — same as
/// the unconstrained reference greedy output.
#[test]
fn constrained_02_tools_single_matches_reference_and_is_forced() {
    run_constrained_case("02_tools_single", IdValues::new(), 12288);
}

/// Fixture 03 (after the households result, the kitchen group id is
/// known): `pause` with that id, matching the unconstrained reference.
#[test]
fn constrained_03_tools_multiturn_matches_reference_and_is_forced() {
    run_constrained_case("03_tools_multiturn", ids(&["RINCON_KITCHEN01:1"]), 12288);
}

/// Unit test (no GPU/model — just the tokenizer): a genuinely fresh
/// `GrammarConstraint` has nothing forced yet — position 0 is the
/// free-text-vs-array-call decision point (`docs/ENGINE.md`: "decided by
/// the first non-whitespace byte: anything other than `[` switches the
/// whole output to unconstrained text"), so *every* non-`[` byte is also a
/// legal continuation there and `forced_run()` correctly returns `None`.
///
/// After `[{"` (the array/object/key-quote opener — every byte of which is
/// itself ambiguous against optional whitespace, `ArrWs`/`NameValueStart`,
/// so none of it is forced either, same as the leading `[`; this is why
/// `run_constrained_case`'s real fixtures see those bytes sampled
/// normally, not jump-forwarded), the tool-call key literal `name":` has
/// exactly one legal continuation byte at every position (up to the `"`
/// that would start the value, itself another whitespace-ambiguous byte)
/// — so it's fully forced in one call, independent of how the tokenizer
/// happens to chunk it into BPE pieces (byte-exact: decoding the forced
/// run's ids and joining them must equal that literal string exactly).
#[test]
fn forced_run_after_leading_bracket_covers_json_scaffolding() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let vocab = TokenVocab::from_tokenizer(&tokenizer);
    let tools = sonos_tools();
    let grammar = Grammar::for_tools(&tools, &ids(&["RINCON_KITCHEN01:1"]));
    let mut constraint = GrammarConstraint::new(&grammar, &tokenizer, &vocab);

    assert!(
        constraint.forced_run().is_none(),
        "a fresh state's first byte is the free-text-vs-array decision point, nothing should be forced yet"
    );

    let opener = tokenizer.encode("[{\"", false).expect("encode `[{\"`");
    for &t in &opener {
        constraint.advance(t);
    }

    let run = constraint.forced_run().expect("the `name\":` key literal after `[{\"` should be forced");
    let text = tokenizer.decode(&run, false).expect("decode forced run");
    assert_eq!(text, "name\":");
}
