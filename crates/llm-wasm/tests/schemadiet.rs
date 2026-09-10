//! Tool-schema token diet tests: token counts at levels 0/1/2 against the
//! real tokenizer + chat template, property-level schema equivalence at
//! level 1, and grammar-parse equivalence at level 1.

use llm_wasm::grammar;
use llm_wasm::schemadiet::{diet_tools, DietLevel};
use llm_wasm::template::{ChatTemplate, Message, Tool};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
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

fn load_tools(name: &str) -> Vec<Value> {
    let path = fixtures_dir().join(name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

fn as_tools(list: &[Value]) -> Vec<Tool> {
    list.iter()
        .map(|t| {
            Tool::from_mcp(
                t.get("name").and_then(Value::as_str).unwrap_or_default(),
                t.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                t.get("inputSchema").cloned().unwrap_or(Value::Null),
            )
        })
        .collect()
}

fn load_template() -> Option<ChatTemplate> {
    let cfg_path = model_dir().join("tokenizer_config.json");
    if !cfg_path.exists() {
        println!("skipping: no tokenizer_config.json at {cfg_path:?} (set LLM_MODEL_DIR)");
        return None;
    }
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    Some(ChatTemplate::from_tokenizer_config(&cfg).expect("template should compile"))
}

fn load_tokenizer() -> Option<llm_wasm::tokenizer::Tokenizer> {
    let path = model_dir().join("tokenizer.json");
    if !path.exists() {
        println!("skipping: no tokenizer.json at {path:?} (set LLM_MODEL_DIR)");
        return None;
    }
    Some(
        llm_wasm::tokenizer::Tokenizer::from_json(&std::fs::read(&path).unwrap())
            .expect("tokenizer should load"),
    )
}

/// Render the prefix the way `template.rs` does for a system + tools turn
/// (system message, tools, `add_generation_prompt: true`), matching how
/// `agent.rs`/`web.rs` build the first prompt of a conversation.
fn render_prefix(tmpl: &ChatTemplate, tools: &[Tool]) -> String {
    let messages = vec![Message::system(
        "You are a helpful assistant with access to tools.",
    )];
    tmpl.render_prompt(&messages, tools, true)
        .expect("render should succeed")
}

fn token_count(tokenizer: &llm_wasm::tokenizer::Tokenizer, text: &str) -> usize {
    tokenizer.encode(text, false).unwrap().len()
}

/// Token counts at levels 0/1/2 for both fixtures, printed with
/// `-- --nocapture`; asserts level 1 strictly reduces tokens vs level 0.
#[test]
fn level1_reduces_token_count_for_both_fixtures() {
    let Some(tmpl) = load_template() else {
        return;
    };
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };

    for (label, file) in [("13-tool", "tools-12.json"), ("34-tool", "tools.json")] {
        let raw = load_tools(file);

        let l0 = as_tools(&diet_tools(&raw, DietLevel::Level0));
        let l1 = as_tools(&diet_tools(&raw, DietLevel::Level1));
        let l2 = as_tools(&diet_tools(&raw, DietLevel::Level2));

        let t0 = token_count(&tokenizer, &render_prefix(&tmpl, &l0));
        let t1 = token_count(&tokenizer, &render_prefix(&tmpl, &l1));
        let t2 = token_count(&tokenizer, &render_prefix(&tmpl, &l2));

        println!(
            "{label} ({} tools): level0={t0} level1={t1} level2={t2} tokens \
             (level1 saves {}, level2 saves {} more)",
            raw.len(),
            t0.saturating_sub(t1),
            t1.saturating_sub(t2)
        );

        assert!(
            t1 < t0,
            "{label}: expected level1 ({t1}) < level0 ({t0})"
        );
        assert!(
            t2 <= t1,
            "{label}: expected level2 ({t2}) <= level1 ({t1})"
        );
    }
}

/// Per-property signature: (name, required?, enum values if any, JSON
/// Schema `type`). This is the semantic surface a valid tool call is
/// checked against — level 1 must not change it for a single property.
fn property_signatures(schema: &Value) -> BTreeSet<(String, bool, Vec<String>, String)> {
    let required: BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|props| {
            props
                .iter()
                .map(|(name, p)| {
                    let ty = p
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("string")
                        .to_string();
                    let mut enum_values: Vec<String> = p
                        .get("enum")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    enum_values.sort();
                    (name.clone(), required.contains(name), enum_values, ty)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// For every tool in both fixtures: the set of (property names, required,
/// enums, types) is identical before/after level 1 dieting.
#[test]
fn level1_preserves_property_signatures() {
    for file in ["tools-12.json", "tools.json"] {
        let raw = load_tools(file);
        let dieted = diet_tools(&raw, DietLevel::Level1);
        assert_eq!(raw.len(), dieted.len(), "{file}: tool count changed");

        for (before, after) in raw.iter().zip(dieted.iter()) {
            let name = before.get("name").and_then(Value::as_str).unwrap_or("?");
            let sig_before = property_signatures(before.get("inputSchema").unwrap());
            let sig_after = property_signatures(after.get("inputSchema").unwrap());
            assert_eq!(
                sig_before, sig_after,
                "{file}: tool `{name}` property signature changed at level 1"
            );
        }
    }
}

/// `grammar::tools_from_json` parses the dieted (level 1) tool list into
/// the same tools, same required flags, and same enums as the raw list —
/// read-only use of `grammar.rs`, owned by another worker.
#[test]
fn level1_dieted_tools_parse_identically_in_grammar() {
    for file in ["tools-12.json", "tools.json"] {
        let raw = load_tools(file);
        let dieted = diet_tools(&raw, DietLevel::Level1);

        let raw_tools = grammar::tools_from_json(&raw);
        let dieted_tools = grammar::tools_from_json(&dieted);

        assert_eq!(raw_tools.len(), dieted_tools.len(), "{file}: tool count");

        for (rt, dt) in raw_tools.iter().zip(dieted_tools.iter()) {
            assert_eq!(rt.name, dt.name, "{file}: tool name mismatch");

            let rprops: BTreeSet<String> = rt.properties.iter().map(|p| p.name.clone()).collect();
            let dprops: BTreeSet<String> = dt.properties.iter().map(|p| p.name.clone()).collect();
            assert_eq!(rprops, dprops, "{file}: `{}` property names", rt.name);

            for rp in &rt.properties {
                let dp = dt
                    .properties
                    .iter()
                    .find(|p| p.name == rp.name)
                    .unwrap_or_else(|| panic!("{file}: `{}.{}` missing after diet", rt.name, rp.name));
                assert_eq!(
                    rp.required, dp.required,
                    "{file}: `{}.{}` required mismatch",
                    rt.name, rp.name
                );

                let renum = enum_of(&rp.kind);
                let denum = enum_of(&dp.kind);
                assert_eq!(
                    renum, denum,
                    "{file}: `{}.{}` enum mismatch",
                    rt.name, rp.name
                );
            }
        }
    }
}

fn enum_of(kind: &grammar::PropKind) -> Option<Vec<String>> {
    match kind {
        grammar::PropKind::String { enum_values, .. } => {
            enum_values.as_ref().map(|v| {
                let mut v = v.clone();
                v.sort();
                v
            })
        }
        _ => None,
    }
}

/// Level 2's cross-tool sentence dedup measurably saves further tokens on
/// the 34-tool fixture, which repeats several sentences verbatim across
/// tools (e.g. "Group volume and player volume are linked...").
#[test]
fn level2_dedup_saves_tokens_on_34_tool_fixture() {
    let Some(tokenizer) = load_tokenizer() else {
        return;
    };
    let raw = load_tools("tools.json");
    let l1 = diet_tools(&raw, DietLevel::Level1);
    let l2 = diet_tools(&raw, DietLevel::Level2);

    let desc_len = |list: &[Value]| -> usize {
        list.iter()
            .filter_map(|t| t.get("description").and_then(Value::as_str))
            .map(|d| token_count(&tokenizer, d))
            .sum()
    };

    let t1 = desc_len(&l1);
    let t2 = desc_len(&l2);
    println!("34-tool description tokens: level1={t1} level2={t2}");
    assert!(t2 < t1, "expected level2 description tokens ({t2}) < level1 ({t1})");
}

/// Level 2 dedup only ever drops whole sentences it has seen before in an
/// earlier tool — it never introduces text, and each surviving sentence
/// was present verbatim (mod whitespace) in the original description.
#[test]
fn level2_dedup_only_drops_seen_sentences() {
    let raw = load_tools("tools.json");
    let l2 = diet_tools(&raw, DietLevel::Level2);
    let mut seen: HashSet<String> = HashSet::new();

    for (before, after) in raw.iter().zip(l2.iter()) {
        let name = before.get("name").and_then(Value::as_str).unwrap_or("?");
        let before_collapsed = before
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let after_desc = after.get("description").and_then(Value::as_str).unwrap_or("");

        for sentence in after_desc.split_inclusive(['.', '!', '?']) {
            let sentence = sentence.trim();
            // Skip re-split artifacts from this test's own naive
            // splitter on abbreviations (e.g. "e.g." re-splits into "e."
            // and "g."), which `diet_description`'s abbreviation-aware
            // splitter never treats as sentence boundaries in the first
            // place — see `schemadiet::split_sentences`.
            let is_abbreviation_fragment = sentence.len() <= 2
                && sentence.ends_with('.')
                && sentence[..sentence.len() - 1].chars().all(|c| c.is_lowercase());
            if sentence.is_empty() || is_abbreviation_fragment {
                continue;
            }
            assert!(
                before_collapsed.contains(sentence),
                "`{name}`: level2 sentence not present in the original description: {sentence:?}"
            );
            assert!(
                seen.insert(sentence.to_string()),
                "`{name}`: level2 kept a sentence already seen in an earlier tool: {sentence:?}"
            );
        }
    }
}
