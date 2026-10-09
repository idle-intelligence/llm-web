//! `render` must give, byte for byte, the text transformers'
//! `apply_chat_template(messages, tools=tools, add_generation_prompt=...,
//! tokenize=False)` gives. The fixtures hold each model's chat template and
//! transformers' output for a set of conversations: plain and multi-turn
//! chat, a system message, non-ASCII and HTML-like text, tools, tool calls
//! (object and string arguments, parallel calls) and tool results. They are
//! written by `scripts/gen_chat_fixtures.py`.

use lean_chat::{render, Message};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Message>,
    tools: Option<Vec<Value>>,
    add_generation_prompt: bool,
    rendered: String,
}

#[derive(Deserialize)]
struct Fixture {
    model: String,
    template: String,
    cases: Vec<Case>,
}

fn check(file: &str) {
    let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(!fixture.cases.is_empty());
    for case in &fixture.cases {
        let ours = render(&fixture.template, &case.messages, case.tools.as_deref(), case.add_generation_prompt).unwrap_or_else(|e| panic!("[{} {}] render failed: {e:#}", fixture.model, case.name));
        if ours != case.rendered {
            let at = ours.bytes().zip(case.rendered.bytes()).take_while(|(a, b)| a == b).count();
            let from = at.saturating_sub(40);
            panic!(
                "[{} {}] differs from transformers at byte {at}\n ours: {:?}\n   hf: {:?}",
                fixture.model,
                case.name,
                String::from_utf8_lossy(&ours.as_bytes()[from..(at + 40).min(ours.len())]),
                String::from_utf8_lossy(&case.rendered.as_bytes()[from..(at + 40).min(case.rendered.len())])
            );
        }
    }
    println!("{}: {} cases byte-equal", fixture.model, fixture.cases.len());
}

#[test]
fn qwen2_5_matches_transformers() {
    check("qwen2_5.json");
}

#[test]
fn qwen3_matches_transformers() {
    check("qwen3.json");
}

#[test]
fn smollm2_matches_transformers() {
    check("smollm2.json");
}
