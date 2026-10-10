//! `render` must give, byte for byte, the text transformers'
//! `apply_chat_template(messages, tools=tools, add_generation_prompt=...,
//! tokenize=False)` gives. The fixtures hold each model's chat template and
//! transformers' output for a set of conversations: plain and multi-turn
//! chat, a system message, non-ASCII and HTML-like text, tools, tool calls
//! (object and string arguments, parallel calls) and tool results, on the
//! Qwen2.5, Qwen3, SmolLM2, Mistral v0.3 (which writes `bos_token` and
//! `eos_token` into the text) and Llama 3.2 (which writes today's date with
//! `strftime_now`; the fixture pins the date and time transformers saw, and
//! `render_at` renders at the same one) templates. They are written by
//! `scripts/gen_chat_fixtures.py`.

use chrono::NaiveDateTime;
use lean_chat::{render, render_at, ChatTemplate, Message};
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
    now: Option<String>,
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    cases: Vec<Case>,
}

fn check(file: &str) {
    let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(!fixture.cases.is_empty());
    let template = ChatTemplate { source: fixture.template.clone(), bos_token: fixture.bos_token.clone(), eos_token: fixture.eos_token.clone() };
    let now = fixture.now.as_deref().map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").unwrap());
    for case in &fixture.cases {
        let rendered = match now {
            Some(now) => render_at(&template, &case.messages, case.tools.as_deref(), case.add_generation_prompt, now),
            None => render(&template, &case.messages, case.tools.as_deref(), case.add_generation_prompt),
        };
        let ours = rendered.unwrap_or_else(|e| panic!("[{} {}] render failed: {e:#}", fixture.model, case.name));
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

#[test]
fn mistral_v0_3_matches_transformers() {
    check("mistral_v0_3.json");
}

#[test]
fn llama3_2_matches_transformers() {
    check("llama3_2.json");
}
