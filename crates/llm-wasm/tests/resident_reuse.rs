//! CPU-only, tokenizer/template-only (no model, no GPU): quantifies how
//! many prompt tokens each step of a multi-step agent turn would need to
//! re-prefill under two restore strategies:
//!
//! (a) restore-to-constant-prefix — what `agent.rs::Agent::generate_attempt`
//!     and `web.rs::WebGenerator::generate_attempt` both do today: every
//!     step restores the KV cache to `prefix_len` (the constant
//!     system+tools preamble, computed once via `prefix_len_for`/
//!     `compute_prefix_len`) and re-prefills the *entire* conversation
//!     tail after it, even though most of that tail was already resident
//!     from the previous step's prefill+decode.
//! (b) longest-common-prefix-with-resident — restore to the longest run of
//!     tokens shared between what's actually resident in the KV cache
//!     (`resident_tokens` = previous step's prompt tokens ++ its generated
//!     tokens, minus the final EOS id, which is never fed through a
//!     forward pass — see `web.rs`'s decode loop, which pushes to
//!     `resident_tokens` only after each `forward_hidden` call and breaks
//!     *before* that call when the sampled token is a stop id) and this
//!     step's newly rendered prompt tokens, then re-prefill only the
//!     remainder.
//!
//! This builds a 3-step agent message history by hand (system + tools
//! preamble, a user utterance, then three `get_households_and_groups_and_players`
//! tool-call/tool-result rounds using the MCP-shaped fixture
//! `fixtures/sonos/results/get_households_and_groups_and_players.json`,
//! exactly as `agent::FixtureCaller` returns it), renders each step's
//! prompt with the real chat template exactly as `agent.rs`/`web.rs` do,
//! and tokenizes with the real tokenizer — no `Generator`/model involved.
//!
//! Also checks (case 4 of the driving question): is the assistant
//! tool-call turn `agent.rs`/`web.rs` re-render into history (from the
//! parsed `ToolCallEntry`/`ToolCallFunction`, via `template.rs`'s
//! `py_tojson`) byte-identical to what the model actually generated? If
//! not, the common prefix is cut short right there.

use llm_wasm::agent::{FixtureCaller, ToolCaller};
use llm_wasm::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use llm_wasm::tokenizer::Tokenizer;
use llm_wasm::tools::format_tool_result;
use serde_json::Value;
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    std::env::var("LLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models/hf/xLAM-2-3b-fc-r"))
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

fn load_parts() -> Option<(ChatTemplate, Tokenizer)> {
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

/// Longest run of leading token ids shared by `a` and `b` — the same
/// technique `agent.rs::Agent::prefix_len_for` /
/// `web.rs::compute_prefix_len` use for the constant system+tools prefix,
/// reused here for the general "how much of the resident cache still
/// applies" question.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// The model's own raw text for a single tool call, exactly the shape
/// `parse_output` expects and the chat template's own tool-call format
/// instructions describe: `[{"name": ..., "arguments": {...}}]` — no
/// trailing `<|im_end|>` here; that's a separate stop token, never fed
/// through a forward pass (see module docs), so it's excluded from what
/// ends up resident in the KV cache.
fn tool_call_text(name: &str, arguments: &Value) -> String {
    format!(
        r#"[{{"name": "{name}", "arguments": {}}}]"#,
        serde_json::to_string(arguments).unwrap()
    )
    // NB: arguments here is `{}`, whose compact serde_json rendering
    // happens to already match Python's `{}` — verified below against the
    // template's own `tojson` output for the identical value.
}

#[test]
fn resident_reuse_savings_across_steps() {
    let Some((template, tokenizer)) = load_parts() else {
        return;
    };
    let tools = tools_12();
    let system_prompt = "You are a helpful home assistant with access to Sonos speaker controls.";
    let mut caller = FixtureCaller::new(results_dir());

    let mut messages = vec![
        Message::system(system_prompt),
        Message::user("what speakers do I have, and is anything playing?"),
    ];

    // The constant system+tools prefix length, computed the same way
    // `Agent::prefix_len_for` does: common leading tokens between the
    // real first render and a throwaway probe utterance under the same
    // tools.
    let probe_messages = vec![Message::system(system_prompt), Message::user("\u{0}prefix-cache-probe\u{0}")];
    let real_render = template.render_prompt(&messages, &tools, true).unwrap();
    let real_tokens_probe_basis = tokenizer.encode(&real_render, false).unwrap();
    let probe_render = template.render_prompt(&probe_messages, &tools, true).unwrap();
    let probe_tokens = tokenizer.encode(&probe_render, false).unwrap();
    let constant_prefix_len = common_prefix_len(&real_tokens_probe_basis, &probe_tokens);

    struct StepRow {
        prompt_tokens: usize,
        common_with_resident: usize,
        prefill_a_constant: usize,
        prefill_b_lcp: usize,
    }
    let mut rows = Vec::new();
    // Step 0 starts warm: the constant system+tools prefix is already
    // resident (persisted across turns via the OPFS-backed KV image, see
    // `web.rs`'s `maybe_export_kv_prefix_to_opfs`/`import_prefix`) — only
    // the per-turn suffix is genuinely new at step 0.
    let mut resident: Vec<u32> = real_tokens_probe_basis[..constant_prefix_len].to_vec();
    // Byte-fidelity check (question 4): re-rendered assistant turn vs raw
    // model text, measured once (the mechanism is the same every step).
    let mut fidelity_checked = false;
    let mut fidelity_mismatch_tokens: Option<usize> = None;

    for step in 0..3usize {
        let prompt = template.render_prompt(&messages, &tools, true).unwrap();
        let prompt_tokens = tokenizer.encode(&prompt, false).unwrap();

        let common_with_resident = common_prefix_len(&resident, &prompt_tokens);
        let prefill_a = prompt_tokens.len() - constant_prefix_len.min(prompt_tokens.len());
        let prefill_b = prompt_tokens.len() - common_with_resident;
        rows.push(StepRow {
            prompt_tokens: prompt_tokens.len(),
            common_with_resident,
            prefill_a_constant: prefill_a,
            prefill_b_lcp: prefill_b,
        });

        // Scripted assistant tool call: repeat the same read tool each
        // step, as the "13-tool Sonos gate" trace does (constant +457
        // tokens/step) — see module docs.
        let name = "get_households_and_groups_and_players";
        let arguments = serde_json::json!({});
        let raw_text = tool_call_text(name, &arguments);

        let entries = vec![ToolCallEntry {
            id: Some(format!("call_{step}")),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: arguments.clone(),
            },
        }];
        messages.push(Message::assistant_tool_calls(entries));

        // Question 4: render *just* this turn (system+tools+user, then
        // this one assistant turn, no generation prompt) and diff its
        // `<|im_start|>assistant\n...<|im_end|>` body against `raw_text`.
        if !fidelity_checked {
            fidelity_checked = true;
            let rendered_with_turn = template.render_prompt(&messages, &tools, false).unwrap();
            let body = rendered_with_turn
                .rsplit("<|im_start|>assistant\n")
                .next()
                .unwrap()
                .trim_end_matches("<|im_end|>");
            if body != raw_text {
                let rendered_tokens = tokenizer.encode(body, false).unwrap();
                let raw_tokens = tokenizer.encode(&raw_text, false).unwrap();
                let lcp = common_prefix_len(&rendered_tokens, &raw_tokens);
                fidelity_mismatch_tokens = Some(raw_tokens.len().saturating_sub(lcp));
            }
        }

        let result = caller.call(name, &arguments).expect("fixture call should succeed");
        messages.push(format_tool_result(name, &result));

        // Update resident: this step's prompt tokens, plus the tokens
        // actually generated (tokenized from raw_text, i.e. excluding the
        // final EOS id — see module docs).
        let generated_tokens = tokenizer.encode(&raw_text, false).unwrap();
        resident = prompt_tokens;
        resident.extend_from_slice(&generated_tokens);
    }

    println!(
        "{:<5} {:>14} {:>20} {:>18} {:>18} {:>10}",
        "step", "prompt_tokens", "common_w/resident", "prefill_a(const)", "prefill_b(lcp)", "saving"
    );
    let mut total_a = 0usize;
    let mut total_b = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let saving = r.prefill_a_constant as isize - r.prefill_b_lcp as isize;
        println!(
            "{:<5} {:>14} {:>20} {:>18} {:>18} {:>10}",
            i, r.prompt_tokens, r.common_with_resident, r.prefill_a_constant, r.prefill_b_lcp, saving
        );
        total_a += r.prefill_a_constant;
        total_b += r.prefill_b_lcp;
        assert!(
            r.prefill_b_lcp <= r.prefill_a_constant,
            "step {i}: longest-common-prefix restore must never re-prefill more than constant-prefix restore"
        );
    }
    let total_saving = total_a - total_b;
    let saving_pct = 100.0 * total_saving as f64 / total_a as f64;
    println!("totals: prefill_a={total_a} prefill_b={total_b} saving={total_saving} ({saving_pct:.1}%)");

    match fidelity_mismatch_tokens {
        None => println!("fidelity: re-rendered assistant tool-call turn is byte-identical to raw model text"),
        Some(n) => println!(
            "fidelity: re-rendered assistant tool-call turn DIFFERS from raw model text by {n} trailing tokens"
        ),
    }

    // Step 0 has no resident cache yet, so both strategies must agree.
    assert_eq!(rows[0].prefill_a_constant, rows[0].prefill_b_lcp);
    assert!(
        total_saving > 0,
        "expected the longest-common-prefix restore to save tokens across steps 1..3"
    );
}

/// The degenerate case `agent.rs::generate_attempt`/`web.rs::generate_attempt`
/// both now guard against (the s02 "List all my speakers." browser-gate
/// bug fixed alongside this test, introduced by the longest-common-prefix
/// restore this module quantifies the savings of): the repeat-guard /
/// forced-text-answer retry path re-renders the exact same prompt a
/// previous step already prefilled in full, so
/// `common_prefix_len(resident, prompt_tokens)` equals `prompt_tokens.len()`
/// — the whole prompt, not just a prefix of it. Restoring the cache to
/// that length and prefilling the (empty) remainder would leave no
/// last-position logit to sample from. Both loops instead step the
/// restore point back one token so there's always exactly one token — the
/// prompt's last — left to prefill.
#[test]
fn fully_cached_prompt_prefills_exactly_one_token() {
    let Some((template, tokenizer)) = load_parts() else {
        return;
    };
    let tools = tools_12();
    let messages = vec![
        Message::system("You are a helpful home assistant with access to Sonos speaker controls."),
        Message::user("List all my speakers."),
    ];
    let prompt = template.render_prompt(&messages, &tools, true).unwrap();
    let prompt_tokens = tokenizer.encode(&prompt, false).unwrap();
    assert!(!prompt_tokens.is_empty());

    // Resident cache already holds exactly this prompt (the re-render on
    // the repeat-guard / forced-text-answer path).
    let resident = prompt_tokens.clone();
    let common_with_resident = common_prefix_len(&resident, &prompt_tokens);
    assert_eq!(
        common_with_resident,
        prompt_tokens.len(),
        "resident cache should cover the whole re-rendered prompt in this scenario"
    );

    // The fix: clamp the restore point back one token when the common
    // prefix would otherwise cover the whole prompt (mirrors
    // `agent.rs::generate_attempt`'s `prefix_len` clamp and
    // `web.rs::generate_attempt`'s `effective_prefix` clamp).
    let mut effective_prefix = common_with_resident;
    if effective_prefix == prompt_tokens.len() {
        effective_prefix -= 1;
    }
    let prefill_len = prompt_tokens.len() - effective_prefix;
    assert_eq!(
        prefill_len, 1,
        "fully-cached prompt must prefill exactly its final token, not 0 and not the whole prompt"
    );
}
