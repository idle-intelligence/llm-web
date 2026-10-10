//! Token-exact parity through the engine's own entry point, `Lean::generate`
//! (text in, text out), against the HF transformers fixtures
//! (`crates/lean-core/reference/gen_fixture*.py`, which load the same GGUF in transformers).
//!
//! For each fixture case with a `prompt`, the prompt is rendered through the
//! model's chat template (the text a chat client would pass), and:
//! - `Lean::token_count` of that text equals the fixture's `input_ids`
//!   length (the engine tokenizes exactly as transformers did);
//! - the greedy reply equals the fixture's `greedy_continuation` decoded,
//!   on the GPU and on the CPU;
//! - the streamed `on_token` pieces concatenate to the reply.
//!
//! One engine per backend serves every case in turn, so later cases also go
//! through the prefix reuse path (their prompts share the chat template's
//! leading system text with the previous call).
//!
//! The `no_retokenize` long-tools cases carry no `prompt` and are skipped.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_TOKENIZER_DIR=/path/to/Qwen2.5-0.5B-Instruct \
//! cargo test -p lean --release --test fixture_parity_generate -- --ignored qwen25_0_5b
//!
//! LEAN_GGUF_LLAMA_360M_Q4_0=/path/to/SmolLM2-360M-Instruct-Q4_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA_360M=/path/to/SmolLM2-360M-Instruct \
//! cargo test -p lean --release --test fixture_parity_generate -- --ignored smollm2_360m
//! ```

use lean::{Backend, GenParams, GgufSource, Lean, LoadOptions, TokenizerSource};
use lean_chat::{read_chat_template, render, Message};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    prompt: Option<String>,
    input_ids: Vec<u32>,
    greedy_continuation: Vec<u32>,
    #[serde(default)]
    no_retokenize: bool,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn load_fixture(path: &str) -> Fixture {
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parsing {path}: {e}"))
}

fn load(gguf: &str, tokenizer_dir: &str, backend: Backend) -> Lean {
    let opts = LoadOptions { gguf: GgufSource::Path(gguf.into()), tokenizer: TokenizerSource::Path(format!("{tokenizer_dir}/tokenizer.json")), max_ctx: 512, backend };
    pollster::block_on(Lean::load(opts)).expect("loading lean")
}

fn check_generate_matches_fixture(gguf: &str, tokenizer_dir: &str, fixture_path: &str) {
    let tokenizer = Tokenizer::from_file(format!("{tokenizer_dir}/tokenizer.json")).expect("loading tokenizer.json");
    let chat_template = read_chat_template(&format!("{tokenizer_dir}/tokenizer_config.json")).expect("reading chat_template");
    let fixture = load_fixture(fixture_path);

    let mut checked = 0;
    for backend in [Backend::WebGpu, Backend::Cpu] {
        let mut lean = load(gguf, tokenizer_dir, backend);
        for case in &fixture.cases {
            if case.no_retokenize {
                continue;
            }
            let prompt = case.prompt.as_deref().expect("case has neither prompt nor no_retokenize");
            let text = render(&chat_template, &[Message::user(prompt)], None, true).expect("rendering the prompt");
            assert_eq!(lean.token_count(&text).unwrap(), case.input_ids.len(), "[{}] token_count != fixture input_ids length", case.name);

            let expected = tokenizer.decode(&case.greedy_continuation, true).expect("decoding greedy_continuation");
            let params = GenParams { max_new_tokens: case.greedy_continuation.len() as u32, ..GenParams::default() };
            let mut streamed = String::new();
            let reply = pollster::block_on(lean.generate(&text, &params, |t| streamed.push_str(t), None)).unwrap();
            assert_eq!(streamed, reply, "[{}] {} on_token pieces must concatenate to the reply", case.name, lean.backend());
            assert_eq!(reply, expected, "[{}] {} reply != HF reference continuation", case.name, lean.backend());
            println!("[{}] {}: generate == HF reference ({} tokens)", case.name, lean.backend(), case.greedy_continuation.len());
            checked += 1;
        }
    }
    assert!(checked > 0, "no prompt-bearing cases found in {fixture_path}");
}

#[test]
#[ignore = "needs LEAN_GGUF and LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn generate_matches_hf_fixture_qwen25_0_5b() {
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR").expect("set LEAN_TOKENIZER_DIR to run this test");
    check_generate_matches_fixture(&gguf, &tokenizer_dir, concat!(env!("CARGO_MANIFEST_DIR"), "/../lean-core/reference/fixture.json"));
}

#[test]
#[ignore = "needs LEAN_GGUF_LLAMA_360M_Q4_0 and LEAN_TOKENIZER_DIR_LLAMA_360M on disk; never committed to this repo"]
fn generate_matches_hf_fixture_smollm2_360m() {
    let gguf = std::env::var("LEAN_GGUF_LLAMA_360M_Q4_0").expect("set LEAN_GGUF_LLAMA_360M_Q4_0 to run this test");
    let tokenizer_dir = std::env::var("LEAN_TOKENIZER_DIR_LLAMA_360M").expect("set LEAN_TOKENIZER_DIR_LLAMA_360M to run this test");
    check_generate_matches_fixture(&gguf, &tokenizer_dir, concat!(env!("CARGO_MANIFEST_DIR"), "/../lean-core/reference/fixture_llama_360m_q4_0.json"));
}
