//! `TextStream` deltas concatenate to the full decode for a multilingual
//! string (multi-byte characters split across byte-level BPE tokens) on the
//! Qwen2.5 and SmolLM2 tokenizers. Needs the tokenizer files on disk:
//! `LEAN_TOKENIZER_DIR=... LEAN_TOKENIZER_DIR_LLAMA_360M=... cargo test -p
//! lean --release --lib -- --ignored text_stream`.

use tokenizers::Tokenizer;

use crate::turn::TextStream;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run this test"))
}

/// Accents, CJK and emoji: several of these characters are split across
/// byte-level BPE tokens, so decoding one id at a time yields U+FFFD halves.
const MULTILINGUAL: &str = " Hello, world! Ça va très bien. 東京は日本の首都です。 Emoji: 🎉🚀 and naïve café.\nNew line.";

fn text_stream_roundtrip(tok_var: &str) {
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", env(tok_var))).expect("loading tokenizer.json");
    let ids = tokenizer.encode(MULTILINGUAL, false).unwrap().get_ids().to_vec();
    let full = tokenizer.decode(&ids, true).unwrap();
    let mut s = TextStream::new();
    let mut deltas: Vec<String> = ids.iter().map(|&id| s.push(&tokenizer, id).unwrap()).collect();
    deltas.push(s.finish(&tokenizer).unwrap());
    assert_eq!(deltas.concat(), full);
    assert!(deltas.iter().all(|d| !d.contains('\u{FFFD}')), "a delta carries half a character: {deltas:?}");
    let per_id: String = ids.iter().map(|&id| tokenizer.decode(&[id], true).unwrap()).collect();
    println!("{} ids; per-id decode equals full decode: {}", ids.len(), per_id == full);
}

#[test]
#[ignore = "needs LEAN_TOKENIZER_DIR on disk; never committed to this repo"]
fn text_stream_qwen25() {
    text_stream_roundtrip("LEAN_TOKENIZER_DIR");
}

#[test]
#[ignore = "needs LEAN_TOKENIZER_DIR_LLAMA_360M on disk; never committed to this repo"]
fn text_stream_smollm2() {
    text_stream_roundtrip("LEAN_TOKENIZER_DIR_LLAMA_360M");
}

