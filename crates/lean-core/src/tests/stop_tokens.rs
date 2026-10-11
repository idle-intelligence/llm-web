//! The ids generation stops on (`Qwen2Config::eos_token_ids`, read from the
//! GGUF alone) against the `eos_token_id` list of the model's own
//! `generation_config.json`, which is what HF transformers' `generate` stops
//! on. Llama 3.2's GGUF names only `<|eot_id|>` as eos while its generation
//! config also lists `<|end_of_text|>` and `<|eom_id|>`.
//!
//! A unit test because `config_from_gguf` and the GGUF reader are private.
//! Ignored by default because it needs the GGUF and tokenizer files on disk.
//! Each model runs when both of its environment variables (the parity
//! tests' names) are set:
//!
//! ```sh
//! LEAN_GGUF_LLAMA32_1B_Q8_0=/path/to/Llama-3.2-1B-Instruct-Q8_0.gguf \
//! LEAN_TOKENIZER_DIR_LLAMA32_1B=/path/to/Llama-3.2-1B-Instruct \
//! cargo test -p lean-core -- --ignored stop_ids
//! ```

use std::fs::File;
use std::io::BufReader;

use crate::config::config_from_gguf;
use crate::gguf::GgufReader;

const MODELS: [(&str, &str); 7] = [
    ("LEAN_GGUF_LLAMA32_1B_Q8_0", "LEAN_TOKENIZER_DIR_LLAMA32_1B"),
    ("LEAN_GGUF_LLAMA32_3B_Q8_0", "LEAN_TOKENIZER_DIR_LLAMA32_3B"),
    ("LEAN_GGUF_QWEN3", "LEAN_TOKENIZER_DIR_QWEN3"),
    ("LEAN_GGUF_QWEN3_1_7B", "LEAN_TOKENIZER_DIR_QWEN3_1_7B"),
    ("LEAN_GGUF_QWEN25_3B", "LEAN_TOKENIZER_DIR_QWEN25_3B"),
    ("LEAN_GGUF_LLAMA_360M_Q8_0", "LEAN_TOKENIZER_DIR_LLAMA_360M"),
    ("LEAN_GGUF_LLAMA_1_7B_Q8_0", "LEAN_TOKENIZER_DIR_LLAMA_1_7B"),
];

#[test]
#[ignore]
fn stop_ids_match_generation_config() {
    let mut ran = 0;
    for (gguf_var, dir_var) in MODELS {
        let (Ok(gguf), Ok(dir)) = (std::env::var(gguf_var), std::env::var(dir_var)) else {
            eprintln!("{gguf_var} or {dir_var} not set, skipping");
            continue;
        };
        let mut reader = GgufReader::open(BufReader::new(File::open(&gguf).unwrap())).unwrap();
        let config = config_from_gguf(&mut reader).unwrap();
        let mut ours = config.eos_token_ids.clone();
        ours.sort_unstable();

        let gen: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/generation_config.json")).unwrap()).unwrap();
        let mut hf: Vec<u32> = match &gen["eos_token_id"] {
            serde_json::Value::Array(ids) => ids.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            v => vec![v.as_u64().unwrap() as u32],
        };
        hf.sort_unstable();
        hf.dedup();
        eprintln!("{gguf_var}: lean {ours:?} HF {hf:?}");
        assert_eq!(ours, hf, "[{gguf}] stop ids differ from generation_config.json");
        assert!(!ours.contains(&config.bos_token_id) || hf.contains(&config.bos_token_id), "[{gguf}] bos is a stop id");
        ran += 1;
    }
    assert!(ran > 0, "no model set; see the module doc");
}
