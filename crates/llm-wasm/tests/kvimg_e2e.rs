//! Native end-to-end test for prefix KV images (`docs/ENGINE.md` "Prefix
//! KV images"): exports the 13-tool system+tools prefix (`dtype: "q8_0"`),
//! imports it into a fresh `KvCache`, and checks that greedy generation
//! over the suffix of a real eval fixture's prompt matches a fresh
//! full-prompt prefill + generate exactly (q8_0's per-block quantization
//! error should not flip an argmax at any of the compared positions).
//!
//! Skips (prints + returns) instead of failing when the GGUF/tokenizer
//! files aren't present, matching `tests/full_forward.rs`'s convention.
//! Run serially (`--test-threads=1`) — same GPU-model-loading rule as
//! `full_forward.rs`.
#![cfg(feature = "wgpu")]

use std::path::Path;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use llm_wasm::eval::{self, ToolOrder, ToolSet};
use llm_wasm::gguf::Q4ModelLoader;
use llm_wasm::kvimg::{prefix_key, Dtype, Header, KvImage};
use llm_wasm::model::LlmModel;
use llm_wasm::template::{ChatTemplate, Message};
use llm_wasm::tokenizer::Tokenizer;

const SYSTEM: &str = "You are a helpful home assistant with access to Sonos speaker controls.";
const N_GENERATE: usize = 16;

fn model_dir() -> String {
    std::env::var("LLM_MODEL_DIR")
        .unwrap_or_else(|_| "./models/gguf/xlam-2-3b-fc-r".to_string())
}

fn hf_dir() -> String {
    std::env::var("LLM_HF_DIR")
        .unwrap_or_else(|_| "./models/hf/xLAM-2-3b-fc-r".to_string())
}

fn fixtures_dir() -> String {
    format!("{}/../../fixtures/sonos", env!("CARGO_MANIFEST_DIR"))
}

fn cases_path() -> String {
    format!("{}/../../eval/utterances.json", env!("CARGO_MANIFEST_DIR"))
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
    let parts = loader.load_deferred(device).expect("load gguf tensors");
    drop(loader);
    Some(parts.finalize(device).expect("finalize model on GPU"))
}

/// Renders the system+tools prefix the same way `bin/llm-agent.rs`'s
/// `kv-export` subcommand and `web.rs`'s `render_kv_prefix_tokens` do:
/// common leading tokens between two content-free probe renders.
fn render_kv_prefix(template: &ChatTemplate, tokenizer: &Tokenizer, tools: &[llm_wasm::template::Tool]) -> Vec<u32> {
    let render = |u: &str| -> Vec<u32> {
        let messages = vec![Message::system(SYSTEM), Message::user(u)];
        let prompt = template.render_prompt(&messages, tools, true).unwrap();
        tokenizer.encode(&prompt, false).unwrap()
    };
    let a = render("kv-export-probe-alpha");
    let b = render("totally-different-probe-beta");
    let prefix_len = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    a[..prefix_len].to_vec()
}

/// Prefill `prefix_tokens` into a throwaway cache and write a `.kvimg`
/// byte buffer under `dtype` — the in-process equivalent of `llm-agent
/// kv-export`, without going through a file.
fn build_kv_image(model: &LlmModel, prefix_tokens: &[u32], dtype: Dtype, model_fingerprint: &str, prefix_key_str: &str) -> Vec<u8> {
    let mut cache = model.new_cache(prefix_tokens.len());
    let _ = model.forward_hidden(prefix_tokens, &mut cache).expect("prefill prefix");
    let layers = cache.export_prefix(prefix_tokens.len());

    let header = Header {
        model_fingerprint: model_fingerprint.to_string(),
        prefix_key: prefix_key_str.to_string(),
        tokens: prefix_tokens.to_vec(),
        n_layers: model.config().num_layers,
        n_kv_heads: model.config().num_kv_heads,
        head_dim: model.config().hidden_size / model.config().num_heads,
        dtype: dtype.as_str().to_string(),
        engine: "llm-wasm/test".to_string(),
        created: "test".to_string(),
    };

    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, dtype, layer_refs).expect("write kv image");
    buf
}

/// Import `image_bytes` into a fresh cache and greedy-generate `N_GENERATE`
/// tokens over `suffix`. Returns `(generated_ids, import_duration)`.
fn generate_from_image(model: &LlmModel, image_bytes: &[u8], suffix: &[u32], max_ctx: usize) -> (Vec<u32>, std::time::Duration) {
    let (header, data_offset) = KvImage::read_header(image_bytes).expect("read kv image header");
    let mut cache = model.new_cache(max_ctx);

    let t_import = Instant::now();
    let mut layers = Vec::with_capacity(header.n_layers);
    for layer in 0..header.n_layers {
        layers.push(KvImage::layer_f32(image_bytes, &header, data_offset, layer).expect("decode kv image layer"));
    }
    cache.import_prefix(&layers, header.tokens.len());
    let import_dur = t_import.elapsed();

    let out = model.generate(suffix, N_GENERATE, &[], &mut cache).expect("generate from imported prefix");
    (out, import_dur)
}

#[test]
fn kv_image_import_matches_fresh_prefill_greedy() {
    let device = WgpuDevice::default();
    let Some(model) = load_model(&device) else {
        return;
    };

    let hf = hf_dir();
    let tok_path = format!("{hf}/tokenizer.json");
    let cfg_path = format!("{hf}/tokenizer_config.json");
    if !Path::new(&tok_path).exists() {
        eprintln!("skipping: {tok_path} not found");
        return;
    }
    let tokenizer = Tokenizer::from_json(&std::fs::read(&tok_path).unwrap()).expect("load tokenizer");
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).expect("parse tokenizer_config.json");
    let template = ChatTemplate::from_tokenizer_config(&cfg).expect("compile chat template");

    let fixtures = fixtures_dir();
    let all_tools = eval::load_all_tools(&fixtures).expect("load fixtures/sonos/tools.json");
    let tools = eval::select_tools_ordered(&all_tools, ToolSet::Twelve, ToolOrder::ListingFirst, &fixtures)
        .expect("select 13-tool subset"); // "Twelve" selects fixtures/sonos/tools-12.json, which has 13 entries

    let prefix_tokens = render_kv_prefix(&template, &tokenizer, &tools);
    assert!(!prefix_tokens.is_empty(), "empty system+tools prefix");
    println!("prefix: {} tokens", prefix_tokens.len());

    let cases = eval::load_cases(cases_path()).expect("load eval/utterances.json");
    let case = cases.iter().find(|c| c.id == "s02").expect("fixture s02 present in eval/utterances.json");
    let full_prompt_tokens = {
        let messages = vec![Message::system(SYSTEM), Message::user(&case.utterance)];
        let prompt = template.render_prompt(&messages, &tools, true).unwrap();
        tokenizer.encode(&prompt, false).unwrap()
    };
    assert_eq!(
        full_prompt_tokens[..prefix_tokens.len()],
        prefix_tokens[..],
        "fixture s02's rendered prompt does not start with the computed system+tools prefix"
    );
    let suffix = full_prompt_tokens[prefix_tokens.len()..].to_vec();
    println!("suffix ({}): {} tokens", case.id, suffix.len());

    let max_ctx = full_prompt_tokens.len() + N_GENERATE + 8;
    let model_fingerprint = "test-model-fingerprint";
    let prefix_key_str = prefix_key(model_fingerprint, &tokenizer.decode(&prefix_tokens, false).unwrap());

    // -- baseline: fresh full-prompt prefill + generate, no prefix caching --
    let t_fresh = Instant::now();
    let mut cache_fresh = model.new_cache(max_ctx);
    let out_fresh = model
        .generate(&full_prompt_tokens, N_GENERATE, &[], &mut cache_fresh)
        .expect("fresh full-prompt generate");
    let fresh_prefill_s = t_fresh.elapsed().as_secs_f64();
    println!("fresh full-prompt prefill+generate: {fresh_prefill_s:.3}s, out={out_fresh:?}");

    // -- q8_0 image: build, import, generate --
    let q8_image = build_kv_image(&model, &prefix_tokens, Dtype::Q8_0, model_fingerprint, &prefix_key_str);
    println!(
        "q8_0 image: {} bytes ({:.1} MB)",
        q8_image.len(),
        q8_image.len() as f64 / 1e6
    );
    let (out_q8, import_dur) = generate_from_image(&model, &q8_image, &suffix, max_ctx);
    println!("q8_0 import: {:.1}ms, out={out_q8:?}", import_dur.as_secs_f64() * 1000.0);

    if out_q8 == out_fresh {
        println!("PASS: q8_0-imported generation matches fresh full prefill exactly ({N_GENERATE} tokens)");
        return;
    }

    // Diverged: find the first differing position and check whether f32
    // (exact, no quantization error) also diverges — if it does too, the
    // divergence isn't q8_0's fault (e.g. a KV-cache-offset bug), so it's
    // worth knowing which case this is.
    let mismatch_pos = out_q8.iter().zip(out_fresh.iter()).position(|(a, b)| a != b);
    let f32_image = build_kv_image(&model, &prefix_tokens, Dtype::F32, model_fingerprint, &prefix_key_str);
    let (out_f32, _f32_import_dur) = generate_from_image(&model, &f32_image, &suffix, max_ctx);
    let f32_matches = out_f32 == out_fresh;

    panic!(
        "q8_0-imported generation diverged from fresh full prefill at position {mismatch_pos:?}:\n\
         fresh: {out_fresh:?}\n\
         q8_0:  {out_q8:?}\n\
         f32:   {out_f32:?} (matches fresh: {f32_matches})"
    );
}
