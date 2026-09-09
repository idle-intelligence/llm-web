//! Native CLI for llm-wasm. `required-features = ["native"]` in Cargo.toml
//! keeps this out of the wasm32-unknown-unknown / wasm-pack build.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use burn::backend::wgpu::WgpuDevice;
use clap::{Parser, Subcommand};
use llm_wasm::gguf::Q4ModelLoader;

#[derive(Parser)]
#[command(name = "llm-agent", about = "xLAM-2-3b-fc-r native CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run greedy generation over a token-id prompt (or plain text, with `--tokenizer`).
    Run {
        #[arg(long)]
        gguf: PathBuf,
        /// JSON file containing a flat array of token ids (see
        /// `fixtures/reference/rendered/*.tokens.json`).
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long, default_value_t = 32)]
        max_new: usize,
        /// Optional tokenizer.json for decoding generated ids to text.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 12288)]
        max_ctx: usize,
    },
    /// Evaluate the model against reference logits/outputs.
    Eval,
    /// Print GGUF header/tensor info for a model file.
    GgufInfo {
        gguf: PathBuf,
    },
    /// Load once, time prefill of `tokens` then `decode_steps` greedy decode
    /// steps from the resulting KV cache; reports prefill tok/s and median
    /// decode ms/token (kernel benchmarking — see docs/BENCHMARKS.md).
    Bench {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long, default_value_t = 32)]
        decode_steps: usize,
        #[arg(long, default_value_t = 12288)]
        max_ctx: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run {
            gguf,
            tokens,
            max_new,
            tokenizer,
            max_ctx,
        } => run(&gguf, &tokens, max_new, tokenizer.as_deref(), max_ctx),
        Commands::Eval => {
            println!("eval: not implemented (fixture wiring is a later phase)");
            Ok(())
        }
        Commands::GgufInfo { gguf } => gguf_info(&gguf),
        Commands::Bench {
            gguf,
            tokens,
            decode_steps,
            max_ctx,
        } => bench(&gguf, &tokens, decode_steps, max_ctx),
    }
}

fn gguf_info(path: &std::path::Path) -> anyhow::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let loader = Q4ModelLoader::new(reader)?;
    let r = loader.reader();

    println!("GGUF version: {}", r.version());
    println!("Tensor count: {}", r.tensor_count());
    println!();
    println!("-- metadata --");
    let mut keys: Vec<&String> = r.metadata().keys().collect();
    keys.sort();
    for k in keys {
        match r.metadata().get(k).unwrap() {
            llm_wasm::gguf::GgufValue::Array { elem_type, len } => {
                println!("{k} = Array(elem_type={elem_type}, len={len})");
            }
            other => println!("{k} = {other:?}"),
        }
    }

    println!();
    match llm_wasm::gguf::config_from_gguf(r) {
        Ok(cfg) => {
            println!("-- LlmConfig (derived) --");
            println!("{cfg:#?}");
        }
        Err(e) => println!("-- LlmConfig derivation failed: {e} --"),
    }

    println!();
    println!("-- tensors ({}) --", r.tensor_names().len());
    println!("{:<32} {:>20} {:>8}", "name", "shape", "dtype");
    for name in r.tensor_names() {
        let info = r.tensor_info(name).unwrap();
        println!(
            "{:<32} {:>20?} {:>8}",
            name,
            info.shape(),
            info.dtype().name()
        );
    }

    Ok(())
}

fn run(
    gguf_path: &std::path::Path,
    tokens_path: &std::path::Path,
    max_new: usize,
    tokenizer_path: Option<&std::path::Path>,
    max_ctx: usize,
) -> anyhow::Result<()> {
    let device = WgpuDevice::default();

    let t0 = std::time::Instant::now();
    let file = File::open(gguf_path)?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;
    let parts = loader.load_deferred(&device)?;
    drop(loader);
    let model = parts.finalize(&device)?;
    eprintln!("model load: {:.2}s", t0.elapsed().as_secs_f32());

    let tokens_json = std::fs::read_to_string(tokens_path)?;
    let prompt_ids: Vec<u32> = serde_json::from_str(&tokens_json)?;
    println!("prompt: {} tokens", prompt_ids.len());

    let mut cache = model.new_cache(max_ctx);
    let stop_ids = model.config().eos_token_ids.clone();

    let t1 = std::time::Instant::now();
    let generated = model.generate(&prompt_ids, max_new, &stop_ids, &mut cache);
    let dt = t1.elapsed().as_secs_f32();
    eprintln!(
        "generated {} tokens in {:.2}s ({:.1} ms/token overall, includes prefill)",
        generated.len(),
        dt,
        1000.0 * dt / generated.len().max(1) as f32
    );

    println!("generated ids: {generated:?}");

    if let Some(tok_path) = tokenizer_path {
        let bytes = std::fs::read(tok_path)?;
        let tok = llm_wasm::tokenizer::Tokenizer::from_json(&bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let text = tok
            .decode(&generated, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("generated text: {text}");
    }

    Ok(())
}

fn bench(
    gguf_path: &std::path::Path,
    tokens_path: &std::path::Path,
    decode_steps: usize,
    max_ctx: usize,
) -> anyhow::Result<()> {
    let device = WgpuDevice::default();

    let t0 = std::time::Instant::now();
    let file = File::open(gguf_path)?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;
    let parts = loader.load_deferred(&device)?;
    drop(loader);
    let model = parts.finalize(&device)?;
    eprintln!("model load: {:.2}s", t0.elapsed().as_secs_f32());

    let tokens_json = std::fs::read_to_string(tokens_path)?;
    let prompt_ids: Vec<u32> = serde_json::from_str(&tokens_json)?;
    println!("prompt: {} tokens", prompt_ids.len());

    let mut cache = model.new_cache(max_ctx);

    let t_prefill = std::time::Instant::now();
    let hidden = model.forward_hidden(&prompt_ids, &mut cache);
    let last = hidden.narrow(1, prompt_ids.len() - 1, 1);
    let logits = model.lm_head(last);
    // burn's wgpu backend dispatches asynchronously — force a sync readback
    // here (native-only `into_data()`, never do this in WASM) so
    // `prefill_dt` measures actual GPU completion, not just queue time.
    let mut logits_vec = llm_wasm::model::logits_to_vec(logits);
    let prefill_dt = t_prefill.elapsed().as_secs_f32();
    let prefill_tok_s = prompt_ids.len() as f32 / prefill_dt;
    println!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s)",
        prompt_ids.len(),
        prefill_dt,
        prefill_tok_s
    );

    let mut decode_ms: Vec<f32> = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        let next = llm_wasm::sample::greedy(&logits_vec);
        let t_step = std::time::Instant::now();
        let hidden = model.forward_hidden(&[next], &mut cache);
        let logits = model.lm_head(hidden);
        // Force sync so this step's GPU work is actually complete before
        // the next timer starts (native-only sync readback — see
        // model.rs's `logits_to_vec` doc comment; not a WASM code path).
        logits_vec = llm_wasm::model::logits_to_vec(logits);
        decode_ms.push(t_step.elapsed().as_secs_f32() * 1000.0);
    }

    decode_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = decode_ms[decode_ms.len() / 2];
    let mean: f32 = decode_ms.iter().sum::<f32>() / decode_ms.len() as f32;
    println!(
        "decode: {} steps, median {:.1} ms/token, mean {:.1} ms/token",
        decode_steps, median, mean
    );

    Ok(())
}
