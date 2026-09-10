//! Native CLI for llm-wasm. `required-features = ["native"]` in Cargo.toml
//! keeps this out of the wasm32-unknown-unknown / wasm-pack build.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use clap::{Parser, Subcommand};
use llm_wasm::gguf::{q4_matmul, set_skip_matvec_for_bench, Q4ModelLoader, Q4Tensor};

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

/// Deterministic xorshift PRNG — matches `tests/q4_matmul.rs`'s generator,
/// duplicated here since this bin has no test-only dependency on that crate.
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

/// Random Q4_0 bytes of shape `[n, k]` — content doesn't matter for a pure
/// throughput measurement (the dequant/matvec math has no data-dependent
/// branches), only the byte count.
fn random_q4_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 18;
    let mut rng = Xorshift::new(seed);
    (0..n * bytes_per_row)
        .map(|_| (rng.next_u32() & 0xFF) as u8)
        .collect()
}

/// P1a (docs/BENCHMARKS.md): isolated K1 matvec (M=1) throughput at the
/// model's four matmul shapes, 100 iterations each, one sync at the end.
/// Returns `(k, n, ms_per_call, gb_per_s)` per shape.
fn bench_matvec_isolated(device: &WgpuDevice) -> Vec<(usize, usize, f64, f64)> {
    let shapes = [
        (2048usize, 151936usize), // lm_head / tied embedding
        (2048, 11008),            // ffn_gate / ffn_up
        (11008, 2048),            // ffn_down
        (2048, 2048),             // attn_q / attn_output
    ];
    let iters = 100;
    let mut results = Vec::with_capacity(shapes.len());

    for &(k, n) in &shapes {
        let bytes = random_q4_bytes(n, k, 0x5EED ^ (k as u64) ^ ((n as u64) << 20));
        let weights = Q4Tensor::from_q4_bytes(&bytes, [n, k], device).expect("upload Q4 weights");
        let input: Tensor<Wgpu, 3> =
            Tensor::<Wgpu, 1>::from_floats(vec![0.01f32; k].as_slice(), device).reshape([1, 1, k]);

        // Warm-up: pipeline compile + first dispatch, synced so it doesn't
        // bleed into the timed loop.
        let _ = q4_matmul(input.clone(), &weights)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        let t0 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(q4_matmul(input.clone(), &weights));
        }
        // Single sync at the end (queue is FIFO on one queue — this waits
        // for all `iters` dispatches, not just the last).
        let _ = last.unwrap().into_data().into_vec::<f32>().unwrap();
        let dt = t0.elapsed().as_secs_f64() / iters as f64;

        let bytes_per_call = (n as f64) * (k as f64) * 18.0 / 32.0;
        let gb_per_s = bytes_per_call / dt / 1e9;
        results.push((k, n, dt * 1000.0, gb_per_s));
        println!(
            "  matvec K={k:>6} N={n:>6}: {:>7.3} ms/call, {:>6.1} GB/s",
            dt * 1000.0,
            gb_per_s
        );
    }
    results
}

/// P1b (docs/BENCHMARKS.md): analytical dispatch-count estimate — list every
/// GPU op in one decoder layer and multiply by `num_layers`. Not an
/// instrumented count (Burn's wgpu backend doesn't expose a dispatch
/// counter to this crate); each line's multiplicity is read directly off
/// `model.rs`'s decoder-block/attention/RoPE/FFN code paths.
fn print_dispatch_estimate(num_layers: usize) {
    let per_layer: &[(&str, usize)] = &[
        ("RMSNorm (attn_norm + ffn_norm, ~4 dispatches each: mean/var/rsqrt/mul)", 8),
        ("Q4Linear matmuls (q,k,v,o,gate,up,down)", 7),
        ("q/k/v bias adds", 3),
        ("RoPE (apply_rope x2 for q,k: mul_scalar+cat+mul+mul+add each)", 10),
        ("repeat_kv cat (k_all, v_all)", 2),
        ("KV cache slice_assign (k, v)", 2),
        ("attention core (QK^T matmul, scale mul, mask compare+fill, softmax~3, PV matmul)", 8),
        ("residual adds (attn, ffn)", 2),
        ("SwiGLU (silu, mul)", 2),
    ];
    let per_layer_total: usize = per_layer.iter().map(|(_, n)| n).sum();
    println!("  per decoder layer:");
    for (name, n) in per_layer {
        println!("    {n:>3}  {name}");
    }
    println!("  per-layer total: {per_layer_total}");
    println!(
        "  x {num_layers} layers = {}, + out_norm(~4) + lm_head matmul(1) = ~{}",
        per_layer_total * num_layers,
        per_layer_total * num_layers + 4 + 1
    );
}

fn bench(
    gguf_path: &std::path::Path,
    tokens_path: &std::path::Path,
    decode_steps: usize,
    max_ctx: usize,
) -> anyhow::Result<()> {
    let device = WgpuDevice::default();

    println!("-- P1a: isolated K1 matvec throughput (100 iters/shape) --");
    let matvec_results = bench_matvec_isolated(&device);

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

    println!("-- P1b: GPU dispatch count per decode token --");
    print_dispatch_estimate(model.config().num_layers);

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

    // P1: prefill breakdown — matmul-call sum vs (attention + everything
    // else combined). Re-runs prefill on a throwaway cache with all
    // `q4_matmul` dispatches skipped (see `set_skip_matvec_for_bench`'s doc
    // comment); `prefill_dt - prefill_skip_dt` isolates the matmul-call
    // cost, `prefill_skip_dt` is attention + softmax + norms + RoPE +
    // everything else combined (not split further — see docs/BENCHMARKS.md
    // P1's note on the time budget for a 3-way split).
    set_skip_matvec_for_bench(true);
    let mut prefill_bench_cache = model.new_cache(max_ctx);
    let t_pf_skip = std::time::Instant::now();
    let hidden2 = model.forward_hidden(&prompt_ids, &mut prefill_bench_cache);
    let last2 = hidden2.narrow(1, prompt_ids.len() - 1, 1);
    let logits2 = model.lm_head(last2);
    let _ = llm_wasm::model::logits_to_vec(logits2);
    let prefill_skip_dt = t_pf_skip.elapsed().as_secs_f32();
    set_skip_matvec_for_bench(false);
    let prefill_matmul_dt = prefill_dt - prefill_skip_dt;
    println!(
        "prefill breakdown: matmul-calls {:.2}s, attention+rest {:.2}s (of {:.2}s total)",
        prefill_matmul_dt, prefill_skip_dt, prefill_dt
    );

    // P1c: decode per-token wall-time breakdown.
    println!("-- P1c: decode per-token wall-time breakdown --");
    let t_embed = std::time::Instant::now();
    let embed_tensor = model.embed_tokens(&[prompt_ids[0]]);
    let _ = embed_tensor.into_data().into_vec::<f32>().unwrap();
    let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;

    set_skip_matvec_for_bench(true);
    let mut rest_cache = model.new_cache(max_ctx);
    // warm-up
    let _ = model.forward_hidden(&prompt_ids[..1], &mut rest_cache);
    let rest_iters = 8;
    let t_rest = std::time::Instant::now();
    let mut last_rest = None;
    for _ in 0..rest_iters {
        let h = model.forward_hidden(&prompt_ids[..1], &mut rest_cache);
        last_rest = Some(model.lm_head(h));
    }
    let _ = llm_wasm::model::logits_to_vec(last_rest.unwrap());
    let rest_ms = t_rest.elapsed().as_secs_f64() * 1000.0 / rest_iters as f64;
    set_skip_matvec_for_bench(false);

    // Sum-of-isolated-matvecs estimate for one decode step (M=1), from
    // P1a's numbers. matvec_results order: [(2048,151936) head,
    // (2048,11008) gate/up, (11008,2048) down, (2048,2048) q/o]. k/v_proj
    // (N=256, K=2048) isn't one of the 4 measured shapes — estimated by
    // scaling the measured 2048x2048 GB/s (same bandwidth-bound regime) by
    // byte count.
    let (_, _, head_ms, _) = matvec_results[0];
    let (_, _, gate_up_ms, _) = matvec_results[1];
    let (_, _, down_ms, _) = matvec_results[2];
    let (_, _, qo_ms, qo_gbps) = matvec_results[3];
    let kv_bytes = 256.0 * 2048.0 * 18.0 / 32.0;
    let kv_ms = kv_bytes / (qo_gbps * 1e9) * 1000.0;
    let per_layer_matvec_ms = 2.0 * qo_ms + 2.0 * gate_up_ms + down_ms + 2.0 * kv_ms;
    let num_layers = model.config().num_layers as f64;
    let all_layers_matvec_ms = per_layer_matvec_ms * num_layers;
    let matvec_sum_ms = all_layers_matvec_ms + head_ms;

    println!(
        "  embedding dequant+upload (CPU): {embed_ms:.3} ms\n\
         \x20 36 layers of matvecs (sum of P1a isolated numbers x counts): {all_layers_matvec_ms:.2} ms\n\
         \x20 everything else on GPU (RMSNorm, RoPE, attention, SwiGLU, residuals; skip-matvec measurement): {rest_ms:.2} ms\n\
         \x20 final head matvec (N=151936, K=2048, from P1a): {head_ms:.3} ms\n\
         \x20 matvec_sum (36 layers + head) = {matvec_sum_ms:.2} ms"
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

    // P1c: sync readback of logits alone (151936 f32 = 600KB). Lower-bound
    // estimate: a zeros tensor with no pending compute of its own, so this
    // measures the `into_data()` transfer/wait itself, not GPU-completion
    // wait for a real forward pass (that's folded into `decode_ms` above).
    let vocab = model.config().vocab_size;
    let readback_tensor: Tensor<Wgpu, 3> = Tensor::zeros([1, 1, vocab], &device);
    let _ = readback_tensor
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap(); // warm-up
    let readback_iters = 20;
    let t_rb = std::time::Instant::now();
    for _ in 0..readback_iters {
        let _ = readback_tensor.clone().into_data().into_vec::<f32>().unwrap();
    }
    let readback_ms = t_rb.elapsed().as_secs_f64() * 1000.0 / readback_iters as f64;
    println!("  sync readback of logits (into_data, {vocab} f32): {readback_ms:.3} ms");

    Ok(())
}
