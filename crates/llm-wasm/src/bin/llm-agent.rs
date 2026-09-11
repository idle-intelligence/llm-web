//! Native CLI for llm-wasm. `required-features = ["native"]` in Cargo.toml
//! keeps this out of the wasm32-unknown-unknown / wasm-pack build.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Duration;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use clap::{Parser, Subcommand};
use llm_wasm::agent::{Agent, FixtureCaller, Generator};
use llm_wasm::eval::{self, ToolOrder, ToolSet};
use llm_wasm::gguf::{
    bench_matmul_variant, q4_dequant_scratch_to_vec, q4_matmul, q4_matmul_naive_forced,
    set_skip_matvec_for_bench, Q4ModelLoader, Q4Tensor, BENCH_STRATEGY_NAMES,
};
use llm_wasm::kv::KvCache;
use llm_wasm::kvimg::{self, Dtype, Header, KvImage};
use llm_wasm::model::LlmModel;
use llm_wasm::template::{ChatTemplate, Message, Tool};
use llm_wasm::tokenizer::Tokenizer;

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
        /// Print a warning and generate unconstrained (schema-constrained
        /// decoding needs a tool schema, which a bare token-id prompt
        /// doesn't have — see `Eval`'s `--constrained` for the real thing).
        #[arg(long)]
        constrained: bool,
    },
    /// Run the Sonos MCP agent eval harness (`eval/README.md`) against the
    /// real (Burn+wgpu) model, prefix-caching the constant system+tools
    /// prefix across cases with the same tool set.
    Eval {
        #[arg(long)]
        gguf: Option<PathBuf>,
        /// Directory containing `tokenizer.json` + `tokenizer_config.json`.
        #[arg(long = "model-dir")]
        model_dir: Option<PathBuf>,
        /// `all` (34 tools, `fixtures/sonos/tools.json`) or `12`
        /// (`fixtures/sonos/tools-12.json`).
        #[arg(long, default_value = "all")]
        tools: String,
        /// `alphabetical` (`tools.json`'s own order, filtered for `--tools
        /// 12`) or `listing-first` (`get_households_and_groups_and_players`
        /// first — `tools-12.json`'s order for `--tools 12`,
        /// `fixtures/sonos/tools-listing-first.json`'s order for `--tools
        /// all`). Folded into the label/filename and the parameters block
        /// (see docs/ENGINE.md "Known issues / fixed").
        #[arg(long = "tool-order", default_value = "listing-first")]
        tool_order: String,
        #[arg(long, default_value = "eval/utterances.json")]
        cases: PathBuf,
        #[arg(long, default_value = "fixtures/sonos")]
        fixtures: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Comma-separated case ids to run instead of the full set, e.g.
        /// `s01,m01`.
        #[arg(long)]
        only: Option<String>,
        #[arg(long = "max-new-tokens", default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long = "max-steps", default_value_t = 6)]
        max_steps: usize,
        #[arg(long = "max-ctx", default_value_t = 12288)]
        max_ctx: usize,
        /// System prompt for the agent (verbatim into the report's
        /// parameters block).
        #[arg(
            long,
            default_value = "You are a helpful home assistant with access to Sonos speaker controls."
        )]
        system: String,
        /// Run label, folded into the output filename
        /// (`eval/results/<date>-<tools>-<label>-native.md`) and recorded
        /// in the parameters block.
        #[arg(long, default_value = "default")]
        label: String,
        /// Schema-constrained decoding (docs/ENGINE.md "Schema-constrained
        /// decoding"): jump-forward through grammar-forced spans instead
        /// of sampling them one token at a time.
        #[arg(long)]
        constrained: bool,
    },
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
    /// Session 10: loads once, then runs a fixed sequence of prefills at
    /// given prefix lengths of `tokens` (fresh `KvCache` per call, so each
    /// timing is prefill-only), reporting per-call wall time — used to
    /// check whether `pad_m_bucket`'s M-bucketing keeps `Tensor::matmul`'s
    /// `autotune` from re-tuning on every distinct prefill length (see
    /// docs/BENCHMARKS.md Session 10).
    AutotuneSweep {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        /// Comma-separated prefix lengths to prefill in order, e.g.
        /// "2225,2225,531,531,640".
        #[arg(long)]
        lengths: String,
        #[arg(long, default_value_t = 12288)]
        max_ctx: usize,
    },
    /// Session 15 (docs/BENCHMARKS.md): sweeps M x `cubek_matmul::Strategy`
    /// at fixed production-representative (K,N) shapes, reporting GFLOP/s
    /// for each combination plus the naive per-element-dequant kernel and
    /// an isolated dequant-only timing — no GGUF load needed, weights are
    /// synthetic random Q4_0 bytes (content doesn't affect matmul
    /// throughput, only shape).
    PrefillSweep {
        /// Comma-separated M values, e.g. "64,128,256,460,512,1024,2048,2304".
        #[arg(long, default_value = "64,128,256,460,512,1024,2048,2304")]
        lengths: String,
        /// Timed repeats per (shape, strategy, M) after one untimed warm-up.
        #[arg(long, default_value_t = 3)]
        reps: usize,
    },
    /// Build-time export of a prefix KV image (docs/ENGINE.md "Prefix KV
    /// images"): renders the system+tools prefix exactly as the agent loop
    /// does, prefills it natively, and writes `<out-dir>/<prefix_key>.kvimg`
    /// (+ a human-readable `.json` sidecar) so any engine instance can
    /// import it instead of running prefill token-by-token.
    KvExport {
        #[arg(long)]
        gguf: Option<PathBuf>,
        #[arg(long = "model-dir")]
        model_dir: Option<PathBuf>,
        /// Path to an MCP `tools/list`-shaped JSON array — e.g.
        /// `fixtures/sonos/tools-12.json`, `fixtures/sonos/tools.json`, or
        /// any other file with the same shape.
        #[arg(long)]
        tools: PathBuf,
        #[arg(long, default_value = "You are a helpful assistant with access to tools.")]
        system: String,
        #[arg(long, default_value = "q8_0")]
        dtype: String,
        #[arg(long = "out-dir")]
        out_dir: Option<PathBuf>,
        /// `listing-first` (move `get_households_and_groups_and_players`
        /// to the front if present, per docs/ENGINE.md "Known issues /
        /// fixed") or `as-is` (`--tools` file's own order, unchanged).
        #[arg(long = "tool-order", default_value = "listing-first")]
        tool_order: String,
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
            constrained,
        } => {
            if constrained {
                eprintln!(
                    "--constrained ignored: `run` has no tool schema to constrain against \
                     (bare token-id prompt) — use `eval --constrained` instead"
                );
            }
            run(&gguf, &tokens, max_new, tokenizer.as_deref(), max_ctx)
        }
        Commands::Eval {
            gguf,
            model_dir,
            tools,
            tool_order,
            cases,
            fixtures,
            out,
            only,
            max_new_tokens,
            max_steps,
            max_ctx,
            system,
            label,
            constrained,
        } => run_eval(
            gguf,
            model_dir,
            &tools,
            &tool_order,
            &cases,
            &fixtures,
            out,
            only,
            max_new_tokens,
            max_steps,
            max_ctx,
            &system,
            &label,
            constrained,
        ),
        Commands::GgufInfo { gguf } => gguf_info(&gguf),
        Commands::Bench {
            gguf,
            tokens,
            decode_steps,
            max_ctx,
        } => bench(&gguf, &tokens, decode_steps, max_ctx),
        Commands::AutotuneSweep {
            gguf,
            tokens,
            lengths,
            max_ctx,
        } => autotune_sweep(&gguf, &tokens, &lengths, max_ctx),
        Commands::PrefillSweep { lengths, reps } => prefill_sweep(&lengths, reps),
        Commands::KvExport {
            gguf,
            model_dir,
            tools,
            system,
            dtype,
            out_dir,
            tool_order,
        } => kv_export(gguf, model_dir, &tools, &system, &dtype, out_dir, &tool_order),
    }
}

fn autotune_sweep(
    gguf_path: &std::path::Path,
    tokens_path: &std::path::Path,
    lengths: &str,
    max_ctx: usize,
) -> anyhow::Result<()> {
    let device = WgpuDevice::default();
    let file = File::open(gguf_path)?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;
    let parts = loader.load_deferred(&device)?;
    drop(loader);
    let model = parts.finalize(&device)?;

    let tokens_json = std::fs::read_to_string(tokens_path)?;
    let prompt_ids: Vec<u32> = serde_json::from_str(&tokens_json)?;

    let lens: Vec<usize> = lengths
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("lengths must be comma-separated integers"))
        .collect();

    for &m in &lens {
        assert!(
            m <= prompt_ids.len(),
            "requested length {m} exceeds fixture length {}",
            prompt_ids.len()
        );
        let ids = &prompt_ids[..m];
        let mut cache = model.new_cache(max_ctx);
        let t = std::time::Instant::now();
        let hidden = model.forward_hidden(ids, &mut cache)?;
        let last = hidden.narrow(1, m - 1, 1);
        let logits = model.lm_head(last);
        let _ = llm_wasm::model::logits_to_vec(logits)?;
        let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("M={m:5}: {dt_ms:8.1} ms ({:.1} tok/s)", m as f64 / (dt_ms / 1000.0));
    }
    Ok(())
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
    let generated = model.generate(&prompt_ids, max_new, &stop_ids, &mut cache)?;
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

/// Session 15 (docs/BENCHMARKS.md): M x `Strategy` sweep at fixed
/// production-representative (K,N) shapes — `(2048,2048)` (attn_q/o/k/v
/// grouped-shape stand-in) and `(2048,11008)` (ffn_gate/up, the largest
/// projection). Reports GFLOP/s per (shape, strategy, M); also runs the
/// naive per-element-dequant kernel (`q4_matmul_naive_forced`) for
/// comparison and times `q4_dequant_scratch_to_vec` alone once per shape.
fn prefill_sweep(lengths: &str, reps: usize) -> anyhow::Result<()> {
    let device = WgpuDevice::default();
    let ms: Vec<usize> = lengths
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("lengths must be comma-separated integers"))
        .collect();
    let shapes: &[(usize, usize, &str)] = &[(2048, 2048, "attn(2048x2048)"), (2048, 11008, "ffn_gate(2048x11008)")];

    for &(k, n, label) in shapes {
        let bytes = random_q4_bytes(n, k, 0x5EED ^ (k as u64) ^ ((n as u64) << 20));
        let weights = Q4Tensor::from_q4_bytes(&bytes, [n, k], &device).expect("upload Q4 weights");

        // Dequant-only timing, once per shape (readback included — real
        // GPU-completion sync point, not just submission).
        let t0 = std::time::Instant::now();
        let _ = q4_dequant_scratch_to_vec(&weights, &device);
        let dequant_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("dequant-only {label}: {dequant_ms:.2} ms ({:.1} GB/s)", (n as f64 * k as f64 * 18.0 / 32.0) / (dequant_ms / 1000.0) / 1e9);

        for &m in &ms {
            let x: Tensor<Wgpu, 3> =
                Tensor::<Wgpu, 1>::from_floats(vec![0.01f32; m * k].as_slice(), &device).reshape([1, m, k]);

            // naive per-element-dequant kernel
            {
                let out = q4_matmul_naive_forced(x.clone(), &weights);
                let _ = out.into_data().into_vec::<f32>().unwrap();
                let mut best = f64::MAX;
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    let out = q4_matmul_naive_forced(x.clone(), &weights);
                    let _ = out.into_data().into_vec::<f32>().unwrap();
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                let gflops = 2.0 * m as f64 * n as f64 * k as f64 / best / 1e9;
                println!(
                    "{label:>20} M={m:>5} strategy=naive          {:>8.1} ms  {:>7.1} GFLOP/s",
                    best * 1000.0,
                    gflops
                );
            }

            for &strat in BENCH_STRATEGY_NAMES {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let out = bench_matmul_variant(x.clone(), &weights, &device, strat);
                    out.into_data().into_vec::<f32>().unwrap()
                }));
                if result.is_err() {
                    println!("{label:>20} M={m:>5} strategy={strat:<18} FAILED (panic)");
                    continue;
                }
                let mut best = f64::MAX;
                let mut ok = true;
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let out = bench_matmul_variant(x.clone(), &weights, &device, strat);
                        out.into_data().into_vec::<f32>().unwrap()
                    }));
                    if result.is_err() {
                        ok = false;
                        break;
                    }
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                if !ok {
                    println!("{label:>20} M={m:>5} strategy={strat:<18} FAILED (panic)");
                    continue;
                }
                let gflops = 2.0 * m as f64 * n as f64 * k as f64 / best / 1e9;
                println!(
                    "{label:>20} M={m:>5} strategy={strat:<18} {:>8.1} ms  {:>7.1} GFLOP/s",
                    best * 1000.0,
                    gflops
                );
            }
        }
    }
    Ok(())
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
///
/// Session 9 F1/F2: RoPE (10 -> 1) and SwiGLU's elementwise step (2 -> 1)
/// were each replaced by one fused WGSL dispatch (`gguf::rope_fused`,
/// `gguf::silu_mul_fused`) — see model.rs's `Q4Attention::forward`/
/// `Q4FeedForward::forward`.
fn print_dispatch_estimate(num_layers: usize) {
    let per_layer: &[(&str, usize)] = &[
        ("RMSNorm (attn_norm + ffn_norm, ~4 dispatches each: mean/var/rsqrt/mul)", 8),
        ("Q4Linear matmuls (q,k,v,o,gate,up,down)", 7),
        ("q/k/v bias adds", 3),
        ("RoPE (fused kernel, F1 Session 9 — was 10: apply_rope x2 for q,k)", 1),
        ("repeat_kv cat (k_all, v_all)", 2),
        ("KV cache slice_assign (k, v)", 2),
        ("attention core (QK^T matmul, scale mul, mask compare+fill, softmax~3, PV matmul)", 8),
        ("residual adds (attn, ffn)", 2),
        ("SiLU*up (fused kernel, F2 Session 9 — was 2: silu, mul)", 1),
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
    let hidden = model.forward_hidden(&prompt_ids, &mut cache)?;
    let last = hidden.narrow(1, prompt_ids.len() - 1, 1);
    let logits = model.lm_head(last);
    // burn's wgpu backend dispatches asynchronously — force a sync readback
    // here (native-only `into_data()`, never do this in WASM) so
    // `prefill_dt` measures actual GPU completion, not just queue time.
    let mut logits_vec = llm_wasm::model::logits_to_vec(logits)?;
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
    let hidden2 = model.forward_hidden(&prompt_ids, &mut prefill_bench_cache)?;
    let last2 = hidden2.narrow(1, prompt_ids.len() - 1, 1);
    let logits2 = model.lm_head(last2);
    let _ = llm_wasm::model::logits_to_vec(logits2)?;
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
    let embed_tensor = model.embed_tokens(&[prompt_ids[0]])?;
    let _ = embed_tensor.into_data().into_vec::<f32>().unwrap();
    let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;

    set_skip_matvec_for_bench(true);
    let mut rest_cache = model.new_cache(max_ctx);
    // warm-up
    let _ = model.forward_hidden(&prompt_ids[..1], &mut rest_cache)?;
    let rest_iters = 8;
    let t_rest = std::time::Instant::now();
    let mut last_rest = None;
    for _ in 0..rest_iters {
        let h = model.forward_hidden(&prompt_ids[..1], &mut rest_cache)?;
        last_rest = Some(model.lm_head(h));
    }
    let _ = llm_wasm::model::logits_to_vec(last_rest.unwrap())?;
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
        let hidden = model.forward_hidden(&[next], &mut cache)?;
        let logits = model.lm_head(hidden);
        // Force sync so this step's GPU work is actually complete before
        // the next timer starts (native-only sync readback — see
        // model.rs's `logits_to_vec` doc comment; not a WASM code path).
        logits_vec = llm_wasm::model::logits_to_vec(logits)?;
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

/// Native `Generator` adapter over the real `LlmModel`: implements
/// `generate_with_cached_prefix` by tracking the token ids currently
/// resident in `cache` as a valid prefix and, when a step's `prefix_len`
/// hint matches that stored prefix exactly, `KvCache::restore()`-ing to it
/// and re-prefilling only the new suffix instead of the whole prompt (see
/// `agent.rs`'s "Prefix caching" module docs and `kv.rs`'s
/// `snapshot`/`restore`). A prefix mismatch (different tool set, or the
/// very first call before anything is cached) falls back to a full
/// `restore(0)` + full-prompt prefill, after which the new prefix is
/// recorded for subsequent calls to reuse.
struct NativeGenerator {
    model: LlmModel,
    cache: KvCache,
    cached_prefix: Vec<u32>,
    last_prefill: Duration,
    last_decode: Duration,
}

impl NativeGenerator {
    fn new(model: LlmModel, max_ctx: usize) -> Self {
        let cache = model.new_cache(max_ctx);
        Self {
            model,
            cache,
            cached_prefix: Vec::new(),
            last_prefill: Duration::ZERO,
            last_decode: Duration::ZERO,
        }
    }
}

impl Generator for NativeGenerator {
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        stop_ids: &[u32],
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with_cached_prefix(prompt_ids, 0, max_new_tokens, stop_ids)
    }

    fn generate_with_cached_prefix(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
        stop_ids: &[u32],
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(!prompt_ids.is_empty(), "generate called with an empty prompt");
        let prefix_len = prefix_len.min(prompt_ids.len());
        let reuse = prefix_len > 0
            && prefix_len == self.cached_prefix.len()
            && prompt_ids[..prefix_len] == self.cached_prefix[..];

        let t_prefill = std::time::Instant::now();
        let suffix_start = if reuse {
            self.cache.restore(prefix_len);
            prefix_len
        } else {
            self.cache.restore(0);
            0
        };
        let suffix = &prompt_ids[suffix_start..];
        let hidden = self.model.forward_hidden(suffix, &mut self.cache)?;
        let last = hidden.narrow(1, suffix.len() - 1, 1);
        let logits = self.model.lm_head(last);
        let mut logits_vec = llm_wasm::model::logits_to_vec(logits)?;
        self.last_prefill = t_prefill.elapsed();

        if !reuse {
            self.cached_prefix = prompt_ids[..prefix_len].to_vec();
        }

        let t_decode = std::time::Instant::now();
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = llm_wasm::sample::greedy(&logits_vec);
            out.push(next);
            if stop_ids.contains(&next) {
                break;
            }
            let hidden = self.model.forward_hidden(&[next], &mut self.cache)?;
            let logits = self.model.lm_head(hidden);
            logits_vec = llm_wasm::model::logits_to_vec(logits)?;
        }
        self.last_decode = t_decode.elapsed();

        Ok(out)
    }

    fn last_call_timing(&self) -> (Duration, Duration) {
        (self.last_prefill, self.last_decode)
    }

    /// Same prefix-reuse prefill as `generate_with_cached_prefix`, but the
    /// decode tail is `model.rs`'s jump-forward loop
    /// (`LlmModel::decode_with_constraint`) instead of plain greedy —
    /// see `docs/ENGINE.md` "Schema-constrained decoding".
    fn generate_constrained(
        &mut self,
        prompt_ids: &[u32],
        prefix_len: usize,
        max_new_tokens: usize,
        stop_ids: &[u32],
        constraint: Option<&mut dyn llm_wasm::grammar::Constraint>,
    ) -> anyhow::Result<llm_wasm::agent::GenerateOutput> {
        anyhow::ensure!(!prompt_ids.is_empty(), "generate called with an empty prompt");
        let prefix_len = prefix_len.min(prompt_ids.len());
        let reuse = prefix_len > 0
            && prefix_len == self.cached_prefix.len()
            && prompt_ids[..prefix_len] == self.cached_prefix[..];

        let t_prefill = std::time::Instant::now();
        let suffix_start = if reuse {
            self.cache.restore(prefix_len);
            prefix_len
        } else {
            self.cache.restore(0);
            0
        };
        let suffix = &prompt_ids[suffix_start..];
        let hidden = self.model.forward_hidden(suffix, &mut self.cache)?;
        let last = hidden.narrow(1, suffix.len() - 1, 1);
        let logits = self.model.lm_head(last);
        let logits_vec = llm_wasm::model::logits_to_vec(logits)?;
        self.last_prefill = t_prefill.elapsed();

        if !reuse {
            self.cached_prefix = prompt_ids[..prefix_len].to_vec();
        }

        let t_decode = std::time::Instant::now();
        let (ids, stats) =
            self.model
                .decode_with_constraint(logits_vec, max_new_tokens, stop_ids, &mut self.cache, constraint)?;
        self.last_decode = t_decode.elapsed();

        Ok(llm_wasm::agent::GenerateOutput {
            ids,
            model_steps: stats.model_steps,
            forced_tokens: stats.forced_tokens,
        })
    }
}

const DEFAULT_GGUF_SUFFIX: &str =
    "Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf";
const DEFAULT_MODEL_DIR_SUFFIX: &str = "Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r";

fn home_relative(suffix: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(format!("{home}/{suffix}"))
}

fn git_commit_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn machine_line() -> String {
    let kernel = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!("Apple M2, 16 GB unified memory, macOS (Darwin {kernel})")
}

fn today() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown-date".to_string())
}

/// Parse a `tools/list`-shaped JSON array (`{"name", "description",
/// "inputSchema"}` per entry — MCP shape, same as `web.rs`'s `parse_tools`
/// and `eval.rs`'s `load_mcp_tools`, duplicated here since this bin takes
/// an arbitrary `--tools` file rather than one of the two fixed fixture
/// files `eval.rs`'s loader is built around) and apply `--tool-order`.
fn load_tools_generic(path: &std::path::Path, tool_order: &str) -> anyhow::Result<Vec<Tool>> {
    let raw: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading --tools {path:?}: {e}"))?,
    )?;
    let mut tools: Vec<Tool> = raw
        .into_iter()
        .map(|t| {
            let name = t["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tool entry missing string `name`: {t}"))?;
            let description = t["description"].as_str().unwrap_or("");
            Ok(Tool::from_mcp(name, description, t["inputSchema"].clone()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    match tool_order {
        "as-is" => {}
        "listing-first" => {
            if let Some(pos) = tools
                .iter()
                .position(|t| t.function.name == "get_households_and_groups_and_players")
            {
                let listing = tools.remove(pos);
                tools.insert(0, listing);
            }
        }
        other => anyhow::bail!("--tool-order must be `listing-first` or `as-is`, got `{other}`"),
    }
    Ok(tools)
}

#[allow(clippy::too_many_arguments)]
fn kv_export(
    gguf: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    tools_path: &std::path::Path,
    system: &str,
    dtype_arg: &str,
    out_dir: Option<PathBuf>,
    tool_order: &str,
) -> anyhow::Result<()> {
    let gguf_path = gguf.unwrap_or_else(|| home_relative(DEFAULT_GGUF_SUFFIX));
    let model_dir = model_dir.unwrap_or_else(|| home_relative(DEFAULT_MODEL_DIR_SUFFIX));
    let out_dir = out_dir.unwrap_or_else(|| home_relative("Code/idle-intelligence/models/kv"));
    std::fs::create_dir_all(&out_dir).map_err(|e| anyhow::anyhow!("creating --out-dir {out_dir:?}: {e}"))?;

    let dtype = match dtype_arg {
        "f32" => Dtype::F32,
        "q8_0" => Dtype::Q8_0,
        other => anyhow::bail!("--dtype must be `f32` or `q8_0`, got `{other}`"),
    };

    let t0 = std::time::Instant::now();
    let tools = load_tools_generic(tools_path, tool_order)?;
    println!("tools: {} ({tool_order})", tools.len());

    // -- load model --
    let device = WgpuDevice::default();
    let t_load = std::time::Instant::now();
    let file = File::open(&gguf_path).map_err(|e| anyhow::anyhow!("opening --gguf {gguf_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;

    // Model-identity fingerprint: header bytes (magic through the
    // tensor-info table — a few KB) + file size, NOT a hash of the full
    // 1.7GB+ file (see kvimg.rs's "Hashing" module docs) — cheap enough to
    // recompute on every run, so no on-disk cache is needed.
    let file_len = loader.reader().file_len();
    let header_bytes = loader.reader_mut().header_bytes()?;
    let model_fingerprint = kvimg::gguf_header_fingerprint(file_len, &header_bytes);
    eprintln!("model fingerprint: {model_fingerprint} ({:.1}s)", t0.elapsed().as_secs_f32());

    let parts = loader.load_deferred(&device)?;
    drop(loader);
    let model = parts.finalize(&device)?;
    eprintln!("model load: {:.2}s", t_load.elapsed().as_secs_f32());

    // -- load tokenizer + chat template --
    let tok_path = model_dir.join("tokenizer.json");
    let cfg_path = model_dir.join("tokenizer_config.json");
    let tokenizer = Tokenizer::from_json(&std::fs::read(&tok_path)?)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {tok_path:?}: {e}"))?;
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let template = ChatTemplate::from_tokenizer_config(&cfg)?;

    // -- render the system+tools prefix: common leading tokens between two
    // content-free probe utterances, same technique `web.rs`'s
    // `compute_prefix_len`/`run_eval`'s `prefix_tokens` use, just with two
    // probes instead of a real utterance + one probe (kv-export has no
    // utterance of its own — the whole point is a prefix that's the same
    // regardless of the user turn that follows it). --
    let render = |u: &str| -> anyhow::Result<Vec<u32>> {
        let messages = vec![Message::system(system), Message::user(u)];
        let prompt = template.render_prompt(&messages, &tools, true)?;
        tokenizer.encode(&prompt, false).map_err(|e| anyhow::anyhow!("{e}"))
    };
    let a = render("kv-export-probe-alpha")?;
    let b = render("totally-different-probe-beta")?;
    let prefix_len = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    anyhow::ensure!(
        prefix_len > 0,
        "empty common prefix — system+tools rendering produced no shared leading tokens"
    );
    let prefix_tokens = a[..prefix_len].to_vec();
    let prefix_text = tokenizer
        .decode(&prefix_tokens, false)
        .map_err(|e| anyhow::anyhow!("decoding prefix tokens: {e}"))?;
    let prefix_key = kvimg::prefix_key(&model_fingerprint, &prefix_text);

    // -- prefill natively --
    let mut cache = model.new_cache(prefix_tokens.len());
    let t_prefill = std::time::Instant::now();
    let _hidden = model.forward_hidden(&prefix_tokens, &mut cache)?;
    let prefill_s = t_prefill.elapsed().as_secs_f32();
    let layers = cache.export_prefix(prefix_tokens.len());

    let header = Header {
        model_fingerprint: model_fingerprint.clone(),
        prefix_key: prefix_key.clone(),
        tokens: prefix_tokens.clone(),
        n_layers: model.config().num_layers,
        n_kv_heads: model.config().num_kv_heads,
        head_dim: model.config().hidden_size / model.config().num_heads,
        dtype: dtype.as_str().to_string(),
        engine: format!("llm-wasm/{}", env!("CARGO_PKG_VERSION")),
        created: today(),
    };

    let mut buf = Vec::new();
    let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    KvImage::write(&mut buf, &header, dtype, layer_refs)?;

    let out_path = out_dir.join(format!("{prefix_key}.kvimg"));
    std::fs::write(&out_path, &buf)?;
    let sidecar_path = out_dir.join(format!("{prefix_key}.json"));
    std::fs::write(&sidecar_path, serde_json::to_string_pretty(&header)?)?;

    let total_s = t0.elapsed().as_secs_f32();
    println!("prefix_key: {prefix_key}");
    println!("tokens: {}", prefix_tokens.len());
    println!(
        "size: {} bytes ({:.1} MB), dtype={}",
        buf.len(),
        buf.len() as f64 / 1e6,
        dtype.as_str()
    );
    println!("prefill: {prefill_s:.2}s, total: {total_s:.2}s");
    println!("wrote: {}", out_path.display());
    println!("wrote: {}", sidecar_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval(
    gguf: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    tools_arg: &str,
    tool_order_arg: &str,
    cases_path: &std::path::Path,
    fixtures_dir: &std::path::Path,
    out: Option<PathBuf>,
    only: Option<String>,
    max_new_tokens: usize,
    max_steps: usize,
    max_ctx: usize,
    system: &str,
    label: &str,
    constrained: bool,
) -> anyhow::Result<()> {
    let gguf_path = gguf.unwrap_or_else(|| home_relative(DEFAULT_GGUF_SUFFIX));
    let model_dir = model_dir.unwrap_or_else(|| home_relative(DEFAULT_MODEL_DIR_SUFFIX));

    let tool_set = match tools_arg {
        "all" => ToolSet::All,
        "12" => ToolSet::Twelve,
        other => anyhow::bail!("--tools must be `all` or `12`, got `{other}`"),
    };
    let tool_order = match tool_order_arg {
        "alphabetical" => ToolOrder::Alphabetical,
        "listing-first" => ToolOrder::ListingFirst,
        other => anyhow::bail!("--tool-order must be `alphabetical` or `listing-first`, got `{other}`"),
    };
    // Folded into the label so it lands in both the filename and the
    // report's `label=...` parameters-block line (see docs/ENGINE.md
    // "Known issues / fixed").
    let label = format!("{label}-{tool_order}");
    let label = label.as_str();

    let date = today();
    let out_path = out.unwrap_or_else(|| {
        let tag = match tool_set {
            ToolSet::All => "all",
            ToolSet::Twelve => "12",
        };
        PathBuf::from(format!("eval/results/{date}-{tag}-{label}-native.md"))
    });

    // -- load model --
    let device = WgpuDevice::default();
    let t0 = std::time::Instant::now();
    let file = File::open(&gguf_path)
        .map_err(|e| anyhow::anyhow!("opening --gguf {gguf_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let mut loader = Q4ModelLoader::new(reader)?;
    let parts = loader.load_deferred(&device)?;
    drop(loader);
    let model = parts.finalize(&device)?;
    eprintln!("model load: {:.2}s", t0.elapsed().as_secs_f32());

    // -- load tokenizer + chat template --
    let tok_path = model_dir.join("tokenizer.json");
    let cfg_path = model_dir.join("tokenizer_config.json");
    let tokenizer = Tokenizer::from_json(&std::fs::read(&tok_path)?)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {tok_path:?}: {e}"))?;
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let template = ChatTemplate::from_tokenizer_config(&cfg)?;

    // -- load tools + cases --
    let all_tools = eval::load_all_tools(fixtures_dir)?;
    let tools = eval::select_tools_ordered(&all_tools, tool_set, tool_order, fixtures_dir)?;
    let mut cases = eval::load_cases(cases_path)?;
    if let Some(only) = &only {
        let ids: std::collections::HashSet<&str> = only.split(',').map(str::trim).collect();
        cases.retain(|c| ids.contains(c.id.as_str()));
    }

    let results_dir = fixtures_dir.join("results");
    let generator = NativeGenerator::new(model, max_ctx);
    let caller = FixtureCaller::new(&results_dir);

    // Common system+tools prefix token length, for the report's parameters
    // block — mirrors `Agent::prefix_len_for` (agent.rs), computed here
    // before `template`/`tokenizer` move into `Agent::new`.
    let first_utterance = cases.first().map(|c| c.utterance.as_str()).unwrap_or("");
    let prefix_tokens = {
        let render = |utterance: &str| -> anyhow::Result<Vec<u32>> {
            let messages = vec![
                llm_wasm::template::Message::system(system),
                llm_wasm::template::Message::user(utterance),
            ];
            let prompt = template.render_prompt(&messages, &tools, true)?;
            tokenizer.encode(&prompt, false)
        };
        let a = render(first_utterance)?;
        let b = render("\u{0}prefix-cache-probe\u{0}")?;
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    };

    let mut agent = Agent::new(template, tokenizer, generator, caller, system, max_new_tokens);
    agent.set_max_steps(max_steps);
    agent.set_constrained(constrained);

    // -- run, printing progress as it goes --
    let mut results = Vec::with_capacity(cases.len());
    for case in &cases {
        if tool_set == ToolSet::Twelve && !case.tools12_ok {
            println!("{}: skipped (tools12_ok=false)", case.id);
            results.push(skipped_case_result(case));
            continue;
        }
        let mut caller = FixtureCaller::new(&results_dir);
        let result = eval::run_case(&mut agent, &tools, case, &mut caller);
        let decode_s = result.decode_ms_total / 1000.0;
        let decode_tok_s = if decode_s > 0.0 {
            result.tokens_generated as f64 / decode_s
        } else {
            0.0
        };
        let calls: Vec<String> = result
            .calls_made
            .iter()
            .map(|c| format!("{}({})", c.name, c.arguments))
            .collect();
        let share = eval::forced_share(result.forced_tokens, result.tokens_generated);
        println!(
            "{}: correct={} steps={} prefill_s={:.3} decode_tok_s={:.2} total_s={:.3} tokens_gen={} forced_tokens={} forced_share={:.2} model_steps={} retries={} tool_errors={} calls=[{}]",
            result.id,
            result.correct,
            result.steps,
            result.prefill_ms_total / 1000.0,
            decode_tok_s,
            result.total_ms / 1000.0,
            result.tokens_generated,
            result.forced_tokens,
            share,
            result.model_steps,
            result.retries,
            result.tool_errors,
            calls.join(", "),
        );
        results.push(result);
    }

    let report = aggregate_report(
        results,
        tool_set,
        &gguf_path,
        &date,
        label,
        system,
        max_new_tokens,
        max_steps,
        prefix_tokens,
    );
    let mut markdown = eval::render_markdown(&report);

    let commit = git_commit_hash();
    let machine = machine_line();
    markdown.push_str(&format!(
        "\n- commit: {commit}\n- machine: {machine}\n- gguf: {}\n- model-dir: {}\n",
        gguf_path.display(),
        model_dir.display(),
    ));

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, &markdown)?;
    println!("wrote {}", out_path.display());

    Ok(())
}

fn skipped_case_result(case: &eval::EvalCase) -> eval::CaseResult {
    // Mirrors `eval.rs`'s private `skipped_result` (not `pub`), duplicated
    // here since `run_eval` prints progress per-case and needs the same
    // skip record `eval::run_all` would produce internally.
    eval::CaseResult {
        id: case.id.clone(),
        correct: false,
        skipped: true,
        reason: "tools12_ok=false: not solvable under the 12-tool subset".to_string(),
        steps: 0,
        calls_made: Vec::new(),
        prefill_ms_total: 0.0,
        decode_ms_total: 0.0,
        tokens_generated: 0,
        total_ms: 0.0,
        forced_tokens: 0,
        model_steps: 0,
        retries: 0,
        tool_errors: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn aggregate_report(
    results: Vec<eval::CaseResult>,
    tool_set: ToolSet,
    gguf_path: &std::path::Path,
    date: &str,
    label: &str,
    system_prompt: &str,
    max_new_tokens: usize,
    max_steps: usize,
    prefix_tokens: usize,
) -> eval::EvalReport {
    let scored: Vec<&eval::CaseResult> = results.iter().filter(|r| !r.skipped).collect();
    let n = scored.len().max(1) as f64;
    let correct_pct = scored.iter().filter(|r| r.correct).count() as f64 / n * 100.0;
    let mean_steps = scored.iter().map(|r| r.steps as f64).sum::<f64>() / n;
    let mean_prefill_s = scored.iter().map(|r| r.prefill_ms_total / 1000.0).sum::<f64>() / n;
    let mean_total_s = scored.iter().map(|r| r.total_ms / 1000.0).sum::<f64>() / n;
    let decode_tok_s_values: Vec<f64> = scored
        .iter()
        .filter_map(|r| {
            let decode_s = r.decode_ms_total / 1000.0;
            (decode_s > 0.0).then(|| r.tokens_generated as f64 / decode_s)
        })
        .collect();
    let mean_decode_tok_s = if decode_tok_s_values.is_empty() {
        0.0
    } else {
        decode_tok_s_values.iter().sum::<f64>() / decode_tok_s_values.len() as f64
    };
    let mean_model_steps = scored.iter().map(|r| r.model_steps as f64).sum::<f64>() / n;
    let mean_forced_share = scored
        .iter()
        .map(|r| eval::forced_share(r.forced_tokens, r.tokens_generated))
        .sum::<f64>()
        / n;
    let total_retries = scored.iter().map(|r| r.retries).sum();
    let total_tool_errors = scored.iter().map(|r| r.tool_errors).sum();

    let model = gguf_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| gguf_path.display().to_string());

    eval::EvalReport {
        model,
        tool_set,
        backend: "native".to_string(),
        date: date.to_string(),
        results,
        correct_pct,
        mean_steps,
        mean_prefill_s,
        mean_decode_tok_s,
        mean_total_s,
        mean_model_steps,
        mean_forced_share,
        total_retries,
        total_tool_errors,
        label: label.to_string(),
        system_prompt: system_prompt.to_string(),
        max_new_tokens,
        max_steps,
        prefix_tokens,
    }
}
