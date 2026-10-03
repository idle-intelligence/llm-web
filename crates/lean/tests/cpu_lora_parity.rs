//! CPU backend against the GPU backend and against HF transformers + PEFT
//! on llm-life's per-cell call shape: the `a-norules-300` LoRA adapter on
//! q/k/v/o (`CpuModel::apply_lora`), a resident prefix prefilled once and
//! rewound by resetting `kv_len`, 64 per-cell prompts packed block-diagonally
//! per forward (`cpu::forward_chunk_spec` with caller positions and mask),
//! and the `[dead, alive]` logits read through the token-embedding rows
//! (`CpuModel::embed_head_sliced`). All 512 (8 neighbours, self) cases at
//! 64 per forward; every 8th case at one cell per forward (the CPU runs one
//! cell's ~30 tokens in a forward, so all 512 one at a time would hold the
//! GPU lock for minutes).
//!
//! The reference is llm-life's fixture (`hf_peft_reference.json`, written by
//! its `tools/parity/hf_peft_reference.py`: transformers GGUF-loaded base +
//! PEFT, float32, eager attention); it carries the token ids, so this test
//! needs no tokenizer and no llm-life code.
//!
//! ```sh
//! LEAN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
//! LEAN_LORA_BIN=/path/to/lora-a-norules-300.bin \
//! LEAN_HF_PEFT_FIXTURE=/path/to/llm-life/crates/llm-life-lean/tests/fixtures/hf_peft_reference.json \
//! cargo test -p lean --release --features threads -- --ignored --nocapture cpu_lora
//! ```

use lean::cpu::{forward_chunk_spec as cpu_chunk, CpuKvCache, CpuModel};
use lean::engine::Engine;
use lean::model::{build_rope_tables, forward_chunk_spec as gpu_chunk, ForwardSpec, GpuModel, KvCache};
use serde::Deserialize;

const DEAD_ALIVE: [u32; 2] = [15, 16]; // "0", "1" in the Qwen2 tokenizer
const CHUNK: usize = 64;

#[derive(Deserialize)]
struct PerCell {
    name: String,
    prefix_ids: Vec<u32>,
    prompt_ids: Vec<Vec<u32>>,
    logits: Vec<[f32; 2]>,
}

#[derive(Deserialize)]
struct Fixture {
    per_cell: Vec<PerCell>,
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k} to run this test"))
}

/// llm-life's `variant_a::pack_chunk`: blocks back to back, each block's
/// positions restarting at the prefix length, each row attending the whole
/// prefix plus its own block causally. Returns the tokens, the spec and the
/// answer row (last token) of each block.
fn pack(prompts: &[&[u32]], prefix_len: usize) -> (Vec<u32>, ForwardSpec, Vec<usize>) {
    let t: usize = prompts.iter().map(|p| p.len()).sum();
    let kv = prefix_len + t;
    let (mut tokens, mut positions, mut answers) = (Vec::with_capacity(t), Vec::with_capacity(t), Vec::new());
    let mut allowed = vec![false; t * kv];
    for p in prompts {
        let start = tokens.len();
        for i in 0..p.len() {
            let row = (start + i) * kv;
            allowed[row..row + prefix_len].fill(true);
            allowed[row + prefix_len + start..row + prefix_len + start + i + 1].fill(true);
            positions.push((prefix_len + i) as u32);
        }
        tokens.extend_from_slice(p);
        answers.push(start + p.len() - 1);
    }
    (tokens, ForwardSpec::default().with_positions(positions).with_allowed(&allowed, t, kv), answers)
}

/// (answers that differ, max |logit diff|).
fn compare(got: &[[f32; 2]], want: &[[f32; 2]]) -> (usize, f32) {
    let mut differ = 0;
    let mut max = 0f32;
    for (g, w) in got.iter().zip(want) {
        differ += ((g[1] > g[0]) != (w[1] > w[0])) as usize;
        max = max.max((g[0] - w[0]).abs()).max((g[1] - w[1]).abs());
    }
    (differ, max)
}

fn truth(k: usize) -> bool {
    let n = (0..8).filter(|b| (k >> b) & 1 != 0).count();
    if (k >> 8) & 1 != 0 { n == 2 || n == 3 } else { n == 3 }
}

#[test]
#[ignore = "needs LEAN_GGUF, LEAN_LORA_BIN and LEAN_HF_PEFT_FIXTURE on disk; never committed to this repo"]
fn cpu_lora_matches_gpu_and_hf_peft() {
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(env("LEAN_HF_PEFT_FIXTURE")).unwrap()).unwrap();
    let cfg = fixture.per_cell.iter().find(|c| c.name == "a-norules-300").expect("a-norules-300 in the fixture");
    let lora = std::fs::read(env("LEAN_LORA_BIN")).unwrap();
    let gguf = env("LEAN_GGUF");
    let p = cfg.prefix_ids.len();
    let cell_max = cfg.prompt_ids.iter().map(|x| x.len()).max().unwrap();
    let max_ctx = p + CHUNK * cell_max;

    let mut cpu = CpuModel::load(&gguf).expect("load CPU model");
    cpu.apply_lora(&lora).expect("CPU apply_lora");
    assert!(cpu.has_lora());
    let mut cpu_cache = CpuKvCache::new(&cpu.config, max_ctx);
    let _ = cpu_chunk(&cpu, &mut cpu_cache, &cfg.prefix_ids, &ForwardSpec::default());
    assert_eq!(cpu_cache.kv_len, p);

    let engine = Engine::new().expect("wgpu engine init");
    let mut gpu = GpuModel::load(&engine, &gguf, true).expect("load GPU model");
    gpu.apply_lora(&engine, &lora).expect("GPU apply_lora");
    let (cos, sin) = build_rope_tables(gpu.config.head_dim, gpu.config.rope_theta, max_ctx);
    let (cos, sin) = (engine.buf_f32(&cos, "rope_cos"), engine.buf_f32(&sin, "rope_sin"));
    let mut gpu_cache = KvCache::new(&engine, &gpu.config, max_ctx as u32);
    let _ = pollster::block_on(gpu_chunk(&engine, &gpu, &mut gpu_cache, &cfg.prefix_ids, &cos, &sin, &ForwardSpec::default()));

    // 64 cells per forward, all 512 cases, both backends.
    let (mut cpu_batched, mut gpu_batched) = (Vec::new(), Vec::new());
    let t0 = std::time::Instant::now();
    for chunk in cfg.prompt_ids.chunks(CHUNK) {
        let prompts: Vec<&[u32]> = chunk.iter().map(|x| &x[..]).collect();
        let (tokens, spec, answers) = pack(&prompts, p);
        cpu_cache.kv_len = p;
        let h = cpu_chunk(&cpu, &mut cpu_cache, &tokens, &spec);
        let l = cpu.embed_head_sliced(&h, tokens.len(), &DEAD_ALIVE);
        cpu_batched.extend(answers.iter().map(|&r| [l[r * 2], l[r * 2 + 1]]));
    }
    let cpu_batched_s = t0.elapsed().as_secs_f64();
    for chunk in cfg.prompt_ids.chunks(CHUNK) {
        let prompts: Vec<&[u32]> = chunk.iter().map(|x| &x[..]).collect();
        let (tokens, spec, answers) = pack(&prompts, p);
        gpu_cache.kv_len = p as u32;
        let h = pollster::block_on(gpu_chunk(&engine, &gpu, &mut gpu_cache, &tokens, &cos, &sin, &spec));
        let l = pollster::block_on(gpu.embed_head_sliced(&engine, &h, tokens.len() as u32, &DEAD_ALIVE));
        gpu_batched.extend(answers.iter().map(|&r| [l[r * 2], l[r * 2 + 1]]));
    }

    // One cell per forward against the rewound prefix, every 8th case (CPU).
    let sampled: Vec<usize> = (0..512).step_by(8).collect();
    let t1 = std::time::Instant::now();
    let cpu_single: Vec<[f32; 2]> = sampled
        .iter()
        .map(|&k| {
            let ids = &cfg.prompt_ids[k];
            cpu_cache.kv_len = p;
            let h = cpu_chunk(&cpu, &mut cpu_cache, ids, &ForwardSpec::default());
            let l = cpu.embed_head_sliced(&h, ids.len(), &DEAD_ALIVE);
            let r = ids.len() - 1;
            [l[r * 2], l[r * 2 + 1]]
        })
        .collect();
    let cpu_single_s = t1.elapsed().as_secs_f64();
    let pick = |v: &[[f32; 2]]| -> Vec<[f32; 2]> { sampled.iter().map(|&k| v[k]).collect() };
    let (want_sampled, cpu_batched_sampled) = (pick(&cfg.logits), pick(&cpu_batched));

    let mut failed = false;
    for (label, got, want) in [
        ("CPU 64/forward vs HF+PEFT", &cpu_batched, &cfg.logits),
        ("CPU 1/forward  vs HF+PEFT", &cpu_single, &want_sampled),
        ("GPU 64/forward vs HF+PEFT", &gpu_batched, &cfg.logits),
        ("CPU 64/forward vs GPU", &cpu_batched, &gpu_batched),
        ("CPU 1/forward  vs CPU 64/forward", &cpu_single, &cpu_batched_sampled),
    ] {
        let (differ, max) = compare(got, want);
        println!("[cpu_lora] {label:<33} {:>3} cases: {differ} answers differ, max |logit diff| {max:.3e}", got.len());
        failed |= differ != 0;
    }
    let correct = cpu_batched.iter().enumerate().filter(|(k, l)| (l[1] > l[0]) == truth(*k)).count();
    let correct_single = cpu_single.iter().zip(&sampled).filter(|(l, &k)| (l[1] > l[0]) == truth(k)).count();
    println!("[cpu_lora] CPU correct vs the rule: {correct}/512 (64/forward), {correct_single}/{} (1/forward)", sampled.len());
    println!(
        "[cpu_lora] CPU time (not a benchmark: shared machine): 64/forward {:.1} s ({:.1} ms/cell), 1/forward {:.1} s ({:.1} ms/cell), prefix {p} tokens",
        cpu_batched_s,
        cpu_batched_s * 1e3 / 512.0,
        cpu_single_s,
        cpu_single_s * 1e3 / sampled.len() as f64
    );
    assert!(!failed, "answers differ (see above)");
    assert_eq!(correct, 512);
    assert_eq!(correct_single, sampled.len());

    // Clearing the adapter returns the base model's numbers (one chunk,
    // both backends, from a fresh prefix prefill each).
    cpu.clear_lora();
    gpu.clear_lora();
    assert!(!cpu.has_lora());
    let prompts: Vec<&[u32]> = cfg.prompt_ids[..CHUNK].iter().map(|x| &x[..]).collect();
    let (tokens, spec, answers) = pack(&prompts, p);
    cpu_cache.kv_len = 0;
    let _ = cpu_chunk(&cpu, &mut cpu_cache, &cfg.prefix_ids, &ForwardSpec::default());
    let h = cpu_chunk(&cpu, &mut cpu_cache, &tokens, &spec);
    let lc = cpu.embed_head_sliced(&h, tokens.len(), &DEAD_ALIVE);
    gpu_cache.kv_len = 0;
    let _ = pollster::block_on(gpu_chunk(&engine, &gpu, &mut gpu_cache, &cfg.prefix_ids, &cos, &sin, &ForwardSpec::default()));
    let h = pollster::block_on(gpu_chunk(&engine, &gpu, &mut gpu_cache, &tokens, &cos, &sin, &spec));
    let lg = pollster::block_on(gpu.embed_head_sliced(&engine, &h, tokens.len() as u32, &DEAD_ALIVE));
    let base_cpu: Vec<[f32; 2]> = answers.iter().map(|&r| [lc[r * 2], lc[r * 2 + 1]]).collect();
    let base_gpu: Vec<[f32; 2]> = answers.iter().map(|&r| [lg[r * 2], lg[r * 2 + 1]]).collect();
    let (differ, max) = compare(&base_cpu, &base_gpu);
    let (_, moved) = compare(&base_cpu, &cpu_batched[..CHUNK]);
    println!("[cpu_lora] after clear_lora, CPU vs GPU base model, 64 cases: {differ} answers differ, max |logit diff| {max:.3e} (adapter moved logits by up to {moved:.3})");
    assert_eq!(differ, 0);
    assert!(max < 1e-2, "CPU and GPU base models diverge after clear_lora: {max}");
    assert!(moved > 0.1, "clear_lora left the adapter in place");
}
