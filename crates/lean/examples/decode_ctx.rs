//! Native greedy decode speed against context length. For each context length
//! the model prefills that many prompt tokens (the `short` fixture prompt
//! repeated), then decodes `--decode` tokens with the pipelined greedy loop and
//! prints ms per token, tok/s and a hash of the generated ids. The first
//! `--logit-steps` single decode steps of each length can be written to
//! `--dump FILE` (f32 logits) or compared against such a file with `--cmp FILE`
//! (prints the largest absolute difference).
//!
//! `LEAN_GGUF=... cargo run --release -p lean --example decode_ctx -- [--ctx 128,512,1024,2048] [--decode 64] [--reps 3] [--logit-steps 4] [--dump F | --cmp F]`

use lean_core::engine::Engine;
use lean_core::model::{argmax, build_rope_tables, decode_greedy_pipelined, forward_decode_step, forward_prefill, GpuModel, KvCache};

const PROMPT: [u32; 36] = [
    151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ctxs: Vec<u32> = arg(&args, "--ctx").unwrap_or_else(|| "128,512,1024,2048".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let n_decode: u32 = arg(&args, "--decode").map(|s| s.parse().unwrap()).unwrap_or(64);
    let reps: usize = arg(&args, "--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let logit_steps: usize = arg(&args, "--logit-steps").map(|s| s.parse().unwrap()).unwrap_or(4);
    let dump = arg(&args, "--dump");
    let cmp = arg(&args, "--cmp");
    let gguf = std::env::var("LEAN_GGUF").expect("set LEAN_GGUF");

    let engine = pollster::block_on(Engine::new_async_with(false))?;
    let model = GpuModel::load(&engine, &gguf, true)?;
    let max_ctx = *ctxs.iter().max().unwrap() + n_decode + logit_steps as u32 + 8;
    let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
    let cos_buf = engine.buf_f32(&cos, "rope_cos");
    let sin_buf = engine.buf_f32(&sin, "rope_sin");
    let mut cache = KvCache::new(&engine, &model.config, max_ctx);

    let reference: Option<Vec<u8>> = cmp.as_ref().map(|p| std::fs::read(p).unwrap());
    let mut dumped: Vec<u8> = Vec::new();
    let mut ref_off = 0usize;
    let mut worst = 0f32;

    println!("ctx  ms/token_median  min  max  tok/s_median  ids_hash");
    for &ctx in &ctxs {
        let ids: Vec<u32> = (0..ctx as usize).map(|i| PROMPT[i % PROMPT.len()]).collect();
        let mut per_token = Vec::new();
        let mut hash = 0u64;
        for r in 0..=reps {
            model.pool.reset();
            let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
            let first = argmax(&logits);
            let t = std::time::Instant::now();
            let (out, _) = pollster::block_on(decode_greedy_pipelined(&engine, &model, &mut cache, first, n_decode, false, &cos_buf, &sin_buf, None, |_| true));
            let ms = t.elapsed().as_secs_f64() * 1e3 / out.len() as f64;
            if r > 0 {
                per_token.push(ms);
            }
            hash = out.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
        }
        per_token.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = per_token[per_token.len() / 2];
        println!("{ctx:5} {med:8.2} {:8.2} {:8.2} {:8.1}  {hash:016x}", per_token[0], per_token[per_token.len() - 1], 1e3 / med);

        model.pool.reset();
        let logits = pollster::block_on(forward_prefill(&engine, &model, &mut cache, &ids, &cos_buf, &sin_buf, None));
        let mut next = argmax(&logits);
        for _ in 0..logit_steps {
            let l = pollster::block_on(forward_decode_step(&engine, &model, &mut cache, next, &cos_buf, &sin_buf, None));
            for v in &l {
                dumped.extend_from_slice(&v.to_le_bytes());
            }
            if let Some(r) = &reference {
                for v in &l {
                    let b = f32::from_le_bytes(r[ref_off..ref_off + 4].try_into().unwrap());
                    ref_off += 4;
                    worst = worst.max((v - b).abs());
                }
            }
            next = argmax(&l);
        }
    }
    if let Some(p) = dump {
        std::fs::write(p, &dumped)?;
    }
    if reference.is_some() {
        println!("max |logit diff| vs reference over {} steps per length: {worst:.3e}", logit_steps);
    }
    Ok(())
}
