// Dumps the CPU prefill's last-row logits and four greedy decode steps'
// logits, as raw f32 little-endian, for several prompt lengths.
// Usage: cpu_last_logits <gguf> <out file> [lengths, comma list]
use std::io::Write;
use std::time::Instant;

use lean_core::cpu::{forward_decode_step, forward_prefill, CpuKvCache, CpuModel};

fn argmax(l: &[f32]) -> u32 {
    let mut b = 0;
    for i in 1..l.len() {
        if l[i] > l[b] {
            b = i;
        }
    }
    b as u32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = CpuModel::load(&args[1]).unwrap();
    let mut out = std::fs::File::create(&args[2]).unwrap();
    let lens: Vec<usize> = args.get(3).map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 3, 5, 36, 256]);
    let vocab = model.config.vocab_size as u32;
    for n in lens {
        let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % vocab.min(30000)).collect();
        let mut cache = CpuKvCache::new(&model.config, n + 8);
        let t = Instant::now();
        let mut logits = forward_prefill(&model, &mut cache, &ids);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut toks = vec![];
        for _ in 0..5 {
            for v in &logits {
                out.write_all(&v.to_le_bytes()).unwrap();
            }
            let id = argmax(&logits);
            toks.push(id);
            logits = forward_decode_step(&model, &mut cache, id);
        }
        println!("len {n} prefill_ms {ms:.1} tokens {toks:?}");
    }
}
