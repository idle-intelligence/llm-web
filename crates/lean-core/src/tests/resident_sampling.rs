//! The GPU-resident sampler (`model::sample_gpu`: penalty, top-k tree, draw)
//! against its CPU reference (`resident_sampling::sample_reference`) on
//! random logits, with ties, at three vocabulary sizes (one, two and three
//! levels of the top-k tree). Skips when no adapter is available.

use crate::engine::Engine;
use crate::model::{sample_gpu, Pool, SampleStep};
use crate::resident_sampling::{sample_reference, GpuSampling};

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

#[test]
fn gpu_sampler_matches_reference() {
    let Ok(engine) = Engine::new() else {
        eprintln!("no wgpu adapter, skipping");
        return;
    };
    let pool = Pool::new(engine.device.clone(), engine.queue.clone());
    let mut st = 4242u64;
    let (mut mismatches, mut total) = (0, 0);
    for case in 0..240u64 {
        let vocab = [900usize, 3000, 49152, 151936][(case % 4) as usize];
        let coarse = case % 3 == 0;
        // Every fifth case also carries masked logits (-3.4e38), as a
        // constrained decode leaves them.
        let masked = case % 5 == 1;
        let logits: Vec<f32> = (0..vocab)
            .map(|_| {
                let r = (lcg(&mut st) % 24001) as f32 / 1000.0 - 12.0;
                if masked && lcg(&mut st) % 10 < 7 {
                    -3.4e38
                } else if coarse {
                    (r * 2.0).round() / 2.0
                } else {
                    r
                }
            })
            .collect();
        let mut counts = vec![0u32; vocab];
        for _ in 0..(lcg(&mut st) % 300) {
            counts[(lcg(&mut st) % vocab as u64) as usize] += 1 + (lcg(&mut st) % 3) as u32;
        }
        let p = GpuSampling {
            temperature: [0.3f32, 0.7, 1.0, 1.7][(lcg(&mut st) % 4) as usize],
            top_k: [1u32, 2, 10, 40, 63][(lcg(&mut st) % 5) as usize],
            top_p: [1.0f32, 0.9, 0.5, 0.05][(lcg(&mut st) % 4) as usize],
            repetition_penalty: [1.0f32, 1.1, 1.3][(lcg(&mut st) % 3) as usize],
            seed: lcg(&mut st) ^ (lcg(&mut st) << 32),
        };
        let position = (lcg(&mut st) % 400) as u32;
        let want = sample_reference(&logits, &counts, &p, position);

        // A pooled buffer, like the decode step's logits: the pool's cached
        // bind groups would keep pointing at a replaced one.
        let logits_buf = pool.data("test_logits", vocab);
        engine.queue.write_buffer(&logits_buf, 0, bytemuck::cast_slice(&logits));
        let counts_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_counts"),
            size: 4 * vocab as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        engine.queue.write_buffer(&counts_buf, 0, bytemuck::cast_slice(&counts));
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
        let mut pass = engine.begin_pass(&mut encoder, "test");
        let step = SampleStep { counts: &counts_buf, sampling: &p, position };
        let out = sample_gpu(&engine, &pool, &mut pass, &logits_buf, vocab as u32, &step);
        drop(pass);
        engine.queue.submit(Some(encoder.finish()));
        let got = pollster::block_on(engine.read_u32(&out));
        let after = pollster::block_on(engine.read_buffer(&counts_buf, vocab));
        assert_eq!(after[got as usize].to_bits(), counts[got as usize] + 1, "the chosen id's count is incremented");
        total += 1;
        if got != want {
            mismatches += 1;
            eprintln!("case {case} vocab {vocab} {p:?} position {position}: gpu {got} reference {want}");
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} of {total}");
}
