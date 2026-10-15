//! Sampling that runs entirely on the GPU (`model::decode_sampled_pipelined`)
//! and its CPU reference. The reference follows the shaders operation for
//! operation (`sample_penalty.wgsl`, `topk_stage.wgsl`, `sample_final.wgsl`):
//! multiplications by host-computed reciprocals instead of divisions, a
//! division-free draw, a polynomial `exp`, and a counter-based generator
//! keyed by (seed, position), so a token depends only on the logits, the
//! history, the seed and its position. It is the first-token sampler of a
//! resident decode and the oracle of the kernel tests.
//!
//! The semantics differ from `lean::sampling::sample` (the full-logits CPU
//! sampler) in two ways: the random stream is this module's, and ties at the
//! k-th logit are cut at exactly k survivors (lower id first) instead of
//! all being kept.

/// Sampling settings of a resident decode. `top_k` is in `1..=63`; the
/// repetition penalty counts every occurrence of an id in the history.
#[derive(Debug, Clone, Copy)]
pub struct GpuSampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl GpuSampling {
    pub fn inv_temp(&self) -> f32 {
        1.0 / self.temperature.max(1e-5)
    }

    pub fn inv_penalty(&self) -> f32 {
        1.0 / self.repetition_penalty
    }
}

fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// The draw in `[0, 1)` for the token at `position`.
pub fn uniform_draw(seed: u64, position: u32) -> f32 {
    let mut h = pcg((seed as u32) ^ 0x9e3779b9);
    h = pcg(h ^ (seed >> 32) as u32);
    h = pcg(h ^ position);
    h = pcg(h ^ 0x85ebca6b);
    (h >> 8) as f32 * (1.0 / 16777216.0)
}

/// `exp(x)` for `x <= 0`, same operations as the shader.
pub fn exp_det(x: f32) -> f32 {
    if x < -80.0 {
        return 0.0;
    }
    let t = x * std::f32::consts::LOG2_E;
    let n = t.floor();
    let f = t - n;
    let mut p = 1.525273e-5_f32;
    for c in [1.540353e-4_f32, 1.333355e-3, 9.618129e-3, 5.550411e-2, 2.402265e-1, 6.931472e-1, 1.0] {
        p = p * f + c;
    }
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// The whole step on the CPU: penalty (once per occurrence in `counts`, a
/// count per vocabulary id), top-k, temperature, softmax, top-p, draw.
pub fn sample_reference(logits: &[f32], counts: &[u32], p: &GpuSampling, position: u32) -> u32 {
    let inv_penalty = p.inv_penalty();
    let mut cands: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut v = v;
            if p.repetition_penalty != 1.0 {
                for _ in 0..counts.get(i).copied().unwrap_or(0) {
                    v = if v > 0.0 { v * inv_penalty } else { (v * p.repetition_penalty).max(-3.4e38) };
                }
            }
            (v, i as u32)
        })
        .collect();
    let order = |a: &(f32, u32), b: &(f32, u32)| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1));
    let k = p.top_k as usize;
    cands.select_nth_unstable_by(k - 1, order);
    cands.truncate(k);
    cands.sort_by_key(|c| c.1);
    let inv_temp = p.inv_temp();
    let raw: Vec<f32> = cands.iter().map(|c| c.0.max(-1.0e30) * inv_temp).collect();
    let m = raw.iter().cloned().fold(raw[0], f32::max);
    let ws: Vec<f32> = raw.iter().map(|&s| exp_det(s - m)).collect();
    let sum = ws.iter().fold(0.0f32, |a, &w| a + w);
    let mut keep = vec![true; k];
    let mut kept_sum = sum;
    if p.top_p < 1.0 {
        keep = vec![false; k];
        let target = p.top_p * sum;
        let mut cum = 0.0f32;
        for _ in 0..k {
            let mut best = usize::MAX;
            for i in 0..k {
                if !keep[i] && (best == usize::MAX || ws[i] > ws[best]) {
                    best = i;
                }
            }
            keep[best] = true;
            cum += ws[best];
            if cum >= target {
                break;
            }
        }
        kept_sum = (0..k).filter(|&i| keep[i]).fold(0.0f32, |a, i| a + ws[i]);
    }
    let target = uniform_draw(p.seed, position) * kept_sum;
    let mut cum = 0.0f32;
    let mut last_kept = cands[0].1;
    for i in 0..k {
        if keep[i] {
            last_kept = cands[i].1;
            cum += ws[i];
            if target < cum {
                return cands[i].1;
            }
        }
    }
    last_kept
}
