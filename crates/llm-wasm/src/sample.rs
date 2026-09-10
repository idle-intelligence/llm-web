//! Token sampling on CPU-resident logits (after a GPU->CPU readback of the
//! last position's `[vocab]` row — see `model.rs::lm_head` callers). Greedy
//! is what `tests/full_forward.rs` and `generate()` use for reference-match
//! testing (deterministic); temperature/top-p/top-k/repetition-penalty match
//! xLAM-2-3b-fc-r's `generation_config.json` defaults (docs/MODELS.md §1:
//! `temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05`) for
//! actual agent use, not exercised by the reference-comparison tests.

/// Greedy decode: argmax over `logits`.
pub fn greedy(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Top-K argmax indices with their (unmodified) logit values, descending.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // NaN-safe descending sort: map NaN to -inf for the comparison only (so
    // a stray NaN logit sorts last, never wins top-k) and use `total_cmp`
    // for the rest so the comparator is a total order (never panics).
    idx.sort_unstable_by(|&a, &b| {
        let ka = if logits[a].is_nan() { f32::NEG_INFINITY } else { logits[a] };
        let kb = if logits[b].is_nan() { f32::NEG_INFINITY } else { logits[b] };
        kb.total_cmp(&ka)
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i as u32, logits[i])).collect()
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        // xLAM-2-3b-fc-r generation_config.json defaults, docs/MODELS.md §1.
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repetition_penalty: 1.05,
        }
    }
}

/// Small deterministic xorshift64* PRNG — avoids pulling in the `rand`
/// crate for what's otherwise a single `next_f32() in [0,1)` call per step.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Apply repetition penalty (HF convention: divide positive logits, multiply
/// negative ones, for each id present in `recent_tokens`), temperature
/// scaling, top-k truncation, then nucleus (top-p) sampling.
pub fn sample(
    logits: &[f32],
    recent_tokens: &[u32],
    cfg: &SamplingConfig,
    rng: &mut Rng,
) -> u32 {
    let mut scaled: Vec<f32> = logits.to_vec();

    if cfg.repetition_penalty != 1.0 {
        for &id in recent_tokens {
            let v = &mut scaled[id as usize];
            *v = if *v > 0.0 {
                *v / cfg.repetition_penalty
            } else {
                *v * cfg.repetition_penalty
            };
        }
    }

    if cfg.temperature > 0.0 && cfg.temperature != 1.0 {
        for v in scaled.iter_mut() {
            *v /= cfg.temperature;
        }
    }

    let k = cfg.top_k.min(scaled.len()).max(1);
    let mut candidates = top_k(&scaled, k);

    // Softmax over the top-k candidates.
    let max_v = candidates.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = candidates.iter().map(|&(_, v)| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Nucleus (top-p): keep the smallest prefix whose cumulative prob >= top_p.
    if cfg.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= cfg.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
        probs.truncate(cutoff);
        let renorm: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= renorm;
        }
    }

    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return candidates[i].0;
        }
    }
    candidates.last().map(|&(id, _)| id).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_ignores_nan_and_is_deterministic() {
        // NaN comparisons are always false, so `greedy`'s strict `>` never
        // selects a NaN and the first (not last) true maximum wins.
        let logits = [1.0, f32::NAN, 3.0, 3.0, f32::NAN, 2.0];
        assert_eq!(greedy(&logits), 2);
    }

    #[test]
    fn top_k_sorts_nan_without_panicking() {
        let logits = [1.0, f32::NAN, 3.0, -f32::NAN, 2.0];
        let top = top_k(&logits, 3);
        assert_eq!(top.len(), 3);
        // The two non-NaN finite values that matter should be ranked first.
        assert_eq!(top[0].0, 2);
        assert_eq!(top[1].0, 4);
        assert_eq!(top[2].0, 0);
    }
}
