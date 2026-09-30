//! CPU-side sampling over a decode step's full logits vector. Reads back
//! `vocab_size * 4` bytes per step (the same readback `forward_decode_step`
//! already does for the mask/logit-inspection callers - see its doc comment
//! in `model.rs`) rather than adding a GPU sampling kernel: sampling only
//! runs when a caller asks for non-greedy decoding, and the existing
//! `forward_decode_step`/GPU-argmax split already gives every caller a
//! cheap way to stay on the fast (`forward_decode_step_argmax`, 4-byte
//! readback) path for the common greedy case. `SamplingParams::is_greedy`
//! is the switch every call site (`generate.rs::decode_loop`) uses to pick
//! between the two readback paths - greedy always takes
//! `forward_decode_step_argmax`, so it stays bit-identical to the
//! pre-existing greedy-only decode loop.
//!
//! `Rng64` is a small splitmix64 generator (no external `rand` dependency -
//! this crate has none, and a PRNG this simple needs none) used only for
//! reproducible sampling given a seed; it is not cryptographically strong
//! and is not meant to be.

/// temperature <= 0.0 means greedy (argmax, matching `model::argmax`'s
/// lowest-index tie-break). `top_k == 0` disables top-k filtering.
/// `top_p >= 1.0` disables nucleus filtering. `repetition_penalty == 1.0`
/// disables the repetition penalty.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0 }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// splitmix64 - fixed, public-domain construction (Steele/Lea/Flood 2014).
/// Used only to turn a `u64` seed into a stream of pseudo-random `u64`s.
pub struct Rng64(u64);

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Rng64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        // 24 bits of precision is enough for a sampling draw (f32 mantissa
        // is 24 bits including the implicit leading one).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Ties broken toward the lower index - matches `model::argmax`/`web.rs`'s
/// `argmax` helper, so `SamplingParams::default()` (greedy) callers that
/// happen to go through this function instead of the GPU-argmax fast path
/// (native unit tests do, to keep the comparison self-contained) see
/// identical output.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Samples one token id from `logits` (mutated in place: penalty/temperature
/// scaling and top-k/top-p masking all happen in place before softmax).
/// `history` is every token id fed into the KV cache so far (prompt +
/// generated), used only by the repetition penalty. Order (matching HF
/// `transformers`' default `LogitsProcessorList` construction order):
/// repetition penalty -> temperature -> top-k -> top-p -> categorical draw.
pub fn sample(logits: &mut [f32], params: &SamplingParams, history: &[u32], rng: &mut Rng64) -> u32 {
    if params.is_greedy() {
        return argmax(logits);
    }

    if params.repetition_penalty != 1.0 {
        for &id in history {
            if let Some(v) = logits.get_mut(id as usize) {
                *v = if *v > 0.0 { *v / params.repetition_penalty } else { *v * params.repetition_penalty };
            }
        }
    }

    let temp = params.temperature.max(1e-5);
    for v in logits.iter_mut() {
        *v /= temp;
    }

    if params.top_k > 0 && (params.top_k as usize) < logits.len() {
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[params.top_k as usize - 1];
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax over whatever survived top-k.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if params.top_p < 1.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &idx) in order.iter().enumerate() {
            cum += probs[idx];
            if cum >= params.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> = order[..cutoff].iter().copied().collect();
        let mut kept_sum = 0.0f32;
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            } else {
                kept_sum += *p;
            }
        }
        if kept_sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= kept_sum;
            }
        }
    }

    let draw = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if draw < cum {
            return i as u32;
        }
    }
    // Floating point rounding fallback: last non-zero-probability index.
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_matches_argmax() {
        let mut logits = vec![0.1, 0.9, 0.05, -1.0];
        let params = SamplingParams::default();
        let mut rng = Rng64::new(42);
        assert_eq!(sample(&mut logits, &params, &[], &mut rng), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 7 };
        let base = vec![1.0, 2.0, 0.5, 3.0, 0.1];
        let mut a = base.clone();
        let mut rng_a = Rng64::new(params.seed);
        let out_a = sample(&mut a, &params, &[], &mut rng_a);
        let mut b = base.clone();
        let mut rng_b = Rng64::new(params.seed);
        let out_b = sample(&mut b, &params, &[], &mut rng_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn top_k_restricts_to_k_candidates() {
        let params = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0, seed: 3 };
        let mut logits = vec![0.1, 5.0, 0.2, 0.3];
        let mut rng = Rng64::new(params.seed);
        // top_k=1 must always pick the single highest-logit index, same as
        // greedy, regardless of the random draw.
        assert_eq!(sample(&mut logits, &params, &[], &mut rng), 1);
    }
}
