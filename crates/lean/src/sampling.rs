//! Sampling over a decode step's logits. `sample` works on the full logits
//! vector (`vocab_size * 4` bytes read back per step); it serves `top_k` 0
//! and 64 and up. For `1 <= top_k <= 63` the whole step runs in
//! `lean_core::resident_sampling` instead (`SamplingParams::resident`), on
//! the GPU or the CPU. Greedy decoding (`SamplingParams::is_greedy`) takes
//! `forward_decode_step_argmax` and never gets here.
//!
//! `Rng64` is a small splitmix64 generator (no external `rand` dependency -
//! this crate has none, and a PRNG this simple needs none) used only for
//! reproducible sampling given a seed; it is not cryptographically strong
//! and is not meant to be.

use lean_core::resident_sampling::GpuSampling;

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

/// Samples one token id from `logits` (mutated in place: penalty and
/// temperature scaling happen in place). `history` is every token id fed
/// into the KV cache so far (prompt + generated), used only by the
/// repetition penalty, which is applied once to each distinct id of it, as
/// HF `transformers`' `RepetitionPenaltyLogitsProcessor` does. Order
/// (matching HF `transformers`' default `LogitsProcessorList` construction
/// order): repetition penalty -> temperature -> top-k -> top-p ->
/// categorical draw.
///
/// Top-k keeps every logit `>= ` the k-th largest scaled logit, so ties at
/// the boundary keep more than k entries. The k-th largest is found by
/// linear-time selection and everything after it (softmax, top-p, the
/// draw) runs over the survivors only, in vocabulary order.
pub fn sample(logits: &mut [f32], params: &SamplingParams, history: &[u32], rng: &mut Rng64) -> u32 {
    if params.is_greedy() {
        return argmax(logits);
    }

    if params.repetition_penalty != 1.0 {
        let mut ids = history.to_vec();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            if let Some(v) = logits.get_mut(id as usize) {
                *v = penalize(*v, params.repetition_penalty);
            }
        }
    }

    let temp = params.temperature.max(1e-5);
    for v in logits.iter_mut() {
        *v /= temp;
    }

    let vocab = logits.len();
    let mut cands: Vec<(u32, f32)>;
    if params.top_k > 0 && (params.top_k as usize) < vocab {
        let threshold = kth_largest(logits, params.top_k as usize);
        cands = logits.iter().enumerate().filter(|(_, &v)| v >= threshold).map(|(i, &v)| (i as u32, v)).collect();
    } else {
        cands = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    }
    draw_from(&mut cands, vocab, params.top_p, rng)
}

impl SamplingParams {
    /// The settings of the resident sampler (`lean_core::resident_sampling`,
    /// used on every backend) when these parameters qualify: non-greedy with
    /// `1 <= top_k <= 63` and `top_k` below the vocabulary size. Its random
    /// stream and its tie rule at the k-th logit differ from `sample`'s.
    pub(crate) fn resident(&self, vocab: usize) -> Option<GpuSampling> {
        let k = self.top_k;
        if self.is_greedy() || k == 0 || k as usize > MAX_RESIDENT_TOP_K || k as usize >= vocab {
            return None;
        }
        Some(GpuSampling { temperature: self.temperature, top_k: k, top_p: self.top_p, repetition_penalty: self.repetition_penalty, seed: self.seed })
    }
}

const MAX_RESIDENT_TOP_K: usize = lean_core::model::TOPK_CANDS - 1;

/// Occurrences of each vocabulary id in `history` (ids outside the
/// vocabulary are ignored).
pub(crate) fn id_counts(vocab: usize, history: &[u32]) -> Vec<u32> {
    let mut counts = vec![0u32; vocab];
    for &id in history {
        if let Some(c) = counts.get_mut(id as usize) {
            *c += 1;
        }
    }
    counts
}

/// The repetition penalty of one logit.
pub(crate) fn penalize(v: f32, penalty: f32) -> f32 {
    if v > 0.0 {
        v / penalty
    } else {
        v * penalty
    }
}

/// The `k`-th largest value (1-based) of `values`, by selection.
pub(crate) fn kth_largest(values: &[f32], k: usize) -> f32 {
    let mut tmp: Vec<f32> = values.to_vec();
    let (_, kth, _) = tmp.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap());
    *kth
}

/// Softmax, top-p and the categorical draw over `cands`, which are the
/// (id, temperature-scaled logit) survivors of top-k in ascending id order.
/// `vocab` is the full vocabulary size (the rounding fallback returns its
/// last id). Sums run in id order, so the result does not depend on how
/// the survivors were found.
pub(crate) fn draw_from(cands: &mut [(u32, f32)], vocab: usize, top_p: f32, rng: &mut Rng64) -> u32 {
    let max_logit = cands.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = cands.iter().map(|&(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if top_p < 1.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &idx) in order.iter().enumerate() {
            cum += probs[idx];
            if cum >= top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let mut keep = vec![false; probs.len()];
        for &i in &order[..cutoff] {
            keep[i] = true;
        }
        let mut kept_sum = 0.0f32;
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
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
            return cands[i].0;
        }
    }
    // Floating point rounding fallback: the last vocabulary id.
    (vocab - 1) as u32
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

    /// The full-sort implementation `sample` replaced, kept as the oracle.
    mod oracle {
        use super::super::*;
        pub fn sample_old(logits: &mut [f32], params: &SamplingParams, history: &[u32], rng: &mut Rng64) -> u32 {
            if params.is_greedy() {
                return argmax(logits);
            }

            if params.repetition_penalty != 1.0 {
                let mut ids = history.to_vec();
                ids.sort_unstable();
                ids.dedup();
                for id in ids {
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
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// Random logits, some on a coarse grid so that many values tie, with
    /// large and small vocabularies and k at, below and above the vocabulary.
    #[test]
    fn matches_full_sort_oracle_on_random_logits_and_ties() {
        let mut st = 12345u64;
        for case in 0..400u64 {
            let vocab = [5usize, 17, 64, 300, 2000][(lcg(&mut st) % 5) as usize];
            let coarse = case % 3 == 0;
            let base: Vec<f32> = (0..vocab)
                .map(|_| {
                    let r = (lcg(&mut st) % 20001) as f32 / 1000.0 - 10.0;
                    if coarse { (r * 2.0).round() / 2.0 } else { r }
                })
                .collect();
            let history: Vec<u32> = (0..(lcg(&mut st) % 40)).map(|_| (lcg(&mut st) % (vocab as u64 + 3)) as u32).collect();
            let params = SamplingParams {
                temperature: [0.3f32, 0.7, 1.0, 1.7][(lcg(&mut st) % 4) as usize],
                top_k: [0u32, 1, 2, 5, 40, 64, 5000][(lcg(&mut st) % 7) as usize],
                top_p: [1.0f32, 0.9, 0.5, 0.05][(lcg(&mut st) % 4) as usize],
                repetition_penalty: [1.0f32, 1.1, 1.3][(lcg(&mut st) % 3) as usize],
                seed: lcg(&mut st),
            };
            let mut rng_a = Rng64::new(params.seed);
            let mut rng_b = Rng64::new(params.seed);
            for step in 0..6 {
                let mut a = base.clone();
                let mut b = base.clone();
                let ta = oracle::sample_old(&mut a, &params, &history, &mut rng_a);
                let tb = sample(&mut b, &params, &history, &mut rng_b);
                assert_eq!(ta, tb, "case {case} step {step} vocab {vocab} params {params:?}");
            }
        }
    }
}
