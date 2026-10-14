//! Sampling over a decode step's logits. `sample` works on the full logits
//! vector (`vocab_size * 4` bytes read back per step); for `0 < top_k < 64`
//! the top-k runs on the GPU instead and `sample_candidates` finishes the
//! step from the 64 best logits plus the repetition-penalty history's
//! logits (`generate.rs::decode_loop`), returning the same token. Greedy
//! decoding (`SamplingParams::is_greedy`) takes `forward_decode_step_argmax`
//! and never gets here.
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

/// Samples one token id from `logits` (mutated in place: penalty and
/// temperature scaling happen in place). `history` is every token id fed
/// into the KV cache so far (prompt + generated), used only by the
/// repetition penalty, which is applied once per occurrence. Order
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
        for &id in history {
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

/// Largest `top_k` the GPU top-k path serves: it returns
/// `lean_core::model::TOPK_CANDS` candidates and needs `top_k` below that so
/// a tie with the k-th value shows up as a tie inside the candidates.
pub(crate) const GPU_TOPK_MAX: u32 = lean_core::model::TOPK_CANDS as u32 - 1;

impl SamplingParams {
    /// Whether a sampling step can run its top-k on the GPU and read back
    /// only the candidates: `0 < top_k <= GPU_TOPK_MAX` (top_k below the
    /// vocabulary size is checked by the caller).
    pub(crate) fn gpu_topk(&self) -> bool {
        !self.is_greedy() && self.top_k > 0 && self.top_k <= GPU_TOPK_MAX
    }
}

/// The repetition-penalty history as distinct ids with their occurrence
/// counts, in first-occurrence order. The penalty is applied once per
/// occurrence, so a count is how many times to apply it to that id.
pub(crate) struct HistoryCounts {
    vocab: usize,
    slot: std::collections::HashMap<u32, usize>,
    pub ids: Vec<u32>,
    pub counts: Vec<u32>,
}

impl HistoryCounts {
    pub fn new(vocab: usize, history: &[u32]) -> Self {
        let mut h = HistoryCounts { vocab, slot: Default::default(), ids: Vec::new(), counts: Vec::new() };
        for &id in history {
            h.push(id);
        }
        h
    }

    /// Ids outside the vocabulary are ignored, as `sample` ignores them.
    pub fn push(&mut self, id: u32) {
        if id as usize >= self.vocab {
            return;
        }
        match self.slot.get(&id) {
            Some(&i) => self.counts[i] += 1,
            None => {
                self.slot.insert(id, self.ids.len());
                self.ids.push(id);
                self.counts.push(1);
            }
        }
    }
}

/// What the GPU returns for one sampling step (see
/// `lean_core::model::TopkStep`), as plain data.
pub(crate) struct Candidates<'a> {
    /// Best logits among the ids not in the history, best first.
    pub cands: &'a [(u32, f32)],
    /// Raw logit of each id in `HistoryCounts::ids`, same order.
    pub hist_logits: &'a [f32],
}

/// `sample` for a step whose top-k was selected on the GPU. Gives the token
/// `sample` gives on the full logits, or `None` when the candidates cannot
/// prove they hold every logit `sample` would keep (the caller then reads
/// the full logits and calls `sample`). The penalty, temperature scaling,
/// softmax, top-p and the draw all run here on copies of the GPU's values,
/// in the same order and arithmetic as `sample`, so the result does not
/// depend on how the GPU rounds. `params.top_k` must satisfy `gpu_topk` and
/// be below `vocab`.
///
/// The k-th largest scaled logit is taken over the candidates plus the
/// penalised history values. A logit outside the candidates is at most the
/// last candidate's, so it is below that k-th value, hence correctly
/// dropped, unless the two are equal (a tie with the boundary that the
/// candidates cannot see) and the tied entries have a non-zero softmax
/// weight; that case returns `None`. When fewer than `TOPK_CANDS` real
/// candidates came back, every non-history logit is among them.
pub(crate) fn sample_candidates(c: &Candidates, hist: &HistoryCounts, vocab: usize, params: &SamplingParams, rng: &mut Rng64) -> Option<u32> {
    let temp = params.temperature.max(1e-5);
    let mut merged: Vec<(u32, f32)> = Vec::with_capacity(c.cands.len() + hist.ids.len());
    for ((&id, &count), &raw) in hist.ids.iter().zip(&hist.counts).zip(c.hist_logits) {
        let mut v = raw;
        if params.repetition_penalty != 1.0 {
            for _ in 0..count {
                v = penalize(v, params.repetition_penalty);
            }
        }
        merged.push((id, v / temp));
    }
    for &(id, raw) in c.cands {
        merged.push((id, raw / temp));
    }
    let k = params.top_k as usize;
    let values: Vec<f32> = merged.iter().map(|m| m.1).collect();
    let threshold = kth_largest(&values, k);
    let complete = c.cands.len() < lean_core::model::TOPK_CANDS;
    if !complete {
        let last = c.cands[c.cands.len() - 1].1 / temp;
        if threshold <= last {
            let max_logit = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if (threshold - max_logit).exp() > 0.0 {
                return None;
            }
        }
    }
    let mut surv: Vec<(u32, f32)> = merged.into_iter().filter(|m| m.1 >= threshold).collect();
    surv.sort_by_key(|m| m.0);
    Some(draw_from(&mut surv, vocab, params.top_p, rng))
}

/// One application of the repetition penalty to one logit.
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

    /// The GPU's selection, on the CPU: the best 64 non-history logits
    /// (value descending, lower id first), plus the history's raw logits.
    fn gpu_view(base: &[f32], hist: &HistoryCounts) -> (Vec<(u32, f32)>, Vec<f32>) {
        let mut rest: Vec<(u32, f32)> = base.iter().enumerate().filter(|(i, _)| !hist.ids.contains(&(*i as u32))).map(|(i, &v)| (i as u32, v)).collect();
        rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        rest.truncate(lean_core::model::TOPK_CANDS);
        (rest, hist.ids.iter().map(|&i| base[i as usize]).collect())
    }

    /// `sample_candidates` returns the token `sample` returns on the full
    /// logits whenever it returns one, and returns one in the common case.
    #[test]
    fn candidates_match_full_sample_on_random_logits_and_ties() {
        let mut st = 777u64;
        let (mut some, mut none) = (0, 0);
        for case in 0..600u64 {
            let vocab = [70usize, 100, 300, 2000, 5000][(lcg(&mut st) % 5) as usize];
            let coarse = case % 4 == 0;
            let base: Vec<f32> = (0..vocab)
                .map(|_| {
                    let r = (lcg(&mut st) % 20001) as f32 / 1000.0 - 10.0;
                    if coarse { (r * 2.0).round() / 2.0 } else { r }
                })
                .collect();
            let history: Vec<u32> = (0..(lcg(&mut st) % 80)).map(|_| (lcg(&mut st) % (vocab as u64 + 3)) as u32).collect();
            let params = SamplingParams {
                temperature: [0.3f32, 0.7, 1.0, 1.7][(lcg(&mut st) % 4) as usize],
                top_k: [1u32, 2, 5, 40, 63][(lcg(&mut st) % 5) as usize],
                top_p: [1.0f32, 0.9, 0.5][(lcg(&mut st) % 3) as usize],
                repetition_penalty: [1.0f32, 1.1, 1.3][(lcg(&mut st) % 3) as usize],
                seed: lcg(&mut st),
            };
            let hc = HistoryCounts::new(vocab, &history);
            let (cands, hist_logits) = gpu_view(&base, &hc);
            let mut rng_a = Rng64::new(params.seed);
            let mut rng_b = Rng64::new(params.seed);
            for step in 0..5 {
                let mut full = base.clone();
                let want = sample(&mut full, &params, &history, &mut rng_a);
                match sample_candidates(&Candidates { cands: &cands, hist_logits: &hist_logits }, &hc, vocab, &params, &mut rng_b) {
                    Some(got) => {
                        some += 1;
                        assert_eq!(want, got, "case {case} step {step} vocab {vocab} params {params:?}");
                    }
                    None => {
                        none += 1;
                        // The caller falls back to the full sample, which draws once.
                        let mut again = base.clone();
                        let fb = sample(&mut again, &params, &history, &mut rng_b);
                        assert_eq!(want, fb);
                    }
                }
            }
        }
        assert!(some > 2000, "the candidate path should serve most steps: {some} served, {none} fell back");
    }
}
