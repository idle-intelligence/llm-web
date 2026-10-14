//! Shared streaming decode loop, used by `turn::gpu_turn` (what
//! `Lean::generate` runs on the GPU) and this crate's native tests
//! (`tests/streaming_sampling.rs`) - the reason this loop lives in its own
//! plain (non-wasm-gated) module rather than inline in `web.rs`: a
//! `js_sys::Function`/`JsValue` callback can't be constructed or called from
//! a native `cargo test`, so the loop itself takes plain Rust closures
//! (`FnMut(u32)` for the per-token callback, `FnMut() -> bool` for the abort
//! check) and `web.rs` adapts those to/from JS at the wasm boundary.
//!
//! Greedy (`SamplingParams::is_greedy()`) takes the 4-byte-readback fast
//! path, pipelined (`model::decode_greedy_pipelined`: each step is
//! submitted before the previous id is read back), with the same tokens
//! as the step-by-step `forward_decode_step_argmax` loop - see
//! `tests/streaming_sampling.rs::greedy_streaming_matches_whole_reply`.
//! Non-greedy sampling with `0 < top_k < 64` selects the top-k on the GPU
//! (`forward_decode_step_topk`) and reads back only the candidates and the
//! repetition-penalty history's logits; the penalty, temperature, top-p and
//! the draw run on the CPU (`sampling::sample_candidates`) and give the
//! token `sampling::sample` gives on the full logits. Other non-greedy
//! settings (`top_k` 0 or 64 and up) read back the full logits vector every
//! step (`forward_decode_step`) and draw via `sampling::sample`.

use lean_core::engine::Engine;
use lean_core::model::{argmax, decode_greedy_pipelined, forward_decode_step, forward_decode_step_topk, GpuModel, KvCache};
use crate::sampling::{sample, sample_candidates, Candidates, HistoryCounts, Rng64, SamplingParams};

/// Runs the decode loop starting from `first_logits` (the logits already
/// computed at the last prefilled position - the caller has just run
/// `forward_prefill`/`forward_prefill_suffix`), picking each next token
/// (`SamplingParams`-driven), streaming it through `on_token` as soon as
/// it's chosen, and stopping on end-of-sequence, `max_new_tokens`, or
/// `should_stop()` returning true (checked once per step, before the next
/// GPU dispatch - the abort path). `history` is the running list of every
/// token id already fed into `cache` (the caller seeds it with the prompt/
/// conversation-prefix ids); this function both reads it (repetition
/// penalty) and appends each newly generated id to it, so the caller's
/// `history` reflects the full sequence afterward. Returns the generated ids
/// (not including any eos token, matching the original `generate()`'s
/// contract).
#[allow(clippy::too_many_arguments)]
pub async fn decode_loop(
    engine: &Engine,
    model: &GpuModel,
    cache: &mut KvCache,
    first_logits: Vec<f32>,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    mask: Option<&wgpu::Buffer>,
    max_new_tokens: u32,
    params: &SamplingParams,
    history: &mut Vec<u32>,
    mut on_token: impl FnMut(u32),
    mut should_stop: impl FnMut() -> bool,
) -> Vec<u32> {
    if params.is_greedy() {
        let first = argmax(&first_logits);
        return decode_greedy_pipelined(engine, model, cache, first, max_new_tokens, true, cos, sin, mask, |id| {
            if should_stop() {
                return false;
            }
            history.push(id);
            on_token(id);
            true
        })
        .await
        .0;
    }
    let mut rng = Rng64::new(params.seed);
    let vocab = model.config.vocab_size;
    // Top-k on the GPU when it applies: only the candidates come back.
    let gpu_topk = params.gpu_topk() && (params.top_k as usize) < vocab;
    let mut counts = HistoryCounts::new(vocab, history);
    let mut next_id = sample(&mut first_logits.clone(), params, history, &mut rng);

    let mut generated = Vec::with_capacity(max_new_tokens as usize);
    for _ in 0..max_new_tokens {
        if model.config.eos_token_ids.contains(&next_id) || should_stop() {
            break;
        }
        generated.push(next_id);
        history.push(next_id);
        counts.push(next_id);
        on_token(next_id);
        if gpu_topk {
            let step = forward_decode_step_topk(engine, model, cache, next_id, cos, sin, mask, &counts.ids).await;
            let cands = Candidates { cands: &step.cands, hist_logits: &step.hist_logits };
            next_id = match sample_candidates(&cands, &counts, vocab, params, &mut rng) {
                Some(id) => id,
                None => {
                    let mut logits = step.read_logits(engine, model, &counts.ids).await;
                    sample(&mut logits, params, history, &mut rng)
                }
            };
        } else {
            let mut logits = forward_decode_step(engine, model, cache, next_id, cos, sin, mask).await;
            next_id = sample(&mut logits, params, history, &mut rng);
        }
    }
    generated
}
