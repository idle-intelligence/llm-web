# lean box check: test suite and decode timing on a second client

Follow-on to `docs/runs/2026-09-28-lean-decode-breakdown.md` (Session 4,
argmax fold + QKV fusion). That session's changes were only verified for
correctness and timed on the Apple M2 (Metal). This run repeats both the
full gate suite and the decode timing sweep on a different client - a Linux
desktop with an RTX 3080 (10GB), Vulkan backend via wgpu - to check the
changes hold outside the M2.

## Parameters

- Commit tested: `376144ace9448f1ffe73098afbaa068ee3b3992e` ("docs: session
  4 - argmax fold and QKV fusion, measured before/after"), committed files
  only (`git archive HEAD`).
- Models: Qwen2.5-0.5B-Instruct (`qwen2.5-0.5b-instruct-q4_0.gguf`),
  Qwen2.5-3B-Instruct official (`qwen2.5-3b-instruct-q4_0.gguf`, Q6_K
  `output.weight`), Qwen3-1.7B (`Qwen3-1.7B-Q8_0.gguf`), SmolLM2-360M-Instruct
  (`SmolLM2-360M-Instruct-Q4_0.gguf`).
- `lean-cli --tokens 16 --kernel fast`, release build, GPU confirmed idle
  (0% utilization, 462MiB/10240MiB used) and no other GPU job running
  immediately before both the test suite and the timing sweep.
- Timing: 5 repeats per model (one process per model, all fixture cases run
  per process), median of 5 taken per case. `tok_match=false` at `--tokens
  16` is expected (the fixture's full greedy continuation is 32 tokens;
  `top1_match=true` on every case/run is the correctness signal actually
  checked here, matching this project's own documented convention) and is
  not evidence of a bug.

## Test suite (all pass, 0 failures)

Non-ignored (`cargo test --release -p lean`): 13 passed, 0 failed
(`cpu_kernels`, `lora`, `chat_template`, `quant`, `model::grid1d_tests` unit
tests).

Ignored fixture/gate tests (`cargo test --release -p lean -- --ignored
--test-threads=1`), 1 passed each, 0 failed:

| test | result | time |
|---|---|---:|
| fixture_parity | pass | 18.45s |
| fixture_parity_llama_1_7b | pass | 9.10s |
| fixture_parity_llama_360m | pass | 6.40s |
| fixture_parity_qwen25_3b | pass | 16.24s |
| fixture_parity_qwen3 | pass | 13.99s |
| fixture_parity_qwen3_1_7b | pass | 24.85s |
| kv_snapshot (both cases) | pass | 38.10s |
| logit_mask (all 3 cases) | pass | 4.63s |

Nothing failed on this client that passes on the Mac. Every required test
also ran markedly faster in wall-clock terms than the equivalent Mac runs
recorded in `docs/runs/2026-09-28-lean-decode-breakdown.md` (e.g.
`fixture_parity_qwen3_1_7b` 24.85s here vs 314.53s there,
`fixture_parity_qwen25_3b` 16.24s here vs 79.94s there) - consistent with
this being a discrete desktop GPU rather than a laptop's integrated/shared
one, not itself a code-correctness finding.

## Decode ms/tok, median of 5, vs. earlier box numbers

Earlier box numbers are from `docs/runs/2026-09-28-lean-perf-2.md` (Sessions
3-4, same RTX 3080/Vulkan machine, pre-Session-4-QKV-fusion commits) and
`docs/runs/2026-09-28-lean-q6k.md`. `long_tools_single` does not exist in
the Qwen2.5-3B or SmolLM2-360M fixtures, so `long` is used for those two
models, matching the earlier box docs' own convention for models without
that case.

| model | case (prompt tokens) | this run: decode ms/tok (5 vals) | median | decode dispatches/step | earlier box ms/tok | change |
|---|---|---|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | short (36) | 16.11, 15.92, 16.09, 16.00, 16.08 | 16.08 | 292 | 23.6-25.1 | faster |
| Qwen2.5-0.5B-Instruct | long_tools_single (2225) | 17.13, 17.18, 17.13, 17.23, 17.27 | 17.18 | 316 | 24.68-24.86 | faster |
| Qwen2.5-3B-Instruct | short (36) | 50.75, 50.88, 51.03, 50.78, 50.82 | 50.82 | 436 | 82.28 | faster |
| Qwen2.5-3B-Instruct | long (86) | 51.16, 51.26, 51.37, 51.09, 51.16 | 51.16 | 436 | 77.50 | faster |
| Qwen3-1.7B | short (15) | 36.41, 36.30, 34.57, 36.43, 36.67 | 36.41 | 396 | 49.8-59.0 | faster |
| Qwen3-1.7B | long_tools_single (2225) | 38.63, 38.65, 38.77, 37.75, 38.91 | 38.65 | 424 | 54.89 | faster |
| SmolLM2-360M-Instruct | short (37) | 21.85, 22.37, 21.94, 22.40, 22.48 | 22.37 | 388 | not previously measured on this client | n/a |
| SmolLM2-360M-Instruct | long (86) | 22.97, 22.90, 22.92, 23.01, 22.94 | 22.94 | 388 | not previously measured on this client | n/a |

Additional cases captured this run, no earlier-box comparison recorded for
them: Qwen2.5-0.5B `non_english` (54 tok) median 15.96, `long_tools_multiturn`
(2354 tok) median 17.17; Qwen2.5-3B `non_english` (54 tok) median 50.85;
Qwen3-1.7B `long` (65 tok) median 36.74, `non_english` (33 tok) median
36.54; SmolLM2-360M `non_english` (63 tok) median 22.62.

## Observations

- Every model/case measured this run is faster than the earlier box
  baseline, by 25-40% depending on model. This is the opposite of the Mac
  finding in `docs/runs/2026-09-28-lean-decode-breakdown.md`'s Session 4,
  which measured the same two commits (argmax fold, QKV fusion) as a
  wall-clock *regression* on 3 of 4 models on the M2 (Qwen2.5-3B +32.6%,
  Qwen3-1.7B +29.3%, SmolLM2-360M +19.7% at their respective worst cases),
  despite an identical dispatch-count reduction on both clients. On this
  client the same dispatch-count reduction (e.g. Qwen2.5-0.5B: 292
  dispatches/step here, matching the Mac doc's own post-fusion count)
  correlates with a real decode-time win instead of a regression.
- Qwen2.5-3B shows the largest improvement in absolute terms: 82.28/77.50
  ms/tok (earlier box, pre-fusion) down to ~50.8-51.2 ms/tok now - roughly
  a 35-38% drop, the opposite direction from this same commit range's
  measured +32.6% Mac regression on the same model.
- Decode dispatch counts (292/316 for Qwen2.5-0.5B, 436 for Qwen2.5-3B, 396/424
  for Qwen3-1.7B, 388 for SmolLM2-360M) were identical across all 5 repeats
  per model/case, as expected (deterministic dispatch count, no
  timing-driven kernel choice).
- SmolLM2-360M has no prior box measurement to compare against in the
  existing run docs (it was only benchmarked on the Mac previously), so its
  row here is a first box baseline, not a before/after comparison.
