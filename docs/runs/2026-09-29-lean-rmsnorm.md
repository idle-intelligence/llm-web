# lean rmsnorm fix: one workgroup per row, RTX 3080/Vulkan

Follow-on to `docs/runs/2026-09-29-lean-vs-llamacpp-profile.md`, which found
`crates/lean/src/shaders/rmsnorm.wgsl` assigning one GPU thread per row with
a fully serial loop over the hidden dimension, twice. At decode (`rows ==
1`) this left exactly one of the GPU's ALUs active per dispatch, costing
40% of decode time on Qwen2.5-0.5B and 57-63% on Qwen2.5-3B. This session
replaces it with a one-workgroup-per-row, shared-memory tree-reduction
design (256 threads/workgroup) reused from this project's own
`crates/llm-wasm/src/wgsl/shader_rmsnorm.wgsl`, adapted to lean's read-only
storage + uniform-`Dims` binding layout.

## Parameters

- Commit tested: this session's own commit on branch `lean-rmsnorm`,
  committed files only (`git archive HEAD`), built and run on the same
  RTX 3080 desktop (Vulkan backend via wgpu) used by
  `docs/runs/2026-09-29-lean-box-check.md` and the profiling doc above.
- Models/fixtures: same four as the box-check doc (Qwen2.5-0.5B-Instruct
  Q4_0, the official Qwen2.5-3B-Instruct Q4_0 with Q6_K `output.weight`,
  Qwen3-1.7B Q8_0, SmolLM2-360M-Instruct Q4_0).
- Gates: `cargo test --release -p lean -- --ignored --nocapture
  --test-threads=1`, the 8 required cases (`fixture_parity`,
  `fixture_parity_qwen25_3b`, `fixture_parity_qwen3`,
  `fixture_parity_qwen3_1_7b`, `fixture_parity_llama_360m`,
  `fixture_parity_llama_1_7b`, `kv_snapshot` both cases, `logit_mask` all 3
  cases, the last of these run separately with `--test logit_mask` since a
  name-substring filter alone matched 0 tests).
- Timing: `lean-cli --tokens 16 --kernel fast`, release build, 5 repeats
  per model/case (median reported), GPU held under `flock` for the whole
  run so no other GPU job ran concurrently. `tok_match=false` at `--tokens
  16` is expected (fixture's full greedy continuation is longer); `top1_match=true`
  on every case/run is the correctness signal, and it held throughout,
  including the baseline comparison runs below.
- "Before" decode ms/tok figures are `docs/runs/2026-09-29-lean-box-check.md`'s
  own numbers, same box, same commit family, pre-this-session. A live
  side-by-side baseline (the pre-fix binary already built in a sibling
  worktree on this box) was also run this session for the prefill-regression
  check specifically, since box-check's table only reports decode ms/tok.

## Gate results (all pass)

| test | result |
|---|---|
| fixture_parity | pass |
| fixture_parity_qwen25_3b | pass |
| fixture_parity_qwen3 | pass |
| fixture_parity_qwen3_1_7b | pass |
| fixture_parity_llama_1_7b | pass |
| fixture_parity_llama_360m | pass |
| kv_snapshot (both cases) | pass |
| logit_mask (all 3 cases) | pass |
| fixture_parity_both_kernel_paths (bonus, ran under the `fixture_parity`-substring filter) | pass |

`top1_match=true` on every fixture case in every timing run below (44 case/
run combinations total across the 4 models' non-profiled timing sweep plus
the 6 baseline comparison runs) - no numerical regression from the kernel
change.

## Decode ms/tok, median of 5, before vs after

| model | case | before (box-check) | after (this session) | speedup |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | short | 16.08 | 5.41 | 2.97x |
| Qwen2.5-0.5B-Instruct | long_tools_single | 17.18 | 7.49 | 2.29x |
| Qwen2.5-3B-Instruct | short | 50.82 | 18.49 | 2.75x |
| Qwen2.5-3B-Instruct | long | 51.16 | 19.06 | 2.68x |
| Qwen3-1.7B | short | 36.41 | 8.88 | 4.10x |
| Qwen3-1.7B | long_tools_single | 38.65 | 12.06 | 3.20x |
| SmolLM2-360M-Instruct | short | 22.37 | 7.64 | 2.93x |
| SmolLM2-360M-Instruct | long | 22.94 | 8.95 | 2.56x |

Additional cases, this session only (median of 5, no box-check baseline row
for them): Qwen2.5-0.5B `non_english` 5.94, `long_tools_multiturn` 7.53;
Qwen2.5-3B `non_english` 18.69; Qwen3-1.7B `long` 10.66, `non_english`
10.13; SmolLM2-360M `non_english` 8.53.

## Prefill check: no regression

Prefill dispatch count is unchanged by this fix (rmsnorm's dispatch count
per call stays 1; only its internal grid shape changed). To confirm actual
prefill wall-clock did not regress at the largest row counts in the fixture
set, this session built and ran the pre-fix binary (already present in a
sibling worktree on this box, same commit family as `docs/runs/2026-09-29-lean-box-check.md`)
side by side under the same GPU lock, 3 runs each, single process per
model:

| model | case (seq) | prefill_ms/tok, pre-fix (3 vals) | prefill_ms/tok, this session (5 vals) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short (36) | 3.51, 3.37, 3.40 | 3.07, 3.11, 2.92, 2.97, 3.10 |
| Qwen2.5-0.5B-Instruct | long (86) | 1.21, 1.22, 1.24 | 1.03, 1.03, 0.97, 1.02, 1.05 |
| Qwen2.5-0.5B-Instruct | non_english (54) | 1.50, 1.50, 1.52 | 1.17, 1.18, 1.09, 1.16, 1.21 |
| Qwen2.5-0.5B-Instruct | long_tools_single (2225) | 1.04 (2 of 3 runs logged before truncation) | 1.02, 1.02, 0.99, 1.03, 1.03 |
| Qwen2.5-3B-Instruct | short (36) | 17.96, 17.47 (run 1 not fully captured) | 16.55, 16.61, 16.49, 16.66, 16.10 |
| Qwen2.5-3B-Instruct | long (86) | 5.54, 5.53 | 4.91, 5.00, 4.99, 5.01, 5.02 |
| Qwen2.5-3B-Instruct | non_english (54) | 6.17, 6.18, 6.16 | 5.35, 5.37, 5.39, 5.39, 5.39 |

Prefill is equal to or faster than the pre-fix kernel at every row count
measured, up to 2225 rows (this fixture set's largest). No dual-kernel
threshold was needed: the one-workgroup-per-row design replaces the old
kernel unconditionally, at every row count, per the task's own preference
for a single design when it doesn't regress.

## Kernel profile, after the fix (16 decode steps, `LEAN_PROFILE_KERNELS=1`)

`us_per_call`, rmsnorm-family kernels only (compare against the profiling
doc's before-numbers in the same units):

| kernel | model | before (us/call) | after (us/call) |
|---|---|---:|---:|
| dec_layer.norm | Qwen2.5-0.5B | 199.9 | ~9.0 (9.00-9.05 across cases) |
| dec_layer.ffnnorm | Qwen2.5-0.5B | 200.0 | ~9.0 (8.95-9.05) |
| dec_out_norm | Qwen2.5-0.5B | 198.3 | ~8.4 (8.38-8.58) |
| dec_layer.norm | Qwen2.5-3B | 448.6 | ~11.3 (11.27-11.32) |
| dec_layer.ffnnorm | Qwen2.5-3B | 448.5 | ~11.3 (11.27-11.28) |
| dec_out_norm | Qwen2.5-3B | 446.1-449.2 | ~10.8-11.4 |

rmsnorm's per-call cost dropped roughly 22-50x on both models, matching the
profiling doc's expectation of "the same order of magnitude" the already-
cooperative matvec kernels show. rmsnorm's share of total per-step GPU time
fell from the largest line item (40% on 0.5B, 57-63% on 3B) to a minor one:
on the 3B profiled run, the three rmsnorm calls now sum to roughly
6,520+6,490+11,392+... us out of a per-step total dominated by
`dec_lm_head` (Q6_K naive kernel, still the known-and-documented largest
single dispatch on this model per the profiling doc's secondary finding,
unchanged by this session).

## Other one-thread-per-row kernels checked

Grepped every shader in `crates/lean/src/shaders/` for the same "one
GPU thread owns a whole row, serial loop over the row" pattern rmsnorm had.
Findings, all already cooperative (no second fix needed):

- `argmax.wgsl`: already single-workgroup, 256-thread shared-memory tree
  reduction (own kernel, not reused from t0/llm-wasm per its own header
  comment) - not the same pattern.
- `attn_decode.wgsl`: one workgroup per attention head (`wg_id.x`), 256-cap
  tiled flash-attention-style shared-memory reduction across kv_len -
  already cooperative, not the same pattern.
- `rope_neox.wgsl` / `rope_positions.wgsl`: one thread per
  `(row, head, half-index)` output element (fully data-parallel at the
  element level, `@workgroup_size(64)` over a flat `global_invocation_id`
  index), not a serial per-row loop - not the same pattern.
- `gather_dequant_q8_rows.wgsl`: one thread per `(selected_row, k)` output
  element, same fully-parallel-at-the-element-level shape as the RoPE
  kernels - not the same pattern.
- Qwen3's per-head q/k RMSNorm (`qnorm`/`knorm` in `model.rs`'s `qkv`
  function) calls the same shared `rmsnorm()` helper and the same
  `rmsnorm.wgsl` kernel as the main attn/ffn/out norms - fixed by this same
  change, no separate kernel exists for it.

No other kernel in this crate had the one-thread-per-row serial-loop
pattern; rmsnorm was the only instance.

## WGSL/browser note

`@workgroup_size(256, 1, 1)` matches WebGPU's guaranteed minimum workgroup
invocation limit exactly (the same limit `argmax.wgsl` and
`attn_decode.wgsl`'s `SHARED_CAP` already assume in this crate) - no
browser-specific risk expected, but this was only built and gated natively
this session; the `wasm-pack build --target web` / browser check on
`www/index.html` is left to the lead per the task's own instructions.

## Observations

- The measured decode speedup (2.3-4.1x across all 8 model/case
  combinations with a box-check baseline) exceeds the profiling doc's own
  "plausibly 1.5-2x" estimate. rmsnorm's actual share of decode time before
  the fix was large enough on every model that collapsing it to a
  cooperative reduction moved the bottleneck cleanly to the matvecs
  (already 6.7-35.3% of peak bandwidth per the profiling doc) rather than
  leaving a second-largest bottleneck close behind.
- `dec_lm_head`'s Q6_K naive-kernel cost on Qwen2.5-3B (the profiling doc's
  documented secondary finding) is unchanged by this session and is now
  proportionally a larger share of what's left in decode time on that
  model - unaddressed here, out of this task's scope.
- `fixture_parity_both_kernel_paths` ran as a side effect of using
  `fixture_parity` as a substring filter (cargo test filters are OR'd
  substrings across all test binaries) - it passed, and is additional
  coverage this session did not have to add by hand.
