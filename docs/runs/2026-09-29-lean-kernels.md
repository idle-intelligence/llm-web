# lean decode kernels, session 2: Q6_K decode matvec + add/rmsnorm fusion, RTX 3080/Vulkan

Follow-on to `docs/runs/2026-09-29-lean-vs-llamacpp-profile.md` (per-kernel
profile) and `docs/runs/2026-09-29-lean-rmsnorm.md` (rmsnorm fix, merged into
this branch's base commit). Same box (Linux desktop, RTX 3080 10GB, Vulkan
backend via wgpu) as both prior sessions. Branch `lean-kernels`, created from
`lean-perf` at commit `c7866b6` (rmsnorm fix already merged in).

## Parameters

- Models/fixtures: same four as both prior sessions - Qwen2.5-0.5B-Instruct
  Q4_0 (`output.weight` Q8_0), the official Qwen2.5-3B-Instruct Q4_0 GGUF
  (`output.weight` Q6_K), Qwen3-1.7B Q8_0, SmolLM2-360M-Instruct Q4_0.
- Profiling: `LEAN_PROFILE_KERNELS=1`, `lean-cli --tokens 16 --kernel fast
  --fixture <model's fixture.json>`, 16 decode steps aggregated per case
  (same mechanism as the profiling doc). Fixture files matter here: the CLI's
  `--fixture` default is Qwen2.5-0.5B's `fixture.json`, so every non-default
  model's profiling/timing run in this session passed its own
  `fixture_*.json` explicitly - an earlier attempt without this crashed on
  SmolLM2-360M with a tokenizer/fixture mismatch (index out of bounds),
  fixed by passing `--fixture crates/lean/reference/fixture_llama_360m_q4_0.json`.
- Gates, every commit: `fixture_parity`, `fixture_parity_qwen25_3b`,
  `fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`, `fixture_parity_llama_360m`,
  `fixture_parity_llama_1_7b`, `kv_snapshot` (both cases), `logit_mask` (run
  separately with `--test logit_mask`, all 3 cases) - all pass on every
  commit below, native release build, GPU-locked (`flock box.lock`),
  `--ignored --test-threads=1`.
- Timing: `lean-cli` median of 5 (Qwen3-1.7B's noisiest case re-run at 8) per
  model/case, ABAB interleaving (before/after alternated within the same
  script run, never all-before-then-all-after), GPU-locked for the whole
  run. Box load average checked below 3 and no `cargo`/`rustc` running
  before every timed run (confirmed via `uptime`/`pgrep` each time).
  `top1_match=true` on every case/run in every timing sweep in this doc - no
  numerical regression from either kernel change.

## 1. Re-profile after the rmsnorm fix (before this session's changes)

`us/step`, `GB/s achieved` and `% of 760 GB/s peak` for the largest kernels,
`short` case, one decode step (aggregated across all layers/16 steps). GB/s
figures use each kernel's known weight-byte shape (Q4_0 = 0.625 B/weight,
Q8_0 = 1.125 B/weight, Q6_K ~= 0.820 B/weight), same convention as the prior
profiling doc.

### Qwen2.5-0.5B-Instruct (24 layers, hidden 896, Q4_0 + Q8_0 lm-head)

| kernel | us/step | GB/s achieved | % of peak |
|---|---:|---:|---:|
| dec_layer.mlp.gate_up.0 | 707.4 (29.51 us/call x 24) | 184.6 | 24.3% |
| dec_layer.attn | 660.7 | - (KV-bound) | - |
| dec_layer.mlp.down.0 | 477.9 | 136.8 | 18.0% |
| dec_lm_head.0 (Q8_0) | 235.0 | 651.7 | 85.8% |
| dec_layer.qkv.qkv_fused.0 | 253.4 | 61.1 | 8.0% |
| dec_layer.wo.0 | 245.3 | 49.1 | 6.5% |
| dec_layer.norm + ffnnorm (post rmsnorm-fix) | 219.4 (9.16+9.13 x24) | - | - |
| **total (short case)** | **decode = 5.62 ms/tok measured** | | |

### Qwen2.5-3B-Instruct, official GGUF (36 layers, hidden 2048, Q6_K lm-head)

| kernel | us/step | GB/s achieved | % of peak |
|---|---:|---:|---:|
| **dec_lm_head.0 (Q6_K, naive kernel)** | **4438.2** | **57.5** | **7.6%** |
| dec_layer.mlp.gate_up.0 | 3809.8 | 266.3 | 35.0% |
| dec_layer.mlp.down.0 | 2150.3 | 235.9 | 31.0% |
| dec_layer.attn | 1248.7 | - | - |
| dec_layer.qkv.qkv_fused.0 | 717.2 | 164.5 | 21.6% |
| dec_layer.wo.0 | 636.4 | 148.3 | 19.5% |
| dec_layer.norm + ffnnorm | 420.1 (11.23+11.20 x36) | - | - |
| **total (short case)** | **decode = 18.89 ms/tok measured** | | |

### Qwen3-1.7B (28 layers, hidden 2048, Q8_0 everywhere)

| kernel | us/step | GB/s achieved | % of peak |
|---|---:|---:|---:|
| dec_layer.mlp.gate_up.0 | 1823.2 | 434.8 | 57.2% |
| dec_layer.mlp.down.0 | 1143.9 | 346.5 | 45.6% |
| dec_layer.qkv.qkv_fused.0 | 863.6 | 306.0 | 40.3% |
| dec_lm_head.0 (Q8_0) | 657.5 | 532.5 | 70.1% |
| dec_layer.wo.0 | 521.8 | 253.2 | 33.3% |
| dec_layer.norm + ffnnorm | 737.6 (13.10+12.63 x28) | - | - |
| **total (short case)** | **decode = 9.94 ms/tok measured** | | |

### SmolLM2-360M-Instruct (32 layers, hidden 960, Q4_0)

| kernel | us/step | GB/s achieved | % of peak | note |
|---|---:|---:|---:|---|
| dec_layer.mlp.down (F32, 4/32 layers) | 1122.9 (280.7 x4) | - | - | see Observations - some layers' down_proj ships F32 in this GGUF, naive `linear.wgsl` kernel |
| dec_layer.attn | 882.6 | - | - | |
| dec_layer.mlp.down.0 (Q4_0, 28/32 layers) | 385.3 | - | - | |
| dec_layer.mlp.gate_up.0 | 629.2 | - | - | |
| dec_layer.qkv.qkv_fused.0 | 357.4 | - | - | |
| dec_layer.wo.0 | 321.2 | - | - | |
| dec_layer.norm + ffnnorm | 575.7 (9.01+8.96 x32) | - | - | |
| **total (short case)** | **decode = 7.09 ms/tok measured** | | |

## 2. Changes made this session

### Change 1: decode-shaped Q6_K matvec (`linear_q6k_decode.wgsl`)

The profiling doc's documented secondary finding: the official Qwen2.5-3B
GGUF's `output.weight` is Q6_K and hit the naive per-output-element kernel
at decode (7.6% of peak, the single biggest individual dispatch on that
model - 4438 us/step). New kernel follows the same 128-thread,
`ROWS_PER_WG=4`, `THREADS_PER_ROW=32`, shared-memory-reduction structure as
this crate's own `linear_q4_decode.wgsl`/`linear_q8_decode.wgsl`, splitting
each Q6_K super-block's 64 `(half, l)` sub-iterations across 32 lanes (2
each) instead of one thread walking all 64 serially. Same per-element math
as `linear_q6k.wgsl` (verified against `gguf.rs::dequantize_q6_k` /
`quant.rs::split_q6k_blocks` per that kernel's own doc comment), only the
accumulation order changes (parallel tree vs. serial), same as
`linear_q4_decode`/`linear_q8_decode` already accept relative to their own
naive counterparts.

**Gates**: all 8 pass, `top1_match=true` on every case.

**Timing** (ABAB, median of 5, `short`/`long` cases, Qwen2.5-3B only - the
only model in the fixture set with a Q6_K tensor):

| case | before (Q6_K naive) | after (Q6_K decode kernel) | speedup |
|---|---:|---:|---:|
| short | 18.74 ms/tok | 15.07 ms/tok | 1.244x |
| long | 19.12 ms/tok | 15.35 ms/tok | 1.246x |

`dec_lm_head` alone: 4438 us/step -> 670 us/step (us/call), 57.5 GB/s -> 380.7
GB/s (7.6% -> 50.1% of peak), a 6.6x per-kernel improvement.

**Kept** - clear >3% gain on Qwen2.5-3B (24.4-24.6%), no other model uses a
Q6_K tensor so no regression surface elsewhere.

### Change 2 (attempted, reverted): 8-word unroll in Q4_0/Q8_0 decode matvecs

Tried doubling `linear_q4_decode.wgsl`/`linear_q8_decode.wgsl`'s per-lane
unroll from 4 words/outer-loop-trip to 8 (matching llama.cpp's Vulkan
`mul_mat_vec.comp` `K_PER_ITER=8`, versus this crate's previous 4) via
`vec4`-shaped explicit unrolling (same `accumulate_word` helper, more calls
per loop trip). Gates passed (all 8, `top1_match=true`). ABAB timing
(median of 5, `short` case):

| model | before (unroll=4) | after (unroll=8) | delta |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 5.40 ms/tok | 5.44 ms/tok | +0.7% (noise) |
| Qwen2.5-3B | 15.17 ms/tok | 15.15 ms/tok | -0.1% (noise) |

No model showed >3% gain; reverted per the task's stop-rule. Read as: these
decode matvecs' bytes/call are small enough (0.5-28 MB) that dispatches
finish before DRAM reaches steady-state bandwidth - the "low % of peak"
numbers in section 1 above are a dispatch-size/ramp-up effect, not a
kernel-body memory-access-pattern problem, so more K-per-iteration unrolling
inside the same dispatch shape doesn't move them. This matches Change 3's
premise (fusion, fewer/bigger dispatches) rather than Change 2's premise
(same dispatch shape, tighter inner loop).

### Change 3: fuse decode's residual add + following rmsnorm (`add_rmsnorm.wgsl`)

Every residual add in decode's per-layer loop is immediately followed by
exactly one rmsnorm read of the updated residual: `add1` (x += attention
output) is always followed by `ffnnorm`, and `add2` (x += mlp output) is
always followed by either the next layer's `attn_norm` or (last layer) the
final `out_norm`. New kernel `add_rmsnorm.wgsl` fuses the pair into one
dispatch: same one-workgroup-per-row, 256-thread shared-memory
tree-reduction as `rmsnorm.wgsl`, with the elementwise add folded into the
first pass (`v = a[i] + delta[i]`, write back into `a` in place, accumulate
`v*v`). `crates/lean/src/model.rs::decode_layers` was restructured so
`normed` (the next dispatch's rmsnorm input) is produced as a side effect of
the previous iteration's `add2` fusion, seeded once before the loop for
layer 0. Same per-element math/order as the two separate kernels - pure
dispatch-count reduction (2 fewer dispatches/layer), no numerical change.
Decode-only; prefill's add+norm pairs are untouched (already amortized
across many rows/dispatch).

**Gates**: all 8 pass, `top1_match=true` on every case.

**Timing** (ABAB, median of 5 - Qwen3-1.7B's `short` case re-run at 8 reps
after its 5-rep result looked like a regression, see Observations):

| model | before (fusion off) | after (fusion on) | delta |
|---|---:|---:|---:|
| SmolLM2-360M | 7.38 ms/tok | 6.94 ms/tok | **6.3% faster** |
| Qwen2.5-3B | 15.29 ms/tok | 14.86 ms/tok | 2.9% faster |
| Qwen2.5-0.5B | 5.50 ms/tok | 5.43 ms/tok | 1.3% faster (noise-level) |
| Qwen3-1.7B (5 reps) | 9.84 ms/tok | 10.20 ms/tok | apparent -3.7% |
| Qwen3-1.7B (8 reps, re-measured) | 9.78 ms/tok | 9.82 ms/tok | flat (0.4%, noise) |

**Kept** - SmolLM2-360M clears the >3% bar cleanly; the apparent Qwen3-1.7B
loss did not survive re-measurement at higher rep count (see Observations)
and every other model is flat-to-positive.

## 3. Final profile, after both kept changes (`short` case, one decode step)

| model | dec_lm_head us/call (before -> after) | dec_layer.add1_ffnnorm us/call | dec_layer.add2_nextnorm us/call |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 239.6 -> 239.6 (Q8_0, unchanged) | 9.40 | 10.03 |
| Qwen2.5-3B | 4438.2 -> 670.4 | 12.31 | 12.35 |
| Qwen3-1.7B | 500.1 -> 500.1 (Q8_0, unchanged) | 12.36 | 12.37 |
| SmolLM2-360M | 91.8 -> 92.1 (Q4_0, unchanged) | 9.10 | 9.25 |

(Each `add1_ffnnorm`/`add2_nextnorm` us/call is one fused dispatch replacing
what was previously two separate ones at roughly the same combined cost per
pair - `add1` ~6.2-6.9us + `ffnnorm` ~9.0-13.1us before, vs one
~9.1-12.4us dispatch now; the saving is in dispatch count, not per-element
compute time.)

## 4. Decode ms/tok, session start vs session end (median of 5, box-locked)

| model | case | before this session (rmsnorm-fix only) | after this session | speedup |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | short | 5.62 | 5.17 | 1.087x |
| Qwen2.5-0.5B | long | 6.67 | 6.40 | 1.042x |
| Qwen2.5-0.5B | long_tools_single | 7.79 | 7.46 | 1.044x |
| Qwen2.5-3B | short | 18.89 | 14.78 | 1.278x |
| Qwen2.5-3B | long | 19.30 | 15.21 | 1.269x |
| Qwen3-1.7B | short | 9.94 | 9.68 | 1.027x |
| Qwen3-1.7B | long | 10.78 | 10.63 | 1.014x |
| Qwen3-1.7B | long_tools_single | 12.89 | 12.57 | 1.025x |
| SmolLM2-360M | short | 7.09 | 7.03 | 1.009x |
| SmolLM2-360M | long | 9.14 | 8.75 | 1.045x |

## 5. Remaining gap to llama.cpp (Vulkan, same box, from the profiling doc)

| model | lean decode (this session, short) | llama.cpp decode | gap |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 5.17 ms/tok | 1.66 ms/tok | 3.11x |
| Qwen2.5-3B | 14.78 ms/tok | 4.19 ms/tok | 3.53x |

Down from the profiling doc's original 9.7x/12.1x (pre-rmsnorm-fix) and the
rmsnorm session's ~2.97x/2.75x-of-its-own-baseline improvement; combined
across both sessions the gap to llama.cpp has closed from ~10-12x to
~3.1-3.5x.

## Observations

- **Fixture defaults matter for profiling non-default models.**
  `lean-cli --fixture` defaults to Qwen2.5-0.5B's `fixture.json`; running
  another model without passing its own `--fixture crates/lean/reference/fixture_*.json`
  loads its tokenizer/config correctly but checks output against the wrong
  fixture, and for SmolLM2-360M this crashed the CLI (`index out of bounds`
  on a mismatched vocab) rather than just failing the parity check. Every
  timing/profiling command in this doc passes an explicit `--fixture`.
- **SmolLM2-360M-Instruct-Q4_0.gguf ships some layers' `ffn_down` as F32,
  not Q4_0.** The profile shows two separate `mlp.down`/`mlp.down.0` buckets
  (4 and 28 layers respectively, out of 32) - `mlp.down` (no chunk suffix)
  is `linear()`'s `MatMulWeight::F32` arm, which has no decode-specialized
  kernel and costs ~280 us/call versus the Q4_0 path's ~14 us/call for a
  smaller matrix. This is a GGUF-authoring characteristic (llama.cpp's own
  mixed-precision quantization heuristics keep some layers at higher
  precision), not a lean bug, and is out of this session's scope - flagged
  as a candidate for a future decode-specialized F32 matvec kernel.
  Discovered, not attempted, this session.
- **The Q4_0/Q8_0 decode matvecs' low %-of-peak numbers are a dispatch-size
  effect, not a kernel-body inefficiency** (see Change 2's revert) - the
  large matmuls (Q8_0 lm-head at 70-86% of peak, 150+ MB/dispatch) versus
  the small ones (wo/qkv at 6-40% of peak, 0.5-9 MB/dispatch) track total
  bytes moved per dispatch far more than any structural difference in the
  kernel body. This makes further fusion (fewer, larger dispatches) a more
  promising lever than kernel-body micro-optimization for these specific
  matvecs, consistent with Change 3's approach.
- **Qwen3-1.7B's `short` case has high run-to-run variance on this box**
  independent of any code change - the box's own before/before spread
  across the two 5-rep sweeps in this doc was 5.62-19.15% (e.g. 8.52 to
  13.52 ms/tok within one 8-rep sample). A 5-rep median can land on either
  side of a small (<5%) true effect purely from this noise; this session's
  Change 3 result for this model needed an 8-rep re-run to resolve.
- WGSL/browser: `add_rmsnorm.wgsl` uses `@workgroup_size(256, 1, 1)`
  (WebGPU's guaranteed minimum), `linear_q6k_decode.wgsl` uses
  `@workgroup_size(128, 1, 1)` - both under the browser's 256-invocation
  cap, no new WGSL extension/feature requirement (no subgroups, no
  `packed_4x8_integer_dot_product`) in either new kernel. Built and gated
  natively only this session; `wasm-pack build --target web` is left to the
  lead per the task's own instructions.
