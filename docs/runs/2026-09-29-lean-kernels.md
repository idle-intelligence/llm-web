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

## 6. Session 2 continued: closing further on llama.cpp

Same branch (`lean-kernels`), continuing from commit `d17a3e3` (this doc's
own section 1-5 state). Same box, same gates, same ABAB/median-of-5
(8-rep re-run on any model whose 5-rep result looked marginal or noisy)
methodology as above. Candidates taken in the task's given order.

### Change 4: decode-shaped F32 matvec (`linear_f32_decode.wgsl`)

Targets the F32-residency observation from section 2's Observations:
`SmolLM2-360M-Instruct-Q4_0.gguf` ships 4 of 32 `ffn_down` tensors as F32,
which fell through to `linear.wgsl`'s naive one-thread-per-output-element
kernel at decode. New kernel (`crates/lean/src/shaders/linear_f32_decode.wgsl`)
reuses the same 128-thread, 4-rows/workgroup, 32-lanes/row, shared-memory
tree-reduction structure as `linear_q4_decode.wgsl`/`linear_q8_decode.wgsl`,
adapted to a plain (undequantized) f32 weight row - no per-block dequant,
just a coalesced cooperative dot product. Wired into `model.rs`'s
`MatMulWeight::F32` arm, picked when `fast && rows == 1` (falls back to the
existing naive `linear.wgsl` kernel for prefill, unchanged).

**Gates**: all 8 pass, `top1_match=true` on every case.

**Timing** (ABAB, median of 5, SmolLM2-360M only - the only model in the
fixture set with an F32-resident tensor):

| case | before (naive linear.wgsl) | after (linear_f32_decode) | speedup |
|---|---:|---:|---:|
| short | 7.07 ms/tok | 5.84 ms/tok | 1.211x |
| long | 8.50 ms/tok | 7.22 ms/tok | 1.177x |

**Kept** - both cases clear the >3% bar by a wide margin (17.7-21.1%), no
other model has an F32-resident tensor so no regression surface elsewhere.
Commit: `lean: decode-shaped F32 matvec (linear_f32_decode.wgsl)`.

### Change 5 (attempted, reverted): ROWS_PER_WG=1 for Q4_0 decode matvecs

Candidate 2's first half: tried shrinking `linear_q4_decode.wgsl`'s
`ROWS_PER_WG` from 4 to 1 (`WG_SIZE`/`@workgroup_size` from 128 to 32),
giving 4x more workgroups for the same total thread count on Qwen2.5-0.5B's
small Q4_0 matvecs (`wo`, `qkv_fused`, out_dim 896-1152) - the hypothesis
being that more, smaller workgroups let the GPU's scheduler hide dispatch
launch/ramp-up latency better than fewer, larger ones. Applied uniformly
(not shape-gated, since this was a probe of the mechanism before committing
to a two-kernel-variant integration). Gates passed (all 8, `top1_match=true`).

ABAB timing (median of 5, Qwen2.5-0.5B, `short`/`long_tools_single` -
`gate_up`/`down` also route through this same kernel, so this measures the
whole model's Q4_0 decode path, not just the small matvecs):

| case | before (ROWS_PER_WG=4) | after (ROWS_PER_WG=1) | delta |
|---|---:|---:|---:|
| short | 5.11 ms/tok | 5.22 ms/tok | -2.2% (regression, noise-level) |
| long_tools_single | 7.49 ms/tok | 7.46 ms/tok | +0.4% (flat, noise) |

No model showed >3% gain (one showed a small regression); reverted per the
task's stop-rule. Read as: at Qwen2.5-0.5B's `wo`/`qkv_fused` shapes
(out_dim 896-1152), `out_dim / 4` workgroups (224-288) already exceeds the
RTX 3080's 68 SMs by 3-4x, and this kernel's shared-memory footprint (128
`f32`s = 512B/workgroup) is small enough that occupancy was never the
limiter - matches section 2's Observations note that these dispatches are
bytes-moved-bound, not occupancy-bound, so redistributing the same total
thread count across more/smaller workgroups doesn't help. Not committed;
`linear_q4_decode.wgsl`/`model.rs`'s dispatch left unchanged.

The epilogue-fusion half of candidate 2 (folding `wo`'s residual add into
its own kernel) was not attempted this session - `wo`'s output already
feeds `add_rmsnorm.wgsl` (Change 3, prior session), so the residual add is
already fused with the *following* rmsnorm; fusing it a second time into
`wo`'s own matvec kernel would need a three-way fused kernel
(matvec+add+rmsnorm) and was judged out of scope for the remaining time in
this session.

### Change 6: skip unchanged uniform-buffer writes in `Pool::uniform()`

Candidate 3: per-step fixed overhead. `crates/lean/src/pool.rs`'s
`uniform()` called `queue.write_buffer()` on every call, even when the
value was byte-identical to the previous call at that cache key - true for
every decode-time `linear`/`rmsnorm`/`add_rmsnorm`/`silu_mul_fused` dims
uniform, since decode's shape (`m`/`k`/`n`/`blocks_per_row`/`n_offset`/etc.)
never changes step to step; only position-dependent uniforms (RoPE's
position, attention's `kv_len`) actually change per step. Added a
`uniform_cache: RefCell<HashMap<String, Vec<u8>>>` alongside the existing
buffer cache: `uniform()` now compares the new value's bytes against the
last-written bytes for that key and skips `queue.write_buffer` when they
match. Cleared in `Pool::reset()` alongside the other caches. Pure CPU-side
per-step overhead reduction - the GPU-side buffer contents are identical
either way, no numerical change.

**Gates**: all 8 pass, `top1_match=true` on every case.

**Timing** (ABAB, median of 5, all four models, `short`/`long` cases,
against a clean baseline re-exported at the commit before this change):

| model | case | before | after | speedup |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | short | 5.11 | 4.83 | 1.058x |
| Qwen2.5-0.5B | long | 6.32 | 5.95 | 1.062x |
| Qwen2.5-3B | short | 14.70 | 13.76 | 1.068x |
| Qwen2.5-3B | long | 15.23 | 14.25 | 1.069x |
| Qwen3-1.7B | short | 9.89 | 8.93 | 1.108x |
| Qwen3-1.7B | long | 10.48 | 9.98 | 1.050x |
| SmolLM2-360M | short | 5.95 | 5.64 | 1.055x |
| SmolLM2-360M | long | 7.28 | 6.88 | 1.058x |

**Kept** - every model/case clears the >3% bar (5.0-10.8%), every AFTER run
beat its paired BEFORE run within the same round for every model/case (no
mixed-direction rounds), no numerical change. Commit: `lean: skip redundant
uniform buffer writes when the value is unchanged`.

### Change 7 (attempted, reverted): SPLIT_CHUNK 128 -> 64/256

Candidate 4: decode attention at long context. Re-profiling confirmed the
premise - on Qwen2.5-0.5B's `long_tools_single` case (kv_len=2225),
`dec_layer.attn.split` is the single largest line item in the per-kernel
table at 1559.6 us/step (24 layers x 64.984 us/call), well ahead of
`mlp.gate_up` (685.6 us/step) and every other kernel, consistent with the
task's "long_tools_single is ~40% slower than short" observation.
`shaders/attn_decode_split.wgsl`'s own doc comment shows `num_splits =
min(MAX_SPLITS, ceil(kv_len / SPLIT_CHUNK))`, `SPLIT_CHUNK = 128` - tried
both directions (256: fewer, larger-chunk splits; 64: more, smaller-chunk
splits) as a shape-derived constant change, no kernel-body rewrite.

Tested against a clean re-exported baseline, all 8 gates passing on both
variants (`top1_match=true` throughout). ABAB timing, Qwen2.5-0.5B
`long` (kv_len=86) and `long_tools_single` (kv_len=2225), median of 8 (the
`long_tools_single` 5-rep result was re-run at 8 to confirm direction):

| case | SPLIT_CHUNK=128 (base) | SPLIT_CHUNK=256 | SPLIT_CHUNK=64 |
|---|---:|---:|---:|
| long (kv_len=86) | 5.98 | not re-measured (clearly worse below) | 5.54 (-7.4%) |
| long_tools_single (kv_len=2225) | 6.905 | 8.22 (+19.0%, regression) | 6.58 (-4.7%) |

`SPLIT_CHUNK=256` regressed both cases and was dropped without further
testing. `SPLIT_CHUNK=64` looked like a real >3% win on Qwen2.5-0.5B
(4.7-7.4% faster, every round faster, not just the median) - but
`SPLIT_CHUNK=64` also moves the split-path threshold itself (`kv_len <=
SPLIT_CHUNK` decides single-workgroup vs split-K), so `kv_len=65-86` ranges
that previously used the cheap single-workgroup kernel now take the
split-K path on every model, not just Qwen2.5-0.5B. Checked against the
other two models sharing this code path:

| model (head_dim) | case (kv_len) | before | SPLIT_CHUNK=64 | delta |
|---|---|---:|---:|---:|
| Qwen3-1.7B (128) | long (65) | 8.58 | 9.79 | **+14.1% (regression)** |
| SmolLM2-360M (64) | long (86) | 6.76 | 6.13 | -9.3% (improvement) |

Qwen3-1.7B's `head_dim=128` split kernel (`attn_decode_split_128`) regressed
at exactly the newly-crossed threshold (`kv_len=65`, `num_splits=2`,
`chunk=33` - a tiny split whose reduce-pass and launch overhead isn't
covered by 33 keys' worth of work at this kernel's wider `workgroup_size(128)`
shape). Violates the task's "no loss elsewhere" rule; reverted. Not
committed; `SPLIT_CHUNK` in `model.rs` left at 128.

Read as: `SPLIT_CHUNK` is not a single global-best constant across this
project's `head_dim` variants - a `head_dim`-conditioned threshold (e.g. a
smaller `SPLIT_CHUNK` only for `head_dim=64`) is architecturally
defensible (still shape-derived, not measured-speed-derived) and would
plausibly capture the Qwen2.5-0.5B/SmolLM2-360M win without the Qwen3-1.7B
regression, but was not implemented this session - the two-constant version
needs its own full ABAB pass across all four models' `short`/`long`/
`long_tools_*` cases to confirm the boundary doesn't move the regression
onto some other `kv_len`, and that work was not completed in the time
available. Flagged as the clear next step for whoever picks this back up.
The redundant-KV-read structural issue (each of the `n_rep` query heads
sharing a `kv_head` re-reads the same K/V range from its own workgroup,
`n_rep`=7 on Qwen2.5-0.5B) was identified as the more fundamental cause of
`attn.split`'s cost but a fix requires fusing the head group into one
workgroup (a real kernel rewrite, not a constant change) and was judged too
large/risky to attempt and verify in the time remaining this session.

## 7. Remaining gap to llama.cpp, end of session 2

| model | lean decode (session 2 end, short) | llama.cpp decode | gap |
|---|---:|---:|---:|
| Qwen2.5-0.5B | ~4.83 ms/tok | 1.66 ms/tok | ~2.91x |
| Qwen2.5-3B | ~13.76 ms/tok | 4.19 ms/tok | ~3.28x |

Down from the task's starting point of 5.17x/14.78x (3.11x/3.53x gap to
llama.cpp) via changes 4 and 6 above (change 5 and 7 reverted, no effect).
Combined across all of this branch's sessions, the gap has closed from the
original ~10-12x to ~2.9-3.3x.

## 8. Session 3

Same branch (`lean-kernels`), continuing from commit `000f33b` (this doc's
own section 1-7 state). Same box (RTX 3080/Vulkan), same gates, same
ABAB/median-of-5 (8 reps for any model/case with noticeable spread)
methodology. Target: decode attention at long context, per the task's own
"redundant-KV-read" and "head_dim-conditioned SPLIT_CHUNK" leads from
session 2's Observations and Change 7.

### Change 8 (attempted, reverted): GQA head-group fusion in decode attention

The premise from session 2: in both `attn_decode.wgsl` (single-workgroup
path) and `attn_decode_split.wgsl` (split-K path), each of the `n_rep`
query heads sharing a `kv_head` (`n_rep`=7 on Qwen2.5-0.5B, up to 8 on this
project's other GQA models) ran in its own workgroup and independently
re-read the same `kv_head`'s K/V range from global memory - `n_rep`-fold
redundant K/V traffic. Rewrote both shaders so one workgroup processes a
`kv_head`'s whole query-head group at once: Q for all `n_rep` heads loaded
into shared memory once, each thread's K row read from global memory
exactly once per tile and fanned out to all `n_rep` heads' dot products via
a small per-rep register array, each V element read once per (thread,
tile-key) and fanned out to all `n_rep` accumulators. Per-head online-softmax
state (`m`/`l`/`acc`, each `array<f32, MAX_N_REP=8>`) and the tree-reduction
shape were otherwise unchanged, just looped once per head per tile instead
of once per (head, workgroup). Workgroup memory: `q_shared`/`tile_scores`
(`array<f32, 1024>` each, `MAX_N_REP=8 * HEAD_DIM_STRIDE=128`) plus
`reduce_buf` (`array<f32, 256>`) = 9216B, under the 16KB default limit.
Dispatch changed from `(n_heads, 1, 1)` / `(n_heads, num_splits, 1)` to
`(n_kv_heads, 1, 1)` / `(n_kv_heads, num_splits, 1)`. `attn_decode_reduce.wgsl`
(pass 2, already per-query-head) needed no change.

**Gates**: all 8 pass, `top1_match=true` on every case (including
`long_tools_single`/`long_tools_multiturn`, which exercise the split-K
path).

**Timing** (ABAB, median of 5, all 4 models, every fixture case including
long-context ones where available):

| model | case | before (unfused) | after (fused) | delta |
|---|---|---:|---:|---|
| Qwen2.5-0.5B | short | ~4.7 ms/tok | ~6.4 ms/tok | -36% |
| Qwen2.5-0.5B | long (kv_len=86) | ~5.4 ms/tok | ~8.8 ms/tok | -63% |
| Qwen2.5-0.5B | long_tools_single (kv_len=2225) | ~6.7 ms/tok | ~9.4 ms/tok | -40% |
| Qwen2.5-3B | short/long | ~13.0-14.2 ms/tok | ~16.0-18.3 ms/tok | -25% to -30% |
| Qwen3-1.7B | short/long/long_tools_single | ~8.6-11.6 ms/tok | ~9.3-12.8 ms/tok | -8% to -15% |
| SmolLM2-360M | short/long | ~5.2-7.2 ms/tok | ~6.2-9.0 ms/tok | -15% to -25% |

**Reverted** - every case on every model regressed; no case cleared the
>3% gain bar. Root cause: collapsing `n_heads` (or `n_heads * num_splits`
on the split path) workgroups down to `n_kv_heads` (or `n_kv_heads *
num_splits`) cuts workgroup count by exactly `n_rep` (up to 8x) while
making each surviving workgroup do `n_rep`x more serial work (the per-rep
softmax reduction loop). This project's decode attention already dispatches
few workgroups (14-16 query heads, or 2-8 KV heads) against a 68-SM GPU -
it is occupancy-bound, not K/V-bandwidth-bound, at these problem sizes.
Cutting workgroup count by up to 8x costs far more in occupancy than the
saved K/V reads are worth, even at `kv_len=2225` where bandwidth would be
expected to matter more. Confirmed uniform across every model, case, and
both attention paths - a clean revert per the task's stop-rule. Not
committed; `attn_decode.wgsl`, `attn_decode_split.wgsl`, and `model.rs`'s
two `attn_decode` dispatch calls restored to be byte-identical to commit
`000f33b`.

### Change 9 (kept): head_dim-conditioned `SPLIT_CHUNK`

The cheaper fallback from session 2's Change 7: a single global
`SPLIT_CHUNK=128` is not a best fit for every `head_dim` this project
compiles. Session 2 found `SPLIT_CHUNK=64` (applied uniformly) was a real
win on `head_dim=64` models but regressed `head_dim=128` (Qwen3-1.7B, at
the newly-crossed `kv_len=65` threshold: `attn_decode_split_128`'s wider
`workgroup_size(128)` kernel doesn't amortize launch/reduce overhead at the
resulting 33-key chunks). Conditioning the constant on `head_dim` - still a
static model-architecture fact known at dispatch time, never a timing
measurement - should capture the `head_dim=64` win without moving the
`head_dim=128` threshold at all.

`crates/lean/src/model.rs`: replaced the `SPLIT_CHUNK` constant with
`fn split_chunk(head_dim: u32) -> u32 { if head_dim <= 64 { 64 } else { 128 } }`,
threaded through `decode_split_plan(kv_len, head_dim)` (added the `head_dim`
parameter) and its one call site in `attn_decode`. For `head_dim=128`
models this is bit-for-bit identical to the old global constant (`128`
either way), so no behavior change is possible there by construction - the
"no loss elsewhere" risk from session 2's uniform attempt is closed by
design, not just by measurement.

**Gates**: all 8 pass, `top1_match=true` on every case.

**Parity at long context**: a synthetic 5630-token prompt (repeated
sentence, well past the task's 4096-token target and past `MAX_SPLITS`'s
own saturation point for both old and new `SPLIT_CHUNK` values) run on
Qwen2.5-0.5B against both the pre-session baseline and this change:
identical greedy continuation text (`"The quick brown fox jumps over the
lazy"`) and decode time within noise (9.32 vs 9.29 ms/tok) - expected,
since at this length `num_splits` saturates at `MAX_SPLITS=32` for both old
and new `SPLIT_CHUNK` (`kv_len / 64` and `kv_len / 128` both far exceed 32),
so the two configurations converge to identical dispatch parameters well
before 5630 tokens.

**Timing** (ABAB, median of 8 for Qwen2.5-0.5B/Qwen3-1.7B/SmolLM2-360M, 5
for Qwen2.5-3B - all 8 rounds ran in the same session, direction checked
per-round, not just on the median):

| model (head_dim) | case (kv_len) | before (global 128) | after (conditioned) | delta |
|---|---|---:|---:|---|
| Qwen2.5-0.5B (64) | long (86) | 5.845 | 5.400 | **+7.6%** |
| Qwen2.5-0.5B (64) | long_tools_multiturn (2354) | 6.785 | 6.535 | **+3.7%** |
| Qwen2.5-0.5B (64) | long_tools_single (2225) | 6.795 | 6.505 | **+4.3%** |
| Qwen2.5-0.5B (64) | non_english (54) | 5.370 | 5.245 | +2.3% |
| Qwen2.5-0.5B (64) | short (36) | 4.720 | 4.665 | +1.2% |
| SmolLM2-360M (64) | long (86) | 6.790 | 5.730 | **+15.6%** |
| SmolLM2-360M (64) | non_english (63) | 6.205 | 5.700 | **+8.1%** |
| SmolLM2-360M (64) | short (37) | 5.395 | 5.340 | +1.0% |
| Qwen2.5-3B (128) | long (86) | 14.220 | 14.100 | +0.8% |
| Qwen2.5-3B (128) | non_english (54) | 13.830 | 13.780 | +0.4% |
| Qwen2.5-3B (128) | short (36) | 13.600 | 13.720 | -0.9% |
| Qwen3-1.7B (128) | long (65) | 9.715 | 9.750 | -0.4% |
| Qwen3-1.7B (128) | long_tools_single (2225) | 11.530 | 11.580 | -0.4% |
| Qwen3-1.7B (128) | non_english (33) | 9.355 | 9.390 | -0.4% |
| Qwen3-1.7B (128) | short (15) | 8.985 | 8.955 | +0.3% |

**Kept** - every `head_dim=64` long-context case clears the >3% bar
(3.7-15.6%), with the biggest wins on the cases with the most split-K
overhead to amortize (SmolLM2-360M `long`, Qwen2.5-0.5B's `long_tools_*`).
Every `head_dim=128` case is flat within this box's own measured noise
floor (session 2 logged 5.62-19.15% round-to-round spread on Qwen3-1.7B
`short` alone) - the largest regression anywhere is Qwen2.5-3B `short` at
-0.9%, well inside that noise band, and per-round direction (not just the
median) favored `after` in 6-7 of 8 rounds on the winning cases. A
kernel-level profile of SmolLM2-360M's `long` case confirms the mechanism:
`dec_layer.attn` (single-workgroup, ~26-38us/call at kv_len=86-89 with the
old threshold) is replaced by `dec_layer.attn.split` +
`dec_layer.attn.reduce` at the same kv_len once `SPLIT_CHUNK` drops to 64 -
exactly the intended earlier crossover to the split-K path for `head_dim=64`
models. Committed as `lean: head_dim-conditioned SPLIT_CHUNK for decode
attention split-K`.

### Remaining gap to llama.cpp, end of session 3

`short`-case decode is unaffected by Change 9 (both old and new
`SPLIT_CHUNK` route `short`'s small `kv_len` through the un-split kernel
identically), so the gap on the task's headline `short` numbers is
unchanged from session 2's end:

| model | lean decode (short) | llama.cpp decode | gap |
|---|---:|---:|---:|
| Qwen2.5-0.5B | ~4.67-4.83 ms/tok | 1.66 ms/tok | ~2.8-2.9x |
| Qwen2.5-3B | ~13.7-13.8 ms/tok | 4.19 ms/tok | ~3.3x |

Change 9's gain is entirely on long-context cases outside this headline
pair (Qwen2.5-0.5B `long`/`long_tools_*`, SmolLM2-360M `long`/`non_english`),
where it closes 3.7-15.6% of the remaining gap on `head_dim=64` models
specifically. The GQA head-group fusion (Change 8) that would have targeted
the `short`-case gap directly was reverted as an occupancy regression (see
above) - closing the `short`-case gap further needs an approach that adds
parallelism rather than removing it (e.g. batching multiple independent
dispatches, or a genuinely different attention decomposition), not
attempted this session.
