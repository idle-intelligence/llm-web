# Benchmarks — Q4_0 kernel work (K1-K4)

Research-log style: machine, commit, command, then a data table per commit. Analysis
lives in commit messages and inline WGSL/Rust comments, not here.

## Machine

- Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- `CARGO_BUILD_JOBS=4`, release profile, `--features native,wgpu`
- Model: xLAM-2-3b-fc-r, Q4_0 GGUF (`/Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf`)
- Prefill fixture: `fixtures/reference/rendered/02_tools_single.tokens.json` (2225 tokens)
- Command: `./target/release/llm-agent bench --gguf <path> --tokens <fixture> --decode-steps N`
  (loads the model once, times a full prefill then N greedy decode steps, forcing a
  sync readback each step so timings reflect GPU completion, not queue time)

## Baseline (pre-session, C3 checkpoint, naive kernel only)

| commit | prefill tok/s (02) | decode ms/token | greedy-match exact |
|---|---|---|---|
| `6828687` (C3) | 29.2 | 307 | y |

## K1 — decode matvec (M=1), commit `80fea4b`

Cooperative Q4_0 matvec kernel (`wgsl/shader_q4_matvec.wgsl`): 256 threads/workgroup,
8 output rows/workgroup, 32 threads/row splitting K across Q4_0 blocks, x staged into
workgroup shared memory 1024 elements (4KB) at a time, shared-memory tree reduction.
Dispatched whenever `B*M==1`. A subgroupAdd() variant
(`wgsl/shader_q4_matvec_subgroup.wgsl`) exists behind `gguf.rs::has_subgroup_support()`
but nothing calls `set_subgroup_support(true)` in this crate (that requires probing
`wgpu::Features::SUBGROUP` in `web.rs` at device-init time, outside this task's owned
files) — **all numbers below use the portable shared-memory variant.**

**Subgroup-availability finding**: not empirically checked against Chrome/WebGPU this
session (no browser harness wired up here) — sts-web's own `web/bindings.rs:162-163`
is the only place in either codebase that calls `set_subgroup_support`, gated on
`wgpu::Features::SUBGROUP` at device creation, and that code path is native/WebGPU
device-init plumbing, not something this crate's `llm-agent` CLI exercises. Treat this
as unverified for this model; the portable variant is the safe default either way.

`cargo test --release --features wgpu --test q4_matmul`: `test_q4_matvec_m1_large_n`
(N=151936, K∈{2048,11008}) plus the pre-existing `test_q4_matmul_synthetic_shapes`
(M=1, K,N∈{2048,11008}) and `test_q4_matmul_real_gguf_token_embd` (M=1, real
token_embd.weight) — all pass, all now exercise K1's kernel since `q4_matmul`'s
dispatch is transparent to callers.

| prefill tok/s (02) | decode ms/token (median, 32 steps) | greedy-match exact |
|---|---|---|
| 26.3 (unchanged — K1 doesn't touch M>1) | 184 (was 307, 1.67x) | y |

## K3 — attention chunk size, commit `e203273`

`ATTN_QUERY_CHUNK` 128 -> 256. Doesn't hit the `SimpleCyclicCmma` shared-memory panic
on this adapter (512 was reported to panic at C3; not re-verified after this change).
`full_forward`'s greedy-match tests pass (bit-identical to 128 — chunking is a pure
reshape, not a numerical change).

Also verified (no fix needed): `generate()` and the new `bench` subcommand already
narrow hidden states to the last prefill position before `lm_head` — only
`forward_logits` (a documented test-only convenience) computes the full `T x vocab`
matrix, and nothing in the production/bench path calls it.

| prefill tok/s (02) | decode ms/token | greedy-match exact |
|---|---|---|
| 26.5 (8-step sample; noise-level vs 26.3) | 183.6 (8-step sample) | y |

Conclusion: attention chunking isn't prefill's bottleneck. The naive Q4_0 matmul
kernel is (see K2).

## K2 — prefill matmul (M>1), commit `95a8c42` — attempted, not shipped

`wgsl/shader_q4_tiled.wgsl`: classic shared-memory-tile GEMM, TK=32 (one Q4_0 block),
dequantizing a `[TN,TK]` weight tile into workgroup shared memory once and reusing it
across `TM` input rows (vs naive's per-row redundant dequant). Correct — see
`test_q4_matmul_tiled_k2_shapes` (M∈{64,128,256}, K,N∈{2048,11008} synthetic, vs CPU
reference) — but never closed the performance gap with the naive kernel, so it is
**not** wired into `q4_matmul`'s default dispatch. `q4_matmul_tiled_forced` exists
only for tests.

Isolated micro-benchmark (`bench_tiled_matmul_shape`, `#[ignore]`d,
`cargo test --release --features wgpu --test q4_matmul -- --ignored
bench_tiled_matmul_shape --nocapture`), M=2225/K=2048/N=11008 (matches
`02_tools_single`'s length and the ffn_gate/up shape):

| variant | TM/TN | MICRO | dequant | ms/call | GFLOP/s |
|---|---|---|---|---|---|
| v1 | 64/64 | 4x4 | scalar `read_u8`, one nibble/read | full-model prefill only: 369s total (6.0 tok/s) | — |
| v2 | 128/128 | 8x8 | scalar `read_u8` | 3324.5 | 30.18 |
| v3 | 64/64 | 4x4 | vectorized u32-word (8 values/read) | 2486.4 | 40.35 |

Naive kernel's effective throughput at full-model granularity: 13.35 TFLOP total
prefill work (≈2 × 3B params × 2225 tokens) / 84s measured = **~159 GFLOP/s**. v3 is
still ~4x below that. M2's GPU peak is on the order of 3.6 TFLOP/s FP32, so both the
naive kernel (≈4% of peak) and the tiled attempt (≈1% of peak) are far from
compute-bound — this is a kernel-efficiency problem, not a fundamental ceiling.

Suspected but **not verified** root cause: two `workgroupBarrier()` calls per 32-element
(one Q4_0 block) K-step leave very little independent ALU work (512 FMAs across 256
threads) to hide barrier/sync latency behind; widening TK per barrier pair (at the
cost of either much larger shared memory or multi-block-per-tile dequant complexity)
is the next thing to try, not attempted here for lack of remaining session budget.

| prefill tok/s (02) | decode ms/token | greedy-match exact |
|---|---|---|
| 25.7 (naive kernel, K2 not dispatched) | 193.7 (K1 kernel, unaffected) | y |

(With K2's v1 wired into production dispatch, measured full prefill was 368.93s / 6.0
tok/s — a 4.4x regression vs the 84s naive baseline. Reverted before committing as the
default.)

## K4 — f16 KV cache

Not attempted this session. Time/risk trade-off: converting the 906MB f32 KV cache
(`kv.rs`'s module doc comment has the math) to f16 requires an f32->f16->f32
round-trip on every cached K/V value read back into attention, and the C2 checkpoint's
own findings (`git log` on `model.rs`) already show several near-tied argmax positions
from Q4_0 quantization noise alone — adding f16 KV rounding risks flipping additional
greedy-decode tokens, which the task's acceptance bar (exact greedy-token match on
02/03) does not tolerate. Given K1 succeeded and K2 didn't reach a shippable state,
the remaining session budget went to K2 debugging instead. Left for future work.

## Session 2 — measured breakdown (P1) + kernel bandwidth fix (K5)

Nobody had measured *where* the 184ms/token and 84s prefill time actually went. This
session adds instrumentation to `llm-agent bench` (`--decode-steps`) that reports:
(a) isolated K1 matvec throughput at the model's four shapes, 100 iters/shape, one
sync at the end; (b) an analytical GPU-dispatch-count estimate (list every op in one
decoder layer × `num_layers`, since Burn's wgpu backend exposes no dispatch counter);
(c) a decode per-token wall-time split, using a bench-only `set_skip_matvec_for_bench`
flag (`gguf.rs`) that skips every `q4_matmul` kernel launch — the resulting decode
step's wall time is "everything else" (RMSNorm, RoPE, attention, SwiGLU elementwise,
residual adds, KV cache writes); (d) the same flag applied to prefill, splitting
matmul-call time from attention+everything-else (not split further — see budget note
below); (e) the sync-readback cost of `into_data()` alone on a 151936-f32 tensor.

### P1 — measured breakdown (before K5), `02_tools_single` (2225 tokens), 32 decode steps

**Isolated K1 matvec (M=1), 100 iters/shape:**

| K | N | ms/call | GB/s |
|---|---|---|---|
| 2048 | 151936 (lm_head) | 6.569 | 26.6 |
| 2048 | 11008 (gate/up) | 0.541 | 23.4 |
| 11008 | 2048 (down) | 0.558 | 22.7 |
| 2048 | 2048 (q/o) | 0.144 | 16.4 |

All four shapes land at **16-27 GB/s** against the M2's ~100GB/s unified-memory peak
(16-27% of peak) — well above the ~11 GB/s stt-wasm extrapolation in `docs/ENGINE.md`
§10, but still far short of bandwidth-bound. Confirms the shader review's diagnosis
(misaligned/uncoalesced nibble reads from the 18-bytes/block interleaved layout) as
the mechanism, not a different bottleneck.

**GPU dispatch count per decode token (analytical estimate — see `print_dispatch_estimate`
in `llm-agent.rs` for the itemized per-op list):** ~44 ops/decoder layer × 36 layers +
out_norm(~4) + lm_head(1) ≈ **~1589 dispatches/token**. RoPE (10/layer, mostly from
`rotate_half`'s `cat`) and RMSNorm (8/layer, Burn's unfused mean/var/rsqrt/mul chain)
are the two largest non-matmul contributors — both candidates for WGSL fusion (not
attempted this session, see "next most valuable" below).

**Decode per-token wall-time split (median of 32 steps, 184-192ms measured that run):**

| component | ms |
|---|---|
| embedding dequant+upload (CPU) | 0.43 |
| 36 layers of matvecs (sum of isolated numbers × per-layer counts; k/v_proj's N=256,K=2048 shape isn't one of the 4 measured — estimated by scaling the measured 2048×2048 GB/s by byte count) | 70.7 |
| everything else on GPU (skip-matvec measurement: RMSNorm+RoPE+attention+SwiGLU+residuals+cache writes) | 75.4 |
| final head matvec (N=151936, K=2048) | 6.6 (included in the 70.7 sum's total) |
| **matvec_sum (36 layers + head)** | **77.3** |
| sync readback of logits alone (`into_data`, 151936 f32 = 600KB) | 0.21 |

`matvec_sum (77) + everything-else (75) ≈ 152ms`, vs the measured real decode step of
~184ms — a ~32ms/step gap not accounted for by either bucket, most plausibly CPU-side
per-dispatch/queue-submission overhead across ~1589 GPU dispatches (not something
`skip_matvec_for_bench`'s bypass — which still submits every non-matmul op — isolates
further within this session's time budget). **The two buckets are roughly equal size**:
Q4 matvec bandwidth and "everything else" (dominated by RoPE/RMSNorm dispatch count
and the KV cache's per-step `slice_assign`, see "next most valuable" below) are
comparably expensive — fixing only the matvec kernel caps the achievable win at ~2x,
not the ~5-8x the ≤40ms/token target would need.

**Prefill breakdown (`02_tools_single`, 2225 tokens, before K5):** matmul-calls
78.18s, attention+everything-else 6.18s, of 84.36s total — **prefill is >90% matmul
time**, confirming `docs/ENGINE.md` §10's compute-bound diagnosis and that the naive
kernel's per-element redundant dequant (K2's finding) is the lever, not attention
chunking (already ruled out by K3) or cache-copy overhead.

### K5 — Q4Tensor buffer repack (aligned nibbles + separate f32 scales), commit below

Per shader-review finding: GGUF's on-disk Q4_0 block is 18 bytes (2-byte f16 scale +
16 bytes nibbles); 18 isn't a multiple of 4, so every other block's nibble data starts
mid-word, forcing every WGSL read through `read_u32_unaligned`'s two-load path with no
coalescing across lanes reading adjacent blocks (consecutive lanes' byte offsets are
18 apart, not 4). Fix (`gguf.rs::Q4Tensor::from_q4_bytes`): repack at load time into
two aligned buffers — `scales: array<f32>` (one per block) and `weights: array<u32>`
(nibbles only, exactly 4 u32/block, scale stripped) — applied uniformly to all four
kernels (naive, matvec, matvec_subgroup, tiled) so there's one weight layout crate-wide
and no duplicated GPU memory (net size increase: 20 bytes/block vs 18, ~11%). This also
moots the shader review's alignment-padding concern (point 3) — the old code's
"pad to 4 bytes" branch is gone entirely since both new buffers are inherently aligned.

The read-only (`read` vs `read_write`) binding-qualifier cleanup the review also asked
for (point 3) was **attempted and reverted**: cubecl-wgpu suballocates multiple logical
buffers from shared physical arenas, and wgpu's usage tracker is whole-buffer, not
per-range — marking some bindings `read` while `output` stayed `read_write` triggered
`wgpu error: ... conflicting usages ... STORAGE_READ_ONLY ... STORAGE_READ_WRITE` as
soon as two logical buffers happened to share a physical arena (surfaced immediately
in `cargo test --test q4_matmul`, all four tests). Reverted to `read_write` everywhere
(matches the pre-K5 baseline) — this runtime's memory model doesn't support the
finer-grained qualifier split within one arena buffer.

**Correctness**: `cargo test --release --features wgpu --test q4_matmul` (all 4
non-ignored tests) and `cargo test --release --features wgpu --test full_forward --
--test-threads=1` (all 3 greedy-match tests) pass, bit-identical to pre-K5.

**Before/after** (`02_tools_single`, 2225 tokens, 32 decode steps; decode ms/token
varied 184-208 across repeated runs both before and after K5 — machine-load noise on
a shared M2, not attributable to the change with confidence at this sample size):

| | prefill tok/s | prefill matmul-time | decode ms/token (median) | isolated matvec GB/s (K=2048,N=2048) |
|---|---|---|---|---|
| before K5 | 26.4 | 78.2s | 183.9 | 16.4 |
| after K5 | 29.6 | 68.0s | 191-208 (noisy, ~flat-to-slightly-worse) | 17.5-24.5 |

**K5 is a genuine ~11-13% prefill win** (78.2s → 68.0s matmul-call time, 26.4 → 29.6
tok/s) — the naive kernel does `blocks_per_row` redundant reads per output element, so
removing the misaligned-read tax pays off once per element, M times over during
prefill (M=2225). **Decode is a wash**: K1's matvec issues only ~4 aligned-word reads
per (row, block) pair regardless, so removing the misalignment tax saves less per call,
and the extra binding (5 buffers vs 4 per dispatch, ~1589 dispatches/token) plausibly
adds enough CPU-side per-dispatch overhead to offset the saving — not confirmed within
this session's budget (would need per-dispatch CPU profiling, not just wall-clock GPU
totals, to separate the two effects). Isolated matvec GB/s **did not reach the
reviewer's 60-100GB/s target** — the repack removes the unaligned-load penalty but
still has each lane read its own block's 4 words with a stride-4-words-per-lane access
pattern (not true word-index-interleaved coalescing across lanes, which the reviewer's
fuller suggestion called for and which this session's budget didn't reach — see below).

## Summary

| kernel | status | prefill tok/s (02) | decode ms/token | target | met? |
|---|---|---|---|---|---|
| baseline (C3) | — | 29.2 | 307 | — | — |
| K1 (matvec, M=1) | shipped | 26.3 | 184 | ≤40 ms/token | no |
| K3 (chunk=256) | shipped | 26.5 | 183.6 | ≥300 tok/s | no |
| K2 (tiled, M>1) | attempted, reverted | 25.7 (naive) | 193.7 | ≥300 tok/s | no |
| K4 (f16 KV) | not attempted | — | — | — | — |
| K5 (Q4Tensor buffer repack) | shipped | 29.6 | ~190 (noisy, flat) | — | — |

Neither target (decode ≤40ms/token, prefill ≥300 tok/s on the 2225-token prompt) was
met this session. Decode improved 1.67x (K1); prefill improved a further ~11-13% this
session (K5) after being unchanged from baseline through K2/K3. Greedy-token-match
numerics are exact and unchanged by every shipped change (K1, K3, K5) — verified via
`cargo test --release --features wgpu --test full_forward -- --test-threads=1` after
each commit.

**Single most valuable change not reached this session**: `kv.rs::KvCache::append`
does `self.k[layer].clone().slice_assign(ranges, k)` — the explicit `.clone()` before
`slice_assign` guarantees the tensor's reference count is ≥2 at the call, which
prevents Burn/cubecl from mutating in place; every decode step this likely copies the
*entire* per-layer `[1, n_kv_heads, max_ctx, head_dim]` cache tensor (12.58MB per
layer at `max_ctx=12288`) rather than writing only the new `T` rows, ×36 layers ×2
(k,v) ≈ 900MB of copy traffic every single decode token regardless of how few tokens
are actually new. This is a plausible dominant contributor to the "everything else"
75ms bucket measured above (a bandwidth estimate at ~24GB/s system throughput puts
900MB at ~37ms, roughly half that bucket) and was found by inspection while writing
this section, not measured in isolation — the next session should replace the clone
with `std::mem::replace(&mut self.k[layer], <placeholder>)` before `slice_assign` (or
confirm whether Burn's wgpu backend does in-place `slice_assign` at all when given a
uniquely-owned tensor) and re-run the P1c decode breakdown to quantify the win before
committing to a rewrite.

## Session 3

Scope this session: D1 (KV cache in-place write) and D2a (RMSNorm fusion). D2b/c/d
(RoPE fusion, SiLU*up fusion, residual folding), D3 (coalesced matvec), D4 (prefill
scratch-dequant+matmul) not attempted — see "next most valuable" below. All numbers
`02_tools_single` (2225-token prompt) unless noted; machine is shared with another
worker's concurrent native smoke-test runs per the task brief, so single before/after
pairs carry real noise (see D1's finding) — multiple runs are reported where taken.

### D1 — KV cache in-place write, commit `92fbdb5`

`kv.rs::append` did `self.k[layer].clone().slice_assign(...)`: the `.clone()` left
`self.k[layer]` holding a second reference to the handle at the exact moment
`slice_assign` ran. Traced into `burn-cubecl-0.20.1/src/kernel/index/slice_assign.rs:110-114`:
`slice_assign` calls `tensor.can_mut()` (an `Arc` strong-count check on the underlying
buffer handle) and only mutates in place if it's the sole owner, otherwise falls back
to `tensor.copy()` — a full-buffer copy of the `[1, n_kv_heads, max_ctx, head_dim]`
tensor (12.58MB/layer at `max_ctx=12288`), confirming last session's "next most
valuable" hypothesis mechanistically. Fix: `std::mem::replace(&mut self.k[layer],
<tiny placeholder>)` before calling `slice_assign` on the taken-out (now uniquely
owned) tensor, then reassign the result — drops the second reference before
`can_mut()` runs.

**Isolated bench** (`kv.rs::bench::bench_append`, `cargo test --release --features
wgpu --lib kv::bench::bench_append -- --ignored --nocapture`; one layer, T=1, real
`n_kv_heads=2/head_dim=128/max_ctx=12288` shape, forced per-call sync via
`into_data()`):

| variant | ms/call |
|---|---|
| old (`.clone()` pattern, inlined manually for comparison) | 1.2032 |
| new (`mem::replace` fix) | 1.2024 |

**No measurable difference in isolation.** Two candidate explanations, neither
confirmed within this session's budget: (a) the forced per-iteration `into_data()`
sync in the bench dominates the measured time (CPU↔GPU round-trip latency, not the
copy itself) at this single-layer/12MB scale, masking a real but small difference; or
(b) `can_mut()` is false in both cases for a reason unrelated to the `Tensor` clone —
e.g. cubecl-wgpu's shared-arena suballocator (already implicated in K5's read/read_write
binding-qualifier revert) reporting the buffer as shared regardless of the Rust-level
`Arc` count. Not distinguished here.

**End-to-end** (`02_tools_single`, full prefill+decode, `--test-threads=1`):

| | prefill tok/s | decode ms/token (20 steps to EOS) | greedy-match |
|---|---|---|---|
| before (HEAD, K5 baseline) | 35.97 | 213.8 | 20/20 |
| after D1 | 29.17 | 194.9 | 20/20 |

Prefill tok/s (which D1's change cannot affect — it's decode-only code) swung 35.97
vs 29.17 across these two runs with zero prefill-path code difference, a ~20% delta —
confirming the isolated-bench finding that this machine's noise floor is large enough
to swallow a real effect of D1's size, if any. **Kept anyway**: the change matches
burn-cubecl's documented contract and cannot regress correctness (verified: greedy
match still exact). Numerics: `full_forward` 20/20 (02), 28/28 (03) — unchanged.

### D2a — fused RMSNorm, commit `2a69490`

`RmsNormLayer::forward` called `burn::nn::RmsNorm::forward`, which lowers to ~8
separate wgpu dispatches (cast/square/mean_dim/add/sqrt/div/mul) — 2 RMSNorms/layer x
36 layers + `out_norm` was the largest non-matmul contributor to Session 2's ~1589
dispatches/token estimate. New `wgsl/shader_rmsnorm.wgsl`: one workgroup/row (B*T rows
total), 256 threads, shared-memory tree reduction for sum-of-squares, one dispatch for
the whole `[B,T,hidden]` tensor. Matches burn-nn 0.20's `Y = X / sqrt(mean(X^2) + eps)
* gamma` exactly (no mixed-precision cast needed — this model's whole pipeline is f32).
`gguf.rs::rmsnorm_fused` dispatches it; `RmsNormLayer::forward` narrowed from generic
`<const D: usize>` to concrete `Tensor<Wgpu, 3>` (every call site was already 3D).

**Correctness**: new `tests/full_forward.rs::test_rmsnorm_fused_matches_reference`
(hidden=2048, 5 rows — exercises the prefill-shaped multi-row path — non-trivial
random gamma so a weight-indexing bug wouldn't hide behind all-ones) — max relative
error **2.97e-7** vs burn-nn's reference, two orders of magnitude under the 1e-5 bar.
Full greedy-match suite: 01 unchanged (pre-existing small-model quantization-noise
divergence, not a regression — see C2/C3 history), 02 20/20, 03 28/28, both exact.

**Dispatch count**: not re-measured with the analytical estimator (`bin/llm-agent.rs`'s
`print_dispatch_estimate` is owned by the other worker this session, not touched) —
by inspection, RMSNorm goes from ~8 dispatches x 73 calls (36 layers x 2 + out_norm) =
~584 to 1 x 73 = 73, a ~511-dispatch reduction, taking the ~1589/token estimate to
roughly **~1078/token** (estimated, not measured).

**End-to-end** (`02_tools_single`, with D1 also applied — not isolated from D1's
noise):

| | prefill tok/s | decode ms/token | greedy-match |
|---|---|---|---|
| D1 only | 29.17 | 194.9 | 20/20 |
| D1 + D2a (run 1) | 29.6* | 188.3 | 20/20 |
| D1 + D2a (`03_tools_multiturn`, run 1) | 26.95 | 181.6 | 28/28 |

(*from the combined 3-fixture suite run; 02's own line reports "37.20 tok/s" in the
raw log — two different runs in this session disagreed by ~8 tok/s on the identical
D1+D2a code, same noise-floor caveat as D1's table.)

Directionally consistent with a real (if partially noise-masked) improvement — every
D1+D2a decode number this session (188.3, 181.6) came in below every D1-only or
pre-D1 number (194.9, 213.8) — but not a clean isolated measurement given the shared
machine. No regression in any run.

### Not attempted this session

- **D2b/c (RoPE rotate-half fusion, SiLU*up fusion)**: same pattern as D2a, next
  highest dispatch-count wins per Session 2's per-op breakdown (RoPE ~10/layer, mostly
  `rotate_half`'s `cat`). Straightforward given D2a's kernel skeleton (one workgroup
  per row, elementwise) — highest-ROI remaining work.
- **D2d (residual add folded into matvec)**: requires adding an optional `+residual`
  binding + info-buffer flag to `shader_q4_matvec.wgsl`/`shader_naive.wgsl` and
  `gguf.rs::q4_matmul`'s signature — more invasive than D2a-c, not started.
- **D3 (coalesced matvec)**: not attempted. Isolated matvec GB/s was last measured at
  16-27 GB/s (Session 2, unaffected by this session's changes); the reviewer's
  word-index-interleaved-across-lanes target (≥60 GB/s) still open.
- **D4 (prefill scratch-dequant + matmul)**: not attempted. Prefill's >90% matmul-time
  bottleneck (Session 2 finding) is unchanged by D1/D2a (neither touches the M>1 naive
  kernel).

### Summary

| change | status | prefill tok/s (02, noisy) | decode ms/token (02) | greedy-match | numerics verified |
|---|---|---|---|---|---|
| K5 baseline (session 2 end) | — | 29.6 | ~190 | y | y |
| D1 (KV cache in-place) | shipped, no isolated measured win | 29-36 (noise) | 195-214 (noise) | y | y (bit-identical logic path when it does write in place; argmax-identical either way) |
| D2a (fused RMSNorm) | shipped | 27-37 (noise) | 181-188 | y | y (2.97e-7 rel. err. vs reference) |

Neither this session's changes nor Session 2's moved decode ms/token or prefill tok/s
outside this machine's run-to-run noise band in a way a single before/after pair can
prove — the directional evidence (every post-D2a decode number beat every pre-D2a
number) is suggestive, not conclusive. **Single most valuable next change**: D2b
(RoPE fusion) — same low-risk elementwise-kernel pattern as D2a, next-largest
dispatch-count item, and (unlike D1) has a plausible mechanism to show up in wall time
regardless of GPU-copy-vs-view uncertainty, since it removes real dispatch-submission
overhead rather than a memory-bandwidth cost that may already be masked by unified
memory. After D2b/c/d close out D2, D3's coalesced-matvec rewrite is the largest
remaining lever (16-27 GB/s measured vs ≥60 GB/s target, decode's matvec bucket is
still ~half of total step time per Session 2's P1c breakdown) — but it's also the
highest-risk item (K2's tiled-matmul attempt shows this GPU/driver combination doesn't
respond to naive tiling/vectorization the way the reviewer's model predicted; treat
D3's target as aspirational, not guaranteed, and budget time to fall back to "ship the
best measured variant" rather than chasing 60GB/s if early attempts plateau like K2
did).
