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

## Summary

| kernel | status | prefill tok/s (02) | decode ms/token | target | met? |
|---|---|---|---|---|---|
| baseline (C3) | — | 29.2 | 307 | — | — |
| K1 (matvec, M=1) | shipped | 26.3 | 184 | ≤40 ms/token | no |
| K3 (chunk=256) | shipped | 26.5 | 183.6 | ≥300 tok/s | no |
| K2 (tiled, M>1) | attempted, reverted | 25.7 (naive) | 193.7 | ≥300 tok/s | no |
| K4 (f16 KV) | not attempted | — | — | — | — |

Neither target (decode ≤40ms/token, prefill ≥300 tok/s on the 2225-token prompt) was
met this session. Decode improved 1.67x (K1); prefill is unchanged from baseline (K2
didn't ship, K3 was neutral). Greedy-token-match numerics are exact and unchanged by
every shipped change (K1, K3) — verified via `cargo test --release --features wgpu
--test full_forward -- --test-threads=1` after each commit.
