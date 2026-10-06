# 2026-10-03 lean-prefill: short-prompt prefill matmuls for a mobile GPU

Branch `lean-prefill`, from `lean-next` at b63ff51.

## Starting point

Reported for b63ff51 (the lean-mobile kernels) on an Android phone
(Chrome, Adreno 6xx), Qwen2.5-0.5B-Instruct Q4_0, 36-token prompt,
`backends.html?backend=webgpu&diag=1`. Not re-measured here.

| quantity | value |
|---|---|
| prefill | 954 ms |
| prefill GPU split (gate_up / down / qkv / o_proj / attn / lm_head) | 464 / 230 / 58 / 44 / 57 / 11 ms |
| bandwidth probe read | 19.1 GB/s |

Same prompt on the M2 browser (docs/runs/2026-10-02-lean-mobile.md): gate_up
26.2, down 13.1, attn 15.5 ms.

## Analysis of the previous small-M kernel (M = 2..63)

- Per layer, gate_up is a 36 x 896 by 896 x 9728 matmul. Its Q4_0 weights
  are about 4.9 MB per layer, 118 MB for 24 layers. The kernel ran 8-row
  groups, so the weights were read 5 times at M = 36: about 590 MB, about
  31 ms at the phone's 19.1 GB/s probe. The measured 464 ms is about 15x
  that, so the weight re-reads do not explain it.
- For every Q4_0 word (8 weights) each thread loaded 16 x vec4s (64 floats)
  for 8 rows from the storage buffer: one global float load per
  multiply-add. Over gate_up that is M x N x K x 4 B = 1.25 GB of x loads
  per layer, 30 GB per prefill.
- Phone/M2 ratios at the same prompt: gate_up 17.7x, down 17.6x, qkv 15x,
  versus 6.5x for the bandwidth-bound decode matvec (lm_head) and 3.7x for
  attention. 464 ms for 7.5 GMAC is about 16 GMAC/s.
- Working hypothesis: the kernel is limited by the x loads, which on the
  phone go to a cache level far from the ALUs, and not by DRAM bandwidth
  or arithmetic.

## Change

| commit | change |
|---|---|
| 652f98c | `linear_q4_small_m.wgsl` rewritten (Q4_0 and its Q8 override). A workgroup covers 8 query rows x 64 output columns with 64 threads: 16 column lanes x 4 K-split lanes. Each step stages 4 blocks (128 k) of the 8 x rows in workgroup memory. A thread loads one vec4<u32> per column per block (a whole Q4_0 block), dequantises it once and dots it against the 8 staged rows (broadcast reads). It keeps 8 rows x 4 columns in 8 vec4 accumulators. The 4 K-split partial sums are added through the same workgroup array (8 KiB in all). The grid is (row groups, column tiles), so workgroups that read the same columns are dispatched next to each other |
| 160d015 | ENGINE_BUILD 2026-10-03-prefill-01 on every loading URL |

Per thread and k4 slot: 8 workgroup-memory vec4 loads for 128
multiply-adds (the previous kernel made 16 global vec4 loads for 64).

## Parameters

- Machine: the M2 laptop (Metal). The timed runs held the gpu and cargo
  locks and started when the 1-minute load average was below 3 with no
  rustc running. No other Chrome for Testing process was running.
- Browser: Playwright's Chrome for Testing, headless, flags
  `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal
  --ignore-gpu-blocklist`. Adapter reported: apple / metal-3.
- Page: `crates/lean/www/backends.html?backend=webgpu&diag=1`,
  Qwen2.5-0.5B-Instruct Q4_0, 36-token prompt, 64 greedy tokens. Base
  served from a b63ff51 tree (ENGINE_BUILD 2026-10-03-next-01), new from
  160d015 (ENGINE_BUILD 2026-10-03-prefill-01). The served wasm hashes were
  checked against the builds. Runs interleaved base, new (5 each).
- Native: `cargo run --release -p lean --example prefill_sweep -- --lens
  8,16,28,36,48,63 --reps 3 --decode 8`, base and new binaries
  interleaved (5 rounds). Each value is the median of 3 reps after one
  warm-up. The table gives the median of the 5 round medians, with the min
  and max of those medians.

## Results

### Browser, M2, ABAB x5 (load average 1.9-2.0)

| | base b63ff51 | this branch |
|---|---|---|
| prefill ms, median (min / max) | 78.0 (73.3 / 83.3) | 68.9 (67.9 / 74.4) |
| prefill warm ms, median (min / max) | 66.3 (65.9 / 66.6) | 60.9 (60.7 / 61.0) |
| decode ms/token, median | 7.1 | 7.1 |
| token hash (all 10 runs) | a454748c60e23841 | a454748c60e23841 |
| GPU split gate_up, median | 26.2 | 21.8 |
| GPU split down | 13.1 | 12.2 |
| GPU split qkv | 3.8 | 4.3 |
| GPU split o_proj | 2.7 | 2.6 |
| GPU split attn | 15.5 | 15.4 |
| GPU split lm_head | 1.7 | 1.7 |

Values are the page's result rows; GPU splits in ms.

One run of an experiment build (not on the branch) that sends every
prefill from 2 rows up to the 64x64 tiled kernel (`SMALL_M_MAX_ROWS = 2`,
ENGINE_BUILD 2026-10-03-prefill-01-tiled): prefill 94.4 ms, warm 83.1 ms,
gate_up 31.5, down 18.9, qkv 8.5, o_proj 3.6 ms, same token hash.

### Native, M2, ABAB x5 (load average 1.5-2.0)

| prompt tokens | base b63ff51 | this branch |
|---|---|---|
| 8 | 41.2 (40.8 / 41.4) | 41.4 (40.8 / 43.9) |
| 16 | 115.9 (111.5 / 129.9) | 160.8 (85.4 / 162.9) |
| 28 | 140.1 (112.6 / 160.1) | 116.0 (88.6 / 134.1) |
| 36 | 148.7 (134.3 / 179.9) | 122.8 (121.9 / 125.3) |
| 48 | 124.2 (120.0 / 177.6) | 142.4 (123.4 / 160.4) |
| 63 | 159.5 (158.7 / 170.6) | 159.6 (155.8 / 160.8) |
| decode ms/step | 9.0 (8.2 / 12.3) | 9.6 (8.2 / 10.9) |

Prefill wall ms.

### Development note (M2 native, under load 5-10, not comparable)

The first version of the rewrite kept the accumulators in module-scope
`var<private>` variables updated by a helper function. In the same
session, base / that version / final took 148 / 456 / 96 ms for the
36-token prefill, with gate_up 42 / 210 / 26 ms. With the accumulators as
function-local `var`s, the Metal build ran at the base's speed. With the
reduction array merged into the x tile (12.8 to 8 KiB of workgroup memory),
it ran faster than the base.

### Phone (Adreno 6xx, Android Chrome, 36 tokens)

Run by hand, values as reported back. All runs gave token hash
a454748c60e23841.

| build | prefill | prefill warm | gpu span | gate_up | down | qkv | o_proj | attn | lm_head | decode ms/token | probe read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03-next-01 (b63ff51) | 953.7 | 943.5 | | 464.0 | 229.8 | 57.6 | 43.9 | 56.6 | | | |
| 2026-10-03-prefill-01 (160d015) | 565.7 | 469.2 | 440.9 | 203.3 | 109.9 | 29.8 | 21.7 | 59.1 | 10.7 | 62.9 | 17.8 |
| 2026-10-03-prefill-01-tiled (experiment) | 2075.1 | 2038.4 | | 1054.1 | 603.3 | 166.4 | 112.5 | 55.4 | | | |

### Gates (native, M2, at 652f98c)

| test | result |
|---|---|
| fixture_parity (both kernel paths) | ok (179.1 s) |
| fixture_parity_qwen25_3b | ok (77.9 s) |
| fixture_parity_qwen3 | ok (151.3 s) |
| fixture_parity_qwen3_1_7b | ok (369.4 s) |
| fixture_parity_llama_360m | ok (22.2 s) |
| kv_snapshot | ok, 2 tests (109.0 s) |
| logit_mask | ok, 3 tests (5.4 s) |
| pool_reuse | ok (4.6 s) |
| embed_head_sliced (incl. adapter swap) | ok, 2 tests (4.8 s) |
| cargo clippy -D warnings, native all targets and wasm32 `web` | clean |
| backends.html webgpu, M2 | a454748c60e23841, same as transformers |

## Observations

- On the M2 browser the rewrite takes the 36-token prefill from 78.0 to
  68.9 ms (warm 66.3 to 60.9). gate_up drops from 26.2 to 21.8 ms and down
  from 13.1 to 12.2 ms. qkv rises from 3.8 to 4.3 ms: at N = 1152 the
  kernel runs 18 column tiles x 5 row groups = 90 workgroups of 64 threads.
- The native wall-time sweep has spreads of 30-50 ms at most lengths in
  both builds. Only 36 tokens (148.7 vs 122.8) and 8 / 63 tokens (equal)
  are stable enough to read. The browser's GPU split is the M2 number to
  use.
- The 64x64 tiled kernel at 36 rows was slower on the M2 than both small-M
  kernels (gate_up 31.5 ms). Whether that holds on the phone is what the
  experiment build measures.
- On the phone the rewrite took gate_up from 464.0 to 203.3 ms (-56%),
  down from 229.8 to 109.9 (-52%), qkv from 57.6 to 29.8 and o_proj from
  43.9 to 21.7 ms. Prefill went from 953.7 to 565.7 ms (warm 943.5 to
  469.2). x traffic from the storage buffer fell from about 1.25 GB to
  about 4.9 MB per layer for gate_up at 36 rows.
- The 64x64 tiled kernel at 36 rows took 1054.1 ms for gate_up on the
  phone, 5.2x the new small-M kernel. That kernel reads each weight once
  per 64-row tile, where the small-M kernel reads it 5 times at 36 rows.
  Per multiply-add, it makes more workgroup-memory loads, and more of them
  to distinct addresses (its weight tile loads are not broadcasts).
- On Metal, module-scope `var<private>` accumulators updated through a
  function made the kernel 5x slower than function-local variables.

## Checkpoint 2

### Changes

| commit | change |
|---|---|
| 1a5c955 | `attn_prefill_tiled.wgsl`: causal GQA prefill attention, 64 threads per (head, 8 query rows), 8 lanes per row, K/V tiles of 8 keys and the tile's q rows staged as vec4s in workgroup memory (rows padded by one vec4), online softmax per key tile, fixed vec4 accumulators. Pipelines for head_dim 64 and 128 (`HEAD_DIM` override); other head_dims keep `attn_prefill.wgsl`. Also names the small-M kernel's rows/columns per workgroup (`SMALL_M_ROWS`, `SMALL_M_COLS`) at its dispatch sites |
| 504ca0a | ENGINE_BUILD 2026-10-03-prefill-02 on every loading URL |

### Small-M matmul variants (experiments, not on the branch)

Generated from one template: R query rows x C columns per thread, 16
column lanes x 4 K-split lanes, 64 threads, x staged in workgroup memory.
Per thread and k4 slot: R workgroup-memory vec4 loads for 4RC
multiply-adds. Weights are read ceil(M / R) times.

| name | R x C | columns per workgroup | x loads per multiply-add (vs prefill-01) | weight reads at M = 36 | accumulator floats per thread |
|---|---|---|---|---|---|
| prefill-01 (branch) | 8 x 4 | 64 | 1x | 5 | 32 |
| A | 4 x 8 | 128 | 0.5x | 9 | 32 |
| B | 8 x 8 | 128 | 0.5x | 5 | 64 |
| C | 16 x 4 | 64 | 1x | 3 | 64 |

A first version of the template unrolled the 8 k4 slots of a block in the
source, instead of the runtime loop prefill-01 uses. On the M2 (native,
36 tokens, load 1.5-1.7) its 8 x 4 build ran gate_up in 162-164 ms against
45-52 ms for prefill-01; with a runtime slot loop the same 8 x 4 build ran
at prefill-01's speed (gate_up 24-46 vs 45-46 ms, bimodal in both). This
is the same 5x Metal slowdown as the `var<private>` version in checkpoint 1.

### Native, M2, prefill attention (one run each, load 1.4)

| prompt tokens | prefill-01 attn | tiled attn | prefill-01 wall | tiled wall |
|---|---|---|---|---|
| 36 | 33.03 | 4.28 | 119.56 | 80.94 |
| 128 | 70.35 | 10.21 | 270.00 | 194.44 |
| 512 | 683.02 | 120.11 | 1248.04 | 698.02 |

GPU split ms (`--split`) and wall ms (median of 3).

### Browser, M2, prefill-01 vs prefill-02, ABAB x5 (load average 2.1-2.8), and one run of each variant

| | prefill-01 (160d015) | prefill-02 (504ca0a) | A 4x8 | B 8x8 | C 16x4 |
|---|---|---|---|---|---|
| prefill ms | 74.7 (68.0 / 77.2) | 52.9 (52.0 / 60.9) | 89.1 | 161.0 | 183.0 |
| prefill warm ms | 60.8 (60.3 / 61.5) | 45.6 (45.5 / 45.7) | 72.4 | 147.0 | 166.9 |
| gate_up | 21.8 | 21.8 | 34.7 | 69.9 | 84.4 |
| down | 12.2 | 12.3 | 20.9 | 41.5 | 44.1 |
| qkv | 4.3 | 4.3 | 7.7 | 21.7 | 24.1 |
| o_proj | 2.6 | 2.6 | 4.2 | 8.1 | 8.5 |
| attn | 15.4 | 0.7 | 0.7 | 0.7 | 0.7 |
| decode ms/token | 7.1 | 7.1 | 7.1 | 7.1 | 7.1 |
| token hash | a454748c60e23841 (all) | a454748c60e23841 (all) | a454748c60e23841 | a454748c60e23841 | a454748c60e23841 |

Medians (min / max) over 5 runs for the first two columns; GPU split in ms.
The variants are built from 504ca0a (with the tiled attention) and differ
from prefill-02 only in the small-M kernel.

### Phone, checkpoint 2

| build | prefill | prefill warm | gate_up | down | qkv | o_proj | attn |
|---|---|---|---|---|---|---|---|
| 2026-10-03-prefill-02 | pending | | | | | | |
| 2026-10-03-prefill-02-A | pending | | | | | | |
| 2026-10-03-prefill-02-B | pending | | | | | | |
| 2026-10-03-prefill-02-C | pending | | | | | | |

### Gates (native, M2, at 1a5c955)

| test | result |
|---|---|
| fixture_parity (both kernel paths) | ok (58.5 s) |
| fixture_parity_qwen25_3b | ok (40.5 s) |
| fixture_parity_qwen3 | ok (62.8 s) |
| fixture_parity_qwen3_1_7b | ok (182.0 s) |
| fixture_parity_llama_360m | ok (13.9 s) |
| kv_snapshot | ok, 2 tests (44.4 s) |
| logit_mask | ok, 3 tests (2.9 s) |
| pool_reuse | ok (3.0 s) |
| embed_head_sliced (incl. adapter swap) | ok, 2 tests (3.6 s) |
| cargo clippy -D warnings, native all targets and wasm32 `web` | clean |
| backends.html webgpu, M2 | a454748c60e23841, same as transformers |

### Observations

- Prefill attention on the M2 browser went from 15.4 to 0.7 ms at 36
  tokens. Natively it went from 683 to 120 ms at 512 tokens. The M2
  browser prefill went from 74.7 to 52.9 ms (warm 60.8 to 45.6).
- The fixture tests with long prompts ran in less time: fixture_parity
  179.1 to 58.5 s, qwen3_1_7b 369.4 to 182.0 s, kv_snapshot 109.0 to
  44.4 s.
- On the M2, all three matmul variants are slower than prefill-01: A by
  1.6x on gate_up, B by 3.2x, C by 3.9x. B and C hold 64 accumulator floats
  per thread, against 32 for prefill-01 and A.
- The phone runs of A, B and C test separately: half the workgroup-memory
  loads per multiply-add (A, B), and fewer weight reads (C: 3 instead of 5
  at 36 rows; A: 9).
