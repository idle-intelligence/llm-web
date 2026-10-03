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

| build | prefill | gate_up | down | qkv | o_proj | attn | lm_head |
|---|---|---|---|---|---|---|---|
| b63ff51 (reported) | 954 | 464 | 230 | 58 | 44 | 57 | 11 |
| 2026-10-03-prefill-01 | pending | | | | | | |
| 2026-10-03-prefill-01-tiled (experiment) | pending | | | | | | |

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
- The working hypothesis (x loads, not weight bandwidth) is tested only by
  the phone run of 2026-10-03-prefill-01. Its x traffic from the storage
  buffer is K x 4 B per row and column tile (about 4.9 MB per layer for
  gate_up at 36 rows, down from 1.25 GB).
- On Metal, module-scope `var<private>` accumulators updated through a
  function made the kernel 5x slower than function-local variables.
