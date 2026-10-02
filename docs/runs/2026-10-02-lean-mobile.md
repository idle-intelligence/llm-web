# 2026-10-02 lean-mobile: prefill and decode kernels for a mobile GPU

Branch `lean-mobile`, from `lean-backends` at 539451c, with 7a103f0
(write_buffer uploads) and 55ac755 (kv_scatter) cherry-picked.

## Starting point (the phone measurement this session works from)

Reported for the branch base on an Android phone (Chrome, Adreno 6xx,
shader-f16, timestamp-query), Qwen2.5-0.5B-Instruct Q4_0, 36-token prompt,
`backends.html?backend=webgpu&diag=1`. Not re-measured here: the phone run
of this branch is pending.

| quantity | value |
|---|---|
| prefill cold / warm | 5930 / 5868 ms |
| prefill GPU (gate_up / down / qkv / o_proj / attn / lm_head) | 5840 ms (3330 / 1676 / 400 / 308 / 54 / 53) |
| decode | 72.7 ms/token |
| decode GPU span (gate_up / lm_head / down / qkv / o_proj / attn) | 49.9 ms (18.6 / 15.1 / 8.3 / 2.3 / 1.9 / 1.6) |
| decode encode+submit (CPU) / wait | 10.0 / 62.3 ms |
| bandwidth probe read / copy | 19.1 / 17.9 GB/s |

## Changes (one commit each)

| commit | change |
|---|---|
| 3936151 | `linear_q4_small_m.wgsl`: Q4_0 prefill for 2..63 rows. Started from the lean-spec branch's `linear_q4_decode_batched.wgsl`; row groups of 8 on workgroup_id.y, word dequantised once per 8 rows, vec4 x loads, 4 lanes per output column |
| 368c789 | `linear_q4_tiled_rb.wgsl` rewritten: 64x64 tile, 16x16 threads, 4x4 per thread, one stage per 32-value Q4_0 block, [64][8] vec4 tiles with an XOR swizzle, whole-vec4 shared stores. Used for every Q4_0 prefill from 64 rows; the 64x64 `linear_q4_tiled.wgsl` (used from 512 rows) is removed |
| 4ce7881 | prefill's one-row lm_head on the decode matvec (was the naive kernel) |
| 1222ec9 | `linear_q4_decode.wgsl`: one whole Q4_0 block per lane per step (vec4<u32> qs, 8 vec4 x loads, one scale), 8 lanes per row, 16 rows per workgroup |
| 896df34 | `linear_q8_decode.wgsl`: same layout for Q8_0 (lm_head, Qwen3 layers) |
| b39c35d | cherry-pick of 55ac755: multi-row K/V cache write as one dispatch |
| 642a57b | `rope_kv_decode.wgsl`: decode RoPE of q and k plus the K/V cache write in one dispatch; a decode step is one compute pass |
| 684cc64 | decode gate/up matvec writes silu(gate) * up (GATE_UP override), the silu dispatch is gone |
| 36f427b | `Pool::use_kv_cache`: bind groups survive across generations on the same KvCache; the browser engine no longer calls `pool.reset()` per generation |
| 1a1752f | Q8_0 prefill on the same two kernels (Q8 overrides) and the same 64-row rule; `linear_q8_tiled_rb.wgsl` removed |
| f6c9784 | greedy decode: argmax id copied to the readback buffer in the step's own encoder (one submit per token) |

Per decode layer, WebGPU commands before: 10 dispatches, 4 buffer copies,
one pass end and begin. After: 8 dispatches, no copies, no pass break.

## Parameters

- Machine: the M2 laptop (Metal). Timed runs held the gpu and cargo locks,
  with no other Chrome for Testing process and no rustc running; load
  averages are listed per table.
- Browser: Playwright's Chrome for Testing 153.0.8010.12, headless, flags
  `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal
  --ignore-gpu-blocklist`. Adapter reported: apple / metal-3.
- Page: `crates/lean/www/backends.html?backend=webgpu&diag=1`, model
  Qwen2.5-0.5B-Instruct Q4_0, 36-token prompt (fixture case `short`), 64
  greedy tokens. Base served from a 539451c tree (ENGINE_BUILD
  2026-10-02-backends-03), new from this branch (ENGINE_BUILD
  2026-10-02-mobile-02); interleaved base, new, base, new (5 each).
- Native: `cargo run --release -p lean --example prefill_sweep` (this
  branch's commit 89aa6a3; the same file copied into the base tree), same
  prompt repeated to length, median of 3 reps after one warm-up, decode =
  median of 31 steps after a 36-token prefill.

## Results

### Browser, M2, ABAB x5 (load average 2.2-2.7)

| | base 539451c | this branch |
|---|---|---|
| prefill ms, median (min / max) | 159.0 (158.4 / 166.0) | 82.1 (73.6 / 82.9) |
| prefill warm ms, median (min / max) | 154.0 (151.3 / 179.0) | 66.3 (66.2 / 69.1) |
| decode ms/token, median (min / max) | 11.0 (11.0 / 11.3) | 7.1 (7.1 / 7.2) |
| decode encode+submit median | 0.2 ms | 0.2 ms |
| token hash (all 10 runs) | a454748c60e23841 | a454748c60e23841 |

Values are the page's result rows `prefill ms`, `prefill warm`,
`decode ms/token`, `decode encode+submit median` and `token hash`.

### Browser, M2, GPU time per op group (diag split, one run each)

| op group | base prefill | new prefill | base decode/step | new decode/step |
|---|---|---|---|---|
| qkv | 11.9 | 3.9 | 0.5 | 0.3 |
| attn | 15.5 | 15.5 | 0.6 | 0.6 |
| o_proj | 7.2 | 2.7 | 0.4 | 0.3 |
| gate_up (new decode: incl. silu) | 69.2 | 26.2 | 3.4 | 1.9 |
| silu | 0.5 | 0.5 | 0.1 | - |
| down | 38.7 | 13.1 | 1.2 | 0.9 |
| lm_head | 4.9 | 1.7 | 2.1 | 1.6 |
| in-pass total | 148.6 | 64.3 | 9.0 | 6.3 |
| decode gpu span median (unsplit) | | | 9.7 | 6.1 |

All values ms.

### Native, M2, Qwen2.5-0.5B Q4_0, final sweep (ABAB x2)

Round 1 load 2.09 (base) / 1.41 (new); round 2 load 1.51 / 1.84. Round 2
base had wide spreads at 512 and 1024 (min/max 5348/6361 and
16687/26655 ms) and a higher decode time than round 1; treat round 2 as
disturbed.

| prompt tokens | base r1 | new r1 | base r2 | new r2 |
|---|---|---|---|---|
| 8 | 141.2 | 41.2 | 142.9 | 40.8 |
| 16 | 216.6 | 129.1 | 213.8 | 121.1 |
| 36 | 364.9 | 123.3 | 365.4 | 130.4 |
| 64 | 361.9 | 166.8 | 364.7 | 160.1 |
| 128 | 645.2 | 268.5 | 645.1 | 269.6 |
| 256 | 1267.8 | 477.8 | 1268.4 | 550.8 |
| 512 | 5234.1 | 1253.5 | 5902.5 | 1466.1 |
| 1024 | 11422.7 | 3566.1 | 21470.6 | 3824.9 |
| decode ms/step | 11.32 | 8.34 | 14.39 | 8.01 |

Prefill ms, median of 3.

### Native, M2, kernel choice per prompt length (during development)

Whole-prefill ms (median of 3) with the Q4_0 linear forced onto one kernel
for every M >= 2 (temporary local override, not in the branch). Load 1.2-2.1.
The 8- and 4-lane columns are the second of two interleaved rounds.

| prompt tokens | base | small-M, 32 lanes/col | small-M, 8 lanes | small-M, 4 lanes (final) | 64x64 tiled, final |
|---|---|---|---|---|---|
| 8 | 115.0 | 87.0 | 53.7 | 53.7 | - |
| 16 | 207.6 | 126.0 | 86.1 | 88.1 | - |
| 36 | 328.0 | 203.9 | 125.2 | 121.9 | 151.7 |
| 64 | 360.9 | 313.0 | 187.6 | 147.9 | 156.2 |
| 128 | 640.8 | 601.8 | 319.4 | 270.3 | 271.4 |
| 256 | 1249.3 | 1207.6 | 578.1 | 516.6 | 488.5 |
| 512 | 5231.5 | 2762.2 | - | 1373.8 | 1289.1 |
| 1024 | 14745.8 | 6840.0 | - | 3774.7 | 3812.8 |

The first tiled rewrite (scalar [TK][64] tiles, before the vec4 swizzle)
measured 219.9 / 235.5 / 427.6 / 831.4 / 2207.8 / 5267.5 ms at 36 / 64 /
128 / 256 / 512 / 1024 tokens in the same session.

### Native, M2, Qwen3-0.6B Q8_0 (ABAB x2, load 1.2-1.6)

| | base r1 | new r1 | base r2 | new r2 |
|---|---|---|---|---|
| prefill 36 tokens | 450.1 | 138.2 | 448.3 | 132.7 |
| prefill 128 tokens | 890.0 | 398.3 | 890.9 | 397.1 |
| prefill 512 tokens | 4290.1 | 2340.7 | 4288.1 | 2333.9 |
| decode ms/step | 15.19 | 11.01 | 16.08 | 11.03 |

### Gates (native, M2, final branch state)

| test | result |
|---|---|
| fixture_parity | ok (116.0 s) |
| fixture_parity_qwen25_3b | ok (40.9 s) |
| fixture_parity_qwen3 | ok (104.7 s) |
| fixture_parity_qwen3_1_7b | ok (227.9 s) |
| fixture_parity_llama_360m | ok (14.2 s) |
| kv_snapshot | ok, 2 tests (52.2 s) |
| logit_mask | ok, 3 tests (3.8 s) |
| fixture_parity_llama_360m_cpu | ok (19.9 s) |
| pool_reuse (new) | ok (3.3 s); fails with the `use_kv_cache` check disabled |
| backends.html webgpu, M2 | a454748c60e23841, same as transformers |
| backends.html threads, M2 | a454748c60e23841, same as transformers |
| cargo clippy -D warnings, native all targets and wasm32 `web` | clean |

## Observations

- The phone's prefill was 98% two MLP matmuls on the old 32x32 tiled
  kernel (5006 of 5840 ms). On the M2 browser the same op groups went from
  107.9 to 39.3 ms (gate_up + down), qkv from 11.9 to 3.9 ms.
- The small-M kernel with 32 lanes per output column (the decode layout)
  was 1.6x slower at 36 tokens than with 4 or 8 lanes (203.9 vs 121.9 /
  125.2 ms): at K = 896 (qkv, gate_up) each lane had about one word per
  row, so the reduction dominated.
- Small-M (4 lanes) and the final 64x64 tiled kernel cross between 64 and
  256 rows on the M2 (147.9 vs 156.2 at 64, 270.3 vs 271.4 at 128, 516.6
  vs 488.5 at 256). The branch switches at 64 rows: below that the tile is
  mostly padding, above it each dequantised weight stage is reused across
  64 rows instead of 8, which matters more on a device with the phone's
  19 GB/s than on the M2.
- A 64x64 tile filled with per-component stores into vec4 shared memory
  from four threads broke parity on Metal (top-1 mismatch in
  fixture_parity); scalar tiles and then whole-vec4 stores both pass.
- The old 64x64 kernel from 512 rows up was slower than every replacement
  measured here (5231.5 ms at 512 tokens vs 1289.1 for the new tiled
  kernel).
- Decode GPU span on the M2 browser went from 9.7 to 6.1 ms per token
  (gate_up 3.4 -> 1.9 including silu, lm_head 2.1 -> 1.6). Decode
  encode+submit stayed at 0.2 ms on the M2, so the CPU-side change
  (fewer commands per layer, one pass, one submit, no per-generation
  bind group rebuild) can only be judged on the phone.
- Prefill attention is unchanged (15.5 ms of 64.3 at 36 tokens on the M2
  browser; 502-762 ms of the 512-token native prefill depending on the
  variant). Two attempts at a faster prefill attention (pipeline-constant
  head_dim with 64-row tiles; K/V tiles in shared memory) did not improve
  it and are not on the branch.
