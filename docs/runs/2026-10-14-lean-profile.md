# Where the time goes: per-group GPU time and rooflines, M2 (Metal) and RTX 3080 (Vulkan)

Measurement only. No kernel changed. Data first, then a ranked list of headroom items, each with the arithmetic behind it.

## Parameters

- Models (all Q4_0 GGUF, greedy, one request): SmolLM2-1.7B-Instruct (24 layers, hidden 2048, 32 heads, 32 KV heads, head_dim 64, intermediate 8192, vocab 49152), SmolLM2-360M-Instruct (32 layers), Qwen2.5-0.5B-Instruct (24 layers, hidden 896, 14 heads, 2 KV heads, vocab 151936). Tensor types in the SmolLM2-1.7B file: 21 of 24 `ffn_down` are Q4_0 and 3 are Q4_1 (loaded as F32 on the GPU, 67 MB each); the tied embedding / lm head is Q6_K (82.6 MB). The KV cache is F32, head-major.
- Harness: `crates/lean/examples/profile_roofline.rs` (this commit). Per-group GPU time comes from pass-level timestamp queries (`Engine::set_diag(true, true)`, one compute pass per op group, only `TIMESTAMP_QUERY` needed, which Metal grants). The per-dispatch table (`LEAN_PROFILE_KERNELS=1`, needs `TIMESTAMP_QUERY_INSIDE_PASSES`) is not available on Metal; it is available on the RTX 3080 and is shown there. The per-dispatch profile now also covers the batched prefill (it covered decode only before).
- Decode: prompt of N tokens prefilled, 2 warm-up steps, then 32 steps; the group time is the median over steps of the sum of that group's passes. "wall" is the median step time of 32 unsplit steps (`forward_decode_step_argmax`, one submit and one readback per step). Prefill: 1 warm-up, then 3 repetitions, median.
- A split run costs more than an unsplit one (extra pass boundaries), so group times are for attribution; wall is the step time. "sum of passes" is the sum of in-pass GPU time, "span" first pass begin to last pass end.
- Ceilings measured with `crates/lean/examples/peak_bench.rs` (own wgpu device, 256 MiB buffers, 9 repetitions, pass timestamps): read-only reduction, copy, and an F32 FMA loop (8 independent vec4 chains). No tuning loop.
- Mac: M2 laptop, 16 GB, Metal through wgpu 26, release build. 1-minute load average at each run is 1.5 to 1.9 (run script waits for below 2.5). Peak benchmark first run (during load 2.6 to 3.1 from other builds) gave 531 GFLOP/s FMA; with quiet load it gave 1057 GFLOP/s three times in a row, which is the value used.
- Linux test machine: Ryzen 9 3900X, RTX 3080 10 GB, Vulkan through wgpu 26, release build, GPU idle before each run (`nvidia-smi`), 1-minute load 1.4 to 1.7, clocks 1935 to 1950 MHz during runs. Sources and the three GGUF files were md5-checked against the Mac copies.
- Browsers (Mac): Chromium 153 (Playwright bundled, headless, `--use-angle=metal`), WebKit (Playwright bundled, Safari 26.6 build, headless), Firefox Developer Edition 159 (headed, puppeteer-core, throwaway profile). wasm built from this tree, md5 5cf5941146783dc429f9a9f06c0343c6, served by a no-store static server and fetched back with the same md5. Harness `scripts/bench`. 3 page loads per browser for the main phases (3 repetitions inside each), 2 loads for the 512-token decode. Load averages at the browser runs: 3.1 to 1.9 (Chromium and WebKit), 1.4 to 1.2 (Firefox).
- Exactness: browser exactness hash efa85acfe7945318 in all 9 main loads (3 browsers x 3); 128-token decode hash 941622dd5a82970f (Chromium) and 512-token decode hash 9731a7102ed3b8d7 in all 6 loads that ran it.

## Ceilings

| Device | Read (GB/s) | Copy, read+write (GB/s) | F32 FMA vec4 (GFLOP/s) | F32 FMA scalar x16 (GFLOP/s) |
|---|---:|---:|---:|---:|
| Apple M2 (Metal) | 91.4 best, 87.0 to 90.8 median over runs | 88.0 to 91.5 | 1057 (3 of 3 quiet runs) | 865 |
| RTX 3080 (Vulkan) | 711 | 705 | 18870 | 15240 |

## Dispatches

| Model | Decode, per step | Prefill, per forward (any length) | Prefill per token, 100 / 500 tokens |
|---|---:|---:|---:|
| SmolLM2-1.7B | 220 | 387 | 3.87 / 0.77 |
| SmolLM2-360M | 292 | 515 | 5.15 / 1.03 |
| Qwen2.5-0.5B | 220 | 387 | 3.87 / 0.77 |

Decode per layer: qkv (fused), rope+KV write, attention split, attention reduce, o_proj, add+norm, gate/up/SwiGLU, down, add+norm = 9 dispatches; plus embedding, lm head, argmax and the first norm = 4. Confirmed by the per-dispatch counts below.

## Decode, SmolLM2-1.7B, ms per step by context

Mac M2 (Metal)

| ms per step | ctx 128 | ctx 512 | ctx 1024 | ctx 2048 |
|---|---:|---:|---:|---:|
| qkv | 2.59 | 2.43 | 2.43 | 2.44 |
| rope | 0.10 | 0.10 | 0.10 | 0.10 |
| attn | 2.13 | 5.86 | 10.18 | 19.26 |
| o_proj | 1.02 | 0.94 | 0.93 | 0.93 |
| add+norm | 0.58 | 0.57 | 0.58 | 0.58 |
| gate_up | 6.41 | 5.86 | 5.89 | 5.90 |
| down | 5.41 | 4.97 | 4.98 | 4.98 |
| lm_head | 2.64 | 2.63 | 2.63 | 2.63 |
| argmax | 0.06 | 0.06 | 0.06 | 0.06 |
| sum of passes (split run) | 21.03 | 23.43 | 27.73 | 37.00 |
| span (split run) | 23.70 | 26.18 | 30.88 | 40.21 |
| wall per step (unsplit run) | 19.99 | 24.47 | 28.95 | 38.04 |

RTX 3080 (Vulkan)

| ms per step | ctx 128 | ctx 512 | ctx 1024 | ctx 2048 |
|---|---:|---:|---:|---:|
| qkv | 0.59 | 0.59 | 0.59 | 0.59 |
| rope | 0.09 | 0.09 | 0.10 | 0.10 |
| attn | 0.79 | 0.95 | 1.25 | 7.42 |
| o_proj | 0.34 | 0.34 | 0.34 | 0.34 |
| add+norm | 0.44 | 0.44 | 0.44 | 0.44 |
| gate_up | 0.87 | 0.87 | 0.87 | 0.87 |
| down | 1.20 | 1.20 | 1.19 | 1.19 |
| lm_head | 0.22 | 0.22 | 0.21 | 0.21 |
| argmax | 0.06 | 0.06 | 0.06 | 0.06 |
| sum of passes (split run) | 4.60 | 4.77 | 5.07 | 11.23 |
| span (split run) | 5.79 | 5.95 | 6.25 | 12.41 |
| wall per step (unsplit run) | 9.35 | 9.57 | 9.52 | 15.74 |

## Decode matvec groups against the measured read ceiling (context 128)

Bytes per step: qkv 169.9 MB (Q4_0, 18 bytes per 32 weights), o_proj 56.6 MB, gate_up 453.0 MB, down 399.5 MB (21 Q4_0 layers 198.2 MB plus 3 F32 layers 201.3 MB), lm_head 82.6 MB (Q6_K, 210 bytes per 256 weights). Total 1161.6 MB.

Mac

| group | MB read per step | ms (ctx 128) | GB/s | % of measured read ceiling |
|---|---:|---:|---:|---:|
| qkv | 169.9 | 2.59 | 65.5 | 72 |
| o_proj | 56.6 | 1.02 | 55.6 | 61 |
| gate_up | 453.0 | 6.41 | 70.7 | 77 |
| down | 399.5 | 5.41 | 73.9 | 81 |
| lm_head | 82.6 | 2.64 | 31.3 | 34 |
| all five | 1161.6 | 18.06 | 64.3 | 70 |

RTX 3080

| group | MB read per step | ms (ctx 128) | GB/s | % of measured read ceiling |
|---|---:|---:|---:|---:|
| qkv | 169.9 | 0.59 | 286.8 | 40 |
| o_proj | 56.6 | 0.34 | 165.3 | 23 |
| gate_up | 453.0 | 0.87 | 520.9 | 73 |
| down | 399.5 | 1.20 | 333.3 | 47 |
| lm_head | 82.6 | 0.22 | 383.7 | 54 |
| all five | 1161.6 | 3.22 | 360.9 | 51 |

## Decode attention, ms per step by context

KV bytes read per step (F32), SmolLM2-1.7B: 50 / 201 / 403 / 805 MB at context 128 / 512 / 1024 / 2048. SmolLM2-360M (5 KV heads): 10 / 42 / 84 / 168 MB. Qwen2.5-0.5B (2 KV heads): 3 / 13 / 25 / 50 MB. The last row of each table is the time to read the 1.7B KV bytes once at the measured read ceiling.

| model | ctx 128 | ctx 512 | ctx 1024 | ctx 2048 |
|---|---:|---:|---:|---:|
| SmolLM2-1.7B | 2.13 | 5.86 | 10.18 | 19.26 |
| SmolLM2-360M | 1.66 | 4.11 | 6.56 | 11.96 |
| Qwen2.5-0.5B | 1.08 | 2.13 | 4.43 | 8.47 |
| KV bytes floor 1.7B (ms) | 0.55 | 2.20 | 4.41 | 8.81 |

| model | ctx 128 | ctx 512 | ctx 1024 | ctx 2048 |
|---|---:|---:|---:|---:|
| SmolLM2-1.7B | 0.79 | 0.95 | 1.25 | 7.42 |
| SmolLM2-360M | 1.00 | 1.10 | 1.22 | 1.78 |
| Qwen2.5-0.5B | 0.74 | 0.81 | 0.90 | 1.28 |
| KV bytes floor 1.7B (ms) | 0.07 | 0.28 | 0.57 | 1.13 |

3080, SmolLM2-1.7B, finer steps (separate runs, 16 steps each): attention 1.34 ms at context 1100, 1.94 at 1300, 3.37 at 1536, 5.86 at 1800, 7.45 at 2048; wall per step 9.85, 9.65, 13.07, 15.13, 16.83 ms.

## Per-dispatch GPU time, RTX 3080, SmolLM2-1.7B (`LEAN_PROFILE_KERNELS=1`, per layer-call in microseconds)

| Kernel | Count per step | us per call, context 512 | us per call, context 2048 | Bytes per call | GB/s at context 512 |
|---|---:|---:|---:|---:|---:|
| qkv fused (Q4_0) | 24 | 27.8 | 27.9 | 7.08 MB | 254 |
| attention split | 24 | 37.6 | 302.9 | KV: 8.4 MB at 512, 33.6 MB at 2048 | 223 at 512, 111 at 2048 |
| attention reduce | 24 | 7.6 | 11.7 | | |
| rope + KV write | 24 | 7.0 | 7.4 | | |
| o_proj (Q4_0) | 24 | 17.4 | 17.5 | 2.36 MB | 135 |
| add + rmsnorm (two per layer) | 48 | 12.3 | 12.3 | | |
| gate/up/SwiGLU (Q4_0) | 24 | 39.4 | 39.7 | 18.87 MB | 479 |
| down, Q4_0 layers | 21 | 46.3 | 46.4 | 9.44 MB | 204 |
| down, F32 layers (from Q4_1) | 3 | 101.0 | 100.9 | 67.1 MB | 665 |
| lm head (Q6_K) | 1 | 217 | 218 | 82.6 MB | 381 |
| argmax | 1 | 63 | 63 | | |

Prefill, same device, per call (us): length 100: gate_up 939, down (Q4_0) 690, down (F32) 744, q/k/v 171 to 172 each, o_proj 171, attention 36.7, SwiGLU 17.4, rmsnorm 11.3 to 11.7, add 7.2 to 7.5; length 500: gate_up 3943, down (Q4_0) 2236, down (F32) 3482, q/k/v 542 to 544 each, o_proj 543, attention 463, SwiGLU 75, rmsnorm 22.5, add 20.9 to 21.9, rope 17.9 to 18.2, KV scatter 16.7 to 18.2. Dispatch counts match the table above (16 per layer plus 3).

## Prefill, SmolLM2-1.7B, ms per forward

FLOPs counted as 2 x tokens x weight parameters of the group x 24 layers (the F32 down layers included in "down"). Peak is the measured vec4 FMA ceiling.

Mac M2

| group | len 100 ms | GFLOP/s | % peak | len 500 ms | GFLOP/s | % peak |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 109.32 | 553 | 52 | 430.84 | 701 | 66 |
| o_proj | 36.47 | 552 | 52 | 143.22 | 703 | 66 |
| gate_up | 280.49 | 574 | 54 | 1132.63 | 711 | 67 |
| down | 142.93 | 563 | 53 | 569.85 | 707 | 67 |
| attn | 15.11 | | | 261.41 | | |
| norm | 0.58 | | | 2.31 | | |
| rope | 0.69 | | | 4.32 | | |
| kv | 1.34 | | | 5.91 | | |
| silu | 2.34 | | | 14.31 | | |
| add+norm | 0.99 | | | 5.67 | | |
| add | 0.49 | | | 3.29 | | |
| lm_head | 2.65 | | | 2.65 | | |
| TOTAL_PASS_SUM | 593.67 | | | 2577.57 | | |
| TOTAL_SPAN | 625.20 | | | 2620.48 | | |
| wall | 620.90 | | | 2616.06 | | |

RTX 3080

| group | len 100 ms | GFLOP/s | % peak | len 500 ms | GFLOP/s | % peak |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 9.13 | 6614 | 35 | 36.68 | 8232 | 44 |
| o_proj | 4.06 | 4957 | 26 | 12.92 | 7792 | 41 |
| gate_up | 22.55 | 7142 | 38 | 93.39 | 8623 | 46 |
| down | 16.75 | 4807 | 25 | 56.59 | 7115 | 38 |
| attn | 0.82 | | | 10.90 | | |
| norm | 0.21 | | | 0.47 | | |
| rope | 0.12 | | | 0.64 | | |
| kv | 0.17 | | | 0.62 | | |
| silu | 0.35 | | | 1.72 | | |
| add+norm | 0.29 | | | 0.88 | | |
| add | 0.11 | | | 0.43 | | |
| lm_head | 0.24 | | | 0.26 | | |
| TOTAL_PASS_SUM | 54.80 | | | 215.37 | | |
| TOTAL_SPAN | 60.42 | | | 223.50 | | |
| wall | 60.52 | | | 226.89 | | |

## Other models

Mac M2, ms per step, context 128 / 512 / 1024 / 2048 (wall per step, unsplit): SmolLM2-360M 8.30 / 11.06 / 12.51 / 18.61; Qwen2.5-0.5B 7.89 / 9.43 / 10.92 / 15.45. RTX 3080: SmolLM2-360M 5.39 / 5.53 / 5.54 / 8.46; Qwen2.5-0.5B 5.12 / 7.46 / 7.93 / 8.07.

Prefill wall, ms (length 100 / 500): Mac SmolLM2-360M 209 / 684, Qwen2.5-0.5B 187 / 699; RTX 3080 SmolLM2-360M 23.4 / 57.8, Qwen2.5-0.5B 25.7 / 61.0.

Attention share of prefill pass time at 500 tokens, Mac: SmolLM2-1.7B 261 of 2578 ms (10%), SmolLM2-360M 166 of 647 ms (26%), Qwen2.5-0.5B 117 of 659 ms (18%). RTX 3080: 10.9 of 215 (5%), 7.4 of 46 (16%), 5.3 of 50 (11%).

Mac decode, share of pass time at context 128 (SmolLM2-360M / Qwen2.5-0.5B): gate_up 25% / 28%, lm_head 9% / 23%, attention 24% / 15%, down 17% / 14%.

Full per-group tables for these two models are in the harness output (`seg` lines); the Mac split run of the 360M model shows span 15.9 ms against pass sum 7.1 ms at context 128 (292 dispatches, 32 layers, many small passes), so its split attribution carries large per-pass overhead.

## Browsers, SmolLM2-1.7B Q4_0, M2, totals per phase

Median (min to max) over page loads. Prompt tokens 100 and 500 cold; suffix = 100 or 500 cached tokens plus 24 or 124 new.

| Metric | Chromium 153 | WebKit (Safari 26.6) | Firefox Dev 159 |
|---|---:|---:|---:|
| Cold prefill 100, ttft ms | 508.8 (508.0 to 509.1) | 619.2 (619.0 to 620.1) | 596.7 (596.4 to 597.3) |
| Cold prefill 500, ttft ms | 2117.5 (2112.4 to 2119.1) | 2707.2 (2706.1 to 2727.5) | 2584.8 (2582.5 to 2586.6) |
| Suffix 100 + 24, ttft ms | 149.3 (149.1 to 149.4) | 187.0 (186.9 to 187.0) | 173.4 (173.2 to 173.6) |
| Suffix 500 + 124, ttft ms | 580.3 (579.3 to 580.5) | 789.8 (789.8 to 793.7) | 743.9 (743.1 to 745.6) |
| Decode 128 tokens, tok/s | 47.6 (47.5 to 47.6) | 50.8 (50.7 to 50.8) | 53.6 (53.4 to 53.8) |
| Decode 512 tokens, tok/s | 43.3 (43.3, 43.3) | 45.3 (45.3, 45.3) | 47.9 (48.1, 47.6) |

Native Mac, same model, same kernels (this harness, release): cold prefill 100 tokens 620.9 ms, 500 tokens 2616 ms; decode wall 19.99 ms per step at context 128 (50.0 tok/s), 24.47 ms at 512 (40.9 tok/s).

Only totals per phase were taken in the browsers. Per-group GPU time in Chromium (timestamp-query) was not taken, and the RTX 3080 browsers were not run in this session.

## Observations

- Mac decode at short context is GPU-bound: unsplit wall 19.99 ms against 21.03 ms pass sum at context 128. On the RTX 3080 the unpipelined step is not GPU-bound: wall 9.35 ms against 5.79 ms span (3.6 ms outside the GPU at context 128); the pipelined decode loop hides it.
- Matvec bandwidth on the Mac is 55 to 81% of the measured 91.4 GB/s read ceiling for the Q4_0 groups and 34% for the Q6_K lm head. On the 3080 the large fused gate/up kernel reaches 73% of 711 GB/s, while the smaller Q4_0 kernels reach 23% (o_proj) to 47% (down).
- The F32 down layers (Q4_1 in the file) run at 665 GB/s on the 3080 (94%) and take 101 us for 3 layers against 46 us per Q4_0 layer that holds a seventh of the bytes.
- Mac attention time grows linearly with context (about 9.4 us per context token per step for the 1.7B model, 0.8 ms per layer at 2048) and reaches 52% of the step at 2048. On the 3080 the 1.7B model has a steep rise beyond context 1100 (1.3 ms to 7.5 ms at 2048, 5.6x for 1.9x context); the 360M and 0.5B models do not show it (1.78 and 1.28 ms at 2048). The cause was not isolated (the run used 32 KV heads, F32 cache, head stride max_ctx x 256 bytes with max_ctx = context + 40; whether the stride or the split plan causes it is untested).
- Tiny kernels cost about 12 us each regardless of size on both devices (add+norm: 12.0 us per pass on the Mac, 12.3 us per dispatch on the 3080).
- Same WGSL, same Mac: Chromium (Dawn) cold prefill 100 tokens is 508.8 ms against native wgpu 620.9 ms (18% less), 500 tokens 2117 against 2616 ms (19% less). Native wall time is within 1% of its own span, so it is not idle gaps; the cause (shader translation, bounds-check code, scheduling) was not isolated.
- Browser order on the M2: Chromium fastest at prefill (509 against 597 and 619 ms), Firefox Dev 159 fastest at decode (53.6 against 50.8 and 47.6 tok/s).

## Where the headroom is

Estimates are the difference between a measured time and a floor computed from bytes or FLOPs at a measured ceiling; the floor is not claimed to be reachable. M2 SmolLM2-1.7B Q4_0 unless stated.

1. Prefill matmul (96% of Mac prefill pass time at 100 tokens, 569 of 594 ms, and 88% at 500 tokens, 2276 of 2578 ms). Achieved 553 to 574 GFLOP/s at 100 tokens (52 to 54% of 1057) and 701 to 711 at 500 (66 to 67%). At 85% of the ceiling (898 GFLOP/s): 322 GFLOP takes 0.36 s against 0.57 s (saves 0.21 s, 34% of the 100-token forward) and 1.61 TFLOP takes 1.79 s against 2.28 s (saves 0.49 s, 19% of the 500-token forward). On the 3080 the groups reach 25 to 46% of 18.9 TFLOP/s; at 70% (13.2 TFLOP/s), length 500 matmuls take 122 ms against 199 ms (saves 77 ms, 34% of 227 ms).
2. Decode attention at long context (KV is F32). Mac: measured 5.86 / 10.18 / 19.26 ms at 512 / 1024 / 2048 against a read floor of 2.20 / 4.41 / 8.81 ms, so 3.7 / 5.8 / 10.5 ms per step, which is 15% / 20% / 27% of the unsplit step (24.47 / 28.95 / 38.04 ms). Storing K/V in f16 halves the bytes: a further 1.1 / 2.2 / 4.4 ms on the floor. 3080: 7.42 ms at 2048 against a 1.13 ms floor, 6.3 ms, 51% of the 12.41 ms GPU span; 0.95 ms against 0.28 ms at 512.
3. Prefill attention. Mac, 500 tokens: 261 ms for 24.6 GFLOP of causal attention, 94 GFLOP/s (9% of 1057), 10% of the forward for the 1.7B model and 26% and 18% for the 360M and 0.5B models. At 50% of the ceiling the 1.7B attention would take 47 ms (saves 214 ms, 8% of 2616 ms); for the 360M model 166 ms at 15.4 GFLOP is 93 GFLOP/s, the the 360M attention is 26% of its forward (about 135 ms of 684 ms). The cost grows with the square of the length.
4. Q4_1 `ffn_down` loaded as F32 (3 of 24 layers in the SmolLM2 files, also in the 360M file). 201.3 MB read per step against 28.3 MB as Q4_0: 173 MB, 1.9 ms at 91.4 GB/s (9.5% of the 19.99 ms step at context 128), plus 173 MB of GPU memory (13% of the 1346 MB weight total). On the 3080 about 0.24 ms (4% of the span). Needs a Q4_1 matvec and matmul kernel.
5. Q6_K lm head matvec on the Mac. 2.64 ms for 82.6 MB (31 GB/s) against 0.90 ms at the ceiling: 1.7 ms, 8.7% of the step at context 128 (4.6% at 2048). On the 3080 it is 0.22 ms against 0.12 ms, so 1.7% of the span. Qwen2.5-0.5B's Q8_0 head runs at 88 GB/s (144.6 MB in 1.65 ms), at the ceiling.
6. Q4_0 matvec efficiency on the Mac. The four groups other than the head read 1079 MB in 15.4 ms (69.9 GB/s). At 85% of the ceiling (77.7 GB/s) they take 13.9 ms: saves 1.5 ms (8% of the step). On the 3080 the five groups take 3.22 ms against a 1.63 ms floor at 711 GB/s: 1.6 ms, 28% of the GPU span, visible once decode is not CPU-bound.
7. Small kernels (about 12 us each). 3080 per step, from the per-dispatch table: add+norm 0.59 ms, rope+KV write 0.17 ms, attention reduce 0.18 ms, together 0.94 ms, 17% of the 5.4 ms profiled step. Mac: add+norm 0.58 ms plus rope 0.10 ms, 3% of the step. Removing a dispatch saves about 12 us times 24 layers = 0.29 ms per step on both.
8. Unpipelined step on the 3080: 3.6 ms outside the GPU per step (38% of the 9.35 ms wall at context 128). The block-readback loop already used by the browser pages is the mitigation; no per-token gain beyond it was measured here.

Not measured: Chromium timestamp-query per-group time, the RTX 3080 browsers, and the effect of any of the above changes.
