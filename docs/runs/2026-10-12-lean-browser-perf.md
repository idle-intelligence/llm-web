# Lean in the browser: Firefox, Safari engine and Chromium, before and after

Branch work on prefill and decode submission. Data first, observations after.

## Parameters

- Model: SmolLM2-1.7B-Instruct, Q4_0 GGUF (994 MB), maxCtx 2048, greedy.
- Harness: `scripts/bench` (bench.html and bench_worker.js, served by serve.mjs). Phases: cold prefill (fresh prompt of N tokens, 1 new token, time to first token), suffix prefill (prefix of N tokens cached, then the extra tokens, 24 new on 100 and 124 new on 500), decode (64 tokens, ms per token between first and last), exactness (greedy replies to three fixed questions, hashed).
- Cold and suffix: 3 repetitions inside each page load; table values are the median of 3 page loads, interleaved A B A B A B (before = origin/main wasm, after = this branch, tag final2, 3 decode steps in flight).
- wasm hash of the after build, identical on the build machine and as served on the M2: 02bc858c7d7a1a1d. Both builds served from a no-store static server under distinct URL tags.
- M2 laptop (16 GB), 1-minute load average 1.1 to 1.9 at run time. Chromium: Playwright bundled headless Chromium 153 (Metal via ANGLE). Safari engine: Playwright bundled WebKit (Safari 26.6 build), headless, WebGPU present. Firefox on the M2: see below.
- Linux test machine (RTX 3080, 10 GB): Firefox 154 headed on Wayland, fresh profile with WebGPU forced on. One GPU job at a time, GPU idle before each run.
- Gates before these runs: lean-core fixture_parity (ok), `cargo test` for lean-core, lean, lean-chat in release (ok), `cargo clippy --all-targets -- -D warnings` and the two wasm32 clippy invocations (ok after the `is_multiple_of` fix), wasm build with simd128 retained in `target_features`.

## Results: M2, Chromium (ms; decode in tok/s)

| Metric | Before median (min-max) | After median (min-max) |
|---|---|---|
| Cold prefill 100 tokens, ttft | 1985 (1983-1993) | 509 (508-509) |
| Cold prefill 500 tokens, ttft | 11205 (11168-11218) | 2110 (2108-2116) |
| Suffix prefill, 100 prefix + 24 new, ttft | 523 (522-525) | 149 (149-149) |
| Suffix prefill, 500 prefix + 124 new, ttft | 3176 (3168-3180) | 580 (579-580) |
| Decode tok/s | 48.0 (47.9-48.0) | 48.0 (47.9-48.2) |
| Exactness hash | efa85acfe7945318 | efa85acfe7945318 |
| Decode text hash | 8d1540a175c3a480 | 8d1540a175c3a480 |

## Results: M2, WebKit (Safari engine)

| Metric | Before median (min-max) | After median (min-max) |
|---|---|---|
| Cold prefill 100 tokens, ttft | 1856 (1855-1857) | 618 (614-619) |
| Cold prefill 500 tokens, ttft | 10649 (10641-10657) | 2708 (2707-2709) |
| Suffix prefill, 100 + 24, ttft | 488 (487-490) | 187 (187-187) |
| Suffix prefill, 500 + 124, ttft | 3110 (3107-3112) | 790 (790-790) |
| Decode tok/s | 48.8 (48.7-48.9) | 51.3 (51.3-51.4) |
| Exactness hash | efa85acfe7945318 | efa85acfe7945318 |
| Decode text hash | 8d1540a175c3a480 | 8d1540a175c3a480 |

## Results: Linux test machine, Firefox 154 (RTX 3080)

| Metric | Before | After, 3 page loads (median, min-max) |
|---|---|---|
| Model load | fails: "Not enough memory left", engine aborts (2 of 2 attempts) | loads, 2427 ms (2418-2468) |
| Cold prefill 100 tokens, ttft | n/a | 58 (57-59) |
| Cold prefill 500 tokens, ttft | n/a | 220 (220-220) |
| Suffix prefill, 100 + 24, ttft | n/a | 23 (23-24) |
| Suffix prefill, 500 + 124, ttft | n/a | 70 (70-70) |
| Decode tok/s | n/a | 227.4 (226.6-228.3) |
| Exactness hash | n/a | efa85acfe7945318 |
| Decode text hash | n/a | 8d1540a175c3a480 |
| Fresh-engine check (reuse equals fresh load) | n/a | same reply, mini phase |

## Results: Firefox on the M2 (earlier runs, installed Firefox 157, headed; not repeated)

These come from the previous session on intermediate builds, same model and prompts, one page load each, 2 repetitions. The final build was not re-measured here (see observations).

| Build | Cold 100 ttft | Cold 500 ttft | Suffix 100+24 ttft | Suffix 500+124 ttft | Decode tok/s |
|---|---|---|---|---|---|
| origin/main | 1820 | not recorded | 522 | not recorded | 9.4 |
| batched suffix prefill | 953 | 4172 | 319 | 1140 | 9.3 |
| plus 3 steps in flight (sampling decode, 48 tokens) | not run | not run | not run | not run | 9.4 |

## Observations

- Prefill is the large win and it shows in all three engines: the batched suffix pass and the tiled F32 matmul cut cold prefill 3.9x (Chromium 100 tokens), 5.3x (Chromium 500), 3.0x and 3.9x (WebKit), and suffix prefill 3.5x to 5.5x. The gain grows with prompt length (Chromium 500-token cold 11205 to 2110 ms, against 100-token 1985 to 509 ms).
- Decode on Chromium is unchanged (48.0 tok/s both), so Chromium did not regress. Decode on WebKit gains 5% (48.8 to 51.3), outside the run-to-run spread of 0.2 tok/s.
- Output is token-exact: exactness and decode hashes are identical before and after, on Chromium, WebKit and Firefox (Linux).
- On Firefox for Linux the old build could not load the 1.7B model at all (device out of memory during weight upload); with 8 MB upload pieces and an empty submit after each, it loads in 2.4 s and decodes at 227 tok/s. The cause of the failure is inferred from the error and from the fix; the upload change was not tested in isolation.
- On the M2, Firefox decode stays near 9.4 tok/s (106 ms per token) across origin/main, batched prefill and in-flight decode steps, while Chromium and WebKit reach 48 to 51 tok/s on the same GPU. The per-token time is flat against changes in the amount of work queued, so the bound there is a per-step cost in Firefox's Metal path, not kernel time. No change in this branch moves it.
- Firefox on the M2 could not be re-run with the final build: the Playwright Firefox build reports "WebGPU is disabled by blocklist" and headless installed Firefox finds no adapter; only a headed run of the installed Firefox gets an adapter, which was not repeated under the rule to use bundled browsers only.
- Repeated engine loads in one Firefox page (the bench "fresh" phase loads the 1.7B model four times) abort with out-of-memory on the 10 GB card, because freeing an engine does not destroy its GPU buffers and Firefox releases them on garbage collection. Separate page loads are fine; the fresh-engine check was therefore run as the two-load mini phase on Firefox.
- No kernel, workgroup size or schedule depends on the browser or on a measurement; the same wasm was served to all three engines.

## What is left

- Firefox decode on macOS (about 9.4 tok/s against 48 on the same machine in other engines): needs profiling of the submit and read path in that browser; the in-flight steps did not change it.
- Chromium on the Linux test machine and Firefox with the final build on the M2 are not in the tables.

## Engine free destroys the GPU device

`Engine` now implements `Drop` as `device.destroy()`. Each engine owns its device, so freeing a `Lean` releases every buffer, texture and bind group at once instead of when the browser collects the wrapper objects (Firefox does so late, which ran the 10 GB card out of memory on repeated loads).

- Harness: `scripts/bench`, phase `loads` (`phases=loads&reps=4&decode=4000`): load SmolLM2-1.7B Q4_0, answer one fixed question greedily (32 tokens), free, wait 4 s, repeat 4 times. GPU memory is read from `nvidia-smi` (whole card) at each "settled k" status.
- Linux test machine (RTX 3080), Chromium 152 headed on Xwayland with Vulkan, card at 505 MiB before the run. wasm built from the same tree with and without the `Drop` impl.

| Build | GPU MiB after free, loads 2, 3, 4 (load 1 not sampled) | Peak MiB | Reply hash, 4 loads |
|---|---|---|---|
| Without destroy | 3329, 3263, 3269 | 3993 | a1b7eb2ee7a6aded x4 |
| With destroy | 989, 947, 933 | 3265 | a1b7eb2ee7a6aded x4 |

- Exactness (phases `exact,fresh`, with destroy): exactness hash efa85acfe7945318 and fresh-engine hash efa85acfe7945318, equal to the table above.
- Freeing now logs one "device lost: destroyed" per engine on the device's `lost` promise (reason `destroyed`); it is the expected result of the destroy.
- Chromium collected the buffers on its own at roughly constant size, so it showed no growth even without the change; the change returns the memory at the free call. Firefox on the Linux test machine was not run for this check (unverified there).

## 2026-10-14: greedy decode loop on the GPU, ids read back in blocks

Change: each greedy step already read its input id from the GPU buffer the previous argmax wrote. Now each step also copies its id into a per-block GPU buffer, and the host maps one block of `GREEDY_BLOCK` = 8 ids instead of one id per step, with two blocks submitted ahead (earlier: one 4-byte map per step, three steps ahead). Stopping inside a block (stop token, callback abort) discards the later steps and sets the KV length to what the step-by-step loop leaves. Greedy only; sampling is unchanged. No kernel change (the argmax kernel is the same), no public API change.

### Parameters

- Model SmolLM2-1.7B-Instruct Q4_0, maxCtx 2048, greedy. Harness `scripts/bench`, phases cold (prompt 100, 2 reps, ttft), decode (128 or 512 new tokens, mean tok/s from first to last token), exact.
- Before: wasm of the previous commit (md5 1b6a5aba2fd8b6bcc6089803fd7a61c4). After: block size 8, wasm md5 82f5a213fa8c2df85b74c1564f66b5d3, rebuilt twice to the same hash; the build served by the local preview server was fetched back and has the same hash.
- 5 page loads per cell, interleaved before, after, before, after. Each run on its own server port with cross-origin isolation headers. Medians, min-max in brackets.
- M2 laptop, 16 GB. The 1-minute load average was 2.4 to 5.0 during the runs (another process held two cores for the whole session) and throughput fell from 48 tok/s (first loads) to about 42 (128 tokens) and 34 (512 tokens) over the hour in all three browsers, before and after alike. Pairs are therefore compared within the same interleaved iteration.
- Chromium: Playwright bundled headless Chromium 153, `--use-angle=metal`. WebKit: Playwright bundled WebKit (Safari 26.6 build), headless. Firefox: Firefox Developer Edition 159, headed, driven by puppeteer-core, fresh throwaway profile, `dom.webgpu.enabled`. Playwright's bundled Firefox 155, headed with WebGPU forced on, was run once on the before build.

### Results: decode tok/s

| Browser | Tokens | Before | After |
|---|---|---|---|
| Chromium 153 | 128 | 42.83 (38.91-48.34) | 42.02 (39.13-47.88) |
| Chromium 153 | 512 | 34.20 (33.42-43.80) | 33.66 (33.01-43.55) |
| WebKit (Safari 26.6) | 128 | 42.10 (38.73-50.85) | 42.32 (39.83-50.77) |
| WebKit (Safari 26.6) | 512 | 34.41 (33.51-44.75) | 34.10 (33.17-34.42) |
| Firefox Dev 159 | 128 | 43.81 (42.03-54.03) | 43.71 (42.25-53.49) |
| Firefox Dev 159 | 512 | 35.36 (34.83-41.58) | 34.93 (33.56-35.28) |

First loads of each cell, before the machine slowed: Chromium 48.2 and 47.9 (128 tokens), WebKit 50.9 and 50.8, Firefox Dev 159 54.0 and 53.5, before and after.

### Results: cold prefill 100 tokens, ttft ms, median of 5

| Browser | Before | After |
|---|---|---|
| Chromium | 579 | 571 |
| WebKit | 744 | 730 |
| Firefox Dev 159 | 699 | 694 |

### Results: block size, 128 tokens, tok/s, 3 loads interleaved

| Browser | K=4 | K=8 | K=16 |
|---|---|---|---|
| Chromium | 36.94 (36.84-40.64) | 36.65 (36.52-38.25) | 37.30 (37.20-37.36) |
| Firefox Dev 159 | 43.04 (42.79-43.40) | 42.79 (42.49-43.31) | 42.50 (42.20-42.50) |

### Correctness

- Decode text hash for 128 and 512 tokens is the same before and after in every run (941622dd5a82970f and 9731a7102ed3b8d7); exactness hash efa85acfe7945318 in all 60 runs plus the 18 block-size runs; 64-token decode hash 8d1540a175c3a480 on all three browsers; loads phase reply hash a1b7eb2ee7a6aded x3 (Chromium, after, with the destroy-on-drop change).
- Native on the M2 (Metal, release): `cargo test -p lean --release -- --ignored` with Qwen2.5-0.5B, SmolLM2-360M and SmolLM2-1.7B Q4_0: lib 6 of 6, `engine_api` 14 of 14 (reuse equivalence, diverging prefixes, CPU equals GPU, overflow recovery), `fixture_parity_generate` Qwen2.5-0.5B and SmolLM2-360M ok (the Llama-3.2-1B case was not run, its model is not on this machine); lean-core `fixture_parity` ok; non-ignored suites of lean-core, lean, lean-chat pass.
- `pipelined_greedy_matches_step_loop` now stops after 1, 7, 8, 9, 16, 17 and 40 tokens (block edges and mid-block) and checks ids, the returned last id, the KV length, and that the next step continues identically.
- clippy `-D warnings`: native all targets, wasm32 web, wasm32 web+diag, all clean.

### Observations

- Firefox Developer Edition 159 and Playwright's Firefox 155 decode at 54 to 55 tok/s on the before build (first loads), in line with Chromium (48) and WebKit (51) on this machine. The 9.4 tok/s (106 ms per token) measured earlier was on Firefox 157 release; it did not reproduce on 155 or 159 here, so the block readback could not be shown to remove that stall. The release Firefox on this machine was not run (not used for tests).
- On the browsers that were measured the change is neutral: differences between before and after are inside the run-to-run spread in every cell, and block sizes 4, 8 and 16 are not distinguishable.
- Tokens now reach the callback in groups of up to 8.

### What is left

- Check the build on Firefox 157 release (the browser with the slow readback) with the chat page: expected effect is one map wait per 8 tokens instead of per token; unverified.
- Quiet-machine rerun of the tables above.

## 2026-10-14: Q4_1 kept quantized on the GPU, Q6_K decode matvec

### Parameters

- Model: SmolLM2-1.7B-Instruct "Q4_0" GGUF. Tensor mix measured from the file: 165 Q4_0 (1560 M weights), 3 Q4_1 (`ffn_down` of three layers, 50.3 M weights, 31.5 MB), 1 Q6_K (tied embedding and output, 100.7 M weights, 82.6 MB), 49 F32 (norms, 0.4 MB).
- Before: tree at the logit_dump commit (no kernel change). After: this branch. The 3 Q4_1 tensors were dequantized to F32 at load (201.3 MB, read in full every step); the Q6_K head ran the scalar byte-by-byte matvec.
- Q4_1: the nibbles repack as for Q4_0, `d` and `m` are stored as interleaved f32 pairs in the scales buffer, and the Q4_0 matvec and matmul kernels gain a `Q41` override (value = nibble * d + m): decode with one and with four rows per lane, short and long prefill, and the naive reference kernel. The dequantized values are the same as the F32 load.
- Q6_K: units of 64 weights read as vec4<u32>, 8 lanes per row with consecutive lanes on consecutive units, four 6-bit values assembled per u32 and converted with the exponent trick.
- Native decode: `crates/lean/examples/profile_roofline` (decode, 32 steps after 2 warm-up steps; group times from pass timestamps in a split run, wall from unsplit steps), release build, interleaved before/after (ABAB), 5 pairs per context on the M2, median with min and max. Prefill: len 100 and 500, 3 pairs. M2 load average 1.4 to 2.5 at the runs (run script waits for below 2.5), under the shared GPU lock. RTX 3080: 3 pairs at context 128 under the shared GPU lock, 1-minute load 0.8 to 1.5; context 512 and the box test suites were not run (the box stopped answering over SSH for the last hours of this work).
- Browsers (M2): Chromium 153 (`--use-angle=metal`), WebKit (Playwright build), Firefox Developer Edition 159 headed with a throwaway profile; harness `scripts/bench`, phases `decode,exact` at 128 tokens (3 loads per variant on Chromium, 2 on WebKit and Firefox) and `decode` at 512 tokens (1 load each), before and after interleaved. Served bytes checked by fetching `lean_bg.wasm` back from each server: before 5cf5941146783dc429f9a9f06c0343c6, after 62bf3b6f8cfe82925e6fd87841ebd840 (ENGINE_BUILD 2026-10-14-quant-01 on all four loading URLs).
- Ceilings (from the profile run doc): M2 read 91.4 GB/s, RTX 3080 read 711 GB/s.

### Results: native decode, M2 (Metal), SmolLM2-1.7B, median [min-max] of 5

| ms | ctx 128 before | ctx 128 after | ctx 512 before | ctx 512 after |
|---|---:|---:|---:|---:|
| wall per step | 20.00 [19.98-20.02] | 17.02 [17.00-17.03] | 24.59 [24.58-24.64] | 21.54 [21.52-21.58] |
| tok/s from wall | 50.0 | 58.8 | 40.7 | 46.4 |
| down (24 layers, group time) | 4.89 [4.88-4.90] | 3.12 [3.12-3.13] | 4.85 [4.83-4.86] | 3.09 [3.09-3.09] |
| lm_head (group time) | 2.63 [2.63-2.63] | 0.89 [0.89-0.89] | 2.63 [2.63-2.63] | 0.89 [0.89-0.89] |
| qkv | 2.42 | 2.41 | 2.41 | 2.41 |
| o_proj | 0.95 | 0.95 | 0.94 | 0.94 |
| gate_up | 5.82 | 5.82 | 5.77 | 5.76 |
| attn | 2.07 | 2.07 | 5.81 | 5.80 |

GPU-resident weights: 1346.4 MB before, 1182.8 MB after.

| Group, M2, ctx 128 | MB read per step before | GB/s before | % of 91.4 | MB after | GB/s after | % of 91.4 |
|---|---:|---:|---:|---:|---:|---:|
| down | 399.5 | 81.7 | 89 | 229.7 | 73.6 | 81 |
| lm_head | 82.6 | 31.4 | 34 | 82.6 | 92.8 | 101 |

### Results: native prefill, M2, wall ms, median of 3

| Prompt | before | after |
|---|---:|---:|
| 100 tokens | 625.78 [625.45-626.08] | 620.67 [620.52-621.27] |
| 500 tokens | 2623.86 [2622.88-2626.75] | 2620.77 [2618.77-2620.82] |

### Results: native decode, RTX 3080 (Vulkan), ctx 128, group time in ms (3 pairs with the final kernels; wall range over 6 runs each)

| ms | before | after |
|---|---:|---:|
| down | 1.193 to 1.203 | 1.025 to 1.035 |
| lm_head | 0.2147 to 0.2163 | 0.1388 to 0.1394 |
| qkv, o_proj, gate_up, attn | 0.589, 0.340, 0.867, 0.786 | 0.590, 0.341, 0.870, 0.786 |
| wall per step | 6.98 to 9.43 | 6.77 to 9.13 |

| Group, 3080, ctx 128 | MB per step before | GB/s before | % of 711 | MB after | GB/s after | % of 711 |
|---|---:|---:|---:|---:|---:|---:|
| down | 399.5 | 334 | 47 | 229.7 | 223 | 31 |
| lm_head | 82.6 | 385 | 54 | 82.6 | 595 | 84 |

GPU-resident weights on the 3080 build: same as the M2, 1346.4 MB before and 1182.8 MB after.

### Results: browsers on the M2, decode tok/s, median [min-max]

| Browser | 128 tokens before | 128 after | 512 tokens before | 512 after |
|---|---:|---:|---:|---:|
| Chromium 153 | 47.6 [47.5-47.7] (n=3) | 68.3 [68.2-68.4] (n=3) | 43.3 (n=1) | 59.8 (n=1) |
| WebKit | 50.7 [50.7-50.8] (n=2) | 61.4 [61.3-61.6] (n=2) | 45.3 (n=1) | 53.5 (n=1) |
| Firefox Dev 159 | 53.7 [53.7-53.7] (n=2) | 66.1 [66.0-66.2] (n=2) | 48.5 (n=1) | 58.5 (n=1) |

Hashes: exactness efa85acfe7945318 in all 20 loads, 128-token decode hash 941622dd5a82970f in all 14 loads that decoded 128 tokens, 512-token decode hash 9731a7102ed3b8d7 in all 6 loads that decoded 512, before and after alike. Browser load averages 1.0 to 1.5.

### Correctness

- Logits against the before build on SmolLM2-1.7B, fixed 100-token prompt, prefill plus 16 greedy decode steps, all 49152 logits compared (`crates/lean/examples/logit_dump.rs`): max |diff| 5.1e-5, same 16 greedy tokens. With the Q4_1 change alone: 5.1e-5 at 100 tokens (also with the four-rows-per-lane decode kernel forced on the Q4_1 tensors: 5.1e-5, bit-identical to the one-row kernel). Q6_K kernel against the previous Q6_K kernel: 1.1e-5 (100 tokens), 9.1e-6 (20 tokens). Naive reference path (12 tokens, 4 steps): 3.2e-3 against the before build, whose naive path runs the F32 kernel; the two differ in summation order over K = 8192.
- Against HF transformers: `fixture_parity_llama_1_7b` on the Q4_0 GGUF itself (3 Q4_1 and a Q6_K tensor; fixture `fixture_llama_1_7b_q4_0.json`, HF dequantizes the same file), naive and fast kernels: max |prefill top-20 logit - HF| 4.7e-5 and 4.2e-5 (RTX 3080), 4.9e-5 and 5.3e-5 (M2), greedy continuations equal. The Q8_0 file of the same model: 6.3e-5 and 6.7e-5 (3080), 5.4e-5 and 6.5e-5 (M2). New test: `generate_matches_hf_fixture_smollm2_1_7b` in `fixture_parity_generate`.
- M2 native, ignored suites: `fixture_parity_qwen25_3b` (Q6_K output tensor), `fixture_parity_llama_360m`, `fixture_parity` (both kernel paths), `fixture_parity_generate` for Qwen2.5-0.5B, SmolLM2-360M and SmolLM2-1.7B, `engine_api` 14 of 14: all pass. RTX 3080: `fixture_parity_llama_1_7b`, `fixture_parity` ok; the rest of the box suite was not completed (box unreachable).
- Non-ignored suites of lean-core, lean, lean-chat pass (release), including the new `q4_1_split_matches_reference_dequant`. clippy `-D warnings` clean: native all targets, wasm32 web, wasm32 web+diag.

### Kernel variants tried for the Q6_K head (ms for the whole head, ctx 128)

| Variant (lanes per row x units per lane) | M2 | RTX 3080 |
|---|---:|---:|
| previous scalar kernel | 2.63 | 0.215 |
| one block per lane, 8 lanes | 1.06 | 0.260 |
| consecutive units, 32 x 1 | 1.98 | 0.135 |
| consecutive units, 16 x 2 | 1.18 | 0.131 |
| consecutive units, 8 x 4 (chosen) | 0.89 | 0.139 |
| consecutive units, 4 x 8 | 0.87 | 0.146 |
| consecutive units, 2 x 16 | 0.89 | 0.345 |

(M2 variant rows measured at load 1.0 to 2.1, one run each; 3080 rows two runs each.)

### Observations

- The head was bound by ALU on the M2, not by memory: the scalar kernel did about 30 operations per weight; at the measured 1057 GFLOP/s that is the 2.63 ms. The new kernel does 5 to 6 per weight and reads 82.6 MB in 0.89 ms, at the read ceiling. On the 3080 the same kernel takes the head from 54% to 84% of the read ceiling; the consecutive-lane layout matters there (one block per lane was 1.2x slower than before on the 3080) and is neutral on the M2 at 8 x 4.
- Keeping the three Q4_1 tensors quantized removes 169.8 MB from the per-step reads (399.5 to 229.7 MB for `down`) and 163.6 MB from GPU memory (13% of 1346.4 MB). On the M2 the `down` group falls from 4.89 to 3.12 ms; its rate drops from 89% to 81% of the read ceiling because the Q4_1 kernel does a fused multiply-add per weight more than the F32 matvec. Prefill is unchanged within noise (`down` 143.63 to 142.75 ms at 100 tokens).
- On the 3080 `down` reads at 31% of the read ceiling after the change: the one-row-per-lane matvec for 2048-row outputs launches 128 workgroups. That was already the case for the 21 Q4_0 layers and is not part of this change.
- Whole step at ctx 128 on the M2: 1042 MB (weights plus KV) in 17.02 ms is 61 GB/s, 67% of the read ceiling (before: 1212 MB in 20.00 ms, 61 GB/s, 66%). The remaining gap is qkv (77% of the read ceiling), o_proj (65%), gate_up (85%), down (81%) and attention.
- The browser gains (Chromium +43%, WebKit +21%, Firefox +23% at 128 tokens) are larger than the native +18% (50.0 to 58.8 tok/s); native and browser use different shader compilers (naga for Metal, Tint for Chromium, the browsers' own for WebKit and Firefox), and the cause was not isolated.
- On the RTX 3080 box a prefill-only run (8 tokens, 0 decode steps, same binary, same file) gives different logits from process to process: 11 of 40 runs differed from the most common value, in the before build as well, with small-M kernels disabled too. All 562 uploaded buffers (1.37 GB) read back identical to the uploaded bytes in 12 of 12 runs, and repeated prefills inside one process agree with each other (12 of 12). The M2 gave one value in 20 of 20 runs. Cause not found. It limits the box to checking logits by tolerance (the differences are about 1e-3), not by hash. A build there also died once with a rustc SIGSEGV, which matches the flaky-RAM suspicion.

### Quant types the loader does not keep quantized

The loader (`gguf.rs::GgmlDtype`) reads F32, F16, Q4_0, Q4_1, Q8_0 and Q6_K and rejects every other type while parsing the tensor index, so a file holding any other type does not load at all. Before this change Q4_1, F16 and F32 matmul weights became F32 on the GPU; after it only F16 and F32 do. The CPU backend still dequantizes Q4_1 to F32.

| Type | Bytes per weight | As F32 | Ratio | State |
|---|---:|---:|---:|---|
| Q4_1 | 0.625 | 4 | 6.4x | kept quantized now |
| F16 | 2 | 4 | 2x | loaded as F32 |
| Q5_0 | 0.6875 | 4 | 5.8x | file rejected |
| Q5_1 | 0.75 | 4 | 5.3x | file rejected |
| Q4_K | 0.5625 | 4 | 7.1x | file rejected |
| Q5_K | 0.6875 | 4 | 5.8x | file rejected |
| Q3_K | 0.4297 | 4 | 9.3x | file rejected |
| Q2_K | 0.3281 | 4 | 12.2x | file rejected |

Measured tensor mixes of GGUFs on disk (weights per type, native bytes against the F32 expansion):

| File | Type: tensors, MB native (MB as F32) |
|---|---|
| SmolLM2-1.7B Q4_0 | Q4_0: 165, 877.7 (6241); Q4_1: 3, 31.5 (201); Q6_K: 1, 82.6 (403); F32: 49 |
| SmolLM2-360M Q4_0 | Q4_0: 220, 171.4 (1219); Q4_1: 4, 6.1 (39); Q8_0: 1, 50.1 (189); F32: 65 |
| SmolLM2-360M Q4_K_M (does not load) | Q5_0: 176, 158.8 (924); Q4_K: 16, 22.1 (157); Q6_K: 16, 32.3 (157); Q8_0: 17, 55.4 (208); F32: 65 |
| Qwen2.5-3B Q4_0 | Q4_0: 253, 1735.7 (12343); Q6_K: 1, 255.3 (1245); F32: 181 |
| Qwen2.5-0.5B Q4_0 | Q4_0: 169, 277.9 (1976); Q8_0: 1, 144.6 (545); F32: 121 |
| Qwen3-1.7B Q8_0 | Q8_0: 197, 1828.0 (6882); F32: 113 |

### What is left

- Q5_0 and Q4_K (the two types behind the Q4_K_M file above, 181 MB of its 268 MB) are the next types worth a kernel: a "Q4_K_M" file is what most Hub models ship, and none of them loads today.
- Q4_1 on the CPU backend is still dequantized to F32 at load.
- Box: context 512, the remaining test suites, and the process-to-process logit variation above.
