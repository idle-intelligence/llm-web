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

## 2026-10-14: decode attention against context length

Change: the split-K pass 1 kernel (`attn_decode_split.wgsl`) read each key row with one thread per key, so neighbouring threads were 256 bytes apart (SmolLM2-1.7B, head_dim 64). Now 8 adjacent threads share one key and read contiguous vec4 slices of its row; the tile max and sum use broadcast reads of workgroup memory instead of two barrier trees; the V accumulation is unrolled by four keys. The split count and chunk size (`decode_split_plan`) are unchanged, so there is no new tuning input and no API change. Workgroup size stays at head_dim (64 or 128), no subgroups.

### Parameters

- Model SmolLM2-1.7B-Instruct Q4_0 (32 query heads, 32 KV heads, head_dim 64, F32 KV cache), greedy. Native harness: `crates/lean/examples/decode_ctx.rs` (prefill N tokens of a repeated prompt, then 64 pipelined greedy steps, median of 3 loads of the loop per cell; 5 interleaved process runs, before then after). Browser harness: `scripts/bench` phase decode, 128 / 512 / 1900 new tokens (the story ends by itself at 963 tokens, so the 1900 row is a 963-token run), 5 page loads per cell (3 for 1900), before then after interleaved.
- Before: shader of the previous commit. After: this branch. wasm md5 before 120efbdcac048db0523e3a0053c52a01, after 9d1dae68c4b92487ff03dbf701dd6ba0 (built once on the Linux test machine, copied to the M2 with matching md5, served under distinct tags and fetched back).
- M2 laptop: 1-minute load average 0.8 to 2.7 for the native runs; 1.0 to 6.7 for the browser runs (other workers built and ran on it). Linux test machine (RTX 3080): load under 1 for native, GPU behind a lock. Chromium 153 (Metal, headless) and WebKit via Playwright; Firefox Developer Edition 159 headed with a throwaway profile on the M2; Chromium 152 headed on Vulkan on the Linux machine.

### Results: native decode, ms per token (median, min-max)

| Context | M2 before | M2 after | RTX 3080 before | RTX 3080 after |
|---|---|---|---|---|
| 128 | 21.5 (21.3-21.6) | 21.0 (20.9-23.2) | 4.61 (4.55-4.64) | 4.28 (4.24-4.30) |
| 512 | 25.2 (25.1-25.2) | 22.8 (22.7-23.6) | 4.82 (4.78-4.86) | 4.47 (4.43-4.48) |
| 1024 | 28.8 (28.6-28.8) | 25.3 (25.2-25.4) | 5.26 (5.23-5.29) | 4.78 (4.73-4.78) |
| 2048 | 38.2 (37.9-38.3) | 33.3 (33.2-33.4) | 11.45 (11.41-11.46) | 5.59 (5.56-5.60) |

M2 before has 4 runs (the first one timed out waiting for the GPU lock), after has 5. A quieter earlier M2 session gave before 19.3 / 23.5 / 27.7 / 36.8 ms and after 18.9 / 21.3 / 24.6 / 32.5 ms at the same four contexts (single run each).

RTX 3080, finer contexts, ms per token:

| Context | 1024 | 1100 | 1300 | 1536 | 1800 | 2048 |
|---|---|---|---|---|---|---|
| Before | 5.25 | 5.47 | 6.02 | 7.34 | 9.93 | 11.41 |
| After | 4.77 | 4.85 | 4.98 | 5.11 | 5.31 | 5.60 |

### Results: browser decode, tok/s (median, min-max)

M2:

| Browser | New tokens | Before | After |
|---|---|---|---|
| Chromium 153 | 128 | 47.3 (39.2-47.6) | 46.7 (37.9-48.9) |
| Chromium 153 | 512 | 37.1 (36.7-38.3) | 39.5 (39.1-42.2) |
| Chromium 153 | 963 | 33.5 (32.0-34.7) | 38.3 (37.9-39.8) |
| WebKit | 128 | 39.5 (38.1-40.9) | 40.4 (38.1-41.7) |
| WebKit | 512 | 39.3 (39.0-39.4) | 41.1 (40.9-41.6) |
| WebKit | 963 | 34.5 (34.2-34.6) | 36.7 (36.4-37.1) |
| Firefox Dev 159 | 128 | 45.3 (42.2-45.5) | 46.0 (43.4-49.5) |
| Firefox Dev 159 | 512 | 41.5 (40.2-41.8) | 43.7 (42.9-43.9) |
| Firefox Dev 159 | 963 | 35.5 (35.4-36.0) | 39.0 (38.6-39.1) |

RTX 3080, Chromium 152 headed (Vulkan), 5 loads:

| New tokens | Before | After |
|---|---|---|
| 128 | 226.1 (225.1-227.1) | 243.3 (241.9-243.6) |
| 512 | 221.4 (220.9-221.6) | 241.9 (241.5-242.3) |
| 963 | 212.6 (212.4-212.9) | 233.5 (233.4-233.6) |

### Correctness

- Decode text hashes before and after are identical in every browser run: 941622dd5a82970f (128 tokens), 9731a7102ed3b8d7 (512), da4a25fb6ebc98b8 (963); exactness hash efa85acfe7945318 before and after on the Linux machine's Chromium. Native greedy id hashes per context (128 to 2048, and 33 to 300 around the 64-key split threshold) are identical before and after on both machines; a 60-token chat reply is identical on the RTX 3080.
- Largest absolute logit difference against the previous kernel, over the first 4 to 6 decode steps after each prefill: SmolLM2-1.7B 2.3e-5 (M2) and 4.0e-5 / 4.7e-5 (RTX 3080, contexts 128 to 2048 and 33 to 300); Qwen3-1.7B Q8_0 (head_dim 128) 5.2e-5 (M2), with identical id hashes at 512 and 2048.
- Tests on the M2 (release): lean-core, lean, lean-chat suites pass; with models on disk: engine_api 14 of 14, fixture_parity_generate Qwen2.5-0.5B and SmolLM2-360M, fixture_parity (both kernel paths), llama 360M (GPU and CPU), llama 1.7B, Qwen2.5-3B, Qwen3-0.6B, Qwen3-1.7B, kv_snapshot, logit_mask, pool_reuse, all ok. Not run (missing model, LoRA or config files on this machine): the Llama-3.2 cases, the LoRA tests, two lean-core unit tests that read files. clippy `-D warnings` as in CI: native all targets, wasm32 web, wasm32 web+diag, clean.
- A first version of the shader named a local `dot` and then called the builtin `dot()`. naga (native) accepted it, Chromium's Tint rejected the module, the pipeline was invalid and the browser's greedy reply stopped after 25 tokens with a different hash. Fixed in the next commit; the browser tables above are from the fixed build. Native tests cannot catch this class of error; the browser decode hash can.

### Observations

- The 3080 curve before the change was not linear: 5.25 ms at 1024, then 7.3 at 1536 and 11.4 at 2048, against 4.2 ms of non-attention time, so attention went from 1.0 to 7.2 ms for 2x the keys. After the change the same curve is linear (attention 0.6 ms at 1024, 1.3 at 2048, read floor about 1.1 ms). The cause is not proven (no hardware counters); the old kernel read each 256-byte key row one float at a time with neighbours a row apart, so every load touched a separate cache line and relied on later loads of the same row hitting in cache. With more split workgroups resident at once (the split count grows with the context up to 32) that reuse would be lost. The change removes the dependence on reuse, and the cliff goes with it.
- On the M2 the attention part at 2048 fell from about 17.5 to 12.5 ms of the added step time (38.2 and 33.3 ms against about 21 at 128), the read floor being about 8.8 ms (F32 KV, 32 KV heads). Decode at 2048 context is still 1.6x slower than at 128 there.
- Browser gains follow the context the run reaches: none at 128 tokens (within spread), 6 to 7% at 512, 9 to 14% at 963 on M2 browsers, 9 to 10% on the RTX 3080.

### What is left

- M2 attention at 2048 is about 1.4x its read floor; an f16 or int8 KV cache would lower the floor itself but changes numerics and is not part of this change.
- Long prompt with a short reply was measured natively only (decode after prefill of N tokens), not through the browser pages.
- Firefox on the Linux machine was not run.

## 2026-10-14: f16 prefill matmul when the device grants shader-f16

Change: `linear_q4_tiled_f16.wgsl`, the 64x64 register-blocked prefill tile with both shared tiles in `vec4<f16>` (dequantised weights `(q - 8) * scale` rounded once to f16, activations rounded to f16 on staging) and the dot products of one whole Q4_0 block (32 values) summed in f16 before they are widened into the f32 accumulators. Accumulation across blocks, the bias and the output stay f32. The Q8_0 prefill uses the same kernel through its `Q8` override. The device requests `shader-f16` whenever the adapter offers it; without it the F32 tiled kernels run, unchanged. Decode, the small-M prefill kernel (fewer than 64 rows) and attention are untouched. `LEAN_F16=0` keeps the F32 kernels on an f16 adapter (native A/B only; browsers have no such switch).

### Parameters

- SmolLM2-1.7B-Instruct Q4_0, maxCtx 2048, greedy. Browser harness `scripts/bench` (phase cold: prompts of 100, 500 and 1000 tokens, 3 repetitions inside each page load, then the suffix prefill of the same prompt with a quarter more tokens appended; the table value is the median of the 3 repetitions). Native: `prefill_sweep --lens 100,500,1000 --reps 3 --suffix`, whose suffix case is the second half of the prompt after the first half is resident.
- 5 page loads (native: 5 process runs) per cell, interleaved after, before, after, before. Medians with min and max. Before is the wasm built from the previous commit (md5 82cf52894b4891dc98fb959fdfb82aac), after is this branch (md5 c859bc9ea6b623f2368ff0d1c81d20cc), served from a no-store static server under distinct tags; the bench page records `device.features.has("shader-f16")` for each load.
- M2 laptop, 16 GB, 1-minute load average 0.9 to 2.1 during the runs, one GPU job at a time. Chromium 153 (Playwright bundled, headless, `--use-angle=metal`), WebKit (Playwright bundled, headless), Firefox Developer Edition 159 (headed, throwaway profile, `dom.webgpu.enabled`), native wgpu on Metal.

### Results: cold prefill, ttft ms (median, min-max)

| Prompt | Chromium before | Chromium after | WebKit before | WebKit after | Firefox Dev 159 before | Firefox Dev 159 after |
|---|---|---|---|---|---|---|
| 100 tokens | 587 (511-616) | 440 (405-465) | 619 (618-619) | 473 (473-474) | 596 (595-738) | 453 (453-518) |
| 500 tokens | 2408 (2217-2475) | 1919 (1682-2059) | 2710 (2705-2861) | 2110 (2108-2110) | 2591 (2587-3140) | 1985 (1975-2372) |
| 1000 tokens | 4912 (4835-4985) | 3901 (3534-4061) | 5971 (5939-6541) | 4769 (4767-4772) | 5628 (5616-6503) | 4417 (4400-4939) |

### Results: suffix prefill, ttft ms (prefix cached, then a quarter more tokens)

| Case | Chromium before | Chromium after | WebKit before | WebKit after | Firefox Dev 159 before | Firefox Dev 159 after |
|---|---|---|---|---|---|---|
| 100 + 24 | 176 (150-186) | 166 (149-178) | 187 (187-187) | 187 (187-187) | 174 (173-222) | 173 (173-206) |
| 500 + 124 | 653 (617-668) | 540 (473-578) | 790 (788-846) | 642 (642-643) | 744 (743-883) | 596 (593-699) |
| 1000 + 249 | 1397 (1378-1436) | 1150 (1046-1192) | 1884 (1878-2013) | 1587 (1586-1587) | 1748 (1745-1937) | 1449 (1444-1565) |

The 100 + 24 case runs 24 rows, below the 64-row threshold of the small-M kernel, so it does not use the changed kernel.

### Results: native wgpu on Metal, prefill ms (median, min-max, n = 5)

| Case | F32 kernels (LEAN_F16=0) | f16 kernels |
|---|---|---|
| 100 tokens | 660.5 (626.6-664.0) | 521.8 (484.1-522.7) |
| 500 tokens | 2847.9 (2825.4-2953.9) | 2320.9 (2024.7-2323.5) |
| 1000 tokens | 6187.7 (6144.5-6366.2) | 4827.4 (4430.1-4881.1) |
| suffix 250 after 250 | 1518.8 (1502.1-1635.1) | 1218.1 (1104.4-1223.7) |
| suffix 500 after 500 | 3409.7 (3354.0-3426.4) | 2652.6 (2501.3-2667.1) |
| suffix 50 after 50 (small-M kernel) | 333.9 (300.0-334.9) | 345.5 (301.4-352.6) |
| decode ms per step | 20.0 (20.0-20.1) | 20.1 (20.0-20.5) |

The native f16 runs drifted upward between the first and the last run (484 to 522 ms at 100 tokens) while the F32 runs did too (627 to 664), so the min of each column and the median both show the same ratio of about 0.79.

### f16 availability

| Engine | adapter offers shader-f16 | device granted | path taken |
|---|---|---|---|
| Chromium 153 on the M2 (Metal) | yes | yes | f16 |
| WebKit (Playwright build) on the M2 | yes | yes | f16 |
| Firefox Developer Edition 159 on the M2 | yes | yes | f16 |
| Native wgpu 26 on Metal (M2) | yes | yes | f16 |
| Chromium 153 on the Linux test machine (RTX 3080, Vulkan, driver 610.57) | no | no | F32 |
| Native wgpu 26 on Vulkan (RTX 3080) | no | no | F32 |

On the Linux test machine the F32 path was re-measured with the same wasm pair (headed Chromium, 2 repetitions): cold 100 tokens 59.9 ms (after build) against 60.4 ms (before), 500 tokens 215.8 against 214.8, same exactness hash, so the capability gate leaves that machine's numbers unchanged.

### Correctness

- `crates/lean-core/tests/f16_parity.rs` (ignored test, needs the GGUF and a fixture name) runs every fixture case twice, F32 kernels then f16 kernels, and fails on any prefill top-1 or greedy-continuation token that differs from the F32 run or from the reference fixture. All greedy continuations were identical (first divergence none) on: Qwen2.5-0.5B Q4_0 (5 cases, including the 2225 and 2354 token tool prompts), SmolLM2-360M Q4_0, SmolLM2-1.7B Q8_0, Qwen2.5-3B Q4_0, Qwen3-1.7B Q8_0 and Qwen3-0.6B Q8_0 (including the 2225 token prompt).
- Largest absolute difference of the last-position logits, f16 against F32 kernels: 0.0457 (Qwen2.5-0.5B, 2354 tokens), 0.0552 (Qwen3-0.6B, 2225 tokens), 0.0503 (Qwen3-1.7B, 2225 tokens), 0.0336 (SmolLM2-1.7B Q8_0, 86 tokens), 0.0266 (Qwen2.5-3B, 86 tokens), 0.0216 (SmolLM2-360M, 86 tokens). Cases shorter than 64 rows use the small-M kernel and show 0 difference. Against the reference fixture's top-20 logits (HF transformers) the F32 kernels stay within 1.2e-4, the f16 kernels within 0.030 (largest: Qwen3-1.7B, 2225 tokens).
- `fixture_parity` and the sibling fixture tests assert 1e-3 on the prefill top-20 logits. With f16 kernels that assertion fails on the first case of 64 or more rows (Qwen2.5-0.5B `long`: 3.0e-3), while the top-1 and all greedy tokens in the same test would still match. Those tests need `LEAN_F16=0` to run their 1e-3 gate on an f16 adapter; their greedy assertions pass either way. Not changed here.
- `engine_api` with f16 kernels on: 14 of 14. `fixture_parity_generate` (Qwen2.5-0.5B, SmolLM2-360M): 2 of 2 (the Llama-3.2-1B case needs files that are not on this machine).
- Browser exactness hash efa85acfe7945318 in all 30 browser loads (Chromium, WebKit, Firefox, before and after).
- clippy `-D warnings`: native all targets, wasm32 web, wasm32 web+diag, clean (run on the Linux test machine).

### Observations

- The three browsers that offer shader-f16 on the M2 take the f16 path and gain 20 to 25% on cold prefill at 100, 500 and 1000 tokens (Chromium 25%, 21%, 21%; WebKit 24%, 22%, 20%; Firefox 24%, 23%, 22%), and 17 to 22% on suffix prefill of 500 or more rows.
- A window of 4 slots instead of a whole block for the f16 partial sum cost 4% in time and lowered the logit error by about 15%; a window of 1 was slower than the F32 kernel. The error is dominated by rounding the activations and the weights to f16, not by the f16 sum.
- A 128 x 64 tile with 8 x 4 outputs per thread (more reuse per shared load) ran 11 to 16% slower than the 64 x 64 tile on the M2 (2949 ms against 2026 ms at 500 tokens, one run each) and was dropped.
- Native decode, Chromium decode and the exactness hash are unchanged.

### What is left

- Greedy exactness was measured on the fixtures and bench prompts above; it is not guaranteed for arbitrary text, because the logits move by up to 0.055. A near-tie in a prompt not tested could resolve differently from the F32 kernels (and from HF).
- The 1e-3 prefill logit gates in the fixture tests do not hold with f16 kernels; they need `LEAN_F16=0` or a looser gate for the top-20 values.
- f16 on the RTX 3080 was not measured: neither native wgpu nor Chromium 153 offers shader-f16 on that machine and driver.
- The small-M kernel (under 64 rows) and prefill attention still run in f32.

## 2026-10-15: sampled decode (the chat page's sampler) on the GPU

### Problem

The chat page samples with temperature 0.7, top-k 40, top-p 0.9, repetition penalty 1.1, seed 0, and it read the whole logits vector back every token (Qwen vocabulary 151936, about 600 KB), then sorted the whole vocabulary on the CPU for top-k. Qwen2.5-0.5B decoded at about 70 tok/s sampled while greedy decodes at 130 to 170 tok/s on the same build. Every earlier speed measurement was greedy.

### Parameters

- Bench: `scripts/bench/bench.html` with `phases=decode,sampled` (new `sampled` phase: GEN_PARAMS of `web/lean-chat-worker.js`, seed 0, prompt "Write a long story about a lighthouse keeper.", 128 and 256 new tokens, 3 repetitions per load, rate over the whole run from the first to the last token, sha256 of the reply). Loads: 5 per cell for Qwen2.5-0.5B Q4_0 and SmolLM2-1.7B Q4_0, 3 for Qwen3-0.6B Q8_0 and SmolLM2-360M Q4_0. The base, step 2 and final builds alternate load by load (ABAB order). Values are the median over all repetitions, with min to max in brackets.
- Browsers on the M2: Playwright Chromium 153 (`--use-angle=metal`, headless), Playwright WebKit (headless), Firefox Developer Edition 159 (headed, throwaway profile). Native: the ignored test `crates/lean/tests/sampled_native.rs` on wgpu Metal.
- Builds compared: base is browser-perf at the reuse fix plus the bench phase; step 2 is the GPU top-k with a 64-candidate readback (sampling still finished on the CPU); final is the whole step on the GPU (this section's last commit). Served wasm bytes were checked against the built files by sha256 before each series.
- The machine was shared with another worker's GPU runs through the lock script; one-minute load average between 1 and 4 during the runs. SmolLM2-1.7B runs in WebKit and Firefox show dips that hit all builds; read the medians with the spread.

### Change

1. CPU sampler: top-k by linear-time selection, softmax and top-p over the survivors only. Same tokens as the full-sort code for the same seed (unit test against the old implementation on random logits with ties).
2. GPU top-k (tree of 1024-item bitonic sorts, 3 dispatches at Qwen vocabulary size), host-side penalty, top-p and draw over 64 candidates. Same tokens as step 1, with a fallback to the full readback when the candidates cannot rule out a tie at the k-th logit.
3. The whole step on the GPU for top_k 1 to 63: penalty from a per-id count buffer, top-k, temperature, softmax, top-p and the draw in three kernels at the end of the decode step. The chosen id feeds the next step's embedding on the GPU and one id is read back per step, with two steps in flight, so every token streams as soon as its step resolves. The kernels use only multiplications and additions (no division, a polynomial exp) and a counter-based generator keyed by seed and token position. `lean_core::resident_sampling::sample_reference` is the same computation on the CPU: it chooses the first token (from the prefill logits) and serves the CPU backend, so the GPU and CPU backends give the same reply for the same seed. Step 2's candidate readback is removed. top_k 0 and 64 and up keep the full-logits path.
4. Semantics that differ from before: the random stream is different (a seed gives a different reply than in earlier builds), and ties at the k-th logit are cut at exactly k, lower token id first, where the CPU sampler keeps every tied value.

### Results: sampled decode, 128 new tokens, tok/s

| model | browser | before (base) | step 2 (candidates) | after (final) | n loads b/s2/f |
|---|---|---|---|---|---|
| qwen25-05b | chromium | 75 (74-75) | 139 (110-147) | 165 (154-166) | 5 / 5 / 5 |
| qwen25-05b | webkit | 66 (66-67) | 118 (95-119) | 126 (124-127) | 5 / 5 / 5 |
| qwen25-05b | firefox | 64 (59-64) | 111 (67-118) | 144 (127-147) | 5 / 5 / 5 |
| qwen3-06b | chromium | 55 (54-55) | 85 (78-86) | 92 (80-94) | 3 / 3 / 3 |
| qwen3-06b | webkit | 49 (49-49) | 72 (63-73) | 75 (68-77) | 3 / 3 / 3 |
| qwen3-06b | firefox | 49 (48-50) | 53 (52-63) | 87 (77-88) | 3 / 3 / 3 |
| smol-360m | chromium | 119 (117-120) | 162 (142-164) | 186 (177-188) | 3 / 3 / 3 |
| smol-360m | webkit | 101 (99-101) | 120 (98-132) | 143 (138-144) | 3 / 3 / 3 |
| smol-360m | firefox | 81 (64-89) | 79 (62-117) | 157 (155-162) | 3 / 3 / 3 |
| smol-17b | chromium | 57 (57-58) | 64 (61-66) | 69 (53-70) | 5 / 5 / 5 |
| smol-17b | webkit | 51 (32-52) | 47 (43-50) | 49 (44-61) | 5 / 5 / 5 |
| smol-17b | firefox | 52 (39-52) | 44 (42-51) | 55 (52-66) | 5 / 5 / 5 |

### Results: sampled decode, 256 new tokens, tok/s

| model | browser | before (base) | step 2 (candidates) | after (final) | n loads b/s2/f |
|---|---|---|---|---|---|
| qwen25-05b | chromium | 75 (75-75) | 132 (92-147) | 164 (147-165) | 5 / 5 / 5 |
| qwen25-05b | webkit | 67 (66-67) | 107 (84-118) | 126 (123-126) | 5 / 5 / 5 |
| qwen25-05b | firefox | 64 (62-65) | 69 (66-83) | 143 (131-145) | 5 / 5 / 5 |
| qwen3-06b | chromium | 55 (55-55) | 86 (83-87) | 92 (80-96) | 3 / 3 / 3 |
| qwen3-06b | webkit | 49 (49-49) | 72 (65-73) | 75 (66-77) | 3 / 3 / 3 |
| qwen3-06b | firefox | 50 (49-50) | 53 (50-54) | 87 (78-88) | 3 / 3 / 3 |
| smol-360m | chromium | 119 (118-119) | 162 (151-163) | 182 (167-187) | 3 / 3 / 3 |
| smol-360m | webkit | 100 (100-101) | 125 (108-129) | 140 (139-143) | 3 / 3 / 3 |
| smol-360m | firefox | 85 (66-88) | 67 (63-73) | 157 (132-161) | 3 / 3 / 3 |
| smol-17b | chromium | 56 (56-57) | 64 (52-65) | 66 (54-69) | 5 / 5 / 5 |
| smol-17b | webkit | 50 (32-50) | 45 (40-49) | 48 (45-59) | 5 / 5 / 5 |
| smol-17b | firefox | 49 (42-50) | 46 (42-57) | 54 (53-65) | 5 / 5 / 5 |

### Results: greedy decode in the same runs (96 tokens, tok/s, base against final)

Unchanged by this work, within the spread: Qwen2.5-0.5B 168 / 169 (Chromium), 135 / 135 (WebKit), 154 / 154 (Firefox); Qwen3-0.6B 95 / 95, 80 / 80, 90 / 90; SmolLM2-360M 197 / 197, 154 / 153, 173 / 173; SmolLM2-1.7B 71 / 71, 63 / 56, 67 / 65.

### Results: native wgpu on Metal, sampled tok/s (median of 6 repetitions, min to max)

| Model | New tokens | base | step 2 | final |
|---|---|---|---|---|
| qwen25-05b | 128 | 70 (69-73) | 117 (104-124) | 124 (123-126) |
| qwen25-05b | 256 | 72 (72-73) | 117 (106-121) | 125 (112-125) |
| qwen3-06b | 128 | 54 (53-54) | 76 (75-77) | 75 (74-76) |
| qwen3-06b | 256 | 54 (54-55) | 77 (76-78) | 75 (74-76) |
| smol-360m | 128 | 90 (78-97) | 94 (92-114) | 94 (93-120) |
| smol-360m | 256 | 92 (81-94) | 95 (85-98) | 100 (97-104) |
| smol-17b | 128 | 56 (55-56) | 61 (61-62) | 59 (55-60) |
| smol-17b | 256 | 54 (54-54) | 60 (60-60) | 58 (58-59) |

### Results: one token per step against 8 per readback

Two builds identical except for the number of steps per readback (1 against 8), Qwen2.5-0.5B and SmolLM2-1.7B, 5 loads each; sampled tok/s at 128 tokens:

| Model | Browser | 8 per readback | 1 per readback |
|---|---|---|---|
| Qwen2.5-0.5B | Chromium | 157 (121-164) | 160 (130-166) |
| Qwen2.5-0.5B | WebKit | 123 (103-126) | 125 (103-127) |
| Qwen2.5-0.5B | Firefox | 146 (145-147) | 144 (141-149) |
| SmolLM2-1.7B | Chromium | 68 (38-70) | 52 (44-70) |
| SmolLM2-1.7B | WebKit | 50 (38-52) | 51 (45-53) |
| SmolLM2-1.7B | Firefox | 54 (38-60) | 55 (38-60) |

Greedy decode of 256 tokens, SmolLM2-1.7B, ABAB, 5 loads each, tok/s: Chromium 69.6 (69.2-69.7) with 8 per readback and 69.6 (69.3-69.9) with 1; Firefox 58.6 (57.2-59.8) and 59.8 (57.6-60.5); WebKit 59.3 (49.8-60.3) and 52.4 (49.1-59.5). Greedy Qwen2.5-0.5B in the 5-load series above: 168 (144-169) against 167 (160-170) in Chromium, 134 and 134 in WebKit, 154 and 155 in Firefox.

### Results: determinism

Reply hashes (first 8 bytes of sha256 of the reply text) for the 128 and 256 token budgets are identical in Chromium, WebKit, Firefox Developer Edition 159, native GPU (Metal) and, for Qwen2.5-0.5B, the native CPU backend:

| Model | base and step 2 | final |
|---|---|---|
| Qwen2.5-0.5B | c9207b389a5307f3, 92a2127691969159 | 6be21500b4902e60, 6ca9499e057eec9c |
| Qwen3-0.6B | edb8a4754f4f4871, 54281234dd4dccac | 48b4d208e2578ff2, 37c1686d60bdcfc3 |
| SmolLM2-360M | e8b194b9b50b01fa, 150899e76dd89f53 | b3a375abc0f60f5f, 12de46ad620f0fd2 |
| SmolLM2-1.7B | 41495f42a537a8b2, ad060b92b43c36f2 | 9c953cc07f554df1, c20fc6a649aa22ba |

Base and step 2 give the same hash in every cell (step 1 and 2 keep the old stream). The final hashes are the same in each cell across the three browsers and native, and repeat across repetitions and loads.

### Results: chat page end to end

`web/index.html` served locally with the final wasm (SmolLM2-360M, the page's default; page's own sampling), story prompt, then a second question, then New chat, then a third question, in the three browsers: all steps completed, New chat emptied the conversation, and the three browsers produced the same text for the same turns (the page reports 242 tokens at 186, 140 and 160 tok/s for the story in Chromium, WebKit and Firefox). The second reply of this 360M model wanders off topic after one sentence, with the previous build as well (compared in Chromium).

### Correctness

- `sampling::tests::matches_full_sort_oracle_on_random_logits_and_ties`: 400 random cases (vocabulary 5 to 2000, ties, k at, below and above the vocabulary), 6 draws each, tokens equal to the previous implementation.
- `lean_core::tests::resident_sampling::gpu_sampler_matches_reference`: 240 random cases (vocabulary 900, 3000, 49152, 151936; ties; masked logits; top-p, penalty counts, k from 1 to 63) on Metal, kernel token equal to `sample_reference` in every case and the chosen id's count incremented.
- `resident_sampling_matches_cpu_reference` (ignored model test): 96-token decodes on four prompts with seeds, equal to the step-by-step CPU reference over the full logits; run on Qwen2.5-0.5B. `full_logits_sampling_is_reproducible` for top_k 0 and 64.
- Ignored model tests run on this machine: lean unit tests (8), `engine_api` (17, GPU and CPU, reuse and overflow), `fixture_parity_generate` (Qwen2.5-0.5B, SmolLM2-360M and 1.7B), lean-core `fixture_parity` and `fixture_parity_qwen3`, `f16_parity`, `kv_snapshot`, `logit_mask`, `pool_reuse`: all pass. The Llama-3.2 and a few other fixture tests need files that are not on this machine and were not run.
- The bench `reuse` phase passes (reply of a freshly loaded engine equals the reuse replies) for each browser, model and build (48 of 48 runs, one per browser, model and build).
- clippy `-D warnings`: native all targets, wasm32 web, wasm32 web+diag, clean.

### Observations

- Sampled decode now runs at the greedy rate within 5 to 10% on Qwen2.5-0.5B (Chromium 165 against 169, WebKit 126 against 135, Firefox 144 against 154), from 75, 66 and 64 tok/s. The read-back of the 600 KB logits and the full-vocabulary sort account for the whole gap.
- Step 2 already removes most of the CPU work, but its per-token round trip varies by browser: Firefox ranged 67 to 118 tok/s on Qwen2.5-0.5B (median 111, 69 at 256 tokens). Keeping the choice on the GPU removes that variation (final: 127 to 147).
- Gain of final over step 2 on Qwen2.5-0.5B: Chromium +19% (165 against 139), WebKit +7%, Firefox +30%. On SmolLM2-1.7B the forward pass dominates; the median gain is +8% in Chromium (69 against 64) and +25% in Firefox, and WebKit shows no change (49 against 47).
- Reading 8 tokens per readback gave no measurable gain over one token per step for sampled or greedy decode in any of the three browsers, so sampled decode streams one token at a time. The SmolLM2-1.7B Chromium line of the table above (68 against 52) comes from dips in single loads that hit both builds at different times; the later greedy ABAB series on the same model shows 69.6 against 69.6.
- Native Metal does not need the GPU-resident step: step 2 and final measure within 6% of each other (native step 2 is slightly faster on Qwen3-0.6B and SmolLM2-1.7B, final on Qwen2.5-0.5B); both are 1.7 times the base on Qwen2.5-0.5B.
- SmolLM2-1.7B WebKit and Firefox, where the forward pass leaves little to win, do not improve beyond the spread.

### What is left

- top_k 0 (pure temperature sampling over the whole vocabulary) and top_k of 64 and up still read the full logits and sort on the CPU; no GPU path for them.
- The kernels' draw agrees with the CPU reference to the last bit unless a device fuses multiply-add differently in the polynomial exp; no mismatch in 240 kernel cases and none in any browser hash, but a draw within a few units of the last place of a cumulative-probability boundary could resolve differently on a device that does.
- The Safari-only garbage output with Qwen2.5-0.5B was not reproduced in Playwright WebKit (sensible text, same hash as Chromium); not investigated here.
- Real Safari, Android browsers and the RTX 3080 were not measured.
