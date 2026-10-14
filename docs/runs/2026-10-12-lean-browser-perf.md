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
