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
