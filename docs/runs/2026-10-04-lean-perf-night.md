# 2026-10-04 lean-perf-night: decode on WebGPU and the CPU threads backend

Branch `lean-perf-night`, from `lean-main` at 0ab85d7 (lean-main's
94a16ff, the streaming `cache.put` in `backends_common.js`, merged in at
48cec77).

## Starting point

Reported for lean-main (ENGINE_BUILD 2026-10-03-main-01) on an Android
phone (Chrome, Adreno 6xx), Qwen2.5-0.5B-Instruct Q4_0, 36-token prompt,
`backends.html?diag=1`. Not re-measured here.

| quantity | value |
|---|---|
| WebGPU prefill | 514.8 ms (gate_up 203.9, down 111.5, qkv 30, o_proj 21, attn 10.1) |
| WebGPU decode | 62.8 ms/token |
| WebGPU decode GPU span | 40.5 ms (gate_up 14.9, lm_head 12.4, down 7.0) |
| WebGPU decode encode+submit (CPU) | 8.1 ms |
| bandwidth probe read | 18.9 GB/s |
| Firefox, CPU threads backend | decode 282.4 ms/token, thread pool init 3977.7 ms |
| Chrome, CPU threads backend (same phone, earlier build) | decode 97.4 ms/token |

## Changes

| commit | change |
|---|---|
| b5a4ee1 | `linear_q4_decode` / `linear_q8_decode` rewritten with 4 output rows per lane: a lane loads a block's 32 x values once and dots them against 4 rows' blocks (x loads per weight byte 8x to 2x for Q4_0, 4x to 1x for Q8_0). Q8_0 gets the gate/up kernel with silu(gate) * up in its epilogue (`linear_q8_decode_swiglu`), as Q4_0 already had |
| 2945369 | pipelined greedy decode (`model::decode_greedy_pipelined`): each step's embedding reads its token id from the previous step's argmax buffer, copied on the encoder, so step t+1 is encoded and submitted before step t's id is read back. Same tokens, stop rule and final `kv_len` as the step loop. Used by `generate`, the greedy path of `decode_loop` (`generateStream`, `chatGenerate`) and a new `decodeGreedy(tokenId, steps)`. New test `streaming_sampling::pipelined_greedy_matches_step_loop` |
| b68d112 | backends page: WebGPU decodes with `decodeGreedy`; with `diag=1` it then repeats the same tokens step by step for the per-step split and checks the ids are the same. ENGINE_BUILD 2026-10-04-night-01 |
| 5356f0b | CPU threads backend: one parallel region per decode step (`cpu_team.rs`). Pool thread 0 runs the step, the other pool threads spin and claim 32-column blocks of each matvec through one epoch-tagged CAS word, instead of one rayon `par_iter` per matvec (about 170 per token) with the pool parking in between. Same `dot_row` per output, bit-identical to the serial path. Prefill keeps rayon |
| 0debe36 | ENGINE_BUILD 2026-10-04-night-02 (pkg-mt rebuilt) |
| 48cec77 | merge of lean-main (streaming `cache.put`); ENGINE_BUILD 2026-10-04-night-03 |
| 7c9080b, c918cde | the 4-row kernels only for outputs of 4096 rows or more (`DECODE_R4_MIN_ROWS`: MLP gate/up and the head); narrower outputs (qkv, o_proj, down) go back to the one-row kernels. ENGINE_BUILD 2026-10-04-night-04 |

## Parameters

- M2 laptop (Metal). Timed runs held the gpu, browser and cargo locks and
  started with a 1-minute load average under 3 and no rustc or cargo
  running; load averages are listed per table.
- M2 browser: Playwright's Chrome for Testing (headless, `--use-angle=metal`,
  adapter apple / metal-3) and Playwright's Firefox 155 (headless; no
  WebGPU, CPU backends only). `navigator.hardwareConcurrency` is 8 in both.
- Linux desktop with an RTX 3080: Chromium headed on Xwayland (Vulkan,
  adapter nvidia / ampere), 24 hardware threads, the box's page runner,
  one page load at a time under the box lock. GPU idle before the runs
  (0 %, 515 MiB).
- Page: `crates/lean/www/backends.html`, Qwen2.5-0.5B-Instruct Q4_0,
  36-token prompt, 64 greedy tokens (63 decode steps). Base = lean-main
  0ab85d7's pkg and pkg-mt (ENGINE_BUILD 2026-10-03-main-01, wasm sha256
  94201fe07f56b1da / 4a151403b00fd4b1) served from a snapshot. Served wasm
  bytes were hashed against each build before the runs.
- Native: `cargo run --release -p lean --example prefill_sweep -- --lens 36
  --reps 3 --decode 63` (the branch's copy prints the pipelined decode
  too), base and branch binaries interleaved.

## wasm bytes

| build | pkg/lean_bg.wasm | pkg-mt/lean_bg.wasm | `/Users/` strings |
|---|---|---|---|
| 2026-10-03-main-01 (base) | 94201fe07f56b1da | 4a151403b00fd4b1 | 0 |
| 2026-10-04-night-01 | 9226e2b2be03d7b1 | 9ff331b5c42494b9 | 0 |
| 2026-10-04-night-02, -03 | 9226e2b2be03d7b1 | c3aabd1b57483432 | 0 |
| 2026-10-04-night-03-oldmv (experiment: night-03 with lean-main's decode matvecs) | 986c531c0b089ba0 | c3aabd1b57483432 | 0 |
| 2026-10-04-night-04 | 03b244d406cbbd88 | 6cacd768eea5230d | 0 |

## Results

### M2 browser, WebGPU, ABAB x5 (load 1.6-2.3)

| | base main-01 | night-01 (4-row kernels everywhere, pipelined) |
|---|---|---|
| prefill ms | 53.1 (51.8 / 56.7) | 52.9 (52.2 / 61.5) |
| decode ms/token | 7.2 (7.1 / 7.3) | 6.5 (6.4 / 6.5) |
| decode step by step ms/token (diag pass) | | 7.6 (7.5 / 7.8) |
| decode gpu span median | 6.1 (6.1 / 6.2) | 6.4 (6.2 / 6.4) |
| split/step qkv | 0.3 | 0.4 |
| split/step o_proj | 0.3 | 0.3 |
| split/step gate_up | 2.0 | 1.8 |
| split/step down | 0.9 | 1.1 |
| split/step lm_head | 1.6 | 1.6 |
| token hash | a454748c60e23841 (5/5) | a454748c60e23841 (5/5), step-by-step ids same (5/5) |

Medians (min / max), `diag=1`, the page's result rows; ms.

One run of night-03-oldmv (one-row kernels, pipelined): decode 6.2
ms/token, prefill 52.6 ms, same hash.

### M2 browser, WebGPU, night-04, ABAB x5 (load 2.0-2.4)

| | base main-01 | night-04 |
|---|---|---|
| prefill ms | 53.3 (52.0 / 60.9) | 54.0 (51.8 / 59.1) |
| decode ms/token | 7.1 (7.1 / 7.1) | 6.1 (6.1 / 6.2) |
| decode step by step ms/token (diag pass) | | 7.3 (7.2 / 7.5) |
| decode gpu span median | 6.1 (6.1 / 6.1) | 6.0 (6.0 / 6.1) |
| split/step qkv | 0.3 | 0.3 |
| split/step o_proj | 0.3 | 0.3 |
| split/step gate_up | 1.9 | 1.8 |
| split/step down | 0.9 | 0.9 |
| split/step lm_head | 1.6 | 1.6 |
| token hash | a454748c60e23841 (5/5) | a454748c60e23841 (5/5), step-by-step ids same (5/5) |

### M2 browser, night-04, every backend (one run each, load 2.1-2.7)

| browser | backend | prefill ms | decode ms/token | token hash |
|---|---|---|---|---|
| Chrome for Testing | webgpu | see above | 6.1 | a454748c60e23841 |
| Chrome for Testing | threads | 343.7 | 22.8 | a454748c60e23841 |
| Chrome for Testing | single | 1043.4 | 77.1 | a454748c60e23841 |
| Firefox 155 | threads | 441.2 | 24.9 | a454748c60e23841 |
| Firefox 155 | single | 1149.3 | 78.9 | a454748c60e23841 |
| Firefox 155 | auto (no WebGPU: picked threads) | | 24.7 | a454748c60e23841 |

### M2 browser, CPU threads backend, ABAB x5 (load 1.5-3.0)

| browser | base main-01 decode ms/token | night-02/03 decode ms/token | base prefill ms | night prefill ms |
|---|---|---|---|---|
| Firefox 155 | 37.8 (36.4 / 38.0) | 25.5 (25.1 / 26.4) | 440.9, 439.5 | 438.4, 445.5 |
| Chrome for Testing | 26.4 (26.3 / 26.5) | 23.1 (23.0 / 24.5) | 344.5, 347.2 | 347.6, 345.4 |

Decode: medians (min / max) of 5 interleaved runs each. Prefill: the 2
runs of the last two rounds (the first three rounds did not record it).
All 20 runs gave a454748c60e23841.

Single-thread backend (base, one run each): Chrome 77.3 ms/token, prefill
1043.6 ms; Firefox 78.6 ms/token, prefill 1155.4 ms. Thread pool init
(`diag=1`, one run each): Chrome 12.8 ms, Firefox 18.8 ms (base), 17.7 ms
(night-03).

A team for prefill too (not on the branch): Chrome prefill 401.1 / 402.9
ms and Firefox 506.7 / 497.5 ms against 342.0 / 342.8 and 440.4 / 478.8
for the base in the same rounds.

### M2 native, ABAB x5 (load 1.5-1.6)

| | base main-01 | night (4-row kernels everywhere) |
|---|---|---|
| prefill 36 tokens ms | 79.91 (77.54 / 80.27) | 79.99 (79.12 / 129.47) |
| decode ms/step, step loop | 8.98 (8.40 / 9.16) | 8.43 (8.11 / 9.20) |
| decode ms/step, pipelined | | 8.08 (8.00 / 8.18) |

Native decode GPU split (`--split`, one run each per round, 3 rounds,
load 1.5-2.2), ms per token:

| build | qkv | o_proj | gate_up | down | lm_head |
|---|---|---|---|---|---|
| base | 0.40-0.56 | 0.31-0.42 | 2.39-3.14 | 1.09-1.33 | 1.62-1.64 |
| 4-row everywhere | 0.50-0.51 | 0.45-0.47 | 2.53-2.55 | 1.20-1.22 | 1.67-1.76 |
| night-04 rule | 0.41-0.53 | 0.31-0.38 | 2.54-2.93 | 1.10-1.23 | 1.67 |

### RTX 3080, Chromium (Vulkan)

| run | base main-01 | night-02 (4-row everywhere) | night-03-oldmv (one-row) | night-04 |
|---|---|---|---|---|
| WebGPU decode ms/token, ABAB x5 vs base | 5.18 (5.04 / 5.22) | 3.50 (3.44 / 3.61) | | |
| WebGPU prefill ms, same runs | 129.4 (126.0 / 134.1) | 130.7 (127.4 / 134.5) | | |
| WebGPU decode ms/token, `diag=1`, ABAB x5 (night-02 vs oldmv) | | 3.56 (3.49 / 3.58) | 3.02 (2.89 / 3.14) | |
| WebGPU decode ms/token, `diag=1`, ABAB x5 (oldmv vs night-04) | | | 3.14 (2.97 / 3.20) | 3.05 (2.95 / 3.09) |
| decode gpu span median (diag) | | 3.1 (run 1) | 2.5 (all 10) | 2.4 (all 5) |
| split/step qkv / o_proj / gate_up / down / lm_head | | 0.4 / 0.4 / 0.5 / 1.1 / 0.3 (run 1) | 0.3 / 0.2 / 0.6 / 0.6 / 0.3 | 0.3 / 0.2 / 0.5 / 0.6 / 0.3 |
| WebGPU decode ms/token, no diag, 5 page loads | | | | 3.07 (2.84 / 3.19) |
| WebGPU prefill ms, same loads | | | | 126.9 (110.9 / 135.7) |
| CPU threads decode ms/token, ABAB x3 (one night-04 run) | 72.59 (72.28 / 73.83) | 27.07 (27.03 / 29.46) | | 27.6 |
| CPU threads prefill ms | 478.2 (454.3 / 502.1) | 460.4 (440.6 / 476.4) | | 531.9 |
| single thread decode ms/token (one run) | | 127.6 | | 126.6 |

With `diag=1` the reported prefill on the 3080 is 13-15 ms, against
110-136 ms without: the diag path drains the queue after engine creation
and after the weight upload before the timed prefill.

Values are the page's JSON fields `decodeMsPerTok`, `prefillMs` and the
diag rows. Every 3080 page load gave a454748c60e23841 (exact):
base 5 + 3, night-02 10 WebGPU + 4 threads + 1 single, night-03-oldmv 10, night-04 10 WebGPU + 1 threads + 1 single.

### Gates (native, M2)

At 7c9080b (night-04 code), one test binary per locked run:

| test | features | result | test time |
|---|---|---|---|
| fixture_parity (both kernel paths) | default | ok | 58.5 s |
| fixture_parity_qwen25_3b | default | ok | 40.0 s |
| fixture_parity_qwen3 | default | ok | 55.1 s |
| fixture_parity_qwen3_1_7b | default | ok | 164.4 s |
| fixture_parity_llama_360m | default | ok | 13.9 s |
| fixture_parity_llama_1_7b | default | ok | 35.4 s |
| kv_snapshot | default | ok, 2 tests | 40.3 s |
| logit_mask | default | ok, 3 tests | 4.0 s |
| pool_reuse | default | ok | 2.9 s |
| embed_head_sliced (LEAN_LORA_BIN2 set) | default | ok, 2 tests | 4.3 s |
| lora_parity | default | ok, 2 tests | 3.5 s |
| fixture_parity_llama_360m_cpu | default | ok | 13.2 s |
| fixture_parity_llama_360m_cpu | threads | ok | 4.5 s |
| cpu_lora_parity | threads | ok | 115.9 s |
| cpu_lora_parity | default | ok | 300.1 s |
| streaming_sampling (incl. pipelined_greedy_matches_step_loop) | default | ok, 4 tests | 6.3 s |
| lib tests | default / threads | ok, 17 / 21 | 0.2 s |
| clippy -D warnings, native all targets (default, threads) and wasm32 `web` | | clean | |

The same list passed at 0debe36 (4-row kernels everywhere).

## Firefox CPU threads: diagnosis

- On the M2, Firefox and Chrome run the single-thread backend at the same
  speed (78.6 vs 77.3 ms/token), so the SIMD128 kernels compile to the
  same speed in both. The threaded build is compiled with `+simd128` too
  (`scripts/build_lean_mt.sh` RUSTFLAGS), and Firefox has had wasm SIMD
  since version 89.
- With threads, Firefox was 1.43x slower than Chrome on the M2 (37.8 vs
  26.4 ms/token). The difference is in the threading, not the arithmetic.
- A decode step made about 170 rayon `par_iter` calls (7 matvecs per layer
  plus the head). Between calls rayon parks its idle threads after a few
  rounds of looking for work; on wasm32 `thread::yield_now` is a no-op, so
  those rounds take microseconds and the threads are asleep at every call.
  Each call then costs a wake (`Atomics.notify` to `memory.atomic.wait32`)
  of the pool threads, and the calling thread, which is not in the pool,
  blocks on a latch until the call ends.
- With one region per step (5356f0b) Firefox went from 37.8 to 25.5
  ms/token and Chrome from 26.4 to 23.1 on the M2; on the 3080 machine
  (24 threads, Linux Chromium) from 72.6 to 27.1 ms/token. The per-call
  wake cost is larger in Firefox than in Chrome, and grows with the thread
  count.
- On the phone, Firefox threads (282.4 ms/token) were about as slow as
  Chrome's single-thread backend (257.4) and 2.9x slower than Chrome
  threads (97.4): the same mechanism, with a larger per-wake cost, would
  explain it. Not verified on the phone.
- Thread pool init: 18.8 ms in Firefox on the M2 against 3977.7 ms on the
  phone. Not reproduced on the desktop, so not diagnosed. `initThreadPool`
  starts one worker per hardware thread, and each one imports `lean.js`
  and instantiates the shared module; the page runs it after the model
  download, not alongside it.

## Observations

- night-04 against lean-main: M2 browser decode 7.1 to 6.1 ms/token, 3080
  decode 5.18 to 3.05-3.07 ms/token, prefill unchanged on both. The GPU
  span barely moves (M2 6.1 to 6.0, 3080 2.5 to 2.4 against the one-row
  build): the gain is pipelining, which removes the per-token CPU encode
  and readback round trip from the critical path.
- With the 4-row kernels everywhere pipelining gave 6.5 ms/token on the M2
  and 3.50 on the 3080; with the one-row kernels 6.2 (one run) and 3.02.
- The 4-row kernels made the narrow matvecs slower on both desktop GPUs:
  on the 3080 down went from 0.6 to 1.1 ms and o_proj from 0.2 to 0.4 ms
  per token, on the M2 browser down from 0.9 to 1.1 ms. An 896-row output
  is 28 workgroups of 64 threads with 4 rows per lane, against 56 of 128
  with one. gate_up went from 2.0 to 1.8 ms on the M2 browser and from
  0.6 to 0.5 ms on the 3080. Hence the row-count rule in night-04.
- The phone question the 4-row kernels test is whether the decode matvecs
  there are limited by x loads (8x the weight bytes for Q4_0 with one row
  per lane): the phone's gate_up ran at about 7.9 GB/s (117.7 MB in 14.9
  ms) and the Q8_0 head at about 11.7 GB/s (144.6 MB in 12.4 ms), against
  an 18.9 GB/s probe, and Q4_0 (8x) ran slower than Q8_0 (4x).
- Pipelining hides the CPU encode (8.1 ms on the phone) behind the GPU
  work as long as encode is shorter than the GPU span (40.5 ms).
- The CPU team made prefill slower when tried there (M2 Chrome 342 to
  401 ms), so prefill keeps rayon.
- Prefill matmuls were not changed: the checkpoint-2 small-M variants A,
  B and C are still served from lean-prefill's servers and still need
  phone numbers before another variant is worth writing.

## Phone test URLs (for TC)

All on the M2, COOP/COEP (`scripts/serve_coi.py`), Qwen2.5-0.5B Q4_0, add
`&diag=1` for the split. Compare within one session.

| port | build | what it tests |
|---|---|---|
| 8833 | 2026-10-04-night-04 | the branch: pipelined decode, 4-row kernels for gate/up and the head, CPU team |
| 8834 | 2026-10-04-night-03-oldmv | same, one-row decode kernels everywhere (isolates the 4-row kernels) |
| 8835 | 2026-10-04-night-03 | same, 4-row kernels everywhere |
| 8831 | 2026-10-03-main-01 | lean-main, for a same-session baseline |
| 8801 / 8802 / 8803 | 2026-10-03-prefill-02-A / -B / -C | lean-prefill's small-M variants (prefill only), not run on a phone yet |

Pages: `/www/backends.html?backend=webgpu&diag=1` (Chrome),
`/www/backends.html?backend=threads&diag=1` (Chrome and Firefox).
