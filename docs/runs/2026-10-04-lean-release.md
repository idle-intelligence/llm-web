# 2026-10-04 lean-release: lean-perf-night + lean-chat-api in one branch

Branch `lean-release`, from `lean-perf-night` at 3e39484 (which already
contains lean-main 0ab85d7 and lean-main's streaming `cache.put`, 94a16ff),
with `lean-chat-api` at b4d1f09 merged in.

## Commits

| commit | change |
|---|---|
| 5db6077 | merge of lean-chat-api (resolutions below) |
| c32132d | `scripts/build_lean_mt.sh`: 2.5 GiB shared memory maximum, `?v=ENGINE_BUILD` on every worker-helper import, home-path remap in the script |
| 5b7ea03 | ENGINE_BUILD `2026-10-04-release-01` on every loading URL; `chat_worker.js` reports `wasmMemoryBytes` in its `ready` and `done` messages |
| (this doc) | run log |

## Merge resolutions

- `src/web.rs`, `LeanEngine::generate`: the one content conflict.
  lean-perf-night had replaced the step loop with
  `decode_greedy_pipelined`; lean-chat-api had kept the step loop and
  routed each id through `TokenSink` (text deltas, callback errors
  rethrown). Resolved to the pipelined decode with the sink inside its
  `on_token` closure: the closure returns `false` once the sink has
  stopped (the callback threw), otherwise it calls `sink.token(id)`, and
  `sink.finish()?` runs after the decode, so a callback error rejects the
  call as on lean-chat-api. The stop point and the generated ids are those
  of the step loop: a stopped sink ends the decode before the next id is
  pushed, and `decode_greedy_pipelined` rewinds `kv_len` for the step it
  had already submitted.
- `generateStream` and both engines' `chatGenerate` merged without
  conflict: they reach `generate::decode_loop`, whose greedy path is
  lean-perf-night's pipelined decode. Its closure checks `should_stop`
  (the `AbortFlag` or a stopped sink) before each id and calls
  `on_token(id)`, which lean-chat-api's call sites pass as
  `sink.token(id)`, so the pipelined path emits a text delta per token and
  honours the `AbortFlag`. The CPU chat path (`chat::cpu_decode_loop`) and
  lean-perf-night's `cpu_team.rs` do not overlap.
- `www/*`: every conflict was the ENGINE_BUILD tag line (night-07 against
  chat-01), resolved to the chat side and then bumped to `release-01` in
  5b7ea03. lean-perf-night's `decodeGreedy` changes in
  `backends_worker.js` merged without conflict.

## build_lean_mt.sh

- `--max-memory`: SmolLM2-1.7B Q4_0 trapped with `unreachable` during
  `load` on the threads build at 1 GiB. Measured with a 4 GiB build
  (`chat.html?backend=threads&model=smollm2-1.7b`, max_ctx 2048, 8 pool
  threads, Chromium): `wasmMemoryBytes` 2091450368 after load and
  2106130432 after three turns. The CPU backend keeps the 0.93 GiB of
  quantized weights resident and allocates a float32 KV cache of 384 KiB
  per position (0.75 GiB at 2048). 2 GiB (2147483648) would leave 40 MiB;
  2.5 GiB (2684354560, 40960 pages) leaves 0.54 GiB for longer contexts
  and more pool threads. `MAX_MEMORY` overrides it.
- Cache bust: the script now requires `ENGINE_BUILD` and patches all three
  untagged URLs in the chain: `lean.js` importing `workerHelpers.js`,
  `workerHelpers.js` spawning its own worker script, and the worker
  importing `lean.js`. It fails if a patch did not apply.
- Remap: `--remap-path-prefix=$HOME=/home` is in the script's RUSTFLAGS,
  and it fails if `lean_bg.wasm` still contains the home directory.

## Parameters

- M2 laptop (Metal). Timed runs held the gpu, browser and cargo locks and
  started with a 1-minute load average under 3 and no rustc or cargo
  running; load averages per table.
- M2 browsers: Playwright's Chrome for Testing (headless,
  `--use-angle=metal`, adapter apple / metal-3) and Firefox 155 (headless,
  CPU backends only).
- Linux desktop with an RTX 3080: Chromium on Xwayland (Vulkan, adapter
  nvidia / ampere), 24 hardware threads, one page load at a time under the
  box lock.
- Page: `crates/lean/www/backends.html` (Qwen2.5-0.5B-Instruct Q4_0,
  36-token prompt, 64 greedy tokens, no `diag`), and `www/chat.html`
  (3-turn greedy conversation from the chat API run, max 48 tokens per
  turn, max_ctx 2048).
- Base = lean-main 0ab85d7's pkg and pkg-mt (ENGINE_BUILD
  2026-10-03-main-01, wasm sha256 94201fe07f56b1da / 4a151403b00fd4b1),
  served from the lean-perf-night snapshot; served bytes hashed before the
  runs.
- Native: `prefill_sweep --lens 36 --reps 3 --decode 63`, base binary built
  from 0ab85d7, interleaved with the release binary.
- wasm builds: `RUSTFLAGS="-C target-feature=+simd128
  --remap-path-prefix=$HOME=/home" wasm-pack build crates/lean --target web
  --release --no-default-features --features web` and
  `ENGINE_BUILD=2026-10-04-release-01 scripts/build_lean_mt.sh`.

## wasm bytes

| file | sha256 | `/Users/` strings |
|---|---|---|
| pkg/lean_bg.wasm | bd2ddf09595b8da9fba0387d11b908ec83d12c45339ae1a2f509f01d5f5d6343 | 0 |
| pkg-mt/lean_bg.wasm | 05fd4d98001e6888b7e9a993ea969831e9660635741caf4c19e97789e6d1dc4a | 0 |

Served on port 8840 at `?v=2026-10-04-release-01` with the same hashes;
the pkg-mt glue declares `maximum:40960` pages.

## Results

### Native gates, M2 (`--release`, `--ignored`, one test binary per locked run)

| test | features | result | test time |
|---|---|---|---|
| fixture_parity (both kernel paths) | default | ok | 58.2 s |
| fixture_parity_qwen25_3b | default | ok | 39.9 s |
| fixture_parity_qwen3 | default | ok | 55.1 s |
| fixture_parity_qwen3_1_7b | default | ok | 166.1 s |
| fixture_parity_llama_360m | default | ok | 13.9 s |
| fixture_parity_llama_1_7b | default | ok | 35.4 s |
| fixture_parity_llama_360m_cpu | default | ok | 13.4 s |
| fixture_parity_llama_360m_cpu | threads | ok | 4.9 s |
| kv_snapshot | default | ok, 2 tests | 40.5 s |
| logit_mask | default | ok, 3 tests | 4.0 s |
| pool_reuse | default | ok | 2.7 s |
| streaming_sampling (incl. pipelined_greedy_matches_step_loop) | default | ok, 4 tests | 6.4 s |
| chat_api (incl. SmolLM2-1.7B Q4_0) | default | ok, 5 tests | 17.3 s |
| chat_api (incl. SmolLM2-1.7B Q4_0) | threads | ok, 5 tests | 9.6 s |
| embed_head_sliced (LEAN_LORA_BIN2 = a-norules-300) | default | ok, 2 tests | 4.3 s |
| lora_parity | default | ok, 2 tests | 3.3 s |
| cpu_lora_parity | threads | ok | 118.7 s |
| cpu_lora_parity | default | ok | 300.4 s |
| lib tests | default / threads | ok, 17 / 22 | 0.2 / 0.3 s |
| clippy -D warnings, native all targets | default, threads | clean | |
| clippy -D warnings, wasm32 `web` | | clean | |

### llm-life gates (lean-migration f0fa7bf, lean patched to lean-release with `--config patch...`, no file changed)

| test | features | per-cell / grid result | vs HF+PEFT |
|---|---|---|---|
| per_cell_base_fewshot | default | 175/512 | 0 differ, max 5.341e-5 |
| per_cell_base_rules | default | 215/512 | 0 differ, max 6.866e-5 |
| per_cell_a_norules | default | 512/512 | 0 differ, max 2.060e-4 |
| per_cell_a_rules | default | 509/512 | 0 differ, max 2.499e-4 |
| whole_grid | default | 157/256, 256/256, 154/256, 256/256, 677/1024, 1024/1024 | 0 differ, max 4.058e-4 |
| cpu_per_cell_a_norules | lean/threads | 512/512 | 0 differ, max 3.242e-4 |
| cpu_per_cell_base_rules | lean/threads | 215/512 | 0 differ, max 8.965e-5 |
| cpu_whole_grid | lean/threads | same counts as whole_grid | 0 differ, max 4.809e-4 |
| cpu_whole_grid | default (one thread) | same counts as whole_grid | 0 differ, max 4.809e-4 |

### M2 browser, backends.html, one run each (load 2.09-2.21)

| browser | backend | prefill ms | decode ms/token | token hash |
|---|---|---|---|---|
| Chrome for Testing | webgpu | 85.1 | 6.1 | a454748c60e23841 |
| Chrome for Testing | threads | 350.1 | 24.0 | a454748c60e23841 |
| Chrome for Testing | single | 1041.9 | 77.2 | a454748c60e23841 |
| Firefox 155 | threads | 420.3 | 24.1 | a454748c60e23841 |

### M2 browser, chat.html, 3 turns, max 48 tokens

| model | browser, backend | turn tokens | reply == streamed == shown | console errors | wasm memory after turn 3 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | Chrome, webgpu | 48, 19, 16 | yes | 0 | 324927488 |
| Qwen2.5-0.5B Q4_0 | Chrome, threads | 48, 19, 16 | yes | 0 | 613679104 |
| SmolLM2-360M Q4_0 | Chrome, webgpu | 7, 28, 27 | yes | 0 | 113508352 |
| SmolLM2-360M Q4_0 | Chrome, threads | 7, 28, 27 | yes | 0 | 528613376 |
| SmolLM2-1.7B Q4_0 | Chrome, threads | 7, 42, 22 | yes | 0 | 2106130432 |
| SmolLM2-1.7B Q4_0 | Firefox, threads | 7, 42, 22 | yes | 0 | 2106130432 |
| SmolLM2-1.7B Q4_0 | Chrome, webgpu | 7, 42, 22 | yes | 0 | 174718976 |

| check | result |
|---|---|
| stop mid-reply, webgpu, Qwen2.5-0.5B | stopped after 8 tokens, streamed == reply; next turn "The dragon was learning to cook." |
| stop mid-reply, threads, Qwen2.5-0.5B | stopped after 7 tokens, streamed == reply; next turn "The dragon was cooking." |

### M2 browser, backends.html, base main-01 against release-01, ABAB

| browser, backend | rounds (load) | base prefill ms | release prefill ms | base decode ms/token | release decode ms/token |
|---|---|---|---|---|---|
| Chrome, webgpu | 5 (2.76-2.99) | 85.0 (83.4 / 87.4) | 85.0 (84.0 / 90.5) | 7.1 (7.1 / 7.1) | 6.1 (6.1 / 6.2) |
| Chrome, threads | 5 (2.45-2.81) | 343.4 (341.7 / 347.1) | 346.1 (341.5 / 356.2) | 26.4 (26.3 / 26.4) | 22.7 (22.4 / 23.2) |
| Firefox, threads | 3 (2.51-2.97) | 439.9 (439.5 / 440.1) | 418.7 (416.0 / 418.8) | 36.9 (36.9 / 37.4) | 24.3 (23.9 / 24.4) |

Medians (min / max) of the page's `prefillMs` and `decodeMsPerTok`. All
26 runs gave a454748c60e23841.

### M2 native, prefill_sweep, ABAB x5 (load 2.37-2.78)

| | base 0ab85d7 | release |
|---|---|---|
| prefill 36 tokens ms | 79.63 (77.85 / 80.28) | 79.59 (77.72 / 79.66) |
| decode ms/step, step loop | 8.95 (8.09 / 9.06) | 8.62 (8.01 / 9.07) |
| decode ms/step, pipelined | | 9.10 (8.58 / 9.37) |

Against lean-perf-night's night-04 binary, ABAB x3 (load 1.71-2.08):
pipelined 8.79 (8.65 / 8.83) for night-04 and 9.04 (8.96 / 9.15) for
release; step loop 8.77, 9.30, 8.39 against 8.42, 9.47, 8.67.

### RTX 3080, native gates (`--release`, `--ignored`, under the box lock)

Every test in the M2 native table above except clippy and the
single-thread cpu_lora_parity passed: fixture_parity (all six),
kv_snapshot, logit_mask, pool_reuse, streaming_sampling (4),
chat_api (5, incl. SmolLM2-1.7B Q4_0) default and threads,
fixture_parity_llama_360m_cpu default and threads, embed_head_sliced,
lora_parity, cpu_lora_parity (threads), lib tests 17 / 22. GPU idle before
the run (0 %, 515 MiB). Log: `data/2026-10-04-lean-release/rtx3080-native-gates.log`.

### RTX 3080, Chromium (Vulkan), base main-01 against release-01, ABAB, one page load per run

| backend | rounds | base prefill ms | release prefill ms | base decode ms/token | release decode ms/token |
|---|---|---|---|---|---|
| webgpu | 5 | 129.9 (108.2 / 132.8) | 120.4 (107.0 / 132.1) | 5.16 (4.92 / 5.19) | 3.03 (2.77 / 3.15) |
| threads | 3 | 476.3 (456.8 / 552.0) | 443.1 (412.1 / 460.1) | 71.74 (71.26 / 72.78) | 24.00 (23.71 / 24.35) |

Release single thread, one load: prefill 1673.1 ms, decode 127.89
ms/token. Every one of the 17 page loads gave a454748c60e23841 and
`matchesReference` true, including the 5 release WebGPU loads. GPU idle
before the runs (0 %, 515 MiB). Log: `data/2026-10-04-lean-release/rtx3080-browser.log`.

Raw logs for the M2 tables are in `data/2026-10-04-lean-release/` too
(the single-thread llm-life `cpu_whole_grid` row was read from its
console output, which the log does not hold).

## Observations

- The merge keeps both branches' numbers: on the M2 browser the release
  build decodes at 6.1 ms/token on WebGPU (base 7.1), 22.7 on Chrome
  threads (base 26.4) and 24.3 on Firefox threads (base 36.9), the
  lean-perf-night values, with lean-chat-api's chat replies token for
  token (same turn lengths and replies as its run doc on both models and
  every backend run).
- SmolLM2-1.7B Q4_0 loads on the threads build and gives the WebGPU
  replies in Chrome and Firefox; the memory reached 2106130432 bytes.
- Native on the M2 the pipelined decode measured 9.04-9.10 ms/step here,
  against 8.08 in lean-perf-night's run at load 1.5; night-04's binary
  measured 8.79 in the same rounds. The merge does not change `model.rs`
  or the kernels. In the browser, where the readback round trip is
  longer, pipelining is the 7.1 to 6.1 gain.
- On the 3080 machine release decodes at 3.03 ms/token on WebGPU (base
  5.16) and 24.00 on threads (base 71.74), the lean-perf-night night-04/07
  values (3.05-3.08 and 23.06-23.63).
- The llm-life gates give the same counts and max logit differences as
  lean-main's run on every row.

## What TC pushes

`lean-release` (HEAD = this doc's commit). It contains lean-main,
lean-perf-night and lean-chat-api; pushing it makes those three branches
redundant. pkg/pkg-mt are build outputs (gitignored); the pages expect
ENGINE_BUILD `2026-10-04-release-01`, and a future rebuild bumps the tag
on every loading URL and passes it to `build_lean_mt.sh`.
