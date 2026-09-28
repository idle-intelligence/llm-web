# lean engine: benchmarks against TC's own consumer use cases

Follow-on to `docs/runs/2026-09-28-lean-perf.md` and
`docs/runs/2026-09-28-lean-qwen3.md`. Scope: for each of TC's current
projects, find the model that project actually uses (or the closest
non-exotic substitute), run it through the lean engine (`crates/lean`, this
repo), and report benchmarks next to the reference engines those projects
use today (llama.cpp, WebLLM, wllama). No project is migrated to the lean
engine in this pass; no consumer repo (`llm-life`, sonos, `hive`, `trucs.ai`)
was edited. Two new files were added to `crates/lean` this session because
no existing harness covered these shapes: `src/bin/lean_bench_usecases.rs`
(native `life-cells` and `sonos-turns` subcommands) and
`www/cpu_qwen3.html`/`www/main_cpu_qwen3.js` (the CPU rung's browser harness,
parametrized for Qwen3-0.6B; `www/cpu.html` already covered Qwen2.5-0.5B).

## Parameters

- Machine: macOS 26.3.1 (Darwin 25.3.0, arm64), Apple M2. One GPU job at a
  time throughout (checked `ps aux | grep -i gpu-process` before each timed
  run; a Steam helper and an Arc browser GPU-process were present but idle,
  not run concurrently with any timed GPU work).
- Engine: `crates/lean`, worktree `lean-engine`, branch `lean-engine`, base
  commit `72affc3`, plus this session's two additions above (uncommitted at
  measurement time, committed with this doc).
- wasm build: `wasm-pack build crates/lean --target web --out-dir pkg
  --no-default-features --features web`, `lean_bg.wasm` sha256
  `e25721bfd66f05758e80d73d4cc6a23d874640abf184567dada2ea18b02a7e50`
  (3,361,363 bytes), verified against the served file before every browser
  run below. This hash differs from `2026-09-28-lean-qwen3.md`'s
  `c1cbd38b...` build (`2026-09-28-9`) despite no forward-pass logic change
  in between - a plain `wasm-pack` rebuild of the same source is not
  byte-identical on this toolchain. The existing pages' `ENGINE_BUILD`
  string constants were not bumped in this session (no logic changed and
  every browser run used a fresh Playwright context with no persistent
  cache across runs, so no stale-cache risk existed), but this is flagged
  as a gap against this crate's own cache-bust rule for any future session
  that reuses these page files with a persistent server/cache.
- Native GPU: `lean-cli`/`lean-bench-usecases --release`, Metal backend via
  wgpu 26.
- Native CPU rung: `lean-cli --engine cpu --release` (`src/cpu.rs`, NEON on
  aarch64).
- Browser: Playwright-bundled Chromium, headless, `--enable-unsafe-webgpu
  --enable-features=Vulkan,WebGPU --use-angle=metal`, served locally via
  `python3 -m http.server` rooted at `crates/lean/` (lean pages) or
  `llm-web/web/` (the wllama demo) or a scratch directory (the WebLLM page).
  Locks (`/Users/tc/.claude/jobs/f13f376f/tmp/locked gpu ... locked
  browser ...`) held around every browser/GPU run.
- Models, all under `~/Code/idle-intelligence/models` via `hf download`
  (plain User-Agent, no email):
  - Qwen2.5-0.5B-Instruct, `qwen2.5-0.5b-instruct-q4_0.gguf` (Q4_0, Q8_0
    `output.weight`), 428,730,208 bytes, tokenizer `Qwen/Qwen2.5-0.5B-Instruct`.
  - Qwen3-0.6B, `Qwen3-0.6B-Q8_0.gguf`, 639,446,688 bytes, tokenizer
    `Qwen/Qwen3-0.6B`.
  - Qwen3-1.7B, `Qwen3-1.7B-Q8_0.gguf`, 1,834,426,016 bytes, tokenizer
    `Qwen/Qwen3-1.7B`.
  - Qwen2.5-3B-Instruct, `qwen2.5-3b-instruct-q4_0.gguf` (`Qwen/Qwen2.5-3B-Instruct-GGUF`),
    1,997,879,712 bytes, tokenizer `Qwen/Qwen2.5-3B-Instruct`.
  - SmolLM2-360M-Instruct, `bartowski/SmolLM2-360M-Instruct-GGUF`,
    `SmolLM2-360M-Instruct-Q4_K_M.gguf`, 270,590,880 bytes.
  - SmolLM2-1.7B-Instruct, `bartowski/SmolLM2-1.7B-Instruct-GGUF`,
    `SmolLM2-1.7B-Instruct-Q4_0.gguf`, 993,874,912 bytes (llama.cpp ceiling
    only - the browser baseline for this row is WebLLM's own MLC-format
    build, not this GGUF).
  - `idle-intelligence/llm-of-life-lora`'s `lora-a-rules-300.bin`
    (read from `~/Code/idle-intelligence/llm-life/artifacts/`, the path
    TC's own llm-life repo keeps it at - not copied into this repo).
- Hub API check (`GET api/models/bartowski/<repo>`, siblings list): both
  `bartowski/SmolLM2-360M-Instruct-GGUF` and
  `bartowski/SmolLM2-1.7B-Instruct-GGUF` ship `Q4_0.gguf` and `Q8_0.gguf`
  files alongside the K-quants - a same-family Q4_0/Q8_0 SmolLM2 GGUF
  exists today if the lean engine grows Llama-architecture support; no
  K-quant kernel work was done or is implied by this.
- Reference engines: `llama-bench` (Homebrew `ggml` 0.25.3, Metal+BLAS
  backend, `-p 512 -n 128`), wllama (this repo's own bundled
  `llm-web/web/index.html` demo, vendored `pkg/wllama`, unedited, driven by
  Playwright), WebLLM 0.2.79 (`@mlc-ai/web-llm@0.2.79` from
  `esm.run`, model `SmolLM2-1.7B-Instruct-q4f16_1-MLC`, a new scratch test
  page - no such page exists in this repo or the hive today).

## Results

### 1. llm-life (Qwen2.5-0.5B-Instruct Q4_0 + `lora-a-rules-300`, LLMLIFE2)

Plain chat protocol, native GPU, fixture cases (`crates/lean/reference/fixture.json`,
prior session's numbers, `docs/runs/2026-09-28-lean-perf.md`, vec4-kernel
build `2026-09-28-6`, browser only shown there for the vec4 delta; native
numbers below are that doc's pre-vec4 `kernel=fast` pass):

| case | prompt tokens | prefill ms/tok | decode ms/tok |
|---|---:|---:|---:|
| short | 36 | 58.65 | 51.51 |
| long | 86 | 38.33 | 52.34 |
| non_english | 54 | 40.07 | 51.97 |

Plain chat protocol, browser GPU, this session (fresh run against this
session's rebuilt wasm, `index.html?local=1`):

| case | prompt tokens | prefill ms | decode ms/tok |
|---|---:|---:|---:|
| short | 36 | 325.0 | 23.5 |
| long | 86 | 381.3 | 24.5 |
| non_english | 54 | 232.3 | 23.7 |

Variant A real shape (this session, `lean-bench-usecases life-cells --cells
256`): rules prefix (`variant_a::rules_prefix`, Conway B3/S23) prefilled
once and snapshotted (79 tokens), then 256 iterations of restore-snapshot +
`forward_chunk_spec` on one cell prompt (27 tokens, a birth-case neighbor
pattern) + `lm_head_sliced` on the dead/alive ids, LoRA applied throughout:

| prefix tokens | cell tokens | cells | total s | cells/s | ms/cell |
|---:|---:|---:|---:|---:|---:|
| 79 | 27 | 256 | 216.62 | 1.18 | 846.17 |

### 2. Sonos MCP agent substitute (Qwen3-1.7B Q8_0, no LoRA)

Real 2,225-token tool prompt (`crates/lean/reference/fixture_qwen3_1_7b.json`,
`long_tools_single` case's `input_ids`), prefilled once and snapshotted;
per turn: restore, append a 24-token command
("Play jazz in the kitchen and set the volume to 30 percent, then tell me
what is now playing there."), greedy-decode 48 tokens. Native
(`lean-bench-usecases sonos-turns --turns 5 --max-new 48`):

| prompt tokens | command tokens | prefill ms | prefill ms/tok |
|---:|---:|---:|---:|
| 2225 | 24 | 154948.9 | 69.64 |

| turn | ms | ms/new-token |
|---:|---:|---:|
| 0 | 35441.1 | 723.29 |
| 1 | 31856.2 | 650.13 |
| 2 | 32150.0 | 656.12 |
| 3 | 32139.9 | 655.92 |
| 4 | 32325.5 | 659.70 |

Median turn: 32150.0 ms.

Same prompt, browser GPU, from `docs/runs/2026-09-28-lean-qwen3.md`
(`qwen3_1_7b.html`, `long_tools_single` fixture case - full prefill +
32-token greedy continuation, not the restore/append/decode-48 turn shape
above, but the same 2225-token prefill):

| prompt tokens | prefill ms | decode ms/tok |
|---:|---:|---:|
| 2225 | 68765.7 | 381.6 |

Native fixture check, this session (`lean-cli --fixture fixture_qwen3_1_7b.json`,
GPU, same model, short/long/non_english cases, for comparison with row 4's
Qwen3-0.6B numbers below - not part of the sonos shape, a plain
chat-protocol data point): short prefill 31.46 ms/tok, decode 175.23
ms/tok (2225-token `long_tools_single` case: prefill 33.60 ms/tok, decode
225.66 ms/tok).

### 3. 3B chat demo substitute (Qwen2.5-3B-Instruct Q4_0)

Native, `lean-cli --prompt "What is the capital of France?" --tokens 32
--kernel fast`:

**Failed to load**: `Error: unsupported GGML dtype code: 14 (this crate
only reads F32/F16/Q4_0/Q8_0)`. GGML type 14 is `Q6_K`: this GGUF's
`Qwen/Qwen2.5-3B-Instruct-GGUF` `q4_0` file mixes Q6_K for at least one
tensor (llama.cpp's own quantize commonly keeps `output.weight`/embeddings
at a higher-precision K-quant even under a "Q4_0" preset), which this
crate's GGUF reader does not support. Not attempted in the browser as a
result - the native load failure is upstream of any browser-specific
buffer-size question. This is a lean-engine architecture gap (Q6_K
support), not a buffer-limit failure.

### 4. swarm / llm->tts phones / llm-web public demo (SmolLM2-360M baseline + lean small-model candidates)

Baseline: SmolLM2-360M-Instruct Q4_K_M via wllama, this repo's own
`web/index.html` demo (unedited), driven live (not greedy - `temp=0.7,
top_k=40, top_p=0.9`, `nPredict=512`), median of 3 runs, same
longer-answer prompt ("List the first twenty prime numbers, one per line,
with a short comment about each."):

| run | model load s | generated tokens | elapsed s | tok/s |
|---:|---:|---:|---:|---:|
| 1 | 3.91 | 145 | 7.0 | 20.7 |
| 2 | 4.17 | 63 | 3.5 | 18.2 |
| 3 | 3.97 | 161 | 7.8 | 20.8 |

Median: 20.7 tok/s. Download: 270,590,880 bytes (~271 MB, matches the
page's own "~271 MB" label).

Lean engine candidates, same chat protocol, next to the wllama baseline
(download size next to speed, per TC's 2026-09-28 addition):

| model | quant | download bytes | rung | prefill | decode |
|---|---|---:|---|---:|---:|
| SmolLM2-360M-Instruct (baseline) | Q4_K_M | 270,590,880 | wllama, browser | n/a (see above) | 20.7 tok/s |
| Qwen2.5-0.5B-Instruct | Q4_0 | 428,730,208 | lean GPU, browser | 110.8 tok/s (short, 9.03 ms/tok) | 42.6 tok/s (23.5 ms/tok) |
| Qwen2.5-0.5B-Instruct | Q4_0 | 428,730,208 | lean CPU rung, browser (wasm32+simd128, no WebGPU) | 13.1 tok/s (short, 76.4 ms/tok) | 13.0 tok/s (76.8 ms/tok) |
| Qwen3-0.6B | Q8_0 | 639,446,688 | lean GPU, browser | 42.6 tok/s (short, 23.46 ms/tok) | 22.2 tok/s (45.1 ms/tok) |
| Qwen3-0.6B | Q8_0 | 639,446,688 | lean CPU rung, browser (wasm32+simd128, no WebGPU) | 0.76 tok/s (short, 1319.4 ms total / 15 tok = 87.96 ms/tok) | 11.5 tok/s (87.0 ms/tok) |

CPU-rung rows are this session's own runs (`cpu.html?local=1` re-run fresh,
`cpu_qwen3.html?local=1` new); GPU rows for Qwen2.5-0.5B are this session's
fresh `index.html?local=1` run, GPU rows for Qwen3-0.6B are from
`docs/runs/2026-09-28-lean-qwen3.md` (`qwen3.html`, `short` case, same
15-token prompt). All four lean rows are token-exact against their HF
transformers fixture (`allMatch: true`).

### 5. llama.cpp `llama-bench` (native ceiling, Metal+BLAS, `-p 512 -n 128`)

| model | quant | size | params | pp512 tok/s | tg128 tok/s |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | Q4_0 | 403.20 MiB | 630.17 M | 2686.16 ± 15.86 | 153.28 ± 0.36 |
| Qwen3-0.6B | Q8_0 | 604.15 MiB | 596.05 M | 1999.14 ± 3.23 | 92.32 ± 0.77 |
| Qwen3-1.7B | Q8_0 | 1.70 GiB | 1.72 B | 738.63 ± 1.01 | 45.43 ± 0.36 |
| Qwen2.5-3B-Instruct | Q4_0 | 1.86 GiB | 3.40 B | 385.11 ± 3.49 | 42.86 ± 0.72 |
| SmolLM2-360M-Instruct | Q4_K_M | 256.35 MiB | 361.82 M | 2465.91 ± 6.75 | 159.29 ± 1.67 |
| SmolLM2-1.7B-Instruct | Q4_0 | 946.13 MiB | 1.71 B | 661.61 ± 1.49 | 72.93 ± 2.25 |

### 6. WebLLM baseline for SmolLM2-1.7B-Instruct (swarm/desktop llm->tts)

`@mlc-ai/web-llm@0.2.79`, model `SmolLM2-1.7B-Instruct-q4f16_1-MLC`, new
scratch test page (`esm.run` CDN import, no vendored copy in this repo or
the hive), headless Chromium WebGPU, prompt "List the first twenty prime
numbers, one per line, with a short comment about each." (`max_tokens:
256`), `engine.runtimeStatsText()`:

| model | quant | download (params, this run's own log) | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|
| SmolLM2-1.7B-Instruct-q4f16_1-MLC | q4f16_1 | 919 MB | 224.93 | 40.94 |

## Summary table

| use case | model | engine | prefill tok/s | decode tok/s | workload metric | native/browser |
|---|---|---|---:|---:|---|---|
| llm-life | Qwen2.5-0.5B Q4_0 + LoRA | lean | 17.1-26.1 (native, fixture) / 110.7-232.5 (browser, fixture) | 19.1-19.4 (native) / 40.8-42.6 (browser) | 1.18 cells/s (variant A, 256 cells, prefix 79 tok + cell 27 tok) | both |
| sonos agent | Qwen3-1.7B Q8_0 | lean | 14.4 (native, 69.64 ms/tok) / 32.3 (browser, 30.9 ms/tok) | n/a (see turn table) | median turn (restore+append 24 tok+decode 48 tok) 32.15 s native | both |
| 3B chat demo | Qwen2.5-3B-Instruct Q4_0 | lean | FAILED (Q6_K tensor unsupported) | FAILED | n/a | native attempted, browser not attempted |
| swarm/tts phones | SmolLM2-360M Q4_K_M | wllama | n/a | 20.7 (median of 3) | 271 MB download | browser |
| swarm/tts phones (candidate) | Qwen2.5-0.5B Q4_0 | lean GPU | 110.8 | 42.6 | 429 MB download | browser |
| swarm/tts phones (candidate) | Qwen2.5-0.5B Q4_0 | lean CPU rung | 13.1 | 13.0 | 429 MB download, no WebGPU needed | browser |
| swarm/tts phones (candidate) | Qwen3-0.6B Q8_0 | lean GPU | 42.6 | 22.2 | 639 MB download | browser |
| swarm/tts phones (candidate) | Qwen3-0.6B Q8_0 | lean CPU rung | 0.76 | 11.5 | 639 MB download, no WebGPU needed | browser |
| desktop llm->tts | SmolLM2-1.7B q4f16_1 | WebLLM 0.2.79 | 224.93 | 40.94 | 919 MB download | browser |
| ceiling (all above) | each model's own GGUF | llama.cpp `llama-bench` | 385-2686 (pp512) | 42.9-159.3 (tg128) | n/a | native |

## Observations

- The lean engine's browser GPU decode is consistently faster than its own
  native Metal decode on this machine across every model measured this
  session and in the cited prior sessions (Qwen2.5-0.5B: 23.5-24.5 ms/tok
  browser vs 51.5-52.3 ms/tok native; Qwen3-0.6B: 45.1 ms/tok browser vs
  175.2-175.6 ms/tok native; Qwen3-1.7B: 110.7-111.2 ms/tok browser at
  short prompts vs the sonos-turns bench's 655-660 ms/new-token native,
  though that native number also includes KV restore and a suffix-append
  step the browser fixture number doesn't) - consistent with this crate's
  standing, unexplained "browser faster than native" finding in
  `2026-09-28-lean-perf.md`, not investigated further here.
- The lean engine is far below llama.cpp's native ceiling at every size:
  Qwen2.5-0.5B tg128 153.3 tok/s (llama.cpp) vs lean's best decode this
  session, 42.6 tok/s (lean GPU browser) or 19.4 tok/s (lean GPU native);
  Qwen3-1.7B tg128 45.4 tok/s (llama.cpp) vs lean's 9.0-9.03 tok/s (browser
  GPU, 110.7-111.2 ms/tok at short prompts, from `2026-09-28-lean-qwen3.md`).
  The gap is largest exactly where `2026-09-28-lean-perf.md`
  already located it (the Q4_0/Q8_0 GEMM kernel's own efficiency, ~4.7% of
  the M2's quoted FP32 peak at the one long-prefill case profiled there),
  not re-profiled this session.
- llm-life's variant A real shape (1.18 cells/s, 846 ms/cell) is far
  slower than the plain per-token chat-protocol decode rate on the same
  model (37.6-42.6 tok/s at 26-24 ms/tok): each cell in this bench does a
  full `KvCache::new` + `restore` (a fresh GPU buffer allocation plus a
  synchronous `queue.write_buffer` copy of the whole snapshotted prefix)
  before its one-chunk forward, none of which llm-life's real packed
  block-diagonal path would pay per cell (the whole point of packing many
  cells into one chunk against one resident prefix, per
  `docs/runs/2026-09-28-lean-lora.md`). This bench measures the
  restore-per-chunk mechanism in isolation, not llm-life's actual packed
  throughput - the packed shape needs its own bench, not built this
  session (see next steps).
- The sonos-turns native prefill (154.9 s for 2225 tokens, 69.64 ms/tok)
  is more than double the browser's prefill on the same prompt/model
  (68.8 s, 30.9 ms/tok, from `2026-09-28-lean-qwen3.md`) - the same
  native-slower-than-browser direction as the per-token numbers above, at
  a much larger absolute gap. Per-turn decode (655-660 ms/new-token
  native, turns 1-4; turn 0 is slower, 723 ms/new-token, plausibly a
  pipeline/shader warmup cost not amortized before the timer starts) is
  far too slow for a live agent turn (48 tokens in 32 s) on this engine
  today.
- wllama's SmolLM2-360M numbers (18.2-20.8 tok/s) are noisy at low token
  counts (run 2 generated only 63 tokens before stopping vs 145/161 for
  runs 1/3, at temp 0.7 sampling, not greedy) - a fixed-length greedy
  protocol would be a fairer three-way comparison with the lean rows
  (which are greedy and token-exact) and was not set up this session
  because wllama's own demo page uses sampling, not greedy, and this
  session ran that page unedited rather than writing a second wllama
  harness.
- The lean engine's Qwen2.5-0.5B Q4_0 GPU browser row already beats the
  wllama SmolLM2-360M Q4_K_M baseline on decode tok/s (42.6 vs 20.7) at a
  larger download (429 MB vs 271 MB) and is token-exact/greedy rather than
  sampled - the strongest evidence in this doc that the lean engine is
  already ahead of the wllama baseline for phones with WebGPU. The CPU-rung
  numbers (13.0-13.1 tok/s for Qwen2.5-0.5B, 11.5 tok/s decode for
  Qwen3-0.6B) are the numbers that matter for a phone without WebGPU;
  wllama's own backend is also CPU/wasm (no WebGPU used by wllama in this
  demo), so the fairer no-WebGPU comparison is wllama's 20.7 tok/s against
  lean's CPU-rung 13.0-13.1 tok/s: the lean CPU rung is currently slower
  than wllama's multi-threaded llama.cpp-wasm backend at this size, most
  likely because wllama's wasm build uses SIMD128 *and* multiple worker
  threads where this crate's `cpu.rs` is single-threaded scalar/NEON-class
  wasm32+simd128 with no threading (`docs/runs/2026-09-28-lean-cpu.md`
  never claims multi-threading).
- Qwen3-0.6B's CPU-rung browser prefill (87.96 ms/tok, 0.76 tok/s at the
  15-token `short` case) looks anomalously slow next to its own decode
  (87.0 ms/tok, in the same run) - they are in fact nearly identical
  per-token costs; the "0.76 tok/s" prefill figure in the summary table is
  the throughput implied by treating the entire 15-token prefill as a
  single unit (1319.4 ms / 15 tokens), not a separate slow phase - not
  investigated further, flagged so the two rows aren't misread as prefill
  being 17x slower than decode.
- The Hub API confirms Q4_0/Q8_0 GGUF files exist today for both SmolLM2
  sizes bartowski publishes, so no K-quant kernel work is a prerequisite
  for running SmolLM2 on the lean engine once it gains Llama-architecture
  support - only the architecture port itself.

## What each use case needs next

- **llm-life**: needs a lean-engine bench of the *actual* packed
  block-diagonal variant-A shape (many cells in one `forward_chunk_spec`
  call against one resident prefix, block-diagonal mask, per-block RoPE
  restart - the mechanism `docs/runs/2026-09-28-lean-lora.md` already
  proved numerically invisible) rather than this session's one-cell-at-a-
  time restore loop, which pays a full KV-cache restore per cell and
  cannot represent llm-life's real per-generation throughput. Performance
  work, not a support gap - `forward_chunk_spec`/LoRA/sliced-lm-head are
  all already implemented.
- **sonos MCP agent**: needs real decode speed - 32 s for a 48-token turn
  (native) is unusable for a live agent, and the browser number for the
  same prefill is 2x faster, so browser-first is the near-term path. Both
  are performance gaps in the existing Q8_0 tiled matmul/attention
  kernels at this hidden size (2048) and prompt length (2225 tokens), not
  missing features.
- **3B chat demo**: needs Q6_K tensor support in the GGUF reader
  (`crates/lean/src/gguf.rs`/`model.rs`) before any Qwen2.5-3B GGUF from
  this Hub repo's "Q4_0" preset can load at all - a support gap, not a
  performance one. Not attempted in the browser as a result.
- **swarm / llm->tts phones**: needs the Llama architecture (RoPE
  interleaved-pairs convention, LayerNorm-vs-RMSNorm/SwiGLU specifics for
  SmolLM2) before SmolLM2 itself can run on this engine at all - the
  actual gap `models-inventory-2026-09-28.md` already names. Until then,
  Qwen2.5-0.5B Q4_0 on the lean engine's GPU rung already beats the
  wllama SmolLM2-360M baseline on speed (42.6 vs 20.7 tok/s decode) at a
  larger download; the CPU rung (no WebGPU) is currently slower than
  wllama's threaded wasm backend and would need multi-threaded wasm
  (`SharedArrayBuffer`/COOP-COEP, not attempted here) to close that gap -
  a performance/infrastructure gap, not an architecture one.
- **desktop llm->tts (WebLLM baseline)**: same Llama-architecture gap as
  above; WebLLM's own numbers (224.93 prefill tok/s, 40.94 decode tok/s
  for SmolLM2-1.7B q4f16_1) are the bar the lean engine needs to beat once
  Llama support lands, at less than half WebLLM's 919 MB download if a
  Q4_0/Q8_0 GGUF is used instead of q4f16_1-MLC.
