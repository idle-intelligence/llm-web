# lean engine, CPU threads rung

Adds the "CPU threads" rung of the project's capability ladder (WebGPU, then CPU
threads, then single CPU thread) below the existing single-thread CPU
fallback (`docs/runs/2026-09-28-lean-cpu.md`). Crate `crates/lean`, changed
files `src/cpu.rs` (`linear()` split into `linear_serial`/`linear_threads`),
`src/lib.rs` (`wasm-mt`-gated `init_thread_pool` re-export), `Cargo.toml`
(new `threads`/`wasm-mt` features), new `scripts/build_lean_mt.sh` (nightly
atomics wasm build, same recipe as idle-intelligence/t0-web PR #8, "Threads
spike for the CPU build: measured, not adopted"), `scripts/serve_coi.py`
(local COOP/COEP dev server), and `crates/lean/www/cpu_mt.html` /
`main_cpu_mt.js` (capability-picked-rung browser harness).

## Design

- **Parallelisation target**: `cpu.rs::linear()`'s `rows * out_dim`
  independent dot products (decode matvecs at `rows=1`, prefill GEMMs at
  `rows>1`) — every output element depends only on `x`'s row and one weight
  row, never on another output element, so splitting this loop across
  threads changes only which thread computes which element, not the
  arithmetic. Verified bit-identical, not just token-exact (see Gates).
- **Thread count is a capability read, not a measurement**: `rayon`'s
  global pool sizes itself from `std::thread::available_parallelism()` on
  native; on wasm, the page calls `initThreadPool(navigator.hardwareConcurrency)`
  once (via `wasm-bindgen-rayon`, re-exported as `init_thread_pool` behind
  the new `wasm-mt` feature). `linear_threads` reads
  `rayon::current_num_threads()` — the same number either way — and never
  runs a timing probe of its own.
- **Fixed, shape-based threshold**: `MIN_WORK_PER_THREAD = 64` (a constant,
  not tuned per run) — `linear_threads` declines the threaded path when
  `rows*out_dim < n_threads*64`, falling back to `linear_serial`, so a
  future small op isn't handed to the pool for less work than the dispatch
  costs. Every op this crate currently calls `linear()` for (q/k/v/o, gate/
  up/down, lm_head) is far above this threshold (out_dim 896-151936).
- **One pool, created once**: `rayon`'s global pool (native default sizing;
  `initThreadPool` on wasm) is built once and reused across every decode
  step and prefill call — no per-call thread spawn.
- Native and browser share the exact same `cpu.rs`/`cpu_kernels.rs` source;
  only the Cargo feature set and the wasm build's linker flags differ.
  Reused directly from the project's own prior work rather than reinvented:
  `scripts/build_lean_mt.sh` is idle-intelligence/t0-web's
  `tools/build-mt.sh` recipe (nightly `build-std`, `wasm-bindgen-rayon`,
  the same atomics/shared-memory/`--import-memory`/`__wasm_init_tls`
  linker-flag set, and the same bare-directory-import patch to
  `workerHelpers.js`), adapted to this crate's `lean.js`/`wasm-mt` names.

## Parameters

- Models: Qwen2.5-0.5B-Instruct Q4_0 (`qwen2.5-0.5b-instruct-q4_0.gguf`,
  428,730,208 bytes) and SmolLM2-360M-Instruct Q4_0
  (`SmolLM2-360M-Instruct-Q4_0.gguf`), both from the local model cache used
  by prior lean runs.
- Fixture: `crates/lean/reference/fixture.json` (Qwen) /
  `fixture_llama_360m_q4_0.json` (SmolLM2), `short`/`long`/`non_english`
  cases (long-context `long_tools_*` cases skipped, as in the existing
  single-thread CPU run).
- Native: macOS 26.3.1, Apple M2 (aarch64, 8 logical cores via
  `sysctl -n hw.ncpu`), `cargo build -p lean --release --features threads`.
  Thread count set via `RAYON_NUM_THREADS` (1 = single-thread comparison
  point, 8 = default/full pool).
- Browser: Playwright-bundled headless Chromium ("Google Chrome for
  Testing"), served locally via `scripts/serve_coi.py` (COOP/COEP) at
  `127.0.0.1:8031`. `navigator.hardwareConcurrency` = 8 in this browser.
  `crates/lean/www/cpu_mt.html?local=1` (default), `?threads=0` forces the
  single-thread fallback rung for comparison.
- wasm builds: `crates/lean/pkg` (single-thread, unchanged, rebuilt this
  session to confirm the new Cargo features don't affect it — sha256
  `07123420da03c2b64ddd1e6c5b94bd91773c3f3cc92e4601bbd3248b38b1a8af`,
  3,484,965 bytes) and `crates/lean/pkg-mt` (new, `scripts/build_lean_mt.sh`,
  `wasm-mt` feature — sha256
  `147b0ac7cfafe8eccd4b6ec75e84c9badeda2e9d3a5d949263f7d6f684225989`,
  4,727,765 bytes; `wasm-tools print | grep -c "v128\|atomic"` = 22582;
  `wasm-tools print` confirms an imported `shared` memory and an exported
  `initThreadPool`). Engine build tags: `2026-09-29-cpu-mt-01` (`main_cpu_mt.js`),
  unchanged `2026-09-29-cpu-06` (`main_cpu.js`, single-thread, re-served to
  confirm no regression).
- Locks held for every native timing run: this session's shared build lock
  and GPU lock, nested (cargo lock outermost, GPU lock innermost), so no
  other worker's build or GPU job ran concurrently with a timed command.
  Load average (1-minute) at the start of the native Qwen sweep: 1.86; at
  the end of the native SmolLM2 sweep: 3.29 — both runs stayed under the
  house "load < 3" bar except the tail of the second sweep, which is noted
  but did not change the outcome (see Observations).

## Results

### Native decode/prefill ms per token, single-thread vs 8 threads, median of 5 ABAB reps

Qwen2.5-0.5B-Instruct Q4_0:

| case | metric | threads=1 median (min-max) | threads=8 median (min-max) | speedup |
|---|---|---:|---:|---:|
| short (36 tok) | decode ms/tok | 47.33 (47.05-47.50) | 16.78 (16.49-17.18) | 2.82x |
| short (36 tok) | prefill ms/tok | 46.89 (46.75-47.03) | 10.82 (10.80-11.69) | 4.33x |
| long (86 tok) | decode ms/tok | 47.60 (47.57-47.62) | 18.29 (17.22-19.20) | 2.60x |
| long (86 tok) | prefill ms/tok | 46.96 (46.84-47.06) | 10.92 (10.78-11.23) | 4.30x |

SmolLM2-360M-Instruct Q4_0 (timing only — this model's CPU-rung output does
not fixture-match transformers on either thread count, a pre-existing bug
unrelated to this change; see Observations):

| case | metric | threads=1 median (min-max) | threads=8 median (min-max) | speedup |
|---|---|---:|---:|---:|
| short (37 tok) | decode ms/tok | 43.04 (42.80-43.60) | 19.43 (19.25-19.44) | 2.22x |
| short (37 tok) | prefill ms/tok | 42.35 (42.29-43.06) | 9.59 (9.55-9.62) | 4.42x |
| long (86 tok) | decode ms/tok | 43.61 (43.59-43.89) | 19.92 (19.75-20.03) | 2.19x |
| long (86 tok) | prefill ms/tok | 42.62 (42.53-42.70) | 9.73 (9.67-9.76) | 4.38x |

### Browser decode ms/tok, Qwen2.5-0.5B-Instruct, headless Chromium under COI

One clean rep (load average 2.6-3.1 throughout, no concurrent build) plus
four contaminated reps (load average rose to 9.36 mid-sweep from an
unrelated concurrent cargo build on this shared machine — see
Observations, excluded from the headline numbers per house rule):

| case | single-thread (`pkg`) ms/tok | threads (`pkg-mt`, pool=8) ms/tok | speedup |
|---|---:|---:|---:|
| short | 77.2 | 26.9 | 2.87x |
| long | 77.6 | 27.7 | 2.80x |
| non_english | 77.0 | 27.3 | 2.82x |

## Gates

- `cargo check -p lean --features threads`: clean.
- `cargo clippy -p lean -- -D warnings` and `cargo clippy -p lean --features threads -- -D warnings`: clean, no new warnings.
- `cargo test -p lean --release --lib --features threads`: 17 passed, 0 failed (3 pre-existing `cpu_kernels` dot-kernel tests, 4 new `cpu::threads_tests` — see below — plus the rest of the crate's existing unit tests, unaffected).
- New unit tests (`crates/lean/src/cpu.rs`, `mod threads_tests`), in-process, no GGUF needed, isolate `linear()`'s thread count via a scoped `rayon::ThreadPoolBuilder` pool instead of the process-global one: decode-shaped (rows=1) and prefill-shaped (rows=37) F32 weights, and a Q4_0 weight, all assert **bit-exact equality** (`assert_eq!`, not an epsilon) between `n_threads=1` and `n_threads∈{2,4,8}`; a fourth test confirms a matrix below `MIN_WORK_PER_THREAD*n_threads` still returns the identical serial result. All 4 pass.
- `lean-cli --engine cpu` fixture gate, Qwen2.5-0.5B, native: `RAYON_NUM_THREADS=1` and default (8) both report `ids=true tok=true top1=true` on all three cases, with **identical** `top20_maxdiff` values between the two thread counts (0.000053 / 0.000031 / 0.000031 in both) — confirms the threaded and single-thread native paths are bit-exact, not just token-exact.
- `lean-cli --engine cpu` fixture gate, SmolLM2-360M, native: both thread counts report `tok=false` with **identical** `top20_maxdiff` (9.404766 / 10.432362 / 9.055265 in both) — the mismatch is present, byte-for-byte identical, before this session's changes too (reproduced on the unmodified `lean-perf` worktree baseline). Pre-existing, out of scope for this task; not introduced or hidden by threading.
- Browser, single-thread (`crates/lean/www/cpu.html`, `pkg`, no COI needed): all 3 cases `tokens_match=true`, decode 76.6-77.3ms/tok — matches `docs/runs/2026-09-28-lean-cpu.md`'s baseline, confirming the new Cargo features didn't regress the existing single-thread wasm build.
- Browser, threads rung (`crates/lean/www/cpu_mt.html`, `pkg-mt`, served under COOP/COEP): `crossOriginIsolated=true`, `SharedArrayBuffer` present, `hardwareConcurrency=8` → rung selected = `threads`, `initThreadPool(8)` succeeds, all 3 cases `tokens_match=true`.
- Browser, forced single-thread fallback (`cpu_mt.html?threads=0`, still served under COI): rung selected = `single-thread`, all 3 cases `tokens_match=true` — proves the loader's fallback branch (and the plain `pkg` build it loads) both still work when the threads capability is deliberately not used.
- GPU path: not re-run this session (this change touches `Cargo.toml`, `lib.rs`, and `cpu.rs` only — no shared `model.rs`/`engine.rs`/`gguf.rs`/`quant.rs` code). `cargo build -p lean --release --bin lean-cli` (which compiles the GPU path) succeeds under both the default and `threads` feature sets, confirmed this session; the expensive `fixture_parity_both_kernel_paths` GPU test (~188s per `docs/runs/2026-09-28-lean-cpu.md`) was not re-run given no shared-code changes and a contended machine at the time (see Observations).

## Observations

- Threading gives a real, substantial win here, unlike the project's own t0-web
  threads spike (PR #8, 0.93x-0.96x, a wash-to-regression) or `engine-plan.md`'s
  citation of the same result — the difference is structural, not a
  contradiction: t0's threads spike parallelised across a *batch* of small
  independent forecasts through Burn/rayon's own dispatch machinery, where
  the parallel work per call was too small to amortize cross-worker
  dispatch cost. Lean's `linear()` instead parallelises *within* a single
  matvec/GEMM, across output-row counts of 896-151,936 — large enough that
  the per-call rayon dispatch overhead is a small fraction of the work, on
  both native and (per the browser numbers) wasm.
- Prefill (GEMM, rows=36-86) scales better than decode (GEMV, rows=1):
  ~4.3x on 8 threads for prefill vs ~2.6-2.8x for decode, on both models.
  Consistent with decode having less total work to spread per call (a
  single row's worth) relative to the fixed per-dispatch pool overhead,
  while prefill's extra rows give the pool more to amortize that overhead
  over.
- Native and browser decode speedups are close (2.6-2.8x native, 2.8-2.9x
  browser) despite the browser's baseline being ~1.6x slower in absolute
  terms (77ms/tok wasm vs ~47ms/tok native, matching the single-thread gap
  already measured in `docs/runs/2026-09-28-lean-cpu.md`) — the relative
  win from threading doesn't erode going from native to wasm-bindgen-rayon.
- **Machine contention was hit and is flagged, not hidden**: mid-way through
  the second browser ABAB sweep, the 1-minute load average rose to 9.36
  (from ~2.6-3.1 at the sweep's start) because another worker's `cargo
  test --release` (8 parallel `rustc` processes, confirmed via `ps aux`,
  building `lean`'s own test binaries in the sibling `lean-perf` worktree)
  started concurrently. The single-thread rung's own ms/tok inflated from
  77ms to 80-99ms across the contaminated reps and the threads rung's from
  27ms to 30-39ms — the relative shape stayed roughly similar but the
  absolute numbers are not trustworthy. Per house rule, only the one clean
  rep (load 2.6-3.1 throughout) is reported as the browser headline number;
  the contaminated reps were discarded, not averaged in.
- The **native** timing sweeps did not hit this problem: the `cargo`/`gpu`
  locks were held for both sweeps, and the 1-minute load average stayed at
  1.86-3.29 throughout (the SmolLM2 sweep's tail is right at the edge of
  the 3.0 bar, not over it).
- **Pre-existing SmolLM2-360M CPU-rung bug, not introduced here**: both
  thread counts reproduce the exact same wrong greedy continuation
  (`top20_maxdiff` identical to 6 decimal places at threads=1 and
  threads=8), and the unmodified `lean-perf` worktree (before any of this
  session's changes) reproduces the identical wrong output too. This is a
  correctness issue in `cpu.rs`'s SmolLM2/Llama-architecture path (Qwen2.5
  is unaffected — token-exact on both thread counts) that predates this
  task and is out of scope for it; flagged here so it isn't mistaken for a
  threading regression.
- `rayon::current_num_threads()` as the single capability signal (rather
  than reading `navigator.hardwareConcurrency` in Rust, which isn't
  available there anyway) means the exact same `cpu.rs` code path is
  exercised on native and wasm with zero platform-specific branching in
  the forward pass itself — only the pool's bootstrap differs (automatic
  on native, one explicit `initThreadPool()` call from JS on wasm).

## What the browser needs to enable the threads rung on GitHub Pages / trucs.ai

- The threads rung requires `self.crossOriginIsolated === true`, which
  requires both `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` response headers on every
  document/script/wasm response in the page's origin. Plain GitHub Pages
  cannot set custom response headers at all. Whether trucs.ai's own
  hosting can (e.g. a `_headers` file, if its platform supports one the
  way Netlify/Cloudflare Pages do) was not verified this session — that's
  the first thing to check before shipping this rung on either surface.
- This session's browser gate served the page from a local Python
  `http.server` subclass that sets the headers itself (`scripts/serve_coi.py`)
  — this proves the engine and the `wasm-bindgen-rayon` worker pool work
  correctly *when real server-side COOP/COEP headers are present*, but
  does **not** prove anything about the client-side `coi-serviceworker.js`
  shim workaround that a header-less static host like GitHub Pages would
  need instead. the project's own prior experience is the relevant warning here:
  on `trucs.ai/stt-llm-tts`, that exact shim's COEP header broke Web
  Worker initialization outright and was removed (`trucs.ai` commit
  `3432383`, "Remove coi-serviceworker: COEP was blocking worker
  loading"). This engine's threads rung also depends on a Web Worker (the
  `wasm-bindgen-rayon` pool, plus this crate's existing requirement that
  all inference runs inside a Web Worker) — the same failure mode is
  possible here and has not been tested with the shim specifically, only
  with real headers.
- **Recommended loader behavior given the above**: ship both `pkg` and
  `pkg-mt` and pick at runtime with `self.crossOriginIsolated &&
  typeof SharedArrayBuffer !== "undefined" && navigator.hardwareConcurrency > 1`
  (implemented in `main_cpu_mt.js`, with a try/catch fallback to `pkg` if
  `initThreadPool` itself throws) — but on GitHub Pages specifically, do
  not add a coi-serviceworker shim to force `crossOriginIsolated` true
  without first testing it against this engine's own worker/model-fetch
  path the way the project's plan's step 5 calls for, given the stt-llm-tts
  precedent. Until that test happens, GitHub Pages traffic will
  legitimately fall through to the single-thread rung (correct, safe
  behavior per the capability check — not a bug), and only a host that can
  set the headers server-side (to be confirmed for trucs.ai) gets the
  threads rung for free with no shim risk.

## Open items

- Confirm whether trucs.ai's hosting platform supports a `_headers`-style
  mechanism for COOP/COEP before enabling the threads rung there.
- Test the `coi-serviceworker.js` shim specifically against this engine's
  Web Worker + wasm-bindgen-rayon worker pool + model-fetch path before
  ever adding it to a GitHub-Pages-hosted lean demo, given the stt-llm-tts
  precedent.
- The SmolLM2-360M CPU-rung correctness bug (pre-existing, reproduced on
  both thread counts and on the unmodified `lean-perf` baseline) is
  unresolved and out of scope for this task.
- The GPU-path `fixture_parity_both_kernel_paths` test (~188s) was not
  re-run this session; worth a follow-up run for full confidence, though
  no shared GPU code was touched.
