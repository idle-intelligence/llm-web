# Headless harness

`run.mjs` verifies that a wasm LLM build (any page, any GGUF + tokenizer
URLs) loads and runs one inference in Playwright's bundled headless
Chromium — never the user's own browser — and can benchmark decode tok/s.
It's general: `llm-web` is about running LLMs on the web (Burn+wgpu,
WebGPU, GGUF); the MCP agent demo at `web/agent/` is one application of
that, and this harness drives it (or any compatible page) generically
rather than being wired to one model.

`repro.mjs` is a thin back-compat shim that forwards its argv to `run.mjs`
— any flag below also works through it.

## Prerequisites

- A model server exposing the GGUF + tokenizer + template over HTTP, e.g.
  `scripts/serve_models.py --dir <models-dir>` on port 8001.
- A demo page server with COOP/COEP headers (required for WebGPU +
  cross-origin Worker), e.g. `python3 web/agent/serve.py` on port 8002.
- The page must expose `window.__llm = { load, run, reset, state }` (see
  `web/agent/index.html`) — `run.mjs` drives the page through that surface,
  not through DOM clicks, so any page implementing the same contract works.
- Playwright's Chromium under `~/Library/Caches/ms-playwright` (this repo
  does not `npm install` Playwright — the module is borrowed via
  `PLAYWRIGHT_MODULE`, see below).

## `window.__llm` contract

```
window.__llm.load(modelSpec)              -> Promise<info>
  modelSpec = {id, shards: [url...], tokenizerUrl, templateUrl}

window.__llm.run(prompt, opts)             -> Promise<transcript>
  opts = {tools: 'none'|'demo', maxNewTokens, maxSteps}
  transcript = {steps: [{index, promptTokens, text, calls, prefillMs, decodeMs, tokens}...], finalText, totalMs}

window.__llm.reset()
window.__llm.state                         -> {loaded, info}
```

## Run

```
node scripts/headless/run.mjs
```

## Flags

All optional; defaults point at the xLAM-2-3b-fc-r demo on the local dev
servers.

| flag | default | meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:8002/` | page to load |
| `--gguf` | xLAM q4_0 on :8001 | GGUF shard URL |
| `--tokenizer` | xLAM tokenizer.json on :8001 | tokenizer URL |
| `--template` | xLAM tokenizer_config.json on :8001 | chat template URL |
| `--prompt` | `what is the weather in Paris?` | user turn |
| `--tools` | `demo` | `none` (plain chat, no tools passed) or `demo` (the two canned tools in `index.html`) |
| `--max-new` | `64` | max new tokens per step |
| `--max-steps` | `6` | max agent-loop steps |
| `--expect` | (none) | regex; exit 1 if the final text doesn't match |
| `--bench N` | (none) | after the run, do a second `tools=none` run with `--max-new N` and report decode tok/s |
| `--timeout-load` | `600000` (10 min) | ms to wait for `load()` |
| `--timeout-run` | `360000` (6 min) | ms to wait for `run()` |
| `--json <path>` | (none) | write a JSON report (chromium version, adapter info, load ms, steps, tokens, text, tok/s, gpu errors, pass/fail) |
| `--out <path>` | `scripts/headless/out/console.log` | full console log path |

`PLAYWRIGHT_MODULE` env var overrides where the `playwright` module is
resolved from (default: the install in `dusty-games-platform`'s
`node_modules`, since this repo has none of its own).

`scripts/headless/out/` is gitignored — JSON reports and console logs land
there by default.

## Examples

```bash
# Default demo run: tools=demo, expect "Paris" to show up in the final answer
node scripts/headless/run.mjs --expect "Paris" --json scripts/headless/out/demo.json

# Plain-chat bench: no tools, one short turn, then a 32-token decode bench
node scripts/headless/run.mjs --tools none --prompt "Write one sentence about the sea." \
  --max-new 32 --bench 32 --json scripts/headless/out/chat.json
```

## Behaviour

Launches only Playwright's bundled Chromium with `--enable-unsafe-webgpu
--enable-features=WebGPU --use-angle=metal --ignore-gpu-blocklist`, drives
the page through `window.__llm`, and stops at the first `[gpu-debug] ...
failed:` line (the root WebGPU validation error — everything after it is
cascade noise repeated per decode step). Prints `[llm]`/`[llm-worker]`
page/worker log lines and timings as it goes, writes the full console log
to `--out`, writes the JSON report to `--json` if given, and exits non-zero
on failure/timeout/gpu-debug-failure/`--expect` mismatch, zero on pass.

Bench numbers report the *decode* phase's tok/s from the transcript step's
`tokens`/`decodeMs`. Per docs/ENGINE.md's Browser section: the total
wall-clock time (`totalMs`) is trustworthy, but the prefill/decode ms
*split* currently measures submission, not GPU completion, until the
engine syncs after prefill — treat tok/s from this split as directional,
not precise, until that TODO is fixed Rust-side.

## Known gotchas

- `navigator.gpu` is `undefined` on `about:blank`. Always check WebGPU
  availability *after* `page.goto()` has navigated to a real http(s)
  origin — checking before that will falsely report WebGPU as unavailable.
- The page must actually expose `window.__llm` — an older page that only
  has the click-driven UI (no automation surface) won't work with this
  harness.
