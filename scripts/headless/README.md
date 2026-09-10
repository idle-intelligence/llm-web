# Headless repro

`repro.mjs` verifies that the wasm engine loads and completes one inference
in Playwright's bundled headless Chromium, before anyone tests the demo in a
real browser — never the user's own browser.

## Prerequisites

- Model server running: `scripts/serve_models.py` on port 8001.
- Demo server running: `web/agent/serve.py` on port 8002 (COOP/COEP).
- Playwright's Chromium browsers installed under
  `~/Library/Caches/ms-playwright` (this repo does not `npm install`
  Playwright — the module is borrowed via an env var, see below).

## Run

```
node scripts/headless/repro.mjs
```

Optional env vars:

- `DEMO_URL` — page to load (default `http://127.0.0.1:8002/`).
- `PLAYWRIGHT_MODULE` — path/specifier to resolve the `playwright` module
  from (default points at the `playwright` install in
  `dusty-games-platform`'s `node_modules`, since this repo has none of its
  own). Point `NODE_PATH` or this var at a different install if needed.

The script launches only Playwright's bundled Chromium with
`--enable-unsafe-webgpu --enable-features=WebGPU --use-angle=metal
--ignore-gpu-blocklist`, drives the demo page through one load + inference,
and stops at the first `[gpu-debug] ... failed:` line (the root WebGPU
validation error — everything after it is cascade noise repeated per decode
step). It prints the `[llm]` page log and timings as it goes, writes the
full console log to `scripts/headless/out/console.log`, and exits non-zero
on failure/timeout/gpu-debug-failure, zero when the page reports `done`.

## Known gotcha

`navigator.gpu` is `undefined` on `about:blank`. Always check WebGPU
availability *after* `page.goto()` has navigated to a real http(s) origin —
checking before that will falsely report WebGPU as unavailable.
