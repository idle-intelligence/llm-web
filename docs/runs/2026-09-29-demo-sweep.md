# Live-demo browser sweep, 2026-09-29

Headless-Chromium sweep of every published demo page - the trucs.ai
portfolio pages and the idle-intelligence *-web GitHub Pages demos - as a
visitor gets them, not a local build. Playwright's bundled headless
Chromium only (`chrome-headless-shell`), one browser instance at a time,
`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal`
plus `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream`
for any page that wants a microphone. No page was modified; every action
used a control the page itself exposes. Quiet-machine gate checked before
each sweep launch (1-minute load average 1.3-1.9 throughout, no other
Chromium instance running, no `rustc`/`cargo`/training process).

Method per page: navigate, let the page's own top-level script settle
(network-idle or a fixed settle delay - some pages attach their button
listeners only after an async top-level import resolves, so clicking too
early hits a live but unwired button), click the page's own "download"/
"load" control if present, wait for the page's own status line/button-
enabled state to report ready, then drive one representative action the
page itself offers (generate N tokens, transcribe a recording or its own
sample WAV, synthesize a sentence, run a forecast, one simulation step, or
a weather compute). Peak memory is the Chromium process-tree RSS (`ps`),
sampled every 0.5s across the whole page session, restricted to that
session's own new process IDs.

## Results

| page | loads clean (console/page errors) | time to ready | peak Chromium RSS | representative action | latency | backend reported |
|---|---|---:|---:|---|---:|---|
| trucs.ai/llm/ | yes, 0/0 | 16.8s | 0.47GB | generate ("Say hello...") | 15.1s (2 tok, 1.2 tok/s) | download UI only - no explicit backend label |
| trucs.ai/stt/ | yes, 0/0 | 16.1s | 3.26GB | transcribe (fake silent mic, 4s) | 42.1s | download UI only |
| trucs.ai/tts/ | yes, 0/0 | did not reach ready (generate button stayed disabled after status showed "ready") | 0.72GB | not completed | n/a | status: "ready" |
| trucs.ai/sts/ | yes, 0/0 | did not reach ready (record button stayed disabled) | 0.10-0.43GB | not completed | n/a | status: "model will be stored locally: ~3.8GB" |
| trucs.ai/t0/ | yes, 0/0 | 3.6s | 0.54GB | forecast (auto after load) | 3.0s | WebGPU |
| trucs.ai/astres/ | yes, 0/0 | 0.3s | 0.22GB | let simulation run 3s | 3.0s | "WebGPU required. Try Chrome 113+ or Safari 18+." (see Observations) |
| trucs.ai/classifier/ | yes, 0/0 | 3.2s | 0.39GB | classify ("This is a great day.") | 3.2s | not labeled on page |
| trucs.ai/swarm/ | yes, 0/0 | 0.3s | 0.09GB | skipped - static page, no in-browser model | n/a | n/a |
| trucs.ai/knn-weather/ | yes, 0/0 | 0.2s | 0.18GB | geolocate + compute | 5.0s | "fetched 5 observations from 1 source in 715ms, calculated in 3ms" |
| idle-intelligence llm-life/web/ | yes, 0/0 | 0.4s | 0.12GB | one simulation step | 1.5s | not labeled on page |
| idle-intelligence tts-web/web/ | yes, 0/0 | did not reach ready (generate button stayed disabled) | 0.72-0.74GB | not completed | n/a | not labeled on page |
| idle-intelligence stt-web/web/ | yes, 0/0 | did not reach ready ("Load Example WAV" button never enabled) | 3.26-3.51GB | not completed | n/a | not labeled on page |
| idle-intelligence sts-web/web/ | yes, 0/0 | did not reach ready in the completed run | 0.10GB | not completed | n/a | not labeled on page |
| idle-intelligence llm-web/web/ | yes, 0/0 | 9.2s | 0.78GB | generate ("Say hello...") | 15.0s (2 tok, 0.6 tok/s) | CPU |
| idle-intelligence weather-web/web/ | yes, 0/0 | did not reach ready (Compute button stayed hidden in this run) | 0.12GB | attempted (coords + fetch + compute) | 6.0s, result panel empty | not labeled on page; no model download observed - pure WASM interpolation |
| idle-intelligence t0-web/web/ | yes, 0/0 | 3.2s | 0.54GB | forecast | 5.0s | webgpu |

Every page loaded with zero console errors and zero page (uncaught JS)
errors in every run, including the pages whose action did not complete.

## Observations

- **Two pages reached "ready" (trucs.ai/llm/, trucs.ai/stt/) once the
  harness waited for the page's own top-level async import to finish
  before clicking "download".** An earlier attempt at both pages clicked
  immediately after the page's `load` event and got a 90-second timeout
  on every one of six pages (trucs.ai/tts/, trucs.ai/sts/, tts-web,
  stt-web, sts-web, and initially llm/ and stt/ too) with the button
  visibly present but never enabled - the module script's dynamic
  `import('@mlc-ai/web-llm')` (trucs.ai/llm/) hadn't resolved yet, so its
  `addEventListener('click', loadModel)` line hadn't run. Re-running with
  a network-idle/settle wait before the first click fixed `llm/` and
  `stt/` immediately (16-17s to ready, real generation/transcription
  attempted). This is a headless-automation timing artifact, not a claim
  about page correctness - it is called out because it means the four
  remaining "did not reach ready" rows (`tts/`, `sts/`, `tts-web`,
  `stt-web`, `sts-web`) got the same fix and *still* did not enable their
  action button, even though `tts/`'s own status line had already reached
  "ready" - worth a follow-up look at whether those specific pages need an
  extra UI step first (e.g. a voice selection) that this sweep's generic
  driver didn't perform, rather than assuming the fix didn't apply.
- **`trucs.ai/astres/` reports no WebGPU** ("WebGPU required. Try Chrome
  113+ or Safari 18+.") in the same headless Chromium session where
  `trucs.ai/t0/`, `idle-intelligence/t0-web/`, and `trucs.ai/llm/` all
  obtained a working WebGPU adapter (`navigator.gpu.requestAdapter()`
  succeeded in a separate check on `llm/`). This machine's headless
  Chromium is known to lack a GPU adapter in general (native/browser
  timing notes on this box), so a page-to-page difference in whether
  WebGPU reports available is itself the finding worth flagging - not
  resolved here, since t0/t0-web's own "WebGPU" backend label and
  successful low-latency forecast (3.0-5.0s) argue the adapter is real for
  at least some pages in this session.
- **Memory is the widest spread in the sweep.** `trucs.ai/stt/` and
  `idle-intelligence/stt-web/` both push the Chromium process tree to
  3.2-3.5GB even though their action never produced a transcript (fake
  silent microphone audio) - loading the STT model's weights accounts for
  this, independent of whether decoding produced useful output. Every
  other page stayed under 1GB.
- **`idle-intelligence/llm-web/web/` reports `CPU`, not `WebGPU`**, and is
  markedly slower per token than `trucs.ai/llm/` (0.6 tok/s vs 1.2 tok/s on
  a 2-token sample - too small a sample to generalize, but consistent with
  a CPU fallback rung). This is an engine we (this project) own; if the
  capability check is landing on CPU where `trucs.ai/llm/`'s WebGPU path
  is available in the same browser session, that gap is worth checking
  first, since both pages ran in the identical headless Chromium instance
  moments apart.
- **`idle-intelligence/weather-web/web/`'s representative action did not
  produce a result** (`resultPanel` empty) even though the page loaded
  cleanly with zero errors - the coordinate-entry -> fetch -> compute
  sequence this sweep drove may need a longer settle between steps or a
  different trigger than a synthetic `change` event; flagged as
  inconclusive rather than broken, since the page itself never logged an
  error.
- **`trucs.ai/swarm/` does not load a model in the browser at all** - it
  is a static description/"how it works (wip)" page with no wasm/model
  fetch in its network log, so no action or memory measurement applies.
- Pages that already run on an engine this project owns and could
  fix directly: `trucs.ai/llm/` and `idle-intelligence/llm-web/web/` (the
  CPU-vs-WebGPU gap above), and `idle-intelligence/t0-web/web/` /
  `trucs.ai/t0/` (both healthy, WebGPU, fast - no action needed). The
  pages whose action did not complete in this sweep (`tts/`, `sts/`,
  `tts-web`, `stt-web`, `sts-web`, `weather-web`) need a follow-up pass
  with a page-specific driver (not this sweep's generic
  download-then-click) before concluding anything is actually broken for
  a real visitor.
