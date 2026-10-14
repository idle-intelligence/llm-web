// Module Worker behind web/device/index.html. Speaks a small protocol:
//
//   page -> worker:  {type:'detect'}
//                     {type:'run', backend}   // backend: 'webgpu' | 'threads' | 'single' | 'all'
//   worker -> page:  {type:'status', text}
//                     {type:'detected', caps, available, reasons}
//                     {type:'progress', loaded, total}
//                     {type:'result', result}   // one per backend run
//                     {type:'runDone'}           // the requested run(s) are finished
//                     {type:'error', message}
//
// 'detect' only runs capability checks: no model download, no engine
// start. 'run' downloads the model once (cached after that) and then runs
// one short greedy generation per requested backend through the engine's
// public API (Lean.generate on the chat-template-rendered prompt), checked
// against the transformers reference text. 'run' with backend:'all' runs every
// available backend in turn, posting a 'result' after each.
//
// This file has no top-level `import` or `await`: a module worker with a
// top-level await can drop messages posted before it resolves. self.onmessage
// is wired first and buffers into EARLY until the real handler replaces it.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-14-quant-01";
// Rewritten to true by scripts/build.sh when BUILD_THREADS=1 built pkg-mt
// into this deployment; GitHub Pages never sets it.
const MT_BUILT = false;
const MAX_CTX = 256;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function detect() {
  const { capabilities, availableBackends } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const caps = await capabilities(MT_BUILT);
  const { available, reasons } = availableBackends(caps);
  self.postMessage({ type: "detected", caps, available, reasons });
}

// Model bytes + tokenizer, fetched once per worker lifetime and reused
// across however many backends get run (getModel's own Cache API entry
// also makes this free on a second page load).
let modelFiles = null;

async function ensureModel() {
  if (modelFiles) return modelFiles;
  const { MODEL, fetchBytes, fetchText } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  status(`fetching ${MODEL.label} (cached after first load)...`);
  const [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(MODEL.gguf, (loaded, total) => self.postMessage({ type: "progress", loaded, total })),
    fetchText(MODEL.tokenizer),
    fetchText(MODEL.tokenizerCfg),
  ]);
  modelFiles = { ggufBytes, tokenizerJson, tokenizerCfgJson, MODEL };
  return modelFiles;
}

async function runBackend(backend) {
  const { MODEL, median } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const { ggufBytes, tokenizerJson, tokenizerCfgJson } = await ensureModel();

  status(`starting ${backend} backend...`);
  const timing = {};
  const mod = backend === "threads" ? await loadThreads(timing) : await loadSingle(timing);

  status(`loading weights on ${backend} backend...`);
  let t0 = performance.now();
  const lean = await mod.Lean.load({ gguf: ggufBytes, tokenizer: tokenizerJson, maxCtx: MAX_CTX, backend: backend === "webgpu" ? "webgpu" : "cpu" });
  timing.loadMs = performance.now() - t0;

  const template = mod.ChatTemplate.fromTokenizerConfig(tokenizerCfgJson);
  const prompt = mod.chatRender(template, [{ role: "user", content: MODEL.prompt }], null, true);

  status(`running ${MODEL.referenceTokens} greedy tokens on the ${backend} backend...`);
  // One timestamp per piece of streamed text (a piece is one token, except
  // where a character spans several).
  const stamps = [];
  t0 = performance.now();
  const reply = await lean.generate(prompt, { maxNewTokens: MODEL.referenceTokens, temperature: 0 }, () => stamps.push(performance.now()));
  const prefillMs = stamps.length ? stamps[0] - t0 : null;
  const stepMs = stamps.slice(1).map((t, i) => t - stamps[i]);
  const nTokens = lean.tokenCount(reply);
  const decodeMsPerTok = stamps.length > 1 && nTokens > 1 ? (stamps[stamps.length - 1] - stamps[0]) / (nTokens - 1) : NaN;

  const result = {
    backend,
    engineBuild: ENGINE_BUILD,
    prefillMs,
    decodeMsPerTok,
    tokPerSec: 1000 / decodeMsPerTok,
    matchesReference: reply === MODEL.referenceText,
    text: reply,
    timing: {
      wasmInitMs: timing.wasmInitMs,
      threadPoolMs: timing.threadPoolMs,
      loadMs: timing.loadMs,
      prefillMs,
      decodeFirstMs: stepMs.length ? stepMs[0] : null,
      decodeMedianMs: stepMs.length ? median(stepMs) : null,
      decodeMinMs: stepMs.length ? Math.min(...stepMs) : null,
      decodeMaxMs: stepMs.length ? Math.max(...stepMs) : null,
    },
  };
  lean.free();
  self.postMessage({ type: "result", result });
}

// Each wasm build is initialised once per worker and reused: the CPU threads
// build's thread pool can only be built once, so a second run must not
// initialise it again.
let singlePromise = null;
let threadsPromise = null;
function loadSingle(timing) {
  if (!singlePromise) {
    singlePromise = (async () => {
      const t0 = performance.now();
      const mod = await import(`../lean/pkg/lean.js?v=${ENGINE_BUILD}`);
      await mod.default(`../lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
      timing.wasmInitMs = performance.now() - t0;
      return mod;
    })();
  }
  return singlePromise;
}
function loadThreads(timing) {
  if (!threadsPromise) {
    threadsPromise = (async () => {
      const { capabilities } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
      const caps = await capabilities(MT_BUILT);
      let t0 = performance.now();
      const mod = await import(`../lean/pkg-mt/lean.js?v=${ENGINE_BUILD}`);
      await mod.default(`../lean/pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
      timing.wasmInitMs = performance.now() - t0;
      t0 = performance.now();
      await mod.initThreadPool(caps.hardwareConcurrency);
      timing.threadPoolMs = performance.now() - t0;
      return mod;
    })();
  }
  return threadsPromise;
}

async function run(requested) {
  console.log(`[device_worker] engine build ${ENGINE_BUILD}, backend ${requested}`);
  const { capabilities, availableBackends } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const caps = await capabilities(MT_BUILT);
  const { available } = availableBackends(caps);

  const order = requested === "all" ? available : [requested];
  for (const b of order) {
    if (!available.includes(b)) continue;
    await runBackend(b);
  }
  self.postMessage({ type: "runDone" });
}

function dispatch(e) {
  const data = e.data;
  const fn = data && data.type === "detect" ? detect : () => run(data.backend);
  fn().catch((err) => self.postMessage({ type: "error", message: err && err.message ? err.message : String(err) }));
}

self.onmessage = dispatch;
for (const e of EARLY) dispatch(e);
