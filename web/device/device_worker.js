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
// one short generation per requested backend, verified against the
// transformers reference hash (same fixture crates/lean/www's
// backends.html?diag=1 checks). 'run' with backend:'all' runs every
// available backend in turn, posting a 'result' after each.
//
// This file has no top-level `import` or `await`: a module worker with a
// top-level await can drop messages posted before it resolves. self.onmessage
// is wired first and buffers into EARLY until the real handler replaces it.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-04-demos-04";
const N_GEN = 64;
const MAX_CTX = 256;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function detect() {
  const { capabilities, availableBackends } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const caps = await capabilities();
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
  const { MODEL, argmaxJs, sha256Hex, median } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const { ggufBytes, tokenizerJson, tokenizerCfgJson } = await ensureModel();

  status(`starting ${backend} backend...`);
  const timing = {};
  let t0 = performance.now();
  let engine, gpu;
  if (backend === "webgpu") {
    const mod = await import(`../lean/pkg/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    timing.wasmInitMs = performance.now() - t0;
    mod.leanInit();
    engine = await mod.LeanEngine.create();
    gpu = true;
  } else if (backend === "threads") {
    const { capabilities } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
    const caps = await capabilities();
    const mod = await import(`../lean/pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../lean/pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
    timing.wasmInitMs = performance.now() - t0;
    t0 = performance.now();
    await mod.initThreadPool(caps.hardwareConcurrency);
    timing.threadPoolMs = performance.now() - t0;
    mod.leanInit();
    engine = mod.LeanEngineCpu.create();
    gpu = false;
  } else {
    const mod = await import(`../lean/pkg/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    timing.wasmInitMs = performance.now() - t0;
    mod.leanInit();
    engine = mod.LeanEngineCpu.create();
    gpu = false;
  }

  status(`loading weights on ${backend} backend...`);
  t0 = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, MAX_CTX);
  timing.loadMs = performance.now() - t0;

  status(`running ${N_GEN} greedy tokens on the ${backend} backend...`);
  t0 = performance.now();
  const logits = gpu ? await engine.prefillTokens(MODEL.promptIds, []) : engine.prefillTokens(MODEL.promptIds);
  const prefillMs = performance.now() - t0;
  const ids = [argmaxJs(logits)];
  const stepMs = [];
  for (let i = 1; i < N_GEN; i++) {
    const ts = performance.now();
    const id = gpu ? await engine.decodeStepArgmax(ids[i - 1], []) : engine.decodeStepArgmax(ids[i - 1]);
    stepMs.push(performance.now() - ts);
    ids.push(id);
  }
  const decodeMsPerTok = stepMs.reduce((a, b) => a + b, 0) / stepMs.length;
  const hash = await sha256Hex(ids);

  const result = {
    backend,
    engineBuild: ENGINE_BUILD,
    promptLen: MODEL.promptIds.length,
    prefillMs,
    decodeMsPerTok,
    tokPerSec: 1000 / decodeMsPerTok,
    hash,
    matchesReference: hash === MODEL.referenceHash,
    text: engine.decodeIds(Uint32Array.from(ids)),
    timing: {
      wasmInitMs: timing.wasmInitMs,
      threadPoolMs: timing.threadPoolMs,
      loadMs: timing.loadMs,
      prefillMs,
      decodeFirstMs: stepMs[0],
      decodeMedianMs: median(stepMs),
      decodeMinMs: Math.min(...stepMs),
      decodeMaxMs: Math.max(...stepMs),
    },
  };
  self.postMessage({ type: "result", result });
}

async function run(requested) {
  console.log(`[device_worker] engine build ${ENGINE_BUILD}, backend ${requested}`);
  const { capabilities, availableBackends } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const caps = await capabilities();
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
