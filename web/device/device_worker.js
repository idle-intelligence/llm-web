// Module Worker behind web/device/index.html. Speaks a small protocol:
//
//   page -> worker:  {type:'detect'}
//                     {type:'run'}
//   worker -> page:  {type:'status', text}
//                     {type:'detected', caps, backend, canRun1_7b}
//                     {type:'progress', loaded, total}
//                     {type:'done', result}
//                     {type:'error', message}
//
// 'detect' only runs capability checks: no model download, no engine start.
// 'run' does that plus the full download/load/generate/verify flow.
//
// Picks a backend by capability only, same policy as
// crates/lean/www/backends_worker.js and web/lean-chat-worker.js: WebGPU if
// an adapter is granted, else CPU threads if the page is cross-origin
// isolated with SharedArrayBuffer and more than one hardware thread, else
// single-thread CPU (WASM SIMD128). 'run' then runs one short generation on
// that backend and checks the output against the transformers reference
// hash (same fixture backends.html?diag=1 checks).
//
// This file has no top-level `import` or `await`: a module worker with a
// top-level await can drop messages posted before it resolves. self.onmessage
// is wired first and buffers into EARLY until the real handler replaces it.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-04-demos-02";
const N_GEN = 64;
const MAX_CTX = 256;

function status(text) {
  self.postMessage({ type: "status", text });
}

function backendFor(caps) {
  return caps.hasAdapter ? "webgpu" : caps.threadsCapable ? "threads" : "single";
}

async function detect() {
  const { capabilities } = await import(`./device_common.js?v=${ENGINE_BUILD}`);
  const caps = await capabilities();
  const backend = backendFor(caps);
  self.postMessage({ type: "detected", caps, backend, canRun1_7b: backend === "webgpu" });
}

async function run() {
  const { MODEL, fetchBytes, fetchText, argmaxJs, sha256Hex, capabilities, median } = await import(
    `./device_common.js?v=${ENGINE_BUILD}`
  );

  console.log(`[device_worker] engine build ${ENGINE_BUILD}`);
  status(`checking this device...`);
  const caps = await capabilities();
  const order = [
    caps.hasAdapter ? "webgpu" : null,
    !caps.hasAdapter && caps.threadsCapable ? "threads" : null,
    "single",
  ].filter(Boolean);

  status(`fetching ${MODEL.label} (cached after first load)...`);
  const [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(MODEL.gguf, (loaded, total) => self.postMessage({ type: "progress", loaded, total })),
    fetchText(MODEL.tokenizer),
    fetchText(MODEL.tokenizerCfg),
  ]);

  let engine = null;
  let gpu = false;
  let backend = null;
  const skipped = [];
  const timing = {};
  for (const c of order) {
    try {
      status(`starting ${c} backend...`);
      let t0 = performance.now();
      if (c === "webgpu") {
        const mod = await import(`../lean/pkg/lean.js?v=${ENGINE_BUILD}`);
        await mod.default(`../lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
        timing.wasmInitMs = performance.now() - t0;
        mod.leanInit();
        engine = await mod.LeanEngine.create();
        gpu = true;
      } else if (c === "threads") {
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
      status(`loading weights on ${c} backend...`);
      t0 = performance.now();
      engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, MAX_CTX);
      timing.loadMs = performance.now() - t0;
      backend = c;
      break;
    } catch (e) {
      skipped.push(`${c}: ${e && e.message ? e.message : e}`);
      engine = null;
    }
  }
  if (!engine) throw new Error(`no backend could start (${skipped.join("; ")})`);

  status(`running ${N_GEN} greedy tokens on the ${backend} backend...`);
  let t0 = performance.now();
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

  const canRun1_7b = backend === "webgpu";
  self.postMessage({
    type: "done",
    result: {
      backend,
      skipped,
      caps,
      engineBuild: ENGINE_BUILD,
      modelLabel: MODEL.label,
      promptLen: MODEL.promptIds.length,
      prefillMs,
      decodeMsPerTok,
      tokPerSec: 1000 / decodeMsPerTok,
      hash,
      matchesReference: hash === MODEL.referenceHash,
      text: engine.decodeIds(Uint32Array.from(ids)),
      canRun1_7b,
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
    },
  });
}

function dispatch(e) {
  const fn = e.data && e.data.type === "detect" ? detect : run;
  fn().catch((err) => self.postMessage({ type: "error", message: err && err.message ? err.message : String(err) }));
}

self.onmessage = dispatch;
for (const e of EARLY) dispatch(e);
