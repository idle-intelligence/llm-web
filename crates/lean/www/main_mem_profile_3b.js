// Qwen2.5-3B-Instruct counterpart of main_mem_profile.js: load-only (no
// prefill/decode - a 1.9GB GGUF's prefill/decode in the browser is slow and
// this file's only job is the load-time wasm/JS-heap/GPU memory snapshot,
// per this session's Task A). Same local/model URL convention as
// main_qwen25_3b.js (./model_qwen25_3b/).
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-02-backends-03";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/tokenizer_config.json";

const params = new URLSearchParams(location.search);
const local = params.get("local") === "1";
const ggufUrl = local ? "./model_qwen25_3b/qwen2.5-3b-instruct-q4_0.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model_qwen25_3b/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model_qwen25_3b/tokenizer_config.json" : HF_TOKENIZER_CFG;

const out = document.getElementById("out");
function log(line) {
  out.textContent += line + "\n";
  console.log(line);
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return await r.text();
}

// wasmExports.memory.buffer.byteLength = wasm linear memory, in bytes.
// performance.memory (Chrome-only, non-standard) = JS heap, in bytes.
function memSnapshot(label, wasmExports, engine, loaded) {
  const wasmBytes = wasmExports.memory.buffer.byteLength;
  const jsHeap = performance.memory
    ? { usedJSHeapSize: performance.memory.usedJSHeapSize, totalJSHeapSize: performance.memory.totalJSHeapSize, jsHeapSizeLimit: performance.memory.jsHeapSizeLimit }
    : null;
  const gpu = loaded ? JSON.parse(engine.gpuMemoryInfo()) : null;
  const snapshot = { label, wasmBytes, jsHeap, gpu };
  log(`[mem] ${label}: wasm=${(wasmBytes / 1e6).toFixed(1)}MB jsHeap=${jsHeap ? (jsHeap.usedJSHeapSize / 1e6).toFixed(1) + "MB" : "n/a"} ` +
    (gpu ? `gpu.weight=${(gpu.weightBytes / 1e6).toFixed(1)}MB gpu.kvCache=${(gpu.kvCacheBytes / 1e6).toFixed(1)}MB gpu.pool=${(gpu.poolBytes / 1e6).toFixed(1)}MB` : "gpu=n/a (not loaded yet)"));
  return snapshot;
}

async function main() {
  const wasmExports = await (await import(`../pkg/lean.js?v=${ENGINE_BUILD}`)).default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  const { LeanEngine, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean] engine build ${ENGINE_BUILD}, source=${local ? "local" : "huggingface"}`);

  const fetchStart = performance.now();
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const engine = await LeanEngine.create();
  const snapshots = [];
  snapshots.push(memSnapshot("after_device_create", wasmExports, engine, false));

  const loadStart = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 128);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  log(`[lean] model loaded in ${(performance.now() - loadStart).toFixed(0)}ms: ${engine.info()}`);
  snapshots.push(memSnapshot("after_load", wasmExports, engine, true));

  log("[mem] DONE");
  window.__leanResult = { snapshots, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { error: String(e) };
});
