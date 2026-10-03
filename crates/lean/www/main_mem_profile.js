// Single-page-load memory profiler for Qwen2.5-0.5B-Instruct: loads the
// model once, runs the fixture's `short` case then its `long_tools_single`
// case (2225 prompt tokens) back to back against the SAME LeanEngine/KV
// cache, logging a memory snapshot after each stage. Not a fixture-parity
// or timing harness (see main.js for that) - this file exists to separate
// "more GPU/wasm memory at long context" from "more decode time at long
// context", per this session's memory investigation
// (docs/runs/2026-09-28-lean-decode-breakdown.md).
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-04-night-03";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json";

const params = new URLSearchParams(location.search);
const local = params.get("local") === "1";
const ggufUrl = local ? "./model/qwen2.5-0.5b-instruct-q4_0.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model/tokenizer_config.json" : HF_TOKENIZER_CFG;

const DECODE_STEPS = 8;

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
async function fetchJson(url) {
  return JSON.parse(await fetchText(url));
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

  const fixture = await fetchJson(`../reference/fixture.json?v=${ENGINE_BUILD}`);
  const shortCase = fixture.cases.find((c) => c.name === "short");
  const longCase = fixture.cases.find((c) => c.name === "long_tools_single");

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

  const maxCtx = longCase.input_ids.length + DECODE_STEPS + 64;
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, maxCtx);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  log(`[lean] model loaded: ${engine.info()}`);
  snapshots.push(memSnapshot("after_load", wasmExports, engine, true));

  // Short case: prefill + a few decode steps.
  {
    const logits0 = await engine.prefillTokens(shortCase.input_ids, []);
    let cur = (() => {
      let best = 0;
      for (let i = 1; i < logits0.length; i++) if (logits0[i] > logits0[best]) best = i;
      return best;
    })();
    for (let i = 0; i < DECODE_STEPS; i++) cur = await engine.decodeStepArgmax(cur, []);
  }
  snapshots.push(memSnapshot("after_short_prefill_decode", wasmExports, engine, true));

  // Long case: fresh KV cache (prefillTokens resets kv_len to 0), same
  // engine/model/pool instance - this is what a real long-context turn in
  // the SAME page session looks like.
  {
    const logits0 = await engine.prefillTokens(longCase.input_ids, []);
    let cur = (() => {
      let best = 0;
      for (let i = 1; i < logits0.length; i++) if (logits0[i] > logits0[best]) best = i;
      return best;
    })();
    for (let i = 0; i < DECODE_STEPS; i++) cur = await engine.decodeStepArgmax(cur, []);
  }
  snapshots.push(memSnapshot("after_long_prefill_decode", wasmExports, engine, true));

  log("[mem] DONE");
  window.__leanResult = { snapshots, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { error: String(e) };
});
