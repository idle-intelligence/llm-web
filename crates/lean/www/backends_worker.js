// Module Worker behind backends.html. Picks the backend by capability only (never
// by a benchmark): auto = WebGPU if an adapter is granted, else CPU threads
// if the page is cross-origin isolated with SharedArrayBuffer and more than
// one hardware thread, else single-thread CPU (WASM SIMD128). A forced backend
// that is unavailable is an error, not a silent fallback.
//
// Backend loading copies main_cpu_mt.js (pkg-mt + initThreadPool for threads,
// pkg for single thread); the prefill + greedy decode loop copies
// main_cpu.js / main.js.
const ENGINE_BUILD = "2026-10-02-backends-01";
const N_GEN = 64;
const MAX_CTX = 256;

// crates/lean/reference/fixture.json, case "short" ("What is the capital of
// France?", chat-templated), so every backend sees the same ids without
// depending on the tokenizer.
const PROMPT_IDS = [
  151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13,
  151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
];
// SHA-256 of the 64 greedy ids (u32 little-endian) from transformers
// (float32, weights dequantized from the same Q4_0 GGUF, greedy, no EOS stop).
const REFERENCE_HASH = "a454748c60e238419186f89709a2b6eee17bcf53b4253a575dfa32691dad04d5";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json";

function status(text) {
  self.postMessage({ type: "status", text });
}

async function fetchBytes(url) {
  const cache = await caches.open("lean-backends-model-v1");
  let r = await cache.match(url);
  if (!r) {
    r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
    await cache.put(url, r.clone());
  }
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return await r.text();
}

function argmaxJs(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

async function sha256Hex(ids) {
  const buf = new ArrayBuffer(ids.length * 4);
  const view = new DataView(buf);
  ids.forEach((id, i) => view.setUint32(i * 4, id, true));
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function capabilities() {
  const caps = {
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    adapter: "none",
    hasAdapter: false,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        caps.hasAdapter = true;
        const i = adapter.info || {};
        const l = adapter.limits;
        caps.adapter =
          [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" / ") +
          ` (storage buffers/stage ${l.maxStorageBuffersPerShaderStage}, max binding ${l.maxStorageBufferBindingSize}, shader-f16 ${adapter.features.has("shader-f16")})`;
      } else {
        caps.adapter = "navigator.gpu present, no adapter";
      }
    } catch (e) {
      caps.adapter = `requestAdapter failed: ${e && e.message ? e.message : e}`;
    }
  } else {
    caps.adapter = "no navigator.gpu";
  }
  caps.threadsCapable = caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

async function createEngine(backend, caps) {
  if (backend === "webgpu") {
    if (!caps.hasAdapter) throw new Error(`no WebGPU adapter (${caps.adapter})`);
    const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    mod.leanInit();
    return { engine: await mod.LeanEngine.create(), gpu: true };
  }
  if (backend === "threads") {
    if (!caps.threadsCapable) {
      throw new Error(
        `threads need crossOriginIsolated, SharedArrayBuffer and >1 hardware thread ` +
          `(got ${caps.crossOriginIsolated}, ${caps.sharedArrayBuffer}, ${caps.hardwareConcurrency})`
      );
    }
    const mod = await import(`../pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
    await mod.initThreadPool(caps.hardwareConcurrency);
    mod.leanInit();
    return { engine: mod.LeanEngineCpu.create(), gpu: false };
  }
  const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  mod.leanInit();
  return { engine: mod.LeanEngineCpu.create(), gpu: false };
}

async function run({ backend: requested, local }) {
  status(`engine build ${ENGINE_BUILD}, checking capabilities...`);
  const caps = await capabilities();
  const candidates =
    requested === "auto"
      ? [caps.hasAdapter && "webgpu", caps.threadsCapable && "threads", "single"].filter(Boolean)
      : [requested];

  status(`fetching model (${local ? "local" : "huggingface"})...`);
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(local ? "./model/qwen2.5-0.5b-instruct-q4_0.gguf" : HF_GGUF),
    fetchText(local ? "./model/tokenizer.json" : HF_TOKENIZER),
    fetchText(local ? "./model/tokenizer_config.json" : HF_TOKENIZER_CFG),
  ]);

  const skipped = [];
  let engine, gpu, backend;
  for (const c of candidates) {
    try {
      status(`starting ${c} backend...`);
      ({ engine, gpu } = await createEngine(c, caps));
      status(`loading weights on ${c} backend...`);
      engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, MAX_CTX);
      backend = c;
      break;
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      if (requested !== "auto") throw new Error(`${c} backend unavailable: ${msg}`);
      skipped.push(`${c}: ${msg}`);
      engine = null;
    }
  }
  ggufBytes = null;

  status(`running ${N_GEN} greedy tokens on ${backend} backend...`);
  const t0 = performance.now();
  const logits = gpu ? await engine.prefillTokens(PROMPT_IDS, []) : engine.prefillTokens(PROMPT_IDS);
  const t1 = performance.now();
  const ids = [argmaxJs(logits)];
  for (let i = 1; i < N_GEN; i++) {
    ids.push(gpu ? await engine.decodeStepArgmax(ids[i - 1], []) : engine.decodeStepArgmax(ids[i - 1]));
  }
  const t2 = performance.now();

  const hash = await sha256Hex(ids);
  return {
    requested,
    backend,
    skipped,
    engineBuild: ENGINE_BUILD,
    caps,
    info: engine.info(),
    promptLen: PROMPT_IDS.length,
    prefillMs: t1 - t0,
    decodeMsPerTok: (t2 - t1) / (N_GEN - 1),
    ids,
    hash,
    matchesReference: hash === REFERENCE_HASH,
    text: engine.decodeIds(Uint32Array.from(ids)),
  };
}

self.onmessage = async (e) => {
  try {
    self.postMessage({ type: "done", result: await run(e.data) });
  } catch (err) {
    self.postMessage({ type: "error", text: err && err.message ? err.message : String(err) });
  }
};
