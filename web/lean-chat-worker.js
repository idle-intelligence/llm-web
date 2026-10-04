// Module Worker behind web/index.html's chat UI. Speaks a small protocol:
//
//   page -> worker:  {type:'load'} | {type:'chat', text} | {type:'stop'} | {type:'reset'}
//   worker -> page:  {type:'status', text, ready, progress?}
//                     {type:'token', text}
//                     {type:'done'}
//                     {type:'error', message}
//
// Backend is picked by capability only (never a benchmark), same policy as
// crates/lean/www/backends_worker.js: WebGPU if an adapter is granted, else
// CPU threads if the page is cross-origin isolated with SharedArrayBuffer
// and more than one hardware thread, else single-thread CPU (WASM SIMD128).
//
// Both LeanEngine (GPU) and LeanEngineCpu (threads/single) expose the same
// multi-turn chatGenerate/chatReset/AbortFlag surface as of
// crates/lean/src/chat.rs + src/web.rs (merged from lean-chat-api,
// ENGINE_BUILD 2026-10-04-demos-01) - every backend gets real
// conversation memory, sampling and a working Stop button. on_token's
// second argument (`text`) is the already-decoded delta for this token;
// this worker streams that directly and never calls decodeIds per token
// itself (an earlier version did, which produced empty output for
// SmolLM2's byte-level BPE tokenizer - fixed upstream in lean, not here).
//
// This file intentionally has NO top-level `import` or `await`: a module
// worker with a top-level await can drop messages posted before it
// resolves (2026-09-22 llm-life incident — see
// docs/runs/2026-09-28-lean-web.md). self.onmessage is wired, as a plain
// synchronous function, before anything else runs; it buffers into EARLY
// until the real handlers are installed, then the buffer is replayed.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-04-demos-01";

// Same model the chat.html test harness validates against
// (crates/lean/www/backends_common.js's "smollm2-360m" entry, referenceHash
// checked token-for-token against transformers) — Q4_0, not the Q4_K_M the
// wllama page used to load, because Q4_0 is the quant lean's GGUF loader is
// validated against.
const GGUF_URL = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_0.gguf";
const TOKENIZER_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json";
const TOKENIZER_CFG_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer_config.json";
const MAX_CTX = 2048;
const MODEL_CACHE = "lean-smollm2-360m-q4_0-v1";

function status(text, ready, progress) {
  self.postMessage({ type: "status", text, ready: !!ready, progress });
}

async function capabilities() {
  const caps = {
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    hasAdapter: false,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      caps.hasAdapter = !!adapter;
    } catch {
      caps.hasAdapter = false;
    }
  }
  caps.threadsCapable = caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

let engine = null;
let backend = null; // 'webgpu' | 'threads' | 'single'
let AbortFlagCtor = null;
let abortFlag = null;

async function createEngine(which) {
  if (which === "webgpu") {
    const mod = await import(`./lean/pkg/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`./lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    mod.leanInit();
    return { engine: await mod.LeanEngine.create(), AbortFlag: mod.AbortFlag };
  }
  if (which === "threads") {
    const mod = await import(`./lean/pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`./lean/pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
    await mod.initThreadPool(navigator.hardwareConcurrency);
    mod.leanInit();
    return { engine: mod.LeanEngineCpu.create(), AbortFlag: mod.AbortFlag };
  }
  const mod = await import(`./lean/pkg/lean.js?v=${ENGINE_BUILD}`);
  await mod.default(`./lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  mod.leanInit();
  return { engine: mod.LeanEngineCpu.create(), AbortFlag: mod.AbortFlag };
}

function backendLabel(b) {
  if (b === "webgpu") return "WebGPU";
  if (b === "threads") return "CPU threads";
  return "CPU single-thread";
}

async function load() {
  status(`engine build ${ENGINE_BUILD}, checking capabilities...`);
  const caps = await capabilities();
  const candidates = [caps.hasAdapter && "webgpu", caps.threadsCapable && "threads", "single"].filter(Boolean);

  status("fetching model (~219 MB, cached after first load)...");
  const { getModel } = await import("./lib/model-cache.js");
  const [ggufBytes, tokenizerBytes, tokenizerCfgBytes] = await getModel(
    [GGUF_URL, TOKENIZER_URL, TOKENIZER_CFG_URL],
    {
      cache: MODEL_CACHE,
      onProgress: (loaded, total) => status("downloading model...", false, { loaded, total }),
    }
  );
  const tokenizerJson = new TextDecoder().decode(tokenizerBytes);
  const tokenizerCfgJson = new TextDecoder().decode(tokenizerCfgBytes);

  const skipped = [];
  for (const c of candidates) {
    try {
      status(`starting ${c} backend...`);
      const r = await createEngine(c);
      status(`loading weights on ${c} backend...`);
      r.engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, MAX_CTX);
      engine = r.engine;
      AbortFlagCtor = r.AbortFlag;
      backend = c;
      break;
    } catch (e) {
      skipped.push(`${c}: ${e && e.message ? e.message : e}`);
      engine = null;
    }
  }
  if (!engine) throw new Error(`no backend available (${skipped.join("; ")})`);

  status(`ready, ${backendLabel(backend)}`, true);
}

async function chat(text) {
  if (!engine) {
    self.postMessage({ type: "error", message: "engine not loaded; send {type:'load'} first" });
    return;
  }
  try {
    // web/index.html sends one fresh prompt per Send, same as the wllama
    // page it replaced (no visible Reset button, no carried-over context).
    engine.chatReset();
    abortFlag = new AbortFlagCtor();
    await engine.chatGenerate(
      text,
      256,
      0.7,
      40,
      0.9,
      1.1,
      0,
      new Uint32Array(0),
      (_id, text) => self.postMessage({ type: "token", text }),
      abortFlag.cloneFlag()
    );
    self.postMessage({ type: "done" });
  } catch (e) {
    self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) });
  } finally {
    abortFlag = null;
  }
}

function stop() {
  if (abortFlag) abortFlag.abort();
}

function reset() {
  if (engine) engine.chatReset();
}

const handlers = {
  load: () => load().catch((e) => self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) })),
  chat: (msg) => chat(msg.text),
  stop: () => stop(),
  reset: () => reset(),
};

self.onmessage = (e) => handlers[e.data.type]?.(e.data);
for (const e of EARLY) handlers[e.data.type]?.(e.data);
