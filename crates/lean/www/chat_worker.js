// Module Worker behind chat.html: multi-turn chat on any backend through the
// one chat API both engines expose (`chatGenerate`/`chatReset`, see
// crates/lean/src/web.rs). Backend choice copies backends_worker.js
// `createEngine()`: by capability only (WebGPU adapter, else CPU threads when
// cross-origin isolated, else one CPU thread), or forced with ?backend=.
//
// Messages in: {type: "load", backend, model}, {type: "send", text, params},
// {type: "stop"}, {type: "reset"}. Out: {type: "status"|"ready"|"piece"|
// "done"|"error", ...}. The text shown is the `text` argument of
// `on_token(id, text)`; the callback never calls back into the engine.
import { capabilities, fetchBytes, fetchText } from "./backends_common.js?v=2026-10-04-release-01";
const ENGINE_BUILD = "2026-10-04-release-01";

const MODELS = {
  "qwen25-0.5b": { dir: "./model/", gguf: "qwen2.5-0.5b-instruct-q4_0.gguf" },
  "smollm2-360m": { dir: "./model_smollm2_360m/", gguf: "SmolLM2-360M-Instruct-Q4_0.gguf" },
  "smollm2-1.7b": { dir: "./model_smollm2_1_7b/", gguf: "SmolLM2-1.7B-Instruct-Q4_0.gguf" },
};
const MAX_CTX = 2048;

let engine = null;
let mod = null;
let wasm = null; // the module's exports, for wasmMemoryBytes
let abortFlag = null;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function createEngine(backend, caps) {
  if (backend === "webgpu") {
    if (!caps.hasAdapter) throw new Error(`no WebGPU adapter (${caps.adapter})`);
    mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
    wasm = await mod.default({ module_or_path: `../pkg/lean_bg.wasm?v=${ENGINE_BUILD}` });
    mod.leanInit();
    return await mod.LeanEngine.create();
  }
  if (backend === "threads") {
    if (!caps.threadsCapable) {
      throw new Error(
        `threads need crossOriginIsolated, SharedArrayBuffer and >1 hardware thread ` +
          `(got ${caps.crossOriginIsolated}, ${caps.sharedArrayBuffer}, ${caps.hardwareConcurrency})`
      );
    }
    mod = await import(`../pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    wasm = await mod.default({ module_or_path: `../pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}` });
    await mod.initThreadPool(caps.hardwareConcurrency);
    mod.leanInit();
    return mod.LeanEngineCpu.create();
  }
  mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  wasm = await mod.default({ module_or_path: `../pkg/lean_bg.wasm?v=${ENGINE_BUILD}` });
  mod.leanInit();
  return mod.LeanEngineCpu.create();
}

async function load(forced, model) {
  const caps = await capabilities();
  const backend = forced || (caps.hasAdapter ? "webgpu" : caps.threadsCapable ? "threads" : "single");
  status(`engine build ${ENGINE_BUILD}, backend ${backend}, fetching ${model}...`);
  engine = await createEngine(backend, caps);
  const m = MODELS[model];
  if (!m) throw new Error(`unknown model ${model}`);
  const [gguf, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(m.dir + m.gguf),
    fetchText(m.dir + "tokenizer.json"),
    fetchText(m.dir + "tokenizer_config.json"),
  ]);
  engine.load(gguf, tokenizerJson, tokenizerCfgJson, MAX_CTX);
  self.postMessage({ type: "ready", backend, model, engineBuild: ENGINE_BUILD, adapter: caps.adapter, info: engine.info(), wasmMemoryBytes: wasm.memory.buffer.byteLength });
}

async function send(text, p) {
  abortFlag = new mod.AbortFlag();
  let tokens = 0;
  let streamed = "";
  const t0 = performance.now();
  try {
    const reply = await engine.chatGenerate(text, p.maxNewTokens, p.temperature, p.topK, p.topP, p.repPenalty, p.seed, new Uint32Array(0), (id, piece) => {
      if (id >= 0) tokens += 1;
      streamed += piece;
      if (piece) self.postMessage({ type: "piece", text: piece });
    }, abortFlag.cloneFlag());
    self.postMessage({ type: "done", reply, streamed, tokens, ms: performance.now() - t0, wasmMemoryBytes: wasm.memory.buffer.byteLength });
  } catch (e) {
    self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) });
  }
  abortFlag = null;
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") await load(msg.backend, msg.model);
    else if (msg.type === "send") await send(msg.text, msg.params);
    else if (msg.type === "stop") abortFlag && abortFlag.abort();
    else if (msg.type === "reset") engine.chatReset();
  } catch (e) {
    self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) });
  }
};
