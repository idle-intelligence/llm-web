// Module Worker behind web/index.html's chat UI. Speaks a small protocol:
//
//   page -> worker:  {type:'load', model, backend} | {type:'chat', text} | {type:'stop'} | {type:'reset'}
//   worker -> page:  {type:'progress', loaded, total}
//                     {type:'ready', backend}
//                     {type:'token', text}
//                     {type:'done'}
//                     {type:'error', message}
//
// `model` is one of web/index.html's MODELS entries (label, gguf,
// tokenizer, tokenizerCfg, cache, maxCtx). `backend` is 'auto', 'webgpu',
// 'threads' or 'single'; 'auto' picks by capability (WebGPU if an adapter
// is granted, else CPU threads if cross-origin isolated with
// SharedArrayBuffer and more than one hardware thread, else single-thread
// CPU). Any other value
// tries only that backend and fails if it can't start — the page only
// offers backends the capability check found available, so this should
// not happen from normal use.
//
// The page owns every status string the visitor sees; this worker never
// sends prose, only the bytes/total a progress bar needs, the backend it
// loaded on, and plain ready/token/done/error signals. The engine build
// tag is console.log only.
//
// Both LeanEngine (GPU) and LeanEngineCpu (threads/single) expose the same
// one-turn `generate(messages, params, onToken, abort)` surface
// (crates/lean/src/chat.rs + src/web.rs) — lean keeps no conversation of
// its own, so this worker holds the conversation (`HISTORY`) and sends
// the whole thing on every call. `onToken(text)` streams already-decoded
// reply text, no token ids. On a `ContextOverflow` error (`e.name ===
// "ContextOverflow"`), this worker drops the oldest user/assistant pair
// and retries — the same client-side trim `examples/chat.rs` uses
// natively. The page's "New chat" button sends {type:'reset'}, which just
// clears `HISTORY`; the engine's own KV cache catches up on the next call
// (prefix reuse finds nothing in common and reprefills from scratch).
//
// This file intentionally has NO top-level `import` or `await`: a module
// worker with a top-level await can drop messages posted before it
// resolves (2026-09-22 llm-life incident). self.onmessage is wired, as a
// plain synchronous function, before anything else runs; it buffers into
// EARLY until the real handlers are installed, then the buffer is replayed.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-09-one-turn-01";
// Rewritten to true by scripts/build.sh when BUILD_THREADS=1 built pkg-mt
// into this deployment; GitHub Pages never sets it.
const MT_BUILT = false;

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
  caps.threadsCapable = MT_BUILT && caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

let engine = null;
let backend = null; // 'webgpu' | 'threads' | 'single'
let AbortFlagCtor = null;
let abortFlag = null;

// Conversation state: lean keeps no history of its own (see this file's
// doc comment), so the worker owns it, same role as examples/chat.rs's
// `history` on the native side.
// Every conversation opens with a short system message; context trimming
// keeps it (see generateWithTrim).
const SYSTEM_MESSAGE = { role: "system", content: "Answer briefly, in one or two sentences." };
let HISTORY = [SYSTEM_MESSAGE];

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

async function load(model, backendChoice) {
  console.log(`[lean-chat-worker] engine build ${ENGINE_BUILD}, model ${model.label}, backend choice ${backendChoice}`);
  let candidates;
  if (backendChoice && backendChoice !== "auto") {
    candidates = [backendChoice];
  } else {
    const caps = await capabilities();
    candidates = [caps.hasAdapter && "webgpu", caps.threadsCapable && "threads", "single"].filter(Boolean);
  }

  const { getModel } = await import("./lib/model-cache.js");
  const [ggufBytes, tokenizerBytes, tokenizerCfgBytes] = await getModel(
    [model.gguf, model.tokenizer, model.tokenizerCfg],
    {
      cache: model.cache,
      onProgress: (loaded, total) => self.postMessage({ type: "progress", loaded, total }),
    }
  );
  const tokenizerJson = new TextDecoder().decode(tokenizerBytes);
  const tokenizerCfgJson = new TextDecoder().decode(tokenizerCfgBytes);

  const skipped = [];
  for (const c of candidates) {
    try {
      console.log(`[lean-chat-worker] starting ${c} backend...`);
      const r = await createEngine(c);
      console.log(`[lean-chat-worker] loading weights on ${c} backend...`);
      r.engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, model.maxCtx);
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

  console.log(`[lean-chat-worker] ready, ${backendLabel(backend)}, engine build ${ENGINE_BUILD}`);
  self.postMessage({ type: "ready", backend });
}

const GEN_PARAMS = { maxNewTokens: 256, temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.1, seed: 0 };

// Drops the oldest user/assistant pair (after any leading system message)
// and retries - the same trim rule examples/chat.rs uses natively. Gives
// up (rethrows) once there is nothing left to drop.
async function generateWithTrim(onToken) {
  for (;;) {
    try {
      return await engine.generate(HISTORY, GEN_PARAMS, onToken, abortFlag.cloneFlag());
    } catch (e) {
      const dropFrom = HISTORY[0] && HISTORY[0].role === "system" ? 1 : 0;
      if (e && e.name === "ContextOverflow" && HISTORY.length > dropFrom + 2) {
        console.log(`[lean-chat-worker] context overflow (needed=${e.needed} available=${e.available}); dropping the oldest turn`);
        HISTORY.splice(dropFrom, 2);
        continue;
      }
      throw e;
    }
  }
}

async function chat(text) {
  if (!engine) {
    self.postMessage({ type: "error", message: "engine not loaded; send {type:'load'} first" });
    return;
  }
  HISTORY.push({ role: "user", content: text });
  try {
    abortFlag = new AbortFlagCtor();
    const reply = await generateWithTrim((text) => self.postMessage({ type: "token", text }));
    // Prompt size of this turn, after any trimming (before the reply joins HISTORY).
    let promptTokens = null;
    try { promptTokens = engine.tokenCount(HISTORY); } catch (_) {}
    HISTORY.push({ role: "assistant", content: reply });
    self.postMessage({ type: "done", promptTokens });
  } catch (e) {
    HISTORY.pop(); // the user turn never got a reply - take it back out.
    self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) });
  } finally {
    abortFlag = null;
  }
}

function stop() {
  if (abortFlag) abortFlag.abort();
}

function reset() {
  HISTORY = [SYSTEM_MESSAGE];
}

const handlers = {
  load: (msg) => load(msg.model, msg.backend).catch((e) => self.postMessage({ type: "error", message: e && e.message ? e.message : String(e) })),
  chat: (msg) => chat(msg.text),
  stop: () => stop(),
  reset: () => reset(),
};

self.onmessage = (e) => handlers[e.data.type]?.(e.data);
for (const e of EARLY) handlers[e.data.type]?.(e.data);
