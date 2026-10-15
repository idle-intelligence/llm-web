// Module Worker behind web/index.html's chat UI. Speaks a small protocol:
//
//   page -> worker:  {type:'load', model, backend} | {type:'chat', text} | {type:'stop'} | {type:'reset'}
//   worker -> page:  {type:'progress', loaded, total}
//                     {type:'ready', backend}
//                     {type:'token', text}
//                     {type:'done', promptTokens, trimmed}
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
// lean's engine is text in, text out (`Lean.generate(text, params,
// onToken, abort)`): it applies no chat template and keeps no
// conversation. This worker holds the conversation (`HISTORY`), renders
// the whole of it through the model's chat template on every call
// (`chatRender`, from lean-chat, in the same wasm package), and passes
// that text to the engine, which reuses the KV cache of the common prefix
// by itself. `onToken(text)` streams already-decoded reply text. On a
// `ContextOverflow` error (`e.name === "ContextOverflow"`), this worker
// drops the oldest exchange (`chatTrimOldestExchange`) and retries, the
// same client-side trim `examples/chat.rs` uses natively. The page's "New chat" button sends {type:'reset'}, which just
// clears `HISTORY`.
//
// This file intentionally has NO top-level `import` or `await`: a module
// worker with a top-level await can drop messages posted before it
// resolves (2026-09-22 llm-life incident). self.onmessage is wired, as a
// plain synchronous function, before anything else runs; it buffers into
// EARLY until the real handlers are installed, then the buffer is replayed.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

const ENGINE_BUILD = "2026-10-15-sampling-merge-01";
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
let lean = null; // the loaded wasm module
let chatTemplate = null;
let abortFlag = null;

// Conversation state: lean keeps no history of its own (see this file's
// doc comment), so the worker owns it, same role as examples/chat.rs's
// `history` on the native side.
// Every conversation opens with a short system message; context trimming
// keeps it (see generateWithTrim).
const SYSTEM_MESSAGE = { role: "system", content: "Answer briefly, in one or two sentences." };
let HISTORY = [SYSTEM_MESSAGE];

// 'threads' runs the CPU forward pass from the threads build (pkg-mt);
// 'webgpu' and 'single' use the default build (pkg).
async function loadModule(which) {
  if (which === "threads") {
    const mod = await import(`./lean/pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`./lean/pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
    await mod.initThreadPool(navigator.hardwareConcurrency);
    return mod;
  }
  const mod = await import(`./lean/pkg/lean.js?v=${ENGINE_BUILD}`);
  await mod.default(`./lean/pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  return mod;
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
      const mod = await loadModule(c);
      console.log(`[lean-chat-worker] loading weights on ${c} backend...`);
      engine = await mod.Lean.load({ gguf: ggufBytes, tokenizer: tokenizerJson, maxCtx: model.maxCtx, backend: c === "webgpu" ? "webgpu" : "cpu" });
      chatTemplate = mod.ChatTemplate.fromTokenizerConfig(tokenizerCfgJson);
      lean = mod;
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

// On a context overflow, drops the oldest exchange (chatTrimOldestExchange
// keeps the system message and the current question) and retries - the
// same rule examples/chat.rs uses natively. Gives up (rethrows) once there
// is nothing left to drop. Returns the reply, the prompt it answered and
// whether any exchange was dropped.
async function generateWithTrim(onToken) {
  let trimmedAny = false;
  for (;;) {
    const prompt = lean.chatRender(chatTemplate, HISTORY, null, true);
    try {
      const reply = await engine.generate(prompt, GEN_PARAMS, onToken, abortFlag.cloneFlag());
      return { reply, prompt, trimmed: trimmedAny };
    } catch (e) {
      const trimmed = e && e.name === "ContextOverflow" ? lean.chatTrimOldestExchange(HISTORY) : undefined;
      if (trimmed) {
        console.log(`[lean-chat-worker] context overflow (needed=${e.needed} available=${e.available}); dropping the oldest turn`);
        HISTORY = trimmed;
        trimmedAny = true;
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
    abortFlag = new lean.AbortFlag();
    const { reply, prompt, trimmed } = await generateWithTrim((text) => self.postMessage({ type: "token", text }));
    // Prompt size of this turn, after any trimming.
    let promptTokens = null;
    try { promptTokens = engine.tokenCount(prompt); } catch (_) {}
    HISTORY.push({ role: "assistant", content: reply });
    self.postMessage({ type: "done", promptTokens, trimmed });
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
