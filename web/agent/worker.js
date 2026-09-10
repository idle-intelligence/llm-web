/**
 * Web Worker hosting the real llm-wasm engine (xLAM-2-3b-fc-r, Burn+wgpu,
 * WebGPU). Speaks the same protocol as the trucs.ai stub
 * (`sonos/llm-worker.js`) verbatim, so it's a drop-in replacement there —
 * only this file's internals differ; message shapes are unchanged.
 *
 * Worker protocol (verbatim, see sonos/llm-worker.js's top-of-file comment):
 *
 * page -> worker
 *   {type:'load', model:{id, shards:[url...], tokenizerUrl, templateUrl}}   // templateUrl: tokenizer_config.json
 *   {type:'run', id, utterance, tools:[MCP tool objects], opts:{maxSteps:6, maxNewTokens:256, temperature:0, toolTimeoutMs:30000}}
 *   {type:'toolResult', id, callId, result}      // or {type:'toolResult', id, callId, error}
 *   {type:'reset'}                                // drop conversation, keep model
 * worker -> page
 *   {type:'progress', loaded, total, shard}
 *   {type:'ready', info:{model, prefixTokens?}}
 *   {type:'step', id, step:{index, promptTokens, text, calls:[{name,args}]|null, prefillMs, decodeMs, tokens}}
 *   {type:'token', id, text}                      // streaming — NOT emitted by this engine, see docs/ENGINE.md
 *   {type:'callTool', id, callId, name, args}
 *   {type:'status', id?, phase:'prefill'|'decode'|'idle'}
 *   {type:'done', id, transcript:{steps, finalText, totalMs}}
 *   {type:'error', id?, message}
 *
 * Re-entrancy: only one of 'load'/'run' may be in flight at a time. A 'run'
 * arriving while one is already running is rejected immediately; a 'load'
 * arriving while a run is in flight is rejected too (loading swaps out the
 * engine from under the run); a 'reset' arriving mid-run is deferred until
 * the run settles, then applied.
 *
 * GPU debug: gpu-debug.js (WebGPU validation error-scope wrapping) is off by
 * default — opt in with ?gpudebug=1 on this worker's URL. Must run before
 * pkg/llm_wasm.js touches the GPU, hence the dynamic import ahead of the pkg
 * import in handleLoad.
 */

const workerSearchParams = new URL(self.location.href).searchParams;
const USE_GPU_DEBUG = workerSearchParams.get('gpudebug') === '1';

let engine = null;
let wasmReady = false;
let initWgpuDevice = null;
let LlmEngine = null;

// Pending tool calls this worker is waiting on: callId -> {resolve, reject}
const pendingToolCalls = new Map();

// Serialization: only one of 'load'/'run' may be in flight at a time.
let busy = false;
let pendingReset = false;

function applyReset() {
  pendingToolCalls.clear();
  if (engine) engine.reset();
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'load':
        if (busy) {
          self.postMessage({ type: 'error', id: msg.id, message: 'load rejected: a run is already in progress' });
          break;
        }
        busy = true;
        try {
          await handleLoad(msg.model);
        } finally {
          busy = false;
          if (pendingReset) {
            pendingReset = false;
            applyReset();
          }
        }
        break;
      case 'run':
        if (busy) {
          self.postMessage({ type: 'error', id: msg.id, message: 'run already in progress' });
          break;
        }
        busy = true;
        try {
          await handleRun(msg);
        } finally {
          busy = false;
          if (pendingReset) {
            pendingReset = false;
            applyReset();
          }
        }
        break;
      case 'toolResult':
        handleToolResult(msg);
        break;
      case 'reset':
        if (busy) {
          pendingReset = true;
          break;
        }
        applyReset();
        break;
      default:
        console.warn('[llm-worker] unknown message type:', msg.type);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: (err.message || String(err)) + (err.stack ? '\n' + err.stack : '') });
  }
};

// Anything that escapes onmessage's try/catch (e.g. a throw during the
// top-level `import('./pkg/llm_wasm.js')`) or a rejected promise that
// nobody awaited would otherwise vanish silently — relay both to the page.
self.addEventListener('error', (e) => {
  console.error('[llm-worker] uncaught error:', e.error || e.message);
  self.postMessage({ type: 'error', message: 'worker error: ' + (e.error?.stack || e.error?.message || e.message) });
});
self.addEventListener('unhandledrejection', (e) => {
  const reason = e.reason;
  console.error('[llm-worker] unhandled rejection:', reason);
  self.postMessage({ type: 'error', message: 'unhandled rejection: ' + (reason?.stack || reason?.message || String(reason)) });
});

// ---------------------------------------------------------------------------
// load: fetch shard URLs (with progress), streaming each shard straight into
// the engine, then tokenizer/template JSON, then initWgpuDevice() +
// LlmEngine.load().
// ---------------------------------------------------------------------------
async function handleLoad(model) {
  if (!wasmReady) {
    if (USE_GPU_DEBUG) await import('./gpu-debug.js');
    const pkg = await import('./pkg/llm_wasm.js');
    initWgpuDevice = pkg.initWgpuDevice;
    LlmEngine = pkg.LlmEngine;
    await pkg.default();
    wasmReady = true;
  }

  const shards = model?.shards || [];
  if (shards.length === 0) throw new Error('load: model.shards is empty — no GGUF URL to fetch');

  if (!model.tokenizerUrl) throw new Error('load: model.tokenizerUrl is missing');
  if (!model.templateUrl) throw new Error('load: model.templateUrl is missing');

  await initWgpuDevice();
  engine = new LlmEngine();

  // Stream each shard straight into the engine instead of buffering it in
  // JS first: peak JS heap is one coalesced chunk, not 2x the shard size.
  // Small network chunks are coalesced to ~16 MB before each
  // appendModelShard call so we don't make one wasm call per TCP segment.
  const COALESCE_BYTES = 16 * 1024 * 1024;
  const PROGRESS_STEP = 8 * 1024 * 1024; // post at least every 8 MB
  for (const url of shards) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`shard fetch failed: ${url} (${res.status})`);
    const total = parseInt(res.headers.get('Content-Length') || '0', 10);
    const reader = res.body.getReader();
    let loaded = 0;
    let lastPosted = 0;
    let pending = [];
    let pendingBytes = 0;
    const flush = () => {
      if (pendingBytes === 0) return;
      if (pending.length === 1) {
        engine.appendModelShard(pending[0]);
      } else {
        const buf = new Uint8Array(pendingBytes);
        let offset = 0;
        for (const chunk of pending) {
          buf.set(chunk, offset);
          offset += chunk.byteLength;
        }
        engine.appendModelShard(buf);
      }
      pending = [];
      pendingBytes = 0;
    };
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending.push(value);
      pendingBytes += value.byteLength;
      loaded += value.byteLength;
      if (pendingBytes >= COALESCE_BYTES) flush();
      if (loaded - lastPosted >= PROGRESS_STEP) {
        lastPosted = loaded;
        self.postMessage({ type: 'progress', loaded, total, shard: url });
      }
    }
    flush();
    self.postMessage({ type: 'progress', loaded, total, shard: url });
  }

  const fetchJsonText = async (url, label) => {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${label} fetch failed: ${url} (${res.status})`);
    return res.text();
  };
  const [tokenizerJson, tokenizerConfigJson] = await Promise.all([
    fetchJsonText(model.tokenizerUrl, 'tokenizer.json'),
    fetchJsonText(model.templateUrl, 'tokenizer_config.json'),
  ]);

  await engine.load(tokenizerJson, tokenizerConfigJson, (stage, step, total) => {
    self.postMessage({ type: 'progress', loaded: step, total, shard: `[gguf] ${stage}` });
  });

  self.postMessage({ type: 'ready', info: { model: model?.id || '[llm-wasm]' } });
}

// ---------------------------------------------------------------------------
// run: start() -> loop on NeedTools (round-trip through the page's real MCP
// client via callTool) -> provideToolResults() -> ... -> Final -> done.
// engine.start()/provideToolResults() block the worker thread while the
// model prefills/decodes; 'status' messages bracket those calls so the page
// can drive a "thinking..." timer instead of looking hung.
// ---------------------------------------------------------------------------
async function handleRun(msg) {
  const { id, utterance, tools, opts } = msg;
  if (!engine) {
    self.postMessage({ type: 'error', id, message: 'engine not loaded — send {type:"load"} first' });
    return;
  }

  const startedAt = performance.now();
  const steps = [];

  try {
    self.postMessage({ type: 'status', id, phase: 'prefill' });
    let outcome = JSON.parse(await engine.start(utterance, JSON.stringify(tools), JSON.stringify(opts || {})));
    self.postMessage({ type: 'status', id, phase: 'idle' });

    while (true) {
      if (outcome.outcome === 'error') {
        self.postMessage({ type: 'error', id, message: outcome.message });
        return;
      }

      const step = {
        index: steps.length,
        promptTokens: outcome.step.promptTokens,
        text: outcome.step.text,
        calls: outcome.outcome === 'needTools' ? outcome.calls.map((c) => ({ name: c.name, args: c.arguments })) : null,
        prefillMs: outcome.step.prefillMs,
        decodeMs: outcome.step.decodeMs,
        tokens: outcome.step.tokens,
      };
      steps.push(step);
      self.postMessage({ type: 'step', id, step });

      if (outcome.outcome === 'final') {
        self.postMessage({
          type: 'done',
          id,
          transcript: { steps, finalText: outcome.text, totalMs: performance.now() - startedAt },
        });
        return;
      }

      // needTools: round-trip every call through the page, in order.
      const results = [];
      for (const call of outcome.calls) {
        try {
          const result = await callToolFromWorker(id, call.call_id, call.name, call.arguments, opts);
          results.push({ call_id: call.call_id, result });
        } catch (err) {
          const message = err.message || String(err);
          // Timeout errors from callToolFromWorker are already fully-formed
          // ("tool <name> timed out after 30 s") — don't double-wrap them.
          self.postMessage({ type: 'error', id, message: message.startsWith('tool ') ? message : `tool call failed: ${message}` });
          return;
        }
      }

      self.postMessage({ type: 'status', id, phase: 'decode' });
      outcome = JSON.parse(await engine.provideToolResults(JSON.stringify(results)));
      self.postMessage({ type: 'status', id, phase: 'idle' });
    }
  } catch (err) {
    console.error('[llm-worker] run failed:', err);
    self.postMessage({ type: 'error', id, message: (err.message || String(err)) + (err.stack ? '\n' + err.stack : '') });
  }
}

const DEFAULT_TOOL_TIMEOUT_MS = 30000;

function callToolFromWorker(runId, callId, name, args, opts) {
  const timeoutMs = opts?.toolTimeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS;
  self.postMessage({ type: 'callTool', id: runId, callId, name, args });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingToolCalls.delete(callId);
      reject(new Error(`tool ${name} timed out after ${Math.round(timeoutMs / 1000)} s`));
    }, timeoutMs);
    pendingToolCalls.set(callId, {
      resolve: (v) => { clearTimeout(timer); resolve(v); },
      reject: (e) => { clearTimeout(timer); reject(e); },
    });
  });
}

function handleToolResult(msg) {
  const pending = pendingToolCalls.get(msg.callId);
  if (!pending) return;
  pendingToolCalls.delete(msg.callId);
  if (msg.error) pending.reject(new Error(msg.error));
  else pending.resolve(msg.result);
}
