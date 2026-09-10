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
 *   {type:'run', id, utterance, tools:[MCP tool objects], opts:{maxSteps:6, maxNewTokens:256, temperature:0}}
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
 */

import init, { initWgpuDevice, LlmEngine } from './pkg/llm_wasm.js';

let engine = null;
let wasmReady = false;

// Pending tool calls this worker is waiting on: callId -> {resolve, reject}
const pendingToolCalls = new Map();

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'load':
        await handleLoad(msg.model);
        break;
      case 'run':
        await handleRun(msg);
        break;
      case 'toolResult':
        handleToolResult(msg);
        break;
      case 'reset':
        pendingToolCalls.clear();
        if (engine) engine.reset();
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
// load: fetch shard URLs (with progress) + tokenizer/template JSON, then
// initWgpuDevice() + LlmEngine.load().
// ---------------------------------------------------------------------------
async function handleLoad(model) {
  if (!wasmReady) {
    await init();
    wasmReady = true;
  }

  const shards = model?.shards || [];
  if (shards.length === 0) throw new Error('load: model.shards is empty — no GGUF URL to fetch');
  const shardBuffers = [];
  for (const url of shards) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`shard fetch failed: ${url} (${res.status})`);
    const total = parseInt(res.headers.get('Content-Length') || '0', 10);
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    let lastReported = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      if (loaded - lastReported >= 8 * 1024 * 1024 || done) {
        self.postMessage({ type: 'progress', loaded, total, shard: url });
        lastReported = loaded;
      }
    }
    self.postMessage({ type: 'progress', loaded, total, shard: url });
    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buf.set(chunk, offset);
      offset += chunk.byteLength;
    }
    shardBuffers.push(buf);
  }

  if (!model.tokenizerUrl) throw new Error('load: model.tokenizerUrl is missing');
  if (!model.templateUrl) throw new Error('load: model.templateUrl is missing');

  const fetchJsonText = async (url, label) => {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${label} fetch failed: ${url} (${res.status})`);
    return res.text();
  };
  const [tokenizerJson, tokenizerConfigJson] = await Promise.all([
    fetchJsonText(model.tokenizerUrl, 'tokenizer.json'),
    fetchJsonText(model.templateUrl, 'tokenizer_config.json'),
  ]);

  await initWgpuDevice();
  engine = new LlmEngine();
  for (const shard of shardBuffers) engine.appendModelShard(shard);

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
          const result = await callToolFromWorker(id, call.call_id, call.name, call.arguments);
          results.push({ call_id: call.call_id, result });
        } catch (err) {
          self.postMessage({ type: 'error', id, message: `tool call failed: ${err.message || err}` });
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

function callToolFromWorker(runId, callId, name, args) {
  self.postMessage({ type: 'callTool', id: runId, callId, name, args });
  return new Promise((resolve, reject) => {
    pendingToolCalls.set(callId, { resolve, reject });
  });
}

function handleToolResult(msg) {
  const pending = pendingToolCalls.get(msg.callId);
  if (!pending) return;
  pendingToolCalls.delete(msg.callId);
  if (msg.error) pending.reject(new Error(msg.error));
  else pending.resolve(msg.result);
}
