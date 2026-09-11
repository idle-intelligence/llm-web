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
 *   {type:'run', id, utterance, tools:[MCP tool objects], opts:{maxSteps:6, maxNewTokens:256, temperature:0, toolTimeoutMs:30000, systemPrompt?, prefillKernel?:'naive'|'pinned'}}
 *   {type:'toolResult', id, callId, result}      // or {type:'toolResult', id, callId, error}
 *   {type:'reset'}                                // drop conversation, keep model
 * worker -> page
 *   {type:'progress', loaded, total, shard}
 *   {type:'ready', info:{model, prefixTokens?}}
 *   {type:'step', id, step:{index, promptTokens, promptTokenIds, text, calls:[{name,args}]|null, prefillMs, decodeMs, tokens}}
 *   {type:'token', id, text}                      // streaming — NOT emitted by this engine, see docs/ENGINE.md
 *   {type:'callTool', id, callId, name, args}
 *   {type:'status', id?, phase:'prefill'|'decode'|'idle'|'kv-image', note?}
 *   {type:'done', id, transcript:{steps, finalText, totalMs}}
 *   {type:'error', id?, message}
 *
 * Prefix KV images (docs/ENGINE.md "Prefix KV images"): before each run's
 * first prefill, the worker checks OPFS then `<modelBase>/kv/<key>.kvimg`
 * for a prebuilt image of the system+tools prefix and imports it instead of
 * running prefill token-by-token (a 'status' message with phase:'kv-image'
 * reports the outcome); on a miss, it exports the freshly-prefilled prefix
 * and saves it to OPFS for next time.
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
// `<modelBase>/kv/` derived from the first GGUF shard URL (see
// `deriveKvBaseUrl`) — where prefix KV images (docs/ENGINE.md "Prefix KV
// images") are fetched from on a cache miss. Null if the shard URL doesn't
// contain a `/gguf/` segment to substitute.
let kvBaseUrl = null;

// Mirrors `LlmEngine::new()`'s Rust-side default (`web.rs`) — used to key
// `prefixKey`/`importKvImage`/`exportKvImage` calls when a run doesn't
// override `opts.systemPrompt`.
const DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant with access to tools.';

/** `http://host/gguf/<model>/foo.gguf` -> `http://host/kv/`. */
function deriveKvBaseUrl(shardUrl) {
  try {
    const u = new URL(shardUrl);
    const idx = u.pathname.indexOf('/gguf/');
    if (idx === -1) return null;
    u.pathname = u.pathname.slice(0, idx) + '/kv/';
    u.search = '';
    u.hash = '';
    return u.toString();
  } catch {
    return null;
  }
}

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

  kvBaseUrl = deriveKvBaseUrl(shards[0]);

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
      let buf;
      if (pending.length === 1) {
        buf = pending[0];
      } else {
        buf = new Uint8Array(pendingBytes);
        let offset = 0;
        for (const chunk of pending) {
          buf.set(chunk, offset);
          offset += chunk.byteLength;
        }
      }
      engine.appendModelShard(buf);
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

  // Model-identity fingerprint for prefix-KV-image matching (docs/ENGINE.md
  // "Prefix KV images") is computed inside `engine.load()` itself, from the
  // GGUF header bytes only (magic through the tensor-info table — a few
  // KB) + file size — NOT a hash of the whole 1.7GB+ model. No
  // crypto.subtle call and no extra JS-side pass over the shard bytes
  // needed here.
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

  const toolsJson = JSON.stringify(tools);
  const systemPrompt = (opts && opts.systemPrompt) || DEFAULT_SYSTEM_PROMPT;
  if (opts && opts.prefillKernel) engine.setPrefillKernel(opts.prefillKernel);

  const startedAt = performance.now();
  const steps = [];

  try {
    self.postMessage({ type: 'status', id, phase: 'kv-image' });
    const { imported } = await maybeImportKvImage(toolsJson, systemPrompt);

    self.postMessage({ type: 'status', id, phase: 'prefill' });
    let outcome = JSON.parse(await engine.start(utterance, toolsJson, JSON.stringify(opts || {})));
    self.postMessage({ type: 'status', id, phase: 'idle' });

    if (!imported) {
      // Fire-and-forget: don't block this run's response on a GPU readback
      // + OPFS write for next time.
      maybeExportKvImage(toolsJson, systemPrompt);
    }

    while (true) {
      if (outcome.outcome === 'error') {
        self.postMessage({ type: 'error', id, message: outcome.message });
        return;
      }

      const step = {
        index: steps.length,
        promptTokens: outcome.step.promptTokens,
        promptTokenIds: outcome.step.promptTokenIds,
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

// ---------------------------------------------------------------------------
// Prefix KV images (docs/ENGINE.md "Prefix KV images"): before the first
// prefill of a given tool set, check OPFS then `<modelBase>/kv/` for a
// prebuilt image and import it instead of running prefill token-by-token;
// on a miss, export the freshly-prefilled prefix after the fact and save it
// to OPFS so the next tab/run for the same tool set hits.
// ---------------------------------------------------------------------------
async function opfsRoot() {
  try {
    return await navigator.storage.getDirectory();
  } catch {
    return null;
  }
}

async function opfsReadKvImage(key) {
  const root = await opfsRoot();
  if (!root) return null;
  try {
    const fileHandle = await root.getFileHandle(`${key}.kvimg`);
    const file = await fileHandle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  } catch {
    return null;
  }
}

async function opfsWriteKvImage(key, bytes) {
  const root = await opfsRoot();
  if (!root) return;
  try {
    const fileHandle = await root.getFileHandle(`${key}.kvimg`, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(bytes);
    await writable.close();
  } catch (err) {
    console.warn('[llm-worker] failed to save kv image to OPFS:', err.message || err);
  }
}

/** Check OPFS then network for a prefix KV image and import it if found and valid. */
async function maybeImportKvImage(toolsJson, systemPrompt) {
  let key;
  try {
    key = engine.prefixKey(toolsJson, systemPrompt);
  } catch (err) {
    console.warn('[llm-worker] prefixKey failed:', err.message || err);
    return { imported: false, key: null };
  }

  let bytes = await opfsReadKvImage(key);
  let source = 'opfs';
  if (!bytes && kvBaseUrl) {
    try {
      const res = await fetch(`${kvBaseUrl}${key}.kvimg`, { cache: 'no-store' });
      if (res.ok) {
        bytes = new Uint8Array(await res.arrayBuffer());
        source = 'network';
      }
      // 404 is the normal "no prebuilt image for this prefix" case — fall
      // through to the miss path below, no error.
    } catch (err) {
      console.warn('[llm-worker] kv image fetch failed:', err.message || err);
    }
  }
  if (!bytes) {
    self.postMessage({ type: 'status', phase: 'kv-image', note: `miss (${key})` });
    return { imported: false, key };
  }

  let ok = false;
  try {
    ok = engine.importKvImage(bytes, toolsJson, systemPrompt);
  } catch (err) {
    console.warn('[llm-worker] importKvImage failed:', err.message || err);
  }
  if (ok) {
    self.postMessage({
      type: 'status',
      phase: 'kv-image',
      note: `loaded ${(bytes.length / 1e6).toFixed(1)} MB from ${source} (${key})`,
    });
    if (source === 'network') await opfsWriteKvImage(key, bytes);
  } else {
    self.postMessage({ type: 'status', phase: 'kv-image', note: `mismatch, ignoring (${key})` });
  }
  return { imported: ok, key };
}

/** Export the just-prefilled prefix and save it to OPFS for next time (fire-and-forget). */
async function maybeExportKvImage(toolsJson, systemPrompt) {
  try {
    const bytes = await engine.exportKvImage(toolsJson, systemPrompt);
    const key = engine.prefixKey(toolsJson, systemPrompt);
    await opfsWriteKvImage(key, bytes);
    self.postMessage({
      type: 'status',
      phase: 'kv-image',
      note: `exported ${(bytes.length / 1e6).toFixed(1)} MB to OPFS (${key})`,
    });
  } catch (err) {
    console.warn('[llm-worker] kv image export failed:', err.message || err);
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
