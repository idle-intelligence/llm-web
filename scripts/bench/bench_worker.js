// Worker behind bench.html. One loaded engine on the requested backend, then:
//   cold prefill   a fresh prompt of N tokens, maxNewTokens 1; time to the first token
//   suffix prefill the same prompt with ~N/4 more tokens appended (prefix reused)
//   decode         greedy 64 tokens from a short prompt, ms per token after the first
//   exact          hash of a fixed 48-token greedy reply (same on every browser and backend)
// phases=loads: `reps` loads and frees of the engine, `decode` ms pause after each (see run()).
// Expects ../lean/pkg (built wasm), model.gguf, tokenizer.json and
// tokenizer_config.json next to the page.
const EARLY = [];
self.onmessage = (e) => EARLY.push(e);

// Panic messages and engine warnings go to console.error; keep them for the report.
const logs = [];
for (const level of ["error", "warn"]) {
  const orig = console[level].bind(console);
  console[level] = (...a) => { const t = a.map(String).join(" ").slice(0, 600); logs.push(t); self.postMessage({ type: "log", text: t }); orig(...a); };
}

// WebGPU validation errors the engine does not look at: log them too.
if (self.GPUAdapter) {
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...a) {
    const device = await requestDevice.apply(this, a);
    device.addEventListener("uncapturederror", (e) => console.error("uncaptured WebGPU error: " + e.error.message + " @" + JSON.stringify(gpuStats)));
    device.lost.then((i) => console.error("device lost: " + i.reason + " " + i.message));
    return device;
  };
}

// Running totals of what the engine hands to the device, printed with the first uncaptured error.
const gpuStats = { createdBytes: 0, createdCount: 0, writtenBytes: 0, writeCount: 0, submits: 0 };
if (self.GPUDevice) {
  const createBuffer = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (d) { gpuStats.createdBytes += d.size; gpuStats.createdCount++; return createBuffer.call(this, d); };
  const writeBuffer = GPUQueue.prototype.writeBuffer;
  GPUQueue.prototype.writeBuffer = function (b, o, data, ...r) { gpuStats.writtenBytes += data.byteLength ?? 0; gpuStats.writeCount++; return writeBuffer.call(this, b, o, data, ...r); };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (c) { gpuStats.submits++; return submit.call(this, c); };
}

const status = (text) => self.postMessage({ type: "status", text });
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : NaN; };

const QUESTIONS = ["What is the capital of France? Answer in one sentence.", "List three prime numbers.", "Translate 'good morning' to Spanish."];
const WORDS = "the quick brown fox jumps over a lazy dog while seven small birds sing near an old stone bridge across the river".split(" ");
function makeText(mod, lean, nTokens, salt) {
  let t = `Document ${salt}. `;
  let i = salt * 7;
  while (lean.tokenCount(t) < nTokens) t += WORDS[i++ % WORDS.length] + " ";
  return t;
}

async function sha(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].slice(0, 8).map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function timeFirstToken(lean, text, maxNew) {
  const t0 = performance.now();
  let first = null;
  let n = 0;
  const reply = await lean.generate(text, { maxNewTokens: maxNew, temperature: 0 }, () => { if (first === null) first = performance.now(); n++; });
  return { ttft: first - t0, total: performance.now() - t0, n, reply };
}

async function run({ backend, build, prompts, reps, decode, mini, phases, sampling }) {
  status("fetching");
  const [gguf, tokenizer, tokenizerCfg] = await Promise.all([
    fetch("./model.gguf").then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
    fetch("./tokenizer.json").then((r) => r.text()),
    fetch("./tokenizer_config.json").then((r) => r.text()),
  ]);
  status("loading wasm");
  const mod = await import(`../lean/pkg/lean.js?v=${build}`);
  await mod.default(`../lean/pkg/lean_bg.wasm?v=${build}`);
  if (phases === "loads") {
    // Load, answer one fixed question, free, wait, repeat. The page status reads
    // "settled <k>" once load k has been freed and the pause is over: an outside
    // sampler of GPU memory reads it there, and it must not grow with k.
    const n = reps;
    const settleMs = decode;
    const template0 = mod.ChatTemplate.fromTokenizerConfig(tokenizerCfg);
    const hashes = [];
    for (let k = 1; k <= n; k++) {
      const l = await mod.Lean.load({ gguf, tokenizer, maxCtx: 2048, backend: backend === "webgpu" ? "webgpu" : "cpu" });
      const reply = await l.generate(mod.chatRender(template0, [{ role: "user", content: QUESTIONS[0] }], null, true), { maxNewTokens: 32, temperature: 0 }, () => {});
      hashes.push(await sha(reply));
      l.free();
      await new Promise((r) => setTimeout(r, settleMs));
      status("settled " + k);
    }
    self.postMessage({ type: "result", result: { build, loads: n, hashes, allEqual: hashes.every((h) => h === hashes[0]) } });
    return;
  }
  const t0 = performance.now();
  const lean = await mod.Lean.load({ gguf, tokenizer, maxCtx: 2048, backend: backend === "webgpu" ? "webgpu" : "cpu" });
  const loadMs = performance.now() - t0;
  const template = mod.ChatTemplate.fromTokenizerConfig(tokenizerCfg);
  const chat = (text) => mod.chatRender(template, [{ role: "user", content: text }], null, true);
  const result = { build, requested: backend, backend: lean.backend, loadMs, cold: [], suffix: [] };

  if (mini) {
    // Two calls that share the chat-template start, then the second on a fresh engine.
    const a = await lean.generate(chat(QUESTIONS[0]), { maxNewTokens: 48, temperature: 0 }, () => {});
    const b = await lean.generate(chat(QUESTIONS[1]), { maxNewTokens: 48, temperature: 0 }, () => {});
    lean.free();
    const l2 = await mod.Lean.load({ gguf, tokenizer, maxCtx: 2048, backend: backend === "webgpu" ? "webgpu" : "cpu" });
    const c = await l2.generate(chat(QUESTIONS[1]), { maxNewTokens: 48, temperature: 0 }, () => {});
    l2.free();
    self.postMessage({ type: "result", result: { build, a, b, c, same: b === c } });
    return;
  }

  // warm-up: shaders and pipelines
  await timeFirstToken(lean, makeText(mod, lean, 40, 99), 4);

  for (const n of phases.includes("cold") ? prompts.split(",").map(Number) : []) {
    const cold = [], suffix = [];
    let tok = 0, tokS = 0;
    for (let r = 0; r < reps; r++) {
      status(`cold ${n} rep ${r}`);
      const base = makeText(mod, lean, n, r + 1);
      tok = lean.tokenCount(base);
      const c = await timeFirstToken(lean, base, 1);
      cold.push(c.ttft);
      const longer = base + makeText(mod, lean, Math.max(8, n >> 2), r + 50);
      tokS = lean.tokenCount(longer) - tok;
      const s = await timeFirstToken(lean, longer, 1);
      suffix.push(s.ttft);
    }
    result.cold.push({ promptTokens: tok, ttftMs: median(cold), msPerTok: median(cold) / tok, all: cold });
    result.suffix.push({ prefixTokens: tok, newTokens: tokS, ttftMs: median(suffix), msPerNewTok: median(suffix) / tokS, all: suffix });
  }

  status("decode");
  if (!phases.includes("decode")) decode = 8;
  // Tokens can reach the callback in bursts (the browser resolves several
  // buffer reads at once), so the rate is the mean over the whole run, from
  // the first token to the last.
  const stamps = [];
  // sampling=1 uses the chat page's sampler (temperature, top-k, top-p, repetition penalty).
  const params = sampling ? { maxNewTokens: decode, temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.1, seed: 0 } : { maxNewTokens: decode, temperature: 0 };
  result.sampling = !!sampling;
  const reply = await lean.generate(chat("Write a long story about a lighthouse keeper."), params, () => stamps.push(performance.now()));
  result.decodeTokens = stamps.length;
  result.decodeMsPerTok = (stamps[stamps.length - 1] - stamps[0]) / (stamps.length - 1);
  result.decodeTokPerSec = 1000 / result.decodeMsPerTok;
  result.decodeHash = await sha(reply);
  // Greedy replies to three fixed questions, hashed together: equal on every
  // browser and backend when the output is token-exact.
  let exact = "";
  for (const q of QUESTIONS) {
    exact += await lean.generate(chat(q), { maxNewTokens: 48, temperature: 0 }, () => {}) + "\n";
  }
  result.exactHash = await sha(exact);
  result.exactText = exact;
  lean.free();
  if (phases.includes("fresh")) {
    // The same questions, each on a freshly loaded engine (nothing reused from
    // an earlier call). The reply must not depend on what the cache held.
    let fresh = "";
    for (const q of QUESTIONS) {
      const l2 = await mod.Lean.load({ gguf, tokenizer, maxCtx: 2048, backend: backend === "webgpu" ? "webgpu" : "cpu" });
      fresh += (await l2.generate(chat(q), { maxNewTokens: 48, temperature: 0 }, () => {})) + "\n";
      l2.free();
    }
    result.freshHash = await sha(fresh);
    result.freshEqualsReuse = fresh === exact;
  }
  self.postMessage({ type: "result", result });
}

self.onmessage = (e) => run(e.data).catch((err) => self.postMessage({ type: "error", logs: logs.slice(0, 12), message: String(err) + " :: " + String(err && err.stack ? err.stack.split("\n").slice(0, 3).join(" ; ") : "") }));
for (const e of EARLY) self.onmessage(e);
