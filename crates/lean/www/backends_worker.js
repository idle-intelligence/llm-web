// Module Worker behind backends.html. Picks the backend by capability only (never
// by a benchmark): auto = WebGPU if an adapter is granted, else CPU threads
// if the page is cross-origin isolated with SharedArrayBuffer and more than
// one hardware thread, else single-thread CPU (WASM SIMD128). A forced backend
// that is unavailable is an error, not a silent fallback.
//
// Backend loading copies main_cpu_mt.js (pkg-mt + initThreadPool for threads,
// pkg for single thread); the prefill + greedy decode loop copies
// main_cpu.js / main.js.
//
// ?diag=1 adds measurements after the default run (which stays exactly as
// without it): engine init split, a warm prefill, the per-token decode split,
// pass-level GPU timings when timestamp-query exists, and a bandwidth probe.
import { MODELS, N_GEN, MAX_CTX, modelUrls, fetchBytes, fetchText, argmaxJs, sha256Hex, capabilities, median, bandwidthProbe } from "./backends_common.js?v=2026-10-02-backends-03";
const ENGINE_BUILD = "2026-10-02-backends-03";

function status(text) {
  self.postMessage({ type: "status", text });
}

async function createEngine(backend, caps, diag, t) {
  if (backend === "webgpu") {
    if (!caps.hasAdapter) throw new Error(`no WebGPU adapter (${caps.adapter})`);
    const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
    let t0 = performance.now();
    await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    t.wasmInitMs = performance.now() - t0;
    mod.leanInit();
    t0 = performance.now();
    const engine = diag ? await mod.LeanEngine.createDiag() : await mod.LeanEngine.create();
    t.createMs = performance.now() - t0;
    return { engine, gpu: true };
  }
  if (backend === "threads") {
    if (!caps.threadsCapable) {
      throw new Error(
        `threads need crossOriginIsolated, SharedArrayBuffer and >1 hardware thread ` +
          `(got ${caps.crossOriginIsolated}, ${caps.sharedArrayBuffer}, ${caps.hardwareConcurrency})`
      );
    }
    const mod = await import(`../pkg-mt/lean.js?v=${ENGINE_BUILD}`);
    let t0 = performance.now();
    await mod.default(`../pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
    t.wasmInitMs = performance.now() - t0;
    t0 = performance.now();
    await mod.initThreadPool(caps.hardwareConcurrency);
    t.threadPoolMs = performance.now() - t0;
    mod.leanInit();
    return { engine: mod.LeanEngineCpu.create(), gpu: false };
  }
  const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  const t0 = performance.now();
  await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  t.wasmInitMs = performance.now() - t0;
  mod.leanInit();
  return { engine: mod.LeanEngineCpu.create(), gpu: false };
}

const ms = (x) => (x === undefined || x === null || Number.isNaN(x) ? "n/a" : `${x.toFixed(1)} ms`);
const split = (d) => `${ms(d.encodeMs + d.waitMs)} (encode+submit ${ms(d.encodeMs)}, wait ${ms(d.waitMs)})`;

// Average per call of a list of diagLast().gpu results, as rows.
function gpuRows(prefix, gpus) {
  const rows = [];
  const n = gpus.length;
  const span = gpus.reduce((a, g) => a + g.spanMs, 0) / n;
  const inPass = gpus.reduce((a, g) => a + g.passSumMs, 0) / n;
  rows.push([`${prefix} gpu span`, `${ms(span)} (in passes ${ms(inPass)}, outside passes ${ms(span - inPass)})`]);
  const seg = new Map();
  for (const g of gpus) for (const [label, t, c] of g.segments) {
    const s = seg.get(label) || [0, 0];
    seg.set(label, [s[0] + t / n, s[1] + c / n]);
  }
  for (const [label, [t, c]] of seg) rows.push([`${prefix} gpu ${label}`, `${ms(t)} (${Math.round(c)} passes)`]);
  return rows;
}

async function runDiag(engine, gpu, ids, prompt, initT, decodeSplits, prefillCold) {
  const rows = [];
  const d = { init: initT };
  rows.push(["wasm init", ms(initT.wasmInitMs)]);
  if (initT.threadPoolMs !== undefined) rows.push(["thread pool init", ms(initT.threadPoolMs)]);
  if (gpu) {
    const info = JSON.parse(engine.diagInfo());
    d.info = info;
    rows.push(["adapter + device", ms(info.deviceMs)]);
    rows.push(["pipeline creation calls", ms(info.pipelinesMs)]);
    rows.push(["queue drain after create", ms(initT.drainCreateMs)]);
    rows.push(["weights parse + upload calls", ms(info.loadWeightsMs)]);
    rows.push(["tokenizer load", ms(info.loadTokenizerMs)]);
    rows.push(["queue drain after load", ms(initT.drainLoadMs)]);
    rows.push(["timestamp-query", info.timestampQuery ? "yes" : "n/a"]);
  } else {
    rows.push(["load (weights + tokenizer)", ms(initT.loadMs)]);
  }

  // Prefill cold is the default run's prefill (first dispatch of every
  // prefill pipeline); warm is the same call again, KV reset.
  status("diagnostics: warm prefill...");
  let t0 = performance.now();
  const logits = gpu ? await engine.prefillTokens(prompt, []) : engine.prefillTokens(prompt);
  const warmMs = performance.now() - t0;
  const warmSame = argmaxJs(logits) === ids[0];
  if (gpu) {
    rows.push(["prefill cold", split(prefillCold)]);
    const w = JSON.parse(engine.diagLast());
    rows.push(["prefill warm", `${split(w)}${warmSame ? "" : ", first token differs"}`]);
    d.prefillWarm = w;
  } else {
    rows.push(["prefill cold", ms(prefillCold.totalMs)]);
    rows.push(["prefill warm", `${ms(warmMs)}${warmSame ? "" : ", first token differs"}`]);
  }
  d.prefillWarmMs = warmMs;

  const stepMs = decodeSplits.map((s) => s.totalMs);
  rows.push(["decode first step", ms(stepMs[0])]);
  rows.push(["decode step median (min/max)", `${ms(median(stepMs))} (${ms(Math.min(...stepMs))} / ${ms(Math.max(...stepMs))})`]);
  if (gpu) {
    const rest = decodeSplits.slice(1);
    rows.push(["decode encode+submit median", ms(median(rest.map((s) => s.encodeMs)))]);
    rows.push(["decode wait median", ms(median(rest.map((s) => s.waitMs)))]);
    rows.push(["decode readback per step", "yes (4-byte argmax id; no GPU-only token feed in lean)"]);
  }
  d.decodeSplits = decodeSplits;

  if (gpu) {
    const info = d.info;
    const STEPS = 8;
    if (info.timestampQuery) {
      // Decode GPU time with the default one-pass-per-layer-half shape,
      // continuing from the warm prefill (same positions as the run).
      status("diagnostics: decode GPU time...");
      engine.diagSet(true, false);
      const gpus = [];
      for (let i = 0; i < STEPS; i++) {
        await engine.decodeStepArgmax(ids[i], []);
        gpus.push(JSON.parse(engine.diagLast()).gpu);
      }
      d.decodeGpu = gpus;
      const span = median(gpus.map((g) => g.spanMs));
      const inPass = median(gpus.map((g) => g.passSumMs));
      rows.push(["decode gpu span median", `${ms(span)} (in passes ${ms(inPass)})`]);

      // Op-group split: one pass per op group (diagnostics only).
      status("diagnostics: prefill and decode split by op group...");
      engine.diagSet(true, true);
      await engine.prefillTokens(prompt, []);
      const pre = JSON.parse(engine.diagLast()).gpu;
      const decs = [];
      for (let i = 0; i < STEPS; i++) {
        await engine.decodeStepArgmax(ids[i], []);
        decs.push(JSON.parse(engine.diagLast()).gpu);
      }
      engine.diagSet(false, false);
      d.prefillSplitGpu = pre;
      d.decodeSplitGpu = decs;
      rows.push(...gpuRows("prefill split", [pre]));
      rows.push(...gpuRows("decode split/step", decs));
    } else {
      rows.push(["gpu time", "n/a (no timestamp-query)"]);
    }
    const rts = [];
    for (let i = 0; i < 5; i++) rts.push(await engine.diagRoundTrip());
    rows.push(["4-byte readback round trip median", ms(median(rts))]);
    const mem = JSON.parse(engine.gpuMemoryInfo());
    rows.push(["weights on GPU", `${(mem.weightBytes / 1e6).toFixed(1)} MB`]);
    d.roundTrips = rts;
    d.gpuMemory = mem;
  }

  status("diagnostics: bandwidth probe...");
  try {
    const bw = await bandwidthProbe();
    d.bandwidth = bw;
    if (bw.error) rows.push(["bandwidth probe", `n/a (${bw.error})`]);
    else {
      rows.push(["probe read 256 MB (read + reduce)", `${bw.readGBs.toFixed(1)} GB/s`]);
      rows.push(["probe copy 128 MB (read + write)", `${bw.copyGBs.toFixed(1)} GB/s`]);
      rows.push(["probe empty submit round trip", ms(bw.emptySubmitMs)]);
    }
  } catch (e) {
    rows.push(["bandwidth probe", `n/a (${e && e.message ? e.message : e})`]);
  }
  return { rows, data: d };
}

async function run({ backend: requested, local, diag, model }) {
  const m = MODELS[model];
  const prompt = m.promptIds;
  status(`engine build ${ENGINE_BUILD}, checking capabilities...`);
  const caps = await capabilities();
  const candidates =
    requested === "auto"
      ? [caps.hasAdapter && "webgpu", caps.threadsCapable && "threads", "single"].filter(Boolean)
      : [requested];

  status(`fetching ${m.label} (${local ? "local" : "huggingface"})...`);
  const urls = modelUrls(model, local);
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([fetchBytes(urls.gguf), fetchText(urls.tokenizer), fetchText(urls.tokenizerCfg)]);

  const skipped = [];
  const initT = {};
  let engine, gpu, backend;
  for (const c of candidates) {
    try {
      status(`starting ${c} backend...`);
      ({ engine, gpu } = await createEngine(c, caps, diag, initT));
      if (diag && gpu) initT.drainCreateMs = await engine.diagRoundTrip();
      status(`loading weights on ${c} backend...`);
      const t0 = performance.now();
      engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, MAX_CTX);
      initT.loadMs = performance.now() - t0;
      if (diag && gpu) initT.drainLoadMs = await engine.diagRoundTrip();
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
  const logits = gpu ? await engine.prefillTokens(prompt, []) : engine.prefillTokens(prompt);
  const t1 = performance.now();
  const prefillCold = diag && gpu ? JSON.parse(engine.diagLast()) : { totalMs: t1 - t0 };
  const ids = [argmaxJs(logits)];
  const decodeSplits = [];
  for (let i = 1; i < N_GEN; i++) {
    const ts = diag ? performance.now() : 0;
    ids.push(gpu ? await engine.decodeStepArgmax(ids[i - 1], []) : engine.decodeStepArgmax(ids[i - 1]));
    if (diag) {
      const s = gpu ? JSON.parse(engine.diagLast()) : {};
      s.totalMs = performance.now() - ts;
      decodeSplits.push(s);
    }
  }
  const t2 = performance.now();

  const hash = await sha256Hex(ids);
  const result = {
    requested,
    backend,
    model,
    modelLabel: m.label,
    skipped,
    engineBuild: ENGINE_BUILD,
    caps,
    info: engine.info(),
    promptLen: prompt.length,
    prefillMs: t1 - t0,
    decodeMsPerTok: (t2 - t1) / (N_GEN - 1),
    ids,
    hash,
    matchesReference: hash === m.referenceHash,
    text: engine.decodeIds(Uint32Array.from(ids)),
  };
  if (diag) {
    const { rows, data } = await runDiag(engine, gpu, ids, prompt, initT, decodeSplits, prefillCold);
    result.diagRows = rows;
    result.diag = data;
  }
  return result;
}

self.onmessage = async (e) => {
  try {
    self.postMessage({ type: "done", result: await run(e.data) });
  } catch (err) {
    self.postMessage({ type: "error", text: err && err.message ? err.message : String(err) });
  }
};
