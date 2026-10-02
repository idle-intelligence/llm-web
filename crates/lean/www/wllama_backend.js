// Reference backend for backends.html (?backend=wllama): wllama (llama.cpp
// compiled to WASM) on the same GGUF file, the same prompt token ids (fed
// with decode(ids), no tokenizer involved), greedy, 64 tokens. Runs on the
// page itself: wllama already runs llama.cpp in its own worker(s).
// Multi-thread when the page is cross-origin isolated (wllama picks its
// multi-thread build itself), with n_threads = hardwareConcurrency, the same
// count the lean threads backend uses; single thread otherwise.
//
// The Wllama wiring (constructor with the single-thread/multi-thread wasm
// path map) is copied from web/index.html; the model comes from the same
// best-effort Cache API path as the lean backends (backends_common.js)
// and is handed to wllama as a Blob instead of loadModelFromHF.
import { Wllama } from "./wllama/index.js?v=2026-10-02-mobile-01";
import { MODELS, N_GEN, MAX_CTX, modelUrls, fetchBytes, sha256Hex, capabilities, bandwidthProbe } from "./backends_common.js?v=2026-10-02-mobile-01";

const ENGINE_BUILD = "2026-10-02-mobile-01";

export async function runWllama({ local, diag, model }, status) {
  const m = MODELS[model];
  const prompt = m.promptIds;
  status(`engine build ${ENGINE_BUILD}, checking capabilities...`);
  const caps = await capabilities();
  const nThreads = caps.threadsCapable ? caps.hardwareConcurrency : 1;

  status(`fetching ${m.label} (${local ? "local" : "huggingface"})...`);
  let bytes = await fetchBytes(modelUrls(model, local).gguf);

  status("starting wllama...");
  const wllama = new Wllama(
    {
      "single-thread/wllama.wasm": `./wllama/single-thread/wllama.wasm?v=${ENGINE_BUILD}`,
      "multi-thread/wllama.wasm": `./wllama/multi-thread/wllama.wasm?v=${ENGINE_BUILD}`,
    },
    { suppressNativeLog: true }
  );
  const tLoad = performance.now();
  await wllama.loadModel([new Blob([bytes])], { n_ctx: MAX_CTX, n_threads: nThreads });
  const loadMs = performance.now() - tLoad;
  bytes = null;

  status(`running ${N_GEN} greedy tokens on wllama...`);
  await wllama.samplingInit({ temp: 0, top_k: 1 }, []);
  const t0 = performance.now();
  await wllama.decode(prompt, {});
  const ids = [(await wllama.samplingSample()).token];
  const t1 = performance.now();
  for (let i = 1; i < N_GEN; i++) {
    await wllama.decode([ids[i - 1]], {});
    ids.push((await wllama.samplingSample()).token);
  }
  const t2 = performance.now();

  const hash = await sha256Hex(ids);
  const text = new TextDecoder().decode(await wllama.detokenize(ids));
  const result = {
    requested: "wllama",
    backend: "wllama",
    model,
    modelLabel: m.label,
    skipped: [],
    engineBuild: ENGINE_BUILD,
    caps,
    wllama: { multithread: wllama.isMultithread(), nThreads: wllama.isMultithread() ? nThreads : 1, loadMs },
    promptLen: prompt.length,
    prefillMs: t1 - t0,
    decodeMsPerTok: (t2 - t1) / (N_GEN - 1),
    ids,
    hash,
    matchesReference: hash === m.referenceHash,
    text,
  };
  if (diag) {
    const rows = [["load (wllama loadModel)", `${loadMs.toFixed(1)} ms`]];
    status("diagnostics: bandwidth probe...");
    try {
      const bw = await bandwidthProbe();
      if (bw.error) rows.push(["bandwidth probe", `n/a (${bw.error})`]);
      else {
        rows.push(["probe read 256 MB (read + reduce)", `${bw.readGBs.toFixed(1)} GB/s`]);
        rows.push(["probe copy 128 MB (read + write)", `${bw.copyGBs.toFixed(1)} GB/s`]);
        rows.push(["probe empty submit round trip", `${bw.emptySubmitMs.toFixed(1)} ms`]);
      }
      result.diag = { bandwidth: bw };
    } catch (e) {
      rows.push(["bandwidth probe", `n/a (${e && e.message ? e.message : e})`]);
    }
    result.diagRows = rows;
  }
  await wllama.exit();
  return result;
}
