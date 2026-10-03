// Per-token decode timing harness, parameterized by URL query, used for
// this session's browser before/after comparison
// (docs/runs/2026-09-28-lean-decode-breakdown.md): loads one model, prefills
// one fixture case, then greedily decodes a fixed number of tokens while
// recording EACH step's wall-clock ms separately (not just an average), so
// a caller can split "cold" (first decode step - includes any
// first-use pipeline/shader compile) from "warm" (steady-state median of
// the rest) from one page load, per this session's before/after protocol.
//
// Query params:
//   model = "05b" | "3b" (which local model files to load)
//   case  = "short" | "long" (which fixture case's input_ids to prefill)
//   steps = number of decode steps to record (default 16)
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-03-prefill-01";

const params = new URLSearchParams(location.search);
const modelKey = params.get("model") || "05b";
const caseName = params.get("case") || "short";
const steps = parseInt(params.get("steps") || "16", 10);

const MODELS = {
  "05b": {
    gguf: "./model/qwen2.5-0.5b-instruct-q4_0.gguf",
    tokenizer: "./model/tokenizer.json",
    tokenizerCfg: "./model/tokenizer_config.json",
    fixture: "../reference/fixture.json",
  },
  "3b": {
    gguf: "./model_qwen25_3b/qwen2.5-3b-instruct-q4_0.gguf",
    tokenizer: "./model_qwen25_3b/tokenizer.json",
    tokenizerCfg: "./model_qwen25_3b/tokenizer_config.json",
    fixture: "../reference/fixture.json",
  },
  "qwen3_1_7b": {
    gguf: "./model_qwen3_1_7b/Qwen3-1.7B-Q8_0.gguf",
    tokenizer: "./model_qwen3_1_7b/tokenizer.json",
    tokenizerCfg: "./model_qwen3_1_7b/tokenizer_config.json",
    fixture: "../reference/fixture.json",
  },
  "smollm2_360m": {
    gguf: "./model_smollm2_360m/model.gguf",
    tokenizer: "./model_smollm2_360m/tokenizer.json",
    tokenizerCfg: "./model_smollm2_360m/tokenizer_config.json",
    fixture: "../reference/fixture.json",
  },
};

const out = document.getElementById("out");
function log(line) {
  out.textContent += line + "\n";
  console.log(line);
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return await r.text();
}
async function fetchJson(url) {
  return JSON.parse(await fetchText(url));
}

async function main() {
  const cfg = MODELS[modelKey];
  if (!cfg) throw new Error(`unknown model ${modelKey}`);

  const { default: init, LeanEngine, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean] engine build ${ENGINE_BUILD}, model=${modelKey}, case=${caseName}, steps=${steps}`);

  const fixture = await fetchJson(`${cfg.fixture}?v=${ENGINE_BUILD}`);
  const c = fixture.cases.find((x) => x.name === (caseName === "long" ? "long_tools_single" : "short"));
  if (!c) throw new Error(`fixture case not found for case=${caseName}`);

  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(cfg.gguf),
    fetchText(cfg.tokenizer),
    fetchText(cfg.tokenizerCfg),
  ]);
  log(`[lean] fetched model (${ggufBytes.length} bytes)`);

  const engine = await LeanEngine.create();
  const maxCtx = c.input_ids.length + steps + 64;
  const loadStart = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, maxCtx);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  const loadMs = performance.now() - loadStart;
  log(`[lean] model loaded in ${loadMs.toFixed(0)}ms: ${engine.info()}`);

  const prefillStart = performance.now();
  const logits0 = await engine.prefillTokens(c.input_ids, []);
  const prefillMs = performance.now() - prefillStart;
  let cur = (() => {
    let best = 0;
    for (let i = 1; i < logits0.length; i++) if (logits0[i] > logits0[best]) best = i;
    return best;
  })();

  const stepMs = [];
  for (let i = 0; i < steps; i++) {
    const t0 = performance.now();
    cur = await engine.decodeStepArgmax(cur, []);
    stepMs.push(performance.now() - t0);
  }

  const cold = stepMs[0];
  const warm = stepMs.slice(1);
  const warmSorted = [...warm].sort((a, b) => a - b);
  const warmMedian = warmSorted.length > 0 ? warmSorted[Math.floor(warmSorted.length / 2)] : NaN;

  log(`[lean] prefill_ms=${prefillMs.toFixed(1)} prompt_len=${c.input_ids.length}`);
  log(`[lean] step_ms=${stepMs.map((x) => x.toFixed(1)).join(",")}`);
  log(`[lean] cold_ms=${cold.toFixed(1)} warm_median_ms=${warmMedian.toFixed(1)}`);

  window.__leanResult = {
    model: modelKey,
    case: caseName,
    promptLen: c.input_ids.length,
    loadMs,
    prefillMs,
    stepMs,
    coldMs: cold,
    warmMedianMs: warmMedian,
    engineBuild: ENGINE_BUILD,
  };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { error: String(e) };
});
