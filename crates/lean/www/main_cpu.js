// Headless parity/perf harness for the lean engine's CPU rung (cpu.rs via
// LeanEngineCpu in web.rs) - no WebGPU involved anywhere in this file, so it
// is meant to be run in a browser context with no GPU adapter (or WebGPU
// disabled outright) to prove the CPU rung doesn't depend on one. Same
// fixture file and match-checking convention as ../www/main.js's GPU
// harness (crates/lean/reference/fixture.json), long_tools_* cases skipped
// by default (see lean_cli.rs's `--long` flag doc comment: they're
// multi-minute on this reference CPU kernel).
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-cpu.md.
const ENGINE_BUILD = "2026-10-02-backends-03";

const params = new URLSearchParams(location.search);
const local = params.get("local") !== "0"; // local model files by default - see this dir's model/
const includeLong = params.get("long") === "1";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json";
const ggufUrl = local ? "./model/qwen2.5-0.5b-instruct-q4_0.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model/tokenizer_config.json" : HF_TOKENIZER_CFG;

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

async function main() {
  log(`[lean-cpu] navigator.gpu present: ${typeof navigator.gpu !== "undefined"} (irrelevant to this harness - LeanEngineCpu never touches it)`);

  const { default: init, LeanEngineCpu, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean-cpu] engine build ${ENGINE_BUILD}, source=${local ? "local" : "huggingface"}`);

  const fixture = JSON.parse(await fetchText(`../reference/fixture.json?v=${ENGINE_BUILD}`));

  const fetchStart = performance.now();
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean-cpu] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const engine = LeanEngineCpu.create();
  const loadStart = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 2500);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  log(`[lean-cpu] model loaded in ${(performance.now() - loadStart).toFixed(0)}ms: ${engine.info()}`);

  function argmaxJs(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  }

  let allMatch = true;
  const rows = [];
  for (const c of fixture.cases) {
    if (!includeLong && c.name.startsWith("long_tools")) {
      log(`[lean-cpu] case=${c.name} skipped (pass ?long=1 to include - multi-minute on this reference CPU kernel)`);
      continue;
    }
    const genStart = performance.now();
    const logits0 = engine.prefillTokens(c.input_ids);
    const firstTokenAt = performance.now();
    const ids = [argmaxJs(logits0)];
    let cur = ids[0];
    for (let i = 1; i < c.greedy_continuation.length; i++) {
      cur = engine.decodeStepArgmax(cur);
      ids.push(cur);
    }
    const genEnd = performance.now();
    const text = engine.decodeIds(Uint32Array.from(ids));
    const prefillMs = firstTokenAt - genStart;
    const decodeMsPerTok = (genEnd - firstTokenAt) / Math.max(1, ids.length - 1);
    const match = JSON.stringify(ids) === JSON.stringify(c.greedy_continuation);
    allMatch = allMatch && match;
    rows.push({ name: c.name, promptLen: c.input_ids.length, match, prefillMs, decodeMsPerTok, text });
    log(
      `[lean-cpu] case=${c.name} prompt_len=${c.input_ids.length} tokens_match=${match} ` +
        `prefill_ms=${prefillMs.toFixed(1)} decode_ms_per_tok=${decodeMsPerTok.toFixed(1)} text=${JSON.stringify(text)}`
    );
  }
  log(allMatch ? "[lean-cpu] ALL FIXTURE CASES TOKEN-MATCH" : "[lean-cpu] FIXTURE MISMATCH");

  window.__leanCpuResult = { allMatch, rows, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean-cpu] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanCpuResult = { allMatch: false, error: String(e) };
});
