// Headless parity/perf harness for the lean engine's CPU rung, exercising
// the capability-picked threads-vs-single-thread ladder rung (the project's ladder:
// WebGPU, then CPU threads, then single CPU thread - this page never
// touches WebGPU). Same fixture file and match-checking convention as
// ../www/main_cpu.js's single-thread harness (crates/lean/reference/fixture.json).
//
// Rung selection is capability-detection only, never a runtime benchmark or
// per-device tuning (see this crate's CPU-fallback plan's correction,
// 2026-09-28): threads rung requires `self.crossOriginIsolated`,
// `typeof SharedArrayBuffer !== "undefined"`, and
// `navigator.hardwareConcurrency > 1` - all three are fixed capability
// reads, not measurements. If any is false, this page falls back to the
// plain single-thread `pkg` build, same as `main_cpu.js`.
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-29-lean-threads.md.
const ENGINE_BUILD = "2026-09-29-cpu-mt-02";

const params = new URLSearchParams(location.search);
const local = params.get("local") !== "0";
const includeLong = params.get("long") === "1";
const forceSingle = params.get("threads") === "0"; // test-only override: force the single-thread rung even when threads capability is present

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
  const coi = self.crossOriginIsolated === true;
  const hasSab = typeof SharedArrayBuffer !== "undefined";
  const hwConcurrency = navigator.hardwareConcurrency || 1;
  const threadsCapable = coi && hasSab && hwConcurrency > 1 && !forceSingle;
  log(
    `[lean-cpu-mt] crossOriginIsolated=${coi} SharedArrayBuffer=${hasSab} hardwareConcurrency=${hwConcurrency} ` +
      `forceSingle=${forceSingle} -> rung=${threadsCapable ? "threads" : "single-thread"}`
  );

  let engine;
  let rung;
  if (threadsCapable) {
    try {
      const mod = await import(`../pkg-mt/lean.js?v=${ENGINE_BUILD}`);
      await mod.default(`../pkg-mt/lean_bg.wasm?v=${ENGINE_BUILD}`);
      await mod.initThreadPool(hwConcurrency);
      mod.leanInit();
      engine = mod.LeanEngineCpu.create();
      rung = "threads";
      log(`[lean-cpu-mt] threads rung initialized: pool size ${hwConcurrency}`);
    } catch (e) {
      log(`[lean-cpu-mt] threads rung init FAILED, falling back to single-thread: ${e && e.message ? e.message : e}`);
      threadsCapable && log(`[lean-cpu-mt] (this is the COEP/worker-init failure mode to watch for - see cpu-fallback-plan step 5)`);
      const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
      await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
      mod.leanInit();
      engine = mod.LeanEngineCpu.create();
      rung = "single-thread (fallback after threads init failure)";
    }
  } else {
    const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
    await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
    mod.leanInit();
    engine = mod.LeanEngineCpu.create();
    rung = "single-thread";
  }
  log(`[lean-cpu-mt] engine build ${ENGINE_BUILD}, rung=${rung}, source=${local ? "local" : "huggingface"}`);

  const fixture = JSON.parse(await fetchText(`../reference/fixture.json?v=${ENGINE_BUILD}`));

  const fetchStart = performance.now();
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean-cpu-mt] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const loadStart = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 2500);
  ggufBytes = null;
  log(`[lean-cpu-mt] model loaded in ${(performance.now() - loadStart).toFixed(0)}ms: ${engine.info()}`);

  function argmaxJs(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  }

  let allMatch = true;
  const rows = [];
  for (const c of fixture.cases) {
    if (!includeLong && c.name.startsWith("long_tools")) {
      log(`[lean-cpu-mt] case=${c.name} skipped (pass ?long=1 to include)`);
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
      `[lean-cpu-mt] case=${c.name} prompt_len=${c.input_ids.length} tokens_match=${match} ` +
        `prefill_ms=${prefillMs.toFixed(1)} decode_ms_per_tok=${decodeMsPerTok.toFixed(1)} text=${JSON.stringify(text)}`
    );
  }
  log(allMatch ? "[lean-cpu-mt] ALL FIXTURE CASES TOKEN-MATCH" : "[lean-cpu-mt] FIXTURE MISMATCH");

  window.__leanCpuMtResult = { allMatch, rung, rows, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean-cpu-mt] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanCpuMtResult = { allMatch: false, error: String(e) };
});
