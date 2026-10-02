// Qwen2.5-3B-Instruct counterpart of main_qwen3_1_7b.js: same headless
// parity/perf harness shape (load wasm + GGUF + tokenizer, replay
// reference/fixture_qwen25_3b.json's cases through LeanEngine, check token
// ids match, report prefill/decode ms), plus one extra long-context timing
// row built from reference/fixture.json's `long_tools_single` case (2225
// prompt tokens) - fixture_qwen25_3b.json itself only has the three short
// cases (see gen_fixture_qwen25_3b.py's doc comment: a 32-token greedy
// decode over a 2225-token prompt through a 3B model on CPU is too slow to
// generate a reference for). Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct
// share the exact same tokenizer.json (verified by hash), so that case's
// `input_ids` are valid input for this model too; there is no HF-generated
// `greedy_continuation` for it against the 3B weights, so this row is
// timing-only, not a token-match check (`tokensMatch: null`).
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-02-rungs-02";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/tokenizer_config.json";

const params = new URLSearchParams(location.search);
const local = params.get("local") === "1";
const ggufUrl = local ? "./model_qwen25_3b/qwen2.5-3b-instruct-q4_0.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model_qwen25_3b/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model_qwen25_3b/tokenizer_config.json" : HF_TOKENIZER_CFG;

// How many tokens to greedily decode for the synthetic long-context timing
// row - matches this session's earlier `long_tools_single` decode-timing
// convention (24 steps), not the 32-token fixture convention used for the
// three real fixture cases below.
const LONG_CTX_DECODE_STEPS = 24;

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
  const { default: init, LeanEngine, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean] engine build ${ENGINE_BUILD}, source=${local ? "local" : "huggingface"}`);

  const fixture = await fetchJson(`../reference/fixture_qwen25_3b.json?v=${ENGINE_BUILD}`);
  const longCtxFixture = await fetchJson(`../reference/fixture.json?v=${ENGINE_BUILD}`);
  const longCtxCase = longCtxFixture.cases.find((c) => c.name === "long_tools_single");

  const fetchStart = performance.now();
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const engine = await LeanEngine.create();
  log(`[lean] device ready: ${engine.info()}`);

  const loadStart = performance.now();
  // max_ctx: longCtxCase's 2225 prompt tokens plus its decode steps, plus
  // headroom.
  const maxCtx = longCtxCase.input_ids.length + LONG_CTX_DECODE_STEPS + 64;
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, maxCtx);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  log(`[lean] model loaded in ${(performance.now() - loadStart).toFixed(0)}ms: ${engine.info()}`);

  let allMatch = true;
  const rows = [];

  for (const c of fixture.cases) {
    const ids = [];
    const genStart = performance.now();
    let firstTokenAt = null;
    const text = await engine.generate(c.prompt, 32, (id) => {
      if (firstTokenAt === null) firstTokenAt = performance.now();
      ids.push(id);
    });
    const genEnd = performance.now();
    const prefillMs = firstTokenAt !== null ? firstTokenAt - genStart : NaN;
    const decodeMsPerTok = ids.length > 0 ? (genEnd - firstTokenAt) / ids.length : NaN;
    const match = JSON.stringify(ids) === JSON.stringify(c.greedy_continuation);
    allMatch = allMatch && match;
    rows.push({ name: c.name, promptLen: c.input_ids.length, tokensMatch: match, prefillMs, decodeMsPerTok, text });
    log(
      `[lean] case=${c.name} prompt_len=${c.input_ids.length} tokens_match=${match} ` +
        `prefill_ms=${prefillMs.toFixed(1)} decode_ms_per_tok=${decodeMsPerTok.toFixed(1)} text=${JSON.stringify(text)}`
    );
  }

  // Long-context timing row (no golden continuation for this model - see
  // this file's header comment).
  {
    const ids = [];
    const genStart = performance.now();
    const logits0 = await engine.prefillTokens(longCtxCase.input_ids, []);
    const firstTokenAt = performance.now();
    let cur = (() => {
      let best = 0;
      for (let i = 1; i < logits0.length; i++) if (logits0[i] > logits0[best]) best = i;
      return best;
    })();
    for (let i = 0; i < LONG_CTX_DECODE_STEPS; i++) {
      ids.push(cur);
      cur = await engine.decodeStepArgmax(cur, []);
    }
    const genEnd = performance.now();
    const prefillMs = firstTokenAt - genStart;
    const decodeMsPerTok = (genEnd - firstTokenAt) / ids.length;
    const text = engine.decodeIds(Uint32Array.from(ids));
    rows.push({ name: "long_tools_single", promptLen: longCtxCase.input_ids.length, tokensMatch: null, prefillMs, decodeMsPerTok, text });
    log(
      `[lean] case=long_tools_single prompt_len=${longCtxCase.input_ids.length} tokens_match=n/a (no 3B reference) ` +
        `prefill_ms=${prefillMs.toFixed(1)} decode_ms_per_tok=${decodeMsPerTok.toFixed(1)} text=${JSON.stringify(text)}`
    );
  }

  log(allMatch ? "[lean] ALL FIXTURE CASES TOKEN-MATCH" : "[lean] FIXTURE MISMATCH");

  window.__leanResult = { allMatch, rows, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { allMatch: false, error: String(e) };
});
