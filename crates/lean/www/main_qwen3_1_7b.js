// Qwen3-1.7B counterpart of main_qwen3.js: same headless parity/perf harness
// shape (load wasm + GGUF + tokenizer, replay reference/fixture_qwen3_1_7b.json's
// cases through LeanEngine, check token ids match, report prefill/decode
// ms) - a separate page/script rather than parametrizing main.js, since the
// two fixtures/models never run together and main.js's KV-snapshot/mask
// sections exercise engine mechanics that are architecture-independent
// (already covered by the Qwen2.5 harness) rather than anything Qwen3-
// specific, so this file only carries the token-match + prefill/decode
// timing loop.
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-04-release-03";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer_config.json";

const params = new URLSearchParams(location.search);
const local = params.get("local") === "1";
const ggufUrl = local ? "./model_qwen3_1_7b/Qwen3-1.7B-Q8_0.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model_qwen3_1_7b/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model_qwen3_1_7b/tokenizer_config.json" : HF_TOKENIZER_CFG;

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
  const { default: init, LeanEngine, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean] engine build ${ENGINE_BUILD}, source=${local ? "local" : "huggingface"}`);

  const fixture = JSON.parse(await fetchText(`../reference/fixture_qwen3_1_7b.json?v=${ENGINE_BUILD}`));

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
  // max_ctx: longest fixture case (long_tools_single, 2225 prompt tokens)
  // plus its 32-token continuation, plus headroom.
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 2300);
  ggufBytes = null; // drop this session's reference to the fetched Uint8Array/ArrayBuffer now that load() (synchronous - every tensor is already uploaded) has returned, so the JS heap can reclaim it - see web.rs's JsBytesReader doc comment.
  log(`[lean] model loaded in ${(performance.now() - loadStart).toFixed(0)}ms: ${engine.info()}`);

  let allMatch = true;
  const rows = [];
  for (const c of fixture.cases) {
    const ids = [];
    const genStart = performance.now();
    let firstTokenAt = null;
    let text;
    if (c.no_retokenize) {
      const logits0 = await engine.prefillTokens(c.input_ids, []);
      firstTokenAt = performance.now();
      let cur = (() => {
        let best = 0;
        for (let i = 1; i < logits0.length; i++) if (logits0[i] > logits0[best]) best = i;
        return best;
      })();
      for (let i = 0; i < c.greedy_continuation.length; i++) {
        ids.push(cur);
        cur = await engine.decodeStepArgmax(cur, []);
      }
      text = engine.decodeIds(Uint32Array.from(ids));
    } else {
      text = await engine.generate(c.prompt, 32, (id) => {
        if (firstTokenAt === null) firstTokenAt = performance.now();
        ids.push(id);
      });
    }
    const genEnd = performance.now();
    const prefillMs = firstTokenAt !== null ? firstTokenAt - genStart : NaN;
    const decodeMsPerTok = ids.length > 0 ? (genEnd - firstTokenAt) / ids.length : NaN;
    const match = JSON.stringify(ids) === JSON.stringify(c.greedy_continuation);
    allMatch = allMatch && match;
    rows.push({ name: c.name, promptLen: c.input_ids.length, match, prefillMs, decodeMsPerTok, text });
    log(
      `[lean] case=${c.name} prompt_len=${c.input_ids.length} tokens_match=${match} ` +
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
