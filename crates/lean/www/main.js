// Headless parity/perf harness for the lean engine's wasm build — not a
// demo page. Loads the wasm module + GGUF + tokenizer, replays the three
// fixture prompts (crates/lean/reference/fixture.json, the same file
// lean-cli's native fixture check uses) through LeanEngine.generate(), and
// reports whether the generated token ids match the native/HF-transformers
// reference plus rough prefill/decode ms/token. Served from crates/lean/ so
// relative paths reach ../pkg (wasm-pack output) and ../reference
// (fixture.json) without duplicating either.
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild — see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-09-28-1";

const HF_GGUF = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json";

const params = new URLSearchParams(location.search);
const local = params.get("local") === "1";
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
  const { default: init, LeanEngine, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  log(`[lean] engine build ${ENGINE_BUILD}, source=${local ? "local" : "huggingface"}`);

  const fixture = JSON.parse(await fetchText(`../reference/fixture.json?v=${ENGINE_BUILD}`));

  const fetchStart = performance.now();
  const [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const engine = await LeanEngine.create();
  log(`[lean] device ready: ${engine.info()}`);

  const loadStart = performance.now();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 256);
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
