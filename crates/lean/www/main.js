// Headless parity/perf harness for the lean engine's wasm build - not a
// demo page. Loads the wasm module + GGUF + tokenizer, replays the three
// fixture prompts (crates/lean/reference/fixture.json, the same file
// lean-cli's native fixture check uses) through LeanEngine.generate(), and
// reports whether the generated token ids match the native/HF-transformers
// reference plus rough prefill/decode ms/token. Served from crates/lean/ so
// relative paths reach ../pkg (wasm-pack output) and ../reference
// (fixture.json) without duplicating either.
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-09-28-4";

// Timing-only, not fixture-checked (no reference continuation exists for
// it): a ~1000-token prompt to measure prefill throughput at a size closer
// to a real chat context than the fixture's 36/54/86-token cases. Built as
// repeated distinct sentences (not one repeated string) so the tokenizer
// doesn't collapse it into a tiny number of repeated-BPE-merge tokens.
const LONG_PROMPT_SENTENCES = [
  "The history of the Roman Empire spans many centuries of political change.",
  "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
  "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
  "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
  "The printing press revolutionized the spread of information across Europe.",
  "Mount Everest is the tallest mountain above sea level on the planet.",
  "The French Revolution reshaped the political landscape of eighteenth century Europe.",
  "DNA carries the genetic instructions used in the growth of living organisms.",
  "The Great Wall of China stretches thousands of kilometers across northern China.",
  "Volcanic eruptions can reshape landscapes and affect climate for years afterward.",
];
function buildLongPrompt(targetSentences) {
  const out = [];
  for (let i = 0; i < targetSentences; i++) {
    out.push(LONG_PROMPT_SENTENCES[i % LONG_PROMPT_SENTENCES.length]);
  }
  return `Summarize the following notes in one paragraph.\n\n${out.join(" ")}`;
}

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
  // max_ctx raised from 256 to 1400: the synthetic ~1000-token timing case
  // below plus its 32-token continuation must fit alongside the fixture's
  // longest case (86 + 32).
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 1400);
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

  // Timing-only long-prompt case (~1000 tokens), not part of allMatch.
  const longPrompt = buildLongPrompt(90);
  const longPromptLen = engine.tokenCount(longPrompt);
  const longIds = [];
  const longGenStart = performance.now();
  let longFirstTokenAt = null;
  const longText = await engine.generate(longPrompt, 32, (id) => {
    if (longFirstTokenAt === null) longFirstTokenAt = performance.now();
    longIds.push(id);
  });
  const longGenEnd = performance.now();
  const longPrefillMs = longFirstTokenAt !== null ? longFirstTokenAt - longGenStart : NaN;
  const longDecodeMsPerTok = longIds.length > 0 ? (longGenEnd - longFirstTokenAt) / longIds.length : NaN;
  const longPrefillTokPerSec = longPromptLen / (longPrefillMs / 1000);
  rows.push({ name: "long_1000", promptLen: longPromptLen, match: null, prefillMs: longPrefillMs, decodeMsPerTok: longDecodeMsPerTok, text: longText });
  log(
    `[lean] case=long_1000 prompt_len=${longPromptLen} (timing-only, not fixture-checked) ` +
      `prefill_ms=${longPrefillMs.toFixed(1)} prefill_tok_per_sec=${longPrefillTokPerSec.toFixed(1)} decode_ms_per_tok=${longDecodeMsPerTok.toFixed(1)}`
  );

  window.__leanResult = { allMatch, rows, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { allMatch: false, error: String(e) };
});
