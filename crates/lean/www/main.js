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
const ENGINE_BUILD = "2026-09-29-02";

// Timing-only, not fixture-checked (no reference continuation exists for
// it): a ~1500-token prompt to measure prefill throughput at a size closer
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
  let [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);
  log(`[lean] fetched model (${ggufBytes.length} bytes) in ${(performance.now() - fetchStart).toFixed(0)}ms`);

  const engine = await LeanEngine.create();
  log(`[lean] device ready: ${engine.info()}`);

  const loadStart = performance.now();
  // max_ctx raised to 2500: the fixture's `no_retokenize` long cases (the
  // Sonos MCP agent's real tool-calling inputs, up to 2354 prompt tokens -
  // see reference/gen_fixture.py's LONG_TOKEN_CASES) plus their 32-token
  // continuation must fit alongside the synthetic ~1000-token timing case
  // below.
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 2500);
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
      // Real Sonos MCP tool-calling prompts (see reference/gen_fixture.py):
      // pre-tokenized with HF's tools-aware chat template, which this
      // engine's chat_template.rs doesn't implement - feed input_ids
      // straight into prefillTokens/decodeStepArgmax instead of
      // engine.generate(prompt, ...)'s own tokenize-from-text path.
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

  // Timing-only long-prompt case (~1500 tokens, per the 2026-09-28
  // long-context perf pass's target checkpoints), not part of allMatch.
  const longPrompt = buildLongPrompt(107);
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
  rows.push({ name: "long_1500", promptLen: longPromptLen, match: null, prefillMs: longPrefillMs, decodeMsPerTok: longDecodeMsPerTok, text: longText });
  log(
    `[lean] case=long_1500 prompt_len=${longPromptLen} (timing-only, not fixture-checked) ` +
      `prefill_ms=${longPrefillMs.toFixed(1)} prefill_tok_per_sec=${longPrefillTokPerSec.toFixed(1)} decode_ms_per_tok=${longDecodeMsPerTok.toFixed(1)}`
  );

  // --- KV snapshot/restore: correctness + ~1000-token timing ---
  function argmaxJs(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  }
  async function greedyContinue(firstLogits, n) {
    const ids = [];
    let cur = argmaxJs(firstLogits);
    for (let i = 0; i < n; i++) {
      ids.push(cur);
      cur = await engine.decodeStepArgmax(cur, []);
    }
    return ids;
  }

  const kvCase = fixture.cases.reduce((a, b) => (b.input_ids.length > a.input_ids.length ? b : a));
  const allIds = kvCase.input_ids;
  const split = Math.floor(allIds.length / 2);
  const prefix = allIds.slice(0, split);
  const suffix = allIds.slice(split);
  const nNew = 12;

  const logitsA0 = await engine.prefillTokens(allIds, []);
  const tokensA = await greedyContinue(logitsA0, nNew);

  await engine.prefillTokens(prefix, []);
  const snapBytes = await engine.snapshotKv();
  await engine.prefillTokens([allIds[0]], []); // perturb the cache so restore is a real test
  engine.restoreKv(snapBytes);
  const snapBytesAfterRestore = await engine.snapshotKv();
  const restoreBytesMatch = snapBytesAfterRestore.length === snapBytes.length && snapBytesAfterRestore.every((b, i) => b === snapBytes[i]);
  const logitsB0 = await engine.appendTokens(suffix, []);
  const tokensB = await greedyContinue(logitsB0, nNew);
  const kvMatch = JSON.stringify(tokensA) === JSON.stringify(tokensB);
  log(
    `[lean] kv_snapshot_restore prefix=${prefix.length} suffix=${suffix.length} restore_bytes_match=${restoreBytesMatch} ` +
      `tokens_match=${kvMatch} tokensA=${JSON.stringify(tokensA)} tokensB=${JSON.stringify(tokensB)}`
  );
  allMatch = allMatch && kvMatch && restoreBytesMatch;

  const kvTimingIds = engine.tokenize(longPrompt);
  const kvT0 = performance.now();
  await engine.prefillTokens(kvTimingIds, []);
  const kvFullPrefillMs = performance.now() - kvT0;
  const kvT1 = performance.now();
  const kvSnap = await engine.snapshotKv();
  const kvSnapshotMs = performance.now() - kvT1;
  await engine.prefillTokens([kvTimingIds[0]], []);
  const kvT2 = performance.now();
  engine.restoreKv(kvSnap);
  const kvRestoreMs = performance.now() - kvT2;
  const kvRoundTripMs = kvSnapshotMs + kvRestoreMs;
  log(
    `[lean] kv_snapshot_timing seq=${kvTimingIds.length} bytes=${kvSnap.length} full_prefill_ms=${kvFullPrefillMs.toFixed(1)} ` +
      `snapshot_ms=${kvSnapshotMs.toFixed(1)} restore_ms=${kvRestoreMs.toFixed(1)} round_trip_ms=${kvRoundTripMs.toFixed(1)} ` +
      `speedup=${(kvFullPrefillMs / kvRoundTripMs).toFixed(1)}x`
  );

  // --- Per-step logit mask: forces an exact target string ---
  const maskPromptIds = engine.tokenize("Reply with a short JSON object.");
  // Array.from: encodeRaw returns a Uint32Array, which JSON.stringify
  // serializes as {"0":.., "1":..} (not `[..]`) - normalize to a plain
  // array so the tokens_match comparison below is meaningful.
  const targetIds = Array.from(engine.encodeRaw('{"ok":true}'));
  await engine.prefillTokens([kvTimingIds[0]], []); // reset cache state before this independent check
  const mask0 = engine.buildMaskBitset([targetIds[0]]);
  const maskLogits0 = await engine.prefillTokens(maskPromptIds, mask0);
  const gotIds = [argmaxJs(maskLogits0)];
  let cur = gotIds[0];
  const maskT0 = performance.now();
  for (let i = 1; i < targetIds.length; i++) {
    const mask = engine.buildMaskBitset([targetIds[i]]);
    cur = await engine.decodeStepArgmax(cur, mask);
    gotIds.push(cur);
  }
  const maskMsPerStep = (performance.now() - maskT0) / Math.max(1, targetIds.length - 1);
  const maskMatch = JSON.stringify(gotIds) === JSON.stringify(targetIds);
  const maskText = engine.decodeIds(gotIds);
  log(`[lean] logit_mask target_match=${maskMatch} forced_text=${JSON.stringify(maskText)} mask_ms_per_step=${maskMsPerStep.toFixed(2)}`);
  allMatch = allMatch && maskMatch;

  window.__leanResult = { allMatch, rows, engineBuild: ENGINE_BUILD, source: local ? "local" : "huggingface" };
}

main().catch((e) => {
  log("[lean] ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanResult = { allMatch: false, error: String(e) };
});
