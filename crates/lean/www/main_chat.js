// Manual-testing chat page for `LeanEngine.chatGenerate` (multi-turn,
// token-streaming, sampling beyond greedy) - not a fixture/parity harness
// like the other www/*.js pages. Loads SmolLM2-360M-Instruct-Q4_0 (small
// enough to iterate on quickly) from local files by default.
//
// Every wasm/js loading URL below carries `?v=ENGINE_BUILD`, bumped in the
// same commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-09-30-stream-01";

const params = new URLSearchParams(location.search);
const local = params.get("local") !== "0"; // local by default for this page

const HF_GGUF = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_0.gguf";
const HF_TOKENIZER = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json";
const HF_TOKENIZER_CFG = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer_config.json";

const ggufUrl = local ? "./model_smollm2_360m/model.gguf" : HF_GGUF;
const tokenizerUrl = local ? "./model_smollm2_360m/tokenizer.json" : HF_TOKENIZER;
const tokenizerCfgUrl = local ? "./model_smollm2_360m/tokenizer_config.json" : HF_TOKENIZER_CFG;

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const promptEl = document.getElementById("prompt");
const sendEl = document.getElementById("send");
const stopEl = document.getElementById("stop");
const resetEl = document.getElementById("reset");

function status(s) {
  statusEl.textContent = s;
  console.log("[lean-chat] " + s);
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

window.__leanChatTokens = 0; // test hook: incremented once per streamed token

async function main() {
  const { default: init, LeanEngine, AbortFlag, leanInit } = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await init(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  leanInit();
  status(`engine build ${ENGINE_BUILD}, fetching model (source=${local ? "local" : "huggingface"})...`);

  const [ggufBytes, tokenizerJson, tokenizerCfgJson] = await Promise.all([
    fetchBytes(ggufUrl),
    fetchText(tokenizerUrl),
    fetchText(tokenizerCfgUrl),
  ]);

  const engine = await LeanEngine.create();
  engine.load(ggufBytes, tokenizerJson, tokenizerCfgJson, 1024);
  status(`ready: ${engine.info()}`);

  let abortFlag = null;

  function appendTurn(cls, text) {
    const span = document.createElement("div");
    span.className = cls;
    span.textContent = text;
    logEl.appendChild(span);
    logEl.scrollTop = logEl.scrollHeight;
    return span;
  }

  async function send() {
    const text = promptEl.value.trim();
    if (!text) return;
    promptEl.value = "";
    sendEl.disabled = true;
    stopEl.disabled = false;
    appendTurn("turn-user", "user: " + text);
    const replySpan = appendTurn("turn-assistant", "assistant: ");

    const temperature = parseFloat(document.getElementById("temperature").value) || 0;
    const topK = parseInt(document.getElementById("topK").value, 10) || 0;
    const topP = parseFloat(document.getElementById("topP").value) || 1;
    const repPenalty = parseFloat(document.getElementById("repPenalty").value) || 1;
    const seed = parseInt(document.getElementById("seed").value, 10) || 0;

    abortFlag = new AbortFlag();
    try {
      await engine.chatGenerate(text, 128, temperature, topK, topP, repPenalty, seed, [], (id) => {
        window.__leanChatTokens += 1;
        replySpan.textContent += engine.decodeIds(Uint32Array.from([id]));
        logEl.scrollTop = logEl.scrollHeight;
      }, abortFlag.cloneFlag());
    } catch (e) {
      appendTurn("turn-assistant", "[error] " + (e && e.message ? e.message : e));
      console.error(e);
    }
    sendEl.disabled = false;
    stopEl.disabled = true;
    abortFlag = null;
  }

  sendEl.addEventListener("click", send);
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send();
  });
  stopEl.addEventListener("click", () => {
    if (abortFlag) abortFlag.abort();
  });
  resetEl.addEventListener("click", () => {
    engine.chatReset();
    logEl.textContent = "";
    window.__leanChatTokens = 0;
  });

  promptEl.disabled = false;
  sendEl.disabled = false;
  resetEl.disabled = false;
  window.__leanChatReady = true;
}

main().catch((e) => {
  status("ERROR: " + (e && e.message ? e.message : e));
  console.error(e);
  window.__leanChatError = String(e);
});
