// Manual-testing chat page for the chat API both engines expose
// (`chatGenerate`/`chatReset`: multi-turn on a kept KV cache, streamed text,
// sampling beyond greedy, stop) - not a fixture/parity harness like the
// other www/*.js pages. Inference runs in chat_worker.js; this file is UI.
//
// ?backend=webgpu|threads|single forces a backend (default: by capability),
// ?model=qwen25-0.5b|smollm2-360m|smollm2-1.7b picks local model files
// (default smollm2-360m), ?max=N the reply length cap (default 128).
//
// Every wasm/js loading URL carries `?v=ENGINE_BUILD`, bumped in the same
// commit as any wasm/model rebuild - see docs/runs/2026-09-28-lean-web.md.
const ENGINE_BUILD = "2026-10-04-release-03";

const params = new URLSearchParams(location.search);
const backend = params.get("backend") || null;
const model = params.get("model") || "smollm2-360m";
const maxNewTokens = parseInt(params.get("max") || "128", 10);

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

function appendTurn(cls, text) {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

const worker = new Worker(`./chat_worker.js?v=${ENGINE_BUILD}`, { type: "module" });
let replyEl = null;
let pending = null;

// Test hooks: __leanChatTokens counts streamed tokens; __leanChat drives
// the page from a headless script.
window.__leanChatTokens = 0;
window.__leanChat = {
  send: (text) => send(text),
  reset: () => reset(),
  stop: () => worker.postMessage({ type: "stop" }),
};

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "status") status(msg.text);
  else if (msg.type === "ready") {
    status(`ready: backend ${msg.backend}, model ${msg.model}, engine build ${msg.engineBuild}, adapter ${msg.adapter}`);
    window.__leanChatReady = msg;
    promptEl.disabled = false;
    sendEl.disabled = false;
    resetEl.disabled = false;
  } else if (msg.type === "piece") {
    replyEl.textContent += msg.text;
    logEl.scrollTop = logEl.scrollHeight;
  } else if (msg.type === "done" || msg.type === "error") {
    if (msg.type === "done") {
      window.__leanChatTokens += msg.tokens;
      status(`${msg.tokens} tokens in ${(msg.ms / 1000).toFixed(1)} s`);
    } else {
      appendTurn("turn-assistant", "[error] " + msg.message);
      if (!window.__leanChatReady) window.__leanChatError = msg.message;
    }
    sendEl.disabled = false;
    stopEl.disabled = true;
    if (pending) {
      pending({ ...msg, shown: replyEl ? replyEl.textContent.replace(/^assistant: /, "") : "" });
      pending = null;
    }
  }
};

function send(text) {
  text = (text ?? promptEl.value).trim();
  if (!text || pending) return Promise.resolve(null);
  promptEl.value = "";
  sendEl.disabled = true;
  stopEl.disabled = false;
  appendTurn("turn-user", "user: " + text);
  replyEl = appendTurn("turn-assistant", "assistant: ");
  const p = {
    maxNewTokens,
    temperature: parseFloat(document.getElementById("temperature").value) || 0,
    topK: parseInt(document.getElementById("topK").value, 10) || 0,
    topP: parseFloat(document.getElementById("topP").value) || 1,
    repPenalty: parseFloat(document.getElementById("repPenalty").value) || 1,
    seed: parseInt(document.getElementById("seed").value, 10) || 0,
  };
  const done = new Promise((resolve) => (pending = resolve));
  worker.postMessage({ type: "send", text, params: p });
  return done;
}

function reset() {
  worker.postMessage({ type: "reset" });
  logEl.textContent = "";
  window.__leanChatTokens = 0;
}

sendEl.addEventListener("click", () => send());
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
stopEl.addEventListener("click", () => worker.postMessage({ type: "stop" }));
resetEl.addEventListener("click", reset);

status(`engine build ${ENGINE_BUILD}, starting worker...`);
worker.postMessage({ type: "load", backend, model });
