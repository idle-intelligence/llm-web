// Backend measurement page: runs one fixed greedy generation (the chosen
// model's fixture "short" prompt, 64 tokens) on the backend the visitor
// picks (?backend=auto|webgpu|threads|single|wllama, ?model=qwen25-0.5b|
// smollm2-360m) and shows the backend used, what the page can see of the
// device, prefill ms, decode ms/token and a hash of the 64 generated ids.
// Token-exact backends give the transformers reference hash.
//
// The lean backends run in backends_worker.js (a module Worker), so the
// threads backend's rayon pool never blocks the main thread; wllama, the
// llama.cpp reference, runs from wllama_backend.js (it has its own
// workers). Served cross-origin isolated (scripts/serve_coi.py) so the
// threaded builds are available. ?diag=1 adds a second table of
// diagnostics under the results; the default output is unchanged.
//
// Every wasm/js loading URL carries `?v=ENGINE_BUILD`, bumped in the same
// commit as any wasm rebuild (backends.html's own script tag included).
import { MODELS, DEFAULT_MODEL } from "./backends_common.js?v=2026-10-03-prefill-02";

const ENGINE_BUILD = "2026-10-03-prefill-02";

const params = new URLSearchParams(location.search);
const backend = ["auto", "webgpu", "threads", "single", "wllama"].includes(params.get("backend")) ? params.get("backend") : "auto";
const model = MODELS[params.get("model")] ? params.get("model") : DEFAULT_MODEL;
const local = params.get("local") !== "0";
const diag = params.get("diag") === "1";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const diagEl = document.getElementById("diag");
for (const a of document.querySelectorAll("#backends a, #models a")) {
  const own = new URLSearchParams(a.getAttribute("href").slice(1));
  const merged = new URLSearchParams(location.search);
  for (const [k, v] of own) merged.set(k, v);
  if ([...own].every(([k, v]) => (k === "backend" ? backend : model) === v)) a.className = "on";
  a.setAttribute("href", "?" + merged.toString());
}

function status(s) {
  statusEl.textContent = s;
  console.log("[lean-backends] " + s);
}

function row(table, k, v) {
  const tr = document.createElement("tr");
  const a = document.createElement("td");
  const b = document.createElement("td");
  a.textContent = k;
  b.textContent = v;
  tr.append(a, b);
  table.appendChild(tr);
}

function show(r) {
  resultsEl.textContent = "";
  row(resultsEl, "backend requested", r.requested);
  row(resultsEl, "backend used", r.backend);
  if (r.skipped.length) row(resultsEl, "backends skipped", r.skipped.join("; "));
  row(resultsEl, "engine build", r.engineBuild);
  row(resultsEl, "hardwareConcurrency", String(r.caps.hardwareConcurrency));
  row(resultsEl, "crossOriginIsolated", String(r.caps.crossOriginIsolated));
  row(resultsEl, "webgpu adapter", r.caps.adapter);
  if (r.wllama) row(resultsEl, "wllama threads", `${r.wllama.nThreads} (${r.wllama.multithread ? "multi-thread" : "single-thread"} build)`);
  row(resultsEl, "prompt tokens", String(r.promptLen));
  row(resultsEl, "prefill ms", r.prefillMs.toFixed(1));
  row(resultsEl, "decode ms/token", `${r.decodeMsPerTok.toFixed(1)} (${r.ids.length - 1} steps)`);
  row(resultsEl, "token hash", r.hash.slice(0, 16));
  row(resultsEl, "same as transformers", r.matchesReference ? "yes" : "no");
  row(resultsEl, "text", r.text);
  if (r.diagRows) {
    diagEl.textContent = "";
    for (const [k, v] of r.diagRows) row(diagEl, k, v);
  }
  window.__leanBackends = r;
  status(`done: ${r.backend}, decode ${r.decodeMsPerTok.toFixed(1)} ms/token, hash ${r.hash.slice(0, 16)}`);
}

function fail(text, e) {
  status("ERROR: " + text);
  console.error(e || text);
  window.__leanBackends = { error: text };
}

if (backend === "wllama") {
  import(`./wllama_backend.js?v=${ENGINE_BUILD}`)
    .then((m) => m.runWllama({ local, diag, model }, status))
    .then(show)
    .catch((e) => fail(e && e.message ? e.message : String(e), e));
} else {
  const worker = new Worker(`./backends_worker.js?v=${ENGINE_BUILD}`, { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "status") status(m.text);
    else if (m.type === "done") show(m.result);
    else if (m.type === "error") fail(m.text);
  };
  worker.onerror = (e) => fail(e.message || "worker failed to start", e);
  worker.postMessage({ backend, local, diag, model });
}
