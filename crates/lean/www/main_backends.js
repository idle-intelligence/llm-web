// Backend measurement page: runs one fixed greedy generation (Qwen2.5-0.5B-
// Instruct Q4_0, the fixture's "short" prompt, 64 tokens) on the backend the
// visitor picks (?backend=auto|webgpu|threads|single) and shows the backend used,
// what the page can see of the device, prefill ms, decode ms/token and a
// hash of the 64 generated ids. Token-exact backends give the same hash.
//
// Inference runs in backends_worker.js (a module Worker), so the threads
// backend's rayon pool never blocks the main thread. Served cross-origin
// isolated (scripts/serve_coi.py) so the threads backend is available.
//
// Every wasm/js loading URL carries `?v=ENGINE_BUILD`, bumped in the same
// commit as any wasm rebuild (backends.html's own script tag included).
const ENGINE_BUILD = "2026-10-02-backends-01";

const params = new URLSearchParams(location.search);
const backend = ["auto", "webgpu", "threads", "single"].includes(params.get("backend")) ? params.get("backend") : "auto";
const local = params.get("local") !== "0";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
for (const a of document.querySelectorAll("#backends a")) {
  if (a.getAttribute("href") === `?backend=${backend}`) a.className = "on";
}

function status(s) {
  statusEl.textContent = s;
  console.log("[lean-backends] " + s);
}

function row(k, v) {
  const tr = document.createElement("tr");
  const a = document.createElement("td");
  const b = document.createElement("td");
  a.textContent = k;
  b.textContent = v;
  tr.append(a, b);
  resultsEl.appendChild(tr);
}

const worker = new Worker(`./backends_worker.js?v=${ENGINE_BUILD}`, { type: "module" });
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") {
    status(m.text);
  } else if (m.type === "done") {
    const r = m.result;
    resultsEl.textContent = "";
    row("backend requested", r.requested);
    row("backend used", r.backend);
    if (r.skipped.length) row("backends skipped", r.skipped.join("; "));
    row("engine build", r.engineBuild);
    row("hardwareConcurrency", String(r.caps.hardwareConcurrency));
    row("crossOriginIsolated", String(r.caps.crossOriginIsolated));
    row("webgpu adapter", r.caps.adapter);
    row("prompt tokens", String(r.promptLen));
    row("prefill ms", r.prefillMs.toFixed(1));
    row("decode ms/token", `${r.decodeMsPerTok.toFixed(1)} (${r.ids.length - 1} steps)`);
    row("token hash", r.hash.slice(0, 16));
    row("same as transformers", r.matchesReference ? "yes" : "no");
    row("text", r.text);
    window.__leanBackends = r;
    status(`done: ${r.backend}, decode ${r.decodeMsPerTok.toFixed(1)} ms/token, hash ${r.hash.slice(0, 16)}`);
  } else if (m.type === "error") {
    status("ERROR: " + m.text);
    console.error(m.text);
    window.__leanBackends = { error: m.text };
  }
};
worker.onerror = (e) => {
  status("ERROR: " + (e.message || "worker failed to start"));
  console.error(e);
  window.__leanBackends = { error: String(e.message) };
};
worker.postMessage({ backend, local });
