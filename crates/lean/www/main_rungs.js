// Rung measurement page: runs one fixed greedy generation (Qwen2.5-0.5B-
// Instruct Q4_0, the fixture's "short" prompt, 64 tokens) on the rung the
// visitor picks (?rung=auto|webgpu|threads|single) and shows the rung used,
// what the page can see of the device, prefill ms, decode ms/token and a
// hash of the 64 generated ids. Token-exact rungs give the same hash.
//
// Inference runs in rungs_worker.js (a module Worker), so the threads
// rung's rayon pool never blocks the main thread. Served cross-origin
// isolated (scripts/serve_coi.py) so the threads rung is available.
//
// Every wasm/js loading URL carries `?v=ENGINE_BUILD`, bumped in the same
// commit as any wasm rebuild (rungs.html's own script tag included).
const ENGINE_BUILD = "2026-10-02-rungs-01";

const params = new URLSearchParams(location.search);
const rung = ["auto", "webgpu", "threads", "single"].includes(params.get("rung")) ? params.get("rung") : "auto";
const local = params.get("local") !== "0";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
for (const a of document.querySelectorAll("#rungs a")) {
  if (a.getAttribute("href") === `?rung=${rung}`) a.className = "on";
}

function status(s) {
  statusEl.textContent = s;
  console.log("[lean-rungs] " + s);
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

const worker = new Worker(`./rungs_worker.js?v=${ENGINE_BUILD}`, { type: "module" });
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") {
    status(m.text);
  } else if (m.type === "done") {
    const r = m.result;
    resultsEl.textContent = "";
    row("rung requested", r.requested);
    row("rung used", r.rung);
    if (r.skipped.length) row("rungs skipped", r.skipped.join("; "));
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
    window.__leanRungs = r;
    status(`done: ${r.rung}, decode ${r.decodeMsPerTok.toFixed(1)} ms/token, hash ${r.hash.slice(0, 16)}`);
  } else if (m.type === "error") {
    status("ERROR: " + m.text);
    console.error(m.text);
    window.__leanRungs = { error: m.text };
  }
};
worker.onerror = (e) => {
  status("ERROR: " + (e.message || "worker failed to start"));
  console.error(e);
  window.__leanRungs = { error: String(e.message) };
};
worker.postMessage({ rung, local });
