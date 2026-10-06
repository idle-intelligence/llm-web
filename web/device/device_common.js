// Shared by device_worker.js: capability checks, the one model this page
// verifies against, and the token hash (the same reference hash for
// SmolLM2-360M-Instruct Q4_0 the lean engine's own parity gates use).
export const MODEL = {
  label: "SmolLM2-360M-Instruct Q4_0",
  name: "SmolLM2-360M-Instruct",
  sizeMB: 219,
  gguf: "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_0.gguf",
  tokenizer: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json",
  tokenizerCfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer_config.json",
  // crates/lean/reference/fixture_llama_360m_q4_0.json, case "short".
  promptIds: [
    1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198,
    1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531, 198,
  ],
  referenceHash: "907b12597274deb6960d713d58e4f55f11c2ab037a841106a3984c067d2ca8fa",
};
export const N_GEN = 64;
export const MAX_CTX = 256;

import { getModel } from "../lib/model-cache.js";

const MODEL_CACHE = "lean-device-model-v1";

export async function fetchBytes(url, onProgress) {
  const [bytes] = await getModel([url], { cache: MODEL_CACHE, onProgress });
  return bytes;
}

export async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return await r.text();
}

export function argmaxJs(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

export async function sha256Hex(ids) {
  const buf = new ArrayBuffer(ids.length * 4);
  const view = new DataView(buf);
  ids.forEach((id, i) => view.setUint32(i * 4, id, true));
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

// mtBuilt: whether this deployment's build included pkg-mt (scripts/
// build.sh BUILD_THREADS=1; GitHub Pages never does). Passed in by the
// caller (device_worker.js's own build-time MT_BUILT constant) rather than
// probed over the network, so a threads-capable browser that just lacks
// the build gets "not built" instead of a pkg-mt fetch that 404s.
export async function capabilities(mtBuilt) {
  const caps = {
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    adapter: "none",
    hasAdapter: false,
    mtBuilt: !!mtBuilt,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        caps.hasAdapter = true;
        const i = adapter.info || {};
        caps.adapter = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" / ") || "WebGPU adapter";
      } else {
        caps.adapter = "navigator.gpu present, no adapter granted";
      }
    } catch (e) {
      caps.adapter = `requestAdapter failed: ${e && e.message ? e.message : e}`;
    }
  } else {
    caps.adapter = "no navigator.gpu (WebGPU not available in this browser)";
  }
  caps.threadsCapable = caps.mtBuilt && caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

// Backends this device can run, in display order, plus a reason string for
// each one that's missing (keyed by backend id) for the page to show next
// to its disabled button.
export function availableBackends(caps) {
  const available = [];
  const reasons = {};
  if (caps.hasAdapter) {
    available.push("webgpu");
  } else {
    reasons.webgpu = caps.adapter;
  }
  if (caps.threadsCapable) {
    available.push("threads");
  } else {
    reasons.threads = !caps.mtBuilt
      ? "not built"
      : !caps.crossOriginIsolated
      ? "not cross-origin isolated"
      : !caps.sharedArrayBuffer
      ? "no SharedArrayBuffer"
      : "one hardware thread";
  }
  available.push("single");
  return { available, reasons };
}

export function median(xs) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
