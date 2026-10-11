// Shared by device_worker.js: capability checks and the one model this page
// verifies against. The reference is transformers' greedy continuation of
// one prompt (crates/lean-core/reference/fixture_llama_360m_q4_0.json, case
// "long": 32 tokens, decoded to text), the same reference the engine's own
// parity gates use.
export const MODEL = {
  label: "SmolLM2-360M-Instruct Q4_0",
  name: "SmolLM2-360M-Instruct",
  sizeMB: 219,
  gguf: "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_0.gguf",
  tokenizer: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json",
  tokenizerCfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer_config.json",
  prompt: "Please list, in order, the first fifteen elements of the periodic table together with their atomic number, their standard atomic weight, and one short fact about each element's most common industrial use, so that a student studying for a chemistry exam next week can memorize the table more easily.",
  referenceTokens: 32,
  referenceText: "Sure, here are the first fifteen elements of the periodic table along with their atomic number, standard atomic weight, and one short fact about each element's most common",
};
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
