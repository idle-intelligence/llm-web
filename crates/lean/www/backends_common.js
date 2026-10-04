// Shared by backends_worker.js (lean backends, in a Worker): the model
// table, best-effort model caching, the token hash, capability checks and
// the WebGPU bandwidth probe used by ?diag=1.

// Prompt ids are the fixture's "short" case ("What is the capital of
// France?", chat-templated), so every backend sees the same ids without
// depending on a tokenizer. Reference hashes: SHA-256 of the 64 greedy ids
// (u32 little-endian) from transformers, float32, weights dequantized from
// the same GGUF, plain argmax loop with no EOS stop
// (crates/lean/reference/gen_backends_hash.py).
export const MODELS = {
  "qwen25-0.5b": {
    label: "Qwen2.5-0.5B-Instruct Q4_0",
    dir: "./model/",
    gguf: "qwen2.5-0.5b-instruct-q4_0.gguf",
    hfGguf: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
    hfTokenizer: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
    hfTokenizerCfg: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json",
    // crates/lean/reference/fixture.json, case "short".
    promptIds: [
      151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13,
      151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
    ],
    referenceHash: "a454748c60e238419186f89709a2b6eee17bcf53b4253a575dfa32691dad04d5",
  },
  "smollm2-360m": {
    label: "SmolLM2-360M-Instruct Q4_0",
    dir: "./model_smollm2_360m/",
    gguf: "SmolLM2-360M-Instruct-Q4_0.gguf",
    hfGguf: "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_0.gguf",
    hfTokenizer: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json",
    hfTokenizerCfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer_config.json",
    // crates/lean/reference/fixture_llama_360m_q4_0.json, case "short".
    promptIds: [
      1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198,
      1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531, 198,
    ],
    referenceHash: "907b12597274deb6960d713d58e4f55f11c2ab037a841106a3984c067d2ca8fa",
  },
};
export const DEFAULT_MODEL = "qwen25-0.5b";
export const N_GEN = 64;
export const MAX_CTX = 256;

export function modelUrls(model, local) {
  const m = MODELS[model];
  return local
    ? { gguf: m.dir + m.gguf, tokenizer: m.dir + "tokenizer.json", tokenizerCfg: m.dir + "tokenizer_config.json" }
    : { gguf: m.hfGguf, tokenizer: m.hfTokenizer, tokenizerCfg: m.hfTokenizerCfg };
}

// Model download + Cache API storage is the shared trucs.ai module (see
// provenance header in ./model-cache.js); this keeps lean's own copy of the
// streaming-cache/length-check logic from drifting from the one other
// pages use. getModel() takes a list of files, so a single GGUF is a
// one-element list; fetchBytes() keeps its old single-url, no-progress
// signature since none of lean's ~10 pages that import it ever used a
// progress callback.
import { getModel } from "./model-cache.js?v=2026-10-04-release-05";

const MODEL_CACHE = "lean-backends-model-v2";

export async function fetchBytes(url) {
  const [bytes] = await getModel([url], {
    cache: MODEL_CACHE,
    onStatus: (m) => console.warn(`[lean-backends] ${m}`),
  });
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

export async function capabilities() {
  const caps = {
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    adapter: "none",
    hasAdapter: false,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        caps.hasAdapter = true;
        const i = adapter.info || {};
        const l = adapter.limits;
        caps.adapter =
          [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" / ") +
          ` (storage buffers/stage ${l.maxStorageBuffersPerShaderStage}, max binding ${l.maxStorageBufferBindingSize}, shader-f16 ${adapter.features.has("shader-f16")})`;
      } else {
        caps.adapter = "navigator.gpu present, no adapter";
      }
    } catch (e) {
      caps.adapter = `requestAdapter failed: ${e && e.message ? e.message : e}`;
    }
  } else {
    caps.adapter = "no navigator.gpu";
  }
  caps.threadsCapable = caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

export function median(xs) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const PROBE_SHADER = `
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let n = arrayLength(&src);
  let stride = nw.x * 256u;
  var acc = vec4<u32>(0u);
  for (var i = g.x; i < n; i = i + stride) {
    acc = acc + src[i];
  }
  out[g.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}`;

// Diagnosis only (?diag=1), never feeds an engine choice: streams a 256 MB
// storage buffer through a read + reduce kernel, and copies 128 MB buffer to
// buffer, on a separate device. Each measurement is one submit holding REPS
// repetitions, timed from submit to onSubmittedWorkDone; one warm-up, then
// the median of 5. Copy GB/s counts bytes read + written.
export async function bandwidthProbe() {
  const SIZE = 256 * 1024 * 1024;
  const COPY = 128 * 1024 * 1024;
  const GROUPS = 4096;
  const REPS = 4;
  if (!navigator.gpu) return { error: "no navigator.gpu" };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { error: "no adapter" };
  const lim = adapter.limits;
  if (lim.maxBufferSize < SIZE) return { error: `maxBufferSize ${lim.maxBufferSize} < 256 MB` };
  let chunk = SIZE;
  while (chunk > lim.maxStorageBufferBindingSize) chunk /= 2;
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: SIZE, maxStorageBufferBindingSize: chunk } });
  try {
    device.pushErrorScope("out-of-memory");
    const src = device.createBuffer({ size: SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const dst = device.createBuffer({ size: COPY, usage: GPUBufferUsage.COPY_DST });
    const out = device.createBuffer({ size: GROUPS * 256 * 4, usage: GPUBufferUsage.STORAGE });
    const oom = await device.popErrorScope();
    if (oom) return { error: `allocation failed: ${oom.message}` };
    const fill = new Uint32Array(4 * 1024 * 1024);
    for (let i = 0; i < fill.length; i++) fill[i] = (i * 2654435761) >>> 0;
    for (let off = 0; off < SIZE; off += fill.byteLength) device.queue.writeBuffer(src, off, fill);

    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: PROBE_SHADER }), entryPoint: "main" } });
    const groups = [];
    for (let off = 0; off < SIZE; off += chunk) {
      groups.push(
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: src, offset: off, size: chunk } },
            { binding: 1, resource: { buffer: out } },
          ],
        })
      );
    }
    const timeSubmit = async (record) => {
      const enc = device.createCommandEncoder();
      record(enc);
      const t = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - t;
    };
    const readOnce = (enc) => {
      for (let r = 0; r < REPS; r++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (const bg of groups) {
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(GROUPS);
        }
        pass.end();
      }
    };
    const copyOnce = (enc) => {
      for (let r = 0; r < REPS; r++) enc.copyBufferToBuffer(src, 0, dst, 0, COPY);
    };
    const measure = async (record) => {
      await timeSubmit(record);
      const ts = [];
      for (let i = 0; i < 5; i++) ts.push(await timeSubmit(record));
      return median(ts);
    };
    const emptyMs = await measure(() => {});
    const readMs = await measure(readOnce);
    const copyMs = await measure(copyOnce);
    return {
      readGBs: (SIZE * REPS) / (readMs * 1e6),
      copyGBs: (2 * COPY * REPS) / (copyMs * 1e6),
      emptySubmitMs: emptyMs,
      bindingChunkMB: chunk / (1024 * 1024),
    };
  } finally {
    device.destroy();
  }
}
