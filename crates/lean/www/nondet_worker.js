// Determinism probe for the WebGPU backend (debug only, not a product
// page). Runs the fixture "short" prompt through the same engine several
// times and hashes every result bit-exactly:
//   1. prefill logits (and the KV cache it leaves), `trials` times;
//   2. teacher-forced decode: after a fresh prefill, each reference token is
//      fed with appendTokens (one decode_layers step, full logits back), and
//      every step's logits are hashed; `trials` times.
//   3. ?ops=1: per-op output checksums of one prefill (debugPrefill), to be
//      compared across page loads so the first op whose output differs can
//      be named.
//   4. ?verify=1: reads every weight buffer back right after load and again
//      after the runs, and lists the ones whose device bytes differ from
//      the bytes uploaded.
// Any hash that differs between trials of the same input is
// non-determinism inside the engine; the report gives the first one.
import { MODELS, MAX_CTX, modelUrls, fetchBytes, fetchText } from "./backends_common.js?v=2026-10-03-main-02";
const ENGINE_BUILD = "2026-10-03-main-01";

// transformers greedy ids for the "short" prompt (the ids behind
// MODELS[...].referenceHash, crates/lean/reference/gen_backends_hash.py).
const REFERENCE_IDS = {
  "qwen25-0.5b": [
    785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 7772, 3283, 304, 279, 1879,
    553, 7042, 13, 12095, 374, 264, 3283, 448, 264, 9080, 3840, 323, 7674, 11, 323, 432, 374, 1083, 264, 3598, 6955,
    323, 4948, 4126, 315, 279, 1879, 13, 1084, 374, 3881, 369, 1181, 26277, 59924, 1741, 438, 279, 468, 3092, 301,
    21938, 11,
  ],
};

const status = (text) => {
  console.log(`[lean-nondet] ${text}`);
  self.postMessage({ type: "status", text });
};

async function hashF32(arr) {
  const f = arr instanceof Float32Array ? arr : Float32Array.from(arr);
  const d = await crypto.subtle.digest("SHA-256", f.buffer.slice(f.byteOffset, f.byteOffset + f.byteLength));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

async function hashBytes(u8) {
  const d = await crypto.subtle.digest("SHA-256", u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function run({ model, trials, steps, ops, verify, local }) {
  const m = MODELS[model];
  const urls = modelUrls(model, local);
  status(`engine build ${ENGINE_BUILD}, fetching ${m.label}...`);
  const [gguf, tok, tokCfg] = await Promise.all([fetchBytes(urls.gguf), fetchText(urls.tokenizer), fetchText(urls.tokenizerCfg)]);
  const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
  await mod.default(`../pkg/lean_bg.wasm?v=${ENGINE_BUILD}`);
  mod.leanInit();
  const engine = await mod.LeanEngine.create();
  if (verify) engine.debugRecordUploads(true);
  engine.load(gguf, tok, tokCfg, MAX_CTX);
  const prompt = m.promptIds;
  const out = { engineBuild: ENGINE_BUILD, model, info: JSON.parse(engine.info()), trials, steps };
  if (typeof engine.debugAdapter === "function") out.adapter = JSON.parse(engine.debugAdapter());

  const ref = REFERENCE_IDS[model];
  if (verify) {
    status("verifying uploads after load...");
    out.uploadsAfterLoad = JSON.parse(await engine.debugVerifyUploads());
  }

  status("prefill trials...");
  out.prefill = [];
  for (let t = 0; t < trials; t++) {
    const logits = await engine.prefillTokens(prompt, []);
    const kv = await engine.snapshotKv();
    out.prefill.push({ logits: await hashF32(logits), kv: await hashBytes(kv), argmax: argmax(logits) });
  }

  status("teacher-forced decode trials...");
  out.decode = [];
  let base = null;
  for (let t = 0; t < trials; t++) {
    await engine.prefillTokens(prompt, []);
    const hashes = [];
    const ams = [];
    const firstDiff = { step: -1, maxAbs: 0 };
    const keep = [];
    for (let s = 0; s < steps; s++) {
      const logits = await engine.appendTokens([ref[s]], []);
      hashes.push(await hashF32(logits));
      ams.push(argmax(logits));
      if (t === 0) keep.push(Float32Array.from(logits));
      else if (firstDiff.step < 0 && hashes[s] !== base.hashes[s]) {
        firstDiff.step = s;
        firstDiff.maxAbs = maxAbsDiff(logits, base.keep[s]);
      }
    }
    const rec = { hashes, argmax: ams, firstDiffVsTrial0: firstDiff };
    if (t === 0) base = { hashes, keep };
    out.decode.push(rec);
    status(`decode trial ${t}: first diff vs trial 0 at step ${firstDiff.step}`);
  }

  if (ops) {
    status("per-op checksums of one prefill...");
    out.ops = JSON.parse(await engine.debugPrefill(prompt));
  }
  if (verify) {
    status("verifying uploads after the forward calls...");
    out.uploadsAfterRun = JSON.parse(await engine.debugVerifyUploads());
  }
  return out;
}

self.onmessage = async (ev) => {
  try {
    const r = await run(ev.data);
    self.postMessage({ type: "done", result: r });
  } catch (e) {
    self.postMessage({ type: "error", text: e && e.stack ? e.stack : String(e) });
  }
};
