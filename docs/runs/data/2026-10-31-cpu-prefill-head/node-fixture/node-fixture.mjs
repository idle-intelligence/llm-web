// lean's CPU backend in Node (wasm32 simd128, one thread, LeanEngineCpu of
// the diag build) against a reference fixture: prefill top-1 and top-20
// values, then the greedy continuation. Writes each case's prefill logits
// (raw f32) to <out>.
// Usage: node node-fixture.mjs <pkg dir> <gguf> <tokenizer.json> <fixture> <max ctx> <out> [skip regex]
import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [pkgDir, gguf, tokPath, fixturePath, maxCtxArg, out, skip] = process.argv.slice(2);
const mod = await import(pathToFileURL(resolve(pkgDir, 'lean.js')).href);
const wasm = await mod.default({ module_or_path: readFileSync(resolve(pkgDir, 'lean_bg.wasm')) });
const mib = () => +(wasm.memory.buffer.byteLength / 1048576).toFixed(1);
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const eng = mod.LeanEngineCpu.create();
eng.load(new Uint8Array(readFileSync(gguf)), readFileSync(tokPath, 'utf8'), Number(maxCtxArg));
console.log(JSON.stringify({ step: 'load', wasmMiB: mib() }));
writeFileSync(out, Buffer.alloc(0));
let allOk = true;
for (const c of fixture.cases) {
  if (skip && new RegExp(skip).test(c.name)) continue;
  const t0 = performance.now();
  let logits;
  try {
    logits = eng.prefillTokens(Uint32Array.from(c.input_ids));
  } catch (e) {
    console.log(JSON.stringify({ case: c.name, seq: c.input_ids.length, error: String(e), wasmMiB: mib() }));
    allOk = false;
    break;
  }
  const prefillMs = performance.now() - t0;
  appendFileSync(out, Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength));
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  let maxDiff = 0;
  c.prefill_top20.ids.forEach((id, k) => (maxDiff = Math.max(maxDiff, Math.abs(logits[id] - c.prefill_top20.values[k]))));
  const got = [best];
  for (let i = 1; i < c.greedy_continuation.length; i++) got.push(eng.decodeStepArgmax(got[i - 1]));
  const tokOk = got.every((t, i) => t === c.greedy_continuation[i]);
  allOk &&= tokOk && best === c.prefill_top20.ids[0];
  console.log(JSON.stringify({ case: c.name, seq: c.input_ids.length, top1: best === c.prefill_top20.ids[0], top20MaxDiff: +maxDiff.toExponential(3), tokens: got.length, tokensMatch: tokOk, prefillMs: Math.round(prefillMs), wasmMiB: mib() }));
}
console.log(JSON.stringify({ allOk }));
process.exit(0);
