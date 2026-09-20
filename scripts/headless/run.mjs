// General headless harness for the llm-web wasm engine: verifies that ANY
// page/model/tokenizer/template combination loads and runs in Playwright's
// bundled headless Chromium, and can benchmark decode tok/s. Not Sonos- or
// xLAM-specific — those live only under fixtures/sonos and eval/.
//
// Uses ONLY Playwright's bundled Chromium, never the user's real browser.
// Run: node scripts/headless/run.mjs [flags]
// See scripts/headless/README.md for flags and examples.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

const URL_ = args.url ?? 'http://127.0.0.1:8002/';
const GGUF = args.gguf ?? 'http://127.0.0.1:8001/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf';
const TOKENIZER = args.tokenizer ?? 'http://127.0.0.1:8001/hf/xLAM-2-3b-fc-r/tokenizer.json';
const TEMPLATE = args.template ?? 'http://127.0.0.1:8001/hf/xLAM-2-3b-fc-r/tokenizer_config.json';
const PROMPT = args.prompt ?? 'what is the weather in Paris?';
const TOOLS_MODE = args.tools ?? 'demo';
// A path to an MCP tools/list-shaped JSON file (e.g.
// fixtures/sonos/tools-12.json) — passed to the page as an actual tools
// array instead of the built-in demo/none modes; page-side canned tool
// results are a generic {ok:true} for anything not one of the two demo
// tools (see web/agent/index.html's toolCaller).
const TOOLS_FILE = args['tools-file'] ?? null;
// Debug-only A/B toggle for the naive-vs-pinned Q4 matmul kernel routing
// (see gguf.rs's force_naive_kernel / web.rs's LlmEngine.setPrefillKernel)
// — a numerical-divergence bisection aid, not a production flag.
const PREFILL_KERNEL = args['prefill-kernel'] ?? null;
const SYSTEM_PROMPT = args.system ?? null;
const MAX_NEW = parseInt(args['max-new'] ?? '64', 10);
const MAX_STEPS = parseInt(args['max-steps'] ?? '6', 10);
const EXPECT = args.expect ? new RegExp(args.expect) : null;
const BENCH_N = args.bench ? parseInt(args.bench, 10) : null;
const REPEAT = args.repeat ? parseInt(args.repeat, 10) : 1;
const TIMEOUT_LOAD = parseInt(args['timeout-load'] ?? String(10 * 60 * 1000), 10);
const TIMEOUT_RUN = parseInt(args['timeout-run'] ?? String(6 * 60 * 1000), 10);
const JSON_PATH = args.json ?? null;
const TOKENS_OUT_PATH = args['tokens-out'] ?? null;
// Debug/tooling entry point (coordinator's "priority fix", 2026-09-11):
// loads the model, then dumps the exact dieted tools JSON + system +
// rendered prefix text `--tools-file`/`--system` would use for a real
// run's prefix-KV-image cache key, without actually running a turn — see
// `web.rs::prefix_inputs`. Writes `{system, tools}` to this path, ready
// for `bin/llm-agent.rs`'s `kv-export --tools <path> --system <system>`.
const DUMP_PREFIX_PATH = args['dump-prefix'] ?? null;
// Browser-path mini-eval gate (coordinator's ask, 2026-09-11): runs a
// handful of `eval/utterances.json` cases through the real page (MCP-
// shaped canned results via index.html's toolCaller fetching
// fixtures/sonos/results/*.json), scoring each with a JS port of
// eval/README.md's rules — minimal, not a byte-exact port of
// crates/llm-wasm/src/eval.rs's scorer. Requires all selected cases to
// pass; see `runMiniEvalGate`.
const CASES_PATH = args.cases ?? null;
const ONLY_IDS = args.only ? String(args.only).split(',').map((s) => s.trim()).filter(Boolean) : null;

// This repo does not `npm install` Playwright itself; point PLAYWRIGHT_MODULE
// at a local Playwright install's `index.mjs` (e.g. a sibling project's
// `node_modules/playwright/index.mjs`), or install it locally and set this
// to `playwright`.
const PLAYWRIGHT_MODULE = process.env.PLAYWRIGHT_MODULE ?? 'playwright';
const { chromium } = await import(PLAYWRIGHT_MODULE);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, 'out');
const CONSOLE_LOG_PATH = args.out ?? path.join(OUT_DIR, 'console.log');

// Defaults to Playwright's own bundled Chromium (whatever `playwright
// install` downloaded, under its cache dir). Override with
// LLM_PLAYWRIGHT_EXECUTABLE for a specific Chrome/Chromium binary.
const EXECUTABLE_PATH = process.env.LLM_PLAYWRIGHT_EXECUTABLE ?? undefined;

const LAUNCH_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=WebGPU',
  '--use-angle=metal',
  '--ignore-gpu-blocklist',
];

const INTERESTING = [
  '[gpu-debug]',
  '[llm]',
  '[llm-worker]',
  '[llm-client]',
  'error',
  'Error',
  'Invalid',
  'Tint',
  'compilation',
];

function isInteresting(text) {
  return INTERESTING.some((needle) => text.includes(needle));
}

// A `[gpu-debug] ... failed:` line is the ROOT validation error. Everything
// after it on the same device is cascade ("is invalid due to a previous
// error") that repeats forever (once per decode step) — stop as soon as we
// see the first one instead of capturing millions of cascade lines.
const GPU_DEBUG_FAILURE = /\[gpu-debug\].*failed:/;

const allLines = [];
let gpuFailureLine = null;
let stopRequested = false;
let stopResolve = null;
const stopSignal = new Promise((res) => { stopResolve = res; });

function record(source, text) {
  const line = `[${source}] ${text}`;
  allLines.push(line);
  if (isInteresting(text)) console.log(line);
  if (!stopRequested && GPU_DEBUG_FAILURE.test(text)) {
    stopRequested = true;
    gpuFailureLine = line;
    stopResolve('gpu-debug-failure');
  }
}

async function checkWebGpuAdapter(page) {
  return page.evaluate(async () => {
    if (!('gpu' in navigator)) return { hasGpu: false };
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { hasGpu: true, adapter: null };
      const info = adapter.info || {};
      return {
        hasGpu: true,
        adapter: {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
          features: [...adapter.features],
          limits: Object.fromEntries(Object.entries(adapter.limits || {}).slice(0, 5)),
        },
      };
    } catch (err) {
      return { hasGpu: true, adapter: null, error: String(err) };
    }
  });
}

async function runOnce(page, { prompt, toolsMode, toolsArray, maxNewTokens, maxSteps, expect, timeoutLoad, timeoutRun, prefillKernel, systemPrompt }) {
  const loadStart = Date.now();
  const loadResult = await Promise.race([
    page.evaluate(
      ({ gguf, tokenizer, template }) =>
        window.__llm.load({ id: 'headless', shards: [gguf], tokenizerUrl: tokenizer, templateUrl: template }),
      { gguf: GGUF, tokenizer: TOKENIZER, template: TEMPLATE }
    ),
    stopSignal.then(() => { throw new Error('gpu-debug-failure during load'); }),
    new Promise((_, rej) => setTimeout(() => rej(new Error('load timeout')), timeoutLoad)),
  ]);
  const loadMs = Date.now() - loadStart;

  const runStart = Date.now();
  const transcript = await Promise.race([
    page.evaluate(
      ({ prompt, toolsMode, toolsArray, maxNewTokens, maxSteps, prefillKernel, systemPrompt }) =>
        window.__llm.run(prompt, {
          tools: toolsArray || toolsMode,
          maxNewTokens,
          maxSteps,
          prefillKernel: prefillKernel || undefined,
          systemPrompt: systemPrompt || undefined,
        }),
      { prompt, toolsMode, toolsArray, maxNewTokens, maxSteps, prefillKernel, systemPrompt }
    ),
    stopSignal.then(() => { throw new Error('gpu-debug-failure during run'); }),
    new Promise((_, rej) => setTimeout(() => rej(new Error('run timeout')), timeoutRun)),
  ]);
  const runMs = Date.now() - runStart;

  let pass = true;
  let expectError = null;
  if (expect && !expect.test(transcript.finalText || '')) {
    pass = false;
    expectError = `final text did not match /${expect.source}/: ${JSON.stringify(transcript.finalText)}`;
  }

  return { loadResult, loadMs, transcript, runMs, pass, expectError };
}

// ---------------------------------------------------------------------------
// Browser-path mini-eval gate — minimal JS port of eval/README.md's
// scoring rules (not a byte-exact port of crates/llm-wasm/src/eval.rs's
// native scorer). `actualCalls` is the flattened, in-order list of
// {name, args} across every `NeedTools` step of one transcript.
// ---------------------------------------------------------------------------
function isMutatingToolName(name) {
  return !(name.startsWith('get_') || name.startsWith('list_'));
}

function argsSubsetMatch(actualArgs, expectedArgs) {
  if (!expectedArgs) return true;
  for (const [k, v] of Object.entries(expectedArgs)) {
    if (JSON.stringify((actualArgs || {})[k]) !== JSON.stringify(v)) return false;
  }
  return true;
}

/** Returns { pass, reason }. */
function scoreCase(caseDef, actualCalls) {
  const { expected, accept } = caseDef;

  if (accept === 'no_mutation') {
    const mutated = actualCalls.find((c) => isMutatingToolName(c.name));
    return mutated
      ? { pass: false, reason: `unexpected mutating call ${mutated.name}` }
      : { pass: true, reason: 'no mutating call, as expected' };
  }
  if (accept === 'any_play') {
    const played = actualCalls.some((c) => c.name.startsWith('play_'));
    return played ? { pass: true, reason: 'some play_* call made' } : { pass: false, reason: 'no play_* call made' };
  }
  if (accept === 'exact') {
    if (actualCalls.length !== expected.length) {
      return { pass: false, reason: `expected exactly ${expected.length} call(s), got ${actualCalls.length}` };
    }
    for (let i = 0; i < expected.length; i++) {
      if (actualCalls[i].name !== expected[i].name || !argsSubsetMatch(actualCalls[i].args, expected[i].args)) {
        return { pass: false, reason: `call ${i} mismatch: expected ${expected[i].name}, got ${actualCalls[i]?.name}` };
      }
    }
    return { pass: true, reason: 'exact match' };
  }

  // subset (default): every expected call must appear, in order, among
  // actualCalls; extra *read* calls in between are free, but an
  // unexpected mutating call where a specific mutating call was expected
  // next is not.
  let ai = 0;
  for (const exp of expected) {
    let found = false;
    while (ai < actualCalls.length) {
      const c = actualCalls[ai++];
      if (c.name === exp.name && argsSubsetMatch(c.args, exp.args)) {
        found = true;
        break;
      }
      if (isMutatingToolName(exp.name) && isMutatingToolName(c.name)) {
        return { pass: false, reason: `expected mutating call ${exp.name}, got ${c.name} instead` };
      }
    }
    if (!found) return { pass: false, reason: `expected call ${exp.name} never happened` };
  }
  return { pass: true, reason: 'all expected calls matched, in order' };
}

async function runMiniEvalGate(page, { casesPath, onlyIds, toolsArray, maxNewTokens, maxSteps, timeoutLoad, timeoutRun, systemPrompt }) {
  const allCases = JSON.parse(await readFile(casesPath, 'utf8'));
  const cases = onlyIds ? allCases.filter((c) => onlyIds.includes(c.id)) : allCases;
  if (onlyIds && cases.length !== onlyIds.length) {
    const missing = onlyIds.filter((id) => !cases.some((c) => c.id === id));
    throw new Error(`--only named case id(s) not found in ${casesPath}: ${missing.join(', ')}`);
  }

  console.log(`\n=== mini-eval gate: ${cases.length} case(s) ===`);

  const loadStart = Date.now();
  await Promise.race([
    page.evaluate(
      ({ gguf, tokenizer, template }) =>
        window.__llm.load({ id: 'headless', shards: [gguf], tokenizerUrl: tokenizer, templateUrl: template }),
      { gguf: GGUF, tokenizer: TOKENIZER, template: TEMPLATE }
    ),
    stopSignal.then(() => { throw new Error('gpu-debug-failure during load'); }),
    new Promise((_, rej) => setTimeout(() => rej(new Error('load timeout')), timeoutLoad)),
  ]);
  console.log(`Model loaded in ${Date.now() - loadStart}ms`);

  const results = [];
  for (const caseDef of cases) {
    await page.evaluate(() => window.__llm.reset());
    const start = Date.now();
    let transcript;
    let errorMessage = null;
    try {
      transcript = await Promise.race([
        page.evaluate(
          ({ utterance, toolsArray, maxNewTokens, maxSteps, systemPrompt }) =>
            window.__llm.run(utterance, { tools: toolsArray, maxNewTokens, maxSteps, systemPrompt: systemPrompt || undefined }),
          { utterance: caseDef.utterance, toolsArray, maxNewTokens, maxSteps, systemPrompt }
        ),
        stopSignal.then(() => { throw new Error('gpu-debug-failure during run'); }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('run timeout')), timeoutRun)),
      ]);
    } catch (err) {
      errorMessage = err.message || String(err);
    }
    const wallMs = Date.now() - start;

    let scored = { pass: false, reason: errorMessage ? `run failed: ${errorMessage}` : 'no transcript' };
    let actualCalls = [];
    if (transcript) {
      actualCalls = transcript.steps.flatMap((s) => s.calls || []);
      scored = scoreCase(caseDef, actualCalls);
    }
    results.push({ id: caseDef.id, ...scored, wallMs, steps: transcript?.steps?.length ?? 0, actualCalls });

    const stepSummary = (transcript?.steps || [])
      .map((s) => `${s.calls ? s.calls.map((c) => c.name).join('+') : 'final'}(${s.prefillMs?.toFixed(0)}/${s.decodeMs?.toFixed(0)}ms)`)
      .join(' -> ');
    console.log(
      `  [${scored.pass ? 'PASS' : 'FAIL'}] ${caseDef.id} "${caseDef.utterance}" — ${wallMs}ms, ` +
      `${transcript?.steps?.length ?? 0} step(s): ${stepSummary || '(none)'} — ${scored.reason}`
    );
  }

  const passCount = results.filter((r) => r.pass).length;
  console.log(`\nmini-eval gate: ${passCount}/${results.length} passed`);
  return { results, passCount, total: results.length };
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  await mkdir(path.dirname(CONSOLE_LOG_PATH), { recursive: true });

  let toolsArray = null;
  if (TOOLS_FILE) {
    toolsArray = JSON.parse(await readFile(TOOLS_FILE, 'utf8'));
    console.log(`Loaded ${toolsArray.length} tools from ${TOOLS_FILE}`);
  }

  const browser = await chromium.launch({
    headless: true,
    executablePath: EXECUTABLE_PATH,
    args: LAUNCH_ARGS,
  });
  const page = await browser.newPage();
  // gpu-debug is gated behind ?gpudebug=1 (web/agent/index.html forwards it
  // to the worker) — the harness needs it ON to catch the root WebGPU
  // validation error instead of only cascade spam.
  const gpuDebugUrl = URL_ + (URL_.includes('?') ? '&' : '?') + 'gpudebug=1';
  await page.goto(gpuDebugUrl, { waitUntil: 'load' });

  const adapterInfo = await checkWebGpuAdapter(page);
  console.log('=== WebGPU adapter check ===');
  console.log(JSON.stringify(adapterInfo, null, 2));

  const report = {
    url: URL_,
    gguf: GGUF,
    tokenizer: TOKENIZER,
    template: TEMPLATE,
    chromiumVersion: browser.version(),
    adapter: adapterInfo,
    pass: false,
  };

  if (!adapterInfo.hasGpu || !adapterInfo.adapter) {
    console.log('WebGPU unavailable/null adapter in headless mode. Aborting.');
    report.error = 'no webgpu adapter';
    await finish(browser, report);
    return;
  }

  page.on('console', (msg) => record('page-console', msg.text()));
  page.on('pageerror', (err) => record('pageerror', err.stack || String(err)));
  page.on('worker', (w) => {
    record('worker-created', w.url());
    w.on('console', (msg) => record('worker-console', msg.text()));
  });

  if (DUMP_PREFIX_PATH) {
    try {
      console.log(`Loading model: ${GGUF}`);
      await Promise.race([
        page.evaluate(
          ({ gguf, tokenizer, template }) =>
            window.__llm.load({ id: 'headless', shards: [gguf], tokenizerUrl: tokenizer, templateUrl: template }),
          { gguf: GGUF, tokenizer: TOKENIZER, template: TEMPLATE }
        ),
        stopSignal.then(() => { throw new Error('gpu-debug-failure during load'); }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('load timeout')), TIMEOUT_LOAD)),
      ]);
      const data = await page.evaluate(
        ({ toolsArray, toolsMode, systemPrompt }) =>
          window.__llm.dumpPrefixInputs({ tools: toolsArray || toolsMode, systemPrompt }),
        { toolsArray, toolsMode: TOOLS_MODE, systemPrompt: SYSTEM_PROMPT }
      );
      await writeFile(DUMP_PREFIX_PATH, JSON.stringify({ system: data.system, tools: data.tools }, null, 2));
      console.log(`prefix key: ${data.prefixKey} (${data.prefixTokens} tokens, diet=${data.diet})`);
      console.log(`Wrote prefix inputs to ${DUMP_PREFIX_PATH}`);
      report.pass = true;
      report.dumpPrefix = data;
    } catch (err) {
      console.error('dump-prefix failed:', err);
      report.error = String(err);
    }
    await finish(browser, report);
    return;
  }

  if (CASES_PATH) {
    try {
      const gate = await runMiniEvalGate(page, {
        casesPath: CASES_PATH,
        onlyIds: ONLY_IDS,
        toolsArray,
        maxNewTokens: MAX_NEW,
        maxSteps: MAX_STEPS,
        timeoutLoad: TIMEOUT_LOAD,
        timeoutRun: TIMEOUT_RUN,
        systemPrompt: SYSTEM_PROMPT,
      });
      report.miniEvalGate = gate;
      report.pass = gate.passCount === gate.total;
    } catch (err) {
      console.error('mini-eval gate failed:', err);
      report.error = String(err);
    }
    await finish(browser, report);
    return;
  }

  try {
    console.log(`Loading model: ${GGUF}`);
    const { loadResult, loadMs, transcript, runMs, pass, expectError } = await runOnce(page, {
      prompt: PROMPT,
      toolsMode: TOOLS_MODE,
      toolsArray,
      maxNewTokens: MAX_NEW,
      maxSteps: MAX_STEPS,
      expect: EXPECT,
      timeoutLoad: TIMEOUT_LOAD,
      timeoutRun: TIMEOUT_RUN,
      prefillKernel: PREFILL_KERNEL,
      systemPrompt: SYSTEM_PROMPT,
    });

    report.loadMs = loadMs;
    report.loadInfo = loadResult;
    report.runMs = runMs;
    report.steps = transcript.steps;
    report.finalText = transcript.finalText;
    report.totalMs = transcript.totalMs;
    report.pass = pass;
    if (expectError) report.expectError = expectError;

    console.log(`Load done in ${loadMs}ms:`, JSON.stringify(loadResult));
    console.log(`Run done in ${runMs}ms. Final text: ${JSON.stringify(transcript.finalText)}`);
    for (const step of transcript.steps) {
      console.log(
        `  step ${step.index}: prompt=${step.promptTokens}tok prefill=${step.prefillMs?.toFixed(1)}ms ` +
        `decode=${step.decodeMs?.toFixed(1)}ms generated=${step.tokens}tok ` +
        `calls=${step.calls ? JSON.stringify(step.calls) : '(final)'}`
      );
    }
    if (expectError) console.log(`FAIL: ${expectError}`);

    // For a native `llm-agent run --tokens <this file>` comparison — see
    // scripts/headless/README.md and docs/BENCHMARKS.md's numerical
    // bisection sessions.
    if (TOKENS_OUT_PATH && transcript.steps[0]?.promptTokenIds) {
      await writeFile(TOKENS_OUT_PATH, JSON.stringify(transcript.steps[0].promptTokenIds));
      console.log(`Wrote step 0 prompt token ids (${transcript.steps[0].promptTokenIds.length}) to ${TOKENS_OUT_PATH}`);
    }

    if (BENCH_N) {
      report.benchRuns = [];
      for (let run = 1; run <= REPEAT; run++) {
        console.log(`\n=== bench run ${run}/${REPEAT}: tools=none, max-new=${BENCH_N} ===`);
        const benchStart = Date.now();
        const benchTranscript = await Promise.race([
          page.evaluate(
            ({ prompt, maxNewTokens }) => window.__llm.run(prompt, { tools: 'none', maxNewTokens, maxSteps: 1 }),
            { prompt: PROMPT, maxNewTokens: BENCH_N }
          ),
          stopSignal.then(() => { throw new Error('gpu-debug-failure during bench'); }),
          new Promise((_, rej) => setTimeout(() => rej(new Error('bench timeout')), TIMEOUT_RUN)),
        ]);
        const benchMs = Date.now() - benchStart;
        const decodeStep = benchTranscript.steps[benchTranscript.steps.length - 1];
        const decodeTokPerSec = decodeStep && decodeStep.decodeMs > 0
          ? (decodeStep.tokens / (decodeStep.decodeMs / 1000))
          : null;
        const benchResult = {
          run,
          maxNewTokens: BENCH_N,
          wallMs: benchMs,
          step: decodeStep,
          decodeTokPerSec,
        };
        report.benchRuns.push(benchResult);
        console.log(`bench run ${run}: ${decodeStep?.tokens} tok, prefill=${decodeStep?.prefillMs?.toFixed(1)}ms decode=${decodeStep?.decodeMs?.toFixed(1)}ms -> ${decodeTokPerSec?.toFixed(2)} tok/s (decode-phase; see ENGINE.md TODO on prefill/decode split accuracy)`);
      }
      // Keep `report.bench` as the last run for backward compatibility with existing readers.
      report.bench = report.benchRuns[report.benchRuns.length - 1];
    }
  } catch (err) {
    console.log('Run failed:', String(err && err.stack ? err.stack : err));
    report.error = String(err && err.message ? err.message : err);
    report.pass = false;
    if (stopRequested) {
      report.gpuDebugFailure = gpuFailureLine;
      console.log('=== ROOT gpu-debug failure ===');
      console.log(gpuFailureLine);
    }
  }

  await page.waitForTimeout(stopRequested ? 500 : 200);
  await finish(browser, report);
}

async function finish(browser, report) {
  await writeFile(CONSOLE_LOG_PATH, allLines.join('\n') + '\n');
  console.log(`Full console log written to ${CONSOLE_LOG_PATH}`);
  await browser.close();

  if (JSON_PATH) {
    await mkdir(path.dirname(JSON_PATH), { recursive: true });
    await writeFile(JSON_PATH, JSON.stringify(report, null, 2) + '\n');
    console.log(`JSON report written to ${JSON_PATH}`);
  }

  process.exitCode = report.pass ? 0 : 1;
}

main().catch((err) => {
  console.error('run.mjs fatal error:', err);
  process.exitCode = 1;
});
