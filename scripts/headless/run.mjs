// General headless harness for the llm-web wasm engine: verifies that ANY
// page/model/tokenizer/template combination loads and runs in Playwright's
// bundled headless Chromium, and can benchmark decode tok/s. Not Sonos- or
// xLAM-specific — those live only under fixtures/sonos and eval/.
//
// Uses ONLY Playwright's bundled Chromium, never the user's real browser.
// Run: node scripts/headless/run.mjs [flags]
// See scripts/headless/README.md for flags and examples.
import { mkdir, writeFile } from 'node:fs/promises';
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
const MAX_NEW = parseInt(args['max-new'] ?? '64', 10);
const MAX_STEPS = parseInt(args['max-steps'] ?? '6', 10);
const EXPECT = args.expect ? new RegExp(args.expect) : null;
const BENCH_N = args.bench ? parseInt(args.bench, 10) : null;
const TIMEOUT_LOAD = parseInt(args['timeout-load'] ?? String(10 * 60 * 1000), 10);
const TIMEOUT_RUN = parseInt(args['timeout-run'] ?? String(6 * 60 * 1000), 10);
const JSON_PATH = args.json ?? null;

const PLAYWRIGHT_MODULE =
  process.env.PLAYWRIGHT_MODULE ??
  '/Users/tc/Code/dusty-bytes/dusty-games-platform/node_modules/playwright/index.mjs';
const { chromium } = await import(PLAYWRIGHT_MODULE);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, 'out');
const CONSOLE_LOG_PATH = args.out ?? path.join(OUT_DIR, 'console.log');

const EXECUTABLE_PATH =
  '/Users/tc/Library/Caches/ms-playwright/chromium-1229/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

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

async function runOnce(page, { prompt, toolsMode, maxNewTokens, maxSteps, expect, timeoutLoad, timeoutRun }) {
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
      ({ prompt, toolsMode, maxNewTokens, maxSteps }) =>
        window.__llm.run(prompt, { tools: toolsMode, maxNewTokens, maxSteps }),
      { prompt, toolsMode, maxNewTokens, maxSteps }
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

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  await mkdir(path.dirname(CONSOLE_LOG_PATH), { recursive: true });

  const browser = await chromium.launch({
    headless: true,
    executablePath: EXECUTABLE_PATH,
    args: LAUNCH_ARGS,
  });
  const page = await browser.newPage();
  await page.goto(URL_, { waitUntil: 'load' });

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

  try {
    console.log(`Loading model: ${GGUF}`);
    const { loadResult, loadMs, transcript, runMs, pass, expectError } = await runOnce(page, {
      prompt: PROMPT,
      toolsMode: TOOLS_MODE,
      maxNewTokens: MAX_NEW,
      maxSteps: MAX_STEPS,
      expect: EXPECT,
      timeoutLoad: TIMEOUT_LOAD,
      timeoutRun: TIMEOUT_RUN,
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
        `decode=${step.decodeMs?.toFixed(1)}ms generated=${step.tokens}tok`
      );
    }
    if (expectError) console.log(`FAIL: ${expectError}`);

    if (BENCH_N) {
      console.log(`\n=== bench: tools=none, max-new=${BENCH_N} ===`);
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
      report.bench = {
        maxNewTokens: BENCH_N,
        wallMs: benchMs,
        step: decodeStep,
        decodeTokPerSec,
      };
      console.log(`bench: ${decodeStep?.tokens} tok in decode=${decodeStep?.decodeMs?.toFixed(1)}ms -> ${decodeTokPerSec?.toFixed(2)} tok/s (decode-phase; see ENGINE.md TODO on prefill/decode split accuracy)`);
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
