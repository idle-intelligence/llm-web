// Headless repro harness for the llm-web agent demo WebGPU failure.
// Uses ONLY Playwright's bundled Chromium, never the user's real Chrome.
// Run: node scripts/headless/repro.mjs
//
// Env vars (all optional):
//   DEMO_URL           page to load (default http://127.0.0.1:8002/)
//   PLAYWRIGHT_MODULE   path/specifier to resolve the `playwright` module
//                       from — this repo has no npm install, so we borrow
//                       Playwright's install from dusty-games-platform by
//                       default. Point NODE_PATH or this var at a different
//                       install if that repo moves.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DEMO_URL = process.env.DEMO_URL ?? 'http://127.0.0.1:8002/';
const PLAYWRIGHT_MODULE =
  process.env.PLAYWRIGHT_MODULE ??
  '/Users/tc/Code/dusty-bytes/dusty-games-platform/node_modules/playwright/index.mjs';
const { chromium } = await import(PLAYWRIGHT_MODULE);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, 'out');
const CONSOLE_LOG_PATH = path.join(OUT_DIR, 'console.log');

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
  if (isInteresting(text)) {
    console.log(line);
  }
  if (!stopRequested && GPU_DEBUG_FAILURE.test(text)) {
    stopRequested = true;
    gpuFailureLine = line;
    stopResolve('gpu-debug-failure');
  }
}

async function launchAndCheckWebGpu(headless) {
  const browser = await chromium.launch({
    headless,
    executablePath: EXECUTABLE_PATH,
    args: LAUNCH_ARGS,
  });
  const page = await browser.newPage();
  await page.goto(DEMO_URL, { waitUntil: 'load' });

  const adapterInfo = await page.evaluate(async () => {
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
          limits: Object.fromEntries(
            Object.entries(adapter.limits || {}).slice(0, 5)
          ),
        },
      };
    } catch (err) {
      return { hasGpu: true, adapter: null, error: String(err) };
    }
  });

  return { browser, page, adapterInfo };
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });

  let headless = true;
  let { browser, page, adapterInfo } = await launchAndCheckWebGpu(headless);

  console.log('=== WebGPU adapter check (headless=true) ===');
  console.log(JSON.stringify(adapterInfo, null, 2));

  if (!adapterInfo.hasGpu || !adapterInfo.adapter) {
    console.log('WebGPU unavailable/null adapter in headless mode. Retrying with headless:false ' +
      '(still Playwright-bundled Chromium only, not the user\'s real Chrome).');
    await browser.close();
    headless = false;
    ({ browser, page, adapterInfo } = await launchAndCheckWebGpu(headless));
    console.log('=== WebGPU adapter check (headless=false) ===');
    console.log(JSON.stringify(adapterInfo, null, 2));
    if (!adapterInfo.hasGpu || !adapterInfo.adapter) {
      console.log('WebGPU still unavailable. Aborting repro — cannot proceed without a GPU adapter.');
      await browser.close();
      process.exitCode = 1;
      return;
    }
  }

  page.on('console', (msg) => record('page-console', msg.text()));
  page.on('pageerror', (err) => record('pageerror', err.stack || String(err)));
  page.on('worker', (w) => {
    record('worker-created', w.url());
    w.on('console', (msg) => record('worker-console', msg.text()));
  });

  await page.click('#load-btn');

  console.log('Waiting for "ready" in page log (timeout 10 min)...');
  try {
    await Promise.race([
      page.waitForFunction(
        () => Array.isArray(window.__llmLog) && window.__llmLog.some((l) => l.includes('ready')),
        { timeout: 10 * 60 * 1000 }
      ),
      stopSignal,
    ]);
    if (stopRequested) {
      console.log('Stopped early: root gpu-debug failure occurred during load:');
      console.log(gpuFailureLine);
    } else {
      console.log('Model reported ready.');
    }
  } catch (err) {
    console.log('Timed out waiting for ready:', String(err));
    const log = await page.evaluate(() => window.__llmLog || []);
    console.log('=== __llmLog at timeout ===');
    console.log(log.join('\n'));
    await writeFile(CONSOLE_LOG_PATH, allLines.join('\n') + '\n');
    await browser.close();
    process.exitCode = 1;
    return;
  }

  if (!stopRequested) {
    await page.fill('#utterance', 'what is the weather in Paris?');
    await page.click('#run-btn');

    console.log('Waiting for done/error in page log, or the first [gpu-debug] ... failed: line (timeout 6 min)...');
    try {
      await Promise.race([
        page.waitForFunction(
          () =>
            Array.isArray(window.__llmLog) &&
            window.__llmLog.some(
              (l) => l.includes('error') || l.startsWith('=')
            ),
          { timeout: 6 * 60 * 1000 }
        ),
        stopSignal,
      ]);
    } catch (err) {
      console.log('Timed out waiting for run outcome:', String(err));
    }
  }

  if (stopRequested) {
    console.log('=== ROOT gpu-debug failure (stopped here, no cascade capture) ===');
    console.log(gpuFailureLine);
    // Give the immediately-following describeDescriptor/compilation-info
    // lines (queued microtasks) a brief moment to flush before we snapshot,
    // without waiting for the cascade that follows on every decode step.
    await page.waitForTimeout(500);
  } else {
    // Give async console messages a moment to flush before we snapshot.
    await page.waitForTimeout(3000);
  }

  const finalLog = await page.evaluate(() => window.__llmLog || []);
  console.log('=== window.__llmLog (final) ===');
  console.log(finalLog.join('\n'));

  console.log('=== ALL interesting console/page/worker lines (in order) ===');
  for (const line of allLines) {
    if (isInteresting(line)) console.log(line);
  }

  console.log(`=== FULL RAW LOG (all ${allLines.length} lines, in order) ===`);
  for (const line of allLines) console.log(line);

  await writeFile(CONSOLE_LOG_PATH, allLines.join('\n') + '\n');
  console.log(`Full console log written to ${CONSOLE_LOG_PATH}`);

  await browser.close();

  if (stopRequested || !finalLog.some((l) => l.includes('done'))) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error('repro.mjs fatal error:', err);
  process.exitCode = 1;
});
