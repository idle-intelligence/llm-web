// Back-compat shim: the original xLAM-specific repro is now the general
// harness's default invocation. See scripts/headless/run.mjs / README.md.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const child = spawn(process.execPath, [path.join(__dirname, 'run.mjs'), ...process.argv.slice(2)], {
  stdio: 'inherit',
  env: process.env,
});
child.on('exit', (code) => { process.exitCode = code ?? 1; });
