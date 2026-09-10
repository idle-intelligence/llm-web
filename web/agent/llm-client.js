/**
 * Page-side wrapper around worker.js, matching trucs.ai's client API
 * (`load`, `run`, `reset`) so worker.js is a drop-in replacement for the
 * sonos stub there. Owns the Worker, correlates run ids, and routes
 * `callTool` messages to a caller-supplied `toolCaller(name, args) ->
 * Promise<result>` function.
 */
export class LlmClient {
  /**
   * @param {object} opts
   * @param {(name: string, args: object) => Promise<any>} opts.toolCaller
   * @param {(evt: object) => void} [opts.onEvent] - raw worker messages, for a log view
   * @param {boolean} [opts.gpuDebug] - wrap GPUDevice calls in error scopes (slower; ?gpudebug=1 on the worker URL)
   */
  constructor({ toolCaller, onEvent, gpuDebug = false } = {}) {
    const workerUrl = new URL('./worker.js', import.meta.url);
    if (gpuDebug) workerUrl.searchParams.set('gpudebug', '1');
    this.worker = new Worker(workerUrl, { type: 'module' });
    this.toolCaller = toolCaller;
    this.onEvent = onEvent || (() => {});
    this.nextRunId = 0;
    this.pendingRuns = new Map(); // id -> {resolve, reject, onStep}
    this.readyPromise = null;
    this.readyResolve = null;

    this.worker.onmessage = (e) => this._handleMessage(e.data);
    this.worker.onerror = (e) => {
      console.error('[llm-client] worker error:', e);
    };
  }

  /** @param {{id:string, shards:string[], tokenizerUrl:string, templateUrl:string}} model */
  load(model) {
    this.readyPromise = new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.worker.postMessage({ type: 'load', model });
    return this.readyPromise;
  }

  /**
   * @param {string} utterance
   * @param {object[]} tools MCP tool objects ({name, description, inputSchema})
   * @param {object} [opts]
   * @param {(step: object) => void} [onStep]
   * @returns {Promise<{steps: object[], finalText: string, totalMs: number}>}
   */
  run(utterance, tools, opts, onStep) {
    const id = String(this.nextRunId++);
    return new Promise((resolve, reject) => {
      this.pendingRuns.set(id, { resolve, reject, onStep: onStep || (() => {}) });
      this.worker.postMessage({ type: 'run', id, utterance, tools, opts: opts || {} });
    });
  }

  reset() {
    this.worker.postMessage({ type: 'reset' });
  }

  _handleMessage(msg) {
    this.onEvent(msg);
    switch (msg.type) {
      case 'progress':
      case 'status':
        break; // surfaced via onEvent only
      case 'ready':
        if (this.readyResolve) {
          this.readyResolve(msg.info);
          this.readyResolve = null;
        }
        break;
      case 'step': {
        const run = this.pendingRuns.get(msg.id);
        if (run) run.onStep(msg.step);
        break;
      }
      case 'callTool': {
        const { id, callId, name, args } = msg;
        Promise.resolve(this.toolCaller ? this.toolCaller(name, args) : Promise.reject(new Error('no toolCaller configured')))
          .then((result) => this.worker.postMessage({ type: 'toolResult', id, callId, result }))
          .catch((err) => this.worker.postMessage({ type: 'toolResult', id, callId, error: String(err && err.message ? err.message : err) }));
        break;
      }
      case 'done': {
        const run = this.pendingRuns.get(msg.id);
        if (run) {
          run.resolve(msg.transcript);
          this.pendingRuns.delete(msg.id);
        }
        break;
      }
      case 'error': {
        if (msg.id && this.pendingRuns.has(msg.id)) {
          this.pendingRuns.get(msg.id).reject(new Error(msg.message));
          this.pendingRuns.delete(msg.id);
        } else if (this.readyReject) {
          this.readyReject(new Error(msg.message));
          this.readyReject = null;
        } else {
          console.error('[llm-client] worker error:', msg.message);
        }
        break;
      }
      default:
        break;
    }
  }
}
