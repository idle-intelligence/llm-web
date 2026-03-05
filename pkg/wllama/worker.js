/**
 * Module code will be copied into worker.
 *
 * Messages between main <==> worker:
 *
 * From main thread to worker:
 * - Send direction: { verb, args, callbackId }
 * - Result direction: { callbackId, result } or { callbackId, err }
 *
 * Signal from worker to main:
 * - Unidirection: { verb, args }
 */
import { createWorker, isSafariMobile } from './utils.js';
import { LLAMA_CPP_WORKER_CODE, WLLAMA_MULTI_THREAD_CODE, WLLAMA_MULTI_THREAD_WORKER_CODE, WLLAMA_SINGLE_THREAD_CODE, } from './workers-code/generated.js';
export class ProxyToWorker {
    logger;
    suppressNativeLog;
    taskQueue = [];
    taskId = 1;
    resultQueue = [];
    busy = false; // is the work loop is running?
    worker;
    pathConfig;
    multiThread;
    nbThread;
    constructor(pathConfig, nbThread = 1, suppressNativeLog, logger) {
        this.pathConfig = pathConfig;
        this.nbThread = nbThread;
        this.multiThread = nbThread > 1;
        this.logger = logger;
        this.suppressNativeLog = suppressNativeLog;
    }
    async moduleInit(ggufFiles) {
        if (!this.pathConfig['wllama.wasm']) {
            throw new Error('"single-thread/wllama.wasm" is missing from pathConfig');
        }
        let moduleCode = this.multiThread
            ? WLLAMA_MULTI_THREAD_CODE
            : WLLAMA_SINGLE_THREAD_CODE;
        moduleCode = moduleCode.replace('var Module', 'var ___Module');
        const runOptions = {
            pathConfig: this.pathConfig,
            nbThread: this.nbThread,
        };
        const completeCode = [
            this.multiThread
                ? `const WLLAMA_MULTI_THREAD_WORKER_CODE = ${JSON.stringify(WLLAMA_MULTI_THREAD_WORKER_CODE)};`
                : '// single-thread build',
            `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
            `function wModuleInit() { ${moduleCode}; return Module; }`,
            LLAMA_CPP_WORKER_CODE,
        ].join(';\n\n');
        this.worker = createWorker(completeCode);
        this.worker.onmessage = this.onRecvMsg.bind(this);
        this.worker.onerror = this.logger.error;
        const res = await this.pushTask({
            verb: 'module.init',
            args: [new Blob([moduleCode], { type: 'text/javascript' })],
            callbackId: this.taskId++,
        });
        // allocate all files
        const nativeFiles = [];
        for (const file of ggufFiles) {
            const id = await this.fileAlloc(file.name, file.blob.size);
            nativeFiles.push({ id, ...file });
        }
        // stream files
        await Promise.all(nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
        }));
        return res;
    }
    async wllamaStart() {
        const result = await this.pushTask({
            verb: 'wllama.start',
            args: [],
            callbackId: this.taskId++,
        });
        const parsedResult = this.parseResult(result);
        return parsedResult;
    }
    async wllamaAction(name, body) {
        const result = await this.pushTask({
            verb: 'wllama.action',
            args: [name, JSON.stringify(body)],
            callbackId: this.taskId++,
        });
        const parsedResult = this.parseResult(result);
        return parsedResult;
    }
    async wllamaExit() {
        if (this.worker) {
            const result = await this.pushTask({
                verb: 'wllama.exit',
                args: [],
                callbackId: this.taskId++,
            });
            this.parseResult(result); // only check for exceptions
            this.worker.terminate();
        }
    }
    async wllamaDebug() {
        const result = await this.pushTask({
            verb: 'wllama.debug',
            args: [],
            callbackId: this.taskId++,
        });
        return JSON.parse(result);
    }
    ///////////////////////////////////////
    /**
     * Allocate a new file in heapfs
     * @returns fileId, to be used by fileWrite()
     */
    async fileAlloc(fileName, size) {
        const result = await this.pushTask({
            verb: 'fs.alloc',
            args: [fileName, size],
            callbackId: this.taskId++,
        });
        return result.fileId;
    }
    /**
     * Write a Blob to heapfs
     */
    async fileWrite(fileId, blob) {
        const reader = blob.stream().getReader();
        let offset = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done)
                break;
            const size = value.byteLength;
            await this.pushTask({
                verb: 'fs.write',
                args: [fileId, value, offset],
                callbackId: this.taskId++,
            }, 
            // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
            [value.buffer]);
            offset += size;
        }
    }
    /**
     * Parse JSON result returned by cpp code.
     * Throw new Error if "__exception" is present in the response
     */
    parseResult(result) {
        const parsedResult = JSON.parse(result);
        if (parsedResult && parsedResult['__exception']) {
            throw new Error(parsedResult['__exception']);
        }
        return parsedResult;
    }
    /**
     * Push a new task to taskQueue
     */
    pushTask(param, buffers) {
        return new Promise((resolve, reject) => {
            this.taskQueue.push({ resolve, reject, param, buffers });
            this.runTaskLoop();
        });
    }
    /**
     * Main loop for processing tasks
     */
    async runTaskLoop() {
        if (this.busy) {
            return; // another loop is already running
        }
        this.busy = true;
        while (true) {
            const task = this.taskQueue.shift();
            if (!task)
                break; // no more tasks
            this.resultQueue.push(task);
            // TODO @ngxson : Safari mobile doesn't support transferable ArrayBuffer
            this.worker.postMessage(task.param, isSafariMobile()
                ? undefined
                : {
                    transfer: task.buffers ?? [],
                });
        }
        this.busy = false;
    }
    /**
     * Handle messages from worker
     */
    onRecvMsg(e) {
        if (!e.data)
            return; // ignore
        const { verb, args } = e.data;
        if (verb && verb.startsWith('console.')) {
            if (this.suppressNativeLog) {
                return;
            }
            if (verb.endsWith('debug'))
                this.logger.debug(...args);
            if (verb.endsWith('log'))
                this.logger.log(...args);
            if (verb.endsWith('warn'))
                this.logger.warn(...args);
            if (verb.endsWith('error'))
                this.logger.error(...args);
            return;
        }
        else if (verb === 'signal.abort') {
            this.abort(args[0]);
        }
        const { callbackId, result, err } = e.data;
        if (callbackId) {
            const idx = this.resultQueue.findIndex((t) => t.param.callbackId === callbackId);
            if (idx !== -1) {
                const waitingTask = this.resultQueue.splice(idx, 1)[0];
                if (err)
                    waitingTask.reject(err);
                else
                    waitingTask.resolve(result);
            }
            else {
                this.logger.error(`Cannot find waiting task with callbackId = ${callbackId}`);
            }
        }
    }
    abort(text) {
        while (this.resultQueue.length > 0) {
            const waitingTask = this.resultQueue.pop();
            if (!waitingTask)
                break;
            waitingTask.reject(new Error(`Received abort signal from llama.cpp; Message: ${text || '(empty)'}`));
        }
    }
}
//# sourceMappingURL=worker.js.map