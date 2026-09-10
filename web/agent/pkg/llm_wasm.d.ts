/* tslint:disable */
/* eslint-disable */

/**
 * Browser-facing agent engine. Single entry point a Web Worker calls —
 * see `web/agent/worker.js` for the message protocol built on it.
 */
export class LlmEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append one GGUF shard (any split the caller likes — a single
     * element is fine for a non-sharded file). Call before `load()`.
     */
    appendModelShard(shard: Uint8Array): void;
    /**
     * JSON string with basic model/device info, for the page's status line.
     */
    info(): string;
    /**
     * Parse the GGUF (from previously `appendModelShard`-ed bytes), upload
     * weights to GPU, and build the tokenizer + chat template.
     *
     * `on_progress`, if a function, is called as `on_progress(stage:
     * string, step: number, total: number)` at each of this method's three
     * phases (`"parsing-gguf"`, `"finalizing-gpu"`, `"ready"`) — coarse
     * progress, since `gguf.rs`'s `load_deferred` loads all 36 transformer
     * layers in one call with no per-layer hook. Byte-level shard fetch
     * progress is the caller's job (see `web/agent/worker.js`'s `load`
     * handler, which reports progress while fetching, before ever calling
     * `appendModelShard`/`load`).
     *
     * GGUF bytes stay as whatever shards were appended — this crate's
     * `Q4ModelLoader::from_shards` reads through a `ShardedCursor`
     * (`gguf.rs`), so the 1.74GB model file need not be one contiguous
     * buffer; the caller can split the fetch into e.g. 64-128MB chunks (or
     * pass one element) — either fits comfortably under wasm32's 4GB
     * address space.
     */
    load(tokenizer_json: string, tokenizer_config_json: string, on_progress: any): Promise<void>;
    /**
     * `#[wasm_bindgen(constructor)]` cannot return `Result` (no fallible
     * JS constructor), so if `initWgpuDevice()` wasn't awaited first this
     * silently falls back to `WgpuDevice::default()` rather than erroring
     * — logged as a `console.warn` since a fallback device here almost
     * certainly means every subsequent GPU call fails or targets the
     * wrong adapter.
     */
    constructor();
    /**
     * Continue the current turn with tool results, keyed by `call_id` from
     * the most recent `NeedTools` outcome. `results_json`:
     * `[{"call_id":"call_0","result":{...}}, ...]`. Same return shape as
     * `start()`.
     */
    provideToolResults(results_json: string): Promise<string>;
    /**
     * Drop the in-progress conversation (messages/tools/pending calls) and
     * the KV cache's resident tokens, keeping the loaded model.
     */
    reset(): void;
    /**
     * Set the system prompt used by subsequent `start()` calls.
     */
    setSystemPrompt(system_prompt: string): void;
    /**
     * Begin a new turn. `tools_json` is a JSON array of MCP `tools/list`
     * entries (`{"name", "description", "inputSchema"}`); `opts_json` is
     * `{"maxNewTokens"?: number, "maxSteps"?: number}` (both optional; `{}`
     * or `"{}"` is fine). Returns a JSON string: `{"outcome":"needTools",
     * "calls":[{"call_id","name","arguments"}...],"step":{...}}`,
     * `{"outcome":"final","text":...,"step":{...}}`, or
     * `{"outcome":"error","message":...}` — see docs/ENGINE.md's Browser
     * section for the full shape and `web/agent/worker.js` for how it's
     * consumed. No `on_token` streaming callback: `model.rs`'s `generate`
     * has no per-token hook at HEAD, so a step's text arrives in one piece
     * when the step completes (see module docs).
     */
    start(utterance: string, tools_json: string, opts_json: string): Promise<string>;
}

/**
 * Initialize the WebGPU device asynchronously. **Must** be called (and
 * awaited) once before constructing any `LlmEngine`. Requests the
 * adapter's full limits, same as stt-web's `initWgpuDevice` — this model's
 * 175MB single-buffer tied lm_head needs `maxStorageBufferBindingSize`
 * above the WebGPU spec default (see docs/ENGINE.md's Browser section).
 */
export function initWgpuDevice(): Promise<void>;

/**
 * Initialize panic hook for readable browser-console error messages.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_llmengine_free: (a: number, b: number) => void;
    readonly initWgpuDevice: () => any;
    readonly llmengine_appendModelShard: (a: number, b: number, c: number) => void;
    readonly llmengine_info: (a: number) => [number, number];
    readonly llmengine_load: (a: number, b: number, c: number, d: number, e: number, f: any) => any;
    readonly llmengine_new: () => number;
    readonly llmengine_provideToolResults: (a: number, b: number, c: number) => any;
    readonly llmengine_reset: (a: number) => void;
    readonly llmengine_setSystemPrompt: (a: number, b: number, c: number) => void;
    readonly llmengine_start: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => any;
    readonly start: () => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2d808c2d349e4bb9: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h39f7e6a28896bbe3: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb2da000e6071c27b: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
