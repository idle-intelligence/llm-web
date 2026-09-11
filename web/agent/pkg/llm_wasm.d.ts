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
     * Export the current cache's system+tools prefix (for `tools_json`/
     * `system`) as a `.kvimg` byte buffer — the counterpart to
     * `import_kv_image`, called on a cache *miss* after the first prefill
     * of a tool set so the worker can save the image to OPFS for next
     * time. Errors (rather than exporting garbage) if `resident_tokens`
     * doesn't currently cover the rendered prefix — call this only after
     * a `start()`/step whose prefill included the full system+tools
     * preamble. `dtype` is always `q8_0` (the format `docs/ENGINE.md`
     * recommends for a one-time browser download — see kvimg.rs module
     * docs).
     */
    exportKvImage(tools_json: string, system: string): Promise<Uint8Array>;
    /**
     * Import a prefix KV image (`bytes`, a full `.kvimg` file as fetched
     * from `<modelBase>/kv/<prefix_key>.kvimg` or OPFS) in place of
     * running prefill for `tools_json`/`system`'s system+tools preamble.
     * Validates `header.model_fingerprint`, `header.prefix_key`, and
     * `header.tokens` against what this engine/model/tools/system would
     * actually render (same match discipline `run_step`'s
     * `effective_prefix` check already applies to `resident_tokens`) —
     * returns `Ok(false)` on any mismatch (caller falls back to normal
     * prefill) rather than importing a wrong prefix. Synchronous:
     * `KvCache::import_prefix` only writes (`from_data`/`slice_assign`),
     * no GPU readback, so no async/await is needed on this path.
     */
    importKvImage(bytes: Uint8Array, tools_json: string, system: string): boolean;
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
     * Prefix-KV-image cache key for `tools_json`/`system` under the
     * currently loaded model (`docs/ENGINE.md` "Prefix KV images"):
     * `sha256(model_fingerprint || rendered_prefix_text)`, computed the same way
     * `bin/llm-agent.rs`'s `kv-export` subcommand computes it when writing
     * an image, so a worker can `fetch(<modelBase>/kv/<key>.kvimg)` before
     * its first prefill of a given tool set. Errors if the model isn't
     * loaded yet (no `model_fingerprint`/tokenizer/template) or `tools_json` is
     * malformed.
     */
    prefixKey(tools_json: string, system: string): string;
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
     * Debug A/B toggle for a browser-only numerical-divergence bisection
     * (see `gguf.rs`'s `force_naive_kernel`): `"naive"` forces the naive
     * per-element Q4 matmul kernel for every prefill matmul regardless of
     * M; anything else (including `"pinned"`, the default) restores
     * production `ForceKernel::Auto` routing. Not used by any production
     * code path.
     */
    setPrefillKernel(kernel: string): void;
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
    readonly llmengine_exportKvImage: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly llmengine_importKvImage: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly llmengine_info: (a: number) => [number, number];
    readonly llmengine_load: (a: number, b: number, c: number, d: number, e: number, f: any) => any;
    readonly llmengine_new: () => number;
    readonly llmengine_prefixKey: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly llmengine_provideToolResults: (a: number, b: number, c: number) => any;
    readonly llmengine_reset: (a: number) => void;
    readonly llmengine_setPrefillKernel: (a: number, b: number, c: number) => void;
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
