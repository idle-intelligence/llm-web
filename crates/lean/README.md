# lean

lean is a small LLM inference engine written in Rust against `wgpu`, with hand-written WGSL kernels for the GPU and SIMD kernels for the CPU. It reads a quantized GGUF file and a Hugging Face tokenizer, and it compiles from one source to a native library and to WebAssembly for the browser. It uses no ML framework.

In a browser it runs on WebGPU or fallback on WASM SIMD CPU. It is the same model on every backend, and it is token-exact against HF transformers: the greedy tokens match the transformers reference on every backend.

[Chat demo](https://idle-intelligence.github.io/llm-web/web/) and [device check](https://idle-intelligence.github.io/llm-web/web/device/); the root [README](../../README.md#quick-start) has the quick start (web page, native CLI, parity gates).

## Backends

A page picks one of three backends, in this order:

1. **WebGPU.** The forward pass runs as WGSL compute kernels, with the weights kept quantized on the GPU and dequantized inside the kernels.
2. **CPU threads.** The CPU forward pass with each matrix product split across a pool of Web Workers (wasm-bindgen-rayon). It needs `SharedArrayBuffer`, so the page must be served cross-origin isolated (the `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers), and it needs more than one hardware thread.
3. **Single CPU thread.** The same CPU forward pass on one thread, with SIMD128 dot-product kernels. It runs wherever WebAssembly does.

The choice is made by capability, never by measuring the device: WebGPU when `navigator.gpu.requestAdapter()` returns an adapter, otherwise CPU threads when `crossOriginIsolated` is true, `SharedArrayBuffer` exists and `navigator.hardwareConcurrency` is above 1, otherwise a single CPU thread. The kernels are the same on every device; there is no autotuning and no per-device code.

Natively the same code runs through `wgpu` (tested on Metal and Vulkan), and the CPU path uses NEON on aarch64 and scalar code elsewhere, on one thread or, with the `threads` feature, on a rayon pool.

## Supported models

The forward pass is the [`lean-core`](../lean-core/README.md) crate; its README lists the architectures (Qwen2/Qwen2.5, Qwen3, Llama) and tensor types (Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K kept quantized; F16 and BF16 kept at two bytes per value; F32). With the `Auto` backend, a model the WebGPU kernels cannot run (a head dimension other than 64 or 128) loads on the CPU.

## The engine: text in, text out

The engine is `Lean`, with the same surface in Rust and in JavaScript:

- **Load.** `Lean::load(LoadOptions { gguf, tokenizer, max_ctx, backend })` (Rust, `async`), `await Lean.load({ gguf, tokenizer, maxCtx, backend })` (JS). `gguf` is a path, bytes or a reader in Rust and a `Uint8Array` in JS, or, in a Web Worker, a `FileSystemSyncAccessHandle` on the file in the origin private file system (Rust: `GgufSource::Reader(Box::new(OpfsReader::new(handle)?))`), read at offsets one tensor at a time so the file never sits whole in JS or wasm memory; `tokenizer` is `tokenizer.json` (a path or its text in Rust, its text in JS); `max_ctx` sizes the KV cache. `backend` defaults to `Auto`: WebGPU when an adapter is granted and the model loads on it, otherwise the CPU forward pass, on a thread pool when the build has threads and the pool has more than one thread, else on one thread. `WebGpu` / `"webgpu"` and `Cpu` / `"cpu"` take only that backend. `backend()` (JS: the `backend` property) says which one runs: `"webgpu"`, `"threads"` or `"single"`.
- **One inference turn.** `generate(text, params, on_token, abort)` tokenizes `text` exactly as given, decodes up to `params.max_new_tokens` tokens, streams each piece of reply text to `on_token` and returns the whole reply. `params` holds `max_new_tokens`, `temperature` (0 = greedy), `top_k`, `top_p`, `repetition_penalty`, `seed` and an optional logit mask (JS: `maxNewTokens`, `temperature`, `topK`, `topP`, `repetitionPenalty`, `seed`, `mask`). `abort` is an `AbortFlag` checked before each token: in JS, pass `flag.cloneFlag()` and call `flag.abort()` from a message handler.
- **Token count.** `token_count(text)` (JS: `tokenCount(text)`) tokenizes without running the model.
- **Prefill.** `prefill(text)` (JS: `await prefill(text)`, resolves to the token count) makes `text` resident in the KV cache without generating, so a later `generate` whose text starts with it prefills only the rest: a fixed system prompt or tools block can be warmed up before the first request.
- **Prefix KV images.** `save_prefix(prefix, out)` writes the KV cache of `prefix` (prefilled if needed) as bytes: a 128-byte header, then each layer's K and V in f32, one tensor per write (JS: `await savePrefix(prefix, onChunk)`). `restore_prefix(prefix, input)` reads such an image back, one tensor at a time, into the cache of a freshly loaded engine (JS: `restorePrefix(prefix, source)`, `source` a `Uint8Array` or a `read(n)` function; `restorePrefix` is synchronous and so must `read(n)` be, which in a browser means a Web Worker reading through an OPFS sync access handle), and the next `generate` gives exactly the reply it would give without the image. `prefix_key(prefix)` (JS: `prefixKey`) is the key to store the image under: a hash of the format version, dtype, engine build, model file header, tokenizer, backend and prefix tokens, so any change to one of them gives a new key. The engine build covers lean's own sources and its workspace `Cargo.lock`, not the lock file of a crate that depends on lean. A restore checks the same fields in the image header and refuses a mismatch with `LeanError::PrefixImage` (JS: an `Error` named `"PrefixImage"`, `code` `"STALE"` with the differing `field`, `"NOT_AN_IMAGE"` or `"TRUNCATED"`), leaving the engine usable; the caller then prefills again and overwrites the image. Where images are kept (OPFS, a download) is the caller's choice. An image is 2 x layers x KV heads x head dim x 4 bytes per prefix token: 25 KB per token for Qwen2.5-0.5B, 229 KB for Qwen3-1.7B.

The engine applies no chat template and keeps no conversation: the caller renders its messages with the model's template and passes the text. The [`lean-chat`](../lean-chat/README.md) crate does that rendering, byte-exact with transformers, and its JavaScript functions (`chatRender`, `chatTrimOldestExchange`) are exported from the same wasm package. Between calls the engine keeps the KV cache of its previous call and reuses the longest token prefix the new text shares with it, so a growing conversation only prefills its new suffix; the output is the same with or without that reuse. **One client per engine**: two clients sharing one engine evict each other's cached prefix (nothing breaks, each call just costs a full prefill), so give each client its own `Lean`.

If `text` plus `max_new_tokens` does not fit `max_ctx`, `generate` returns `LeanError::ContextOverflow { needed, available }` before touching anything, and the engine stays usable. In JS the promise rejects with an `Error` whose `name` is `"ContextOverflow"`, `code` is `"CONTEXT_OVERFLOW"`, and which carries numeric `needed` and `available`. What to drop before retrying is the caller's choice; `web/lean-chat-worker.js` and `examples/chat.rs` drop the oldest exchange.

A minimal worker:

```js
const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
await mod.default({ module_or_path: `../pkg/lean_bg.wasm?v=${ENGINE_BUILD}` });
const lean = await mod.Lean.load({ gguf, tokenizer: tokenizerJson, maxCtx: 2048 });

const abortFlag = new mod.AbortFlag();
const chatTemplate = mod.ChatTemplate.fromTokenizerConfig(tokenizerConfigJson);
const prompt = mod.chatRender(chatTemplate, [{ role: "user", content: text }], null, true);
const params = { maxNewTokens: 128, temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.1, seed: 1 };
const reply = await lean.generate(prompt, params, (piece) => {
  self.postMessage({ type: "piece", text: piece });
}, abortFlag.cloneFlag());
```

For CPU threads the page loads `../pkg-mt/lean.js` instead and calls `await mod.initThreadPool(navigator.hardwareConcurrency)` after `mod.default(...)` and before `Lean.load`. Run the engine in a Web Worker, never on the main thread.

Natively, `load` and `generate` are `async` too; drive them with any executor, e.g. `pollster::block_on(lean.generate(...))`.

### Low-level calls

Built with the `diag` cargo feature (not in the default web package), the wasm also exports `LeanEngine` (WebGPU, `await LeanEngine.create()`, plus `leanInit()` for the panic hook) and `LeanEngineCpu` (`LeanEngineCpu.create()`), which work on raw token ids for a caller that drives its own token loop: `load(ggufBytes, tokenizerJson, maxCtx)`, `encodeRaw`, `decodeIds`, `prefillTokens`, `decodeStepArgmax`, `info`, and on `LeanEngine` also `appendTokens`, `decodeGreedy`, `kvLen`, `buildMaskBitset` (a logit mask that can change at every step), `snapshotKv`/`restoreKv` (store the KV cache of a long fixed prefix and skip its prefill next time) and the diagnostics calls. In Rust, token ids, masks, positions and LoRA adapters are the `lean-core` crate's API.

## Build

```bash
# Native library and the lean-cli binary
cargo build -p lean --release
cargo build -p lean --release --features threads   # CPU path on a rayon pool

# Web: writes crates/lean/pkg (WebGPU + single CPU thread) and assembles _site/
ENGINE_BUILD=<tag> scripts/build.sh
```

`ENGINE_BUILD` is the `?v=` tag that every page puts on its wasm and JS URLs. A rebuild comes with a new tag on every loading URL, otherwise browsers keep running the cached module. `scripts/build.sh` maps the home directory out of the compiled-in source paths and fails if a local path is left in either wasm. It needs `wasm-pack` and the `wasm-bindgen-cli` version that matches `Cargo.lock`; see the comment at its top. `BUILD_THREADS=1 scripts/build.sh` also writes `crates/lean/pkg-mt` (CPU threads), and needs a nightly toolchain with `rust-src`.

## Tests

`fixture_parity_generate` and `engine_api` check the engine against fixtures written by HF transformers (`crates/lean-core/reference/gen_fixture*.py`, which load the same GGUF through transformers). They need model files that are not in the repository, so they are `#[ignore]`d and run with `--ignored` and environment variables pointing at a GGUF and at a directory holding `tokenizer.json` and `tokenizer_config.json`:

| test | what it checks | environment |
|---|---|---|
| `fixture_parity_generate` | `Lean::generate` on GPU and CPU: token count and greedy reply equal the fixture, Qwen2.5-0.5B, SmolLM2-360M and Llama-3.2-1B Q4_0 | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR`, the `_LLAMA_360M` Q4_0 pair, `LEAN_GGUF_LLAMA32_1B_Q4_0`, `LEAN_TOKENIZER_DIR_LLAMA32_1B` |
| `engine_api` | prefix reuse gives the same replies as a fresh engine (growing conversation, diverging prefixes), CPU equals GPU over three turns, `ContextOverflow` counts and recovery | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR`, the `_LLAMA_360M` pair, `LEAN_GGUF_LLAMA_1_7B_Q4_0`, `LEAN_TOKENIZER_DIR_LLAMA_1_7B` |

```bash
LEAN_GGUF=<gguf> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --release --test engine_api -- --ignored
```

The decode loop, sampling and streaming detokenization are checked by unit tests (`src/tests`, same environment variables, `cargo test -p lean --release --lib -- --ignored`).

The forward-pass parity gates (logits and greedy tokens against transformers, every model and backend, LoRA, masks, KV snapshots) are in [`lean-core`](../lean-core/README.md#parity-gates); the chat-template exactness tests are in [`lean-chat`](../lean-chat/README.md). Release gate logs are under `docs/runs/`.

## License

MIT, like the rest of this repository (see `LICENSE` at the repository root). Model weights are not distributed here and keep their own licenses.
