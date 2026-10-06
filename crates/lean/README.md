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

Architectures, from `general.architecture` in the GGUF (`src/config.rs`):

- `qwen2`: Qwen2 and Qwen2.5 (tested: Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct).
- `qwen3`: Qwen3 (tested: Qwen3-0.6B, Qwen3-1.7B).
- `llama`: Llama-architecture models (tested: SmolLM2-360M-Instruct, SmolLM2-1.7B-Instruct).

The GPU attention kernels are compiled for a head dimension of 64 or 128; `LeanEngine` rejects a model with another head dimension at load.

Tensor types (`src/gguf.rs`, `src/quant.rs`, `src/cpu.rs`):

| type | how it runs |
|---|---|
| Q4_0 | kept quantized, dequantized in the kernels (GPU and CPU) |
| Q8_0 | kept quantized, dequantized in the kernels (GPU and CPU) |
| Q6_K | kept quantized, dequantized in the kernels (GPU and CPU); usually the embedding and output tensors of a "Q4_0" GGUF |
| Q4_1 | dequantized to F32 at load (a few tensors of the SmolLM2 "Q4_0" GGUFs) |
| F16, F32 | F32 at load (norms, biases) |

Any other type is rejected at load with an error naming it.

## JavaScript API

The wasm module exports two engines with the same method names wherever both implement a method, so a page can hold either one behind the same calls:

- `LeanEngine` (WebGPU): `await LeanEngine.create()`.
- `LeanEngineCpu` (CPU, one thread or the thread pool, depending on the build): `LeanEngineCpu.create()`.

Both load with `engine.load(ggufBytes, tokenizerJson, tokenizerConfigJson, maxCtx)` and run in a Web Worker, never on the main thread.

- **Chat.** `await engine.chatGenerate(prompt, maxNewTokens, temperature, topK, topP, repetitionPenalty, seed, maskBits, onToken, abortFlag)` adds a user turn, streams the reply and keeps the KV cache for the next turn; only the new part of the conversation is prefilled. `engine.chatReset()` starts a new conversation. `onToken(id, text)` is called once per generated token; `text` is the piece of reply text that token completes, so concatenating every `text` gives the returned reply. If text is still held back when generation ends, one last call passes `id = -1`. The callback must not call back into the engine. A throw from the callback stops generation and rejects the call.
- **Stop.** `new AbortFlag()`; pass `flag.cloneFlag()` to the call and call `flag.abort()` from a message handler. The engine checks it before each token.
- **One-shot generation** (`LeanEngine`): `generate(prompt, maxNewTokens, onToken)` is greedy; `generateStream(prompt, maxNewTokens, temperature, topK, topP, repetitionPenalty, seed, maskBits, onToken, abortFlag)` samples. `LeanEngineCpu.generate` is greedy and synchronous.
- **Logit mask.** `engine.buildMaskBitset(allowedIds)` packs the allowed token ids into a bitset; pass it as `maskBits` (an empty array means no mask). `LeanEngine` also takes a mask on the low-level calls `prefillTokens`, `appendTokens` and `decodeStepArgmax`, so a grammar can change the allowed set at every step. A mask shorter than `vocabSize / 32` words is rejected.
- **KV snapshot** (`LeanEngine`): `await engine.snapshotKv()` returns the KV cache prefix as bytes, `engine.restoreKv(bytes)` puts it back, and `appendTokens(ids, maskBits)` continues from it. A page can store the snapshot of a long fixed prefix and skip its prefill next time.
- **LoRA.** LoRA adapters are in the Rust API (`apply_lora` / `clear_lora` on the GPU model and on the CPU model); they are not exposed to JavaScript yet.
- **Tokens.** `tokenize`, `encodeRaw`, `decodeIds`, `tokenCount`, `kvLen`, `info`.

A minimal worker:

```js
const mod = await import(`../pkg/lean.js?v=${ENGINE_BUILD}`);
await mod.default({ module_or_path: `../pkg/lean_bg.wasm?v=${ENGINE_BUILD}` });
mod.leanInit();
const engine = await mod.LeanEngine.create(); // or mod.LeanEngineCpu.create()
engine.load(gguf, tokenizerJson, tokenizerCfgJson, 2048);

const abortFlag = new mod.AbortFlag();
const reply = await engine.chatGenerate(text, 128, 0.7, 40, 0.9, 1.1, 1, new Uint32Array(0), (id, piece) => {
  if (piece) self.postMessage({ type: "piece", text: piece });
}, abortFlag.cloneFlag());
```

For CPU threads the page loads `../pkg-mt/lean.js` instead and calls `await mod.initThreadPool(navigator.hardwareConcurrency)` after `mod.default(...)` and before `LeanEngineCpu.create()`.

## Build

```bash
# Native library and the lean-cli binary
cargo build -p lean --release
cargo build -p lean --release --features threads   # CPU path on a rayon pool

# Web: writes crates/lean/pkg (WebGPU + single CPU thread) and assembles _site/
ENGINE_BUILD=<tag> scripts/build.sh
```

`ENGINE_BUILD` is the `?v=` tag that every page puts on its wasm and JS URLs. A rebuild comes with a new tag on every loading URL, otherwise browsers keep running the cached module. `scripts/build.sh` maps the home directory out of the compiled-in source paths and fails if a local path is left in either wasm. It needs `wasm-pack` and the `wasm-bindgen-cli` version that matches `Cargo.lock`; see the comment at its top. `BUILD_THREADS=1 scripts/build.sh` also writes `crates/lean/pkg-mt` (CPU threads), and needs a nightly toolchain with `rust-src`.

## Parity gates

The parity tests compare lean against fixtures written by HF transformers (`reference/gen_fixture*.py`, which load the same GGUF through transformers). Each `fixture_parity*` test checks that lean's tokenizer and chat template give the fixture's input ids, that the top-20 logits after prefill agree within 1e-3, and that the greedy continuation is identical. They need model files that are not in the repository, so they are `#[ignore]`d and run with `--ignored` and environment variables pointing at a GGUF and at a directory holding `tokenizer.json` and `tokenizer_config.json`:

| test | what it checks | environment |
|---|---|---|
| `fixture_parity` | Qwen2.5-0.5B Q4_0, both GPU kernel paths | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `fixture_parity_qwen25_3b` | Qwen2.5-3B Q4_0 (Q6_K output tensor) | `LEAN_GGUF_QWEN25_3B`, `LEAN_TOKENIZER_DIR_QWEN25_3B` |
| `fixture_parity_qwen3` | Qwen3-0.6B Q8_0 | `LEAN_GGUF_QWEN3`, `LEAN_TOKENIZER_DIR_QWEN3` |
| `fixture_parity_qwen3_1_7b` | Qwen3-1.7B Q8_0 | `LEAN_GGUF_QWEN3_1_7B`, `LEAN_TOKENIZER_DIR_QWEN3_1_7B` |
| `fixture_parity_llama_360m` | SmolLM2-360M Q4_0 and Q8_0, GPU | `LEAN_GGUF_LLAMA_360M_Q4_0`, `LEAN_GGUF_LLAMA_360M_Q8_0`, `LEAN_TOKENIZER_DIR_LLAMA_360M` |
| `fixture_parity_llama_360m_cpu` | the same on the CPU path (run with and without `--features threads`) | same as above |
| `fixture_parity_llama_1_7b` | SmolLM2-1.7B Q8_0 | `LEAN_GGUF_LLAMA_1_7B_Q8_0`, `LEAN_TOKENIZER_DIR_LLAMA_1_7B` |
| `chat_api` | three-turn chat, CPU tokens equal GPU tokens, on Qwen2.5-0.5B and the SmolLM2 models | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR`, the `_LLAMA_360M` pair, `LEAN_GGUF_LLAMA_1_7B_Q4_0`, `LEAN_TOKENIZER_DIR_LLAMA_1_7B` |
| `logit_mask` | a mask forces an exact string; an all-allowed mask equals no mask | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `kv_snapshot` | restore plus suffix equals a full prefill; snapshot round trip | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `lora_parity` | LoRA on the GPU against a transformers fixture with the LoRA deltas applied | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR`, `LEAN_LORA_BIN` |
| `cpu_lora_parity` | LoRA on the CPU path against HF PEFT | `LEAN_GGUF`, `LEAN_LORA_BIN`, `LEAN_HF_PEFT_FIXTURE` |

```bash
LEAN_GGUF=<gguf> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --release --test fixture_parity -- --ignored
LEAN_GGUF_LLAMA_360M_Q4_0=<gguf> LEAN_GGUF_LLAMA_360M_Q8_0=<gguf> LEAN_TOKENIZER_DIR_LLAMA_360M=<dir> \
  cargo test -p lean --release --features threads --test fixture_parity_llama_360m_cpu -- --ignored
```

`cargo test -p lean` without `--ignored` runs the tests that need no model file (dequantization, chat template, LoRA parsing). Release gate logs are under `docs/runs/`.

## License

MIT, like the rest of this repository (see `LICENSE` at the repository root). Model weights are not distributed here and keep their own licenses.
