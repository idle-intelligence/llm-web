# llm-web

lean is a small LLM inference engine written in Rust against `wgpu`, with hand-written WGSL kernels for the GPU and SIMD kernels for the CPU. It reads a quantized GGUF file and a Hugging Face tokenizer, and it compiles from one source to a native library and to WebAssembly for the browser, with no ML framework. In a browser it runs on WebGPU, then on CPU threads, then on a single CPU thread (WASM SIMD), picked by capability rather than measured; it is the same model on every backend, and it is token-exact against HF transformers.

[**Chat demo →**](https://idle-intelligence.github.io/llm-web/web/) · [**Device check →**](https://idle-intelligence.github.io/llm-web/web/device/)

> **Disclaimer:** lean is an original implementation written from public model configs and GGUF metadata, not a port of llama.cpp or wllama. Models are fetched at run time from their authors' Hugging Face repos under their own licenses; no weights are redistributed here. This project is not affiliated with the Qwen or SmolLM2 teams.

## Quick start

### In a web page

Build the web packages and assemble them into `_site/`:

```bash
ENGINE_BUILD=dev scripts/build.sh    # writes crates/lean/pkg and _site/
python3 scripts/serve.py             # serves _site/ on http://localhost:8030, with cross-origin isolation headers
```

The CPU threads backend is opt-in and needs a nightly toolchain: `BUILD_THREADS=1 scripts/build.sh` also writes `crates/lean/pkg-mt`.

Then open `http://localhost:8030/web/index.html` for the chat demo, or `http://localhost:8030/web/device/` for the device check. The core of the API:

```html
<script type="module">
const mod = await import('./pkg/lean.js');
await mod.default('./pkg/lean_bg.wasm');
const lean = await mod.Lean.load({ gguf: ggufBytes, tokenizer: tokenizerJson, maxCtx: 2048 });
const template = mod.ChatTemplate.fromTokenizerConfig(tokenizerConfigJson);
const prompt = mod.chatRender(template, [{ role: 'user', content: question }], null, true);
const params = { maxNewTokens: 64, temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.1, seed: 0 };
const reply = await lean.generate(prompt, params, (text) => { /* append text */ }, new mod.AbortFlag().cloneFlag());
</script>
```

The engine is text in, text out: `generate` runs one inference turn over the given text and returns the reply. It applies no chat template and keeps no conversation (see [`crates/lean/README.md`](crates/lean/README.md)); a page holds its own message history, renders it with the model's chat template, and sends the whole text each call. `chatRender` comes from [`lean-chat`](crates/lean-chat/README.md), shipped in the same wasm package: it renders messages, and tools, exactly as transformers' `apply_chat_template` does, and `chatTrimOldestExchange` drops the oldest exchange when a conversation no longer fits.

### Native

```bash
cargo build -p lean --release --bin lean-cli
LEAN_GGUF=<path-to-gguf> LEAN_TOKENIZER_DIR=<dir-with-tokenizer.json> \
  cargo run -p lean --release --bin lean-cli -- --prompt "..." --tokens 64
```

### Verify (parity gates)

Greedy tokens matching the transformers reference, on every backend:

```bash
LEAN_GGUF=<gguf> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean-core --release --test fixture_parity -- --ignored
LEAN_GGUF_LLAMA_360M_Q4_0=<gguf> LEAN_GGUF_LLAMA_360M_Q8_0=<gguf> LEAN_TOKENIZER_DIR_LLAMA_360M=<dir> \
  cargo test -p lean-core --release --features threads --test fixture_parity_llama_360m_cpu -- --ignored
```

See [`crates/lean/README.md`](crates/lean/README.md) for the backends and the full JavaScript API, and [`crates/lean-core/README.md`](crates/lean-core/README.md) for the forward pass underneath (token ids, masks, positions, LoRA) and every parity test with its environment variables.

## Architecture

```
lean-chat   messages + tools -> prompt text (the model's own chat template)
    |
    v  text
lean        tokenizer, sampling, streaming, prefix reuse, backend choice, wasm API
    |
    v  token ids
lean-core   GGUF reader, forward pass: WGSL kernels on wgpu, or SIMD kernels on the CPU
```

**Crates.** `lean-core` owns the model: it reads the GGUF, uploads the weights, holds the KV cache and runs the forward pass on token ids, returning logits or a chosen token. `lean` sits on top: it tokenizes with the Hugging Face `tokenizers` crate (`tokenizer.json`), samples, streams text, picks the backend and exports the JavaScript API through wasm-bindgen. `lean-chat` renders chat messages and tools into prompt text with the model's own Jinja template (minijinja), byte for byte what transformers' `apply_chat_template` returns. The engine is text in, text out; the caller owns the conversation and the chat template.

**From a file to a model.** `GpuModel::load` and `CpuModel::load` read the GGUF one tensor at a time from a path, bytes or a reader and upload it, so the whole file is not held in memory as a second copy. The architecture comes from `general.architecture`: `qwen2` (Qwen2 and Qwen2.5), `qwen3` and `llama` (SmolLM2, Llama 3.x, with the GGUF's RoPE scaling). Q4_0, Q4_1, Q5_0, Q8_0, Q4_K and Q6_K stay quantized and are dequantized inside the kernels; F16 and BF16 stay at two bytes per value; any other tensor type is rejected at open with the type and tensor name. Weights wider than a browser's storage-buffer binding limit are split into row chunks bound separately. The tables are in [`crates/lean-core/README.md`](crates/lean-core/README.md#supported-models).

**GPU path.** The kernels are WGSL files in `crates/lean-core/src/shaders`, dispatched through `wgpu` (WebGPU in a browser, Metal and Vulkan natively). Which kernel runs is decided by the shape of the work, with the same rule on every device:

- Prefill of 64 rows or more uses a 64x64 register-blocked tiled matmul. Shorter prefills use a small-M kernel (8 rows by 64 columns per workgroup) because the 64-row tile would be mostly padding.
- Decode uses matvec kernels: four weight rows per lane when the output has at least 4096 rows, one row per lane otherwise. Decode attention is split along the key axis for long contexts (flash-decoding style), with the number of splits computed from the context length.
- When the adapter grants `shader-f16`, the long-prefill tile holds weights and activations in f16 and sums short windows in f16 before accumulating in f32. Adapters without it run the f32 tile. `LEAN_F16=0` turns the f16 tile off for testing.
- Greedy decoding, repetition penalty, top-k and sampling run on the GPU (`sample_penalty`, `topk_stage`, `sample_final`), so a step does not read the logits back, and steps are submitted before the previous token is read.
- Each shader is compiled for the model at hand by `specialize`, which writes build constants (head dimension, block format, fused gate/up) into the WGSL source before the module is created. They are plain `const`s, not pipeline-overridable constants, and pipelines are compiled on first use, so a model pays only for the formats it contains.

**CPU path.** The same forward pass on the CPU, with dot-product kernels for each quantized format: NEON on aarch64, SIMD128 in WebAssembly, scalar code elsewhere. With the `threads` feature each matrix product is split across a pool (rayon natively, Web Workers through wasm-bindgen-rayon in a browser, which needs a cross-origin isolated page). Without it, the same code runs on one thread.

**Choosing a backend.** `Backend::Auto` tries WebGPU first and falls back to the CPU when there is no adapter, when any pipeline fails validation (the engine checks all of them in error scopes before handing itself out), or when the model cannot run on the GPU kernels (a head dimension other than 64 or 128). On the CPU it uses the thread pool if the build has threads and more than one thread is available, otherwise one thread. The choice depends on what the device reports, never on a timing run.

**One source, native and wasm.** The three crates compile for the host and for `wasm32`. The target differences are the wgpu setup (`wasm` and `web` features), the SIMD kernels chosen at compile time, and the thread pool.

**KV cache and prefix reuse.** `lean` keeps the KV cache between calls and reuses the longest token prefix the new text shares with the previous one, so a growing conversation prefills only its new suffix; replies are identical with or without reuse. On SmolLM2-1.7B Q4_0 in Chromium on an M2 laptop, a 500-token prefix plus 124 new tokens reaches its first token in 580 ms, against 2110 ms for the same 500 tokens prefilled cold ([run](docs/runs/2026-10-12-lean-browser-perf.md)); decode on that setup runs at about 69 tokens per second, greedy or sampled (same run log). `lean-core` can also snapshot a KV cache to bytes and restore it, and apply a LoRA adapter (q/k/v/o deltas) on top of the quantized weights, on both the GPU and the CPU path. If the input does not fit the context, `generate` returns a typed `ContextOverflow` error with the needed and available counts and leaves the engine usable; trimming is the caller's choice.

**Correctness.** The reference is Hugging Face transformers. Fixtures are written by transformers from the same GGUF, and the `fixture_parity*` tests require the same input ids, the same top-20 logits after prefill within 1e-3, and the same greedy continuation, on the GPU and on the CPU path, for Qwen2.5, Qwen3, SmolLM2 and Llama 3.2 models in several quantizations. The f32 kernels stay within 1.2e-4 of the reference on the top-20 logits in the 2026-10-12 run. The f16 prefill tile is looser by design: its last-position logits differ from the f32 kernels by 0.02 to 0.055 on the models and prompts measured and from the reference by at most 0.030 on those prompts and by up to about 0.1 on longer batched prompts of several hundred tokens, so the 1e-3 gate does not apply to it. Its gate (`f16_parity`) requires identical greedy tokens, the same top-1 after prefill, and each path's token in the other's top 5 at every generated position; all measured continuations were identical. Prompts shorter than 64 rows use the small-M kernel and show no difference. Run logs for each gate are in [`docs/runs/`](docs/runs/).

**Complemented by agent-web.** lean stays an inference engine: it runs the model and does not decide what to do with the output. [agent-web](https://github.com/idle-intelligence/agent-web) is a tool-calling agent loop for Qwen2.5-Instruct models that runs on lean. It renders tool prompts through lean-chat, parses the model's tool calls, regenerates when a reply is malformed, drives the step-wise loop and scores runs with an offline eval harness. It depends on lean and lean-chat, and uses lean's normal backend selection.

## Models

| Model | Params | Quant | License |
|-------|--------|-------|---------|
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Q4_0 | Apache 2.0 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | Q4_0 | Apache 2.0 |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | Q4_0 | Apache 2.0 |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Q4_0 (Q6_K embedding) | Apache 2.0 |

See [`crates/lean-core/README.md`](crates/lean-core/README.md#supported-models) for the full architecture and tensor-type list.

## Credits

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Apache 2.0).
- [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (Apache 2.0).
- Weights for all of the above are not distributed here; both demo pages fetch them from Hugging Face at run time.

## License

Code in this repository (excluding vendored/third-party components noted above, and excluding any model weights, which are not distributed here) is licensed under the [MIT License](LICENSE).
