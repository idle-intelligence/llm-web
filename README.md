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
mod.leanInit();
const engine = navigator.gpu ? await mod.LeanEngine.create() : mod.LeanEngineCpu.create();
engine.load(ggufBytes, tokenizerJson, tokenizerConfigJson, 2048);
await engine.chatGenerate(prompt, 64, 0.7, 40, 0.9, 1.1, 0, new Uint32Array(0),
  (_id, text) => { /* append text */ }, new mod.AbortFlag().cloneFlag());
</script>
```

### Native

```bash
cargo build -p lean --release --bin lean-cli
LEAN_GGUF=<path-to-gguf> LEAN_TOKENIZER_DIR=<dir-with-tokenizer.json> \
  cargo run -p lean --release --bin lean-cli -- --prompt "..." --tokens 64
```

### Verify (parity gates)

Greedy tokens matching the transformers reference, on every backend:

```bash
LEAN_GGUF=<gguf> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --release --test fixture_parity -- --ignored
LEAN_GGUF_LLAMA_360M_Q4_0=<gguf> LEAN_GGUF_LLAMA_360M_Q8_0=<gguf> LEAN_TOKENIZER_DIR_LLAMA_360M=<dir> \
  cargo test -p lean --release --features threads --test fixture_parity_llama_360m_cpu -- --ignored
```

See [`crates/lean/README.md`](crates/lean/README.md) for the backends, the full JavaScript API, and every parity test and its environment variables.

## Models

| Model | Params | Quant | License |
|-------|--------|-------|---------|
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Q4_0 | Apache 2.0 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | Q4_0 | Apache 2.0 |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | Q4_0 | Apache 2.0 |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Q4_0 (Q6_K embedding) | Apache 2.0 |

See [`crates/lean/README.md`](crates/lean/README.md#supported-models) for the full architecture and tensor-type list.

## Credits

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Apache 2.0).
- [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (Apache 2.0).
- Weights for all of the above are not distributed here; both demo pages fetch them from Hugging Face at run time.

## License

Code in this repository (excluding vendored/third-party components noted above, and excluding any model weights, which are not distributed here) is licensed under the [MIT License](LICENSE).
