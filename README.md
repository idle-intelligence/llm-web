# llm-web

lean is a small LLM inference engine written in Rust against `wgpu`, with hand-written WGSL kernels for the GPU and SIMD kernels for the CPU. It reads a quantized GGUF file and a Hugging Face tokenizer, and it compiles from one source to a native library and to WebAssembly for the browser, with no ML framework. In a browser it runs on WebGPU or falls back to WASM SIMD CPU, picked by capability rather than measured; it is the same model on every backend, and it is token-exact against HF transformers.

[**Chat demo →**](https://idle-intelligence.github.io/llm-web/web/) · [**Device check →**](https://idle-intelligence.github.io/llm-web/web/device/)

> **Disclaimer:** lean is an original implementation written from public model configs and GGUF metadata, not a port of llama.cpp or wllama. Models are fetched at run time from their authors' Hugging Face repos under their own licenses; no weights are redistributed here. This project is not affiliated with the Qwen or SmolLM2 teams.

## Quick start

### In a web page

Build the web packages and serve the repository with cross-origin isolation headers (needed for the CPU threads backend):

```bash
ENGINE_BUILD=dev scripts/build_lean.sh       # crates/lean/pkg: WebGPU + single-thread CPU
ENGINE_BUILD=dev scripts/build_lean_mt.sh    # crates/lean/pkg-mt: CPU threads (needs a nightly toolchain, see the script)
python3 scripts/serve_coi.py
```

Then open [`web/hello/index.html`](web/hello/index.html), a complete, minimal example (loads a model, streams one reply):

```html
<script type="module">
const mod = await import('../../crates/lean/pkg/lean.js');
await mod.default('../../crates/lean/pkg/lean_bg.wasm');
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

See [`crates/lean/README.md`](crates/lean/README.md) for the backends, the full JavaScript API, every parity test and its environment variables, and the `www/` development pages (`backends.html`, `chat.html`).

## Models

| Model | Params | Quant | License |
|-------|--------|-------|---------|
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Q4_0 | Apache 2.0 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | Q4_0 | Apache 2.0 |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | Q4_0 | Apache 2.0 |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Q4_0 (Q6_K embedding) | Apache 2.0 |

See [`crates/lean/README.md`](crates/lean/README.md#supported-models) for the full architecture and tensor-type list.

## llm-wasm (earlier engine)

`crates/llm-wasm/` is an earlier Burn+wgpu implementation of the Qwen2 architecture (GQA attention, RoPE, SwiGLU), with quantized GGUF weights, runtime LoRA adapters, schema-constrained decoding and an MCP-shaped tool-calling agent loop (`agent.rs`/`web.rs`). It runs Qwen2.5-0.5B-Instruct (Q4_0) with runtime LoRA adapters in the browser, and is the engine behind the LLM methods of [llm-life](https://github.com/idle-intelligence/llm-life). Prefix KV cache images let a session restore GPU KV state to the longest matching prompt prefix instead of re-prefilling from scratch. Its demo page (`web/agent/`) is a local dev harness, loading the GGUF/tokenizer from a local model server; it is not deployed publicly. See the crate's own doc comments and `docs/archive/` for the archived tool-calling accuracy numbers.

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features web -p llm-wasm
wasm-pack build crates/llm-wasm --target web --no-default-features --features web
python3 web/agent/serve.py
```

## Credits

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Apache 2.0).
- [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (Apache 2.0).
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT), vendored at `web/vendor/coi-serviceworker.js` so the CPU threads backend works on GitHub Pages.
- Weights for all of the above are not distributed here; both demo pages fetch them from Hugging Face at run time.

## License

Code in this repository (excluding vendored/third-party components noted above, and excluding any model weights, which are not distributed here) is licensed under the [MIT License](LICENSE).
