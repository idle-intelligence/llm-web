# llm-web

A Rust/Burn WebGPU engine for running an LLM tool-calling agent client-side, with a wllama/WASM fallback for a live chat demo.

[**Try the demo →**](https://idle-intelligence.github.io/llm-web/web/)

> **Disclaimer:** The engine (`crates/llm-wasm/`) is an original Burn+wgpu implementation of the Qwen2 architecture, built from the model's public config and GGUF/tokenizer metadata, not a port of wllama or llama.cpp. It targets [Salesforce/xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r), licensed CC-BY-NC-4.0 and released by Salesforce for research purposes only; model weights are not distributed in this repository. This project is not affiliated with or endorsed by Salesforce.

## Status

- The Burn+wgpu engine runs the full Qwen2 forward pass (GQA attention, QKV bias, RoPE, RMSNorm, SwiGLU, tied embeddings) natively and compiles to `wasm32-unknown-unknown` with WebGPU, verified by a native CLI (`llm-agent`) and a browser demo page under `web/agent/`.
- Schema-constrained decoding forces tool calls onto valid JSON matching the tool schema. On a 43-utterance Sonos tool-calling eval, constrained decoding scores 76.7% correct against a 53.3% unconstrained baseline (`docs/BENCHMARKS.md`).
- An MCP-shaped agent loop (`agent.rs`/`web.rs`) drives multi-step tool calls, retries malformed output, and feeds tool errors back to the model.
- Prefix KV cache images let a session restore GPU KV state to the longest matching prompt prefix instead of re-prefilling from scratch.
- The public demo above runs the wllama fallback path (SmolLM2-360M-Instruct), which is deployed to GitHub Pages. The Burn+wgpu engine demo (`web/agent/`) is a local dev page: it loads the GGUF/tokenizer from a local model server and is not currently deployed publicly.

## Models

| Model | Size | Params | Quant | License |
|-------|------|--------|-------|---------|
| [xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r) (Qwen2 arch) | ~1.8 GB (Q4_0 GGUF) | 3B | Q4_0 (Q6_K token embedding) | CC-BY-NC-4.0, research only |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | ~271 MB | 360M | Q4_K_M | Apache 2.0 |

## Structure

```
crates/llm-wasm/   # The engine: GGUF loader, Qwen2 model, WGSL kernels, tokenizer/template/agent, WASM bindings
eval/              # Sonos MCP tool-calling eval harness and results
fixtures/          # Reference tensors, rendered prompts, canned tool results
scripts/headless/  # Playwright-driven harness to load and benchmark a page in headless Chromium
web/agent/         # Local dev demo for the Burn+wgpu engine (xLAM-2-3b-fc-r, tool calling)
pkg/wllama/        # Vendored @wllama/wllama ESM build + WASM binaries
web/index.html     # Public demo page, wllama fallback: download model, chat, streaming output
```

## Build

```bash
# Native (default features: wgpu + native)
cargo build
cargo test

# WASM (web feature; native/CLI excluded)
cargo build --target wasm32-unknown-unknown --no-default-features --features web -p llm-wasm

# wasm-pack, for the browser demo's pkg/
wasm-pack build crates/llm-wasm --target web --no-default-features --features web
```

## Run locally

Public wllama demo:

```bash
npx serve -l 9000 --no-clipboard
# Open http://localhost:9000/web/
```

Engine agent demo (needs a local GGUF + tokenizer server, e.g. `scripts/serve_models.py`, and the COOP/COEP-enabled page server):

```bash
python3 web/agent/serve.py
# Open http://localhost:8002/ (or the port serve.py prints) and point it at your model server
```

Native CLI (run/eval/bench against a local GGUF file):

```bash
cargo run --release --bin llm-agent -- run --gguf <path-to-gguf> --tokens <path-to-tokens.json> --tokenizer <path-to-tokenizer.json>
```

## Credits

- [wllama](https://github.com/nicebyte/wllama) (MIT) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), the vendored browser-inference path under `pkg/wllama/` and `web/index.html`.
- [Salesforce/xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r), the model the engine targets. CC-BY-NC-4.0, research purposes only; weights are not distributed here.
- [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (Apache 2.0), the model served by the public wllama demo.

## License

Code in this repository (excluding vendored/third-party components noted above, and excluding any model weights, which are not distributed here) is licensed under the [MIT License](LICENSE).
