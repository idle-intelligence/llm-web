# llm-web

Two Rust WebGPU inference engines for language models, running client-side in the browser, plus a wllama fallback demo.

- `crates/llm-wasm/` is the original engine: a Burn+wgpu implementation of the Qwen2 architecture, with quantized GGUF weights, runtime LoRA adapters, schema-constrained decoding and a tool-calling agent loop.
- `crates/lean/` is a second, newer engine: a hand-written raw-wgpu + WGSL forward pass (no Burn, no CubeCL), native and wasm from one source, that picks a rung (WebGPU, or a single-threaded CPU fallback with NEON/simd128 kernels) by capability rather than by runtime measurement.

[**Try the demo →**](https://idle-intelligence.github.io/llm-web/web/)

> **Disclaimer:** both engines are original implementations written from public model configs and GGUF metadata, not ports of wllama or llama.cpp. Models are fetched at run time from their authors' Hugging Face repos under their own licenses; no weights are redistributed here. This project is not affiliated with the Qwen team or Salesforce.

## llm-wasm (Burn+wgpu engine)

### Status

- The Burn+wgpu engine runs the full Qwen2 forward pass (GQA attention, QKV bias, RoPE, RMSNorm, SwiGLU, tied embeddings) natively and compiles to `wasm32-unknown-unknown` with WebGPU, verified by a native CLI (`llm-agent`) and a browser demo page under `web/agent/`.
- Runs Qwen2.5-0.5B-Instruct (Q4_0) with runtime LoRA adapters in the browser. This is the engine behind the LLM methods of [llm-life](https://github.com/idle-intelligence/llm-life), where fine-tuned adapters turn the model into a Game of Life update rule.
- Schema-constrained decoding forces tool calls onto valid JSON matching the tool schema. On a 43-utterance Sonos tool-calling eval, constrained decoding scores 76.7% correct against a 53.3% unconstrained baseline (`docs/BENCHMARKS.md`).
- An MCP-shaped agent loop (`agent.rs`/`web.rs`) drives multi-step tool calls, retries malformed output, and feeds tool errors back to the model.
- Prefix KV cache images let a session restore GPU KV state to the longest matching prompt prefix instead of re-prefilling from scratch.
- The public demo above runs the wllama fallback path (SmolLM2-360M-Instruct), which is deployed to GitHub Pages. The Burn+wgpu engine demo (`web/agent/`) is a local dev page: it loads the GGUF/tokenizer from a local model server and is not currently deployed publicly.

### Models

| Model | Size | Params | Quant | License |
|-------|------|--------|-------|---------|
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ~430 MB (Q4_0 GGUF) | 0.5B | Q4_0 | Apache 2.0 |
| [xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r) (Qwen2 architecture, tool calling) | ~1.8 GB (Q4_0 GGUF) | 3B | Q4_0 (Q6_K token embedding) | CC-BY-NC-4.0, research only |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | ~271 MB | 360M | Q4_K_M | Apache 2.0 |

### Structure

```
crates/llm-wasm/   # The engine: GGUF loader, Qwen2 model, WGSL kernels, tokenizer/template/agent, WASM bindings
eval/              # Sonos MCP tool-calling eval harness and results
fixtures/          # Reference tensors, rendered prompts, canned tool results
scripts/headless/  # Playwright-driven harness to load and benchmark a page in headless Chromium
web/agent/         # Local dev demo for the Burn+wgpu engine (xLAM-2-3b-fc-r, tool calling)
pkg/wllama/        # Vendored @wllama/wllama ESM build + WASM binaries
web/index.html     # Public demo page, wllama fallback: download model, chat, streaming output
```

### Build

```bash
# Native (default features: wgpu + native)
cargo build
cargo test

# WASM (web feature; native/CLI excluded)
cargo build --target wasm32-unknown-unknown --no-default-features --features web -p llm-wasm

# wasm-pack, for the browser demo's pkg/
wasm-pack build crates/llm-wasm --target web --no-default-features --features web
```

### Run locally

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

## lean (raw wgpu engine)

`crates/lean/` is a second engine in this repo: a forward pass written directly against `wgpu` and hand-written WGSL, with no Burn and no CubeCL. It compiles from one source to native and to `wasm32-unknown-unknown`, and it picks a rung by capability rather than by runtime measurement or per-device tuning: WebGPU when available, otherwise a single-threaded CPU forward pass (`cpu.rs`) with NEON (aarch64) or simd128 (wasm32) dot-product kernels, otherwise a plain scalar fallback.

### Supported architectures and quant types

Architectures (`src/config.rs`): Qwen2, Qwen3, and Llama (including SmolLM2, which uses the Llama architecture). GGUF tensor dtypes (`src/gguf.rs`): F32, F16, Q4_0, Q8_0, and Q6_K (Q4_1-encoded tensors are read and dequantized to F32). Which dtypes a given tensor uses depends on how the GGUF was quantized; llama.cpp's "Q4_0" quant profile commonly keeps `token_embd.weight`/`output.weight` at Q6_K or Q8_0.

### Build

```bash
# Native
cargo build -p lean
cargo test -p lean

# WASM (web feature; native CLI bins excluded)
cargo build -p lean --target wasm32-unknown-unknown --no-default-features --features web

# wasm-pack, for the www/ demo pages
wasm-pack build crates/lean --target web --no-default-features --features web
```

### Run locally

Native CLI (fixture-parity check or free-form prompt against a local GGUF + tokenizer):

```bash
LEAN_GGUF=<path-to-gguf> LEAN_TOKENIZER_DIR=<dir-with-tokenizer.json> cargo run -p lean --release --bin lean-cli
# or: cargo run -p lean --release --bin lean-cli -- --prompt "..." --tokens 64
```

www demo pages (`crates/lean/www/index.html` and friends: `cpu.html`, `qwen3.html`, `qwen25_3b.html`, `decode_timing.html`, `mem_profile*.html`) load a `wasm-pack`-built `pkg/` plus a locally hosted GGUF and tokenizer with `?local=1`. Serve the crate directory over HTTP and open the page:

```bash
cd crates/lean && python3 -m http.server 8000
# Open http://localhost:8000/www/index.html?local=1
```

### Tests

`cargo test -p lean` always runs the synthetic (non-GPU) tests: `chat_template.rs`, `lora.rs`'s parser tests, `quant.rs`'s dequant-repack tests, and `q6k_reference_dequant.rs`. The GPU-backed parity tests are `#[ignore]`d because they need real model files not committed to this repo; run them explicitly with the matching env vars set, for example:

```bash
LEAN_GGUF=<path> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --test fixture_parity -- --ignored
LEAN_GGUF=<path> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --test kv_snapshot -- --ignored
LEAN_GGUF=<path> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean --test logit_mask -- --ignored
```

Each of `fixture_parity_qwen3.rs`, `fixture_parity_qwen3_1_7b.rs`, `fixture_parity_qwen25_3b.rs`, `fixture_parity_llama_360m.rs`, and `fixture_parity_llama_1_7b.rs` reads its own model-specific env vars (e.g. `LEAN_GGUF_QWEN3`/`LEAN_TOKENIZER_DIR_QWEN3`); see each test file's `env::var` calls for the exact names.

### Performance

Benchmark and profiling runs (native and browser, prefill/decode timing, kernel-level breakdowns) are logged under `docs/runs/` (for example `docs/runs/2026-09-29-lean-vs-llamacpp-profile.md` for a comparison against llama.cpp on the same GGUF). Those docs are the source of truth for any performance number; this README doesn't restate them.

## Credits

- [wllama](https://github.com/nicebyte/wllama) (MIT) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), the vendored browser-inference path under `pkg/wllama/` and `web/index.html`.
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache 2.0) and [Salesforce/xLAM-2-3b-fc-r](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r) (CC-BY-NC-4.0, research only), the models run on the engine so far; weights are not distributed here.
- [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (Apache 2.0), the model served by the public wllama demo.

## License

Code in this repository (excluding vendored/third-party components noted above, and excluding any model weights, which are not distributed here) is licensed under the [MIT License](LICENSE).
