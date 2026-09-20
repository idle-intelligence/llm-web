# llm-web

Self-hosted [wllama](https://github.com/nicebyte/wllama) (v2.1.1) for browser-based LLM inference via WebAssembly.

Part of [Idle Intelligence](https://idleintelligence.org/).

## Demo

[idle-intelligence.github.io/llm-web/web/](https://idle-intelligence.github.io/llm-web/web/)

## Structure

```
pkg/wllama/       # Vendored @wllama/wllama ESM build + WASM binaries
web/index.html    # Demo page — download model, chat, streaming output
```

## Usage

Import from GitHub Pages:

```js
import { Wllama } from 'https://idle-intelligence.github.io/llm-web/pkg/wllama/index.js';

const wllama = new Wllama({
  'single-thread/wllama.wasm': 'https://idle-intelligence.github.io/llm-web/pkg/wllama/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': 'https://idle-intelligence.github.io/llm-web/pkg/wllama/multi-thread/wllama.wasm',
});
```

## Local development

```bash
npx serve -l 9000 --no-clipboard
# Open http://localhost:9000/web/
```

## Credits

- [wllama](https://github.com/nicebyte/wllama) (MIT) — the original vendored
  browser-inference demo under `pkg/wllama/` and `web/index.html`.
- The Rust `llm-wasm` engine (`crates/llm-wasm/`) is an original Burn+wgpu
  implementation of the Qwen2 architecture, written from the model's public
  config and GGUF/tokenizer metadata — not a port of wllama or llama.cpp.
- Evaluation and benchmark fixtures target
  [`Salesforce/xLAM-2-3b-fc-r`](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r)
  (`Qwen2ForCausalLM` architecture, fine-tuned from an unnamed Qwen2.5-class
  base). That model is licensed **CC-BY-NC-4.0** and released by Salesforce
  for **research purposes only** — see its model card for full terms. Model
  weights are not distributed in this repository.

## License

Code in this repository (excluding vendored/third-party components noted
above, and excluding any model weights, which are not distributed here) is
licensed under the [MIT License](LICENSE).
