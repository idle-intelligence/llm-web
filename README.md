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

## License

MIT
