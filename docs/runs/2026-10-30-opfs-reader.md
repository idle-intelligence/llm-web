# 2026-10-30: loading a GGUF from OPFS through `OpfsReader`

## Parameters

- Change: `OpfsReader` (`crates/lean/src/web.rs`), a `Read + Seek` over a `FileSystemSyncAccessHandle`; `Lean.load({gguf})` accepts the handle as well as a `Uint8Array`. Reads of 64 KiB or more go straight from the handle into the wasm buffer; smaller reads are served from one 64 KiB window of the file.
- Base: lean at 4ce38c4. New: the same commit plus this change.
- Machine: the Linux test machine (RTX 3080), Playwright's bundled Chromium 153.0.8010.12, headless, Vulkan flags; `requestAdapter()` reported vendor `nvidia`, architecture `ampere`, not a fallback adapter, in every run below.
- Parity harness: a scratch wasm crate (not in the repo) linking lean with the `web` and `diag` features. The bytes path is lean's own `LeanEngine.load(Uint8Array)` / `LeanEngineCpu.load(Uint8Array)` and `Lean.load({gguf: Uint8Array})`; the OPFS path is `GpuModel::load_from_reader` / `CpuModel::load_from_reader` over `OpfsReader`, and `Lean.load({gguf: handle})`. The GGUF was downloaded into OPFS first and checked against its SHA-256.
- Model: Qwen3-1.7B Q8_0 GGUF, 1,834,426,016 bytes, SHA-256 `061b54da...`, the LFS SHA-256 Hugging Face lists for `Qwen3-1.7B-Q8_0.gguf` in Qwen/Qwen3-1.7B-GGUF.
- Prompt (27 tokens): `<|im_start|>user\nWhat is the capital of France, and why? Answer in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, `maxCtx` 256; `Lean.generate` with `maxNewTokens` 48, `temperature` 0.
- Served bytes: the harness wasm served to the page hashed `76b29bc9...` (curl of the served URL, `data/2026-10-30-opfs/summary.json:parity2.servedSha256`), the same file as the build.
- Non-regression: llm-web's own `web/device/` (Run all) and `web/` chat page, built by `scripts/build.sh` from 4ce38c4 (`ENGINE_BUILD=base4ce38c4`, `lean_bg.wasm` `4f60e072...`) and from this change (`ENGINE_BUILD=opfs2`, `lean_bg.wasm` `1a88e9d1...`), served bytes hashed by curl before the runs, driven by Playwright like a user: pick the model and backend, download, type "What is the capital of France? Answer in one sentence.", Send, read the reply, screenshot.

## Results

Logits of the last prompt position, bytes path against OPFS path (`summary.json:parity2.result`):

| Forward pass | Logits | Bit-identical | Max abs diff | Argmax (bytes / OPFS) |
|---|---:|---|---:|---|
| WebGPU (`LeanEngine` / `GpuModel`) | 151,936 | yes | 0 | 785 / 785 |
| CPU single thread (`LeanEngineCpu` / `CpuModel`) | 151,936 | yes | 0 | 785 / 785 |

`Lean.load` with each source (`summary.json:parity2.result.lean_webgpu`, `lean_cpu`):

| Backend | Load ms, bytes | Load ms, OPFS | Same reply | Same prefix key | Same prefix image (SHA-256 of 6,193,280 bytes) |
|---|---:|---:|---|---|---|
| webgpu | 3,865 | 3,998 | yes | yes | yes |
| single | 511 | 519 | yes | yes | yes |

Load time before the 64 KiB window, same harness, one run each (`parity.json`, harness wasm `80f3d7f7...`): webgpu 3,881 ms bytes, 4,651 ms OPFS; single 501 ms bytes, 1,323 ms OPFS.

Non-regression, Chromium WebGPU and single CPU thread (`summary.json:e2e-base`, `e2e-new`):

| Page | Model | Backend | Base reply | New reply | Base decode tok/s | New decode tok/s |
|---|---|---|---|---|---:|---:|
| web/device | SmolLM2-360M Q4_0 | webgpu | same as transformers: yes | same as transformers: yes | n/a | n/a |
| web/device | SmolLM2-360M Q4_0 | single | same as transformers: yes | same as transformers: yes | n/a | n/a |
| web | Qwen2.5-0.5B Q4_0 | webgpu | The capital of France is Paris. | The capital of France is Paris. | 223.7 | 235.3 |
| web | Qwen2.5-0.5B Q4_0 | single | The capital of France is Paris. | The capital of France is Paris. | 6.9 | 6.9 |
| web | SmolLM2-360M Q4_0 | webgpu | The capital of France is Paris. | The capital of France is Paris. | 213.4 | 207.8 |
| web | SmolLM2-360M Q4_0 | single | The capital of France is Paris. | The capital of France is Paris. | 8.2 | 8.3 |

Same pages in Playwright's bundled Firefox (1543 build), headless on the same machine, which has no WebGPU there; Auto picked the single CPU thread (`summary.json:e2e-firefox-base`, `e2e-firefox-new`):

| Page | Model | Backend | Base reply | New reply | Base decode tok/s | New decode tok/s |
|---|---|---|---|---|---:|---:|
| web/device | SmolLM2-360M Q4_0 | single | same as transformers: yes | same as transformers: yes | n/a | n/a |
| web | Qwen2.5-0.5B Q4_0 | auto (single) | The capital of France is Paris. | The capital of France is Paris. | 6.8 | 7.0 |
| web | Qwen2.5-0.5B Q4_0 | single | The capital of France is Paris. | The capital of France is Paris. | 6.6 | 6.6 |
| web | SmolLM2-360M Q4_0 | auto (single) | The capital of France is Paris. | The capital of France is Paris. | 8.6 | 8.6 |
| web | SmolLM2-360M Q4_0 | single | The capital of France is Paris. | The capital of France is Paris. | 8.6 | 8.9 |

## Observations

- Both forward passes give bit-identical logits from the two sources, and `Lean.load` gives the same reply, prefix key and prefix image byte for byte.
- Without the window, the OPFS load was 770 ms (webgpu) and 822 ms (single) slower than the bytes load: the GGUF header is read in hundreds of thousands of small reads, one handle call each. With the window, the gap is 133 ms and 8 ms.
- The device page matches the transformers reference on both backends for both builds, and the chat page gives the same reply on both builds; decode speeds differ by at most 5% in single runs, on a 7-token reply.
- CPU threads were not built (`BUILD_THREADS` unset), so that backend is not covered here. WebKit could not start on this machine (missing system libraries) and is not covered here.
