# 2026-10-04: one chat API on every lean backend, streamed text

Two problems came up while moving trucs.ai's LLM pages to lean:

1. `LeanEngine` (WebGPU) had `chatGenerate`/`chatReset`, `LeanEngineCpu`
   did not, so a page could not hold a multi-turn chat on the CPU.
2. On WebGPU with SmolLM2-1.7B-Instruct Q4_0, `generate()`'s callback fired
   for every token but the page showed no text.

## Cause of the empty output

The page's callback called `engine.decodeIds([id])` from inside
`engine.generate(...)`. A wasm-bindgen method that takes `&mut self` keeps
the engine borrowed for the whole call, including across its awaits, so
the nested `decodeIds` threw `recursive use of an object detected which
would lead to unsafe aliasing in rust` on every token. The engine called
the callback with `let _ = f.call1(...)`, which dropped the exception, so
nothing reported it.

Repro against the lean-main build (`2026-10-03-main-01`), headless
Chromium, WebGPU, SmolLM2-1.7B-Instruct Q4_0, prompt "Hello! Who are
you?", 16 tokens:

| field | value |
|---|---|
| ids passed to the callback | 19556, 17, 339, 5248, 3511, 308, 34519, 28, 253, 1165, 1789, 1743, 2724, 288, 4237, 351 |
| `decodeIds` calls that threw | 16 of 16 |
| text returned by `generate()` | "Hello! I'm SmolLM, a small language model designed to assist with" |

The ids and the returned text were right; only the per-token text was
lost. The same pattern was in `www/main_chat.js`, whose test hook counted
tokens and so never saw it.

Decoding one id at a time is also wrong on its own: a character that spans
several byte-level BPE tokens comes out as U+FFFD halves. On a
multilingual test string the per-id decode differs from the full decode
for both tokenizers (`text_stream_*` below).

## What changed

- `src/chat.rs` (new), shared by both engines and the native tests:
  - `ChatSession`: the conversation plus the exact ids in the KV cache.
    Each turn renders the whole conversation through the model's chat
    template and prefills only what follows the longest common prefix
    with the cached ids. Any other call that writes the cache
    (`generate`, `prefillTokens`, `appendTokens`, `decodeStepArgmax`,
    `restoreKv`) makes the next turn prefill from position 0.
  - `TextStream`: decodes a sliding window of the accumulated ids and
    returns only the new text, holding back a trailing incomplete
    character. The deltas concatenate to the full decode.
  - `gpu_chat_turn`, `cpu_chat_turn`, `cpu_decode_loop` (the CPU mirror of
    `generate::decode_loop`, with masks applied to the logits as
    `mask_logits.wgsl` does).
- `src/web.rs`:
  - `LeanEngineCpu.chatGenerate` / `chatReset`, same signatures as
    `LeanEngine`'s. The CPU `chatGenerate` is async and yields to the
    event loop after every token (a `MessageChannel` message), so a stop
    message reaching the worker can flip the `AbortFlag`.
  - Every generating call (`generate`, `generateStream`, `chatGenerate`,
    both engines) calls `on_token(id, text)`, where `text` is the reply
    text that token completes. A final call with `id = -1` carries text
    still held back at the end, if any. A callback that throws stops
    generation and the call rejects with the message.
- `www/chat.html`: inference in `chat_worker.js`, backend by capability or
  `?backend=webgpu|threads|single`, `?model=qwen25-0.5b|smollm2-360m|smollm2-1.7b`,
  `?max=N`; the reply shows the `text` argument.

JS surface, identical on `LeanEngine` and `LeanEngineCpu`:

```js
await engine.chatGenerate(prompt, maxNewTokens, temperature, topK, topP,
  repetitionPenalty, seed, maskBits /* Uint32Array, empty = none */,
  (id, text) => { /* id = -1 only for a final flush */ },
  abortFlag.cloneFlag() /* or undefined */);   // -> Promise<string>
engine.chatReset();
```

## Parameters

- M2 laptop. Native tests on Metal, `--release`. Browser: Playwright's
  bundled headless Chromium; WebGPU runs with `--enable-unsafe-webgpu
  --use-angle=metal` (adapter `apple / metal-3`), CPU runs without them
  (no adapter).
- Models: Qwen2.5-0.5B-Instruct Q4_0, SmolLM2-360M-Instruct Q4_0,
  SmolLM2-1.7B-Instruct Q4_0 (bartowski).
- Conversation for both the native test and the browser: "What is the
  capital of France?", "Name two famous museums there, one sentence
  each.", "Écris une phrase en français sur la Seine, avec un emoji.";
  greedy, at most 48 new tokens per turn, max_ctx 1024 native / 2048
  browser.
- wasm builds: `wasm-pack build crates/lean --target web --release
  --no-default-features --features web` and `scripts/build_lean_mt.sh`'s
  steps, both with `--remap-path-prefix` for the home directory.
  ENGINE_BUILD `2026-10-04-chat-01`.

## Results

### Native gates (lean-chat-api at 4b6da77, M2, `--release`)

| test | features | result | test time |
|---|---|---|---|
| fixture_parity (both kernel paths) | default | ok | 58.5 s |
| fixture_parity_qwen25_3b | default | ok | 40.6 s |
| fixture_parity_qwen3 | default | ok | 55.1 s |
| fixture_parity_qwen3_1_7b | default | ok | 155.2 s |
| fixture_parity_llama_360m | default | ok | 13.9 s |
| fixture_parity_llama_1_7b | default | ok | 35.4 s |
| fixture_parity_llama_360m_cpu | default | ok | 13.2 s |
| fixture_parity_llama_360m_cpu | threads | ok | 5.4 s |
| kv_snapshot | default | ok, 2 tests | 40.3 s |
| logit_mask | default | ok, 3 tests | 4.0 s |
| pool_reuse | default | ok | 3.0 s |
| streaming_sampling | default | ok, 3 tests | 4.5 s |
| chat_api (new) | default | ok, 5 tests | 17.3 s |
| chat_api (new) | threads | ok, 5 tests | 10.0 s |
| embed_head_sliced (LEAN_LORA_BIN2 = a-norules-300) | default | ok, 2 tests | 4.3 s |
| lora_parity | default | ok, 2 tests | 3.2 s |
| cpu_lora_parity | threads | ok | 112.6 s |
| lib tests | default | ok, 17 | 0.2 s |
| lib tests | threads | ok, 21 | 0.2 s |
| clippy -D warnings, native all targets | default | clean | |
| clippy -D warnings, native all targets | threads | clean | |
| clippy -D warnings, wasm32 `web` | | clean | |

cpu_lora_parity was run with threads only; this branch does not change
`cpu.rs`, `cpu_kernels.rs` or `lora.rs`.

### chat_api, 3-turn greedy conversation, GPU vs CPU (native)

| model | turn | tokens | GPU ids == CPU ids | deltas concat == reply |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 1 | 48 | yes | yes |
| Qwen2.5-0.5B Q4_0 | 2 | 19 | yes | yes |
| Qwen2.5-0.5B Q4_0 | 3 | 16 | yes | yes |
| SmolLM2-360M Q4_0 | 1 | 7 | yes | yes |
| SmolLM2-360M Q4_0 | 2 | 28 | yes | yes |
| SmolLM2-360M Q4_0 | 3 | 27 | yes | yes |

| text_stream test | ids | deltas concat == full decode | per-id decode == full decode |
|---|---|---|---|
| Qwen2.5 tokenizer | 31 | yes | no |
| SmolLM2 tokenizer | 51 | yes | no |

`smollm2_1_7b_q4_0_gpu_streams_text`: 32 ids, reply "Hello! I'm SmolLM, a
small language model designed to assist with text-based inquiries. I'm
part of the Hugging Face library, a", deltas concatenate to it.

### wasm bytes

| file | sha256 | `/Users/` strings |
|---|---|---|
| pkg/lean_bg.wasm | c815ec8a90803fd8bcf54fcdfab535e9378213573702245783d7c11a4d7f54a5 | 0 |
| pkg-mt/lean_bg.wasm | 6506af5d46622d196859f6b9954d519624b792a7b1450131d7d0c8f323f8c1a2 | 0 |

The files served on port 8807 at `?v=2026-10-04-chat-01` hash to the same
values.

### Browser, M2, www/chat.html, same 3-turn conversation, max 48 tokens

| model | backend | turn tokens | reply == streamed == shown | console errors | replies equal native GPU |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | webgpu | 48, 19, 16 | yes | 0 | yes |
| Qwen2.5-0.5B Q4_0 | threads | 48, 19, 16 | yes | 0 | yes |
| Qwen2.5-0.5B Q4_0 | single | 48, 19, 16 | yes | 0 | yes |
| SmolLM2-360M Q4_0 | webgpu | 7, 28, 27 | yes | 0 | yes |
| SmolLM2-360M Q4_0 | threads | 7, 28, 27 | yes | 0 | yes |
| SmolLM2-360M Q4_0 | single | 7, 28, 27 | yes | 0 | yes |
| SmolLM2-1.7B Q4_0 | webgpu | 7, 42, 22 | yes | 0 | |
| SmolLM2-1.7B Q4_0 (max 16) | single | 7, 16, 16 | yes | 0 | |
| SmolLM2-1.7B Q4_0 | threads | load fails: `unreachable` | | | |

| check | result |
|---|---|
| stop mid-reply, webgpu, Qwen2.5-0.5B | stopped after 8 tokens, next turn answered ("The dragon was learning to cook.") |
| stop mid-reply, threads, Qwen2.5-0.5B | stopped after 7 tokens, next turn answered ("The dragon was cooking.") |
| no `?backend`, no WebGPU flags | picked threads, 3 turns ok |
| no `?backend`, WebGPU flags | picked webgpu, 3 turns ok |
| callback calling `decodeIds` on the new build (CPU) | call rejects: "on_token callback threw: recursive use of an object detected which would lead to unsafe aliasing in rust" |

## Observations

- The empty output was the page's nested `decodeIds` call throwing inside
  the callback (16 of 16 calls) while the engine dropped the exception;
  the ids and the returned text were correct on WebGPU with SmolLM2-1.7B
  Q4_0.
- Over three greedy turns the CPU produces the GPU's ids on both models,
  even though the GPU prefills turns 2 and 3 one token at a time
  (`forward_prefill_suffix`) and the CPU prefills them as one chunk.
- In the browser all three backends give the same replies as the native
  GPU run for both Qwen2.5-0.5B and SmolLM2-360M, and the text shown on the
  page equals the returned reply in every turn.
- Decoding one id at a time differs from the full decode on both
  tokenizers for the multilingual string; the streamed deltas match it.
- SmolLM2-1.7B Q4_0 does not load on the threads build: the wasm traps
  (`unreachable`) during `load`. The single-thread build loads and runs
  it. `scripts/build_lean_mt.sh` links pkg-mt with
  `--max-memory=1073741824` (1 GiB), and the CPU backend keeps the
  ~1 GB of quantized weights resident, so this is likely that cap; not
  changed here.
- The two stopped replies differ in length (8 vs 7 tokens) because the
  stop lands at a different point, so the turn after them differs too.
