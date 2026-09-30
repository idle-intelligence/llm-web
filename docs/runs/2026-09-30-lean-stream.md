# 2026-09-30 - lean: token streaming, sampling, multi-turn chat

Branch `lean-stream`, worktree off `lean-perf` (3c92d34). Commits: 77afb85
(streaming/sampling/chat API + tests + chat.html), plus this session's
follow-up (AbortFlag `cloneFlag` fix + ENGINE_BUILD bump, committed after
this doc).

## What changed

- `crates/lean/src/generate.rs` (new): `decode_loop`, the shared streaming
  decode loop used by both `web.rs` and native tests. Greedy
  (`SamplingParams::is_greedy()`) stays on `forward_decode_step_argmax`
  (4-byte GPU-argmax readback), unchanged from the pre-existing
  `generate()` path.
- `crates/lean/src/sampling.rs` (new): `SamplingParams`
  (temperature/top_k/top_p/repetition_penalty/seed), `sample()`
  (repetition-penalty -> temperature -> top-k -> top-p -> categorical
  draw), `Rng64` (splitmix64, no new crate dependency).
- `crates/lean/src/chat_template.rs`: added `render_conversation` (arbitrary
  multi-turn `(role, content)` history); `render_user_prompt` is now a thin
  wrapper over it.
- `crates/lean/src/model.rs`: added `pub fn argmax` (previously duplicated
  privately in `web.rs`/`lean_cli.rs`/etc).
- `crates/lean/src/web.rs`: `AbortFlag` (JS-shareable abort switch, `Rc<Cell<bool>>`),
  `generateStream` (single-turn streaming + sampling + mask + abort),
  `chatGenerate`/`chatReset` (multi-turn: renders the full conversation each
  turn via `render_conversation`, forwards only the new suffix onto the
  existing KV cache with `forward_prefill_suffix` instead of re-prefilling
  everything, then closes the assistant turn's trailing template text the
  same way).
- Follow-up fix: `AbortFlag::cloneFlag()`. wasm-bindgen destroys a by-value
  class argument's JS-side handle the instant it crosses into wasm
  (`__destroy_into_raw()` in the generated glue) - passing the caller's own
  `AbortFlag` into `chatGenerate`/`generateStream` left it unusable for a
  later `.abort()` call from JS (surfaced as a `null pointer passed to rust`
  browser console error during the Playwright check below). Fix: JS passes
  `flag.cloneFlag()` (a cheap `Rc` clone sharing the same underlying cell)
  instead of `flag` itself; `main_chat.js` updated to match.
- `crates/lean/tests/streaming_sampling.rs` (new, `--ignored`, needs
  `LEAN_GGUF`/`LEAN_TOKENIZER_DIR`): greedy-streaming-matches-whole-reply,
  seeded-sampling-reproducible-across-runs, multi-turn-append-matches-full-reprefill.
- `crates/lean/www/chat.html` + `main_chat.js` (new): manual streaming-chat
  test page, SmolLM2-360M-Instruct-Q4_0 local by default.
- ENGINE_BUILD bumped to `2026-09-30-stream-01` in every `www/*.js` that
  loads `../pkg/lean.js`/`lean_bg.wasm` (all of them load the one rebuilt
  wasm-pack output, including the CPU-rung pages).

## Gate results

| gate | command (`--ignored`, `--release`) | result | wall time |
|---|---|---|---|
| fixture_parity | `cargo test -p lean --test fixture_parity` | pass (1/1) | 124.7s |
| fixture_parity_qwen3 | `cargo test -p lean --test fixture_parity_qwen3` | pass (1/1) | 76.1s |
| fixture_parity_llama_360m | `cargo test -p lean --test fixture_parity_llama_360m` | pass (1/1) | 14.5s |
| kv_snapshot | `cargo test -p lean --test kv_snapshot` | pass (2/2) | 95.7s |
| logit_mask | `cargo test -p lean --test logit_mask` | pass (3/3) | 5.1s |
| streaming_sampling (new) | `cargo test -p lean --test streaming_sampling` | pass (3/3) | 4.9s |

All six run under `LEAN_GGUF`/`LEAN_TOKENIZER_DIR` = the Qwen2.5-0.5B-Instruct
Q4_0 GGUF/tokenizer (except `fixture_parity_qwen3`, Qwen3-0.6B-Q8_0, and
`fixture_parity_llama_360m`, SmolLM2-360M-Instruct Q4_0/Q8_0 - each test's
own required env vars), serialized through the shared `locked gpu`/
`locked cargo` wrappers on the shared Mac (queued behind another worker's
`llm-life` job for part of this session).

Native build/lint: `cargo check -p lean` and `cargo clippy -p lean
--all-targets -- -D warnings` clean; `cargo check`/`cargo clippy -p lean
--no-default-features --features web --target wasm32-unknown-unknown -- -D
warnings` clean.

## wasm-pack build + served-bytes proof

`wasm-pack build crates/lean --target web --out-dir pkg --features web`.

| step | sha256 (lean_bg.wasm) |
|---|---|
| first build (before AbortFlag fix) | `62245f6c7e243845bcf194db13c0885b0f256adf850be3fd7cf557a630f2ae5a` |
| rebuild (after AbortFlag `cloneFlag` fix) | `5a04b6127c8b17631b6f836e995284ba1c476e2b91b82d1c517e00bc7751adb3` |
| served by `python3 -m http.server 8222` at `/pkg/lean_bg.wasm`, fetched via curl after the rebuild | `5a04b6127c8b17631b6f836e995284ba1c476e2b91b82d1c517e00bc7751adb3` (matches) |

## Browser check (headless Playwright, SwiftShader software WebGPU)

`/tmp/pw-venv/bin/python3` + Chromium args `--enable-unsafe-webgpu
--enable-features=Vulkan --use-angle=swiftshader --use-gl=swiftshader
--ignore-gpu-blocklist`, run under `locked browser` only (server itself not
locked, port 8222 - not in the reserved list).

| check | result |
|---|---|
| `www/index.html?local=1` loads rebuilt wasm, device ready, model loaded, first fixture case `tokens_match=true` | pass, 0 console errors |
| `www/chat.html`: page ready (`window.__leanChatReady`), `LeanEngine.info()` reports `numLayers=32 hiddenSize=960 vocabSize=49152` (SmolLM2-360M) | pass |
| `chatGenerate` streams tokens (`onToken` fires per token) | pass - 1 token within 1.5s of send, 24 tokens accumulated by the second turn |
| `AbortFlag`/`cloneFlag`: stop button aborts mid-generation, token count does not advance after abort (24 at abort, still 24 after a 2s wait) | pass |
| console errors during the chat check | 0 (was 1: `null pointer passed to rust`, before the `cloneFlag` fix above) |

`www/index.html?local=1`'s full fixture+kv-snapshot+mask harness was not run
to completion under SwiftShader (multi-minute under software rendering,
out of scope for this check) - the check above only confirms the page still
loads the rebuilt wasm and produces correct early output, matching the
"re-check existing pages still load" ask.

## Not done / left for the next session

- `crates/lean/pkg/` is `.gitignore`d (wasm-pack output is never committed)
  - publishing it (e.g. to gh-pages) for trucs.ai/stt-llm-tts to consume is
    unaddressed.
- No timing/throughput numbers for `generateStream`/`chatGenerate` (this
  session was correctness-only, run on a shared, sometimes-contended Mac -
  no timing claims per the shared-machine rule).
- Wiring lean into trucs.ai/stt-llm-tts (Web Worker, mic capture, replacing
  WebLLM) is frontend-browser-audio's scope, not touched here.
