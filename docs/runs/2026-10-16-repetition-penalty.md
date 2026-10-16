# Repetition penalty: once per distinct id, as transformers

Change: lean applied the repetition penalty once per occurrence of an id in the history, on every path: the full-logits CPU sampler (`lean::sampling::sample`) looped over the history, and the resident sampler (`sample_penalty.wgsl` and its CPU reference `resident_sampling::sample_reference`, used by every backend for top-k in 1..=63) looped `count` times. transformers' `RepetitionPenaltyLogitsProcessor` gathers the logits of `input_ids`, penalizes them and scatters them back, so each distinct id is penalized once however often it occurs. After the change the CPU sampler penalizes the deduplicated history and the kernel and its reference penalize every id whose count is nonzero, once. The counts buffer, the order of operations (penalty, temperature, top-k, top-p, draw), the random streams and greedy decoding are unchanged.

## Parameters

- Before: `browser-perf` at 2f381ca. After: branch `rep-penalty-hf`, fix at f150fb5 (fixture d70bff4, test knob 3b1644f).
- Reference: transformers 5.17.0, torch 2.14.0, CPU, f32. `RepetitionPenaltyLogitsProcessor.__call__` in `transformers/generation/logits_process.py`: `torch.gather(scores, 1, input_ids)` (line 409), `torch.where(score < 0, score * self.penalty, score / self.penalty)` (line 411), `scores.scatter(1, input_ids, score)` (line 412). `generate` builds it with the penalty only (`generation/utils.py` lines 1303-1304), so `prompt_ignore_length` is unset and `input_ids` is the whole sequence, prompt plus generated ids.
- llama.cpp (checkout 38a5b42, 2026-09-15): `llama_sampler_penalties_apply` in `src/llama-sampler.cpp` (lines 2950-2979) looks each candidate up in `token_count` and applies `penalty_repeat` once (`logit <= 0` multiplies, otherwise divides); the count only scales the frequency and presence penalties. The window is the last `penalty_last_n` accepted ids (default 64, `common/common.h` line 239; ring kept in lines 2921-2935), and prompt ids are accepted into it (`tools/server/server-context.cpp` lines 419-424, `tools/completion/completion.cpp` lines 693-695).
- Fixture: `crates/lean-core/reference/gen_fixture_rep_penalty.py` writes `fixture_rep_penalty.json`. Models: SmolLM2-360M-Instruct and Qwen2.5-0.5B-Instruct, the Q4_0 GGUFs lean reads, loaded and dequantized by transformers. Two prompts per model (a five-fruit list, "repeat after me three times"), chat template, plus a 48-token greedy continuation, so the history repeats ids (90 to 96 ids, 35 to 60 distinct, an id occurring up to 10 times). Next-token logits from one f32 forward over the whole sequence; processor applied at penalties 1.1 and 1.3. The fixture stores the raw and processed logit of each distinct history id as f32 bits; the script asserts every other logit is unchanged.
- Parity tests: `cpu_penalty_matches_transformers` (lean, `src/tests/rep_penalty_parity.rs`) runs `sample` at temperature 1, no top-k, no top-p, which leaves the penalized logits in place; `gpu_penalty_matches_transformers` (lean-core, `src/tests/resident_sampling.rs`) runs `sample_gpu` with the counts buffer as `decode_sampled_pipelined` fills it and reads the logits buffer back. Ids outside the history get fixed filler logits. Max abs diff over the whole vocabulary.
- Native sampled: `sampled_native` (chat page sampler: temperature 0.7, top-k 40, top-p 0.9, seed 0; prompt "Write a long story about a lighthouse keeper."), penalty 1.1 and, through `LEAN_PENALTY`, 1.0; 128 and 256 new tokens, three runs each (asserted equal); md5 of the written replies. Backends WebGPU (Metal) and CPU.
- Browsers: `scripts/bench` (bench.html, bench_worker.js) served by `serve.mjs`, phases `exact,decode,sampled`, `decode=128`, `sampledN=128,256`, one load per cell. Hashes: first 8 bytes of the SHA-256 of the reply text. Exactness = greedy replies to three fixed questions (48 tokens each); greedy 128 = the decode phase's reply; sampled = the chat page's sampler with penalty 1.1, seed 0. Playwright 1.63.0 bundled Chromium (headless) and WebKit (headless), Firefox Developer Edition 159.0b1 (headed, throwaway profile, `dom.webgpu.enabled`).
- Builds: `scripts/build.sh`, single-thread package, ENGINE_BUILD `2026-10-16-rp-base` and `2026-10-16-rp-fix` (local tags, not committed; the committed loading URLs are not changed by this branch). Served wasm fetched back from each server, md5: base 649569c777ec820eb906947495b2e27d, fix 7b772b0d856fcab8c8095b8f8c3a5132; both keep `+simd128`.
- Machine: the M2 laptop (16 GB), Rust 1.99.0 (pinned toolchain), release builds. 1-minute load average at the start of the browser cells 2.29 to 4.43; the tok/s figures below are not timing claims.

## Results

### Post-penalty logits against transformers (8 cases: 2 models x 2 prompts x 2 penalties)

| Path | Model | Penalty | Before: max abs diff | Before: logits differing | After: max abs diff | After: logits differing |
|---|---|---|---|---|---|---|
| CPU `sample` | SmolLM2-360M, fruits | 1.1 | 8.754 | 15 | 0 | 0 |
| CPU `sample` | SmolLM2-360M, fruits | 1.3 | 12.79 | 15 | 0 | 0 |
| CPU `sample` | SmolLM2-360M, repeat | 1.1 | 6.384 | 20 | 0 | 0 |
| CPU `sample` | SmolLM2-360M, repeat | 1.3 | 10.41 | 20 | 0 | 0 |
| CPU `sample` | Qwen2.5-0.5B, fruits | 1.1 | 4.193 | 14 | 0 | 0 |
| CPU `sample` | Qwen2.5-0.5B, fruits | 1.3 | 5.579 | 14 | 0 | 0 |
| CPU `sample` | Qwen2.5-0.5B, repeat | 1.1 | 4.226 | 16 | 0 | 0 |
| CPU `sample` | Qwen2.5-0.5B, repeat | 1.3 | 7.835 | 16 | 0 | 0 |

| Path | Model | Penalty | Before: max abs diff | Before: max ulp | After: max abs diff | After: max ulp |
|---|---|---|---|---|---|---|
| GPU `sample_penalty.wgsl` | SmolLM2-360M, fruits | 1.1 | 8.754 | 9149270 | 9.5e-7 | 1 |
| GPU `sample_penalty.wgsl` | SmolLM2-360M, fruits | 1.3 | 12.79 | 25351117 | 9.5e-7 | 1 |
| GPU `sample_penalty.wgsl` | SmolLM2-360M, repeat | 1.1 | 6.384 | 6253347 | 9.5e-7 | 1 |
| GPU `sample_penalty.wgsl` | SmolLM2-360M, repeat | 1.3 | 10.41 | 15919446 | 1.9e-6 | 1 |
| GPU `sample_penalty.wgsl` | Qwen2.5-0.5B, fruits | 1.1 | 4.193 | 10706266 | 1.9e-6 | 1 |
| GPU `sample_penalty.wgsl` | Qwen2.5-0.5B, fruits | 1.3 | 5.579 | 28338650 | 9.5e-7 | 1 |
| GPU `sample_penalty.wgsl` | Qwen2.5-0.5B, repeat | 1.1 | 4.226 | 6644521 | 1.9e-6 | 1 |
| GPU `sample_penalty.wgsl` | Qwen2.5-0.5B, repeat | 1.3 | 7.835 | 18988497 | 9.5e-7 | 1 |

### Native sampled replies (md5 of the 128 and 256 token replies)

| Model | Backend | Penalty | Before | After |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | WebGPU | 1.1 | 3ae5ddc5eae410f0de41158748346338 | 4b5fc0cb72a04e0ec30372e18b7e9a9a |
| Qwen2.5-0.5B Q4_0 | CPU | 1.1 | 3ae5ddc5eae410f0de41158748346338 | 4b5fc0cb72a04e0ec30372e18b7e9a9a |
| SmolLM2-360M Q4_0 | WebGPU | 1.1 | d4f0a9524876ae5f7443eb3bce8e635f | 7f0078a876ee03638311be373fe9ef9d |
| SmolLM2-360M Q4_0 | CPU | 1.1 | d4f0a9524876ae5f7443eb3bce8e635f | 7f0078a876ee03638311be373fe9ef9d |
| SmolLM2-1.7B Q4_0 | WebGPU | 1.1 | 0d08128fd723f6dfb4118e47e9df03b7 | fed4e822d0848e85e39b9618690b133e |
| Qwen2.5-0.5B Q4_0 | WebGPU | 1.0 | 468d2e7c7f2cb3e9e60a656dedfde9e3 | 468d2e7c7f2cb3e9e60a656dedfde9e3 |
| Qwen2.5-0.5B Q4_0 | CPU | 1.0 | 468d2e7c7f2cb3e9e60a656dedfde9e3 | 468d2e7c7f2cb3e9e60a656dedfde9e3 |
| SmolLM2-360M Q4_0 | WebGPU | 1.0 | dcaca5b385094cfc089de6400401c226 | dcaca5b385094cfc089de6400401c226 |
| SmolLM2-360M Q4_0 | CPU | 1.0 | dcaca5b385094cfc089de6400401c226 | dcaca5b385094cfc089de6400401c226 |
| SmolLM2-1.7B Q4_0 | WebGPU | 1.0 | 6fcec55f63e10cd337afaad0e48b227f | 6fcec55f63e10cd337afaad0e48b227f |

### Browsers, WebGPU: greedy hashes (identical in Chromium, WebKit and Firefox Dev 159, before and after)

| Model | Exactness | Greedy 128 |
|---|---|---|
| Qwen2.5-0.5B Q4_0 | 29a934d304c7e868 | fdcf35d7e2636db1 |
| SmolLM2-360M Q4_0 | d908b4954ae69f95 | 3fda34ebd775825a |
| SmolLM2-360M Q4_K_M | c7a02253d192ee24 | 709752f9b3a2ce93 |
| Qwen3-0.6B Q8_0 | b222e3caf4f106d1 | 1480f1d6e12739de |
| SmolLM2-1.7B Q4_0 | efa85acfe7945318 | 941622dd5a82970f |

### Browsers, WebGPU: sampled hashes, penalty 1.1 (identical in Chromium, WebKit and Firefox Dev 159)

| Model | Before 128 | Before 256 | After 128 | After 256 |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 6be21500b4902e60 | 6ca9499e057eec9c | ad78429e765d69c2 | 36d8ee30a1b6df49 |
| SmolLM2-360M Q4_0 | b3a375abc0f60f5f | 12de46ad620f0fd2 | a7e7530ac128108a | 92428b3522ad279c |
| SmolLM2-360M Q4_K_M | d53ef6d61c21b015 | e9180ed2b534849d | 056989c0da688b41 | 820fed8a83c1e5b0 |
| Qwen3-0.6B Q8_0 | 48b4d208e2578ff2 | 37c1686d60bdcfc3 | 3267090046bb8edc | 0969901e7a6fbd1b |
| SmolLM2-1.7B Q4_0 | 9c953cc07f554df1 | c20fc6a649aa22ba | bc7c79f0983441a2 | 5a6686a6f5b39528 |

### Browsers, single CPU thread, after (Chromium, WebKit, Firefox Dev 159)

| Model | Exactness | Greedy 128 | Sampled 128 | Sampled 256 |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 29a934d304c7e868 | fdcf35d7e2636db1 | ad78429e765d69c2 | 36d8ee30a1b6df49 |
| SmolLM2-360M Q4_0 | d908b4954ae69f95 | 3fda34ebd775825a | a7e7530ac128108a | 92428b3522ad279c |

### Tests (after, f150fb5)

| Command | Result |
|---|---|
| `cargo test -p lean-core -p lean -p lean-chat --release` (CI), GPU adapter present | ok |
| `cargo clippy -p lean-core -p lean -p lean-chat --all-targets -- -D warnings` | ok |
| `cargo clippy ... --target wasm32-unknown-unknown --no-default-features --features lean/web,lean-chat/web -- -D warnings` | ok |
| `cargo clippy ... --target wasm32-unknown-unknown --no-default-features --features lean/web,lean/diag,lean-chat/web -- -D warnings` | ok |
| `fixture_parity` (Qwen2.5-0.5B Q4_0) | ok |
| `fixture_parity_llama_360m`, `fixture_parity_llama_360m_cpu` (with and without `--features threads`) | ok |
| `fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`, `fixture_parity_llama_1_7b`, `fixture_parity_qwen25_3b` | ok |
| `fixture_parity_gguf` (SmolLM2-360M Q4_K_M, Qwen3-0.6B Q4_K_M, Qwen2.5-0.5B Q4_K_M, Qwen3-0.6B F16 and BF16) | ok |
| `f16_parity` (Qwen2.5-0.5B, SmolLM2-360M, Qwen3-0.6B fixtures) | ok |
| lean `fixture_parity_generate` (qwen25_0_5b, smollm2_360m, smollm2_1_7b) | ok |
| lean `--lib --ignored` (Qwen2.5-0.5B and SmolLM2-360M): the six `streaming_sampling` tests, including `resident_sampling_matches_cpu_reference`, and `text_stream_qwen25` | ok (`text_stream_smollm2` not run: its variable was not set) |
| lean-core `--lib --ignored`: `quantized_kernels_match_cpu_reference`, `matvec_bandwidth` | ok (four tests not run: their model variables were not set) |

## Observations

- Before the change, the CPU and GPU paths agreed with each other (same max abs diff, 8.754 for the first SmolLM2 case) and both disagreed with transformers on exactly the history ids that occur more than once (14 to 20 per case); single-occurrence ids already matched.
- After it, the CPU sampler equals transformers bit for bit on all 8 cases. The GPU kernel is within one unit in the last place (max abs diff 1.9e-6) because it multiplies a positive logit by the host-computed `1 / penalty` where transformers divides; negative logits are multiplied on both sides and match exactly.
- llama.cpp also penalizes each distinct id once, but only over the last `penalty_last_n` ids (64 by default), prompt included; transformers and lean use the whole sequence.
- With penalty 1.0 the native sampled replies are unchanged (md5 equal before and after, five model and backend pairs); greedy hashes are unchanged for all five models in the three browsers. With penalty 1.1 every sampled hash changes, and the new hashes are the same on WebGPU and on the single CPU thread, in all three browsers and natively.
- The SmolLM2-1.7B reply at penalty 1.1 starts identically before and after ("Once upon a time in the quaint town of Ravenswood, there stood an ancient beacon known as the Lighthouse of Ravenswood. This magnificent structure was not just a") and diverges after about 30 words.
