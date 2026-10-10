# lean-core

lean-core is the forward pass of lean: it loads a quantized GGUF model and runs it on the GPU (`wgpu`, hand-written WGSL kernels, weights kept quantized and dequantized inside the kernels) or on the CPU (SIMD dot kernels: NEON natively on aarch64, SIMD128 in the browser, optionally on a rayon pool), from one source for native and wasm. It works on token ids and logits; text, tokenizers, sampling and chat templates are the [`lean`](../lean/README.md) and [`lean-chat`](../lean-chat/README.md) crates above it.

Use it directly when you need more than text in, text out: scoring a few token ids, packed batches with custom attention masks and positions, LoRA adapters.

## API

GPU (`engine`, `model`):

- `Engine::new()` (native) / `Engine::new_async().await`: the WebGPU/wgpu device and every compute pipeline. `engine.buf_f32(data, label)`, `engine.buf_u32(data, label)` upload a buffer.
- `GpuModel::load(&engine, path, fast_kernels)` / `GpuModel::load_from_reader(&engine, reader, fast_kernels)`: reads the GGUF one tensor at a time and uploads it. `model.config` is the `Qwen2Config`.
- `KvCache::new(&engine, &model.config, max_ctx)`: `kv_len` positions resident out of `max_ctx`; set `kv_len` to rewind. `cache.snapshot(&engine).await` / `cache.restore(&engine, &snapshot)` and `KvSnapshot::to_bytes` / `from_bytes` store and reload a prefix.
- `build_rope_tables(head_dim, theta, max_pos)`: the RoPE cos/sin tables every forward call takes (upload them with `buf_f32`).
- `forward_prefill(&engine, &model, &mut cache, ids, &cos, &sin, mask)` returns the last position's logits; `forward_prefill_suffix` continues from `kv_len`; `forward_decode_step` / `forward_decode_step_argmax` run one token; `decode_greedy_pipelined` runs a greedy loop with each step submitted before the previous id is read back. `mask` is an optional logit mask buffer (`build_mask_bitset(vocab, allowed_ids)`); `argmax(logits)` breaks ties to the lowest id.
- `ForwardSpec::default().with_positions(positions).with_allowed(&allowed, rows, cols)` (or `.with_allowed_bits(bits)`): explicit RoPE positions and an attention mask over `cols = kv_len + rows` keys. `forward_chunk_spec(&engine, &model, &mut cache, ids, &cos, &sin, &spec)` runs it and returns the final hidden states (`[rows, hidden]`, still on the GPU).
- `model.embed_head_sliced(&engine, &hidden, rows, &ids).await`: logits for `ids` only, through the embedding rows (`lm_head_sliced` through the output head).
- `model.apply_lora(&engine, bytes)` / `clear_lora()` / `has_lora()`: a LoRA adapter (q/k/v/o deltas) on top of the quantized weights.
- `model.pool.reset()`: call before reusing a model with a different `KvCache` (the pooled bind groups point at the previous one).

CPU (`cpu`): `CpuModel::load(path)` / `load_from_reader(reader)`, `CpuKvCache::new(&config, max_ctx)` (`kv_len`, `max_ctx()`, `remaining_capacity()`), `forward_prefill`, `forward_decode_step`, `forward_decode_step_argmax`, `forward_chunk_spec(&model, &mut cache, ids, &spec)` (hidden states as a `Vec<f32>`), `model.embed_head_sliced(&hidden, rows, &ids)`, `apply_lora` / `clear_lora` / `has_lora`. The same numbers as the GPU path up to float summation order; greedy tokens match.

Diagnostics used by lean's tools: `Engine::new_async_with(timestamps)`, `set_diag`, `diag_last_gpu`, `read_buffer`, `dispatch_count`, the `LEAN_PROFILE=1` per-op profile (`profile_report::report`), and the debug upload/tap checks.

Features: `wasm` (wgpu on WebGPU), `web` (also `performance.now()` timings in the browser), `threads` (CPU forward pass on a rayon pool), `force_scalar` (dev only).

## Supported models

Architectures, from `general.architecture` in the GGUF (`src/config.rs`):

- `qwen2`: Qwen2 and Qwen2.5 (tested: Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct).
- `qwen3`: Qwen3 (tested: Qwen3-0.6B, Qwen3-1.7B).
- `llama`: Llama-architecture models (tested: SmolLM2-360M-Instruct, SmolLM2-1.7B-Instruct, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct). Llama 3.1/3.2 RoPE scaling is read from the GGUF's `rope_freqs.weight` tensor.

The GPU attention kernels are compiled for a head dimension of 64 or 128; `GpuModel::load` rejects a model with another head dimension.

Tensor types (`src/gguf.rs`, `src/quant.rs`, `src/cpu.rs`):

| type | how it runs |
|---|---|
| Q4_0 | kept quantized, dequantized in the kernels (GPU and CPU) |
| Q8_0 | kept quantized, dequantized in the kernels (GPU and CPU) |
| Q6_K | kept quantized, dequantized in the kernels (GPU and CPU); usually the embedding and output tensors of a "Q4_0" GGUF |
| Q4_1 | dequantized to F32 at load (a few tensors of the SmolLM2 "Q4_0" GGUFs) |
| F16, F32 | F32 at load (norms, biases) |

Any other type is rejected at load with an error naming it.

## Parity gates

The parity tests compare lean against fixtures written by HF transformers (`reference/gen_fixture*.py`, which load the same GGUF through transformers). Each `fixture_parity*` test checks that the prompt rendered by lean-chat and tokenized gives the fixture's input ids, that the top-20 logits after prefill agree within 1e-3, and that the greedy continuation is identical. They need model files that are not in the repository, so they are `#[ignore]`d and run with `--ignored` and environment variables pointing at a GGUF and at a directory holding `tokenizer.json` and `tokenizer_config.json`:

| test | what it checks | environment |
|---|---|---|
| `fixture_parity` | Qwen2.5-0.5B Q4_0, both GPU kernel paths | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `fixture_parity_qwen25_3b` | Qwen2.5-3B Q4_0 (Q6_K output tensor) | `LEAN_GGUF_QWEN25_3B`, `LEAN_TOKENIZER_DIR_QWEN25_3B` |
| `fixture_parity_qwen3` | Qwen3-0.6B Q8_0 | `LEAN_GGUF_QWEN3`, `LEAN_TOKENIZER_DIR_QWEN3` |
| `fixture_parity_qwen3_1_7b` | Qwen3-1.7B Q8_0 | `LEAN_GGUF_QWEN3_1_7B`, `LEAN_TOKENIZER_DIR_QWEN3_1_7B` |
| `fixture_parity_llama_360m` | SmolLM2-360M Q4_0 and Q8_0, GPU | `LEAN_GGUF_LLAMA_360M_Q4_0`, `LEAN_GGUF_LLAMA_360M_Q8_0`, `LEAN_TOKENIZER_DIR_LLAMA_360M` |
| `fixture_parity_llama_360m_cpu` | the same on the CPU path (run with and without `--features threads`) | same as above |
| `fixture_parity_llama_1_7b` | SmolLM2-1.7B Q8_0 | `LEAN_GGUF_LLAMA_1_7B_Q8_0`, `LEAN_TOKENIZER_DIR_LLAMA_1_7B` |
| `fixture_parity_llama32` | Llama-3.2-1B and 3B, Q4_0 and Q8_0, GPU, including a 9044-token prompt (past the 8192 positions of the original context); the prompt is the fixture's HF-rendered chat text | `LEAN_GGUF_LLAMA32_1B_Q4_0`, `LEAN_GGUF_LLAMA32_1B_Q8_0`, `LEAN_TOKENIZER_DIR_LLAMA32_1B`, and the same with `3B` |
| `fixture_parity_llama32_cpu` | the same on the CPU path; the 9044-token case only with `LEAN_LLAMA32_CPU_LONG=1` | same as above |
| `logit_mask` | a mask forces an exact string; an all-allowed mask equals no mask | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `kv_snapshot` | restore plus suffix equals a full prefill; snapshot round trip | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR` |
| `lora_parity` | LoRA on the GPU against a transformers fixture with the LoRA deltas applied | `LEAN_GGUF`, `LEAN_TOKENIZER_DIR`, `LEAN_LORA_BIN` |
| `cpu_lora_parity` | LoRA on the CPU path against HF PEFT | `LEAN_GGUF`, `LEAN_LORA_BIN`, `LEAN_HF_PEFT_FIXTURE` |

```bash
LEAN_GGUF=<gguf> LEAN_TOKENIZER_DIR=<dir> cargo test -p lean-core --release --test fixture_parity -- --ignored
LEAN_GGUF_LLAMA_360M_Q4_0=<gguf> LEAN_GGUF_LLAMA_360M_Q8_0=<gguf> LEAN_TOKENIZER_DIR_LLAMA_360M=<dir> \
  cargo test -p lean-core --release --features threads --test fixture_parity_llama_360m_cpu -- --ignored
```

`cargo test -p lean-core` without `--ignored` runs the tests that need no model file (dequantization against gguf-py's reference blocks, LoRA parsing, kernel unit tests). `embed_head_sliced` and the Q6_K reference test are unit tests in `src/tests/`, since they check private modules. Release gate logs are under `docs/runs/`.

