# CPU prefill: output_norm and the LM head on the last row only, one token_embd table

Change (two commits on top of ee92a41):
- e95db1d: the CPU model keeps `token_embd` once. The embedding gather dequantizes rows of the same `CpuWeight` the tied LM head reads; before, the tied head was a clone of the table.
- 191c00f: `forward_prefill` runs `output_norm` and the LM head on the last row only, as the GPU path does. A prompt of two or more rows keeps `linear_multi_row` for that row (the kernel `linear` picks for several rows), so the logits stay the same bits. `forward_chunk_spec` still returns every row's normed hidden state.

## Parameters

- Before: ee92a41. After: 191c00f (and e95db1d alone, for the bit-identity table).
- Machines: the M2 laptop (macOS, aarch64, NEON kernels) and the Linux test machine (RTX 3080, Ryzen 9 3900X, x86-64: native CPU kernels are the scalar `F4` fallback there; wasm runs use simd128). Rust 1.99.0 (pinned toolchain), release builds, no `RUSTFLAGS`.
- Models (GGUF): SmolLM2-360M-Instruct Q4_0, Q8_0 and Q4_K_M; Qwen2.5-0.5B-Instruct Q4_0 (separate `output.weight`); Qwen3-0.6B Q8_0; Qwen3-1.7B Q8_0 (hidden 2048, 28 layers, 16 heads, 8 KV heads, head dim 128, FFN 6144, vocab 151,936, tied embeddings).
- Native bit-identity: `data/.../native-dumps/cpu_last_logits.rs` (an example program against `lean_core::cpu`, not committed to the crate) writes the prefill logits and four greedy decode steps' logits as raw f32 for synthetic prompts of 1, 2, 3, 5, 36 and 256 tokens. Built at ee92a41, e95db1d and 191c00f, default features and `threads`; run on the M2 laptop.
- Native parity: `lean-cli --engine cpu --long` on `fixture.json`, `fixture_qwen3.json` and `fixture_qwen3_1_7b.json`, and the ignored tests `fixture_parity_llama_360m_cpu`, `fixture_parity_gguf_cpu` (SmolLM2-360M Q4_K_M) and `cpu_lora_parity`, on the M2 laptop.
- wasm parity: lean's web+diag package (`LeanEngineCpu`) in Node 22.23.4, one thread, `data/.../node-fixture/node-fixture.mjs`: prefill top-1 and top-20 against the fixture, then the greedy continuation, each case's prefill logits written as raw f32. Packages: `lean_bg.wasm` SHA-256 `32535f24...` (ee92a41) and `91febdde...` (191c00f). Linux test machine.
- wasm memory: the agent-web `agent-wasm` package (lean's `Lean` on the CPU backend, one thread, in Node) built against each lean commit, running an agent page's sequence: load, a prefix image for an empty tools list (256-token system prompt), then for seven tools (904 positions), then one turn. Release packages: `agent_wasm_bg.wasm` `b6bda6e3...` (ee92a41) and `39fe55ac...` (191c00f); diagnostic build of 191c00f (counting allocator, no wasm-opt) `50beb294...`. `wasmMiB` is `memory.buffer.byteLength`; the wasm32 limit is 4,096 MiB.
- Timing: before and after interleaved (ABAB), five runs each natively, three each in wasm; median, min, max. Load is the 1-minute load average at the start of each run, including the benchmark's own threads.

## Results

### Native logits, before against after (M2 laptop; SHA-256 of all lengths' logits per build)

| Model | ee92a41 | ee92a41 threads | e95db1d | 191c00f | 191c00f threads |
|---|---|---|---|---|---|
| SmolLM2-360M Q4_0 | c19dddec2cbd4d2b | same | same | same | same |
| SmolLM2-360M Q4_K_M | e0e9b13de3be019e | same | same | same | same |
| Qwen2.5-0.5B Q4_0 | 46a0e3359e110dea | same | same | same | same |
| Qwen3-0.6B Q8_0 | 9542fa4a009aea26 | same | same | same | same |
| Qwen3-1.7B Q8_0 | 6620fd04fb9758ca | same | same | same | same |

Source: `native-dumps/sha256.txt`.

### Native parity against transformers, 191c00f (M2 laptop)

| Check | Model | Cases (prompt tokens) | Top-1 | Greedy tokens | Max top-20 diff |
|---|---|---|---|---|---|
| lean-cli, 1 thread | Qwen2.5-0.5B Q4_0 | 36, 86, 54, 2225, 2354 | all | all equal (2225: the fixture's 9 tokens, see below) | 8.9e-5 |
| lean-cli, threads | Qwen3-0.6B Q8_0 | 15, 65, 33, 2225 | all | all equal | 1.16e-4 |
| lean-cli, threads | Qwen3-1.7B Q8_0 | 15, 65, 33, 2225 | all | all equal | 9.9e-5 |
| fixture_parity_llama_360m_cpu, default and threads | SmolLM2-360M Q4_0, Q8_0 | 3 each | ok | ok | test passes |
| fixture_parity_gguf_cpu, default and threads | SmolLM2-360M Q4_K_M | 37, 86, 63 | all | 8, 32, 32 equal | 6.10e-5 |
| cpu_lora_parity, threads | Qwen2.5-0.5B Q4_0 + LoRA | 512 / 64 | 0 answers differ | | CPU 64/forward vs HF+PEFT 3.242e-4, CPU 1/forward 1.583e-4 |

Source: `parity-native-m2.log`, `cpu-lora-parity-m2.log`.

### wasm parity in Node, one thread (Linux test machine)

| Model | Case (tokens) | ee92a41 | 191c00f | Max top-20 diff (both) | Prefill logits |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0, max_ctx 2400 | 36, 86, 54 | pass, 627.2 MiB | pass, 535.6 MiB | 6.8e-5 | same bits |
| Qwen2.5-0.5B Q4_0 | 2225, 2354 | pass, 3,289.1 MiB | pass, 713.3 MiB | 8.8e-5 | same bits |
| SmolLM2-360M Q4_0 | 37, 86, 63 | pass | pass | 4.8e-5 | same bits |
| SmolLM2-360M Q8_0 | 37, 86, 63 | pass | pass | 4.3e-5 | same bits |
| Qwen3-1.7B Q8_0, max_ctx 2300 | 15, 65, 33 | pass, 2,752.9 MiB | pass, 2,362.3 MiB | 6.7e-5 | same bits |
| Qwen3-1.7B Q8_0 | 2225 | `RuntimeError: unreachable` at 2,961.8 MiB | pass, 2,658.4 MiB | 1.13e-4 (after) | |

Source: `node-fixture/fx-*.out`, `node-fixture/sha256.txt`. "pass" is top-1 and every greedy token equal to the fixture.

### wasm memory, agent package on Qwen3-1.7B Q8_0 (Linux test machine, Node, one thread)

| max_ctx | Step | Positions | 191c00f wasm MiB | 191c00f heap peak MiB | 191c00f largest request MiB | ee92a41 wasm MiB | Result (191c00f) |
|---:|---|---:|---:|---:|---:|---:|---|
| 8192 | load | 0 | 3,571.7 | 3,555.1 | 315.3 | 3,887.0 | loaded |
| 8192 | warm, no tools | 256 | 3,604.3 | 3,592.3 | 6.0 | trap | image saved |
| 8192 | warm, seven tools | 904 | 3,670.9 | 3,651.5 | 15.3 | | image saved |
| 8192 | turn | | 3,670.9 | 3,556.3 | 0.6 | | `needTools`, 915 prompt, 177 think, 20 answer tokens |
| 4096 | load | 0 | 2,675.7 | 2,659.1 | 315.3 | 2,991.0 | loaded |
| 4096 | warm, seven tools | 904 | 2,774.9 | 2,755.5 | 15.3 | 4,078.6 | image saved |
| 4096 | turn | | 2,774.9 | 2,660.3 | 0.6 | 4,078.6 | same outcome and token counts |
| 2048 | load | 0 | 2,227.7 | 2,211.1 | 315.3 | 2,543.0 | loaded |
| 2048 | warm, no tools | 256 | 2,260.3 | 2,248.3 | 6.0 | 2,872.5 | image saved |
| 2048 | warm, seven tools | 904 | 2,326.9 | 2,307.5 | 15.3 | 3,630.6 | image saved |
| 2048 | turn | | 2,326.9 | 2,212.3 | 0.6 | | same outcome and token counts |

Source: `node-agent/f2d8192.out`, `f2d4096.out`, `f2d2048.out` (diagnostic build), `f2r8192.out` (release build, same wasm MiB at every step); ee92a41 at 2048 from `timing-linux.log`. The ee92a41 values at 8192 and 4096 come from the same harness run on the same machine on 2026-10-30 (not committed here): at 8192 the first prefill failed an allocation of 148.4 MiB, one `[256, 151,936]` f32 logits buffer.

### CPU prefill time, 256-token synthetic prompt, ms (median, min, max)

| Machine | Build | Model | Threads | ee92a41 | 191c00f | Load range |
|---|---|---|---|---|---|---|
| Linux test machine, native | release | Qwen2.5-0.5B Q4_0 | 1 | 15,249.8 (14,934.2, 15,328.7) | 11,367.4 (11,197.5, 11,545.0) | 1.07 to 6.79 |
| Linux test machine, native | release, `threads` | Qwen3-1.7B Q8_0 | 24 | 7,653.3 (7,627.8, 7,683.1) | 6,873.3 (6,832.6, 6,910.7) | 1.12 to 5.57 |
| M2 laptop, native | release | Qwen2.5-0.5B Q4_0 | 1 | 5,979.7 (5,663.7, 8,054.8) | 5,137.8 (4,284.5, 5,943.9) | 2.49 to 4.40 |
| M2 laptop, native | release | Qwen3-1.7B Q8_0 | 1 | 22,546.8 (19,709.5, 28,159.9) | 18,607.2 (16,630.3, 23,927.4) | 2.58 to 4.34 |

Source: `timing-linux.log`, `timing-m2.log` (`len 256 prefill_ms`).

### wasm prefill time in the agent package, Qwen3-1.7B Q8_0, max_ctx 2048, ms (Linux test machine, Node, one thread, three runs each)

| Step | Rows prefilled | ee92a41 median (runs) | 191c00f median (runs) |
|---|---:|---|---|
| warm, no tools | 256 | 35,966 (35,876, 35,966, 36,484) | 30,629 (30,529, 30,629, 31,237) |
| warm, seven tools | 648 | 100,540 (100,258, 100,540, 100,862) | 86,327 (85,703, 86,327, 86,550) |

Source: `timing-linux.log` (step `ms` differences). Load at the start of the six runs: 5.81, 1.63, 1.24, 1.20, 1.10, 1.38.

### CI commands on 191c00f (M2 laptop)

`cargo test -p lean-core -p lean -p lean-chat --release`, `cargo test -p lean-core --release --features threads`, clippy `-D warnings` native all targets, native all targets with `lean/threads`, wasm32 web, wasm32 web+diag: all exit 0 (`ci-m2.log`).

## Observations

- The last-row logits and the four decode steps after them have the same SHA-256 before and after on all five models, at 1 to 256 prompt tokens, with and without `threads`; in wasm the prefill logits of every case both builds ran are the same bits.
- The wasm memory after load drops by 315.3 MiB at every max_ctx (3,887.0 to 3,571.7 at 8192), the size of the second `token_embd` copy.
- The largest request during a prefill drops from 148.4 MiB (256 rows) and 379.1 MiB (648 rows) to 6.0 and 15.3 MiB, and wasm memory grows by 99.2 MiB over load and both prefills at 8192, against 1,087.6 MiB at 2048 before.
- At max_ctx 8192 the agent sequence completes after the change and returns the same tool call and token counts as the earlier builds at 2048 and 4096; before, it failed in the first prefill.
- Qwen3-1.7B's 2,225-token prefill traps in wasm before the change (at 2,961.8 MiB, about 2.6 GB of logits needed) and passes after it, token-exact.
- The 256-token prefill is 25.5% faster natively on one thread for Qwen2.5-0.5B on the Linux machine, 10.2% with 24 threads on Qwen3-1.7B, and 14.8% (256 rows) and 14.1% (648 rows) faster in wasm on Qwen3-1.7B. The M2 numbers ran at load averages above 3 for most runs and spread by up to 43% within a build.
- `lean-cli` counts `long_tools_single` on Qwen2.5-0.5B as a token mismatch: it generates 32 tokens while the fixture stops at 9 (an end-of-turn token); the first 9 are equal. The comparison takes `got_tokens` whole against the fixture's shorter list, the same at ee92a41.
</content>
</invoke>
