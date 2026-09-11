# ENGINE.md — what stt-wasm transfers to a Qwen2 decoder

Audit of `/Users/tc/Code/idle-intelligence/stt-web/crates/stt-wasm` (read-only, not modified) as
a starting point for a Qwen2.5-3B-style decoder (xLAM-2-3b-fc-r: GQA + q/k/v bias, RoPE, RMSNorm,
SwiGLU, tied embeddings, 151k vocab) driving MCP tool calls in-browser.

Versions in use (`stt-web/Cargo.toml:22-58`): Burn 0.20, cubecl 0.9 (wgpu backend), wgpu 26,
tokenizers 0.22 (declared, unused in WASM — see §6).

## 1. GGUF loader (`src/gguf.rs`)

- Quant types supported: **F32, F16, Q4_0 only** (`GgmlDtype::from_u32`, gguf.rs:139-143). No
  Q4_K, Q8_0, or any k-quant. `byte_size()` (gguf.rs:146-157) hard-codes Q4_0 block math
  (18 bytes / 32-element block: 2-byte f16 scale + 16 bytes nibble data).
- GGUF v2/v3 header only (gguf.rs:196-198); metadata KV pairs are read and **discarded**
  (gguf.rs:213-220, `skip_gguf_value`) — no metadata (rope_theta, head_count, etc.) is read from
  the file. All hyperparameters instead come from the hardcoded `SttConfig::default()`
  (`src/lib.rs:60-80`), verified by hand against `kyutai/stt-1b-en_fr`'s `config.json`. A Qwen2
  port must either add real GGUF metadata parsing or keep hand-copying config fields (current
  pattern).
- Tensor → GPU path: `Q4Tensor::from_q4_bytes` (gguf.rs:414-450) uploads the raw Q4_0 byte blob
  directly to a `wgpu` storage buffer via `client.create_from_slice`, padded to 4-byte alignment
  for `array<u32>` access. No CPU-side dequant for linear weights — dequant happens in the WGSL
  kernel (§2). Norm weights (F32/F16) are fully decoded on CPU and uploaded as a `Tensor<Wgpu,1>`
  (gguf.rs:890-919). The token embedding table is the one exception: `Q4ModelParts::finalize`
  dequantizes the whole `text_emb` table to F32 on CPU and uploads it (gguf.rs:834-847,
  `dequant_embedding_to_gpu` at 843-877) specifically to allow GPU-resident `Tensor::select()`
  lookups and avoid a 236 ms WebGPU readback per frame (comment at gguf.rs:838-840). At Qwen's
  151936×2048 vocab, that dequantized table is 151936×2048×4B ≈ **1.16 GB** in F32 — a real memory
  cost on a 16 GB M2 that stt-wasm's 8001×2048 (~62 MB) table never had to reckon with.
- Two-phase loading (`load_deferred` gguf.rs:763-825, then `Q4ModelParts::finalize` gguf.rs:829
  -877): parse GGUF and stash weights as `Q4TransformerBlock`s / raw bytes, drop the reader (frees
  the parsed file), then build GPU tensors. Comment at gguf.rs:5-9 explains this exists to stay
  under WASM's ~4 GB address space with a large model.
- Tensor-name assumptions are hardcoded strings tied to the kyutai checkpoint's naming, not
  read from GGUF metadata: `emb.{i}.weight` (audio codebooks, gguf.rs:772), `text_emb.weight`
  (gguf.rs:783), `transformer.layers.{i}.norm1.alpha` / `.self_attn.in_proj_weight` /
  `.self_attn.out_proj.weight` / `.norm2.alpha` / `.gating.linear_in.weight` /
  `.gating.linear_out.weight` (gguf.rs:829-870), `out_norm.alpha` (gguf.rs:816),
  `text_linear.weight` (gguf.rs:819). None of these match `llama.cpp`/GGUF-convention Qwen2 names
  (`blk.N.attn_q.weight`, `output_norm.weight`, etc.) — a Qwen2 loader is a rewrite of this
  naming table, not a parameterization of it. Note `docs/TENSOR_NAMING.md` in stt-web describes a
  *different*, un-prefixed naming scheme (`layers.{i}.attention.wq.weight`, plus depformer tensors)
  that doesn't match what `gguf.rs` actually loads — the doc is stale relative to the loader.
- **No linear bias support is wired up.** `Q4Linear` has an `Option<Tensor<Wgpu,1>>` bias field
  (gguf.rs:472) and `forward()` adds it if present (gguf.rs:480-489), but `load_q4_linear`
  (gguf.rs:873-887) always constructs `Q4Linear::new(q4, None)` — no bias tensor is ever read from
  GGUF. Qwen2's q/k/v projections have bias; wiring this up is mechanical (read a `*.bias` F32/F16
  tensor per projection, same pattern as `load_rms_norm`) but is unimplemented today.

## 2. Q4 WGSL kernel (`src/wgsl/shader_naive.wgsl`)

- Exactly **one** shader ships in this crate: the "naive" variant (98 lines). The module doc at
  gguf.rs:9 references a "tiled kernel" as native-only, but no such file exists under
  `crates/stt-wasm/src/wgsl/` — only `shader_naive.wgsl`. A tiled/cooperative variant does exist
  in the vendored reference at `stt-web/refs/voxtral-mini-realtime-rs/src/gguf/shader.wgsl` but is
  not used by stt-wasm.
- Dispatch: one thread per output element. `main()` (shader_naive.wgsl:29) reads `B, M, K, N,
  blocks_per_row` from an `info` uniform-ish storage buffer (gguf.rs:541-546) and computes
  `output[B,M,N] = input[B,M,K] × weights[N,K]^T`. Workgroup size is `16×16×1` = 256 invocations
  (`NAIVE_WG_X`/`NAIVE_WG_Y`, gguf.rs:36-37), chosen to sit at WebGPU's default
  `maxComputeInvocationsPerWorkgroup` limit (comment at shader_naive.wgsl:5). Dispatch grid:
  `wg_x = ceil(N/16)`, `wg_y = ceil(B*M/16)` (gguf.rs:559-560).
- **This kernel is already general over M** (it indexes `bm = gid.y`, `m = bm % M`, `b = bm / M`),
  so it does both matvec (M=1, decode) and matmul (M>1, prefill) — there is no dedicated prefill
  path missing at the shader level. The gap is entirely upstream in `model.rs`'s KV cache (§3),
  which never actually calls this kernel with M>1.
- Block format: Q4_0, 18 bytes/block (2-byte f16 scale + 16 bytes of paired 4-bit values,
  low/high nibble = elements `[i]`/`[i+16]` within the 32-wide block), read via
  `read_u32_unaligned` (shader_naive.wgsl:15-22) since blocks aren't 4-byte aligned relative to
  the buffer. Dequant is inlined per-block, `vec4` dot products against 4 input elements at a time.
- Size limits relevant to Qwen2.5-3B:
  - **Dispatch grid**: WebGPU's per-dimension dispatch limit is 65535 workgroups. For the
    151936-wide lm_head, `wg_x = ceil(151936/16) ≈ 9497` — fine. Even a prefill of, say, 4096
    tokens in one batch gives `wg_y = ceil(4096/16) = 256` — fine. Not a real risk at these sizes.
  - **Storage buffer binding size**: `init_wgpu_device` (bindings.rs:105-160) requests the
    adapter's *full* `adapter.limits()` as `required_limits` (bindings.rs:135), rather than the
    WebGPU default (`maxStorageBufferBindingSize` = 128 MiB, `maxBufferSize` = 256 MiB spec
    minimums). This is the load-bearing detail for Qwen2.5-3B: the lm_head weight alone
    (151936×2048 at Q4_0 ≈ 174 MB) and the down_proj/gate_proj/up_proj at 11008-wide MLP
    (2048×11008 Q4_0 ≈ 23 MB each) would each blow the 128 MiB spec-minimum binding limit if the
    code requested defaults instead. Chrome/Metal on M2 typically reports `maxBufferSize` /
    `maxStorageBufferBindingSize` well above 1 GB, so this should work, but it is untested here —
    a Q4 lm_head tensor (151936×2048/32×18 bytes ≈ **174 MB in one buffer**) is the single largest
    per-tensor allocation this engine would ever need to make, more than 10× any single tensor in
    the 1B STT model (largest there: `in_proj_weight` 3×2048×2048 Q4_0 ≈ 7 MB, or `text_linear`
    8000×2048 Q4_0 ≈ 9.2 MB). This should be checked against actual Chrome/Metal
    `maxStorageBufferBindingSize` before committing to a single-buffer lm_head.
  - The naive kernel has no shared-memory tiling, so it is bandwidth-bound: every output element
    re-reads its full K-length row of weights from global memory with no reuse across the M
    (batch/token) dimension. At M=1 (decode) this is close to optimal (weight-bound). At M>>1
    (prefill) it wastes compute vs. a tiled kernel that reuses weight blocks across multiple query
    rows — this matters more for a 3B decoder's 11008-wide MLP than it did for STT's 8448-wide one.

## 3. Attention / KV cache (`src/model.rs`, `src/stream.rs`)

- GQA is implemented and general (`Q4Attention::expand_kv`, model.rs:476-499): repeats K/V heads
  by `n_heads/n_kv_heads` before the QK^T matmul. The shipped STT checkpoint happens to use
  `num_heads == num_kv_heads == 16` (MHA, `SttConfig::default()`, lib.rs:66-67), so this path is
  present but not exercised by current tests/weights — worth validating with an actual GQA
  checkpoint before trusting it for Qwen2.5-3B (7 KV heads / 16 Q heads in the reference config).
- RoPE (`RoPE::apply_rotation`, model.rs:86-121) uses the **interleaved / GPT-NeoX-pair** variant:
  it reshapes `head_dim` into `[half_dim, 2]` adjacent pairs `(x0,x1), (x2,x3), ...` and rotates
  each pair, then re-interleaves. **This is not the HF Llama/Qwen2 "rotate_half" convention**,
  which splits the head into two contiguous halves `[x0..x_{d/2}]` / `[x_{d/2}..x_d]` and rotates
  across the split. The two are mathematically different unless weights are also permuted to
  match. This is a real, silent-failure-risk gap: porting Qwen2 weights straight into this RoPE
  implementation will produce wrong attention without an explicit note or a permutation of the
  Q/K projection weight rows (interleave ↔ half-split), or a rewrite of `apply_rotation` to the
  half-split convention.
- **KV cache only supports single-token decode, not multi-token prefill.** `KVCache::update`
  (model.rs:176-218) always writes at `[pos..pos+1]` via `slice_assign` — one timestep per call
  regardless of the caller's sequence length (model.rs:200-201). `SttModel::forward` /
  `forward_with_gpu_token` (model.rs:631-661, 675-716) only ever build a `[1,1,dim]` input tensor
  from a single audio+text token pair — there is no code path in this crate that runs the model
  over N>1 tokens at once, despite `apply_causal_mask_with_offset` (model.rs:302-326) being
  written generally for `q_len != kv_len`. Confirmed by `full_forward.rs` and `streaming.rs` tests,
  which only ever call `model.forward()` one frame at a time. **Prefill (loading an MCP
  system/tool-definition prompt of, say, 500-2000 tokens) needs new code**: a batched embedding
  path, a `KVCache::update` that writes M>1 rows in one `slice_assign`, and exercising the Q4
  matmul kernel at M>1 (which, per §2, the kernel itself already supports).
- KV cache layout: pre-allocated `[batch, n_kv_heads, max_len, head_dim]` ring buffer per layer
  (model.rs:139-176), `max_len = sliding_window + 1 = 751` (model.rs:728-730,
  `sliding_window: 750` in `SttConfig::default()`, lib.rs:76). Growth is **not** dynamic — it's a
  fixed-size ring buffer sized at model-creation time from `sliding_window`, with wraparound
  (model.rs:203, `write_pos = (write_pos+1) % max_len`) and a separate monotonic `offset` counter
  used for RoPE position (model.rs:224-227, comment at 136-138 explaining offset vs. `seq_len`
  after wraparound). Max context here is **750 timesteps at the model's own 12.5 Hz-derived
  internal rate**, nowhere near what an MCP agent needs (thousands of tokens of tool schemas +
  conversation). A Qwen2 port needs either a much larger fixed cache (memory cost: `n_layers ×
  2 × n_kv_heads × head_dim × 4 bytes × context_len`; for Qwen2.5-3B-class dims — e.g. 36 layers,
  2 KV heads (GQA), head_dim 128 — at 8192 context that's `36×2×2×128×4×8192 ≈ 604 MB`, plausible
  on 16 GB) or a growable/paged design; ring-buffer wraparound as written would silently corrupt
  a long agent conversation once it exceeds the buffer.
- Sliding-window masking (`apply_sliding_window_mask_with_offset`, model.rs:328-357) is applied
  unconditionally when `sliding_window: Some(_)` is passed to `Q4Attention::new` (always the case
  today, from `config.sliding_window`, gguf.rs:857-863). Qwen2.5-3B doesn't use sliding-window
  attention by default, so this should become `None`/full causal for a Qwen2 port — mechanical.
- Causal masking (`apply_causal_mask_with_offset`, model.rs:302-326) is written to support
  `q_len != kv_len` already (i.e., it's already prefill-shaped), building a full `[q_len, kv_len]`
  additive mask with `-inf` above the diagonal, offset by the cache's absolute position. This part
  needs no change for prefill — the mask math is already correct for M>1; only the KV-cache write
  path (previous bullet) blocks it.

## 4. Burn model structure

- Mix of Burn built-ins and custom cubecl/WGSL: `burn::nn::RmsNorm` is used as-is, wrapped in
  `RmsNormLayer` only to carry GGUF-loaded gamma (model.rs:365-381); `burn::tensor::activation::
  silu` and `softmax` are Burn built-ins (model.rs:15, used at model.rs:521/468); RoPE, the KV
  cache, causal/sliding masks, and GQA head-expansion are **all hand-written** Burn tensor-op code
  (matmul/reshape/slice/cat), not cubecl kernels — only the Q4 dequant+matmul itself
  (`q4_matmul`, gguf.rs:514-575) is a custom cubecl/WGSL `SourceKernel`. So attention math runs as
  regular (F32) Burn ops on top of Q4-dequantized activations; only weight storage/dequant is
  quantized.
- Activation is **SiLU**, not GELU (`Q4FeedForward::forward`, model.rs:517-524:
  `linear_out(silu(gate) * value)` — a standard SwiGLU 2-matrix-gate form, matching what
  Qwen2/Llama-family MLPs need). No GELU appears anywhere in this crate.
- Norm is **RMSNorm only** (`RmsNormLayer`, model.rs:365-382, `eps: 1e-8` — gguf.rs:816/832/891
  all call `load_rms_norm(..., 1e-8, ...)`); no LayerNorm path exists. Qwen2.5 also uses RMSNorm,
  so this transfers directly (only the eps needs matching Qwen2's config, typically 1e-6).
- q/k/v bias: struct-level support exists (`Q4Linear.bias: Option<Tensor<Wgpu,1>>`, gguf.rs:472)
  but is never populated by the loader (§1) — needs a `load_q4_linear_with_bias` variant that also
  reads a `*.bias` F32/F16 tensor.
- Tied embeddings / lm_head: **not implemented as tied.** This model has two independent Q4
  tensors — `text_emb.weight` (embedding, gguf.rs:783-790) and `text_linear.weight` (output head,
  gguf.rs:819, loaded via generic `load_q4_linear`) — loaded from separate GGUF tensor names, with
  no code path that reuses one buffer for both. `xLAM-2-3b-fc-r`/Qwen2.5-3B ties embeddings and
  lm_head (single `model.embed_tokens.weight` used both ways), which is actually a small
  *simplification* opportunity (share one GPU buffer, skip loading a second 174 MB Q4 tensor) but
  requires new plumbing since nothing here currently aliases embedding and head weights, and the
  embedding table needs to support both the CPU-dequant-per-row path (`EmbeddingStore`,
  gguf.rs:750-798, used for input lookups) and the Q4Tensor-as-matmul-weight path (`Q4Tensor`,
  used by `Q4Linear`) simultaneously.

## 5. WASM bindings and worker protocol (`src/web/bindings.rs`, `web/worker.js`)

- `wasm-bindgen` surface on `SttEngine` (bindings.rs:171-676): `new()`; `appendModelShard(shard:
  &[u8])` (223-234) / `loadModel()` (236-269) — multi-shard GGUF load, two-phase per §1;
  `loadMimi(data)` (273-282); `loadTokenizer(data)` (284-292) — loads the SentencePiece `.model`
  bytes into the Rust-side decoder (§6); `loadVad(data)` / VAD threshold setters (293-349);
  `feedAudio(samples: &[f32]) -> String` (350+, main streaming entry point, returns decoded text
  synchronously per call); `getMetrics()` (520+, JSON metrics blob); `reset()` (601+); `isReady()`
  (675+). Plus module-level `initWgpuDevice()` (async, 105-163) that must be awaited before
  constructing `SttEngine`.
- Weight loading is shard-based: `web/worker.js`'s `handleLoad` (worker.js:196-266) fetches one
  or more GGUF shard URLs (`config.shardList` or a single `modelUrl`, worker.js:220-227) through
  `cachedFetch` (Cache-API-backed, worker.js:126-186), calling `engine.appendModelShard(...)` per
  shard, then a single `engine.loadModel()` once all shards are appended. Model, Mimi codec,
  tokenizer, and (optionally) VAD weights are fetched as separate HTTP resources by the worker and
  handed to WASM as raw `Uint8Array`s — WASM itself never fetches anything (no `fetch` calls in
  `bindings.rs`; all I/O is JS-side `cachedFetch`, then passed in as bytes).
- Worker message protocol (documented at worker.js:5-18 and implemented via `self.onmessage` /
  `msgQueue` / `drainQueue`, worker.js:39-118): main→worker `{type:'load'}`, `{type:'audio',
  samples}`, `{type:'stop'}`, `{type:'reset'}`, plus `{type:'audio-port', port}` for a direct
  `MessagePort` from an `AudioWorklet` that bypasses the main thread for audio chunks
  (worker.js:99-113); worker→main `{type:'status', ...}`, `{type:'transcript', ...}`,
  `{type:'error', ...}`. All engine calls are serialized behind a `busy` flag + FIFO queue
  (worker.js:42-88) because the WASM engine takes `&mut self` and wasm-bindgen aborts on
  reentrant calls — the same pattern (one `&mut` engine, message-queued from a single worker)
  would need to carry over, plus new message types for turn-based text (prompt in, streamed
  tokens out) rather than continuous audio frames.

## 6. Where tokenization happens

**Decoding happens in Rust/WASM (`src/tokenizer.rs`), not JavaScript, and there is no encoding
path on either side.** Concretely:

- `crates/stt-wasm/src/tokenizer.rs` implements a hand-rolled SentencePiece **`.model` protobuf
  parser** (`parse_sentencepiece_vocab`, tokenizer.rs:43-69) that extracts only the piece strings
  into `Vec<String>` indexed by token ID, and a **decode-only** `SpmDecoder` (tokenizer.rs:8-35:
  `from_bytes`, `decode`, `vocab_len`) — turning token IDs back into text (▁ → space, byte-fallback
  tokens reassembled via `TextDecoder`). This is what `SttEngine::loadTokenizer`/`feedAudio`
  actually call at runtime (bindings.rs:285-292, and decode calls inside `feed_audio`).
- `web/tokenizer.js` is a **near-identical, unused duplicate**: another decode-only SentencePiece
  `.model` parser (`SpmDecoder` class, tokenizer.js:14-...), but nothing in `web/worker.js` or
  `web/stt-client.js` imports it (`grep -rn "tokenizer.js"` under `web/` matches only the file's
  own class definition) — it appears to be dead/legacy code from before decoding moved into WASM.
- The `tokenizers` crate (HuggingFace Rust tokenizers, workspace pin `0.22`, root
  `Cargo.toml:45`) is declared as an **optional, native-only** dependency in `stt-wasm/Cargo.toml`
  (`tokenizers = { workspace = true, optional = true }`, gated behind the `native-tokenizer`
  feature which is in `default = [...]` but **not** in the `wasm` feature set —
  `stt-wasm/Cargo.toml` features block) with the comment "has C dependencies" explaining the
  exclusion. `grep -rn "tokenizers::" crates/stt-wasm/src/` finds **zero** call sites — the crate
  is declared but never actually used anywhere, native or WASM; encoding isn't implemented at all
  in this codebase (unsurprising: STT only needs to *decode* the small 8001-token output
  vocabulary, never encode arbitrary text).
- Implication for Qwen2: none of this transfers. A 151k-vocab byte-level BPE tokenizer with a
  ~7 MB `tokenizer.json` (merges + vocab) needs a **real encode+decode** implementation, and the
  current architecture's precedent — hand-roll the format parser client-side, keep it out of the
  WASM binary — points toward doing this **in JavaScript** (e.g. `@huggingface/transformers`'s
  bundled tokenizer, or a from-scratch BPE) rather than compiling the `tokenizers` crate into WASM,
  matching how this repo already treats "native has real deps, WASM gets a hand-rolled JS/Rust-but-
  decode-only parser" as its working pattern. Whichever side does it, the worker protocol (§5)
  needs new message shapes: today `feedAudio` takes raw audio floats and returns decoded text
  per-call; a Qwen2 agent instead needs a prompt-in / token-stream-out protocol, and BPE encoding
  of a multi-KB MCP tool-definition prompt is itself nontrivial work sitting on the critical path
  before the first forward pass.

## 7. Tests and reference rig

- `crates/stt-wasm/tests/q4_matmul.rs` (491 lines): CPU-reference Q4_0 dequant
  (`dequantize_q4_0_cpu`, q4_matmul.rs:12-33) compared against the GPU `q4_matmul` kernel output.
  Four tests: `test_q4_matmul_identity_like` (q4_matmul.rs:101, small hand-built case),
  `test_q4_matmul_different_inputs` (158, `max_err < 1.0` tolerance at line 228),
  `test_q4_matmul_larger_realistic` (232, `max_err < 2.0` at line 306, realistic-sized random
  matrices), `test_q4_matmul_with_actual_gguf` (396, loads the real GGUF and checks a real weight
  tensor's dequant+matmul against CPU reference). **Directly reusable as-is** for Qwen2 — the
  kernel and dequant math don't change; only shapes/values do. Note the tolerances are absolute
  max-error, not relative — reasonable for values in a similar magnitude range as STT's, would
  need re-checking against Qwen2's actual activation scale.
- `crates/stt-wasm/tests/full_forward.rs` (157 lines): loads the real GGUF, runs
  `model.forward()` across a few hand-picked audio-token frames, and just checks that **logits
  vary and produce different top-5/argmax** across frames (`test_forward_produces_varying_logits`,
  full_forward.rs:16; `test_forward_no_cache_varying`, 91) — a smoke test, not a numeric match
  against any reference. Reusable pattern (swap in the Qwen2 model) but not a strong correctness
  check by itself.
- `crates/stt-wasm/tests/e2e_pytorch_mimi.rs` (118 lines): the actual PyTorch-reference-comparison
  test. Reads **pre-generated** reference Mimi tokens from `tests/reference/mimi_tokens.json`
  (produced by a Python/PyTorch run offline — no `test_reference.py` currently exists in
  `stt-web/scripts/` despite being referenced in `README.md`; the JSON artifacts already exist
  under `stt-web/tests/reference/` — `config.json`, `mimi_tokens.json`, `text_tokens.json`,
  `transcript.txt`), feeds them through the Rust STT model, and asserts the decoded transcript is
  **non-empty** (`assert!(!transcript.is_empty())`, e2e_pytorch_mimi.rs:116) — not a token-exact or
  logit-tolerance comparison against PyTorch. `crates/stt-wasm/tests/e2e_transcript.rs` (178 lines)
  goes further: it diffs the Rust-predicted token stream against `text_tokens.json`/
  `transcript.txt` word-by-word (e2e_transcript.rs:129+) but tracks match/mismatch counts as
  diagnostic output rather than a hard pass/fail threshold. **There is no logit-level
  numeric-tolerance test against a PyTorch reference anywhere in this crate** — the closest is
  Q4-vs-CPU-dequant in `q4_matmul.rs`. A Qwen2 port would need to build a genuine PyTorch-logits
  comparison harness from scratch (a real gap, not a reuse item), most likely following the
  `e2e_pytorch_mimi.rs` pattern of "dump reference intermediate values as JSON offline, load and
  compare in a Rust test."
- `crates/stt-wasm/tests/e2e_wav.rs` (704 lines) and `tests/streaming.rs` (128 lines) exercise the
  full mic-to-text pipeline (Mimi + STT + tokenizer) on real WAV files and are also where the
  per-frame timing numbers in `stt-web/BENCHMARKS.md` come from (see §8) — not reusable for Qwen2
  (audio-specific), but their timing-harness pattern (`Instant::now()` around the decode loop,
  RTF computation) is.

## 8. Baseline timing (measured natively on this M2)

Ran the existing e2e timing test, which is the same one BENCHMARKS.md's numbers came from:

```
cargo test --release --features wgpu --test e2e_wav test_e2e_wav_bria -- --nocapture
```

(note: `test_e2e_wav_bria` is not `#[ignore]`d, so no `--ignored` flag — that flag silently
filters the test out entirely, `0 passed; ... 5 filtered out`.) Measured on this M2 against
`models/stt-1b-en_fr-q4.gguf` (real Q4_0 GGUF weights) and `web/test-bria.wav` (44.85 s audio):

```
Audio: 44.85s, 1076454 samples
Mimi: 560 frames from 44.85s audio, Mimi RTF: 0.123x (5.517s / 44.85s)
STT RTF:   0.680x (30.518s processing / 44.85s audio)
Total RTF: 0.803x (Mimi: 0.123x, STT: 0.680x)
test test_e2e_wav_bria ... ok, finished in 37.01s
```

560 STT frames in 30.518 s → **54.5 ms/decode-step** average (one frame = one autoregressive
step: single audio+text token in, one text-token logits out), i.e. **~18.3 tok/s** for
single-token (M=1) decode on the 1B/16-layer Q4 model at hidden_size=2048. This matches
`stt-web/BENCHMARKS.md`'s reported steady-state ~50-55 ms/frame for long clips almost exactly —
confirming the checked-in benchmark numbers are reproducible on this machine, not stale. No
independent browser number exists to measure: `stt-web` has no
Playwright or other headless-browser harness anywhere in the repo (`find . -iname "*playwright*"`,
`grep -rn playwright` under `stt-web/` both return nothing) — per the task constraint against
opening a real browser, the only browser figures available are **quoted, not measured**: the
`OVERVIEW.md`/README's real-time budget target (**80 ms/frame**, i.e. the model must be ≥12.5 Hz
to keep up with streaming audio) and `BENCHMARKS.md`'s claim that browser TTFB is reduced "by
~150-400 ms" via GPU warmup (BENCHMARKS.md, "Changes" section) — both qualitative/targets, not a
measured browser tok/s number.

Qwen2.5-3B is roughly 3× the parameter count of this 1B model, with wider MLP (11008 vs 8448) and
deeper attention (36 vs 16 layers in the reference config) — so this baseline should be read as
"per-decode-step cost of the current architecture at 1B scale," not a Qwen2.5-3B estimate; the Q4
matmul kernel's cost scales roughly linearly with total weight bytes touched per step at M=1
(bandwidth-bound), so a rough extrapolation is ~3× this engine's per-token time before accounting
for the wider MLP/attention shapes, i.e. plausibly bandwidth-bound *slower* than 20 tok/s once
ported, before any prefill/tiling optimization work.

## 9. Facts settled after the audit

- **The GGUF actually in use** is `~/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/
  xLAM-2-3b-fc-r-q4_0.gguf` (lowercase filename; a **Mungert** requant, not the Salesforce-
  official file `MODELS.md` §2 audited). Confirmed on disk: 1,742,628,672 bytes ≈ **1.74 GB**,
  **pure F32 + Q4_0** throughout including `token_embd.weight` (the Salesforce Q4_0 file instead
  keeps `token_embd.weight` at Q6_K and is 1.82 GB — a different, unused file). No `output.weight`
  tensor, consistent with `tie_word_embeddings: true`: LM head reuses `token_embd.weight`
  transposed. Since Mungert is what's on disk, the loader only needs Q4_0 support, not Q6_K —
  simpler than `MODELS.md` §2's checklist implied.
- **Tool-call output is a bare JSON array**, no `<tool_call>` wrapper tags (`MODELS.md` §3) — the
  engine's tool-call detector is "does decoded, stripped output start with `[`", then
  `JSON.parse` the whole thing as `[{name, arguments}, ...]`.
- **Chat template needs only minijinja's `json` feature** (for `tojson`, incl. `indent` kwarg) on
  top of default `builtins` — no `loop_controls`, `custom_syntax`, `adjacent_loop_items`,
  `macros`, or `multi_template` (`MODELS.md` §3, "Jinja features used").

## 10. Bandwidth bound

Q4 decode (M=1) is memory-bandwidth-bound, not compute-bound: every output element reads its full
weight row once and does O(1) arithmetic per byte. The measured baseline (§8) gives an effective
achieved bandwidth for the *existing naive kernel*:

- stt-wasm's 1B model: ~0.6 GB of Q4 weights touched per decode step, measured **54.5 ms/step**
  (`stt-web/BENCHMARKS.md`, 120s clip steady state) → **0.6 GB / 0.0545 s ≈ 11 GB/s effective**,
  against an M2's unified-memory peak of roughly **~100 GB/s** — the naive kernel runs at ~11% of
  peak bandwidth.
- xLAM-2-3b-fc-r (Mungert Q4_0, §9) reads **~1.8 GB per decode step**: all 36 layers' Q4_0
  weights plus norms/biases, plus the **tied 151936×2048 Q4_0 head** (151936 × 2048 × 0.5625
  bytes/elem ≈ **175 MB**) reused as the LM-head matmul.
  - **Bandwidth bound at ~near-peak (~90-100 GB/s)**: 1.8 GB / 100 GB/s ≈ **18-20 ms/step**
    (~**50 tok/s**) — achievable only with a kernel that reuses shared memory / cooperative
    reduction close to peak, i.e. not the naive kernel.
  - **On the naive kernel** (extrapolating stt-wasm's measured ~11 GB/s effective rate): 1.8 GB /
    11 GB/s ≈ **150 ms/step** (~**6 tok/s**) — this is the realistic number if the naive kernel is
    ported as-is without a tiled/cooperative replacement.
- **Prefill is compute-bound**, not bandwidth-bound (M>1 reuses each weight row across all N
  prompt tokens): roughly `2 × 3e9 × N` FLOPs for the 34-tool MCP system prompt, where N is the
  rendered token count. `MODELS.md` does not report the rendered length of the 34-tool prompt —
  **N is not yet known; measure it in Phase 1c** (tokenizer is built there) before this can be
  turned into a wall-clock estimate.

## 11. Kernels and pieces to take from siblings

- **sts-web's fast matvec kernels are Q4_K-only, not Q4_0**, all three hardcoded to `M=1`
  (decode only): `sts-web/crates/sts-wasm/src/wgsl/shader_q4k_matvec.wgsl` (shared-memory input
  caching), `shader_q4k_matvec_coop.wgsl` (K-cooperative), `shader_q4k_matvec_subgroup.wgsl`
  (`subgroupAdd()` reduction, 8 subgroups × 32 threads/workgroup on Apple Silicon). All three
  operate on Q4_K's 144-byte/256-element block, not Q4_0's 18-byte/32-element block — **porting
  any of them to our Mungert-Q4_0 file needs a genuine block-decode-math rewrite**, not a flag.
  `sts-web/crates/sts-wasm/src/gguf.rs:141-157`'s `GgmlDtype` loader accepts both Q4_0 and Q4_K,
  but the fast WGSL kernels themselves never branch on dtype — Q4_K only. No benchmark numbers
  exist for these three kernels specifically (`BENCHMARKS.md` lives in `stt-web`, not `sts-web`;
  the 54.5 ms/step figure in §10 is from stt-web's naive Q4_0 kernel). **No Q4_0 variant of the
  cooperative/subgroup kernel exists anywhere across `sts-web`, `stt-web`, or `tts-web`** — the
  only Q4_0 matvec/matmul shader in any of the three repos is stt-web's `shader_naive.wgsl` (§2),
  general over M but bandwidth-inefficient (§10).
- **tts-web's tied-head path** (`tts-web/crates/tada-wasm/src/gguf.rs`): `to_gpu_f32()`
  (~L663-680) dequantizes the entire Q4_0 embedding table to F32 on CPU, row by row, and uploads
  it as one `[vocab_size, dim]` GPU tensor — "for tied lm_head where we need the full table
  resident on GPU for matmul." A **load-time, CPU-side, one-shot dequant**, not a per-step GPU
  cost (~1.16 GB F32 upload once, same magnitude as stt-wasm's `dequant_embedding_to_gpu`, §1).
  `load_f32_weight_any()` (~L782-812) is the general form for any F32/F16/Q4_0/Q8_0 weight.
- **Recommended starting kernel: stt-wasm's `shader_naive.wgsl`, not any sts-web Q4_K kernel.**
  It's the only Q4_0 kernel in any of the three repos, already general over M, and covered by
  `stt-wasm/tests/q4_matmul.rs`. It's bandwidth-inefficient (§10: ~11 GB/s vs. ~100 GB/s peak), so
  the real work is not porting a sibling kernel wholesale but **writing a Q4_0 cooperative/
  subgroup kernel modeled on sts-web's `shader_q4k_matvec_subgroup.wgsl` reduction strategy**
  (shared-memory row caching + `subgroupAdd()`) with Q4_0 block-decode math. tts-web's
  `to_gpu_f32()` is directly reusable as-is for the tied-head load-time dequant either way.

## 12. Gap list for Qwen2.5-3B (xLAM-2-3b-fc-r), revised

| Gap | Size | Notes |
|---|---|---|
| GGUF metadata parsing (currently skipped entirely, §1) | small | Only needed if config should come from the file instead of hardcoded, as today |
| Qwen2 GGUF tensor-name table (new loader, §1) | small–medium | Mechanical rewrite of `load_transformer_layer`/`load_q4_linear` name strings |
| q/k/v bias loading (struct support exists, loader doesn't populate, §1/§4) | small | Add a `*.bias` tensor read alongside each Q4 linear load |
| RoPE convention mismatch: interleaved-pair vs. Llama/Qwen2 rotate-half (§3) | medium | Silent-wrong-output risk if not fixed; either permute weight rows on load or rewrite `apply_rotation` |
| Tokenisation: byte-level BPE, 151k vocab, ~7 MB tokenizer.json, encode+decode (§6, MODELS.md §5) | large | Nothing in stt-wasm transfers — current tokenizer is decode-only, 8001-vocab SentencePiece; likely lands in JS per this repo's own native-vs-WASM precedent |
| Prefill + KV cache: single-token-only → multi-token prefill, plus context large enough for MCP conversations (§3, §10) | large | Kernel itself already supports M>1 (naive kernel, §2); KV cache rewrite (batched write path, growable/bigger-than-751-step buffer) is the real work; prefill is compute-bound (§10) and N (34-tool prompt length) is unmeasured until Phase 1c |
| Q4_0 decode kernel port/rewrite: naive kernel works but is bandwidth-inefficient; no ready-made fast Q4_0 kernel exists in any sibling repo (§10, §11) | medium–large | Recommended path: start from stt-wasm's `shader_naive.wgsl` for correctness, then write a Q4_0 cooperative/subgroup kernel modeled on sts-web's `shader_q4k_matvec_subgroup.wgsl` reduction strategy (Q4_K→Q4_0 block-math rewrite, not a port) — needed to hit the ~18-20 ms/step bound instead of ~150 ms/step |
| Sliding-window mask always-on → make optional/full-causal (§3) | small | `Q4Attention::new(..., sliding_window: None)` for Qwen2's full-context attention |
| Tied embeddings/lm_head (currently two independent tensors in stt-wasm, §4) | small | tts-web's `to_gpu_f32()` (§11) is a directly reusable load-time pattern: one CPU-side dequant pass, one GPU buffer shared between embedding lookup and LM-head matmul |
| MCP tool-call sampling/formatting: detect bare-JSON-array output (MODELS.md §3), greedy or constrained decode | large | No sampling logic beyond argmax exists in stt-wasm (`stream.rs` only reads GPU argmax) — greedy-only; tool-call-structured decoding is new work entirely |
| PyTorch-logits reference/comparison harness (§7) | medium | No numeric-tolerance-vs-PyTorch test exists to copy; would follow `e2e_pytorch_mimi.rs`'s "dump JSON offline, compare in Rust test" pattern but needs building from scratch |
| lm_head at 151936×2048: single 175 MB Q4_0 buffer, untested against actual `maxStorageBufferBindingSize` (§2, §10) | medium | Requesting adapter's full limits (already done in stt-wasm, §2) likely covers it on M2/Metal, but unverified; may need to shard the head matmul across dispatches if not |

**Effort estimate (Phase 1b — decoder + kernels + KV + sampling on this engine): roughly
18-28 worker-days**, revised up from the pre-audit 15-25 range now that the kernel question is
settled: sts-web's fast kernels don't shortcut the work (§11 — Q4_K-only, M=1, need a real
block-math rewrite to serve Q4_0), so hitting the ~50 tok/s bandwidth bound (§10) instead of the
~6 tok/s naive-kernel floor is now budgeted as its own line item. Four items dominate: (1)
**tokenisation** — 151k-vocab BPE encode+decode, a real subproject (likely JS-side per this
repo's precedent), plus a new prompt-in/token-stream-out worker protocol on the critical path;
(2) **prefill/KV cache** — the naive kernel already supports M>1 mathematically, but the
KV-cache rewrite (batched writes, MCP-length-context buffer) is untouched code, and prefill cost
is unmeasured until N (the 34-tool prompt's token count) is known in Phase 1c; (3) **RoPE
rotate-half** — cheap to fix, expensive to catch if missed, so the from-scratch PyTorch-logits
reference harness (§7) must exist from day one; and (4) **the Q4_0 kernel port** — confirmed to
need new WGSL, not a sibling-repo copy, to move from ~150 ms/step (naive) to the ~18-20 ms/step
bound (§10); treating the fast kernel as a stretch goal rather than budgeting it is the main
schedule risk here. Secondary risk unchanged: the 175 MB single-buffer tied head is unverified
against real WebGPU `maxStorageBufferBindingSize` on M2/Chrome.

## 13. Browser (wasm-bindgen bindings, `crates/llm-wasm/src/web.rs`)

### Bindings surface

- `initWgpuDevice(): Promise<void>` — module-level async fn, mirrors stt-web's `initWgpuDevice`
  (`stt-wasm/src/web/bindings.rs`): requests a `BROWSER_WEBGPU` adapter with
  `PowerPreference::HighPerformance`, then a device with the **adapter's full limits** (not spec
  defaults) — required because the tied lm_head is a single ~175MB Q4_0 buffer
  (`maxStorageBufferBindingSize` needs to clear that; unverified on real hardware, same open risk
  docs/ENGINE.md §12 already flagged for the native path). Must be awaited exactly once before
  constructing any `LlmEngine`.
- `new LlmEngine()` — cheap constructor; grabs the device `initWgpuDevice` stashed in a
  `static OnceLock<WgpuDevice>`.
- `engine.appendModelShard(bytes: Uint8Array)` — call once per GGUF shard (see "shard decision"
  below) before `load()`.
- `await engine.load(tokenizerJson: string, tokenizerConfigJson: string, onProgress?: (stage, step, total) => void): Promise<void>` —
  parses the GGUF via `Q4ModelLoader::from_shards` (two-phase: parse+load tensors, drop the
  reader, then `finalize()` onto GPU — same pattern as native and as stt-web), builds the
  `Tokenizer` and `ChatTemplate`, and allocates a `KvCache` sized `DEFAULT_MAX_CTX = 12288` (same
  sizing rationale as `kv.rs`'s doc comment: the 34-tool Sonos prompt plus conversation headroom).
  `onProgress` fires three times (`"parsing-gguf"`, `"finalizing-gpu"`, `"ready"`) — coarse,
  because `gguf.rs`'s `load_deferred` loads all 36 transformer layers in one call with no
  per-layer hook. Byte-level shard *fetch* progress is `worker.js`'s job, before ever calling
  `appendModelShard`.
- `engine.setSystemPrompt(text: string)`.
- `await engine.start(utterance: string, toolsJson: string, optsJson: string): Promise<string>` —
  begins a turn; `toolsJson` is a JSON array of MCP `tools/list` entries
  (`{"name","description","inputSchema"}`), `optsJson` is `{"maxNewTokens"?, "maxSteps"?,
  "systemPrompt"?, "constrained"?, "requireToolCallFirstStep"?, "diet"?}` (all optional;
  `constrained`/`requireToolCallFirstStep`/`diet` default `true` — see "Agent loop"'s
  `require_tool_call_first_step` section for what the latter does and why).
  Returns a JSON **string** (not `serde-wasm-bindgen`, to avoid adding that dependency for a shape
  this simple):
  `{"outcome":"needTools","calls":[{"call_id","name","arguments"}...],"step":{"promptTokens","text","prefillMs","decodeMs","tokens","modelSteps","forcedTokens","retries","toolErrors","toolsForced"}}`,
  `{"outcome":"final","text":...,"step":{...}}`, or `{"outcome":"error","message":...}`.
  **Mirrors `agent.rs::Agent`'s behaviour** (`docs/ENGINE.md` "Agent loop" / "Schema-constrained
  decoding" — this was `web.rs`'s TODO, now done): `LlmEngine::run_step` builds a fresh
  `Grammar::for_tools` every step from the current `tools` + an `IdValues` accumulated across the
  whole conversation from every tool result (`provideToolResults`), wraps it in a
  `GrammarConstraint`, and drives `LlmEngine::generate_attempt`'s decode loop through the same
  jump-forward semantics as `model.rs::LlmModel::decode_with_constraint` (batched `forward_hidden`
  for forced runs of at least `JUMP_MIN_TOKENS = 8`, masked-argmax otherwise) — on by default
  (`opts.constrained`). An empty tool-call array, unparsable `[...]` JSON, or a call naming a tool
  outside `tools` doesn't become an assistant turn: retry 1 forces constrained decoding on (if not
  already), retry 2 appends a generic nudge (`"Respond with a tool call from the list or a final
  answer."`) as a `user` message, and exhausting 2 retries returns `"outcome":"error"` —
  retries don't consume `maxSteps`. `provideToolResults` checks each result with
  `tools::tool_error_message` (an MCP `isError: true` result or a top-level `error` field) and
  feeds an error back as `{"error": "<msg>"}` via `tools::format_tool_error` instead of the normal
  tool-result shape; 3 consecutive tool errors gives up with `"outcome":"error"` instead of
  looping. `start()`/the KV-image entry points (`prefixKey`/`importKvImage`/`exportKvImage`) run
  raw MCP tool lists through `schemadiet::diet_tools(_, DietLevel::Level1)` before `Tool::from_mcp`
  when `opts.diet` (default `true`) is on — all four entry points use the same `self.diet` flag so
  the rendered/cached prefix stays consistent between `start()` and the KV-image path. `step`'s
  `modelSteps`/`forcedTokens` mirror `model.rs::GenerateStats` (a jump-forward run of `k` tokens is
  1 model step, not `k`); `retries`/`toolErrors` mirror `agent.rs::Step`'s same-named fields.

  **2026-09-11 addendum (coordinator-reported browser bug + fixes, both loops).** A live browser
  run ("resume bloupblip") produced six consecutive `get_households_and_groups_and_players({})`
  steps — every id-taking tool was uncallable. Root cause: `grammar.rs::IdValues::collect_from_result`
  didn't parse JSON embedded in a string value, but a real MCP `tools/call` result's payload
  *is* a JSON string inside `content[0].text`
  (`{"content":[{"type":"text","text":"<pretty JSON>"}]}`), not a bare parsed value — native evals
  passed because their fixtures handed the parsed value directly. Fixed generically in `walk`
  (shared by both loops): every string value is now tried as JSON first and recursed into if it
  parses, falling back to the `looks_like_id` heuristic only when it doesn't (a real id string
  never parses as JSON, so no false negatives). `agent::FixtureCaller` and this page's own
  `toolCaller` (`index.html`) were both updated to serve MCP-shaped results, so native evals and
  the browser path now exercise the same parsing shape. Two more fixes landed alongside it:
  - **Generic repeated-call loop guard** (both loops): a well-formed, valid call byte-identical
    to the immediately preceding step's call(s), fed a non-error result, doesn't become a step —
    it retries via the nudge (never the constrained-force retry, since the constraint was already
    satisfied) and gives up with `"model is repeating a call"` after `max_retries`. `Step`/`step`
    gained nothing new for this (it's folded into the existing `retries` count).
  - **Fail-open** (`grammar.rs::Grammar::for_tools_unrestricted_ids` + `callable_tool_names`): if,
    after building the id-restricted grammar, every still-callable tool is a read tool
    (`get_`/`list_`-prefixed) already called this turn — the id rule left nothing new to try —
    the grammar is rebuilt without the id restriction (every `*_id`/`*_ids` property becomes an
    ordinary free string) instead of leaving the model boxed into repeating itself. Both
    `Step.id_rule_relaxed` (native) and `step.idRuleRelaxed` (`web.rs`'s JSON) record when this
    fired.

  **Prefix-key debugging (`LlmEngine::prefixInputs`, `js_name = prefixInputs`).** A second
  coordinator-reported bug: a prefix KV image built from a fixture tools file + eval system
  prompt + listing-first tool order missed against the live page (its own `tools/list` schemas,
  subset order, system prompt) — any byte of difference changes `prefixKey`. `prefixInputs(tools_json,
  system): string` (sync, no GPU) returns `{"tools": <dieted raw JSON>, "system", "diet",
  "prefixText", "prefixTokens", "modelFingerprint", "prefixKey"}` — the exact bytes `prefixKey`/
  `start()` would use — so a caller can save `{system, tools}` and hand it to `kv-export --tools
  <file> --system <system>` to reproduce the key exactly (`kv-export`'s `load_tools_generic`
  doesn't diet its `--tools` input at all, so the dump must already be post-diet). Wired up via
  `web/agent/index.html`'s "Export prefix inputs" button and `worker.js`'s `dumpPrefix` message,
  and `scripts/headless/run.mjs --dump-prefix <path>`. `worker.js` also now logs the prefix key on
  every `run` (`console.log('[llm-worker] prefix key: ...')`, captured by the headless harness),
  and its `status` notes are honest about hit vs. miss instead of a generic "checking" that read
  the same either way: miss is `"no KV image for this tool set (key …); prefilling once, will be
  saved to browser storage for next time"`, hit is `"KV image loaded (N tokens, M MB, T ms, from
  <opfs|network>, key …)"`.

  **Robust OPFS write-back.** The previous design exported+saved a freshly-prefilled prefix to
  OPFS via a JS-side fire-and-forget call made *after* `engine.start()`'s whole step (prefill +
  decode) had already resolved — a decode-time or tool-call failure later in the same turn meant
  the save never happened, even though the prefix itself had been correctly prefilled. Moved
  entirely into `generate_attempt` (`maybe_export_kv_prefix_to_opfs`, a free function — see its
  doc comment for why not an `LlmEngine` method): right after the prefill that makes
  `resident_tokens` cover the rendered system+tools prefix, and *before* the decode loop starts,
  it exports the prefix (`KvCache::export_prefix_async`) and writes `<key>.kvimg` straight to OPFS
  from Rust (`opfs_save_kv_image`, new `web-sys` features: `WorkerGlobalScope`/`WorkerNavigator`/
  `StorageManager`/`FileSystemDirectoryHandle`/`FileSystemFileHandle`/
  `FileSystemGetFileOptions`/`FileSystemWritableFileStream`/`WritableStream`) — no JS
  orchestration needed for the save itself to be durable, and it happens once per session per key
  (`LlmEngine.kv_image_exported_key`, also set by a successful `importKvImage` so an imported
  prefix is never redundantly re-exported).
- `await engine.provideToolResults(resultsJson: string): Promise<string>` — `resultsJson` is
  `[{"call_id","result"}...]`, keyed by the `call_id`s from the prior `needTools` outcome; same
  return shape as `start`.
- `engine.reset()` — drops the in-progress conversation and the KV cache's resident length
  (`cache.restore(0)`), keeps the loaded model.
- `engine.info(): string` — JSON with `numLayers`/`hiddenSize`/`vocabSize`/`maxCtx`/`cacheLen`
  (or `{"loaded": false}` before `load()`).
- No `on_token` streaming callback: `model.rs`'s `generate()` has no per-token hook at HEAD, and
  adding one would mean editing `model.rs` (out of scope — it's phase 1b's file, being edited
  concurrently). A step's full text arrives in one `start`/`provideToolResults` resolution when
  the step completes.

### Load path: bytes -> GPU

The actual `xLAM-2-3b-fc-r-q4_0.gguf` on disk is a **single 1.74GB file**, not pre-sharded.
`gguf.rs`'s `GgufReader`/`Q4ModelLoader` need `Read + Seek` over the whole logical byte range up
front (header + all tensor offsets are read before any tensor's bytes), so there's no streamed/
incremental parse available to consume bytes as they arrive off the wire — `ShardedCursor` reads
`Vec<Vec<u8>>` in memory, `Read+Seek`-compatible, purely to avoid ever forcing one contiguous
>2GB `Vec<u8>` allocation (wasm32 pointer/length fields and some allocators choke well before the
4GB address-space ceiling). **Decision: `worker.js`'s `load` handler streams each shard URL's
fetch reader straight into the engine, coalescing chunks to ~16MB before each `appendModelShard`
call (progress reported at least every 8MB)** — no full-shard JS buffer, and no attempt to stream
tensor-by-tensor GPU upload beyond that. Peak wasm-linear-memory residency is one coalesced chunk
at a time, not the whole 1.74GB shard, plus the GPU-side buffers `finalize()` allocates
(dropped from CPU memory once GPU upload completes, per the existing two-phase `Q4ModelLoader`
design), comfortably clears wasm32's 4GB ceiling with room for the KV cache (~906MB f32 at
12288 ctx, `kv.rs`'s own arithmetic) and working buffers. If disk-side sharding is ever added
(e.g. to parallelize the HTTP fetch), `appendModelShard` already accepts it — `worker.js`'s
`model.shards` is a URL array for exactly this reason — with no Rust-side change needed.

### The step-wise loop, or: why `web.rs` doesn't call into `Agent`

`agent.rs`'s `Agent::start`/`provide_tool_results` are synchronous — correct for native, where
Burn's wgpu backend can block on `Tensor::into_data()` for logits readback. In the browser that
readback is a WebGPU buffer map, asynchronous only (`Tensor::into_data_async().await` — see
`model.rs`'s `logits_to_vec` doc comment). A synchronous `Generator::generate` can't wrap that, so
there's no GPU-backed `Generator` impl for wasm to plug into `Agent`. `LlmEngine::run_step`
(private, in `web.rs`) therefore runs its own async render -> encode -> restore-prefix -> prefill
-> decode -> parse loop directly against `model.rs`'s public `forward_hidden`/`lm_head`,
`kv.rs`'s `KvCache::restore`, and `sample::greedy`/`tools::parse_output` — the same functions
`Agent::step_inner` calls natively, just awaited per-token instead of called synchronously. This
duplicates *orchestration* (message/tools bookkeeping, prefix-length bookkeeping) between
`agent.rs` and `web.rs`, not any model/tokenizer/template/tool-parsing logic. If Burn ever exposes
a sync-over-async escape hatch for wasm32, this duplication could be collapsed by implementing
`Generator` for a wasm-backed type instead.

**TODO — `prefillMs`/`decodeMs` are wall-clock around a blocking call, not phase-accurate.**
Each `step.prefillMs`/`step.decodeMs` the page reports (via `worker.js`'s `status` brackets and
`step.tokens`/`promptTokens`) is measured as elapsed wall-clock time around the single blocking
`await engine.start(...)`/`await engine.provideToolResults(...)` call, split by whatever
`web.rs` internally timestamps between its prefill and decode phases. Because WebGPU submission
is async and the browser can pipeline/queue work, `prefillMs` currently measures how long it took
to *submit* the prefill dispatch, not how long it took the GPU to *complete* it — decode's timer
start can therefore begin before prefill's GPU work has actually finished, skewing the split
between the two numbers. **The `totalMs`/end-to-end number is trustworthy; the prefill/decode
split is not**, until `web.rs` inserts an explicit GPU sync (e.g. an `into_data_async()` readback
or an explicit queue-submitted-work-done fence) between the two phases. Not fixed here — this is
a Rust-side change (`crates/llm-wasm/src/web.rs`), out of scope for the JS-side owner of this
file's Browser section.

### Prefix cache

Both `Agent` (native) and `LlmEngine` (`web.rs`) track `resident_tokens`: the exact token sequence
currently written into the KV cache, position-for-position — the last rendered prompt plus
whatever of its generation was actually forwarded through the model (every generated token except
a trailing stop id, which ends the decode loop *before* it's ever fed through a forward pass, so
it never enters the cache). On every step, before prefilling, both compute `common_prefix_len`
between `resident_tokens` and the newly rendered prompt's tokens — the longest run of leading
tokens the two share, which is a prefix of both by construction and never longer than
`resident_tokens` — and pass that length as the "already resident" hint (`Agent` to
`Generator::generate_constrained`'s `prefix_len` argument; `LlmEngine` directly to
`KvCache::restore()`), prefilling only the remainder.

This reuses far more than just the constant system+tools preamble: within one multi-step agent
turn, each step's prompt is almost entirely a prefix of the *previous* step's prompt-plus-
generation — prior tool calls and tool results included — not just the tools-set-constant part.
Restoring only to the constant prefix (what this cache used to do, keyed by `tools` and computed
once via a throwaway probe render) re-prefills that entire growing conversation tail on every
non-first step. `crates/llm-wasm/tests/resident_reuse.rs` quantifies the difference on the real
tokenizer/template against a 3-step tool-call turn built from the 12-tool Sonos fixture
(`fixtures/sonos/tools-12.json`): restoring to just the constant prefix costs 1233 tokens of
prefill across the three steps; restoring to the longest common prefix with `resident_tokens`
costs 769 — a 37.6% reduction, concentrated in the later steps (step 2 alone: 807 tokens under the
old scheme vs. 377 under the new one, more than half saved) as the conversation tail — and the gap
between the two schemes — grows.

The re-rendered assistant tool-call turn `Agent`/`LlmEngine` append to `messages` after a tool-call
step (built from the parsed `ToolCallEntry`/`ToolCallFunction`, via `template.rs`'s `py_tojson`) is
byte-identical to what the model actually generated for that turn — also verified in
`resident_reuse.rs` — so the common-prefix match isn't cut short there; the only points where it
actually degrades are genuine content changes (a different tool call, a different tool result).

### The three local dev servers

1. `scripts/serve_models.py --dir ~/Code/idle-intelligence/models` (port 8001) — GGUF + tokenizer
   files, Range-request + CORS aware (pre-existing, not owned by this work). Exposes
   `/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf` and
   `/hf/xLAM-2-3b-fc-r/{tokenizer.json,tokenizer_config.json}`.
2. (reserved for a real MCP tool server in a later phase — this demo's two tools are canned
   in-page, no server needed.)
3. `python3 web/agent/serve.py` (port 8002) — serves `web/agent/` (the demo page, `worker.js`,
   `llm-client.js`, `pkg/`) with `Cross-Origin-Opener-Policy: same-origin` /
   `Cross-Origin-Embedder-Policy: credentialless` (WebGPU/cross-origin-Worker requirements) and
   permissive CORS, stdlib-only (mirrors `scripts/serve_models.py`'s style).

Open `http://127.0.0.1:8002/` once both are running.

### Build

```bash
CARGO_BUILD_JOBS=4 wasm-pack build --target web --out-dir web/agent/pkg crates/llm-wasm \
  --no-default-features --features web
```

Note: `--out-dir` is resolved relative to the crate directory being built
(`crates/llm-wasm/`), not the invocation cwd — after the build, move (or symlink) the emitted
`crates/llm-wasm/web/agent/pkg` to `web/agent/pkg` at the repo root if it doesn't land there
directly. `pkg/` is committed (same convention as stt-web) — remove `pkg/.gitignore` (wasm-pack
always emits one with a bare `*`) before staging, or nothing under `pkg/` will be tracked.

### Rebuild after the Q4 repack / cooperative-matvec / fused-RMSNorm changes

`web.rs` was checked against the post-repack `model.rs`/`gguf.rs`/`kv.rs`/`agent.rs` APIs
(repacked Q4 tensors, cooperative M=1 matvec + subgroup variant, fused RMSNorm, in-place KV
writes, `preserve_order` JSON, step timing fields) — no source changes were needed beyond the
subgroup wiring below; `forward_hidden`/`lm_head`/`KvCache::restore`/`into_data_async` all kept
their signatures. `cargo clippy --target wasm32-unknown-unknown --no-default-features --features
web -p llm-wasm` is clean and `wasm-pack build` (command above) succeeds; `pkg/` was rebuilt and
recommitted.

`pkg/` rebuilt 2026-09-10 after review fixes; headless runs pass.

`initWgpuDevice()` now probes `adapter.features().contains(wgpu::Features::SUBGROUP)` and calls
`gguf::set_subgroup_support(...)` before requesting the device, mirroring
`sts-wasm/src/web/bindings.rs`'s `initWgpuDevice` exactly (same feature-diff-then-probe shape
already present in `web.rs` for `MAPPABLE_PRIMARY_BUFFERS`) — the browser now uses the subgroup
matvec kernel when the adapter reports it, falling back to the portable kernel otherwise.
Unverified in a real browser (no headless browser available in this environment) whether any
adapter actually reports `SUBGROUP` support.

### What is untested

No headless browser is available in this environment (Playwright/Chrome are explicitly
off-limits per this task's constraints) — everything above is verified by: `cargo check`/`cargo
clippy --target wasm32-unknown-unknown --no-default-features --features web` (clean), `wasm-pack
build` succeeding, and native `cargo test -p llm-wasm` for the render/encode/parse/prefix-cache
logic `web.rs` shares with `agent.rs`. **Never exercised**: `initWgpuDevice()` actually acquiring
a WebGPU adapter/device in a real browser; `maxStorageBufferBindingSize` actually covering the
175MB tied-head buffer; the full `load()` -> `start()` -> tool-call round trip -> `Final` path
end-to-end; `into_data_async()`'s actual behavior/latency under real WebGPU buffer mapping; Worker
+ module-script + COOP/COEP interaction in a real browser (`type: 'module'` Workers plus
cross-origin isolation headers have historically had browser-specific rough edges). The user's
first click-through in a real browser is the first real signal on all of these.

### `worker.js` hardening (ported from trucs.ai's sonos stub)

`web/agent/worker.js` was hardened to match the error-handling/observability bar set by
trucs.ai's `sonos/llm-worker.js` (same protocol, drop-in-compatible): shard/tokenizer/template
fetches use `cache: 'no-store'` so a stale cached GGUF or tokenizer never masks a real reload;
`load` guards against an empty `model.shards` array and a missing `tokenizerUrl`/`templateUrl`
before fetching anything, and checks `res.ok` on the tokenizer/template responses (not just the
shard); thrown errors carry their stack trace in the `error` message's text; the whole `run` loop
is wrapped in try/catch so an engine-side throw reaches the page as an `error` message instead of
an unhandled rejection; top-level `self.addEventListener('error'|'unhandledrejection')` relay
anything that would otherwise vanish silently (e.g. a throw during the top-level `import` of
`pkg/llm_wasm.js`); shard fetch progress is throttled to report at most every 8MB instead of on
every chunk; and new `{type:'status', phase:'prefill'|'decode'|'idle'}` messages bracket the
blocking `engine.start`/`provideToolResults` calls so `index.html` can drive a page-side
"thinking... (N.Ns)" timer instead of the UI looking hung while the model runs. The
`?stub=1` dry-run path and sonos-specific bits were not ported — this demo's two canned tools
(`get_weather`, `set_thermostat`) already exercise the same UI without needing a fake model path.

## Known issues / fixed

### 2026-09-11: Fused decode-time attention kernel for the production F32 KV cache (Session 14)

`shader_attn_decode_f32.wgsl` fuses decode's (t==1) QK^T -> softmax -> PV into one dispatch per
layer (one workgroup per query head, GQA-aware, f32 accumulation throughout), reading K/V directly
out of `kv.rs`'s `KvDtype::F32` cache tensors via a new `KvCache::f32_layer` raw-handle accessor —
no dequant step (this cache is already f32), replacing ~8 Burn dispatches (QK^T matmul, scale
multiply, causal-mask compare+fill, softmax's several ops, PV matmul via `pv_matmul`'s
contraction-chunking workaround, plus `repeat_kv`'s `cat`) with 1. It's the F32-cache counterpart
of Session 13's `shader_attn_decode_q8.wgsl`, sharing that kernel's three-phase structure (raw
scores + running max, then exp/sum, then PV) and `workgroupUniformLoad` uniformity discipline, but
without any block/word unpacking. Measured (`docs/BENCHMARKS.md` Session 14): a ~10% decode
ms/token win at kv_len~2225, but a regression at kv_len~8140 — 16 workgroups total isn't enough to
keep the GPU saturated once each thread's unstrided per-key V-accumulation loop (phase C) dominates
at long context. `model.rs`'s `Q4Attention::forward` therefore gates the fused path behind
`FUSED_DECODE_ATTN_MAX_KV_LEN = 4096`, falling back to the existing chunked Burn-matmul path beyond
that; a tiled/two-pass variant with more workgroups per head is the likely fix for long context but
wasn't attempted this session.

### 2026-09-10: Burn/cubecl wgpu matmul mis-computes for large-K/small-N shapes (fixed: chunked-attention 11.14 logit divergence)

Symptom (Session 6, docs/BENCHMARKS.md): `full_forward.rs::split_prefill_matches_single_prefill`'s
split=1000 case (prefix 1000 rows, then 1225 rows at offset 1000) showed max-abs
logit diff 11.14 vs a single whole-prompt prefill. Session 6 traced this to
`model.rs::attention_scores_and_values`'s `t > ATTN_QUERY_CHUNK` branch and
exonerated `gguf.rs`/WGSL (both matmul kernels agree with a CPU reference to
~1e-5 on every real weight shape, at every M value from the failing split) but
left it unfixed as a KNOWN BUG, hypothesizing non-contiguous strides from
`Tensor::narrow` on a permuted tensor feeding `Burn::matmul`.

Root cause (Session 7): **not** contiguity — forcing every chunked tensor
through a sync CPU round-trip before `matmul` produced a bit-identical
divergence. Bisecting QK^T -> mask -> softmax -> P@V independently against a
CPU f64 reference (`model.rs::debug_tests::chunked_attention_matches_unchunked_synthetic`,
fully synthetic, no GGUF model) isolated the bug to the P@V matmul
(`probs.matmul(v)`) specifically: QK^T and softmax matched the CPU reference to
~1e-6/~1e-8, but `probs.matmul(v_chunk)` on shape `[1,H,256,1256]` x
`[1,H,1256,16]` did not. Confirmed as a Burn 0.20/cubecl wgpu matmul kernel bug
unrelated to attention entirely: a plain `matmul` on fresh random tensors of
that same shape (no softmax, no masking) diverged from a CPU reference by 33.79
max-abs. Trigger appears to be a large contraction dimension (K = kv_len, low
thousands during prefill) paired with a small output width (N = head_dim).

Fix: `model.rs::pv_matmul` chunks the P@V matmul's contraction (K) dimension
into 256-element blocks and sums partial products instead of one large-K/
small-N `matmul` call, used in both branches of `attention_scores_and_values`.
Verified: all `split_prefill_matches_single_prefill` splits (including two new
ones matching the real MCP tool-result-suffix shapes, 133-token and 531-token
suffixes on `03_tools_multiturn`) now hold the tight 3e-4 bound (measured
7e-5-1.3e-4); `test_forward_02/03` remain greedy-exact; prefill tok/s unchanged
(75.52 tok/s on `02_tools_single`, above Session 5's 73.3 baseline). Did not
patch cubecl itself (out of scope) — this workaround avoids the bad kernel
selection rather than fixing it upstream, so any *other* future call site with
a similarly large-K/small-N matmul shape should chunk its contraction dimension
the same way rather than assuming Burn's `matmul` handles it correctly.

### 2026-09-10: Tint WGSL uniformity rejects `workgroupBarrier`/`subgroupAdd` after a storage-buffer-gated early return

Symptom, real Chrome (Dawn/Tint), first compute dispatch of a run:
`createShaderModule failed: Error while parsing WGSL: :61:9 error:
'workgroupBarrier' must only be called from uniform control flow` /
`control flow depends on possibly non-uniform value` / `reading from
read_write storage buffer 'info' may result in a non-uniform value`.
Native wgpu/Naga does not run this analysis, so the bug was invisible to
`cargo test --features wgpu` and only surfaced in the browser.

Root cause: three fused kernels each had an early `return` guarding
per-workgroup bounds (`row >= rows` in `shader_rmsnorm.wgsl`, `b >= B` in
`shader_q4_matvec.wgsl` and `shader_q4_matvec_subgroup.wgsl`) that executed
*before* the kernel's `workgroupBarrier()`/`subgroupAdd()` calls. The guard
condition (`row`/`b` == a builtin id) is uniform per workgroup, but the
bound it's compared against (`rows`/`B`) is loaded from a `storage` buffer
(`info`), and Tint's WGSL uniformity analysis conservatively taints *every*
storage-buffer load as non-uniform — it cannot prove the load reads the
same address for every invocation, so any branch on that value that can
skip a later barrier is rejected, even when (as here) the skip is
workgroup-uniform in practice.

Fix, same pattern in all three shaders: never branch around a barrier or
subgroup op. Compute a `valid`/`b_valid` flag once, guard only the
loads/accumulation/store with `if (valid)`, and leave every
`workgroupBarrier()`/`subgroupAdd()` call at the top level of `main` (or of
an unconditional loop) so it is always reached by the whole workgroup
regardless of the storage-buffer-derived bound. Invalid lanes contribute 0
to shared-memory reductions, which doesn't change results for valid lanes.
Changed: `crates/llm-wasm/src/wgsl/shader_rmsnorm.wgsl`,
`shader_q4_matvec.wgsl`, `shader_q4_matvec_subgroup.wgsl`. No dispatch-shape
or binding changes, so `gguf.rs` was untouched.

Audited and left as-is: `shader_naive.wgsl` has an early return but no
barriers/subgroup ops at all, so the rule doesn't apply (this is why it was
already known to pass Chrome). `shader_q4_tiled.wgsl` (native-only,
`#[cfg(not(target_arch = "wasm32"))]` in `gguf.rs`, never compiled for
Tint) already followed the correct pattern — its per-element bounds checks
(`n_global < N`, `m_global < M`) gate `if/else` value selection, not a
`return`/`break` that skips a barrier, and its K-loop barriers are
unconditional at loop-body top level.

Rule for future kernels: **guard work, not barriers.** A `workgroupBarrier`
or `subgroupAdd` call must never sit inside an `if`/`for`/`while` whose
condition was computed from a storage-buffer load, and no `return`/`break`/
`continue` derived from such a load may skip a barrier call for only some
invocations. Push the bounds check down into the loads/stores instead.

**Addendum, same day:** the guard-work-not-barriers rule alone is not
sufficient when the *loop itself* is bounded by a storage-buffer value and
contains a barrier — `shader_q4_matvec.wgsl`'s tile loop
(`loop { if (tile_start >= K) { break; } ... workgroupBarrier(); }`) still
tripped Tint in the headless Chromium harness even after the return-before-
barrier fix, because the loop's exit condition depends on `K` (read from
`info`), which Tint taints non-uniform regardless of guard placement. Fix:
stage every control value that drives a barrier-gating loop/branch through
`var<workgroup>` + `workgroupUniformLoad` — thread 0 writes `info`'s fields
into a `var<workgroup> array<...>`, then every thread reads it back via
`let x = workgroupUniformLoad(&wg_array);` (one call for the whole array,
not one per scalar — it already contains its own barrier pair). Tint's
uniformity analysis special-cases `workgroupUniformLoad`'s result as
uniform. Applied to `shader_rmsnorm.wgsl`, `shader_q4_matvec.wgsl`,
`shader_q4_matvec_subgroup.wgsl`. Verified end-to-end in the headless
Chromium harness (`web/agent/`, real Dawn/Tint): full prefill+decode with
a tool call and a follow-up turn completed with no shader errors. Updated
rule: **any control value that gates a loop or branch containing a barrier
or subgroup op must be loaded via `workgroupUniformLoad`, not read directly
from a storage buffer** — the guard-work-not-barriers pattern is still
correct for per-thread bounds checks (`row < N`, `b < B`) that only guard
work, never a barrier.

### 2026-09-10: `eval --tools 12` tool preamble silently reordered (not a KV-cache bug)

Symptom: `llm-agent eval --tools 12` on `m01`/`s09` ("Pause the kitchen." / "Resume
playback.") invented a group id directly instead of calling the listing tool
first, while `llm-agent run` on the identical HF-rendered prompt (and HF bf16
itself) called the listing tool first. Initial hypothesis was a split-prefill
bug: the eval harness prefills the constant system+tools prefix once,
`KvCache::snapshot`/`restore`s it, and prefills only the per-turn suffix at a
nonzero `offset` — suspects were RoPE position, the causal mask at `offset >
0`, or `KvCache::append`/`restore` writing/reading the wrong range.

That hypothesis was wrong. `crates/llm-wasm/tests/full_forward.rs`'s
`split_prefill_matches_single_prefill` test (added for this investigation)
prefills a 2225-token prompt whole, then again split at token 2218/1000/2224
with `restore()` in between, and again with `restore()` followed by a
*different* suffix (simulating a second turn) — all three variants match a
fresh single-shot prefill to <1e-4 max-abs-diff in logits and bit-for-bit in
an 8-token greedy continuation. The KV cache / RoPE-offset / causal-mask
machinery in `model.rs`/`kv.rs` is correct at nonzero offset.

The real bug: `s09` is the *first* case run in that eval (no cache reuse
involved at all — a single fresh 2225-token prefill, same code path as
`llm-agent run`) and still produced the wrong tool call, which ruled out
prefix-caching entirely and pointed at prompt construction instead.
`eval::select_tools` (`crates/llm-wasm/src/eval.rs`) built the 12-tool subset
by filtering the full 34-tool list (`tools.json`, alphabetically ordered) down
to the names present in `tools-12.json`, **discarding `tools-12.json`'s own
curated order** (which lists `get_households_and_groups_and_players` — the
listing tool — first) and replacing it with alphabetical order. The model
sees a completely different tool preamble token sequence than the
HF/`llm-agent run` reference (which used `tools-12.json`'s order verbatim),
so it picks a different first action — nothing to do with KV cache offsets.

Fix: `select_tools` now iterates `tools-12.json` in its own order and looks
up each tool's schema in the full 34-tool list, instead of filtering the
34-tool list's order. `llm-agent eval --tools 12 --only s09,m01 --label
split-fix` after the fix: both call `get_households_and_groups_and_players({})`
first (`correct=true`).

### Headless repro

`web/agent/worker.js` imports `./gpu-debug.js` before dynamically importing
`pkg/llm_wasm.js`, wrapping `GPUDevice` methods in error scopes so Chrome's
cascade errors (`[Invalid BindGroupLayout] ... is invalid due to a previous
error`) surface their root validation message instead of only the downstream
noise. `index.html` exposes every log line on `window.__llmLog` for
automation. To reproduce headless with Playwright's bundled Chromium (never
the user's real browser): start `python3 web/agent/serve.py` (port 8002,
COOP/COEP) alongside the model server on :8001, then drive the page with
`chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu',
'--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'] })`
— WebGPU works fine in headless mode on this hardware (Apple M2, metal-3
adapter) as long as the page has navigated to a real http(s) origin first;
`navigator.gpu` is `undefined` on `about:blank`. Click `#load-btn`, wait for
`window.__llmLog` to contain `ready`, fill `#utterance`, click `#run-btn`,
and watch for `[gpu-debug]` lines.

A ready-to-run version of this harness lives at `scripts/headless/repro.mjs`
(see `scripts/headless/README.md`).

## Review fixes 2026-09-10

1. `gguf.rs::GgufReader::open` — `metadata_kv_count`/`tensor_count`/`ndims`
   from the untrusted header no longer drive unbounded `with_capacity`
   (capped at `MAX_CAPACITY_HINT = 1<<16`); `ndims <= 4` is `ensure!`d; every
   tensor's `offset` and `[abs_offset, abs_offset+byte_size)` range is
   checked against the actual file length before any caller can slice it.
2. `GgmlDtype::byte_size`/`GgufTensorInfo::{num_elements,byte_size}` return
   `Result` (`checked_mul` chain, "tensor size overflow"/"element count
   overflow"); `tensor_data` converts the result to `usize` via
   `usize::try_from` with context instead of `as usize` truncation.
   `EmbeddingStore::embed_id`/`embed_id_add_cpu` now return `Result` and
   `ensure!` `id < vocab_size`; propagated through `LlmModel::embed_tokens`
   -> `forward_hidden` -> `forward_logits`/`generate`, and every native
   (`llm-agent.rs`) and WASM (`web.rs`) caller.
3. `model::logits_to_vec` returns `Result` instead of
   `.expect("f32 logits")`; `web.rs`'s async GPU-readback `.expect(...)` maps
   to `JsError` instead. Swept the crate for other runtime-reachable
   `expect`/`unwrap` on untrusted-data paths (GGUF bytes, tokenizer JSON,
   model output) — none left outside `#[cfg(test)]` code and internal GPU
   kernel-launch invariants (not driven by untrusted input).
4. `load_q4_linear`/`load_q4_linear_with_bias` `ensure!(shape.len() == 2, ...)`
   with the tensor name in the message, before indexing `shape[0]`/`shape[1]`.
5. `sample::top_k`'s sort comparator is now NaN-safe: NaN logits are mapped
   to `-inf` for the comparison only (never win top-k), non-NaN values use
   `total_cmp` (total order, never panics). `greedy`'s strict `>` was
   already NaN-safe and deterministic (first true max wins); added tests.
6. `GgmlDtype::byte_size`/`GgufTensorInfo::num_elements` use a `checked_mul`
   chain instead of raw `*`/`.product()` (see finding 2 above — same fix).
7. `LlmEngine::new()` (`#[wasm_bindgen(constructor)]`, can't return `Result`)
   still falls back to `WgpuDevice::default()` when `initWgpuDevice()`
   wasn't awaited first, but now logs a `console.warn` explaining the
   fallback almost certainly means later GPU calls fail; documented on the
   method.
8. Added `tests/gguf_malformed.rs` (8 cases, no GPU): truncated header (two
   variants), absurd `tensor_count`/`metadata_kv_count`, `ndims = 9`, tensor
   offset+size past EOF, offset alone past EOF, and a well-formed-file
   sanity check — all via a small hand-rolled GGUF byte writer. Added
   `sample.rs` unit tests for NaN in `greedy`/`top_k`.
9. Subgroup matvec kernel gating (`shader_q4_matvec_subgroup.wgsl` hardcodes
   `SUBGROUP_SIZE=32`): `web.rs`'s `initWgpuDevice` now only calls
   `gguf::set_subgroup_support(true)` when `wgpu::Features::SUBGROUP` is
   present **and** `adapter.limits().min_subgroup_size ==
   max_subgroup_size == 32`. As of wgpu 26's `BROWSER_WEBGPU` backend these
   limits always come back `0`/`0` (`Limits::default()` — the backend
   doesn't query a real value from the browser), so the subgroup kernel is
   effectively disabled on WebGPU today, with a comment explaining why and
   what would need to change (wgpu, or the WebGPU spec, actually surfacing
   subgroup size) for it to activate.
10. `shader_q4_tiled.wgsl`'s header comment corrected: it previously
    described the shipped kernel as TM=TN=128/MICRO=8, but the constants in
    the file (and what's actually compiled) are TM=TN=64/MICRO=4
    (`docs/BENCHMARKS.md` K2's `v3`, vectorized dequant); the 128/MICRO=8
    variant (`v2`) was measured and found slower, then reverted. No code
    change, comment only.

### Deferred

- **Agent/web duplication** (review finding 7): `web.rs`'s async step loop
  duplicates `agent.rs`'s `Agent::step_inner` orchestration because Burn's
  wgpu tensor readback has no sync-over-async escape hatch on wasm32 (see
  `web.rs`'s module doc comment) — not addressed here.
- **Per-step re-render cost** (review finding 6): each agent step
  re-renders the full chat-template prompt from scratch rather than
  incrementally extending it — not addressed here.

### 2026-09-10: Autotune and prefill shapes

`Tensor::matmul`'s `burn/autotune` (enabled since Session 5) tunes a kernel
strategy per distinct `(M,N,K)` shape and has no persistent cache across a
browser session — every new prefill length M paid a fresh tuning pass, and
since every agent-loop utterance and tool result has a different M, this meant
constant re-tuning rather than a one-time warm-up. Fixed in `gguf.rs`'s
`q4_matmul_dispatch`: the scratch-dequant+matmul path now pads the input's M
up to a fixed bucket (`pad_m_bucket` — multiples of 32 below 128, multiples of
128 at/above) before calling `Tensor::matmul`, and slices the output back to
the real M, so autotune only ever sees a small, reused set of shapes
(128, 256, ..., 2304, ...) regardless of how ragged the actual prefill lengths
are; `SCRATCH_MATMUL_CHUNK_M` dropped from 2225 to 2048 (itself 128-aligned)
so a chunked prefill's remainder chunk stays bucket-aligned too. See
docs/BENCHMARKS.md Session 10 for the in-process bucket-reuse measurement and
the padding-is-numerically-inert test in `tests/q4_matmul.rs`.

**Session 11 update**: bucketing only reduced the *number* of distinct
tuning passes — each one was still a real autotune benchmark, cheap natively
(cubecl caches results on disk at `target/autotune/0.9.0/<device-key>/
burn_cubecl-kernel-matmul-tune-base.json.log`, `CacheConfig::Target`,
see `cubecl-runtime-0.9.0/src/config/cache.rs`) but with no browser
equivalent, so the *first* prefill of a new bucket still paid the full
benchmark-every-candidate cost in Chrome (6.5 min for a 2304-row session).
Inspecting that native cache (`CUBECL_DEBUG_LOG=stdout` env var also
logs autotune picks) shows every CMMA/MMA candidate strategy failing kernel
selection outright on this Metal-via-wgpu backend ("No tile size is
available for the problem"), so the winner at every shape sampled is a
`DoubleUnit` (double-buffered, non-tensor-core) kernel, varying only by
`cubek_matmul::routines::TileSizeSelection` (`MinTileSize` wins at the
dominant M=2048-row scratch-matmul chunk; `MaxTileSize` only wins for
smaller-M remainder chunks). `gguf.rs`'s scratch-matmul path
(`pinned_matmul`) now calls `cubek_matmul::launch::launch_ref` directly with
that `Strategy` pinned as a constant, bypassing Burn's `Tensor::matmul`/
`autotune` entirely for the shapes that actually caused the multi-minute
spike. `burn/autotune` stays enabled for the smaller attention-path matmuls
in `model.rs` (cheap to tune, and empirically *not* safe to run through
cubek's un-benchmarked `Strategy::Auto` fallback — an earlier attempt to
disable `autotune` crate-wide produced numerically wrong attention output).
See docs/BENCHMARKS.md Session 11 for cold/warm numbers.

**Session 15 update**: Session 11's single process-wide pin was chosen only
at the dominant M=2048 chunk shape, and left real throughput on the table at
every other M an agent loop's tool-result prefills actually hit — the
browser gate's ~460-token tool results were measured at ~24 tok/s (~145
GFLOP/s, ~10% of f32 peak on this GPU). `llm-agent prefill-sweep` swept M x
every non-CMMA/MMA `cubek_matmul::Strategy` (CMMA/MMA already dead per
Session 11; `SimpleVecMat`/`DoubleVecMat` newly found dead too — they
require column-major Rhs, but `q4_dequant_scratch`'s output is row-major) at
two production shapes and found three clean regimes: M<=128 the naive
per-element-dequant kernel wins outright (no scratch-dequant/pipeline
overhead to amortize), 129<=M<1024 `DoubleUnit/MaxTileSize` wins by
1.2-1.4x over `MinTileSize`, M>=1024 `MinTileSize` wins (Session 11's
regime, unchanged). `gguf.rs::strategy_for_bucket(m)` now replaces the
single pinned constant with this three-way table (`SCRATCH_MATMUL_MIN_M`
raised 32->129 so M<=128 skips the scratch route entirely, same path as
M==1's matvec); `pinned_matmul`/`scratch_matmul_chunked` call it per chunk,
so a chunked prefill's 2048-row chunks and bucket-aligned remainder each get
their own bucket's winner automatically. See docs/BENCHMARKS.md Session 15
for the full sweep table and before/after numbers (M=460: 26.9->39.8 tok/s
warm, +1.4x, zero regression at M=2225).

## 14. Schema-constrained decoding (`src/grammar.rs`)

A pure state machine — no GPU, no model — that decides, at every decoding
step, which vocab tokens are legal continuations of xLAM-2's tool-call
output. Not wired into the generate loop yet (see "Hook-in point" below);
this section is the contract the eventual wiring builds on.

**The grammar.** `Grammar::for_tools(tools: &[Tool], id_values: &IdValues)`
builds the constraint for one turn. It accepts exactly:

- **Free text** — decided by the first non-whitespace byte: anything other
  than `[` switches the whole output to unconstrained text, accepted until
  EOS. This mirrors `tools.rs::parse_output`'s own detection rule (does the
  stripped output start with `[`).
- **OR** a JSON array of one or more `{"name": ..., "arguments": {...}}`
  call objects, where `name` is one of the tool names and `arguments`
  satisfies that tool's `inputSchema`: every `required` key present (any
  order), no unknown keys, values typed (string / integer / boolean /
  array-of-string), `enum` respected, integers within `minimum`/`maximum`,
  strings within `minLength`.
- Whitespace: both the model's observed style (space after `:`, `, `
  between items — see `fixtures/reference/logits/02_tools_single.json` and
  `03_tools_multiturn.json`'s recorded greedy output) and the fully compact
  form are accepted; anywhere else, no extra whitespace is allowed.

**The id rule.** A string (or array-of-string) property whose name ends in
`_id`/`_ids`, or whose schema `description` contains "ID", must take a
value from `IdValues` — the set of id-shaped strings harvested from earlier
tool results in the conversation (`IdValues::collect_from_result`, generic:
any object key ending in "id" case-insensitively, plus any string that
*looks* id-shaped — has a digit and a `_`/`:`/`-` separator, no whitespace,
length ≥ 6 — no Sonos-specific regex). If that set is empty for a given
property, the property cannot be emitted at all: `Grammar::for_tools` drops
it from the tool's schema, and if it was required, drops the *tool* from
the grammar entirely — its name never appears in the `"name"` choice, so
the model can't even start typing it (`Grammar::can_call`). This is what
`pause` looks like with no known group id: excluded outright, not merely
rejected once `arguments` is wrong.

**Token-level API.**

```rust
let vocab = TokenVocab::from_tokenizer(&tokenizer);      // once per generation
let grammar = Grammar::for_tools(&tools, &id_values);     // once per turn
let mut state = GrammarState::new(&grammar);

loop {
    let mask: TokenMask = state.allowed(&vocab);           // bitset over vocab ids
    let token_id = sample(logits, &mask);                  // not implemented here
    state.advance(token_id, &vocab);
    if state.is_complete() { break; }
}
```

`TokenVocab` precomputes every vocab id's exact byte string once
(`Tokenizer::token_bytes`, added for this — `decode(&[id], false)`, keeping
special tokens rather than silently dropping them) so `allowed()` never
re-tokenizes. `GrammarState::allowed` walks each vocab token's bytes
through the character-level matcher (`Grammar::step`, a hand-written byte
DFA — no generic recursive grammar engine, since the schema shape here is
fixed and flat) from a cheap `Copy` snapshot of the current parser
position; a token is allowed iff every one of its bytes is accepted in
sequence (it need not reach a *complete* grammar state, just not be
rejected outright). EOS is added to the mask separately, only when
`is_complete()`. Property/tool-name/enum/id matching is a linear
prefix-elimination walk over a small candidate list held as `&[String]`
(at most 34 tool names, a handful of properties or enum values per tool) —
no tries built up front, no per-byte allocation.

The one deliberate shared-mask optimization: inside free text, every token
is valid, so `allowed()` returns a precomputed all-ones `TokenMask` instead
of re-walking the vocab — the case the spec called out explicitly ("states
repeat, e.g. inside free text everything is allowed").

Constraints that depend on a variable-length suffix (integer
`minimum`/`maximum`, string `minLength`, array `minItems`) are *not*
checked digit-by-digit or char-by-char; they're checked once, at the
terminal delimiter (the `,`/`}`/`]` right after the value), by re-parsing
the accumulated buffer. This is the same "rejected at the closing
character" shape as the required-field check (test (b)/(d): the `}` that
would close `arguments` is the token that gets rejected, not some byte
earlier in `group_id`/`shuffle`'s absence — there's nothing earlier to
reject). Enum and id values, by contrast, *are* checked byte-by-byte (a
wrong-enum first character is rejected immediately, not at the closing
quote), since the full candidate set is known up front and prefix-matching
it is free.

**Wired up (Session 12).** `sample.rs::greedy_masked`/`top_k_masked` apply
a `TokenMask` to a logits vec before argmax/top-k (disallowed ids simply
never win, no `-inf` rewrite needed since the comparison already skips
them). `grammar.rs`'s `Constraint` trait (`allowed`/`advance`/`forced_run`/
`is_complete`) is the interface `model.rs::LlmModel::generate_with_constraint`
drives; `GrammarConstraint<'g>` is the only implementation, wrapping a
`GrammarState` + the real `Tokenizer` + a `TokenVocab`, with `allowed()`'s
mask recomputed eagerly on every `advance()` so it's a cheap `&self`
reference return. `generate()` is now a thin unconstrained wrapper
(`constraint: None`) over the same function, so every existing
unconstrained caller and `tests/full_forward.rs`'s greedy-exact checks are
untouched.

**Jump-forward.** `forced_run()` does *not* work by checking whether the
`TokenMask` allows exactly one *vocab token* — that condition almost never
holds inside a forced literal, because BPE gives multiple legal vocab
tokens simultaneously at nearly every position (different-length
segmentations of the same forced substring: e.g. `"name"` alone was
observed with `mask.count()` between 3 and 6 at every byte position along
its length, never 1, even though the literal text is fully determined).
Instead, `GrammarConstraint::forced_bytes` walks the grammar's byte-level
DFA (`Grammar::step`) one byte at a time, scanning all 256 possible next
bytes at each position: as long as exactly one byte is ever legal, that
byte is forced, tokenizer-independent. Once branching resumes (>1 legal
byte) or the walk ends, `forced_run()` re-encodes the forced byte span
with the *real tokenizer* (`Tokenizer::encode`, byte-level BPE — a pure
function of the input bytes, no surrounding-context dependence) to get the
canonical token ids, verifies the round-trip (`decode` the result back and
compare bytes — a tokenizer surprise degrades to `None`, i.e. a normal
masked step, never wrong output) and returns those. `model.rs`'s decode
loop appends the whole run via one `forward_hidden` call (`M = run.len()`,
the same multi-token prefill path prefill itself uses) instead of one
decode step per token, then continues from that call's last-position
logits — jump-forward literally skips per-token forward passes for
anything the grammar has already fully decided, at the cost of the
tokenizer re-encode-and-check (cheap: milliseconds, not a GPU op).
Measured on fixture 03 (`tests/constrained.rs`): forced bytes cover the
whole `[{"name": "..."`/`, "arguments": {"..."}}]` scaffolding and any
uniquely-determined enum/id value; genuine choice points (the tool name's
first byte, an enum's first byte when >1 candidate remains) are the only
non-forced steps.

**Agent wiring.** `Agent::set_constrained(bool)` turns this on; `Agent`
accumulates an `IdValues` across the whole conversation from every tool
result (`IdValues::collect_from_result`, called from
`provide_tool_results`), builds a fresh `Grammar::for_tools` every
`step_inner` call from the current `tools` + that running `IdValues`, and
threads it through the new `Generator::generate_constrained` trait method
(default impl: ignores the constraint, behaves exactly like
`generate_with_cached_prefix` — `FixtureGenerator` relies on this).
`Step` gained `model_steps`/`forced_tokens` for eval reporting (a
jump-forward run of `k` tokens is 1 model step, not `k`).

**`web.rs`'s job (not done by this change — that file is owned by the
worker doing the browser wiring).** The web engine's decode loop needs
the same shape `NativeGenerator::generate_constrained` in
`bin/llm-agent.rs` has: build a `TokenVocab` once (cache it — expensive to
rebuild per step), build a `Grammar::for_tools` per step from the current
tool set + an `IdValues` accumulated the same way `Agent` does, wrap it in
a `GrammarConstraint`, and call `LlmModel::generate_with_constraint`
instead of the current unconstrained decode loop when the engine's
constrained flag is set. No new WGSL/WASM concerns — this is pure
CPU-side control flow around the same `forward_hidden`/`lm_head` calls the
unconstrained path already makes, so the existing `into_data_async().await`
readback discipline (no `.into_data()` in WASM) is unaffected either way.

**Limits.** Flat schemas only: `properties` of string/integer/boolean/
array-of-string, no nested objects, arrays only of strings (matches every
schema in `fixtures/sonos/tools.json`). Array-of-string *free* elements
(no id/enum constraint) don't track backslash escapes as carefully as
scalar string values do (accepts any escape pair without validating it's
one of JSON's defined escapes) — acceptable for the id/enum-constrained
elements this schema set actually uses, but a real generic-JSON-string
validator would be stricter. `advance()` on a token the mask didn't allow
is a documented no-op (state unchanged) rather than a panic, since it's the
sampler's job to respect the mask, not this module's job to trust it did.

**Session 12 addendum: don't force into a multi-candidate `QuotedChoice`.**
`forced_bytes`' "exactly one legal next byte" test forces a
multi-candidate id's *shared prefix* too (e.g. all three device ids in
`fixtures/sonos/results` start `RINCON_`, so those 7 bytes are
unambiguous right up to the point the candidates diverge). `forced_run`
then re-encodes that isolated substring with the real tokenizer to get
the ids fed into the KV cache — but BPE segmentation isn't
prefix-invariant, so `Tokenizer::encode("RINCON_", ...)` alone can land on
a different token boundary than the model's own tokenization of the
*full* candidate string ever produces mid-generation. That off-boundary
KV commit biased the very next masked-decode step toward the wrong
candidate in practice (`docs/BENCHMARKS.md` session 12 addendum:
`RINCON_KITCHEN01:1` picked over the correct `RINCON_LIVING01:2` in four
`constrained43` cases). Fix: `Pos::quoted_choice()` lets `forced_bytes`
recognize when the DFA position is inside a `QuotedChoice`'s content
(`QPhase::InContent`) that started with more than one candidate, and it
now stops unconditionally right at the opening quote in that case —
never forcing into the content, not even a byte that's currently
unambiguous. Per-token masked decoding (`GrammarState::allowed`,
unchanged) picks up from there; it already restricts to exactly the
tokens that are prefix-compatible with *some* live candidate, so no
token-boundary distortion is introduced. Also added `model::JUMP_MIN_TOKENS`
(8): a forced run shorter than this is decoded one token at a time via
masked argmax instead of the batched jump-forward path, since a short
run's forced-run computation overhead (DFA walk + tokenizer
encode/decode round trip) isn't worth the batching win — exact either
way, purely a speed knob.

## Agent loop (`src/agent.rs`)

**Malformed-output retry policy.** A live browser run surfaced the model
emitting `[]<|im_end|>` (an empty tool-call array) on two consecutive
steps, each getting appended to history as an empty assistant turn —
wasting 2 steps and teaching the model its own junk output was a valid
conversational turn. `Agent::step_inner` now treats three shapes of model
output as *invalid, not a turn*: an empty tool-call array (`[]`),
output starting with `[` that doesn't parse as `Vec<ToolCall>`
(`tools::parse_output`'s `Err` case), or a tool-call array naming a tool
outside the current `tools` set. None of these get an assistant message
appended to `messages`, and none consume the `max_steps` budget (that's
charged once per `step_inner` call, before the retry loop). Instead
`step_inner` regenerates, changing something each attempt so a
deterministic sampler doesn't just reproduce the same junk:

- **Retry 1**: turns schema-constrained decoding on for this attempt only
  (`generate_attempt(force_constrained: true)`), if it wasn't already on
  and `tools` is non-empty. Doesn't touch `self.constrained`.
- **Retry 2**: appends a short generic nudge as a `user` message
  (`RETRY_NOTE`: "Respond with a tool call from the list or a final
  answer.") and regenerates with whatever constrained setting was already
  active. This message *does* become part of history going forward.
- Exhausting `max_retries` (default 2, `Agent::set_max_retries`) gives up
  with `StepOutcome::Error { message: "model produced no valid call", step:
  None }`.

The accepted step's `Step.retries` records how many retries it took (0 if
the first attempt was already valid).

**Tool-error feedback.** `Agent::provide_tool_results` checks each result
with `tools::tool_error_message` — `Some(msg)` for an MCP `isError: true`
result (message pulled from `content[0].text`, generically) or a
top-level `error` field (string, or an object's `message` key), `None`
otherwise. An error result is *not* passed to `format_tool_result`;
instead `tools::format_tool_error(name, msg)` builds a `tool`-role message
whose `content` is the compact JSON string `{"error": "<msg>"}`, distinct
in shape from a successful result so the model can tell them apart at a
glance. A non-error result resets a running `consecutive_tool_errors`
counter to 0; an error result increments it, and once it exceeds
`MAX_CONSECUTIVE_TOOL_ERRORS` (2) — i.e. a third consecutive tool error —
`provide_tool_results` gives up with `StepOutcome::Error` instead of
feeding that error back and generating again. `Step.tool_errors` records
the running consecutive-error count as of that step (0 unless the
immediately preceding tool result was an error).

**Step budget vs. retries.** `max_steps` (`Agent::set_max_steps`) counts
only `step_inner` calls (one per `start`/`provide_tool_results`
invocation) — the malformed-output retries inside a single `step_inner`
call are capped separately by `max_retries` and never increment
`step_index`.

**Schema-diet hook-in.** `Agent::start_from_mcp(utterance, raw_mcp_tools)`
is a new entry point alongside `start` for callers holding raw MCP
`tools/list` entries rather than pre-built `Tool`s: it runs them through
`schemadiet::diet_tools(_, DietLevel::Level1)` before `Tool::from_mcp`
when `Agent`'s diet flag is on (`Agent::set_diet`, on by default), then
calls `start`. `start`/`run` taking pre-built `Tool`s directly (as every
existing test and the byte-fidelity template tests do) are completely
unaffected — the diet only applies on the `start_from_mcp` path.

**Generic repeated-call loop guard, and the forced text-only fallback
(2026-09-11).** A live browser run ("List all my speakers.") called
`get_households_and_groups_and_players({})`, got a result, then called it
again — the repeat guard (added for the six-consecutive-repeats bug above)
turned that into a hard `StepOutcome::Error`, so a plain read-only question
died with no answer instead of ending in text. Two changes:

- **Broader repeat detection.** `is_repeat` used to compare only against
  the *immediately preceding* step's calls (`last_tool_calls`, now
  removed). It now checks a call (by `name`+`arguments`) against every call
  made so far this turn whose result was *not* an error
  (`Agent::successful_calls_this_turn`, populated in
  `provide_tool_results`'s non-error branch) — a repeat several steps back
  is caught just as well as an immediate one. A call identical to one whose
  *earlier* result *was* an error is deliberately excluded from that list,
  so retrying a failed call still executes normally — retrying a failure is
  legitimate, retrying a success is not.
- **Forced text-only answer instead of `Error`.** When `step_inner`'s
  retry loop exhausts `max_retries` on a call that's a repeat, or an empty
  tool-call array (`[]`) — both cases where the model plausibly already has
  what it needs but won't say so — it no longer gives up with `Error`.
  Instead it makes one more `generate_attempt` with
  `force_text_only: true`, which swaps the whole constraint for
  `Grammar::text_only()` (`grammar.rs`: `Pos::Start` rejects a leading `[`
  outright instead of entering the tool-call-array branch, so only free
  text is reachable) and returns whatever text comes back as
  `StepOutcome::Final`. Any other kind of invalid output (unparsable JSON,
  an unknown tool name) still exhausts to `Error` as before — there's no
  similar evidence the model has an answer ready. `Step.repeat_guard`
  records whether a repeat was caught and nudged past on the way to this
  step's outcome (regardless of whether that outcome was a normal model
  answer or the forced one); `Step.forced_text_answer` records whether
  *this* step's text specifically came from the forced branch rather than
  the model's own choice.

**What `web.rs` must mirror** (not done here — that file's worker owns
it): the same three behaviours, adapted to the step-wise browser loop —
(1) an empty/unparsable/unknown-tool model output must not become an
assistant turn; retry with constrained-on then a `user`-role nudge, capped
at 2; then, for a repeated or empty call specifically, force one more
generation under a text-only grammar and return it as `"outcome":"final"`
rather than erroring (any other kind of invalid output still surfaces
`"outcome":"error"` to the page — see the 2026-09-11 addendum above, which
`run_step`/`force_final_answer` mirror exactly); (2) an MCP tool result carrying
`isError: true` (or an `error` field) must be fed back as a `tool` message
with `{"error": "<msg>"}` content, not `format_tool_result`'s normal
shape, and 3 consecutive tool errors should stop the turn with an error
rather than looping; (3) apply `schemadiet::diet_tools(_, DietLevel::Level1)`
to the raw `tools/list` result before the per-tool `Tool::from_mcp` loop,
by default.

**`require_tool_call_first_step` (2026-09-11) — forcing the first
generation of a turn to attempt a tool call.** Observed on a real browser
run (13-tool Sonos page, build `be8ea94`): the utterance "play nirvana on
bloupblip" (`bloupblip` a real speaker not in the fixture set) produced, on
step 0, 114 tokens of prose refusal — "The provided tools do not include a
function to play a specific artist on a Sonos group... it is not possible..."
— with **zero tool calls made**. The id rule (`Grammar::for_tools`: a tool
whose required `*_id`/`*_ids` property has no known value is dropped from
the grammar entirely — see "Schema-constrained decoding" below) correctly
excluded `play_artist` (no ids known yet, nothing harvested). But the
free-text branch was still open at `Pos::Start`, and one tool *was*
callable — `get_households_and_groups_and_players`, which takes no
arguments and would have supplied the ids `play_artist` needed. The model
never called it; it reasoned itself into a refusal instead of looking
anything up.

The fix has two parts:

- **`grammar.rs`: `Grammar::tools_only`.** Mirrors `Grammar::text_only()`'s
  trick in the opposite direction: `Pos::Start` rejects a leading non-`[`
  byte outright, so the free-text branch never opens — every generation
  under this grammar must be a tool-call array. Unlike `text_only()` (which
  drops the tool set to nothing), `tools_only()` is a chainable modifier —
  `Grammar::for_tools(&tools, &id_values).tools_only()` — so it composes
  with whichever id-restriction (`for_tools` or
  `for_tools_unrestricted_ids`, after fail-open) the grammar was already
  built under.
- **`agent.rs`/`web.rs`: `require_tool_call_first_step` (on by default).**
  On the first generation of a turn — `self.calls_made_this_turn` still
  empty, i.e. no tool has been called yet this turn, including across
  malformed-output retries of that same first step — with schema-constrained
  decoding on and at least one tool still callable after the id rule (and
  its fail-open relaxation), `generate_attempt` rebuilds the step's grammar
  with `.tools_only()`. `Step.tools_forced` (`Step.toolsForced` in
  `web.rs`'s JSON) records whether this step's grammar was forced this way.
  Any *later* step of the same turn has `calls_made_this_turn` non-empty (a
  tool call must have happened to reach it via `NeedTools` ->
  `provide_tool_results`), so the condition is naturally false there — the
  turn can still end with a prose `Final` answer once the model has looked
  something up, exactly as before. `force_final_answer`'s
  `Grammar::text_only()` fallback (the repeat-guard / exhausted-retries
  path) is a separate, later-step code path and is unaffected — it still
  needs to produce text.
- **Interaction with fail-open.** If the id rule (even after fail-open
  relaxation) leaves *no* tool callable at all, `require_tool_call_first_step`
  does not force anything — there is no legal tool call to force, and
  forcing one would be an impossible constraint. `generate_attempt` falls
  back to the normal grammar (free-text branch open) in that case, same as
  if the option were off.

**Trade-off.** A chat-shaped question with tools present ("What can you
do?") now costs one extra step: the grammar forces a tool call on step 0
even though the question doesn't need one, so the model calls some tool
first and only answers in prose on the next step, instead of answering
directly. `eval/utterances.json`'s `s18` documents this cost explicitly.
Weighed against the `bloupblip` failure — a tool call that should have
happened never did — this crate takes the trade: silently unhelpful over
one wasted step.

## Tool-schema token diet (`src/schemadiet.rs`)

`agent.rs`/`web.rs` build `Tool`s from raw MCP `tools/list` entries via
`Tool::from_mcp` (see `template.rs`), and those raw entries carry
JSON-Schema fields the model never needs to decide what a valid call is —
measured with the real xLAM-2 tokenizer + chat template
(`crates/llm-wasm/tests/schemadiet.rs`, `system` + `tools` +
`add_generation_prompt: true`, matching how both loops render the first
prompt of a conversation), the 13-tool Sonos preamble
(`fixtures/sonos/tools-12.json`) is **2499 tokens** and the 34-tool one
(`fixtures/sonos/tools.json`) is **8129 tokens** at level 0 (no diet).

`diet_tools(tools: &[Value], level: DietLevel) -> Vec<Value>` runs over
the raw MCP tool objects *before* `Tool::from_mcp` — a pure
`Vec<Value> -> Vec<Value>` transform, generic across MCP servers (nothing
here is keyed on a Sonos tool or property name):

- **Level 1** (safe, structural only, lossless in meaning): drops
  `annotations`, `$schema`, a schema-level `title` that duplicates
  `description`, `additionalProperties: false`, an empty `required: []`,
  `minLength: 1` on a string (the default), and `minimum`/`maximum` when
  set to the i64 extremes (±9007199254740991 — a schema author's
  "effectively unbounded" spelling, not a real bound; `seek`'s
  `delta_millis`/`position_millis` in the Sonos fixture do this). Every
  property name, `required` entry, `enum`, and `type` survives untouched
  — `tests/schemadiet.rs`'s `level1_preserves_property_signatures` checks
  the full (name, required, enum, type) signature set per tool before/after,
  and `level1_dieted_tools_parse_identically_in_grammar` confirms
  `grammar::tools_from_json` (read-only use, owned by another worker)
  parses the dieted list into the same tools/required/enums as the raw
  one. Measured: **2382 tokens** (13 tools, -117) and **7738 tokens**
  (34 tools, -391).
- **Level 2** (level 1 + description trimming, default OFF because the
  dedup step is meaning-affecting): collapses whitespace runs, then drops
  any description sentence that already appeared verbatim in an earlier
  tool's description (first tool keeps it, later tools lose it) — the
  Sonos payload repeats sentences like "Group volume and player volume
  are linked — changing one affects the other." across 8 tools. Sentence
  splitting is abbreviation-aware (won't split "e.g."/"i.e." into bogus
  fragments that could spuriously collide across tools — see
  `split_sentences`'s doc comment). Measured: **2318 tokens** (13 tools,
  -64 more) and **7407 tokens** (34 tools, -331 more).

**Hook-in point** (not wired up here — other workers own `agent.rs`/
`web.rs`): call `diet_tools(&raw_mcp_tools, DietLevel::Level1)` on the
`tools/list` result before the per-tool loop that calls `Tool::from_mcp`
in both loops. Level 2 is available but should be an explicit opt-in
(e.g. a config flag), not a default, since it changes the literal text a
human — or the model — reads per tool.

## Prefix KV images

Build-time-produced files that let any engine instance load a prefix's KV cache in one shot
instead of running prefill token-by-token, per the idea in
`trucs.ai/.claude/worktrees/sonos-mcp/docs/kv-cache-images.md` (read-only reference). Steps 1-2 of
that plan: the file format supports both `f32` and `q8_0` tensor encodings (§"KV quantisation" in
that doc); reading `q8_0` directly in the attention kernels (rather than dequantizing at import)
is still a later step.

### Format v1

`crates/llm-wasm/src/kvimg.rs`, little-endian: magic `KVIMG\0` (6 bytes), `u32` version=1, `u32`
header length, then that many bytes of UTF-8 JSON header, then zero-padding out to the next
16-byte boundary from the start of the file, then tensor data: for `layer in 0..n_layers`, K then
V, each flat `[n_kv_heads, n_tokens, head_dim]` f32 row-major (the cache's batch=1 axis is
dropped — redundant with the flattened layout). `KvImage::layer_slices_at` returns zero-copy `&[f32]`
views when the data region lands 4-byte aligned in memory, and falls back to an owned copy
otherwise (`Cow<[f32]>`).

Header fields: `model_fingerprint` — deliberately **not** a hash of the whole GGUF file (1.7GB+):
`gguf_header_fingerprint` hashes only the GGUF header region (magic through the end of the
tensor-info table — a few KB, contains every tensor's name/shape/dtype, from
`gguf::GgufReader::header_bytes`/`data_section_offset`) plus the file size, cheap enough to
recompute on every native `kv-export` run and every browser `load()` (no on-disk cache, no
`crypto.subtle` pass over the fetched shard bytes needed). `content_fingerprint`'s coarser
`sha256(file_size || first 1MiB || last 1MiB)` remains available for callers without a parsed GGUF
header handy; either is just an opaque string to this format. `prefix_key`
(`sha256(model_fingerprint || rendered_prefix_text)` — hashing the *rendered* prompt, not the
system prompt and tool list separately, means the key is
sensitive to chat-template version, system-prompt wording, and tool ordering all at once, matching
how `web.rs`'s `resident_tokens`/`common_prefix_len` already treat the rendered prompt as the
unit of comparison), `tokens` (the prefix's token ids, for a resident-tokens equality check before
trusting an import — same check `web.rs`'s `run_step` already does for its in-session prefix
cache), `n_layers`/`n_kv_heads`/`head_dim`/`dtype`/`engine`/`created`.

No `sha2` crate: `cargo tree -i sha2` printed nothing, so `kvimg.rs` implements SHA-256 itself
(~80 lines, FIPS 180-4, tested against the standard empty/`"abc"`/pangram vectors) rather than add
a dependency this worker doesn't own `Cargo.toml` to add anyway.

### `dtype: "q8_0"`

Selected via `Header::dtype`; the container stays version 1 (the header already carries the
layout selector) and v1 `f32` images still read unchanged (`v1_f32_images_still_read` in
`tests/kvimg.rs`). Per tensor (`K` or `V`, flattened `[n_kv_heads, n_tokens, head_dim]` row-major,
same as `f32`): values are grouped into blocks of 32 consecutive values along `head_dim`
(`head_dim` must be a multiple of 32 — `head_dim=128` is 4 blocks/row), each block quantized to
one `f32` scale (`absmax(block)/127`) plus 32 `i8` values packed 4-per-`u32` little-endian (so a
future WGSL shader can read a block as 8 `u32` words directly) = 4 + 32 = 36 bytes per 32 values
(1.125 B/value vs 4 B/value for `f32`, ~3.55x smaller). Within one tensor, all block scales are
written first as a contiguous `f32` array, then all blocks' packed words follow as a contiguous
`u32` array — two separately-bindable buffers (scales, packed data), mirroring the Q4 repack
convention for a future kernel. `KvImage::quantize_q8_0`/`dequantize_q8_0` do the block math;
`KvImage::layer_f32` is the dtype-agnostic per-layer accessor (`Vec<f32>`, copied for `f32`,
dequantized for `q8_0`) an importer should use instead of special-casing dtype itself —
`KvCache::import_prefix` already takes `Vec<f32>`, so no `kv.rs` changes were needed for this
step.

Measured on synthetic K/V-shaped data (mostly `|x| < 10`, ~0.5% outliers up to 100, 4096 blocks):
max-abs error per block stayed within the `absmax/127` bound in every block (asserted in
`q8_0_quantize_dequantize_roundtrip_error_bounds`), mean relative error 0.021. Payload size ratio
vs `f32` for the same dims measured at 3.556x (`q8_0_file_roundtrip_write_read`), matching the
36/32-bytes-per-value math above.

The engine still holds KV in `f32` on the GPU — `KvCache` (`kv.rs`) is unchanged by this step, so
a `q8_0` image is dequantized to `f32` at import (via `layer_f32`) rather than read directly by
the attention kernels. Reading `q8_0` KV directly in the attention shaders (avoiding the
dequantize-at-import copy and cutting GPU-resident KV memory, not just download size) is the next
step, per the WGSL packed-int8 kernel plan in `kv-cache-images.md`'s "KV quantisation" section.
llama.cpp reports q8_0 KV perplexity delta in the 0.002-0.05 range
(`github.com/ggml-org/llama.cpp/discussions/20969`), well inside this project's own eval noise
from tool-order/prompt-wording effects.

### `KvCache` API (`kv.rs`)

`export_prefix(&self, n_tokens) -> Vec<(Vec<f32>, Vec<f32>)>`: one `into_data()` readback per
layer per tensor (not per token), narrowed to `[0, n_tokens)`. **Native/build-time only** — this
is a synchronous GPU readback; a WASM caller must not use it directly (deadlocks the browser) and
should add an `into_data_async` variant before calling it from `web.rs`.

`import_prefix(&mut self, layers, n_tokens)`: one `slice_assign` per layer per tensor (bulk write
for all `n_tokens` rows at once), using the same placeholder-swap discipline `write` uses (see
its doc comment / D1 in `docs/BENCHMARKS.md`) so cubecl mutates the existing buffer in place
instead of copying the whole `max_ctx`-sized tensor. Sets `len = n_tokens` directly, so
`snapshot()` reflects the imported prefix with no further calls needed.

#### KV cache storage dtype (Session 13, docs/BENCHMARKS.md)

`KvCache` now holds K/V in one of two storage modes (`KvDtype`), chosen at construction
(`KvCache::new` uses `DEFAULT_KV_DTYPE = KvDtype::F32` — still `F32`, see the correctness note
below; `KvCache::new_with_dtype` picks explicitly):

- `KvDtype::F32` (**default**) — the original design: one `[1, n_kv_heads, max_ctx, head_dim]`
  Burn tensor per layer per K/V. ~906MB at this model's `max_ctx=12288`.
- `KvDtype::Q8_0` (opt-in, not yet the default) — K/V held as raw cubecl GPU buffers, not Burn
  tensors: per layer, `scales: [n_kv_heads, max_ctx, head_dim/32]` f32 and
  `words: [n_kv_heads, max_ctx, head_dim/4]`
  u32 (one absmax/127 scale + 32 packed-i8 values, 4-per-u32, per 32-element block along
  `head_dim`) — the same on-disk convention `kvimg.rs`'s `dtype: "q8_0"` image format already
  used, just resident on GPU instead of round-tripped through a file. ~255MB at `max_ctx=12288`
  (~3.55x smaller than `F32`).

**Correctness status**: every `Q8_0` kernel is unit-tested correct in isolation (quantize/dequant
round trip within its `absmax/127` per-block bound, a bulk `T=31` write matches 31 separate `T=1`
writes bit-for-bit, the fused decode kernel matches an f64 CPU reference to <=2.5e-4 relative
error at `kv_len` in `{31, 2225, 5000}` — all in `tests/kvimg.rs`), but running the real model
end-to-end (`tests/full_forward.rs`) with `Q8_0` as the cache diverges badly: every prefill
position's argmax disagreed with the F32 reference (vs. 8/31 under `F32`), and last-position
top-5 had zero overlap with the reference. Small per-block quantization error, applied to every
K/V value at every layer from token 0, compounds across 36 residual layers into much larger
divergence than the isolated tests predicted. Not yet root-caused to K vs V specifically (the
task brief's suggested next diagnostic — quantize K only, keep V in f32 — isn't implemented yet).
`DEFAULT_KV_DTYPE` stays `F32` until this is resolved; see `docs/BENCHMARKS.md` Session 13.

`write` (renamed from `append` — it no longer returns the read-back prefix tensors, see
`read_or_dequant_f32`/`q8_layer` below) quantizes new rows on the GPU in `Q8_0` mode
(`gguf::kv_quantize_dispatch`, `wgsl/shader_kv_quantize.wgsl`) — no CPU round trip, one dispatch
each for K and V. `read_or_dequant_f32(layer, kv_len)` is the dtype-agnostic read path used by the
prefill (`M>1`) attention branch: a plain `narrow` in `F32` mode, a GPU dequant-range dispatch
(`gguf::kv_dequant_range_dispatch`, `wgsl/shader_kv_dequant_range.wgsl`) into a fresh f32 tensor in
`Q8_0` mode. `q8_layer(layer)` exposes the raw `(k_scales, k_words, v_scales, v_words)` handles
for the decode-time (`M=1`) fused attention kernel (`model.rs`'s `attn_decode_q8`,
`gguf::attn_decode_q8_dispatch`, `wgsl/shader_attn_decode_q8.wgsl`) to bind directly — QK^T,
softmax, and PV all read K/V straight out of the quantized cache, accumulating in f32, with no
dequant-to-f32 step at all. Prefill still takes the old Burn-matmul attention path (dequantizing
the needed K/V range once per layer per forward first) — a fused prefill kernel is a later step.

`export_prefix`/`import_prefix` keep their `Vec<f32>` signatures regardless of storage dtype (so
`kvimg.rs`/`web.rs` need no changes for this): in `Q8_0` mode `export_prefix` dequantizes via
`read_or_dequant_f32` before the readback, and `import_prefix` quantizes the uploaded f32 data via
the same `kv_quantize_dispatch` kernel `write` uses. `export_prefix_q8`/`import_prefix_q8` are the
new no-dequant counterparts (`Q8_0` mode only) that move already-quantized bytes directly — the
path a future `kvimg.rs`/`web.rs` change would use to import a `dtype: "q8_0"` prefix image
without ever materializing f32, per the task brief's step 1. `import_prefix_q8` currently does a
CPU read-modify-write of each buffer (readback the full per-layer buffer, patch rows `[0,
n_tokens)`, re-upload) rather than a partial GPU-side write — correctness first, since this runs
once per session/prefix load, not per token; cubecl's `ComputeClient` has no partial-buffer-write
primitive to target directly (only `create_from_slice`/`empty`).

### Where this plugs in (wired up)

- `bin/llm-agent.rs`'s `kv-export` subcommand (native only — a build step, not a runtime path):
  `--gguf`, `--model-dir`, `--tools <MCP tools/list JSON>` (any file with that shape, not just the
  two fixed fixtures — `eval.rs`'s tools loader is duplicated locally as `load_tools_generic`
  rather than editing that module), `--system`, `--dtype f32|q8_0` (default `q8_0`), `--out-dir`
  (default `~/Code/idle-intelligence/models/kv/`), `--tool-order listing-first|as-is`. Renders the
  prefix as the common leading tokens between two content-free probe utterances (`"kv-export-
  probe-alpha"` / `"totally-different-probe-beta"` — chosen with **no shared leading text**, since
  an earlier version shared `"kv-export-probe-"` and the token-level common prefix overran into
  probe-specific tokens; **must** match `web.rs`'s `render_kv_prefix_tokens` verbatim, or the two
  sides compute different `prefix_key`s for the same actual prefix), prefills
  it via `forward_hidden`, calls `KvCache::export_prefix`, and writes `<out-dir>/<prefix_key>.kvimg`
  + a `<prefix_key>.json` sidecar (the header, pretty-printed). `model_fingerprint` is computed
  from `gguf::GgufReader::header_bytes` (a few KB, before `load_deferred` consumes the loader) +
  file size via `kvimg::gguf_header_fingerprint` — no on-disk cache needed, since it never reads
  the full 1.7GB GGUF (an earlier version streamed a full-file SHA-256 into
  `<out-dir>/model-hashes.json`; dropped per the coordinator's "do NOT hash the full GGUF" note).
- `web.rs`: `LlmEngine::load()` computes `model_fingerprint` itself, inside the same braced block
  that opens the loader and calls `load_deferred` — `Q4ModelLoader::reader()`/`reader_mut()` expose
  `file_len()`/`header_bytes()` on the wasm-side `GgufReader<ShardedCursor>` exactly as they do
  natively, so this needs no bytes beyond what `appendModelShard` already staged and no JS-side
  pass over the shard buffers. `LlmEngine.prefixKey(tools_json, system)` renders the prefix the
  same way `kv-export` does and returns `sha256(model_fingerprint || prefix_text)`.
  `LlmEngine.importKvImage(bytes, tools_json, system)` validates `header.model_fingerprint`,
  `header.tokens` (exact match against the freshly-rendered prefix — same discipline `run_step`'s
  `effective_prefix` check already applies to `resident_tokens`), and `header.prefix_key`/shape
  before calling `KvCache::import_prefix` and setting `resident_tokens`; returns `false` (not an
  error) on any mismatch so the caller falls back to normal prefill. Synchronous — `import_prefix`
  only writes (`from_data`/`slice_assign`), no GPU readback needed.
  `LlmEngine.exportKvImage(tools_json, system)` is the miss-path counterpart: async (uses the new
  `KvCache::export_prefix_async`, `into_data_async`-based), checks `resident_tokens` already covers
  the rendered prefix, and returns a `.kvimg` byte buffer for the caller to persist.
- `worker.js`: no longer computes anything model-identity-related itself — `model_fingerprint` is
  entirely `web.rs`'s concern now (see above), so `load()` takes no extra param beyond tokenizer/
  template JSON + the progress callback. On each `run`, before `start()`: `prefixKey()`, then check
  OPFS (`navigator.storage.getDirectory()`)
  and then `fetch(<modelBase>/kv/<key>.kvimg, {cache:'no-store'})` (404 is a normal miss, not an
  error) for a matching image; on a hit, `importKvImage()` and (if it came from the network) save a
  copy to OPFS for next time; on a miss, `start()` runs its normal full prefill and the worker
  fire-and-forgets an `exportKvImage()` + OPFS save afterward. `<modelBase>/kv/` is derived from the
  GGUF shard URL by substituting its `/gguf/` path segment (`deriveKvBaseUrl`) — matches
  `scripts/serve_models.py` serving the whole `models/` tree, `kv/` a sibling of `gguf/`/`hf/`.
  `'status'` messages with `phase: 'kv-image'` and a human `note` report the outcome;
  `docs/ENGINE.md`'s worker protocol comment (top of `worker.js`) documents the message shapes.

Debug-only, not part of the KV-image feature itself: `LlmEngine.setPrefillKernel("naive"|"pinned")`
(`web.rs`) / `gguf::set_force_naive_kernel` toggle production `ForceKernel::Auto` routing (which
takes the pinned scratch-dequant + Burn matmul path at prefill's M>=32 shapes) vs forcing the naive
per-element kernel for every prefill matmul — a numerical-divergence A/B bisection aid for browser
runs that produce a different first tool call than native at the same tokens, added alongside this
work at the coordinator's request; not used by any production path. `step.promptTokenIds` (added to
`run_step`'s JSON output) and a per-step top-5 `(id, logit)` console log serve the same
bisection — see `scripts/headless/run.mjs --tokens-out` for extracting a step's prompt token ids
into a file for `llm-agent run --tokens <file>` native comparison.

### Sizes

xLAM-2-3b-fc-r (36 layers, 2 kv heads, head_dim 128), `f32` vs `q8_0` (measured 3.556x ratio, per
above):

| tools-set | seq_len | f32 | q8_0 |
|---|---|---|---|
| 13 tools | ~2300 tok | 170 MB | ~48 MB |
| 34 tools | 8140 tok | 600 MB | ~170 MB |

`f32` is exact; `q8_0` includes the per-32-block `f32` scale overhead (hence not exactly 1/4 of
`f32`). At 13 tools `q8_0` (~48 MB) is a comfortable one-time download; at 34 tools, ~170 MB is
still the size that makes a cold-connection download plausible where 600 MB `f32` was not — see
`kv-cache-images.md`'s "KV quantisation" section for the full rationale (int8 chosen over f16 to
avoid the `shader-f16` WGSL extension, which has known gaps on Firefox/Linux/NVIDIA and Qualcomm).
