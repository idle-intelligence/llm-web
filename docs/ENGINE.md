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
  "systemPrompt"?}` (all optional). Returns a JSON **string** (not `serde-wasm-bindgen`, to avoid
  adding that dependency for a shape this simple):
  `{"outcome":"needTools","calls":[{"call_id","name","arguments"}...],"step":{"promptTokens","text","prefillMs","decodeMs","tokens"}}`,
  `{"outcome":"final","text":...,"step":{...}}`, or `{"outcome":"error","message":...}`.
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

Same idea as `agent.rs`'s (see its module docs): `LlmEngine` renders the actual utterance plus a
throwaway probe utterance under the same `tools` set, finds their common leading token run, and
caches that length keyed by the `tools` list (`Tool` derives `PartialEq`, so a 12-34-entry
`Vec<Tool>` comparison is cheap). Before prefilling, it checks the cached length against
`resident_tokens` (the exact token sequence currently written into the `KvCache`, tracked
position-for-position in `web.rs`) — only trusting the cache when the resident prefix's tokens
actually match the new prompt's leading tokens, falling back to a full prefill
(`effective_prefix = 0`) otherwise. Measured on the real tokenizer/template with the 12-tool
Sonos fixture (`fixtures/sonos/tools-12.json`) and system prompt `"You are a helpful home
assistant with access to Sonos speaker controls."` (see
`crates/llm-wasm/tests/agent.rs::prefix_is_stable_across_utterances_for_same_tools`):

```
utterance A ("pause the kitchen"):                2225 tokens
utterance B ("what's playing in the living room"): 2229 tokens
common prefix:                                     2218 tokens
```

i.e. ~99.7% of the rendered prompt for two different utterances under the same tools is the
constant system+tools preamble — prefilling that once per tools-set change instead of per turn is
the entire point of this cache.

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
