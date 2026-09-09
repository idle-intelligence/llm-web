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

## 9. Gap list for Qwen2.5-3B (xLAM-2-3b-fc-r)

| Gap | Size | Notes |
|---|---|---|
| GGUF metadata parsing (currently skipped entirely, §1) | small | Only needed if config should come from the file instead of hardcoded, as today |
| Qwen2 GGUF tensor-name table (new loader, §1) | small–medium | Mechanical rewrite of `load_transformer_layer`/`load_q4_linear` name strings |
| q/k/v bias loading (struct support exists, loader doesn't populate, §1/§4) | small | Add a `*.bias` tensor read alongside each Q4 linear load |
| RoPE convention mismatch: interleaved-pair vs. Llama/Qwen2 rotate-half (§3) | medium | Silent-wrong-output risk if not fixed; either permute weight rows on load or rewrite `apply_rotation` |
| KV cache: single-token-only → multi-token prefill (§3) | large | New `KVCache::update` path for M>1 writes; new batched embedding path in `model.rs`; kernel itself (§2) already supports M>1 |
| KV cache size/growth: 751-step fixed ring buffer → thousands-of-tokens context for MCP conversations (§3) | medium–large | Either a much bigger fixed buffer (~600 MB budget estimate in §3) or a paged/growable design; current wraparound silently corrupts state past capacity |
| Sliding-window mask always-on → make optional/full-causal (§3) | small | `Q4Attention::new(..., sliding_window: None)` for Qwen2's full-context attention |
| lm_head at 151936×2048: single 174 MB Q4 buffer, untested against actual `maxStorageBufferBindingSize` (§2) | medium | Requesting adapter's full limits (already done, §2) likely covers it on M2/Metal, but unverified; may need to shard the head matmul across dispatches if not |
| Tied embeddings/lm_head (currently two independent tensors, §4) | small–medium | Simplification opportunity (share one buffer) but needs new plumbing since `EmbeddingStore` (CPU row-dequant) and `Q4Tensor`(GPU matmul weight) are different representations today |
| Byte-level BPE tokenizer, 151k vocab, ~7 MB tokenizer.json, encode+decode (§6) | large | Nothing here transfers — current tokenizer is decode-only, 8001-vocab SentencePiece; likely lands in JS per this repo's own native-vs-WASM precedent |
| Worker/wasm-bindgen protocol: audio-frame-in/text-out → prompt-in/token-stream-out (§5) | small–medium | Message shapes and `SttEngine` surface both need new methods; `&mut self` single-flight queuing pattern (worker.js) carries over directly |
| MCP tool-call sampling/formatting (grammar-constrained or JSON-mode decoding for tool calls) | large | No sampling logic beyond argmax exists in this crate (`stream.rs` only ever reads GPU argmax) — greedy-only; tool-call-structured decoding is new work entirely |
| PyTorch-logits reference/comparison harness (§7) | medium | No numeric-tolerance-vs-PyTorch test exists to copy; would follow `e2e_pytorch_mimi.rs`'s "dump JSON offline, compare in Rust test" pattern but needs building from scratch |
| Prefill matmul performance (naive kernel is bandwidth-bound, no M-dimension weight reuse, §2) | medium | Works correctness-wise at M>1 today; a multi-hundred-token MCP prompt prefill would be slow without a tiled/cooperative kernel (exists in `refs/voxtral-mini-realtime-rs` but not ported into this crate) |

**Effort estimate (Phase 1b — decoder + kernels + KV + sampling on this engine): roughly
15-25 worker-days**, dominated by three items: (1) multi-token KV cache/prefill plumbing (KV
cache rewrite + batched embedding path + wiring the already-general Q4 kernel through it — this
touches the most files and needs careful correctness verification against a fresh PyTorch
reference, since there's no existing numeric-tolerance harness to build on, §7); (2) the RoPE
convention mismatch, which is cheap to *fix* but expensive to *catch* if missed (wrong numbers,
not a crash — needs the PyTorch reference harness from day one, not bolted on after); and (3) BPE
tokenization for 151k vocab, which is a real subproject (encode+decode, likely JS-side, plus a new
worker protocol) rather than a config change. Secondary risks worth calling out explicitly: the
174 MB single-buffer lm_head is unverified against real WebGPU `maxStorageBufferBindingSize` on
M2/Chrome (§2 — could force a head-matmul sharding fallback, a few extra days if so); and prefill
throughput on the naive (untiled) kernel is unmeasured and could make MCP tool-schema prompts
(hundreds to low-thousands of tokens) slow to first-token even after correctness is achieved,
since only a matvec-shaped (M=1) workload has ever actually been benchmarked in this codebase
(§8) — porting the tiled kernel referenced in `refs/voxtral-mini-realtime-rs` is the natural
mitigation and should be budgeted in rather than treated as a stretch goal.
