# lean engine: KV snapshot/restore + per-step logit mask

Two engine features for the Sonos MCP agent and llm-life consumers (see
`~/Code/tracker/projects/llm-web/consumers-2026-09-28.md`, gap list items 1
and 2): resident-prefix KV snapshot/restore/export/import, and a per-step
GPU logit mask for constrained decoding. Crate `crates/lean`. Native
reference and gate: `lean-cli`, `cargo test -p lean --release -- --ignored
fixture_parity`.

## Parameters

- Model: Qwen2.5-0.5B-Instruct, GGUF `qwen2.5-0.5b-instruct-q4_0.gguf`
  (Q4_0 weights, Q8_0 `output.weight`), tokenizer from
  `Qwen/Qwen2.5-0.5B-Instruct`.
- Config (from GGUF metadata): 24 layers, hidden 896, 14 heads, 2 KV heads,
  head_dim 64, intermediate 4864, vocab 151936, rope_theta 1e6,
  rms_norm_eps 1e-6.
- Fixture: `crates/lean/reference/fixture.json` (`short` 36 tokens, `long`
  86, `non_english` 54).
- Kernel path: `fast_kernels = true` on both native and browser.
- Native: macOS 26.3.1, Metal backend via wgpu 26, `LEAN_GGUF`/
  `LEAN_TOKENIZER_DIR` pointed at
  `~/Code/idle-intelligence/models/{gguf/Qwen2.5-0.5B-Instruct-GGUF,hf/Qwen2.5-0.5B-Instruct}`.
- Browser: Playwright-bundled Chromium (headless), `--enable-unsafe-webgpu
  --enable-features=Vulkan,WebGPU --use-angle=metal`, model served locally
  (`?local=1`) from `python3 -m http.server` rooted at `crates/lean/`.
- wasm build: `wasm-pack build crates/lean --target web
  --no-default-features --features web`. Harness engine build tag bumped
  to `2026-09-28-5` in `crates/lean/www/main.js`.
  `lean_bg.wasm` sha256 `6d8589d4d993c7e98d0c7b521ac45cabd2f9acf6c966c27dbf68bf2824e11d19`
  (3,335,832 bytes), `lean.js` sha256
  `e248888200faaab33b6d889090e498cb02339692937428e7d2ce80d24f70eaec`.
- One GPU job at a time: native `cargo test`/`lean-cli` runs and the
  headless browser run were never run concurrently.

## API

Rust (`crates/lean/src/model.rs`, all pre-existing forward functions gained
a trailing `mask: Option<&wgpu::Buffer>` parameter):

```rust
pub fn build_mask_bitset(vocab: usize, allowed: &[u32]) -> Vec<u32>;

pub struct KvSnapshot { pub kv_len: u32, pub num_kv_heads: u32, pub head_dim: u32, pub num_layers: u32, pub k: Vec<Vec<f32>>, pub v: Vec<Vec<f32>> }
impl KvSnapshot {
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>;
}
impl KvCache {
    pub async fn snapshot(&self, engine: &Engine) -> KvSnapshot;
    pub fn restore(&mut self, engine: &Engine, snapshot: &KvSnapshot);
}

pub async fn forward_prefill(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32>;
pub async fn forward_prefill_suffix(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32>;
pub async fn forward_decode_step(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> Vec<f32>;
pub async fn forward_decode_step_argmax(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer, mask: Option<&wgpu::Buffer>) -> u32;
```

JS (`crates/lean/src/web.rs`, `LeanEngine`, all async where GPU readback is
involved):

```js
engine.tokenize(prompt) -> Uint32Array          // chat-template render + encode
engine.encodeRaw(text) -> Uint32Array           // raw encode, no template
engine.decodeIds(ids) -> string
engine.buildMaskBitset(allowedIds) -> Uint32Array
await engine.prefillTokens(tokenIds, maskBits) -> Float32Array   // fresh cache, kv_len=0
await engine.appendTokens(tokenIds, maskBits) -> Float32Array    // onto existing cache
await engine.decodeStepArgmax(tokenId, maskBits) -> number
engine.kvLen() -> number
await engine.snapshotKv() -> Uint8Array
engine.restoreKv(bytes)                          // queued writes only, not async
```

`maskBits`/`mask_bits` is `[]` for "no mask" (wasm-bindgen has no
`Option<Vec<u32>>` ergonomics worth the complexity here); a non-empty
bitset is `build_mask_bitset`/`buildMaskBitset`'s packed format.

## Results

Native, `cargo test -p lean --release -- --ignored <name> --test-threads=1`:

| test | result | time |
|---|---|---:|
| `fixture_parity_both_kernel_paths` | ok | 24.50s |
| `kv_snapshot_restore_matches_full_prefill_and_round_trips` | ok | 11.70s |
| `kv_snapshot_timing_1000_token_prefix` | ok | 31.11s |
| `singleton_mask_forces_exact_string` | ok | (in 18.49s combined) |
| `all_allowed_mask_matches_unmasked` | ok | (in 18.49s combined) |
| `mask_upload_per_step_cost` | ok | (in 18.49s combined) |

Native KV snapshot/restore (86-token `long` fixture case, split 43/43,
12-token greedy continuation):

| path | tokens |
|---|---|
| prefill(prefix+suffix) -> generate | `[39814,...]` (matches below) |
| restore + prefill(suffix) -> generate | identical, byte-for-byte export/import round trip identical |

`export_bytes=1056784` for this 43-token snapshot (24 layers x k+v x
2 kv_heads x 43 x 64 x 4 bytes + 16-byte header).

Native KV snapshot timing (synthetic prompt, 1235 tokens):

| metric | value |
|---|---:|
| full prefill (from scratch) | 30522.24 ms |
| snapshot (readback) | 97.65 ms |
| export (`to_bytes`) | 2.84 ms |
| import (`from_bytes`) | 2.60 ms |
| restore (`queue.write_buffer`) | 5.36 ms |
| round trip (snapshot+export+import+restore) | 108.44 ms |
| speedup | 281.5x |

Native mask upload cost (24-step decode loop, vocab 151936, all-allowed
mask vs no mask):

| path | ms/step |
|---|---:|
| unmasked `forward_decode_step_argmax` | 51.591 |
| masked (fresh bitset uploaded every step) | 51.572 |
| overhead | -0.019 (within noise) |

Native `lean-cli --check kv-snapshot` / `--check mask` (19/19-token split,
default prompt): both `PASS`.

Browser (headless Chromium, WebGPU, `?local=1`, engine build
`2026-09-28-5`), full harness (`window.__leanResult.allMatch = true`):

| case | prompt tokens | tokens match | prefill ms | decode ms/tok |
|---|---:|:---:|---:|---:|
| short | 36 | true | 390.3 | 26.4 |
| long | 86 | true | 456.0 | 26.9 |
| non_english | 54 | true | 292.8 | 26.0 |
| long_1000 (timing only) | 1263 | n/a | 4500.8 | 38.7 |

Browser KV snapshot/restore (86-token `long` case, split 43/43, 12-token
greedy continuation): `restore_bytes_match=true`, `tokens_match=true`,
`tokensA=tokensB=[39814,11,1588,525,279,1156,36655,5424,315,279,38345,1965]`.

Browser KV snapshot timing (synthetic prompt, 1263 tokens):

| metric | value |
|---|---:|
| full prefill (from scratch) | 8199.3 ms |
| snapshot | 344.0 ms |
| restore | 26.3 ms |
| round trip | 370.3 ms |
| speedup | 22.1x |

Browser per-step logit mask: `target_match=true`,
`forced_text="{\"ok\":true}"`, `mask_ms_per_step=29.88` (normal masked
decode-step cost, not isolated overhead - see native table above for the
overhead-vs-baseline split).

Build/lint gates: `cargo check -p lean --release` clean, `cargo clippy -p
lean --release --bins --tests -- -D warnings` clean (native), `cargo
check`/`clippy -p lean --release --lib --target wasm32-unknown-unknown
--no-default-features --features web -- -D warnings` clean, `wasm-pack
build crates/lean --target web --no-default-features --features web`
succeeds.

## Observations

- The 1235/1263-token KV-snapshot-timing prefill is timing-only, not
  correctness-checked, for the same pre-existing reason the wasm harness's
  `long_1000` fixture case already is: `attn_prefill.wgsl`'s fixed
  `MAX_SEQ = 256` per-invocation scores array, combined with its
  `@workgroup_size(256)` dispatch, means only query rows 0..255 of a
  from-scratch prefill actually get an attention output computed at
  seq > 256 - a pre-existing limit of this engine, not something this
  session's two features touch or fix.
- An earlier version of `forward_prefill_suffix` batched every suffix
  token's `decode_layers` call into one encoder before a single submit, to
  save dispatch overhead. That hung indefinitely on native Metal with 0%
  CPU (no panic, no error) - isolated with a `--test-threads=1
  --nocapture` run plus explicit `eprintln!` + `stderr().flush()` markers
  after every phase, which showed it never returned from
  `forward_prefill_suffix()`. Root cause: `Pool`'s per-call-site dims
  buffers are updated via `queue.write_buffer`, a queue-timeline operation
  independent of encoder-recording order - writing new `pos_base`/`kv_len`
  values for the same cached key multiple times before a single `submit()`
  clobbers all but the last write, so every dispatch in that one encoder
  ended up reading the *last* suffix token's dims. Fixed by giving each
  suffix token its own encoder/submit (still only reading back the final
  token's logits) - see `model.rs::forward_prefill_suffix`'s doc comment.
  This is now the second time this session touched `Pool`'s "external/
  queue-timing write before submit" hazard class (the first, pre-existing
  one is the mask-buffer bind-group cache staleness fixed by using
  `engine.bind_group` instead of `pool.bind_group` in `mask_logits_gpu`).
- Mask upload/apply cost is not measurable above noise on either target:
  native overhead is negative (-0.019 ms/step, i.e. within run-to-run
  jitter) and the browser's masked-step cost (29.88 ms) is in the same
  range as its unmasked decode steps elsewhere in this run (26-27 ms) once
  prefill/decode context differs are accounted for. The mask kernel is one
  `vocab/256`-workgroup dispatch (~594 workgroups here) added to a step
  that already issues ~15 dispatches per layer x 24 layers.
- Browser full-prefill wall time for the ~1250-token synthetic prompt
  (4500.8-8199.3 ms across two runs) was noisier than the native number
  (30522.24 ms, single run) but consistently much lower - both runs are
  exercising the same known-incomplete `attn_prefill` path at this length,
  so neither number should be read as "prefill throughput" for a correct
  1250-token forward pass, only as the baseline this session's speedup
  ratio is measured against.
- KV snapshot byte size matches the documented format exactly: `16 +
  num_layers * 2 * num_kv_heads * kv_len * head_dim * 4` bytes (e.g.
  43-token snapshot: `16 + 24*2*2*43*64*4 = 1,056,784`, matching
  `export_bytes` above on both native and the CLI check).

## Commits

On branch `lean-engine`:

- `2989e9e` lean: KV snapshot/restore + per-step logit mask in the core forward pass
- `535de3c` lean: --check kv-snapshot/mask native smoke checks in lean-cli
- `7afd1e7` lean: expose kv snapshot/restore and logit mask in the wasm API
- `beae21b` lean: native parity/timing tests for kv snapshot and logit mask
- `e404667` lean: browser harness checks for kv snapshot and logit mask, bump build tag
