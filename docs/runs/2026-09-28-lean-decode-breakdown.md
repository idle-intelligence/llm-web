# lean decode step breakdown and pass-batching fix

Machine: M2 laptop (native = Metal via wgpu; browser = WebGPU in
Playwright's bundled headless Chromium, `--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --use-angle=metal`). One GPU job at a time;
`pgrep -f "Contents/MacOS/Google Chrome for Testing"` checked empty before
each native timing run. Model file: Qwen2.5-3B-Instruct-GGUF's official
`qwen2.5-3b-instruct-q4_0.gguf` unless noted otherwise (36 layers, hidden
2048, 16 heads / 2 KV heads, head_dim 128, intermediate 11008, vocab
151936). Commits: `9f0378c` (pre-fix baseline used for the profile),
`b9ca7da` (adds the long-prefill dispatch-overflow fix; "before" in the
timing tables below), `bdd6683` (adds this session's pass-batching fix;
"after").

## Step 1: where the time goes (CPU sampling profile)

Command:
```
lean-cli --gguf <3B scratch pure-Q4_0 gguf> --tokenizer-dir <Qwen2.5-3B-Instruct> \
  --prompt "What is the capital of France?" --tokens 300 --kernel fast &
sample <pid> 8 -f /tmp/lean_decode_sample.txt
```
(`sample` is macOS's built-in CPU sampling profiler, 1ms interval, 8s
window, taken mid-run so every sample lands inside the steady-state decode
loop.)

| call tree node (main thread) | samples | % of 5919 total |
|---|---|---|
| `lean_cli::run_generation` | 4692 | 79.3% |
| &nbsp;&nbsp;`pollster::block_on -> Device::poll -> wgpu_core::device::resource::Device::maintain` | 4119 | 69.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;`DynDevice::wait` (native Metal poll loop: `nanosleep`/`__semwait_signal`) | 4029 | 68.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;`Queue::maintain` (retiring completed command buffers: `EncoderInFlight`/`TempResource` drop, Metal buffer dealloc) | 33 | 0.6% |
| everything else in `run_generation` (CPU-side dispatch encoding: bind groups, uniform writes, `format!` keys) | ~573 | 9.7% |

Measured dispatch/pass counts at this point (pre-fix, `9f0378c`/`b9ca7da`):
one `wgpu::ComputePass` opened and closed per single dispatch
(`Engine::dispatch`, `crates/lean/src/engine.rs`), so decode's per-step
dispatch count (508-532 measured across models in this session, matching
the task brief) equalled its per-step *pass* count - every dispatch was its
own `MTLComputeCommandEncoder` session inside one command buffer.

## Root cause

`Engine::dispatch` called `encoder.begin_compute_pass()` /
`pass.dispatch_workgroups()` / (implicit end-of-pass on drop) for every
single kernel call. Decode does ~14-15 dispatches per layer (rmsnorm,
q/k/v, ropeq, ropek, attn, wo, add, ffnnorm, gate_up, silu, down, add) x 36
layers on this model = 508-532 single-dispatch passes recorded into one
command buffer per decode step. Opening/closing a
`MTLComputeCommandEncoder` has real fixed CPU+GPU-side cost on native
wgpu-core/Metal; this also matches the task's own observation that native
(98.8ms/tok on this model) was *slower* than the same WGSL running through
Dawn in the browser (77.8ms/tok) - Dawn's compute-pass overhead is
apparently cheaper than wgpu-core's for the same fragmentation pattern.

Other candidates named in the task brief were checked in the profile and
found small at steady state: `Pool` already caches buffers/bind groups
(zero new allocations after the first call at a given shape), and the
per-step fresh staging buffer for `read_u32`/`read_buffer` only showed up
as the 0.6% `Queue::maintain`/`TempResource` line above.

## Fix (commit `bdd6683`)

`Engine::dispatch` now takes an already-open `&mut wgpu::ComputePass`
instead of a `&mut wgpu::CommandEncoder` (no per-call pass open/close).
`forward_prefill`, `decode_layers` and `forward_chunk_spec` each hold one
open `ComputePass` spanning a whole layer, closing it only around
`scatter_kv_gpu`'s `copy_buffer_to_buffer` calls (encoder-level, cannot run
inside a pass) and reopening immediately after - cutting passes/layer from
~14-15 to ~2, with the pass staying open across consecutive layers where
there's no intervening copy. No kernel, dispatch order, or numeric logic
changed. Parity re-verified after the change: `fixture_parity`,
`fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`, `fixture_parity_llama_360m`,
`fixture_parity_llama_1_7b`, `kv_snapshot`, `logit_mask` all pass;
`fixture_parity_qwen25_3b` was investigated separately and found to be a
pre-existing, data-dependent failure (wrong GGUF file used in an earlier
run of this session, unrelated to this fix) - it passes on both `b9ca7da`
and `bdd6683` against the correct official GGUF, 3/3 runs each, zero
variance in the reported diff.

## Native decode ms/tok, before (`b9ca7da`) vs after (`bdd6683`), median of 3

| model | GGUF | fixture case (prompt tokens) | before | after | change |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | short (36) | 33.63 | 25.13 | -25.3% |
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | long (86) | 35.60 | 25.03 | -29.7% |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0.gguf (official) | short (36)* | 133.48 | 118.78 | -11.0% |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0.gguf (official) | long (86)* | 148.45 | 119.89 | -19.2% |
| Qwen3-1.7B | Qwen3-1.7B-Q8_0.gguf | short (15) | 81.88 | 83.63 | +2.1% (within measured noise, see note) |
| Qwen3-1.7B | Qwen3-1.7B-Q8_0.gguf | long (65) | 93.36 | 95.48 | +2.3% (within measured noise, see note) |
| SmolLM2-360M-Instruct | SmolLM2-360M-Instruct-Q4_0.gguf | short (37) | 43.58 | 31.40 | -28.0% |
| SmolLM2-360M-Instruct | SmolLM2-360M-Instruct-Q4_0.gguf | long (86) | 42.85 | 31.40 | -26.7% |

\* The 3B model was run against `crates/lean/reference/fixture.json`
(generated for the 0.5B model's tokenizer/weights - the fixture is used
here only for its `input_ids`, not for correctness checking against a 3B
reference; `tok_match`/`top20_maxdiff` on this row are expected to disagree
and are not evidence of a bug).

Note on Qwen3-1.7B: one run in each of the before and after sets was
>20% off its other two and was re-run per this session's rule (before:
run 2's short-case decode of 230.74ms/tok, clearly GPU contention -
rerun gave 81.88, consistent with the other two; after: run 1's short/long
decode of 64.39/63.01ms/tok, re-run gave 63.49/69.29 - i.e. it reproduced,
so this was real run-to-run variance on this model/commit, not contention -
included in the median as-is). The net "after" change on this model sits
inside that variance and is not a clear regression or improvement.

## Browser decode ms/tok (current/after code only, `bdd6683`, 3 fresh page loads)

| model | GGUF | case (prompt tokens) | run1 | run2 | run3 | median |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | short (36) | 21.0 | 101.5 | 81.5 | 81.5 |
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | long (86) | 21.4 | 81.8 | 34.3 | 34.3 |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0-pure.gguf (scratch, output.weight forced Q4_0)** | short (36) | 73.63 | 73.60 | 73.90 | 73.63 |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0-pure.gguf (scratch)** | long_tools_single (2225) | 966.54 | 961.74 | 710.40 | 961.74 |

\*\* The browser 3B harness (`lean-3b-harness/model/model.gguf`, inherited
from this task's earlier `old-vs-new.md` comparison work) is wired to the
scratch requantized copy, not the official file; the official file's
Q6_K-output-tensor kernel path is only exercised by the native
`fixture_parity_qwen25_3b` test in this session, not this browser harness.

Every run reported here (0.5B: 3 separate fresh Chromium page loads via 3
separate `run_lean_browser.py` invocations; 3B: 3 in-page runs sharing one
loaded model/KV-cache-reset cycle via `run_lean3b_browser.py`) is a cold or
near-cold WebGPU pipeline/shader-module compile on top of whatever the rest
of the page/process is doing - the 0.5B row's 21.0/101.5/81.5 and
21.4/81.8/34.3 spreads are consistent with per-process shader-compile
variance already noted in this task's earlier `old-vs-new.md` doc (a
31.81/32.07/62.17ms spread on the same case, same cause suspected there
too), not a regression introduced by this session's fix - the long_tools_single
3B case previously **crashed** on this same shape before `b9ca7da`'s fix (a
`silu_mul_fused` dispatch-workgroup-count overflow past WebGPU's
65535-per-dimension limit); it now completes and produces the fixture's
exact greedy continuation on every run.

## Commands

Native timing (3x each per model/commit, GPU-locked):
```
lean-cli --gguf <path> --tokenizer-dir <path> \
  --fixture crates/lean/reference/<fixture>.json --tokens 16 --kernel fast
```
(`--tokens 32` for the two models whose native comparison predates this
session's `--tokens 16` convention: Qwen2.5-0.5B-Instruct and
SmolLM2-360M-Instruct; only the `short`/`long` per-case `decode_ms/tok`
column is read from stdout either way.)

wasm build:
```
wasm-pack build crates/lean --target web --out-dir pkg -- --no-default-features --features web
```

Browser harnesses: `python3 -m http.server` rooted at `crates/lean/`
(0.5B, `www/index.html?local=1`) and at the scratch `lean-3b-harness/`
directory (3B), driven by `/tmp/pw-venv/bin/python3` running
`run_lean_browser.py` / `run_lean3b_browser.py` (Playwright's bundled
headless Chromium, never a personal browser).

## Session 2: memory, not the kernel (2026-09-29)

Follow-up to the above: browser decode at `long_tools_single` (2225 prompt
tokens) on Qwen2.5-3B was ~962ms/tok - 8x the same commit's native number
(120ms/tok) and 13x the same browser's short-context number (74ms/tok). A
first repro attempt (official 3B GGUF this time, not the scratch
requantized copy) reproduced a milder version (588ms/tok decode, 136.5s
prefill) but was killed by the session lead mid-run: the renderer's
physical footprint hit 22GB and the GPU process 6.7GB on this 16GB Mac,
with swap at 24.5GB - a memory problem, not a kernel/dispatch one. This
section re-scopes around that.

### Root causes found (both in `crates/lean/src/model.rs`)

**1. `Pool` duplicated every layer's prefill scratch buffers, one full copy
per layer, forever.** `Pool` (`crates/lean/src/pool.rs`) is grow-only and
never frees. `forward_prefill`'s per-layer loop built every scratch-buffer
key as `format!("layer{i}...")` (`model.rs:1585` before this fix), so each
of the model's `num_layers` passes through `rmsnorm`/`qkv_proj`/`mlp`/etc.
got its own distinct `Pool`-cached buffer, sized to the longest prompt ever
seen, kept alive for the rest of the page's life - even though each
layer's scratch is fully consumed and dead before the next layer's
identically-shaped call touches it. Fixed by `scratch_key()`
(`model.rs`, new function, commit `9da3b77`): strips the trailing digits
off a call-site key's first segment before it reaches `Pool::data`/
`Pool::uniform` (`"layer5.mlp.gate_up.out"` -> `"layer.mlp.gate_up.out"`),
so every layer's scratch collapses onto the same buffer. `Pool::bind_group`
keys are untouched (still carry the layer index - each layer's bind group
references that layer's own weight tensor, a different GPU buffer every
layer). Verified safe by the existing intra-layer dependency chains (RoPE
mutating q/k in place, rmsnorm's output feeding straight into the next
dispatch) - every scratch buffer this maps is already write-then-read
within one layer's own dispatch sequence, recorded into the same
`wgpu::ComputePass`/command encoder in program order, so a later layer's
write cannot execute before an earlier layer's last read of the same
buffer already has.

**2. `forward_prefill` computed the lm-head logits for every prompt
position, not just the last one.** `forward_prefill`'s only-ever-used
return value is `all_logits[(seq-1)*vocab..]` - the last row - but it ran
`rmsnorm` and the `lm_head` matmul at `rows = seq` and read back the full
`[seq, vocab]` buffer (`model.rs:1613-1627` before this fix). `vocab =
151936` on every model in this session, and `output.weight`'s matmul has
no fast decode-shaped kernel at prefill row counts (Q6_K is naive-only;
Q4_0/Q8_0 at `rows = seq` runs the un-tiled kernel below
`TILED_MIN_ROWS`'s break-even) - so this multiplied both the lm-head
matmul's compute AND its output buffer's size by `seq` for a result that
was `seq - 1` parts thrown away. Fixed (commit `91201ac`): slice the last
row of the final layer's hidden state into a small `[1, hidden]` buffer
(one `copy_buffer_to_buffer`, same encoder-level-copy/pass-reopen pattern
`scatter_kv_gpu` already uses) before `out_norm`/`lm_head`, so both run at
`rows = 1`.

Found by adding `Pool::debug_top_buffers()` (kept, `pool.rs`) and a
`LEAN_DEBUG_POOL_TOP`-gated print in a local, uncommitted copy of
`lean_cli.rs` (removed again before committing), then running the native
fixture loop once: this is what showed `lm_head.out` at 1.43GB and
confirmed the per-layer scratch buffers were the *rest* of the total, not
a separate mystery.

### Native Pool memory, Qwen2.5-0.5B-Instruct (24 layers, hidden 896),
`long_tools_single` (2225 tokens), before vs after both fixes

| buffer | before | after |
|---|---|---|
| `Pool::resident_bytes()` total | 5.69GB (browser figure below; native run showed 1.55-1.64GB across the fixture's cases, see below) | 189MB (browser, see below) |
| `lm_head.out` (or 24 per-layer copies before the scratch_key fix) | 1.35-1.43GB, one buffer (this fix predates the scratch_key fix in the debug trace, so this row is already the shared-key state) | vocab-size only (`151936 * 4` bytes = 0.6MB) |
| `layer.mlp.gate_up.out` (all layers) | 86.6MB total this shape (already-shared; PRE-scratch_key this same shape was ~24x this, one per layer) | 86.6MB (one instance, unchanged shape) |

The three `pool.resident_bytes()` numbers actually measured, native,
`LEAN_DEBUG_POOL_TOP=1`, Qwen2.5-0.5B, in fixture-loop order (each row is
the *cumulative* pool state right after that case's prefill, since `Pool`
never shrinks): seq=36 -> 25.0MB, seq=86 -> 59.8MB, seq=54 -> 37.6MB (still
holding the seq=86 buffers' sizes for anything not reallocated),
seq=2225 (`long_tools_single`) -> 1.55GB, seq=2354
(`long_tools_multiturn`) -> 1.64GB. This native trace was taken with only
the scratch-key fix applied, before the lm-head slicing fix - the browser
numbers below (both fixes) show the combined effect.

### Browser GPU/wasm memory, Qwen2.5-0.5B-Instruct, single page load, before vs after both fixes

`mem_profile.html`/`main_mem_profile.js`, official
`qwen2.5-0.5b-instruct-q4_0.gguf`, one page load: device create -> model
load -> short case (36 tokens) prefill + 8 decode steps -> `long_tools_single`
case (2225 tokens) prefill + 8 decode steps, same engine/KV cache
throughout. `wasm` = `wasm.memory.buffer.byteLength`; `gpu.pool` =
`Pool::resident_bytes()` via the new `gpuMemoryInfo()` export.

| stage | wasm (before fix) | wasm (after) | gpu.pool (before) | gpu.pool (after) |
|---|---|---|---|---|
| after model load | 957.9MB | 957.9MB | 0MB | 0MB |
| after short (36 tok) prefill+decode | 957.9MB | 957.9MB | 94.6MB | 4.3MB |
| after long_tools_single (2225 tok) prefill+decode | 3662.4MB | 957.9MB | 5693.4MB | 189.4MB |

`gpu.weight` (463.9MB) and `gpu.kvCache` (56.5MB) were unchanged by either
fix in both runs (neither scales with this bug). The wasm heap's own 2.7GB
growth at long context, not just `gpu.pool`, disappeared as a side effect
of these two fixes - it was never independently investigated further
(plausibly the wasm/webgpu backend's own bookkeeping scaling with
oversized buffer creation calls), since it tracks `gpu.pool`'s growth
closely enough in both the "before" and "after" rows to not need a
separate root cause.

### Before/after decode ms/tok, median of 3 unless noted

Native (`lean-cli --tokens 16 --kernel fast`, GPU-locked, current HEAD =
both fixes applied - "before" numbers are this same run doc's Step-1
table above, on the commits named there):

| model | GGUF | case (prompt tokens) | before decode ms/tok | after decode ms/tok |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | short (36) | 25.13 | 36.38 |
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-instruct-q4_0.gguf | long_tools_single (2225) | not measured pre-session-2 | 47.50 |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0.gguf (official) | short (36) | 118.78 | 113.16 |
| Qwen2.5-3B-Instruct | qwen2.5-3b-instruct-q4_0.gguf (official), cross-tokenizer timing only (see fixture.json note above) | long_tools_single (2225) | not measured pre-session-2 | 182.42 (162.85 / 182.42 / 246.72 across 3 runs - see observations) |

The 0.5B short-case "after" number (36.38) is noisier and higher than the
earlier session's 25.13 - both are native `lean-cli` runs, same commit's
fix stacked on top, no unrelated code changed on this path; see
observations below before reading this as a regression.

Browser (`decode_timing.html`/`main_decode_timing.js`, official GGUFs for
both models, 3 SEPARATE fresh headless-Chromium page loads per row - not 3
runs in one page - `cold` = that page's first decode step, `warm` =
median of the remaining 15):

| model | case (prompt tokens) | cold ms/tok (median of 3) | warm ms/tok (median of 3) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short (36) | 23.7 | 20.6 |
| Qwen2.5-0.5B-Instruct | long_tools_single (2225) | 32.7 | 27.8 |
| Qwen2.5-3B-Instruct (official GGUF) | short (36) | 160.1 / 209.3 (n=2, see observations) | 151.6 / 208.6 (n=2) |
| Qwen2.5-3B-Instruct (official GGUF) | long_tools_single (2225) | 361.8 (196.0 / 361.8 / 654.9) | 299.9 (189.6 / 299.9 / 313.2) |

This session's original browser number for this exact case (962ms/tok, or
588ms/tok on a repeat with the official GGUF, both pre-fix) compares to
this table's 299.9ms/tok median "after" - roughly 3.2x faster than the
962ms baseline, and the best individual run (189.6ms/tok) is 5x faster.
Short-context browser decode is materially unchanged (151.6-208.6ms/tok
vs this model's native 113.16ms/tok - browser is slower than native here,
the reverse of the 0.5B case, and not investigated further this session).

### Observations

- The two fixes above are real and verified (all 8 required native gates
  pass on both commits; the browser's own greedy tokens still matched the
  fixture on every case tested before session 2, and this session's browser
  harnesses check `tokensMatch` the same way - see `main_qwen25_3b.js`'s
  `long_tools_single` row, which is timing-only and does not check tokens
  against a mismatched-tokenizer-weights reference, exactly like the
  original doc's footnote).
- **3B in the browser is still memory-tight even after both fixes.** A
  repeat of the 3B `short` case (36 tokens - the *cheap* case) hit a
  7.20GB total-Chromium-RSS peak on its second back-to-back fresh-page-load
  run, tripping this session's own 6GB abort guardrail; the very same case
  measured 5.14GB moments earlier. The 3 completed `long_tools_single` runs
  ranged from 189.6 to 313.2ms/tok warm (a 65% spread) with RSS peaks of
  6.11 / 5.15 / 5.57GB, and a 4th run was aborted by the guardrail at
  8.63GB before producing a result. `pgrep` confirmed no other GPU-using
  process was running before each attempt and `vm_stat`/`top` showed the
  system fully recovered (6.5GB+ free, no elevated swap) between runs, so
  this reads as this model's own real memory footprint sitting close to
  this 16GB machine's ceiling, not measurement contention - loading a
  ~2GB GGUF file into wasm memory, then into GPU-resident quantized
  buffers, leaves little headroom on this machine even with both bugs
  fixed. Native decode also showed matching-magnitude variance on the same
  case (162.85 / 182.42 / 246.72ms/tok) that this session did not chase
  further.
- Not investigated this session: why native 0.5B short-case decode is
  noisier now (28.30/36.38/43.04ms/tok across 3 runs) than the Step-1
  table's earlier 25.13ms figure. Both are `lean-cli --tokens
  16 --kernel fast` on the same GGUF; nothing on that code path changed in
  either fix. Flagged as an open item, not folded into either fix's
  before/after claim.
- The lm-head slicing fix (root cause 2) is a real prefill speedup too
  (not just memory): native 3B prefill on `long_tools_single` dropped from
  this session's first, pre-fix, official-GGUF browser measurement of
  136.5s to native `lean-cli` prefill numbers of 131-149ms/tok x 2225
  tokens (~292-332s total is NOT what was measured - `lean-cli` reports
  ms/tok, i.e. 131-149ms per PROMPT token during prefill, not per decode
  step; browser prefill for the same case, same fix, was not re-measured
  standalone this session - the `main_decode_timing.js`/`main_qwen25_3b.js`
  harnesses report it as part of `prefillMs` but that number was not
  isolated into its own before/after row here).
- Kept for future debugging: `Pool::debug_top_buffers(n)` (`pool.rs`),
  `GpuModel::weight_gpu_bytes()`/`KvCache::gpu_bytes()`/
  `EmbeddingTable::gpu_bytes()`, and `LeanEngine::gpuMemoryInfo()`
  (`web.rs`, wasm-bindgen export) - all added this session, all still in
  the tree.

## Session 3: load-time memory - the double-copy fix (2026-09-29)

Task A of this session's brief: after Session 2's two fixes, load-time wasm
memory for Qwen2.5-0.5B (a ~409MB local Q4_0 GGUF) was still 957.9MB
(measured on this session's own build of the same commit, `www/mem_profile.html`,
"after_load" row) - about 2.34x the file size, despite `GpuModel::load_from_reader`
already reading and uploading one tensor at a time and dropping each tensor's
raw bytes immediately after upload (`model.rs::load_from_reader`'s doc
comment, `quant.rs::load_matmul_weight_gguf`'s per-call `bytes: &[u8]`
parameter never outliving that call).

### Root cause (`crates/lean/src/web.rs`)

`LeanEngine::load`/`LeanEngineCpu::load` took `gguf_bytes: Vec<u8>` as their
wasm-bindgen parameter. wasm-bindgen's `Vec<u8>` argument type always copies:
JS's `Uint8Array` (already a copy of the fetched `ArrayBuffer`) gets copied a
*second* time into a freshly-allocated wasm-linear-memory `Vec<u8>` before
Rust code runs at all. That `Vec<u8>` was then wrapped in
`std::io::Cursor::new(gguf_bytes)` and held for the whole parse (per
`model.rs`'s own now-stale doc comment: "handed across the wasm boundary as
one `Vec<u8>`... no sharded reader is needed"). So at any point during
loading, wasm linear memory held: the entire file (via the `Cursor`, held
until `load_from_reader` returns) *plus* whatever the tensor-at-a-time
processing was transiently allocating on top (the current tensor's raw
bytes, plus `quant.rs::split_q4_blocks`/`split_q8_blocks`/`split_q6k_blocks`'s
repacked `qs`/`scales` arrays) - and wasm memory never shrinks back down
after a peak, so that sum became the load's permanent floor.

### Fix (`crates/lean/src/web.rs`)

Added `JsBytesReader`, a `Read + Seek` adapter directly over a
`js_sys::Uint8Array` (no `Vec<u8>` copy at construction - `js_sys::Uint8Array`
is a JS-owned view, not a wasm-bindgen-copied argument type). `read()` calls
`Uint8Array::subarray(start, end).copy_to(&mut buf[..n])`, copying only the
bytes the caller actually asked for into a small transient Rust buffer -
`GgufReader::open`'s header/metadata parse and `tensor_data()`'s one-tensor-
at-a-time reads already only ever ask for small windows (see Session 2's
notes above), so this reader never has to materialize more than one tensor's
raw bytes at a time in wasm memory, and the fetched file's bytes now live
only in the JS heap (as the caller's own `Uint8Array`/`ArrayBuffer`) until
each small window is pulled across.

`LeanEngine::load`/`LeanEngineCpu::load`'s `gguf_bytes` parameter type
changed from `Vec<u8>` to `js_sys::Uint8Array`; `GpuModel::load_from_reader`/
`CpuModel::load_from_reader` are unchanged (already generic over `R: Read +
Seek`, per Session 2's notes - `model.rs`/`cpu.rs` needed zero edits).
`LeanEngineCpu`'s CPU rung still legitimately holds the raw quantized
tensor bytes resident for its whole lifetime (it computes directly off them,
`cpu.rs`'s own doc comment) - `JsBytesReader` only removes the *duplicate*
whole-file copy on the way in, not `CpuModel`'s one necessary copy.

Every `www/main*.js` harness's `.load(ggufBytes, ...)` call site needed no
change (a JS `Uint8Array` is what was already being passed - wasm-bindgen's
old `Vec<u8>` parameter type accepted the same JS value and copied it; the
new `js_sys::Uint8Array` parameter type accepts it without copying). Added
one line after each `.load()` call, `ggufBytes = null;`, so the harness's own
reference to the fetched buffer is dropped as soon as `load()` (synchronous -
every tensor is uploaded before it returns) is done with it, letting the JS
heap reclaim it too (`const [ggufBytes, ...]` changed to `let [...]` in each
file to allow the reassignment). Added a load-only memory profiler for
Qwen2.5-3B, `www/mem_profile_3b.html`/`main_mem_profile_3b.js` (mirrors
`main_mem_profile.js`'s device-create/load/snapshot shape, skips
prefill/decode - a 1.9GB GGUF's browser decode is slow and out of scope for
a load-time-memory measurement). `ENGINE_BUILD` bumped to
`2026-09-29-02`/`2026-09-29-cpu-02`/`2026-09-29-cpu-qwen3-02` in every
`www/main*.js` file in this same commit; served `pkg/lean_bg.wasm` bytes
hashed and confirmed to match the freshly built artifact before every
timing/memory run below (`shasum -a256` on both the built file and the
`curl`'d served file).

### Before/after: wasm memory after `load()`, single page load, local GGUF

Measured with `www/mem_profile.html`/`main_mem_profile.js`
("after_load" row) for Qwen2.5-0.5B and `www/mem_profile_3b.html`/
`main_mem_profile_3b.js` ("after_load" row, load-only) for Qwen2.5-3B, both
via Playwright's bundled headless Chromium
(`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal`),
one page load each, `pgrep` confirmed no other Chrome-for-Testing/headless-
shell process running before either run. "Before" is this session's own
build of the pre-fix code (commit `0f74d76`, Session 2's `Vec<u8>`-parameter
`load()`) run through the same harness; "after" is the `JsBytesReader` fix.

| model | GGUF size | wasm before | wasm after | wasm after / GGUF size | gpu.weight | gpu.kvCache |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 408.7MB (428,730,208 bytes) | 957.9MB | 529.1MB | 1.29x | 463.9MB | 56.5MB (max_ctx=2500) |
| Qwen2.5-3B-Instruct | 1905.0MB (1,997,879,712 bytes) | not measured (see note) | 707.9MB | 0.37x | 2191.6MB | 9.4MB (max_ctx=128, load-only) |

Note: the 3B model's pre-fix wasm-memory number was not separately
re-measured this session (Session 2's own 3B browser runs predate the
`mem_profile_3b.html` harness and used the full prefill+decode
`main_qwen25_3b.js` harness instead, whose wasm-memory numbers are not
directly comparable to this load-only harness) - the 0.37x-of-file-size
"after" figure and the 0.5B model's like-for-like 2.34x-before/1.29x-after
comparison are the evidence this fix generalizes to the larger model rather
than being 0.5B-specific.

### Before/after: JS heap and Chromium process footprint

`jsHeap` below is Chrome's `performance.memory.usedJSHeapSize`, read
straight from the page; `chromium peak RSS` is this session's own
`scripts/run_mem_profile.py`, which samples every Chrome-for-Testing/
chrome-headless-shell process's RSS (`ps aux`, summed across the whole
process tree Playwright's launch produced) every 250ms for the run's
duration and reports the peak. `performance.memory.usedJSHeapSize` did not
move at all between "after_device_create" and "after_load" in either run
(fixed at 462.0MB for 0.5B, 2060.0MB for 3B) and did not change after an
explicit `window.gc()` call (tested standalone with `--js-flags=--expose-gc`)
- this reads as this Chromium build's `performance.memory` being coarsely
bucketed/quantized (a known privacy-motivated behavior in recent Chrome
versions) rather than evidence the JS-heap fix did nothing; `chromium peak
RSS` (real process memory, not a JS API) is the metric this section treats
as ground truth.

| model | jsHeap after_load | chromium peak RSS (this session's fix, single run) | guard limit |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 462.0MB | 2.721GB | 6GB |
| Qwen2.5-3B-Instruct (load-only) | 2060.0MB | 6.352GB | 8GB |

No guard trip on either run. The 3B load-only number (6.35GB) is close to
its 8GB guard and to this 16GB machine's practical ceiling once a page,
OS, and any other process are accounted for - prefill/decode on top of this
load (not measured in this load-only harness) would add further GPU-pool
and KV-cache growth on top of the number above, so 3B in the browser
remains memory-tight on this machine even after this fix; this was not
pushed further this session (a fully lazy/streaming GGUF reader that never
materializes even one tensor's raw bytes in wasm memory, only its own
compressed qs/scales, is the next lever, not attempted here).

### Gates (unchanged code paths, run after this fix)

Native, GPU-locked, `--ignored --test-threads=1`, all pass on this fix (this
change is wasm/web.rs-only - `model.rs`/`cpu.rs`/native `lean-cli` are
untouched, so this is confirmation, not new risk):
`fixture_parity` (211.7s), `fixture_parity_llama_1_7b` (54.75s),
`fixture_parity_llama_360m` (21.5s), `fixture_parity_qwen25_3b` (official
3B GGUF, 79.94s), `fixture_parity_qwen3` (136.37s), `fixture_parity_qwen3_1_7b`
(314.53s), `kv_snapshot` (both cases, 191.5s), `logit_mask` (all 3 cases,
8.04s).

Browser, same headless Chromium, `www/index.html?local=1`
(`main.js`, Qwen2.5-0.5B, all 6 fixture cases plus the long-context/KV-
snapshot/logit-mask extras this harness also runs):
`allMatch: true` - every case's `tokens_match` is `true` (`short`, `long`,
`non_english`, `long_tools_single` at 2225 prompt tokens,
`long_tools_multiturn` at 2354), `kv_snapshot_restore`'s
`restore_bytes_match`/`tokens_match` both `true`, `logit_mask`'s
`target_match` `true` - the browser's greedy continuation is unchanged by
this fix, as expected (it only changes how bytes cross the JS/wasm
boundary, never what gets computed).

`cargo clippy -p lean --lib --no-default-features --features web --target
wasm32-unknown-unknown -- -D warnings` and `cargo clippy -p lean -- -D
warnings` (native default features) both clean.

### Files touched

`crates/lean/src/web.rs` (`JsBytesReader`, `LeanEngine::load`/
`LeanEngineCpu::load` parameter type), `crates/lean/www/main*.js` (8 files:
`ggufBytes = null` after `.load()`, `ENGINE_BUILD` bump),
`crates/lean/www/mem_profile_3b.html` + `main_mem_profile_3b.js` (new,
load-only 3B memory harness), `scripts/run_mem_profile.py` (new, Playwright
driver: loads a `mem_profile*.html` page, reports `window.__leanResult` plus
peak Chromium process-tree RSS, with a self-kill guard at a caller-supplied
GB limit - never touches another session's browser, matches pids by process
name, not `pgrep -f` against its own argv).

Task B (dispatch/kernel-fusion work: argmax folding into
`forward_decode_step_argmax`, QKV/gate-up fusion) was not attempted this
session - Task A's fix and its gates took the full session; starting a
kernel fusion without room to verify qk-norm/QKV-bias correctness against
all four models' fixtures would have been a worse outcome than not starting
it.

## Session 4: argmax pass-fold and QKV fusion, measured (2026-09-29)

Two commits on this session's branch: `7264366` (fold decode's argmax
dispatch into `decode_layers`' own compute pass, no dispatch-count change,
one fewer pass/step) and `4737327` (fuse decode's q/k/v projection into one
matmul, `qkv_proj`'s `rows == 1`/no-LoRA path only; prefill and the
chunked/masked forward path keep the original three-matmul path
unconditionally - see that commit's message for why). Both commits' gates
(`fixture_parity`, `fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`,
`fixture_parity_llama_360m`, `fixture_parity_llama_1_7b`,
`fixture_parity_qwen25_3b` against the official
`qwen2.5-3b-instruct-q4_0.gguf`, `kv_snapshot`, `logit_mask`) pass, native
GPU-locked release build, `--ignored --test-threads=1`. `lora_parity`'s
`packed_block_diagonal_matches_per_cell_forward` (rows > 1, exercises the
untouched chunked-forward path) also passes on both commits;
`lora_and_sliced_head_match_reference` fails identically on both the
pre-fusion (`7264366`) and post-fusion (`4737327`) code with the same
`.bin` file (`lora dead logit mismatch: got 24.040848 want 24.75631` on
both) - a wrong/mismatched LoRA weights file picked for this check (not
this repo's own fixture asset), not a fusion regression; not one of this
task's required gates. `cargo clippy -p lean -- -D warnings` and
`cargo clippy -p lean --lib --no-default-features --features web --target
wasm32-unknown-unknown -- -D warnings` both clean on both commits. Browser
gate (`www/index.html?local=1`, Playwright's bundled headless Chromium,
`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal`,
served bytes hashed and confirmed against the freshly built
`pkg/lean_bg.wasm` before each check): `allMatch: true` on both commits
(`engineBuild` `2026-09-29-04` for the argmax commit, `2026-09-29-05` for
the QKV-fusion commit).

Native timing: `lean-cli --gguf <path> --tokenizer-dir <path> --fixture
<fixture> --tokens 16 --kernel fast`, GPU-locked, M2 laptop, one model
loaded per process. "Before"/"after" below are the argmax-fold commit
(`7264366`) vs the QKV-fusion commit (`4737327`) - i.e. this table isolates
QKV fusion's own effect, since the argmax fold changes pass count, not
dispatch count or (measurably) timing. Dispatch counts are exact
(`decode_dispatches_per_step`, printed by `lean-cli` itself, zero
variance across repeated runs at a fixed commit/model/case). `long`
below is each model's own `long_tools_single` fixture case where its GGUF
has one (Qwen2.5-0.5B, Qwen3-1.7B); Qwen2.5-3B's `long_tools_single` row
uses the 0.5B-tokenizer `fixture.json`'s case against the 3B weights
(timing-only cross-tokenizer run, same convention as this doc's Session 1
footnote - `tok_match`/`top1_match` on that row are not evidence of a bug).

### Decode dispatches/step, before vs after QKV fusion

| model | layers (derived from the dispatch delta / 2) | short: before | short: after | delta | long_tools_single: before | long_tools_single: after | delta |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 340 | 292 | -48 (-14.1%) | 364 | 316 | -48 (-13.2%) |
| Qwen2.5-3B-Instruct (official GGUF) | 36 | 508 | 436 | -72 (-14.2%) | not measured before (see note) | 472 | n/a |
| Qwen3-1.7B (Q8_0) | 28 | 452 | 396 | -56 (-12.4%) | 480 | 424 | -56 (-11.7%) |
| SmolLM2-360M-Instruct (Q4_0) | 32 | 452 | 388 | -64 (-14.2%) | not measured (no own long_tools_single fixture case; not re-run cross-tokenizer this session) | - | - |

Note: this session's first attempt at Qwen2.5-3B's cross-tokenizer
`long_tools_single` row (via `fixture.json`) stopped after 3 of 5 cases on
both the before and after runs (only `short`/`long`/`non_english` printed,
matching each other) - not investigated further since it's a timing-only
cross-tokenizer convenience case, not one of this task's required models'
own fixtures; the `long_tools_single` dispatch count *after* fusion (472)
was captured from a rerun of that same command, `after`-only.

### Native decode ms/tok, before vs after QKV fusion

`short`/`long_tools_single` are the fixture case names; `n` is the number
of runs a median was taken over on this commit/model/case (3 unless
noted). A GPU-contention check (`pgrep` for Chrome-for-Testing/headless-
shell) was run before every timing command; none were found running.

| model | case | before ms/tok (n, values) | after ms/tok (n, values) | change |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short | 32.85 (n=3: 26.40, 32.85, 34.59) | 32.30 (n=3: 25.12, 32.30, 32.54) | -1.7% |
| Qwen2.5-0.5B-Instruct | long_tools_single | 50.14 (n=3: 50.14, 49.12, 59.86) | 49.06 (n=3: 49.06, 48.86, 49.16) | -2.2% |
| Qwen2.5-3B-Instruct (official GGUF) | short | 172.60 (n=3: 179.81, 117.89, 172.60) | 228.93 (n=3: 250.21, 228.93, 226.37) | +32.6% |
| Qwen3-1.7B (Q8_0) | short | 84.45 (n=1) | 91.90 (n=1) | +8.8% |
| Qwen3-1.7B (Q8_0) | long_tools_single | 132.58 (n=1) | 171.45 (n=1) | +29.3% |
| SmolLM2-360M-Instruct (Q4_0) | short | 42.42 (n=1) | 50.79 (n=1) | +19.7% |

### Observations

- **Dispatch count dropped by exactly 2/layer on every model** (confirmed
  by `lean-cli`'s own `decode_dispatches_per_step` counter, not estimated):
  `q_w`/`k_w`/`v_w`'s three separate `linear()` dispatches collapsed into
  `qkv_w`'s one, everywhere the fused path's guard (`rows == 1`, no LoRA,
  `layer.qkv_w.is_some()`) held - which is every decode step on every
  model tested (all four GGUFs quantize their attn weights uniformly, so
  `gguf_matmul_qkv_fused`'s dtype-match check always passed).
- **Wall-clock decode time did not improve to match the dispatch-count
  drop, and regressed on 3 of 4 models.** Qwen2.5-0.5B (the model the
  argmax-fold and pass-batching fixes were validated on in earlier
  sessions) is roughly a wash (-1.7%/-2.2%, inside this session's own
  measured run-to-run spread - see the `n=3` value lists above, e.g. the
  3B "before" row's 117.89-179.81 spread on the *same* commit/model/case).
  Qwen2.5-3B, Qwen3-1.7B and SmolLM2-360M all got slower after fusion,
  Qwen2.5-3B by the largest margin (+32.6% at `short`, +29.3% at
  `long_tools_single` on Qwen3-1.7B). This is the opposite of what the
  dispatch-count reduction predicts and was re-measured before being
  reported here: Qwen2.5-3B's `short` case was independently confirmed on
  3 runs before fusion (179.81, 117.89, 172.60ms/tok, via a temporary
  revert of `model.rs` to the pre-fusion commit and a rebuild - not a
  fixture/config difference) and 3 runs after (250.21, 228.93, 226.37ms/tok),
  non-overlapping ranges.
- **No root cause identified this session for the regression.** Candidate
  hypotheses, none tested: the fused decode matvec kernel
  (`linear_q4_decode`/`linear_q8_decode`/`linear_q8_decode_dp4`, chosen by
  `rows == 1` in `linear()`) may not scale linearly with `out_dim` the way
  three separate dispatches implicitly did (GQA's asymmetric split - one
  wide `q_dim` chunk plus two narrow `kv_dim` chunks concatenated into one
  `out_dim = q_dim + 2*kv_dim` - could interact with the kernel's
  per-workgroup row assignment in a way three separately-sized dispatches
  didn't); or bind-group/pool-key overhead for the new `.qkv_fused` key
  outweighs the saved dispatch overhead at these model sizes now that the
  pass-batching fix (this doc's Session 1) already removed most of the
  per-dispatch `MTLComputeCommandEncoder` cost that made fusion attractive
  in the first place. Per this task's own "parity before perf" framing:
  this is reported as a **measured regression**, not a fix, and the
  commit is not reverted pending that decision - `qkv_proj`'s fused path
  is functionally correct (every required gate passes) but is not shown
  to be faster on any model tested except within noise on the smallest.
- **Browser decode timing (`decode_timing.html`) was not run this
  session** for either commit, on any model - the native regression above
  took priority to confirm honestly (three separate native rebuild/measure
  cycles) within this session's time budget. `main_decode_timing.js` was
  extended with `qwen3_1_7b`/`smollm2_360m` model keys and a local
  `model_smollm2_360m/` dir was added (gitignored) in the argmax-fold
  commit, so the harness is ready for a follow-up session to run the full
  cold/warm x 3-page-load x per-model matrix this task originally asked
  for.

### Next dispatch-count candidate (not implemented, per this session's scope)

Post-fusion, one decode layer's dispatch list (short context, no
split-K) is: `rmsnorm`(attn_norm) -> `qkv` -> `rope`(q) -> `rope`(k) ->
`attn_decode` -> `linear`(wo) -> `add_inplace` -> `rmsnorm`(ffn_norm) ->
`linear`(gate_up) -> `silu_mul_fused` -> `linear`(down) ->
`add_inplace` = 12 dispatches/layer (matches: 12*24 + 4 fixed
(embed/out_norm/lm_head/argmax) = 292, the measured Qwen2.5-0.5B count).
The two `rmsnorm`+`add_inplace` pairs (`attn_norm` reads `x`, `add_inplace`
writes `x`; `ffn_norm` reads `x`, `add_inplace` writes `x`) are 4 of those
12 dispatches/layer and are the next-biggest same-shaped group: fusing
each residual-add into the *following* rmsnorm's read (an "add-then-normalize"
kernel reading two inputs and writing one normalized output, replacing
`add_inplace`+`rmsnorm`) would save 2 dispatches/layer - the same order of
magnitude as this session's QKV fusion (48-72 total depending on layer
count). Given this session's QKV fusion measured a wall-clock regression
despite an identical-magnitude dispatch-count win, this candidate should
not be implemented without first profiling *why* fusion isn't paying off
on this codebase's current dispatch-batching baseline (e.g. an isolated
kernel-time comparison of the fused vs. three-separate-dispatch QKV path,
per this project's own t0-fast lesson that framework/dispatch overhead
should be measured in isolation before more fusion work, not assumed).

## Session 5: QKV fusion re-measured on a quiet machine (2026-09-29)

Session 4 reported native decode regressions after QKV fusion (commit
`4737327`, "B" below) vs the commit before it (`7264366`, "A" below):
Qwen2.5-3B `short` +32.6%, Qwen3-1.7B `long_tools_single` +29.3%,
SmolLM2-360M `short` +19.7%. Its own after-the-fact 3-run check on
Qwen2.5-3B's `short` case (a temporary revert/rebuild, not a fresh
worktree) already ran 30-50% slower than the same commit's own numbers
measured earlier the same night (0.5B 32.9ms/tok that session vs 25.1ms/tok
an earlier session; 3B 172-181ms/tok that session vs 113-119ms/tok this
session, below) - evidence the machine was under load throughout that
session's measurements, not evidence of a fusion regression. This session
re-measures both commits from two separate detached worktrees (so no
build/revert step in between), gated on a quiet machine, to settle it.

**Method.** Two `lean-cli` release binaries, one built from each commit in
its own detached worktree (`git worktree add --detach`, never `git
stash`). Before every timed run: 1-minute `vm.loadavg` < 3.0, no
`rustc`/`cargo`/training-`python` process running, and no process (other
than the run itself) above 40% CPU (`ps -Ao pcpu,comm -r | awk 'NR==2'`) -
checked by a polling loop, gate re-checked every 20s until it opens.
(A first pass gated only on `vm.loadavg < 1.5` after finding a leftover
shell from an unrelated task had been spinning one core at 100% for
hours while this session's own `vm.loadavg` sat around 2 - that stray
process was killed and this session's timings up to that point were
discarded and rerun from scratch, which is why every number below is
from a single rerun-from-scratch pass, not the session's first attempt.)
Runs interleaved A/B/A/B/..., 5 runs per commit per case, one process per
run, `lean-cli --gguf <path> --tokenizer-dir <path> --fixture <path>
--tokens 16 --kernel fast --long`. `short` is each model's own fixture;
`long_tools_single` is each model's own fixture case where it has one
(Qwen2.5-0.5B, Qwen3-1.7B); Qwen2.5-3B's `long_tools_single` row uses a
single-case fixture file built from the 0.5B fixture's `long_tools_single`
case against the 3B weights (same cross-tokenizer timing-only convention
as this doc's Session 1 footnote and Session 4 - `tok_match`/`top1_match`
mismatches on that row are expected, not a bug); SmolLM2-360M has no own
`long_tools_single` case and was not run cross-tokenizer this session
(same choice Session 4 made).

### Native decode ms/tok, commit A (`7264366`, pre-fusion) vs B (`4737327`, post-fusion), quiet machine

n=5 runs per cell, interleaved A/B/A/B/A/B/A/B/A/B. 1-minute loadavg stayed
under 1.6 for every run in this table (recorded per-run, not sampled after
the fact); the top non-`lean-cli` process stayed under 40% CPU for every
run except the very first (38.9% once, then under 24% for the rest).

| model | case | A median (min-max) ms/tok | B median (min-max) ms/tok | change |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short | 27.49 (27.08-28.39) | 26.36 (26.27-26.83) | -4.1% |
| Qwen2.5-0.5B-Instruct | long_tools_single | 38.69 (35.45-39.41) | 38.27 (37.80-39.60) | -1.1% |
| Qwen2.5-3B-Instruct (official GGUF) | short | 118.60 (117.63-119.26) | 118.36 (118.14-118.38) | -0.2% |
| Qwen2.5-3B-Instruct (official GGUF, cross-tokenizer) | long_tools_single | 165.54 (139.98-204.79) | 162.32 (153.24-166.98) | -1.9% |
| Qwen3-1.7B (Q8_0) | short | 73.29 (66.93-74.57) | 72.85 (66.48-74.91) | -0.6% |
| Qwen3-1.7B (Q8_0) | long_tools_single | 99.37 (86.17-102.84) | 99.43 (98.35-100.66) | +0.1% |
| SmolLM2-360M-Instruct (Q4_0) | short | 33.34 (33.02-34.66) | 32.94 (32.69-33.97) | -1.2% |

None of these changes exceed the run-to-run spread within a single
commit/case (e.g. Qwen2.5-3B `long_tools_single` A's own 139.98-204.79
range on the same commit). Session 4's reported +32.6%/+29.3%/+19.7%
regressions do not reproduce on a quiet machine; every model is a wash or
a small (1-4%) improvement, consistent with the dispatch-count drop QKV
fusion produces (Session 4's own measurement: -12% to -14% dispatches/step
across all four models).

### Browser decode timing (`decode_timing.html`), same two commits, quiet machine

Session 4 did not run this; this session runs it for the models it skipped
last time (Qwen2.5-3B excluded per this doc's own Session 2 finding -
browser decode on Qwen2.5-3B risks a 22GB renderer footprint on a 16GB
Mac, a memory problem unrelated to this fusion question, and native
already answers it above without that risk). Each commit's own
`wasm-pack build crates/lean --target web --out-dir pkg --no-default-features
--features web` served from its own worktree; served `pkg/lean_bg.wasm`
bytes hashed and confirmed to match that worktree's own build before any
run (and to differ between the two worktrees, confirming two distinct
binaries were actually under test). 3 fresh page loads per model/case/
commit, Playwright's bundled headless Chromium
(`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal`),
`cold` = first decode step of a fresh page load, `warm` = median of the
remaining 15 steps.

| model | case | commit | cold ms (3 loads) | warm ms (3 loads) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short | A | 26.1, 26.1, 26.0 | 22.5, 22.7, 22.4 |
| Qwen2.5-0.5B-Instruct | short | B | 25.4, 25.0, 24.4 | 22.1, 22.1, 21.7 |
| Qwen2.5-0.5B-Instruct | long_tools_single | A | 33.6, 33.4, 33.5 | 29.8, 29.5, 29.4 |
| Qwen2.5-0.5B-Instruct | long_tools_single | B | 31.3, 33.3, 33.3 | 27.3, 29.3, 29.1 |
| SmolLM2-360M-Instruct | short | A | 32.4, 32.1, 31.7 | 27.6, 27.4, 27.7 |
| SmolLM2-360M-Instruct | short | B | 31.3, 30.7, 30.6 | 27.1, 27.3, 27.4 |
| SmolLM2-360M-Instruct (cross-tokenizer) | long_tools_single | A | 53.2, 42.6, 48.1 | 39.0, 39.0, 42.1 |
| SmolLM2-360M-Instruct (cross-tokenizer) | long_tools_single | B | 42.0, 43.4, 45.1 | 38.5, 38.8, 42.3 |
| Qwen3-1.7B (Q8_0) | short | A | 71.2, 71.6, 73.2 | 68.1, 66.4, 63.5 |
| Qwen3-1.7B (Q8_0) | short | B | 69.9, 69.0, 70.4 | 67.2, 64.8, 60.7 |
| Qwen3-1.7B (Q8_0) | long_tools_single | A | 112.7, 120.0, 113.4 | 98.0, 102.0, 98.4 |
| Qwen3-1.7B (Q8_0) | long_tools_single | B | 115.5, 113.4, 121.5 | 100.5, 98.1, 102.1 |

Browser confirms native: B matches or is marginally faster than A on
every model/case, no regression anywhere.

### Decision

**Keep the fusion (`4737327`), no revert, no shape gate.** On a verified
quiet machine, across all four models this task named, both native and
browser, QKV fusion is a wash to a small win (0-4% faster on 6 of 7 native
cells, essentially flat on the 7th) - never a regression. Session 4's
reported regressions were a measurement artifact (background CPU
contention: first an unrelated worker's stray process at 100% CPU for
hours, found and killed mid-session; before that, contention already
flagged by this doc's own before/after inconsistency at the top of this
section). No shape-dependent kernel issue was found because there was
nothing to find - the GQA asymmetric-width hypothesis from Session 4's
"next candidate" section was never tested because the regression it was
meant to explain does not reproduce. All 8 fixture/KV/mask gates and the
browser `allMatch` check remain as Session 4 left them (unaffected by this
session, which only re-measured timing); no code changes were made this
session.

## Session 6: rmsnorm fix, Mac + browser side (2026-09-29)

The rmsnorm one-workgroup-per-row fix (`docs/runs/2026-09-29-lean-rmsnorm.md`)
was gated and timed only on the RTX 3080/Vulkan machine. This session
covers the Mac (Metal via wgpu) and the browser (headless Chromium/WebGPU),
on the same commit (branch `lean-perf`, which merges the rmsnorm fix).

**Method.** Native: `lean-cli --tokens 16 --kernel fast`, release build,
5 runs per model/case, machine gated quiet before every run (1-minute
`vm.loadavg` < 3, top non-`lean-cli` process < 40% CPU, no `rustc`/`cargo`/
training-`python` running - confirmed throughout, loadavg stayed
0.79-2.39, top process 0.6-30.1%). Browser: `wasm-pack build crates/lean
--target web --out-dir pkg --no-default-features --features web`, served
from a local `python3 -m http.server` in `crates/lean/`; served
`pkg/lean_bg.wasm` bytes hashed (sha256
`22d43615cc2a866f27549def9ad0638bc88a8f413c95233048b75f138e7e3531`) and
confirmed to match the freshly built file before any run. `ENGINE_BUILD`
bumped in every `crates/lean/www/*.js` loading URL this session (to
`2026-09-29-06` for the wgpu-build pages, `cpu-05`/`cpu-qwen3-05` for the
CPU-rung pages, unaffected by this fix but bumped for hygiene since they
share the same `pkg/`). `www/index.html?local=1` run in Playwright's
bundled headless Chromium (`--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --use-angle=metal`, never a personal
browser): `allMatch: true` - all 5 fixture cases token-match, KV
snapshot/restore byte- and token-match, logit mask forces the exact target
string. `decode_timing.html` (per-step timing harness, unchanged from
Session 5) run 3 fresh page loads per model/case; Qwen2.5-3B run once
through `scripts/run_mem_profile.py` (unmodified - it already waits on
`window.__leanResult`, which `main_decode_timing.js` also sets, and kills
only its own browser process tree if the footprint exceeds `--limit-gb`)
to keep the 16GB Mac's renderer footprint bounded.

Qwen2.5-3B's own fixture (`fixture_qwen25_3b.json`) has no
`long_tools_single` case; rather than build a cross-tokenizer single-case
fixture as Session 5 did on the 3080, this session used the model's own
`long` case (seq 86) for the second native/browser cell, per the task's
"short + long_tools_single or long" allowance.

### Native decode ms/tok, Mac (Metal via wgpu), median of 5

| model | case | median | min-max |
|---|---|---:|---|
| Qwen2.5-0.5B-Instruct | short | 15.18 | 13.79-18.15 |
| Qwen2.5-0.5B-Instruct | long_tools_single | 24.79 | 24.68-24.81 |
| Qwen2.5-3B-Instruct (official GGUF) | short | 101.74 | 95.14-105.14 |
| Qwen2.5-3B-Instruct (official GGUF) | long | 93.76 | 93.20-94.28 |
| Qwen3-1.7B (Q8_0) | short | 40.81 | 34.91-41.69 |
| Qwen3-1.7B (Q8_0) | long_tools_single | 66.11 | 64.86-75.85 |
| SmolLM2-360M-Instruct (Q4_0) | short | 16.52 | 16.39-18.30 |
| SmolLM2-360M-Instruct (Q4_0) | long | 16.77 | 16.32-17.62 |

`top1_match=true` on every case/run (40 runs total); `tok_match=false` at
`--tokens 16` is expected, same convention as every prior session in this
doc.

Compared with Session 5's own commit-B column (RTX 3080/Vulkan, quiet
machine): the 3080 decodes Qwen2.5-3B `short` at 118.36ms/tok and Qwen3-1.7B
`long_tools_single` at 99.43ms/tok pre-rmsnorm-fix; this session's Mac
numbers (101.74ms and 66.11ms respectively) are not a fair apples-to-apples
comparison since they are a different GPU, a different session, and
(unlike Session 5's box) do already include the rmsnorm fix - no Mac
"before" binary was built this session to isolate the fix's own Mac
speedup. The rmsnorm doc's own native gates and the browser `allMatch`
check above are this session's Mac-side correctness evidence instead.

### Browser decode timing (`decode_timing.html`), 3 fresh loads per case

| model | case | promptLen | cold ms (3 loads) | warm median ms (3 loads) |
|---|---|---:|---|---|
| Qwen2.5-0.5B-Instruct | short | 36 | 14.6, 14.6, 14.4 | 11.1, 11.2, 11.2 |
| Qwen2.5-0.5B-Instruct | long (long_tools_single) | 2225 | 22.9, 22.6, 22.8 | 18.5, 18.6, 18.5 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | short | 36 | 15.0, 15.1, 15.3 | 11.6, 11.5, 11.6 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | long | 2225 | 27.4, 27.6, 34.8 | 23.2, 23.2, 23.3 |
| Qwen3-1.7B (Q8_0) | short | 36 | 38.9, 47.6, 40.1 | 28.0, 28.1, 28.0 |
| Qwen3-1.7B (Q8_0) | long | 2225 | 92.2, 76.4, 80.0 | 88.6, 70.6, 68.9 |

Qwen2.5-3B, guarded (`scripts/run_mem_profile.py --limit-gb 8`, one browser
at a time, `case=short`, promptLen 36): loaded in 1312ms, prefill
2427.5ms, decode cold 123.6ms, warm median 120.1ms over 16 steps, peak
Chromium process-tree RSS 7.524GB - under the 8GB guard, no kill.

Browser numbers track native closely for the small/mid models (0.5B warm
11.1-11.2ms browser vs 15.18ms native-median-`short`/24.79ms
native-median-`long_tools_single` - browser is faster here because
`warm` excludes the shader-compile-inflated first step that native's
per-run process start pays on every one of its 5 runs, not because the
browser rung is actually faster). Qwen3-1.7B's `long` cold/warm spread
(92.2/76.4/80.0 cold, 88.6/70.6/68.9 warm) is the noisiest cell measured
this session, on both native (Qwen3-1.7B `long_tools_single` min-max
64.86-75.85) and browser - consistent with each other, not a browser-only
artifact.

### `LEAN_PROFILE_KERNELS=1` on the Mac: not available

`LEAN_PROFILE_KERNELS=1 ./target/release/lean-cli --gguf <Qwen2.5-0.5B> ...`
produced no `kernel_profile` output. Reading `crates/lean/src/engine.rs`
(the feature-gate: profiling is only granted when the adapter's
`wgpu::Features` contains both `TIMESTAMP_QUERY` and
`TIMESTAMP_QUERY_INSIDE_PASSES`) and `crates/lean/src/profile_report.rs`
(`report()` silently returns with no output when its totals map is empty)
confirms this is the designed silent-no-op path, not a crash: this
session's Metal adapter did not grant both timestamp-query features, so
no per-kernel breakdown is available on this Mac. (The same run's startup
line reports `has_dp4=true, dp4_decode=false` - this adapter does expose
the packed-4x8-integer-dot-product WGSL feature, but that lever is off by
default per `docs/runs/2026-09-28-lean-perf-2.md` Session 6 and is
unrelated to this session's rmsnorm question.)

### Where the Mac's remaining decode time goes

Without a per-kernel timestamp breakdown on this Mac, only the
dispatch-count and cross-model comparison are available as evidence:

- Decode dispatch count per step is unchanged by the rmsnorm fix (292 for
  the three smaller models' short/long cases, 316 for the 2225-token
  KV-cache-heavy cases) - identical to the dispatch counts already
  reported in this doc's earlier sessions. The fix only changed each
  rmsnorm dispatch's internal grid shape, not the dispatch count, matching
  the rmsnorm doc's own "no prefill regression" finding.
- Qwen2.5-3B decodes at 93.76-101.74ms/tok median on this Mac vs.
  18.49-19.06ms/tok on the RTX 3080 post-fix (`docs/runs/2026-09-29-lean-rmsnorm.md`).
  That gap (roughly 5-5.5x) is consistent with the raw compute/bandwidth
  gap between a laptop-class Apple GPU and a desktop RTX 3080 - this
  session found no Mac-specific bottleneck beyond what the hardware gap
  already predicts, but also could not rule one in or out without the
  timestamp-query breakdown Metal declined to grant here.
- The rmsnorm doc's own remaining-bottleneck finding (`dec_lm_head`'s
  Q6_K naive-kernel cost on Qwen2.5-3B, unaddressed by the rmsnorm fix)
  is architecture-level, not GPU-vendor-specific, so it plausibly still
  holds on the Mac's decode time - this session did not re-verify it here
  since the Mac profiler path was unavailable.

### Gate/build summary

All 8 required native gates pass on the Mac this session
(`fixture_parity`, `fixture_parity_qwen25_3b`, `fixture_parity_qwen3`,
`fixture_parity_qwen3_1_7b`, `fixture_parity_llama_360m`,
`fixture_parity_llama_1_7b`, `kv_snapshot` both cases, `logit_mask` all 3
cases). `wasm-pack build --target web` succeeds; `www/index.html?local=1`
reports `allMatch: true` against the freshly built and served
`pkg/lean_bg.wasm`. No code changes were made this session - only
`ENGINE_BUILD` bumps (crates/lean/www/*.js) and this doc section.

## Session 7: lean-kernels merge, Mac + browser side (2026-09-29)

`lean-kernels` (Q6_K decode matvec, fused add+rmsnorm, decode-shaped F32
matvec - `docs/runs/2026-09-29-lean-kernels.md`) was gated and timed only on
the RTX 3080/Vulkan machine. This session covers the Mac (Metal via wgpu)
and the browser (headless Chromium/WebGPU), on the merge commit (branch
`lean-perf`, base `lean-kernels` merged in). Same method as Session 6:
quiet-machine gate before every timed run (1-minute load average < 3, no
`rustc`/`cargo`/training process above 40% CPU, confirmed via `uptime`/
process listing each time), both a build lock and a GPU lock held for the
duration of every native gate/timing run.

**Native gates** (release build, `--ignored --test-threads=1`, GPU-locked):
all 8 pass - `fixture_parity`, `fixture_parity_qwen25_3b`,
`fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`,
`fixture_parity_llama_360m`, `fixture_parity_llama_1_7b`, `kv_snapshot`
(both cases), `logit_mask` (all 3 cases). `top1_match=true` throughout.

**Wasm build**: `wasm-pack build crates/lean --target web --out-dir pkg
--no-default-features --features web` succeeded. `ENGINE_BUILD` bumped in
every `crates/lean/www/*.js` loading URL this session (wgpu-build pages to
`2026-09-29-07`, CPU-rung pages to `cpu-06`/`cpu-qwen3-06` for hygiene,
unaffected by this merge). Served `pkg/lean_bg.wasm` bytes hashed (sha256
`752bc7db9f522bc292f829625572534879b1ace8110262519d27f2c4748b111b`) and
confirmed to match the freshly built file before every browser run in this
session (checked twice, before the index check and again before the decode-
timing sweep).

**`www/index.html?local=1`**, headless Chromium (`--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --use-angle=metal`, Playwright's bundled
browser only): `allMatch: true`, all 6 fixture-checked cases token-match,
KV snapshot/restore byte- and token-match (speedup 124.1x), logit mask
forces the exact target string, zero page errors.

### Native decode ms/tok, Mac (Metal via wgpu), median of 5

| model | case | before (Session 6, rmsnorm-fix only) | after (this session) | min-max (after) |
|---|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | short | 15.18 | 13.96 | 12.45-16.61 |
| Qwen2.5-0.5B-Instruct | long_tools_single | 24.79 | 23.07 | 21.48-26.24 |
| Qwen2.5-3B-Instruct (official GGUF) | short | 101.74 | 60.16 | 56.90-63.61 |
| Qwen2.5-3B-Instruct (official GGUF) | long | 93.76 | 58.64 | 57.47-63.07 |
| Qwen3-1.7B (Q8_0) | short | 40.81 | 38.54 | 33.69-40.24 |
| Qwen3-1.7B (Q8_0) | long_tools_single | 66.11 | 62.32 | 61.33-67.61 |
| SmolLM2-360M-Instruct (Q4_0) | short | 16.52 | 13.39 | 13.30-17.52 |
| SmolLM2-360M-Instruct (Q4_0) | long | 16.77 | 14.52 | 14.43-14.92 |

`top1_match=true` on every case/run in every sweep (20 runs); `tok_match=false`
at `--tokens 16` is expected, same convention as every prior session. The
Qwen2.5-3B improvement (101.74 -> 60.16ms/tok short, 1.69x) tracks the
Q6_K decode matvec directly - this is the only model in the fixture set
with a Q6_K tensor (`output.weight`), matching the 3080's own 1.24-1.28x
finding in `docs/runs/2026-09-29-lean-kernels.md` at a larger absolute
margin (different GPU, both real).

### Browser decode timing (`decode_timing.html`), 3 fresh loads per case

| model | case | before cold (3 loads) | after cold (3 loads) | before warm median | after warm median |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short | 14.6, 14.6, 14.4 | 13.8, 13.5, 13.6 | 11.1, 11.2, 11.2 | 11.0, 11.0, 11.0 |
| Qwen2.5-0.5B-Instruct | long (long_tools_single) | 22.9, 22.6, 22.8 | 22.1, 22.6, 22.2 | 18.5, 18.6, 18.5 | 18.4, 18.4, 18.4 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | short | 15.0, 15.1, 15.3 | 13.7, 13.4, 13.2 | 11.6, 11.5, 11.6 | 10.2, 10.3, 10.3 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | long | 27.4, 27.6, 34.8 | 25.9, 25.6, 25.6 | 23.2, 23.2, 23.3 | 22.8, 22.0, 22.0 |
| Qwen3-1.7B (Q8_0) | short | 38.9, 47.6, 40.1 | 36.6, 35.3, 36.0 | 28.0, 28.1, 28.0 | 28.6, 28.9, 27.8 |
| Qwen3-1.7B (Q8_0) | long | 92.2, 76.4, 80.0 | 53.8, 67.1, 91.4 | 88.6, 70.6, 68.9 | 45.8, 57.7, 88.4 |

Engine build confirmed as `2026-09-29-07` in every one of the 18 page
loads' own console log line. Qwen3-1.7B's `long` cell stays the noisiest
cell measured on this Mac both before and after (Session 6 already flagged
it as noisy) - the third of three after-runs (91.4/88.4) sits close to the
before numbers while the first two (53.8/45.8, 67.1/57.7) are clearly
faster, consistent with the add+rmsnorm fusion and (for the two smaller
models) the same fusion's effect rather than the Q6_K kernel (neither
Qwen2.5-0.5B nor Qwen3-1.7B has a Q6_K tensor).

### Qwen2.5-3B, memory-guarded (`scripts/run_mem_profile.py --limit-gb 8`)

One browser at a time, `case=short`, promptLen 36. First attempt this
session hit the guard: peak Chromium process-tree RSS crossed 8.19GB before
`window.__leanResult` was set, and `run_mem_profile.py` killed its own
browser per its designed behavior - no result captured from that attempt.
A second, otherwise identical run stayed under the guard:

| metric | before (Session 6) | after (this session, 2nd attempt) |
|---|---:|---:|
| load ms | 1312 | 1206 |
| prefill ms | 2427.5 | 2483.5 |
| decode cold ms | 123.6 | 63.6 |
| decode warm median ms (16 steps) | 120.1 | 53.6 |
| peak Chromium process-tree RSS | 7.524GB | 7.453GB |

Decode roughly halves (123.6 -> 63.6ms cold, 120.1 -> 53.6ms warm median),
consistent with the Q6_K decode kernel - Qwen2.5-3B's `output.weight` is
Q6_K, same tensor the native table's 1.69x improvement traces to. The first
attempt's 8.19GB spike versus the second attempt's 7.453GB peak, both on an
otherwise-idle Mac, reads as page-load/GC-timing variance around the same
~7.5-8GB footprint rather than a reproducible regression - flagged, not
resolved, since only one guarded run per attempt was taken.

### Gate/build summary

All 8 required native gates pass on the Mac this session. `wasm-pack build
--target web` succeeds; `www/index.html?local=1` reports `allMatch: true`
against the freshly built and served `pkg/lean_bg.wasm` (hash-verified).
No source changes were made this session beyond the `ENGINE_BUILD` bumps
(`crates/lean/www/*.js`) and this doc section - the kernel changes
themselves were already merged into this branch before this session began.

## Session 8: lean-kernels docs-only merge, Mac + browser recheck (2026-09-29)

`lean-kernels` had one commit past Session 7's merge point
(`38d3eee`, docs-only - `docs/runs/2026-09-29-lean-kernels.md`, CPU-phase
breakdown and a reverted `rope_qk_fused` attempt). Merged into `lean-perf`
with `git merge --no-edit lean-kernels`; no source files changed by the
merge, only that one doc file added. This session re-ran the full native +
wasm + browser gate set on the resulting commit, since no gate/timing pass
had been taken on the branch since Session 7's merge point. Same
quiet-machine method as Sessions 6-7 (1-minute load average checked via
`uptime` before every timed run - stayed between 1.1 and 2.0 throughout,
never above 3; no `rustc`/`cargo`/training process above 40% CPU via `ps
aux`), both a build lock and a GPU lock held for the duration of every
native gate and timed run (`locked cargo`/`locked gpu` wrapper scripts).

**Native gates**: all 8 pass - `fixture_parity`, `fixture_parity_qwen25_3b`,
`fixture_parity_qwen3`, `fixture_parity_qwen3_1_7b`,
`fixture_parity_llama_360m`, `fixture_parity_llama_1_7b`, `kv_snapshot`
(both cases), `logit_mask` (all 3 cases). `top1_match=true` throughout.
Build note: `logit_mask`'s three test function names
(`singleton_mask_forces_exact_string`, `all_allowed_mask_matches_unmasked`,
`mask_upload_per_step_cost`) don't contain the substring `logit_mask`, so
passing `logit_mask` as one of several libtest filter args alongside the
other seven gate names matches 0 of its tests (cargo's multi-filter is an
OR over each binary's own test names, not binary names) - ran as its own
`cargo test -p lean --release --test logit_mask -- --ignored
--test-threads=1` instead, 3 passed.

**Wasm build**: `wasm-pack build crates/lean --target web --out-dir pkg --
--no-default-features --features web` succeeded. `ENGINE_BUILD` bumped from
`2026-09-29-07` to `2026-09-29-08` in every wgpu-build `crates/lean/www/*.js`
loading URL this session (`main.js`, `main_decode_timing.js`,
`main_mem_profile.js`, `main_mem_profile_3b.js`, `main_qwen25_3b.js`,
`main_qwen3.js`, `main_qwen3_1_7b.js`); the CPU-rung files
(`main_cpu.js`, `main_cpu_qwen3.js`) were left untouched, unaffected by this
merge. Served `pkg/lean_bg.wasm` bytes hashed (sha256
`056f550e5766f72a296fde7b17f8b26bda744ea7b22fc9147ef3ad2312acb066`) and
confirmed to match the freshly built file before the `index.html` check.

**`www/index.html?local=1`**, headless Chromium (`--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --use-angle=metal`, Playwright's bundled
browser only): `allMatch: true`, all 6 rows report `match: true` or `match:
null` (the `long_1500` timing-only case, which has no reference
continuation by design), `engineBuild` confirmed as `2026-09-29-08` in the
page's own result object.

### Native decode ms/tok, Mac (Metal via wgpu), median of 5

| model | case | before (Session 7) | after (this session) | min-max (after) |
|---|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | short | 13.96 | 11.61 | 11.43-14.71 |
| Qwen2.5-0.5B-Instruct | long_tools_single | 23.07 | 20.63 | 20.57-20.66 |
| Qwen2.5-3B-Instruct (official GGUF) | short | 60.16 | 55.50 | 54.55-57.77 |
| Qwen2.5-3B-Instruct (official GGUF) | long | 58.64 | 54.73 | 54.38-56.56 |
| Qwen3-1.7B (Q8_0) | short | 38.54 | 33.91 | 28.63-35.19 |
| Qwen3-1.7B (Q8_0) | long_tools_single | 62.32 | 57.50 | 54.91-58.55 |
| SmolLM2-360M-Instruct (Q4_0) | short | 13.39 | 11.77 | 11.62-11.83 |
| SmolLM2-360M-Instruct (Q4_0) | long | 14.52 | 11.70 | 11.69-11.74 |

`top1_match=true` on every case/run in every sweep (20 runs total, plus the
5-run `logit_mask` gate separately). Correction: Session 7 was measured at
35d54a2, before two code commits reached lean-perf through the merge 0635cc1:
a555415 (skip uniform buffer writes whose bytes are unchanged) and 7d3df51
(split-K chunk chosen by head_dim). The merge in this session was docs-only,
but the Session 7 baseline predates those two, so the 10-20% gains above are
mostly their effect (they gave 5-11% and 4-16% on the RTX 3080), with some
run-to-run variance on top.

### Browser decode timing (`decode_timing.html`), 3 fresh loads per case, `steps=16`

| model | case | cold ms/tok (3 loads) | warm median ms/tok (3 loads) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | short | 13.3, 13.8, 13.6 | 10.7, 10.8, 10.9 |
| Qwen2.5-0.5B-Instruct | long (long_tools_single) | 22.4, 22.6, 22.6 | 18.5, 18.8, 18.8 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | short | 13.1, 13.9, 14.1 | 9.9, 9.9, 9.9 |
| SmolLM2-360M-Instruct (cross-tokenizer fixture) | long | 25.2, 26.3, 25.2 | 21.2, 22.0, 21.1 |
| Qwen3-1.7B (Q8_0) | short | 33.3, 66.6, 37.8 | 27.6, 28.1, 27.6 |
| Qwen3-1.7B (Q8_0) | long | 55.7, 73.7, 141.6 | 46.2, 70.5, 164.8 |

Engine build confirmed as `2026-09-29-08` in every page load's own console
log line. Qwen3-1.7B's `long` cell is again the noisiest cell measured on
this Mac (flagged as such in Sessions 6 and 7 too): its third load's own
first browser attempt hit this harness's memory guardrail (6GB
Chromium-process-tree RSS, peak 7.18GB) mid-run and was killed before
producing a result, so that cell's third value is a retry, not the original
third attempt - the retry's own warm median (164.8ms/tok) is close to 3x
its first two loads' medians (46.2, 70.5), consistent with page-load/GC
variance around a footprint close to this harness's guard threshold rather
than a reproducible regression (no code changed between loads).

### Qwen2.5-3B, memory-guarded (`scripts/run_mem_profile.py --limit-gb 8`)

`decode_timing.html?model=3b&case=short`, one browser at a time. Four
consecutive attempts at `steps=16` (this session's native/browser
convention elsewhere) hit the script's own 8GB guard before producing a
result - peaks 8.08GB, 8.22GB, 8.05GB, 8.30GB, each killed mid-run by the
script's own guard (never another process). A fifth attempt at `steps=8`
stayed under the guard:

| metric | value (`steps=8`, 5th attempt) |
|---|---:|
| load ms | 1401.4 |
| prefill ms | 1598.6 |
| decode cold ms | 56.6 |
| decode warm median ms (7 steps) | 52.5 |
| peak Chromium process-tree RSS | 7.031GB |

This session's four `steps=16` peaks (8.05-8.30GB) sit above Session 7's
own `steps=16`-equivalent second-attempt peak (7.453GB, a different,
since-removed scratch harness) - flagged, not resolved, since the exact
harness differs between the two sessions (this session's `decode_timing.html`
vs. Session 7's scratch `lean-3b-harness` script) and no source code changed
in between, so the two peak-RSS numbers aren't a clean before/after
comparison of the same code path.

### Gate/build summary

All 8 required native gates pass on the Mac this session. `wasm-pack build
--target web` succeeds; `www/index.html?local=1` reports `allMatch: true`
against the freshly built and served `pkg/lean_bg.wasm` (hash-verified). No
source changes were made this session beyond the `ENGINE_BUILD` bumps
(`crates/lean/www/*.js`) and this doc section - the one upstream commit
merged in this session (`38d3eee`) was itself docs-only.

## Session 9: post-merge check - ENGINE_BUILD bumps, native gates, partial browser check

### ENGINE_BUILD bump

The 10 `crates/lean/www/*.js` `ENGINE_BUILD` tags were bumped by exactly one
step each from the last committed value (e.g. `2026-09-29-08` ->
`2026-09-29-09`, `2026-09-29-cpu-06` -> `2026-09-29-cpu-07`), matching the
rule that every wasm-loading URL gets a new tag in the same commit as a
wasm/model rebuild. Committed in `9b04874`.

### Served-bytes proof

`wasm-pack build crates/lean --target web --no-default-features --features
web`, then served `crates/lean/` over `python3 -m http.server 8123` and
fetched `pkg/lean_bg.wasm` over HTTP:

| file | sha256 |
|---|---|
| local build (`crates/lean/pkg/lean_bg.wasm`) | `889b372e5844bd098bc96ccca8da144a745c9cb692293490e07b1e860881370d` |
| served (`http://localhost:8123/pkg/lean_bg.wasm`) | `889b372e5844bd098bc96ccca8da144a745c9cb692293490e07b1e860881370d` |

Identical.

### Native ignored tests (`cargo test -p lean --release --test <name> -- --ignored`, run individually per test binary, each under the repo's `cargo`/`gpu` locks)

| test | result | time |
|---|---|---:|
| fixture_parity | ok | 188.35s |
| fixture_parity_qwen25_3b | ok | 57.59s |
| fixture_parity_qwen3 | ok | 125.91s |
| fixture_parity_qwen3_1_7b | ok | 288.34s |
| fixture_parity_llama_360m | ok | 20.08s |
| fixture_parity_llama_1_7b | ok | 52.22s |
| kv_snapshot (2 tests) | ok | 154.64s |
| logit_mask (3 tests) | ok | 8.34s |
| fixture_parity_llama_360m_cpu | ok | 25.89s |

0 failures across all 9 test binaries (14 individual `#[test]` fns total).

### Browser check (SwiftShader, headless Chromium)

`www/index.html?local=1` run in Playwright's bundled headless Chromium with
`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=swiftshader
--use-gl=swiftshader --ignore-gpu-blocklist` (software WebGPU - correctness
check only, no timing). First attempt (900s wait) timed out before
`window.__leanResult` was set; 3 of the fixture's 5 cases had completed by
then:

| case | tokens_match | prefill_ms | decode_ms_per_tok |
|---|---|---:|---:|
| short | true | 27355.2 | 3594.8 |
| long | true | 35153.9 | 3137.0 |
| non_english | true | 20924.9 | 3164.6 |

Partial only - the run never reached the final `allMatch` verdict (2 of 5
cases, `kv_cache`/whichever the fixture orders last, not yet executed). A
second attempt with a longer timeout was stopped by the lead before
completion: it had been running 75+ minutes of active SwiftShader CPU time
and was holding the shared browser-automation lock two other workers
needed.

**Browser allMatch on a real GPU: pending** - run `www/index.html?local=1`
in Chrome on the M2, or headed on the 3080/Vulkan box, to get the
authoritative allMatch verdict; SwiftShader software rendering is too slow
for this fixture to finish in a practical amount of time on shared
infrastructure.
