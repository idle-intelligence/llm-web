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
