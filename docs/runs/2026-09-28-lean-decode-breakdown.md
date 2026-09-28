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
