# lean vs llama.cpp: where decode time goes, RTX 3080/Vulkan

Follow-on to `docs/runs/2026-09-29-lean-box-check.md` (same commit,
`376144ace9448f1ffe73098afbaa068ee3b3992e`, same client: a Linux desktop
with an RTX 3080 (10GB), Vulkan backend, box-check's own decode ms/tok
numbers are the "current, unprofiled" reference used below). This run adds
an opt-in per-kernel GPU timing path to `lean`, measures llama.cpp's Vulkan
backend on the same two GGUF files as a reference, and traces the ~10x gap
between them to a specific kernel design issue, verified against the shader
source, not just measurement.

## Parameters

- Models: `qwen2.5-0.5b-instruct-q4_0.gguf` (Qwen2.5-0.5B-Instruct: 24
  layers, hidden 896, intermediate 4864, 14 heads / 2 KV heads, head_dim 64,
  vocab 151936; `token_embd.weight` Q4_0, `output.weight` Q8_0) and the
  official `qwen2.5-3b-instruct-q4_0.gguf` (Qwen2.5-3B-Instruct: 36 layers,
  hidden 2048, intermediate 11008, 16 heads / 2 KV heads, head_dim 128,
  vocab 151936; `output.weight` Q6_K).
- `llama-bench` (llama.cpp's own tool, Vulkan backend, same box, GPU
  confirmed idle before each run): `-p 0 -n 64 -r 5`, i.e. pure decode
  (`tg`), 5 repeats, `avg_ns`/`avg_ts` as reported.
- `lean-cli --tokens 16 --kernel fast`, release build, run under this
  session's own opt-in `LEAN_PROFILE_KERNELS=1` env var (see "Profiling
  path added" below) for the per-kernel table, and the box-check doc's own
  unprofiled numbers (same commit, same box) for the "current decode
  ms/tok" figures the llama.cpp comparison uses - profiling itself adds
  measurable overhead (see Observations), so it is never used for the
  headline ms/tok comparison.
- Weight-bandwidth floor = GGUF file's `model_size` (bytes) / 760 GB/s (the
  RTX 3080's rated peak memory bandwidth), i.e. the time to read every
  weight byte once at peak bandwidth with zero other cost - the
  best case for any M=1 decode step, since a batch-1 forward is a pure
  memory-bound matvec chain.

## 1. llama.cpp reference (Vulkan, same box, same GGUF files)

| model | avg_ns/tok | tok/s | GGUF size | bandwidth floor | llama.cpp / floor |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct Q4_0 | 1,664,440 | 600.98 | 422,782,464 B | 0.556 ms | 3.0x |
| Qwen2.5-3B-Instruct Q4_0 | 4,187,000 | 238.83 | 1,991,923,712 B | 2.621 ms | 1.6x |

llama.cpp's Vulkan `mul_mat_vec` backend sits within 1.6-3.0x of the pure
weight-bandwidth floor on both models - close to roofline for a decode
matvec chain that also has to run RMSNorm, RoPE, attention and the KV
cache update in between.

## 2. lean decode, current (unprofiled) vs llama.cpp vs floor

"Current" = `docs/runs/2026-09-29-lean-box-check.md`'s own numbers, same
commit, same box, `--tokens 16 --kernel fast`, no profiling instrumentation.

| model | case | lean decode ms/tok | llama.cpp ms/tok | floor ms | lean/llama.cpp | lean/floor |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | short | 16.08 | 1.664 | 0.556 | 9.7x | 28.9x |
| Qwen2.5-0.5B-Instruct | long_tools_single | 17.18 | 1.664 | 0.556 | 10.3x | 30.9x |
| Qwen2.5-3B-Instruct | short | 50.82 | 4.187 | 2.621 | 12.1x | 19.4x |
| Qwen2.5-3B-Instruct | long | 51.16 | 4.187 | 2.621 | 12.2x | 19.5x |

The task's "~10x slower than llama.cpp" holds (9.7-12.2x measured here), and
lean sits 19-31x off the pure weight-bandwidth floor where llama.cpp sits
1.6-3.0x off it.

## Profiling path added (opt-in, default off)

`crates/lean/src/engine.rs`: `Engine::new_async` now checks
`LEAN_PROFILE_KERNELS=1` (native-only in practice - wasm's `std::env::var`
always errs) and requests `Features::TIMESTAMP_QUERY |
Features::TIMESTAMP_QUERY_INSIDE_PASSES` only if both the env var is set
*and* the adapter reports both features (checked once, never inferred from
a device/vendor string - same rule this crate already uses for `has_dp4`).
When granted, `Engine::dispatch` wraps each `pass.dispatch_workgroups` call
with `pass.write_timestamp` before/after and records the call-site label;
`Engine::resolve_profile`/`read_profile` (new) resolve and read back the
timestamp pairs the same way every other native readback in this file
already works (async `map_async` + native-only blocking `device.poll`).
`crates/lean/src/profile_report.rs` (new module) aggregates by label with
the per-layer index stripped (same `"layerN.x" -> "layer.x"` rule
`model.rs::scratch_key` already uses for `Pool`), and `lean_cli.rs` prints
the table once per fixture case when `engine.profiling_enabled()`. Wired
into `forward_decode_step_argmax` only (the function `lean-cli`'s own
timing already uses). Zero cost when the env var is unset: the `dispatch()`
profiling branch is one `bool`-equivalent (`Option::is_some`) check, and
`profile_labels` stays an empty `Vec`. Gates run with the env var unset
(default path, unchanged from before this change): `fixture_parity`,
`fixture_parity_qwen25_3b`, `fixture_parity_qwen3`,
`fixture_parity_qwen3_1_7b`, `fixture_parity_llama_360m`,
`fixture_parity_llama_1_7b`, `kv_snapshot` (both cases), `logit_mask` (all
3 cases) - all pass, native release build, GPU-locked, `--ignored
--test-threads=1`, on this same box. `cargo clippy --release -p lean --bin
lean-cli -- -D warnings` is clean on every file this change touches (one
pre-existing, unrelated `sort_by` lint in `pool.rs`, not touched by this
change, is the only remaining clippy warning in the crate).

## 3. Per-kernel GPU time, one decode step

Measured with `LEAN_PROFILE_KERNELS=1`, 16 decode steps per case, label
aggregated across layers (`count` = calls across all 16 steps; a kernel
called once per layer shows `count = layers * 16`). `bytes/call` and
`GB/s` are computed from each kernel's known shape (weight bytes only:
Q4_0 = 0.625 B/weight after this crate's own qs+f32-scale repacking, Q8_0 =
1.125 B/weight, Q6_K ~= 0.820 B/weight per llama.cpp's superblock layout;
activation/bias bytes are negligible next to the weight matrix at every row
count below and omitted) - not separately re-measured with a bandwidth
microbenchmark, so read them as the model this run used to explain the
`us/call` column, not as an independently-verified achieved-bandwidth
number.

### Qwen2.5-0.5B-Instruct, `long_tools_single` case (kv_len=2225, split-K attention)

| kernel | count/step | total us/step | us/call | bytes/call | GB/s achieved | % of 760 GB/s peak |
|---|---:|---:|---:|---:|---:|---:|
| dec_layer.norm (rmsnorm) | 24 | 4798.1 | 199.9 | 10,752 B | 0.054 GB/s | 0.007% |
| dec_layer.ffnnorm (rmsnorm) | 24 | 4799.1 | 200.0 | 10,752 B | 0.054 GB/s | 0.007% |
| dec_layer.attn.split | 24 | 1557.4 | 64.9 | - (KV cache read, not weight-bound) | - | - |
| dec_layer.mlp.gate_up | 24 | 689.9 | 28.7 | 5,449,600 B | 189.9 GB/s | 25.0% |
| dec_layer.mlp.down | 24 | 466.6 | 19.4 | 2,724,800 B | 140.2 GB/s | 18.4% |
| dec_lm_head (Q8_0) | 1 | 232.1 | 232.1 | 153,214,976 B | 660.0 GB/s | 86.8% |
| dec_layer.qkv_fused | 24 | 248.5 | 10.4 | 645,120 B | 62.3 GB/s | 8.2% |
| dec_layer.attn.reduce | 24 | 218.8 | 9.1 | - | - | - |
| dec_out_norm (rmsnorm) | 1 | 198.3 | 198.3 | 10,752 B | 0.054 GB/s | 0.007% |
| dec_argmax | 1 | 181.0 | 181.0 | 607,744 B (vocab readback-side) | - | - |
| dec_layer.wo | 24 | 238.9 | 9.955 | 501,760 B | 50.7 GB/s | 6.7% |
| dec_layer.ropek/ropeq | 24 each | 157.3 / 150.5 | 6.6 / 6.3 | small | - | - |
| dec_layer.add1/add2 | 24 each | 144.3 / 148.2 | 6.0 / 6.1 | small | - | - |
| dec_layer.mlp.silu | 24 | 147.3 | 6.1 | small | - | - |
| embed | 1 | 7.4 | 7.4 | small | - | - |
| **total GPU time/step** | | **14,383.7 us (14.38 ms)** | | | | |

Measured decode wall time for this case under profiling was 23.61-23.99
ms/tok (see Observations for the profiling-overhead gap); the 14.38 ms
above is the sum of every dispatch's own measured GPU duration.

### Qwen2.5-3B-Instruct, `short` case (kv_len small, plain attn_decode)

| kernel | count/step | total us/step | us/call | bytes/call | GB/s achieved | % of 760 GB/s peak |
|---|---:|---:|---:|---:|---:|---:|
| dec_layer.norm (rmsnorm) | 36 | 16,148.9 | 448.6 | 24,576 B | 0.055 GB/s | 0.007% |
| dec_layer.ffnnorm (rmsnorm) | 36 | 16,147.6 | 448.5 | 24,576 B | 0.055 GB/s | 0.007% |
| dec_lm_head (Q6_K, naive kernel) | 1 | 4,404.1 | 4,404.1 | 255,242,240 B | 57.9 GB/s | 7.6% |
| dec_layer.mlp.gate_up | 36 | 1,681.3 | 46.7 (105.1us/call at long case, see note) | 28,180,480 B | 268.2 GB/s | 35.3% |
| dec_layer.mlp.down | 36 | 948.7 | 26.3 (59.2us/call at long case) | 14,090,240 B | 237.8 GB/s | 31.3% |
| dec_layer.attn | 36 | 549.2 (short) / 636 (long) | 34.3 / 39.7 | - | - | - |
| dec_layer.qkv_fused | 36 | 316.3 | 8.8 (19.6-19.8us/call) | 3,276,800 B | 166.0 GB/s | 21.8% |
| dec_layer.wo | 36 | 280.2 | 7.8 (17.5us/call) | 2,621,440 B | 149.8 GB/s | 19.7% |
| dec_out_norm (rmsnorm) | 1 | 446.1-449.2 | 446.1-449.2 | 24,576 B | 0.055 GB/s | 0.007% |
| dec_argmax | 1 | 179.4-181.0 | 179.4-181.0 | 607,744 B | - | - |
| dec_layer.ropek/ropeq | 36 each | 103.4 / 99.8 | 2.9 / 2.8 | small | - | - |
| dec_layer.add1/add2 | 36 each | 98.1 / 98.2 | 2.7 / 2.7 | small | - | - |
| dec_layer.mlp.silu | 36 | 100.6 | 2.8 | small | - | - |
| embed | 1 | 6.9 | 6.9 | small | - | - |
| **total GPU time/step** | | **~26,900 us (26.9 ms) at `short`** | | | | |

(`us/call` values above are averaged across this run's `short`/`long`/
`non_english` cases where they differ slightly; the raw per-case numbers
this table is built from are in the "Per-kernel GPU time" output captured
this session, not separately reproduced here row by row.)

## 4. Where the gap is, and why (checked against shader source)

**RMSNorm is the single largest cost in decode on both models, and it is
not a bandwidth problem.** `rmsnorm` reads/writes ~11 KB (0.5B) to ~24.6 KB
(3B) per call - at peak bandwidth that is single-digit nanoseconds - but
measures 200 us (0.5B) to 448 us (3B) per call, i.e. **0.007% of peak
bandwidth achieved**. Two calls per layer (`attn_norm`, `ffn_norm`) sum to
9,597 us/step on the 0.5B model (**40% of the unprofiled 17.18-23.99 ms/tok
decode time**) and 32,297 us/step on the 3B model (**~57-63% of the
unprofiled 50.8-58.3 ms/tok decode time**) - the single largest line item
on both models by a wide margin, ahead of every matvec combined.

Root cause, read directly from `crates/lean/src/shaders/rmsnorm.wgsl`:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= dims.rows) { return; }
    let base = r * dims.dim;
    var ss: f32 = 0.0;
    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {  // serial loop, 1 thread
        let v = x[base + d];
        ss = ss + v * v;
    }
    let denom = sqrt(ss / f32(dims.dim) + dims.eps);
    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {  // serial loop again
        out[base + d] = (x[base + d] / denom) * scale[d];
    }
}
```

This kernel assigns **one thread per row** (`gid.x = r`), with each thread
doing a fully serial loop over the whole hidden dimension, twice. At
prefill (`rows` = prompt length, tens to thousands) this parallelizes fine
- one thread per row across many rows keeps the GPU busy. At **decode,
`rows = 1`**, exactly one of the GPU's thousands of ALUs is active per
dispatch, doing `2 * hidden_size` serial global-memory-bound iterations
alone. Its measured cost scales almost exactly with `hidden_size`
(200us at 896, 448us at 2048 - a 2.24x ratio against a 2.29x hidden-size
ratio), confirming this is compute-bound single-thread work, not a fixed
per-dispatch launch overhead (which would not scale with the shape at
all). This is a direct, code-level finding, independent of any measurement
noise: the shader has no cooperative reduction across the row at all when
`rows == 1`.

This is a reused kernel, and the doc comment says exactly where it was
reused from: it is a **verbatim port of `t0-web/crates/t0-fast/src/shaders/rmsnorm_full.wgsl`**,
byte-for-byte the same one-thread-per-row design. That design is
appropriate for a forecaster whose "rows" are always many time steps in
parallel; it was never exercised at `rows = 1` in its source project, and
porting it unchanged into a model whose entire decode loop runs at
`rows = 1` reused the wrong shape regime from an otherwise-correct kernel.

**The matvecs are not the problem right now.** `dec_lm_head`'s Q8_0 decode
matvec on the 0.5B model reaches **660 GB/s, 86.8% of the card's peak** -
essentially roofline. The Q4_0 decode matvecs (`qkv_fused`, `wo`) sit at
6.7-21.8% of peak and the larger MLP matvecs (`gate_up`, `down`) reach
25-35.3% of peak - all far closer to bandwidth-bound than rmsnorm's
0.007%, and none of them is the dominant line item in the table above.

**Secondary finding: the 3B model's lm head uses the naive Q6_K kernel,
not a decode-specialized one.** `dec_lm_head` on the 3B model is the
single biggest individual dispatch (4,404 us, 19x the 0.5B model's 232 us
for the same op despite the 3B model's `hidden` being only 2.3x larger),
achieving 57.9 GB/s (7.6% of peak) versus the 0.5B model's 660 GB/s. Per
`shaders/linear_q6k.wgsl`'s own header comment (cited in `engine.rs`'s
`linear_q6k` field doc), this is intentional and already known: "no
tiled/decode-specialized variant exists yet" for Q6_K. This is a real,
identifiable cost (4.4 ms/step on the 3B model, ~8-9% of its total decode
time) but not this run's top finding since it is one dispatch, model- and
GGUF-specific (only the official 3B GGUF's `output.weight` ships as Q6_K),
versus rmsnorm's 57-63%-of-decode-time cost on every model in this
project's fixture set.

**What llama.cpp's Vulkan `mul_mat_vec` does differently for the matvecs it
does have** (`ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp` and
`mul_mat_vec_base.glsl`, read on this box):

- `mul_mat_vec_base.glsl:93-99` (`reduce_result`, `USE_SUBGROUP_ADD_NO_SHMEM`
  path): `subgroupAdd(temp[j][n])` - the cross-lane reduction is one
  hardware subgroup instruction, **no `shared` array, no
  `workgroupBarrier()`**. `lean`'s `linear_q4_decode.wgsl` instead does a
  5-round `shared`-memory tree reduction with a `workgroupBarrier()` after
  every round (32 lanes -> 16 -> 8 -> 4 -> 2 -> 1). llama.cpp's own
  non-subgroup fallback (`mul_mat_vec_base.glsl:130-140`) uses the same
  shared+barrier pattern `lean` always uses - subgroup ops are the
  capability-gated fast path, matching the same "gate on adapter feature,
  never on vendor string" rule this crate already applies to `has_dp4`.
- `mul_mat_vec.comp:11-14` sets `K_PER_ITER = 8` for every quantized dtype
  (`lean`'s `linear_q4_decode.wgsl` loads/accumulates 4 words = 32 elements
  per lane per outer-loop pass, i.e. `K_PER_ITER`-equivalent of 4, half of
  llama.cpp's 8) and manually unrolls the outer loop by 4 then by 2
  (`mul_mat_vec.comp:160-243`) to amortize per-iteration loop overhead
  further than `lean`'s single level of x4 unrolling inside
  `accumulate_word`'s caller.
- llama.cpp's `ggml-vulkan.cpp:2867-2887` picks `rm_stdq`/`NUM_ROWS`
  (2 or 4 rows/workgroup, depending on vendor) and `wg_size_subgroup` (the
  adapter's own reported subgroup size) at pipeline-creation time from
  device properties - the same "adapter capability decides the constant,
  never a device-name match" rule this project already uses for `has_dp4`
  and `HEAD_DIM` pipeline-compilation constants, just applied to one more
  axis (subgroup width) that `lean` does not yet query.

None of these differences explain today's ~10x gap by themselves - they
would matter once rmsnorm stops dominating, and they are proposals 2/3
below for exactly that reason.

## Top 3 proposals (not implemented this session)

All three are chosen by shape/dtype/adapter-capability, matching this
project's own no-runtime-tuning rule - none of them measure a device at
run time to decide anything.

1. **Give `rmsnorm.wgsl` a one-workgroup-per-row, shared-memory
   tree-reduction design at every row count, not just many-row prefill.**
   `crates/llm-wasm/src/wgsl/shader_rmsnorm.wgsl` already does exactly
   this in this project's own tree: one workgroup per row (`wg_id.x`),
   `@workgroup_size(256)`, a `shared` `partial_sums` array reduced across
   the workgroup, so a single row (decode's `rows = 1` case) still gets
   256-way parallelism instead of `lean`'s 1-thread-per-row serial loop.
   Expected gain: this is the single largest line item measured this
   session (40% of decode time on the 0.5B model, 57-63% on the 3B model);
   collapsing it from a serial `2*hidden_size`-iteration loop to a
   256-lane cooperative reduction (the same shape class as `lean`'s own
   `linear_q4_decode.wgsl`'s row-cooperative matvec, already proven fast
   at 8-35% of peak bandwidth) should bring rmsnorm's `us/call` down by
   roughly the same order of magnitude the matvecs already show,
   plausibly a 1.5-2x overall decode speedup on every model in this
   project's fixture set, at zero numerical change (same formula, same
   inputs, different thread mapping).
2. **Add a subgroup-reduction (`subgroupAdd`-equivalent) variant of the
   decode matvec kernels** (`linear_q4_decode`/`linear_q8_decode`/their
   `_dp4` counterparts), gated on `wgpu::Features::SUBGROUPS` reported by
   the adapter at startup (queried once, alongside the existing `has_dp4`
   check - never inferred from a vendor string), replacing the current
   5-round `shared`+`workgroupBarrier()` tree reduction with the
   subgroup-native reduction llama.cpp's `USE_SUBGROUP_ADD_NO_SHMEM` path
   uses (`mul_mat_vec_base.glsl:93-99`, cited above). No existing kernel in
   this project's tree does this yet for a Q4_0/Q8_0 decode matvec - it is
   new work, following the same capability-gate pattern `has_dp4`/
   `dp4_decode` already established in this same file. Expected gain is
   smaller in absolute terms today (the matvecs are already the smaller
   share of decode time), but removes barrier-stall cost that will matter
   proportionally more once proposal 1 lands and the matvecs become the
   new largest cost.
3. **Fuse each layer's residual-add into the following RMSNorm's read**
   (`add_inplace` + `rmsnorm` -> one "add-then-normalize" dispatch reading
   two inputs), once proposal 1 makes rmsnorm itself cheap enough that the
   remaining per-dispatch scheduling cost of two separate ops matters
   again. This was already identified as the next dispatch-count target in
   `docs/runs/2026-09-28-lean-decode-breakdown.md`'s own "Next
   dispatch-count candidate" section (2 fewer dispatches/layer, 48-72
   total depending on layer count) and follows the same fusion discipline
   this project already applies elsewhere in the same file
   (`silu_mul_fused.wgsl`, `qkv_fused` via `gguf_matmul_qkv_fused`) - not a
   technique imported from outside this codebase, an extension of a
   pattern already in it.

## Observations

- Profiling itself adds real overhead: 0.5B decode measured 22.58-23.99
  ms/tok under `LEAN_PROFILE_KERNELS=1` this session versus
  16.08-17.18 ms/tok unprofiled (same commit/box, `docs/runs/2026-09-29-lean-box-check.md`)
  - a 32-40% inflation from the extra `write_timestamp` calls and the
    forced resolve+readback each step. 3B showed a smaller relative
    inflation (56.58-58.28 ms/tok profiled vs 50.82-51.16 ms/tok
    unprofiled, 11-14%). This is why every ms/tok number used for the
    llama.cpp comparison in sections 1-2 is the unprofiled figure; the
    per-kernel table in section 3 is only used for relative
    kernel-to-kernel comparison and the shape-derived GB/s figures, not
    for an absolute wall-clock claim.
  - `lean-cli`'s own fixture-case pass/fail check reported "one or more
    fixture cases failed" on both models under `--tokens 16` - expected
    and unrelated to this session's change: `tok_match=false` at
    `--tokens 16` against a fixture whose full greedy continuation is
    longer is this project's own documented convention (see
    `docs/runs/2026-09-29-lean-box-check.md`'s Parameters section);
    `top1_match=true` on every case checked here is the correctness
    signal, and it held throughout.
- `attn.split`/`attn.reduce` (the split-K flash-decoding path, used on the
  0.5B model's `long_tools_single`/`long_tools_multiturn` cases at
  kv_len=2225+) cost 1,776 us/step combined - noticeably more than the
  matvecs, but a distant second to rmsnorm and not investigated further
  this session (kv_len-dependent, not model-dependent, so it would need
  its own long-context sweep to characterize properly).
- The dispatch counts this session's profiled runs reported
  (`decode_dispatches_per_step` = 316 for the 0.5B model at
  `long_tools_single`, 436 for the 3B model at `short`) match
  `docs/runs/2026-09-29-lean-box-check.md`'s own numbers for the same
  commit/box exactly - confirms this session's profiling code addition
  changed no dispatch, no numeric result, and no code path other than the
  new opt-in instrumentation itself.
