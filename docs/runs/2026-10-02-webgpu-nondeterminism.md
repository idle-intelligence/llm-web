# 2026-10-02: WebGPU backend not token-exact and not deterministic in Chromium on the RTX 3080

Raw results: `docs/runs/data/2026-10-02-webgpu-nondeterminism/` (one JSON per
page load, named as in the tables below).

## Symptom

`www/backends.html?backend=webgpu`, Qwen2.5-0.5B-Instruct Q4_0, fixture
"short" prompt (36 ids), 64 greedy tokens, ENGINE_BUILD
`2026-10-02-backends-03`: two runs in Chromium on the RTX 3080 test machine
(Vulkan) gave two different hashes, neither equal to the transformers
hash `a454748c60e23841`. The threads and single backends on the same machine,
Chrome on the M2 laptop (Metal) and native lean on the 3080 (wgpu, Vulkan)
were exact.

## Setup

- RTX 3080 test machine: Chromium 152.0.7977.75, headed on the
  desktop's X display, flags `--enable-unsafe-webgpu
  --enable-features=Vulkan,WebGPU --use-angle=vulkan --use-vulkan=native
  --enable-dawn-features=allow_unsafe_apis --ignore-gpu-blocklist`, a fresh
  browser profile per page load. The machine's training job was paused
  (SIGSTOP, GPU at 0 % before every batch) and resumed (SIGCONT, GPU back at
  92-100 %) after every batch.
- Pages served with COOP/COEP by `scripts/serve_coi.py`, `?local=1` (model
  from disk). Served wasm hashed with `curl | sha256sum` against the build
  before each batch.
- M2 laptop: Chrome for Testing (Playwright), Metal, under the GPU lock.
- Probe page: `www/nondet.html` (added in this session, debug only). It
  hashes prefill logits and the KV snapshot `trials` times, runs `steps`
  teacher-forced decode steps (`appendTokens` of each reference id, full
  logits back) per trial, `?ops=1` returns per-op checksums of one prefill
  (`debugPrefill`), `?verify=1` reads every uploaded buffer back after load
  and after the runs (`debugVerifyUploads`).

## 1. Regression or pre-existing

| build | wasm sha256 (prefix) | loads | exact | hashes of the others |
|---|---|---:|---:|---|
| 7bb5049 (before the diagnostics commit), ENGINE_BUILD backends-02 | ff5b14de | 8 | 3 | 33e33e67, d85cab4f, 06148a58, 6aac6dca, a6a193c0 |
| 539451c (HEAD), ENGINE_BUILD backends-03 | 3acc9382 | 5 | 4 | 7b5fb645 |

Files: `old7bb_run1..8.json`, `head539_run1..5.json`. The 7bb5049 pkg was
built from `git archive 7bb5049` with its own target directory (a shared one
reused the HEAD artifact because the archived files carry older mtimes; that
first build gave the HEAD hash 3acc9382 and was discarded).

## 2. Capability-driven selection

lean requests no optional feature for compute (only `TIMESTAMP_QUERY` when
diagnostics or profiling ask for it) and asks for the adapter's own limits.
Kernel choice in `model.rs::linear` depends on the weight dtype, `rows` and
the fast flag; attention on `head_dim` and `kv_len`. The only limit that
changes anything is `maxStorageBufferBindingSize` (row chunking in
`quant.rs::chunk_rows`); the largest buffer, `output.weight` Q8_0 qs, is
136,134,656 bytes and fits one binding on all three below. No kernel uses
subgroups or f16. Largest workgroup storage: `linear_q4_tiled.wgsl`, 16 KiB.

| field | Chromium, 3080 | native wgpu, 3080 | Chrome, M2 |
|---|---|---|---|
| backend | BrowserWebGpu | Vulkan, NVIDIA 610.57.04 | BrowserWebGpu |
| device features requested | none | none | none |
| shader-f16 on adapter | no | (n/a) | yes |
| subgroup size | n/a | 32/32 | n/a |
| maxStorageBufferBindingSize | 2147483644 | 2147483647 | 4294967292 |
| maxStorageBuffersPerShaderStage | 16 | 1048576 | 10 |
| maxComputeWorkgroupStorageSize | 49152 | 49152 | 32768 |
| maxComputeInvocationsPerWorkgroup | 1024 | 1024 | 1024 |
| minStorageBufferOffsetAlignment | 256 | 32 | 256 |
| lm_head chunks | 1 | 1 | 1 |

Sources: the Chromium row is
`verify1.json` `adapter`, the native row `native_3080_lean_cli.log` (lean-cli
with `LEAN_DEBUG_ADAPTER=1`), the M2 row `m2_nondet_probe.json` `adapter`. All
three run the same kernels.

## 3. Localisation

| probe | loads | result | files |
|---|---:|---|---|
| 6 prefills + 6 teacher-forced decodes in one load | 1 | all 6 prefill logits hashes equal (df3cf597), KV equal, every decode step's logits equal across trials | `probeA.json` |
| same, 2 trials per load | 5 | prefill logits hash different on every load: 96649935, c3770e70, 4d3b275d, 71132171, b4231ff7 | `probeB1..5.json` |
| per-op checksums of one prefill, compared across loads | 4 | first differing op is a K or V projection output, one or a few whole output columns (all 36 rows): layer5.v col 127, layer5.k cols 109-120, layer11.v col 118 | `ops_per_load_summary.json` |
| read back all 560 uploaded buffers (479,536,128 bytes) after load | 4 | mismatches on 3 loads: blk.7.attn_q scales + blk.13.attn_v qs; blk.7.attn_v scales + blk.9.attn_v qs + blk.9.attn_output scales; blk.5.attn_k qs + blk.11.attn_output scales; none on 1 load (the one whose prefill hash is 96649935). Same list after the forward calls | `verify1..4.json` |

The engine is deterministic within a page load; what differs between loads is
the weight data on the device, before any forward pass runs. Corrupted
buffers are small (14,336 to 100,352 bytes) and different on every load.

## 4. Fix and checks

Change: `Engine::buf_f32`/`buf_u32`/`buf_uniform` and `Pool::uniform` create
the buffer and fill it with `queue.write_buffer` (`Engine::buf_upload`)
instead of `create_buffer_init` (mapped at creation). Before the fix:
`engine.rs:359,367,384` and `pool.rs:212` at 539451c.

| check | build | result | files |
|---|---|---|---|
| upload read-back + probe, write_buffer uploads | wasm 02656c0a | 6 of 6 loads: 0/560 mismatched, prefill logits hash 96649935 on all 6, teacher-forced argmax equal to the reference on all 63 steps | `wb1..6.json` |
| backends.html webgpu, Qwen2.5-0.5B | ENGINE_BUILD backends-04, wasm 98550820 | 5 of 5 a454748c60e23841 | `fix_qwen_webgpu_run1..5.json` |
| backends.html webgpu, SmolLM2-360M Q4_0 | backends-04 | 3 of 3 907b12597274deb6 (reference) | `fix_smol_webgpu_run1..3.json` |
| backends.html threads / single, Qwen2.5-0.5B | backends-04, pkg-mt a98c24de | a454748c60e23841 / a454748c60e23841 | `fix_qwen_threads_run1.json`, `fix_qwen_single_run1.json` |
| Chrome on the M2, webgpu, backends.html | backends-04, wasm 98550820 (served from 8795) | 2 of 2 "same as transformers: yes" | (console output) |
| Chrome on the M2, nondet.html, 2 trials | backends-04 | 0/63 teacher-forced steps differ from the reference, both trials | `m2_nondet_probe.json` |
| native gates on the M2 (release, `--ignored`) | 7a103f0 | fixture_parity 1/1, fixture_parity_qwen25_3b 1/1, fixture_parity_qwen3 1/1, fixture_parity_llama_360m 1/1, kv_snapshot 2/2, logit_mask 3/3 | `native_m2_gates.log` |
| native lean-cli fixture on the 3080 (Vulkan) | e09c0bc + write_buffer change | short, long, non_english, long_tools_multiturn: tokens match; long_tools_single: ids equal the fixture's 9 up to its EOS (151645), then lean-cli keeps generating and reports tok=false | `native_3080_lean_cli.log` |

## Observations

- Not a regression: 7bb5049 was wrong on 5 of 8 loads, HEAD on 1 of 5.
- No kernel differs between Chromium and native on the 3080, and no kernel
  is chosen from a capability that differs between them (section 2).
- Within one load, 6 prefills and 6 x 63 decode steps were bit-identical
  (`probeA.json`): no race inside or between kernels.
- Across loads, the first differing value was always a whole column of a K/V
  projection, and the upload read-back named the matching weight buffers on 3
  of 4 loads (`verify2..4.json`); the clean load gave the clean prefill hash.
- With `write_buffer` uploads, 6 of 6 loads read back clean and gave one
  prefill hash; after the rebuild, 5 of 5 Qwen and 3 of 3 SmolLM2 loads were
  token-exact.
- Decode ms/token in the fixed runs (5.76-5.99) sits next to the HEAD runs
  (5.93-6.03); these were not timing runs (no ABAB, load not controlled) and
  are not a timing claim.
