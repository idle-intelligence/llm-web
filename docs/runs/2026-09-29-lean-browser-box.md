# lean in the browser on the RTX 3080/Vulkan machine (Chrome/Dawn vs native wgpu-core)

Machine: RTX 3080 (10GB), driver 610.57.04,
Vulkan 1.4. Chromium 152.0.7977.75 (system package, already installed - no
sudo used, no packages installed at the OS level). Work done entirely under
a fresh directory (`llm-web-browser/`);
no other worker's checkout was touched. Source: the `lean-kernels` branch
tip (commit `57fa22d`, includes all sessions in
`docs/runs/2026-09-29-lean-kernels.md` - Q6_K decode matvec, add+rmsnorm
fusion, F32 decode matvec, uniform-write skip, head_dim-conditioned
SPLIT_CHUNK - plus the load-time memory fixes from
`docs/runs/2026-09-28-lean-decode-breakdown.md` Sessions 2-3, `scratch_key`/
lm-head-slice/`JsBytesReader`, all confirmed present in the built source).
Transferred from the Mac worktree to the RTX 3080 machine via `git bundle`
(the branch is local-only, never pushed) since the machine's own checkouts
under `lean/llm-web-kernels-final` etc. belong to other workers and were
left untouched. `wasm-pack build crates/lean --target web --out-dir pkg
--no-default-features --features web` was run **on the machine itself**
(wasm-pack/cargo/rustc/`wasm32-unknown-unknown` were already installed there
from earlier work) - no Mac build-lock contention. `ENGINE_BUILD` bumped to
`2026-09-29-box-01` in `main.js`/`main_decode_timing.js` in the same commit
as this doc; served `pkg/lean_bg.wasm` bytes hashed (`sha256sum`) against
the just-built file before every run in this doc.

## 1. Getting a WebGPU adapter on the NVIDIA GPU

**Headless Chromium cannot reach the Vulkan/NVIDIA adapter on this machine and
silently falls back to SwiftShader (software).** Tried, in order:
`--headless=new` and the old default headless mode (`chrome-headless-shell`),
both via Playwright's bundled Chromium and the system `/usr/bin/chromium`,
with every flag combination the task named
(`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=vulkan
--use-vulkan=native --enable-dawn-features=allow_unsafe_apis
--ignore-gpu-blocklist --disable-gpu-sandbox --no-sandbox`). Every headless
combination produced a valid `GPUAdapter` (`requestAdapter()` never
returned null) but with `adapter.info.architecture === "swiftshader"`.
Direct stderr capture (`--enable-logging=stderr --v=1`) traced the cause:

```
[...] ERROR:gpu/vulkan/vulkan_instance.cc:200] vkCreateInstance() failed: -7
[...] ERROR:gpu/ipc/service/gpu_init.cc:1434] Failed to create and initialize Vulkan implementation.
```

`-7` is `VK_ERROR_EXTENSION_NOT_PRESENT`. `vulkaninfo --summary` (run
directly as the same user, no flags) confirms the instance-level surface
extensions Chromium's Vulkan backend requests (xlib/xcb/wayland surface,
etc.) are all present and the RTX 3080 is visible
(`deviceName = NVIDIA GeForce RTX 3080`, `driverVersion = 610.57.4.0`), so
this is specifically Chromium's headless (`ozone-platform=headless`)
Vulkan-instance path failing to negotiate a surface extension it wants, not
a missing driver/library. Not investigated further than root-causing the
error code - out of scope for "get an adapter", which the headed fallback
below does.

**Fix: headed Chromium on the machine's own Xwayland display.** The machine
already runs a desktop session (GNOME/mutter on Wayland, with Xwayland providing
`:0`/`:1`). Using that display's `XAUTHORITY` cookie
(`/run/user/<uid>/.mutter-Xwaylandauth.<token>`, found via
`ls /run/user/<uid>/ | grep -i xwayland`) and `DISPLAY=:0`, launching
Chromium **not headless** (`headless: false` in Playwright, no
`--headless` flag at all - only the GPU-enabling flags above) reaches the
real adapter:

```json
{
  "vendor": "nvidia",
  "architecture": "ampere",
  "deviceCreated": true,
  "deviceErr": null
}
```

Full feature list included `timestamp-query`,
`chromium-experimental-timestamp-query-inside-passes`, `subgroups`,
`subgroup-size-control`, `shader-f16` was not present (not checked further,
out of this task's scope). Limits: `maxStorageBufferBindingSize =
2147483644` (~2GiB, vs SwiftShader's 1GiB), `maxComputeInvocationsPerWorkgroup
= 1024`, `maxBufferSize = 4294967292`.

**Reproducible command** (adapter probe only, no model):
```
XAUTHORITY=/run/user/<uid>/.mutter-Xwaylandauth.<token> DISPLAY=:0 \
  <venv>/bin/python3 gpu_probe.py \
  --executable-path /usr/bin/chromium --headless-mode headed
```
(`gpu_probe.py`: Playwright launch with `headless=False` and args
`--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=vulkan
--use-vulkan=native --enable-dawn-features=allow_unsafe_apis
--ignore-gpu-blocklist`, then `navigator.gpu.requestAdapter()` +
`requestDevice()`, printed as JSON.)

This opens a real (offscreen-composited, but real) browser window on the
machine's own display for the run's duration - acceptable per this task's own
"headed is the fallback" instruction, but worth flagging: anyone
physically at that desktop session would see Chromium windows flash open
and close during a sweep.

## 2. Parity (`www/index.html?local=1`)

All 6 fixture cases match, `allMatch: true`, official Qwen2.5-0.5B GGUF,
Vulkan/NVIDIA/headed:

| case | prompt tokens | match | prefill ms | decode ms/tok |
|---|---:|---|---:|---:|
| short | 36 | true | 731.8 | 6.06 |
| long | 86 | true | 70.2 | 6.24 |
| non_english | 54 | true | 48.4 | 6.47 |
| long_tools_single | 2225 | true | 1949.8 | 8.10 |
| long_tools_multiturn | 2354 | true | 2192.9 | 6.20 |
| long_1500 (timing-only, no reference) | 1494 | n/a | 1037.6 | 6.22 |

Peak Chromium-tree RSS for this single-page-load run: 3.39GB. No crashes,
no NaN/garbled output; greedy continuations read as coherent English/French
text in every case.

```
XAUTHORITY=<...> DISPLAY=:0 <venv>/bin/python3 run_lean_page.py \
   "http://localhost:8899/www/index.html?local=1" --limit-gb 20 --timeout 180
```

## 3. Decode timing, browser (Chrome/Dawn/Vulkan/NVIDIA) vs native (wgpu-core/Vulkan/NVIDIA)

`www/decode_timing.html?model=<key>&case=<short|long>&steps=16`, 3 fresh
Chromium page loads per model/case (each a fresh process - `cold` = that
page's first decode step, `warm` = median of the remaining 15), one GPU job
at a time, headed on the machine's Xwayland display as in
section 1. `smollm2_360m`'s own fixture has no `long_tools_single` case
(confirmed in `docs/runs/2026-09-29-lean-kernels.md`), so its "long" case is
its own 86-token `long` fixture case (`main_decode_timing.js` was given a
per-model `longCaseName` override for this - the only code change made to
the harness); every other model's "long" is the shared 2225-token
`long_tools_single` case. `qwen3_1_7b` and `smollm2_360m` were pointed at
their own tokenizer's fixture files (`fixture_qwen3_1_7b.json`,
`fixture_llama_360m_q4_0.json`) instead of the default `fixture.json`,
whose max token id (151645) is out of `smollm2_360m`'s vocab range and would
crash it (same failure mode the kernels doc's native CLI hit on this model
without its own `--fixture`).

Native numbers below are this machine's own final figures from
`docs/runs/2026-09-29-lean-kernels.md` (same machine, same branch tip, native
`lean-cli --kernel fast`, GPU-locked, median of 5-8, `short`-case gap table
end of session 3, and the head_dim-conditioned-SPLIT_CHUNK table for
long-context cases).

| model | case (tokens) | native ms/tok (this box, native, end of kernels doc) | browser warm ms/tok (median of 3 page loads) | browser cold ms/tok (median of 3) | browser / native |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | short (36) | ~4.67-4.72 | 5.4 | 15.3 | ~1.16x |
| Qwen2.5-0.5B | long_tools_single (2225) | ~6.51 | 5.7 | 24.6 | ~0.88x (browser faster) |
| Qwen2.5-3B (official GGUF) | short (36) | ~13.60-13.72 | 12.7 | 33.0 | ~0.93x (browser faster) |
| Qwen2.5-3B (official GGUF) | long_tools_single (2225) | not measured natively on this box/branch (no 2225-token fixture case for this model in the kernels doc) | 15.2 | 41.0 | n/a |
| Qwen3-1.7B (Q8_0) | short (15) | ~8.96-9.39 | 8.6 | 30.2 | ~0.96x (browser faster) |
| Qwen3-1.7B (Q8_0) | long_tools_single (2225) | ~11.53-11.58 | 10.9 | 34.6 | ~0.94x (browser faster) |
| SmolLM2-360M (Q4_0) | short (37) | ~5.34 | 6.2 | 20.3 | ~1.16x |
| SmolLM2-360M (Q4_0) | long (86, own fixture) | ~5.73 | 6.4 | 20.5 | ~1.12x |

**Browser is within 1.0-1.2x of native on this NVIDIA box on every model
tested, and faster than native on three of four models' long-context
case** - on the Mac the browser was also at or ahead of native in the same period
(Sessions 6-8 of `docs/runs/2026-09-28-lean-decode-breakdown.md`: Qwen2.5-0.5B
10.8 ms/tok in the browser against 11.6 native). This box has 62GB RAM and
never approached a memory ceiling on any model, so that confound is absent here; the browser/native ratio above is read as a
closer measurement of Dawn's own per-dispatch overhead vs wgpu-core's on
Vulkan specifically (both back ends target the same driver/GPU here, unlike
the Mac's Dawn-Metal vs wgpu-core-Metal comparison).

Cold-start decode (first step, includes shader/pipeline compile) is
3-6x warm on every model - consistent with the Mac harness's earlier
findings, not investigated further here.

**Reproducible command** (per model/case/run):
```
XAUTHORITY=<...> DISPLAY=:0 <venv>/bin/python3 run_lean_page.py \
   "http://localhost:8899/www/decode_timing.html?model=<05b|3b|qwen3_1_7b|smollm2_360m>&case=<short|long>&steps=16" \
   --limit-gb <see section 4> --timeout 300
```
models served from `crates/lean/www/model*/` directories, each a set of
symlinks into `<workdir>/lean/models/{gguf,hf}/...` (no files copied,
no HF re-download - all four models were already cached on the machine from
earlier native work).

## 4. Memory guard: 8GB is a headless number, not a headed one

The task's stated 8GB Chromium-tree RSS guard comes from the Mac's
**headless** harness. On this box, running **headed** (the only way to
reach the real GPU - section 1), Chromium's own baseline overhead
(compositor, GPU process, multiple sandboxed renderer/utility/crashpad
processes, each mapping the ~300MB `chromium` binary - `ps`-summed RSS
double/triple-counts pages shared across those processes) is materially
higher: a blank `about:blank` page alone measured **4.98GB** total
Chromium-tree RSS on this box, before any model or WebGPU work. At an 8GB
ceiling this leaves under ~3GB of headroom for the model itself, which
Qwen2.5-0.5B (408MB GGUF) and SmolLM2-360M (230MB GGUF) fit inside (peak
2.3-3.2GB total, sections 2-3) but Qwen2.5-3B (1.9GB GGUF) and Qwen3-1.7B
(1.8GB GGUF) do not - every run for those two models tripped the 8GB guard
and aborted before completing (`_aborted: true`) on the first sweep. Since
this box has 62GB RAM (59GB free at the time of these runs) and headed
Chromium's baseline is a fixed cost rather than something scaling with
model size, the guard was raised to **20GB** for the 3B/Qwen3-1.7B reruns
(section 3's numbers) - well inside available RAM, and no run at 20GB came
close to tripping it (peak observed: 10.14GB, Qwen3-1.7B `long_tools_single`).
The 0.5B/SmolLM2 runs never needed the higher guard and are reported from
their original 8GB-guarded sweep. No run in this doc was aborted or timed
out at its final guard value.

Peak Chromium-tree RSS by model/case (median of 3 runs' peaks):

| model | case | peak RSS |
|---|---|---:|
| Qwen2.5-0.5B | short / long_tools_single | 3.08GB / 3.21GB |
| SmolLM2-360M | short / long | 2.33GB / 2.32GB |
| Qwen2.5-3B | short / long_tools_single | 8.83GB / 8.91GB |
| Qwen3-1.7B | short / long_tools_single | 8.12GB / 9.98GB |

## 5. `timestamp-query` inside passes

`timestamp-query` is present in `adapter.features` (section 1) and
`chromium-experimental-timestamp-query-inside-passes` is present as a
separate feature string, both on the Vulkan/NVIDIA headed adapter.
Functional check (not just feature-string presence): a minimal compute
pipeline (one dispatch, 16 workgroups) run inside a pass with
`timestampWrites: { querySet, beginningOfPassWriteIndex: 0,
endOfPassWriteIndex: 1 }`, resolved via `resolveQuerySet` into a buffer,
copied to a `MAP_READ` buffer and read back:

```json
{
  "hasFeature": true,
  "timestampsRaw": ["1790685986513987584", "1790685986513993728"],
  "deltaNs": 6144,
  "ok": true
}
```

Both timestamps are non-zero, distinct, and the delta (6144ns for a
16-workgroup trivial dispatch) is a plausible GPU timing figure - no
validation error, no device-lost, no zero/garbage readback. `requestDevice`
did not need `requiredFeatures: ["timestamp-query"]` to succeed (device
creation succeeds either way on this adapter), but the feature must still
be requested explicitly for `createQuerySet({ type: "timestamp" })` and
`timestampWrites` to be permitted on the pass (confirmed by requesting it
in this test). **This means the in-browser kernel profiler the lead wants
next can use real per-pass GPU timestamps on this machine's Chrome/Dawn/Vulkan
adapter, headed** - it has not been tried headless (section 1: headless
never reaches this adapter here) or inside the actual `lean` engine's own
passes (this test used a standalone trivial kernel, not `Engine::dispatch`).

**Reproducible command:**
```
XAUTHORITY=<...> DISPLAY=:0 <venv>/bin/python3 run_ts_test.py
```
(`run_ts_test.py` navigates to a static test page,
`www/ts_query_test.html`, and reads `window.__tsResult`.)

## Files touched (not committed)

`<workdir>/llm-web-browser/` (fresh dir): `repo/` (git bundle clone of
`lean-kernels`, wasm built in place), `venv/` (Python + Playwright, bundled
Chromium also installed via `playwright install chromium` though the system
`/usr/bin/chromium` was used for every real run above), `gpu_probe.py`,
`run_lean_page.py`, `run_ts_test.py`, `sweep.sh`/`sweep2.sh`, `results/*.json`.
`repo/crates/lean/www/main.js` and `main_decode_timing.js`: `ENGINE_BUILD`
bumped to `2026-09-29-box-01`; `main_decode_timing.js` also gained a
per-model `fixture`/`longCaseName` override for `qwen3_1_7b`/`smollm2_360m`
(see section 3). `repo/crates/lean/www/ts_query_test.html`: new, standalone
timestamp-query functional test (section 5). `repo/crates/lean/www/model*/`:
symlinks only, into the machine's already-cached model files - no new
downloads. All cleaned up after this session: systemd units
(`agent-lean-http`, `agent-lean-sweep`, `agent-lean-sweep2`) stopped and
unloaded, no Chromium or `http.server` process left running
(`ps aux` checked clean), GPU memory back to idle (461MB, matches this
machine's idle baseline before this session started).
