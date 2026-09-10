# Reference export scripts

venv mon ami. Python 3.12 (torch has no 3.14 wheels yet on this machine;
`/opt/homebrew/bin/python3.12` was used, not the default `python3`).

## Setup

```
cd /Users/tc/Code/idle-intelligence/llm-web
/opt/homebrew/bin/python3.12 -m venv scripts/.venv
scripts/.venv/bin/pip install -r scripts/requirements.txt
```

## Build the input fixtures

```
scripts/.venv/bin/python scripts/make_inputs.py
```

Regenerates the `tools` array of `fixtures/reference/inputs/{02_tools_single,
03_tools_multiturn,04_tools_all}.json` from the real Sonos MCP schemas
(`fixtures/sonos/tools-12.json` for 02/03, `fixtures/sonos/tools.json` for
04), converting each MCP `{"name", "description", "inputSchema"}` entry
verbatim into the transformers/OpenAI function-calling shape
`{"type": "function", "function": {"name", "description", "parameters"}}`.
Messages are left untouched (read from the existing files, not generated).
`01_no_tools.json` has no tools and isn't touched by this script.

## Generate the reference fixtures

```
scripts/.venv/bin/python scripts/export_reference.py
```

Reads `fixtures/reference/inputs/{01_no_tools,02_tools_single,03_tools_multiturn}.json`,
runs the full pipeline (render + tokenize + forward pass + logits + greedy
decode) for each, writes:

- `fixtures/reference/rendered/<name>.txt` — exact `apply_chat_template(...,
  tokenize=False)` output, no trailing newline appended.
- `fixtures/reference/rendered/<name>.tokens.json` — token ids for that
  string.
- `fixtures/reference/logits/<name>.json` — seq_len, dtype/device, argmax
  token id per position, top-5 (id, logit) at the last position, and the
  greedy-decoded (`do_sample=False`) first 32 generated tokens as text +
  ids.
- `/Users/tc/Code/idle-intelligence/models/reference/xlam-2-3b-fc-r/<name>.logits.npy`
  — full-sequence float32 `[seq_len, vocab_size]` forward-pass logits.
  Outside the repo: multi-hundred MB, not committed.

Device/dtype: MPS with bfloat16 if available, else CPU bfloat16. float32 on
MPS was tried first and thrashed into swap on this 16GB M2 (12GB of float32
weights left no headroom; measured ~1 min of actual CPU progress after 25
min wall time) — switched to bfloat16 (~6GB weights) for both MPS and CPU so
all three fixtures share one dtype. The script prints which it picked; the
same info is recorded per-fixture in `fixtures/reference/logits/<name>.json`.

## Render-only (no forward pass, no logits)

```
scripts/.venv/bin/python scripts/export_reference.py --render-only 04_tools_all
```

Loads only the tokenizer (no model weights, no forward pass) and writes just
`fixtures/reference/rendered/04_tools_all.{txt,tokens.json}`. Used for
`04_tools_all.json` (all 34 Sonos MCP tools): at ~8k prefill tokens a full
forward pass would produce a `[8140, 151936]` float32 logits.npy
(multi-GB) purely to check how prefill size scales with tool count — not
worth generating. `--render-only` with no name defaults to
`RENDER_ONLY_NAMES` in the script (currently just `04_tools_all`).

## Model-vs-port parity check (HF greedy generate vs. `llm-agent run`)

```
scripts/.venv/bin/python scripts/parity_eval.py hf    # one model load, all cases
scripts/.venv/bin/python scripts/parity_eval.py port  # sequential llm-agent run invocations
```

Purpose: isolate whether the Sonos-eval score (`eval/results/2026-09-10-summary.md`,
~25% correct, mostly from inventing ids instead of calling the listing tool
first) traces to the model (xLAM-2-3b-fc-r itself) or the port (Burn+wgpu,
Q4_0 GGUF) — by comparing HF bf16 MPS greedy `generate()` against
`llm-agent run` on the byte-identical rendered/tokenized prompt for a
handful of cases, bypassing the eval harness's multi-step/tool-execution/
prefix-cache machinery entirely.

`hf` stage: for a fixed list of case ids from `eval/utterances.json`, builds
`[system, user(utterance)]` + the 12-tool set (`fixtures/sonos/tools-12.json`,
via `make_inputs.mcp_to_function_tools`), renders with
`apply_chat_template(add_generation_prompt=True)`, tokenizes the same way
`export_reference.py` does (`tokenizer(rendered, add_special_tokens=False)`),
writes tokens to `/Users/tc/.claude/jobs/ae1e446d/tmp/parity/<id>.tokens.json`,
then runs one HF bf16 MPS greedy `generate()` per case (one model load for
all cases), writing `<id>.hf.json`. Frees the model (`del` + `gc.collect()` +
`torch.mps.empty_cache()`) before returning.

`port` stage: runs `./target/release/llm-agent run --gguf <default-gguf>
--tokens <id>.tokens.json --max-new 64 --tokenizer <model-dir>/tokenizer.json`
once per case as a separate process, writing `<id>.port.json`. Run this
stage only after the `hf` stage's model is fully freed (they must never
share memory at once on a 16GB M2 — bf16 HF is ~6GB, the port's wgpu backend
is ~3GB GPU).

Results, method, and the model-vs-port conclusion:
`eval/results/2026-09-10-parity.md`.

## Generate the Q4_0-dequant-matched reference

```
scripts/.venv/bin/python scripts/export_reference_dequant.py
```

Purpose: the Rust (Burn/wgpu) port loads the Q4_0 GGUF and dequantises to
f32 at load time, then computes in f32 — it is not comparable byte-for-byte
to `export_reference.py`'s bf16-on-MPS reference, which uses the original
unquantised weights. This script builds a *third* reference that isolates
quantisation noise from port bugs: it loads
`Salesforce/xLAM-2-3b-fc-r` in plain float32 on CPU, then overwrites every
parameter in place with the exact Q4_0-dequantised (Q6_K for
`token_embd.weight`) values read straight out of
`/Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf`
via the `gguf` package's `GGUFReader` + `gguf.quants.dequantize`. The
resulting model's forward pass should track the Rust port far more closely
than the bf16 reference does — any remaining discrepancy vs. the Rust port
is then attributable to the port itself, not quantisation.

**Memory: run this alone.** It holds a full float32 copy of the 3B model on
CPU (~12 GB) for the duration of the run — no MPS/bf16 headroom trick
available here since the whole point is exact f32 compute. Loading uses
`low_cpu_mem_usage=True` and each GGUF tensor is dequantised, copied into
the already-allocated model parameter in place, and freed before moving to
the next tensor, so the transient overhead beyond the base ~12 GB is at
most one tensor at a time (largest: the ~1.2 GB embedding table). Do not
run this alongside another GPU/CPU-heavy job (e.g. a GPU worker sharing
this machine's unified memory) — pick a quiet moment.

Default run (fixture `01_no_tools` only, no hidden states, ~1-2 min once
the model is loaded):

```
scripts/.venv/bin/python scripts/export_reference_dequant.py
```

Also run 02/03 and dump per-layer hidden states (02/03 add a much longer
f32-on-CPU prefill — `03_tools_multiturn` is 2225 tokens, expect 5-10
minutes for that one fixture alone; `--hidden` adds negligible time):

```
scripts/.venv/bin/python scripts/export_reference_dequant.py \
    --inputs 01_no_tools 02_tools_single 03_tools_multiturn --hidden
```

Writes, per fixture `<name>`:

- `fixtures/reference/logits/<name>.dequant.json` — same shape as
  `export_reference.py`'s `<name>.json` (seq_len, argmax per position,
  top-5 at the last position, greedy 32-token continuation), plus a
  `"weights"` field noting this is the dequant-matched run.
- `models/reference/xlam-2-3b-fc-r/<name>.dequant.logits.npy` — full
  `[seq_len, vocab_size]` float32 logits. Outside the repo, not committed.
- `models/reference/xlam-2-3b-fc-r/<name>.dequant.hidden.npz` (`--hidden`
  only) — last-position hidden states at layers {0, 9, 18, 27} (raw,
  pre-next-layer-norm), `layer_35_raw` (raw output of the final
  transformer layer, captured via a forward hook since
  `output_hidden_states`'s last tuple entry is post-final-norm, not raw),
  and `final_normed` (after `model.norm`). For the Rust side to compare
  per-layer later.

After each fixture's forward pass, the script also prints a comparison
against the existing bf16 reference (`fixtures/reference/logits/<name>.json`
and, if present outside the repo, `<name>.logits.npy`): per-position argmax
agreement (`n/seq_len`) and cosine similarity at the last position. This
number is the load-bearing one — it tells us how much of the Rust port's
argmax mismatch against the bf16 reference is explainable by Q4_0
quantisation alone, versus a port bug.
