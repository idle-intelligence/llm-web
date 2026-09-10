# Sonos MCP agent eval

Dry-run tool-call eval for the Sonos LLM agent (Phase 3 of
`sonos/PLAN.md`). No live Sonos speakers involved — the agent runs
against the canned household in `fixtures/sonos/results/`, and each
tool call it makes is answered from a static JSON file for that tool
name (not per-argument), then scored offline against
`eval/utterances.json`.

## Scoring rules

- A run is **correct** if every tool call in the item's `expected` list
  that has a real side effect (anything other than a `get_*` read)
  happens, in order, with exactly the expected `args`.
- Reads (`get_*` calls) are **free under `accept: "subset"`**: the
  agent may call extra read tools (e.g. re-running
  `get_households_and_groups_and_players` to resolve a room name, or
  double-checking `get_now_playing`) without being penalized, as long
  as the required calls still happen with the right args. `expected`
  read steps (e.g. a lookup call, `args: {}`) describe the canonical
  path, not a hard requirement to reproduce verbatim.
- `accept: "exact"` means no extra calls beyond `expected` are
  tolerated — reserved for the handful of items where any additional
  call would be redundant on its face (e.g. a bare
  `get_households_and_groups_and_players` with nothing to look up).
- **Wrong room, wrong group/player id, or a missing required call is
  incorrect** — no partial credit.
- **Max 6 steps** per utterance. A run that exceeds 6 tool calls
  without producing the expected sequence is scored incorrect
  regardless of whether it eventually would have gotten there.
- A read-only multi-step item (e.g. `m07`, "what's the volume in the
  bedroom?") is correct if its final read happens with the right args;
  there's no mutating call to check.

## Metrics per row

One row per utterance per run. Record:

- `correct` — true/false per the rules above
- `steps` — number of tool calls actually made
- `prefill_s` — prefill time in seconds (includes the full rendered
  chat template: system prompt + tool schemas + utterance)
- `decode_tok_s` — decode throughput in tokens/second
- `total_s` — wall clock for the whole exchange (prefill + all decode
  steps + any intermediate re-prefills for tool results)

## Results file convention

`eval/results/<date>.md`, `YYYY-MM-DD` (one file per eval run, not
appended to). Data only, research-log style — no analysis or
narrative, that goes in `sonos/NOTES.md` or the draft post.

Structure:

1. A parameters block: model (name + quant + GGUF file), tool count
   (34 or 12), native or browser, hardware, commit hash of `llm-web`
   at run time, date.
2. One table per (model, tool count, native|browser) combination, rows
   = utterance ids (`s01`..`s10`, `m01`..`m10`), columns = `correct`,
   `steps`, `prefill_s`, `decode_tok_s`, `total_s`.

Example skeleton:

```markdown
# Sonos MCP agent eval — 2026-09-15

## Run: xLAM-2-3b-fc-r Q4_K_M, 34 tools, native

- hardware: M2 16 GB
- commit: <sha>

| id  | correct | steps | prefill_s | decode_tok_s | total_s |
|-----|---------|-------|-----------|---------------|---------|
| s01 | true    | 1     | 0.42      | 48.1          | 0.9     |
| ... |         |       |           |               |         |

## Run: xLAM-2-3b-fc-r Q4_K_M, 12 tools, native

| id  | correct | steps | prefill_s | decode_tok_s | total_s |
|-----|---------|-------|-----------|---------------|---------|
| ... |         |       |           |               |         |
```

## Files

- `utterances.json` — the 20 scored utterances (10 `single`, 10
  `multi`), each with the expected tool-call sequence, canned-fixture
  argument values, accept mode, `tools12_ok` (solvable with the
  12-tool subset in `fixtures/sonos/tools-12.json`), and notes.
- `results/<date>.md` — one file per eval run (see above).

## How to run

The scoring engine itself (`load_cases`, `select_tools`, `run_case`,
`run_all`, `render_markdown`) lives in `crates/llm-wasm/src/eval.rs`.
Wiring it up to the real (Burn+wgpu) `Generator` behind an `llm-agent`
CLI subcommand is the engine worker's job, not this crate's — once
wired, the intended invocation is:

```sh
llm-agent eval \
  --gguf /path/to/xLAM-2-3b-fc-r.Q4_K_M.gguf \
  --tools all|12 \
  --label <name> \
  --system "<system prompt>" \
  --out eval/results/<date>.md
```

- `--gguf` — path to the quantized model to eval.
- `--tools` — `all` for the full 34-tool schema
  (`fixtures/sonos/tools.json`) or `12` for the 12-tool subset
  (`fixtures/sonos/tools-12.json`); see `eval::ToolSet`.
- `--label` — a short name for this run, folded into the default output
  filename (`eval/results/<date>-<tools>-<label>-native.md`) and recorded
  in the report's parameters block. Defaults to `default`.
- `--system` — override the agent's system prompt (default: "You are a
  helpful home assistant with access to Sonos speaker controls."),
  recorded verbatim in the parameters block.
- `--max-new-tokens` / `--max-steps` / `--max-ctx` — generation and
  agent-loop limits (defaults 256 / 6 / 12288), also recorded in the
  parameters block.
- `--out` — where to write the rendered markdown (`eval::render_markdown`'s
  output); defaults to `eval/results/<date>-<tools>-<label>-native.md`.

The rendered report's parameters block also records the tool count, the
token length of the constant system+tools prefix (`prefix_tokens`), the
commit hash, and the machine — see `eval/results/2026-09-10-*-native.md`
for examples.

Until that subcommand exists, `crates/llm-wasm/tests/eval.rs` exercises
the same `eval.rs` API end to end with a scripted `FixtureGenerator`
standing in for the real model.
