# Sonos MCP agent eval

**Tool result compaction:** every tool result is compacted before it
enters a `tool` message (`tools.rs`'s `format_tool_result`, the one
place both `agent.rs` and `web.rs` build it) — if the result, or any
string nested inside it (e.g. an MCP `content: [{"type": "text", "text":
"<json>"}]` item), parses as JSON, that JSON is re-serialized without
newlines or indentation before the whole result is compact-encoded into
the message. This matters because some MCP servers (Sonos's included)
return pretty-printed JSON text: `get_households_and_groups_and_players`
costs 394 tokens pretty vs 241 compact for the canned fixture (see
`crates/llm-wasm/tests/tokenizer.rs`'s
`compacting_households_fixture_reduces_token_count`), and the gap is
larger on a real household with more players/groups — so
`prefix_tokens` and any `prefill_s`/token counts recorded below reflect
post-compaction sizes. A compacted result over `tools::MAX_RESULT_CHARS`
(8000 chars) is never truncated — kept in full — but logs a
`tracing::warn!` on the native side.

**Tool-schema token diet:** the tool preamble itself (the `tools/list`
schemas rendered into the system turn, separate from any tool *result*)
also carries avoidable tokens — `schemadiet.rs`'s `diet_tools` strips
JSON-Schema fields the model doesn't need to decide what a valid call is
(`annotations`, redundant `title`, `additionalProperties: false`, empty
`required: []`, `minLength: 1`, i64-extreme `minimum`/`maximum`) without
changing any property name, `required` list, `enum`, or `type`. Measured
against the real tokenizer + chat template
(`crates/llm-wasm/tests/schemadiet.rs`): the 13-tool Sonos preamble goes
from 2499 to 2382 tokens and the 34-tool one from 8129 to 7738 tokens.
Not wired into this eval's agent loop yet — see `docs/ENGINE.md`
§"Tool-schema token diet" for the hook-in point and level-2 (opt-in,
description-deduping) numbers.

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
- **Args match by subset**: a required call's actual arguments must
  contain every key in `expected.args` with the exact expected value;
  extra keys the model included beyond `expected.args` are ignored.
  This covers schema-required params the utterance doesn't determine
  (e.g. `play_artist`'s `music_service`, which is required but not
  named by "Play music by Nirvana in the kitchen." in `m11`) — any
  value, or omission where the schema allows it, is accepted for keys
  not listed in `expected.args`.
- **Max 6 steps** per utterance. A run that exceeds 6 tool calls
  without producing the expected sequence is scored incorrect
  regardless of whether it eventually would have gotten there.
- A read-only multi-step item (e.g. `m07`, "what's the volume in the
  bedroom?") is correct if its final read happens with the right args;
  there's no mutating call to check.
- **Unquantified volume nudges ("a bit", "a little", "monte le son"
  with no number) use a delta of ±10** via `adjust_group_volume` on
  the resolved group — `+10` for up, `-10` for down. This is a
  convention chosen for the eval, not a value derived from any Sonos
  default; see `s16`, `m23`, `m26`.
- **New accept modes, not yet scored — `pending_harness: true`:**
  `crates/llm-wasm/src/eval.rs`'s scorer does not yet implement these;
  items using them are marked `pending_harness: true` so the current
  scorer skips them (treat them the way `tools12_ok: false` items are
  skipped under the 12-tool set) until the harness is extended.
  - `accept: "no_mutation"` — the item's `expected` is `[]`; correct
    means **no** mutating tool call was made (an unresolvable target,
    e.g. a room that doesn't exist in the household — see `m21`). Read
    calls, or a plain text answer, are fine.
  - `accept: "any_play"` — the item's `expected` is `[]`; correct
    means any single `play_*` call was made (any target, any content)
    — used for fully ambiguous requests with no canonical answer, e.g.
    `m22`, "play something".
- **`lang: "fr"`** marks a French-language utterance; the expected
  call sequence is scored identically to an equivalent English item —
  see `m25`, `m26`.

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

## Run: xLAM-2-3b-fc-r Q4_K_M, 13 tools, native

| id  | correct | steps | prefill_s | decode_tok_s | total_s |
|-----|---------|-------|-----------|---------------|---------|
| ... |         |       |           |               |         |
```

## Files

- `utterances.json` — 43 utterances (17 `single`, 26 `multi`; 2 of the
  `multi` items are French, `lang: "fr"`), each with the expected
  tool-call sequence, canned-fixture argument values, accept mode,
  `tools12_ok` (solvable with the 13-tool subset in
  `fixtures/sonos/tools-12.json`; file still named tools-12.json), and
  notes. 2 items (`m21`, `m22`) use the new `no_mutation`/`any_play`
  accept modes and are marked `pending_harness: true` — see "Scoring
  rules" above.
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
  (`fixtures/sonos/tools.json`) or `12` for the 13-tool subset
  (`fixtures/sonos/tools-12.json`, filename unchanged); see `eval::ToolSet`.
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
