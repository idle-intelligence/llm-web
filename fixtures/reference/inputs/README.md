# Reference inputs

Three fixed `(messages, tools)` pairs used to generate the byte-for-byte /
logit-for-logit ground truth in `../rendered/` and `../logits/`. See
`scripts/export_reference.py`.

- `01_no_tools.json` — system + user, no tools (`"tools": []`).
- `02_tools_single.json` — system + user "pause the kitchen", with a 12-tool
  subset of the Sonos MCP tool list.
- `03_tools_multiturn.json` — same 12 tools; user asks to pause the kitchen;
  assistant calls `get_households_and_groups_and_players`; a `tool`-role
  message returns a small fake two-household/two-room JSON result (Kitchen,
  Living Room with ids). The conversation ends there so the next generation
  is the assistant's follow-up turn (expected: a `pause` call on the Kitchen
  group id).

## `_placeholder: true` — tool schemas are NOT the real Sonos MCP list

No recorded `tools/list` response was found (checked
`trucs.ai/.claude/worktrees/sonos-mcp/sonos/{NOTES.md,client-test.mjs,sonos-mcp.js,index.html}`
read-only; NOTES.md mentions "34 actions" but no schema JSON is captured
anywhere in that worktree). The 12 tool schemas embedded in `02_*` and `03_*`
were written by hand to be plausible and are named after tools referenced in
that worktree's prose/code (`get_households_and_groups_and_players`, `pause`,
`resume`, `set_group_volume`, `get_now_playing`, `skip_to_next_track`,
`play_sonos_favorite`, `get_sonos_favorites`, `set_group_mute`,
`adjust_group_volume`, `get_group_volume`, `play_radio`). Parameter naming
(`group_id`, `household_id`, snake_case) follows the one concrete data point
found — `index.html` calls tools with `{ group_id: ... }` — but descriptions,
types, and the exact parameter set are invented.

**These must be replaced with the real recorded `tools/list` output** once
one is captured from the actual Sonos MCP server, and the reference
rendering/logits in `../rendered/` and `../logits/` regenerated from it.
