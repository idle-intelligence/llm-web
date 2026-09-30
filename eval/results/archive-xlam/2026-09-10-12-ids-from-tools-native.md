# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=ids-from-tools

- system: "You are a helpful home assistant with access to Sonos speaker controls. Identifiers such as group_id, player_id, household_id and favorite ids must come from tool results; if you do not have the identifier you need, first call the tool that lists them."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2257

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 98.567 | 5.52 | 107.998 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 18.524 | 5.13 | 44.486 | matched expected sequence exactly |
| s03 | false | 2 | 9.472 | 5.30 | 22.491 | final read get_sonos_favorites({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 2 | 5.688 | 5.40 | 11.808 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 5.643 | 5.68 | 11.466 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 3 | 12.284 | 5.70 | 22.128 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | true | 5 | 84.816 | 5.48 | 104.556 | all required (mutating) calls made in order with correct args |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | false | 3 | 11.761 | 5.81 | 20.374 | missing required call pause({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m02 | false | 1 | 0.000 | 0.00 | 3.859 | agent error: failed to parse model output: output starts with '[' but is not a valid tool-call array: missing field `arguments` at line 1 column 50 |
| m03 | false | 1 | 0.000 | 0.00 | 3.777 | agent error: failed to parse model output: output starts with '[' but is not a valid tool-call array: missing field `arguments` at line 1 column 50 |
| m04 | false | 2 | 5.469 | 5.46 | 11.708 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 40.052 | 5.30 | 55.159 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 2 | 5.604 | 5.82 | 11.633 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 4 | 63.039 | 5.39 | 82.179 | all required (mutating) calls made in order with correct args |
| m09 | false | 1 | 0.000 | 0.00 | 3.764 | agent error: failed to parse model output: output starts with '[' but is not a valid tool-call array: missing field `arguments` at line 1 column 50 |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |

**Summary**: correct 20.0% (15 scored, 5 skipped) — mean_steps 2.33, mean_prefill_s 24.061, mean_decode_tok_s 5.50, mean_total_s 34.493

- commit: 9884f99
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
