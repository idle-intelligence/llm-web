# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=default

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2218

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 70.917 | 6.81 | 77.540 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 12.331 | 6.37 | 36.372 | matched expected sequence exactly |
| s03 | false | 2 | 7.480 | 5.89 | 19.196 | final read get_sonos_favorites({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 2 | 6.078 | 5.19 | 12.445 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 5.917 | 5.35 | 11.536 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 2 | 5.458 | 5.58 | 10.671 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 2 | 5.038 | 5.86 | 9.826 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | false | 3 | 11.407 | 6.00 | 19.749 | missing required call pause({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m02 | true | 3 | 37.137 | 5.83 | 49.322 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 36.704 | 5.77 | 47.812 | all required (mutating) calls made in order with correct args |
| m04 | false | 2 | 4.891 | 6.18 | 10.239 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 38.197 | 5.73 | 52.343 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 2 | 5.395 | 6.00 | 11.241 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 4 | 60.109 | 5.58 | 78.586 | all required (mutating) calls made in order with correct args |
| m09 | false | 1 | 0.000 | 0.00 | 3.567 | agent error: failed to parse model output: output starts with '[' but is not a valid tool-call array: missing field `arguments` at line 1 column 50 |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |

**Summary**: correct 26.7% (15 scored, 5 skipped) — mean_steps 2.33, mean_prefill_s 20.471, mean_decode_tok_s 5.87, mean_total_s 30.030

- commit: 94420b3
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
