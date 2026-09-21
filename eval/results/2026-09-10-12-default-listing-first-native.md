# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=default-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2218

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 94.443 | 5.39 | 104.093 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 17.940 | 5.28 | 42.203 | matched expected sequence exactly |
| s03 | true | 3 | 40.998 | 5.46 | 59.514 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | true | 3 | 37.334 | 5.56 | 48.507 | all required (mutating) calls made in order with correct args |
| s07 | false | 2 | 5.071 | 6.02 | 10.067 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | true | 3 | 37.089 | 5.64 | 47.381 | all required (mutating) calls made in order with correct args |
| s09 | true | 5 | 83.774 | 5.56 | 103.203 | all required (mutating) calls made in order with correct args |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 37.456 | 5.63 | 47.414 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 38.019 | 5.58 | 50.766 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 38.098 | 5.53 | 49.693 | all required (mutating) calls made in order with correct args |
| m04 | false | 2 | 5.204 | 5.87 | 11.004 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 39.116 | 5.47 | 53.930 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 2 | 5.498 | 5.80 | 11.542 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 38.933 | 5.44 | 52.376 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 38.098 | 5.58 | 49.770 | all required (mutating) calls made in order with correct args |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |

**Summary**: correct 66.7% (15 scored, 5 skipped) — mean_steps 2.80, mean_prefill_s 37.138, mean_decode_tok_s 5.59, mean_total_s 49.431

- commit: ffefd4e
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
