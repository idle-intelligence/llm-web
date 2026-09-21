# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=kernels-v2-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 50.925 | 9.06 | 56.677 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 13.082 | 8.58 | 32.104 | matched expected sequence exactly |
| s03 | true | 3 | 26.837 | 8.83 | 38.294 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 2 | 5.501 | 9.64 | 8.934 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 7.524 | 9.89 | 10.569 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | true | 3 | 26.481 | 9.25 | 32.766 | all required (mutating) calls made in order with correct args |
| s09 | false | 3 | 24.734 | 9.26 | 30.578 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 25.098 | 8.93 | 31.386 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 33.824 | 5.41 | 46.974 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 29.611 | 6.34 | 39.718 | all required (mutating) calls made in order with correct args |
| m04 | false | 2 | 5.746 | 7.24 | 10.318 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 32.897 | 6.35 | 45.678 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 2 | 7.481 | 6.53 | 12.858 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 36.349 | 5.82 | 48.752 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 38.500 | 4.71 | 52.344 | all required (mutating) calls made in order with correct args |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 39.742 | 5.03 | 54.700 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |

**Summary**: correct 56.2% (16 scored, 6 skipped) — mean_steps 2.62, mean_prefill_s 25.271, mean_decode_tok_s 7.55, mean_total_s 34.541

- commit: 94e4aaa
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
