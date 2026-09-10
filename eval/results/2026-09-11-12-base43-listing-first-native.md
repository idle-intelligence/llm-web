# Sonos MCP agent eval — 2026-09-11

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=base43-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 52.076 | 9.42 | 57.608 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 15.128 | 8.76 | 33.749 | matched expected sequence exactly |
| s03 | true | 3 | 34.347 | 8.78 | 45.870 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 2 | 5.536 | 9.83 | 8.902 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 4.645 | 9.64 | 7.766 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | true | 3 | 43.626 | 4.44 | 56.722 | all required (mutating) calls made in order with correct args |
| s09 | false | 3 | 45.831 | 3.66 | 60.646 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 46.280 | 2.21 | 71.775 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 47.347 | 3.06 | 70.644 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 39.743 | 5.26 | 51.923 | all required (mutating) calls made in order with correct args |
| m04 | false | 2 | 6.612 | 5.74 | 12.381 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 47.623 | 5.11 | 63.500 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 2 | 7.795 | 6.01 | 13.636 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 39.998 | 5.01 | 54.391 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 47.990 | 3.16 | 68.606 | all required (mutating) calls made in order with correct args |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 43.940 | 4.67 | 60.041 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s11 | true | 3 | 39.745 | 5.23 | 53.332 | all required (mutating) calls made in order with correct args |
| s12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s13 | true | 3 | 39.311 | 5.46 | 51.231 | all required (mutating) calls made in order with correct args |
| s14 | false | 2 | 8.855 | 5.36 | 17.261 | final read get_now_playing({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| s15 | false | 2 | 9.442 | 4.91 | 20.467 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s16 | false | 2 | 8.023 | 5.78 | 15.307 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":-10}) in order after step 0 |
| s17 | true | 4 | 78.658 | 3.69 | 111.507 | all required (mutating) calls made in order with correct args |
| m13 | true | 4 | 81.161 | 3.20 | 115.865 | all required (mutating) calls made in order with correct args |
| m14 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m15 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m16 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m17 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m18 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m19 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m20 | false | 7 | 65.711 | 5.18 | 96.470 | agent error: agent exceeded max_steps (6) without a final text response |
| m21 | false | 3 | 46.677 | 5.53 | 56.293 | no_mutation mode: mutating call pause({"group_id":"RINCON_GARAGE01"}) was made |
| m22 | false | 7 | 11.769 | 9.22 | 13.110 | agent error: agent exceeded max_steps (6) without a final text response |
| m23 | true | 4 | 78.196 | 2.45 | 122.154 | all required (mutating) calls made in order with correct args |
| m24 | true | 3 | 48.653 | 4.92 | 60.090 | all required (mutating) calls made in order with correct args |
| m25 | true | 3 | 47.170 | 5.54 | 57.674 | all required (mutating) calls made in order with correct args |
| m26 | false | 4 | 80.362 | 5.44 | 96.034 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":10}) in order after step 0 |

**Summary**: correct 53.3% (30 scored, 13 skipped) — mean_steps 3.10, mean_prefill_s 39.075, mean_decode_tok_s 5.56, mean_total_s 54.165

- commit: 3e0e1c9
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: /Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: /Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r
