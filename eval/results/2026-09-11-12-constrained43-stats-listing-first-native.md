# Sonos MCP agent eval — 2026-09-11

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=constrained43-stats-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | tokens_gen | forced_tokens | forced_share | model_steps | retries | tool_errors | reason |
|----|---------|-------|-----------|---------------|---------|------------|----------------|--------------|-------------|---------|-------------|--------|
| s01 | true | 3 | 79.648 | 8.81 | 88.101 | 73 | 8 | 0.11 | 66 | 0 | 0 | final read made with correct args |
| s02 | true | 2 | 15.016 | 8.95 | 33.244 | 163 | 8 | 0.05 | 156 | 0 | 0 | matched expected sequence exactly |
| s03 | true | 3 | 33.956 | 9.18 | 44.986 | 101 | 8 | 0.08 | 94 | 0 | 0 | final read made with correct args |
| s04 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | true | 3 | 33.627 | 9.75 | 40.010 | 62 | 8 | 0.13 | 55 | 0 | 0 | all required (mutating) calls made in order with correct args |
| s07 | true | 3 | 33.589 | 9.83 | 39.518 | 58 | 8 | 0.14 | 51 | 0 | 0 | all required (mutating) calls made in order with correct args |
| s08 | true | 3 | 33.513 | 9.84 | 39.436 | 58 | 8 | 0.14 | 51 | 0 | 0 | all required (mutating) calls made in order with correct args |
| s09 | false | 3 | 33.505 | 10.00 | 38.931 | 54 | 8 | 0.15 | 47 | 0 | 0 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 33.533 | 10.00 | 39.160 | 56 | 8 | 0.14 | 49 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 33.786 | 9.63 | 41.181 | 71 | 8 | 0.11 | 64 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 33.623 | 9.76 | 40.208 | 64 | 8 | 0.12 | 57 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m04 | true | 3 | 33.657 | 9.81 | 39.999 | 62 | 8 | 0.13 | 55 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m05 | false | 3 | 33.942 | 9.38 | 43.025 | 85 | 8 | 0.09 | 78 | 0 | 0 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | true | 3 | 33.696 | 9.81 | 40.144 | 63 | 8 | 0.13 | 56 | 0 | 0 | final read made with correct args |
| m08 | true | 3 | 33.674 | 9.66 | 41.154 | 72 | 8 | 0.11 | 65 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 33.762 | 8.61 | 41.341 | 65 | 8 | 0.12 | 58 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m10 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 34.242 | 9.21 | 42.410 | 75 | 8 | 0.11 | 68 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s11 | true | 3 | 33.852 | 9.28 | 41.530 | 71 | 8 | 0.11 | 64 | 0 | 0 | all required (mutating) calls made in order with correct args |
| s12 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s13 | true | 3 | 33.539 | 9.73 | 40.248 | 65 | 8 | 0.12 | 58 | 0 | 0 | all required (mutating) calls made in order with correct args |
| s14 | true | 3 | 33.736 | 9.56 | 41.603 | 75 | 8 | 0.11 | 68 | 0 | 0 | final read made with correct args |
| s15 | false | 7 | 122.158 | 6.42 | 147.922 | 165 | 8 | 0.05 | 158 | 0 | 0 | agent error: agent exceeded max_steps (6) without a final text response |
| s16 | false | 5 | 79.764 | 7.50 | 98.881 | 143 | 8 | 0.06 | 136 | 0 | 0 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":-10}) in order after step 0 |
| s17 | true | 4 | 60.380 | 6.63 | 78.674 | 121 | 8 | 0.07 | 114 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m13 | true | 4 | 65.130 | 6.34 | 82.681 | 111 | 8 | 0.07 | 104 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m14 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m15 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m16 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m17 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m18 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m19 | skipped | - | - | - | - | - | - | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m20 | true | 5 | 80.393 | 6.82 | 96.155 | 107 | 8 | 0.07 | 100 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m21 | false | 3 | 36.319 | 7.28 | 45.555 | 67 | 8 | 0.12 | 60 | 0 | 0 | no_mutation mode: mutating call pause({"group_id":"RINCON_BEDROOM01:3"}) was made |
| m22 | false | 1 | 0.773 | 5.72 | 2.891 | 12 | 0 | 0.00 | 12 | 0 | 0 | any_play mode: no play_* call with a valid group_id was made |
| m23 | true | 4 | 57.097 | 6.45 | 73.723 | 107 | 8 | 0.07 | 100 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m24 | true | 3 | 42.219 | 5.34 | 52.744 | 56 | 8 | 0.14 | 49 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m25 | true | 3 | 71.977 | 3.42 | 88.990 | 58 | 8 | 0.14 | 51 | 0 | 0 | all required (mutating) calls made in order with correct args |
| m26 | false | 4 | 104.052 | 4.28 | 123.982 | 85 | 8 | 0.09 | 78 | 0 | 0 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":10}) in order after step 0 |

**Summary**: correct 76.7% (30 scored, 13 skipped) — mean_steps 3.30, mean_prefill_s 46.272, mean_decode_tok_s 8.23, mean_total_s 56.948, mean_model_steps 74.07, mean_forced_share 0.103, total_retries 0, total_tool_errors 0

- commit: 24aa6f5
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
