# Sonos MCP agent eval — 2026-09-11

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=constrained43-fix-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | true | 3 | 94.329 | 6.11 | 106.360 | final read made with correct args |
| s02 | true | 2 | 16.542 | 8.89 | 34.898 | matched expected sequence exactly |
| s03 | true | 3 | 44.555 | 4.10 | 69.229 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | true | 3 | 41.986 | 5.34 | 53.643 | all required (mutating) calls made in order with correct args |
| s07 | true | 3 | 39.658 | 5.63 | 50.002 | all required (mutating) calls made in order with correct args |
| s08 | true | 3 | 39.314 | 5.99 | 49.034 | all required (mutating) calls made in order with correct args |
| s09 | false | 3 | 38.778 | 5.80 | 48.131 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 38.961 | 6.01 | 48.317 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 39.525 | 5.78 | 51.845 | all required (mutating) calls made in order with correct args |
| m03 | true | 3 | 38.831 | 5.78 | 49.940 | all required (mutating) calls made in order with correct args |
| m04 | true | 3 | 38.952 | 5.98 | 49.358 | all required (mutating) calls made in order with correct args |
| m05 | false | 3 | 39.934 | 5.37 | 55.790 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | true | 3 | 39.489 | 5.76 | 50.472 | final read made with correct args |
| m08 | true | 3 | 39.383 | 5.71 | 52.028 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 39.055 | 5.68 | 50.532 | all required (mutating) calls made in order with correct args |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 39.370 | 5.65 | 52.680 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s11 | true | 3 | 39.099 | 5.74 | 51.507 | all required (mutating) calls made in order with correct args |
| s12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s13 | true | 3 | 39.132 | 5.63 | 50.705 | all required (mutating) calls made in order with correct args |
| s14 | true | 3 | 38.955 | 5.70 | 52.141 | final read made with correct args |
| s15 | false | 7 | 131.841 | 5.22 | 163.489 | agent error: agent exceeded max_steps (6) without a final text response |
| s16 | false | 5 | 90.038 | 5.36 | 116.795 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":-10}) in order after step 0 |
| s17 | true | 4 | 65.891 | 5.28 | 88.848 | all required (mutating) calls made in order with correct args |
| m13 | true | 4 | 65.952 | 5.43 | 86.427 | all required (mutating) calls made in order with correct args |
| m14 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m15 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m16 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m17 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m18 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m19 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m20 | true | 5 | 86.634 | 5.57 | 105.893 | all required (mutating) calls made in order with correct args |
| m21 | false | 3 | 39.189 | 5.97 | 50.438 | no_mutation mode: mutating call pause({"group_id":"RINCON_BEDROOM01:3"}) was made |
| m22 | false | 1 | 0.992 | 5.25 | 3.293 | any_play mode: no play_* call with a valid group_id was made |
| m23 | true | 4 | 60.599 | 5.64 | 79.610 | all required (mutating) calls made in order with correct args |
| m24 | true | 3 | 39.604 | 5.88 | 49.161 | all required (mutating) calls made in order with correct args |
| m25 | true | 3 | 38.982 | 6.13 | 48.473 | all required (mutating) calls made in order with correct args |
| m26 | false | 4 | 65.413 | 5.57 | 80.717 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":10}) in order after step 0 |

**Summary**: correct 76.7% (30 scored, 13 skipped) — mean_steps 3.30, mean_prefill_s 49.033, mean_decode_tok_s 5.73, mean_total_s 63.325

- commit: 93e3115
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: /Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: /Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r
