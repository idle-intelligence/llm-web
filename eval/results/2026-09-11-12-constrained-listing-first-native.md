# Sonos MCP agent eval — 2026-09-11

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=constrained-listing-first

**Timing contaminated**: this run shared the GPU with another worker's
headless Chromium job for part or all of its duration. Every timing
column below (`prefill_s`, `decode_tok_s`, `total_s`, and the summary's
`mean_prefill_s`/`mean_decode_tok_s`/`mean_total_s`) is **contaminated
(concurrent GPU job)** — not representative of this change's real
performance. `correct`/`steps`/`reason` (and the constrained-decoding
step/forced-token breakdown reported separately in
`tests/constrained.rs`'s fixtures) are unaffected by GPU contention and
stand as measured.

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 3 | 111.570 | 4.21 | 129.476 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 18.381 | 4.54 | 54.300 | matched expected sequence exactly |
| s03 | true | 3 | 43.092 | 4.52 | 65.463 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 3 | 47.632 | 4.93 | 60.658 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 3 | 41.528 | 5.28 | 53.106 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 3 | 43.601 | 5.46 | 54.437 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 3 | 42.570 | 5.08 | 53.438 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 46.352 | 5.15 | 57.445 | all required (mutating) calls made in order with correct args |
| m02 | false | 3 | 47.380 | 4.31 | 64.376 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":20}) in order after step 0 |
| m03 | true | 3 | 42.341 | 5.06 | 55.035 | all required (mutating) calls made in order with correct args |
| m04 | false | 3 | 44.405 | 5.14 | 56.898 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 46.629 | 4.67 | 65.735 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 7 | 132.229 | 6.21 | 160.460 | agent error: agent exceeded max_steps (6) without a final text response |
| m08 | true | 3 | 38.443 | 6.14 | 50.357 | all required (mutating) calls made in order with correct args |
| m09 | false | 3 | 37.974 | 5.80 | 49.554 | missing required call set_group_mute({"group_id":"RINCON_LIVING01:2","muted":false}) in order after step 0 |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 38.182 | 5.84 | 51.399 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s11 | false | 3 | 38.645 | 5.96 | 51.088 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":35}) in order after step 0 |
| s12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s13 | true | 3 | 38.560 | 5.74 | 50.096 | all required (mutating) calls made in order with correct args |
| s14 | true | 3 | 38.233 | 5.84 | 51.284 | final read made with correct args |
| s15 | false | 7 | 131.233 | 5.45 | 168.017 | agent error: agent exceeded max_steps (6) without a final text response |
| s16 | false | 3 | 38.882 | 5.83 | 51.434 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":-10}) in order after step 0 |
| s17 | true | 4 | 64.100 | 5.34 | 87.534 | all required (mutating) calls made in order with correct args |
| m13 | true | 4 | 65.020 | 5.83 | 84.287 | all required (mutating) calls made in order with correct args |
| m14 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m15 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m16 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m17 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m18 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m19 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m20 | false | 7 | 112.629 | 5.52 | 141.131 | agent error: agent exceeded max_steps (6) without a final text response |
| m21 | false | 3 | 38.777 | 6.11 | 47.982 | no_mutation mode: mutating call pause({"group_id":"RINCON_KITCHEN01:1"}) was made |
| m22 | false | 1 | 0.645 | 5.56 | 2.822 | any_play mode: no play_* call with a valid group_id was made |
| m23 | true | 4 | 61.044 | 5.58 | 80.457 | all required (mutating) calls made in order with correct args |
| m24 | true | 3 | 38.907 | 6.06 | 48.341 | all required (mutating) calls made in order with correct args |
| m25 | true | 3 | 38.638 | 6.19 | 48.205 | all required (mutating) calls made in order with correct args |
| m26 | false | 4 | 64.335 | 5.80 | 79.899 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":10}) in order after step 0 |

**Summary**: correct 43.3% (30 scored, 13 skipped) — mean_steps 3.43, mean_prefill_s 53.065, mean_decode_tok_s 5.44, mean_total_s 69.157

- commit: 9da9f53
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: /Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: /Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r
