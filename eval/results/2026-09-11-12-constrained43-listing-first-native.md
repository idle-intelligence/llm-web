# Sonos MCP agent eval — 2026-09-11

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=constrained43-listing-first

**Partial timing contamination**: `decode_tok_s` craters from a ~5.2-5.9
tok/s baseline to 0.53-1.35 tok/s for `m13`, `m20`, `m21`, `m22`, `m23`
(with `s15`-`s17` mildly depressed at 2.31-5.24 just before), then
recovers to 5.79-5.82 tok/s at `m24` onward — consistent with a
concurrent GPU consumer starting partway through this run and clearing
before it finished (the GPU was confirmed idle immediately before this
run started). `prefill_s`/`total_s` for that same span are inflated
accordingly (e.g. `m20` 222.958s prefill, 433.472s total). `correct`,
`steps`, and `reason` are unaffected. Rows outside that span (`s01`-`s17`
excluding `s15`-`s17`, `m01`-`m11`, `m24`-`m26`) are clean.

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2503

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 3 | 100.915 | 4.50 | 117.717 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 23.596 | 4.48 | 60.082 | matched expected sequence exactly |
| s03 | true | 3 | 48.127 | 5.54 | 66.408 | final read made with correct args |
| s04 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s05 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s06 | false | 3 | 40.658 | 5.77 | 51.793 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 3 | 48.396 | 5.47 | 59.604 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 3 | 47.149 | 5.64 | 57.662 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 3 | 47.530 | 5.24 | 58.090 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m01 | true | 3 | 47.913 | 5.25 | 58.866 | all required (mutating) calls made in order with correct args |
| m02 | false | 3 | 47.359 | 5.60 | 60.459 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":20}) in order after step 0 |
| m03 | true | 3 | 48.036 | 5.52 | 59.662 | all required (mutating) calls made in order with correct args |
| m04 | false | 3 | 47.124 | 5.81 | 58.186 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 48.118 | 5.35 | 64.819 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m07 | false | 7 | 160.033 | 5.86 | 189.983 | agent error: agent exceeded max_steps (6) without a final text response |
| m08 | true | 3 | 48.541 | 4.90 | 63.499 | all required (mutating) calls made in order with correct args |
| m09 | false | 3 | 48.032 | 4.99 | 61.559 | missing required call set_group_mute({"group_id":"RINCON_LIVING01:2","muted":false}) in order after step 0 |
| m10 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m11 | true | 3 | 47.046 | 5.55 | 60.970 | all required (mutating) calls made in order with correct args |
| m12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s11 | false | 3 | 46.851 | 5.66 | 59.983 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":35}) in order after step 0 |
| s12 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| s13 | true | 3 | 47.861 | 5.42 | 60.082 | all required (mutating) calls made in order with correct args |
| s14 | true | 3 | 47.170 | 5.59 | 60.827 | final read made with correct args |
| s15 | false | 7 | 172.796 | 2.31 | 259.703 | agent error: agent exceeded max_steps (6) without a final text response |
| s16 | false | 3 | 48.642 | 5.36 | 62.342 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":-10}) in order after step 0 |
| s17 | true | 4 | 80.031 | 5.24 | 103.938 | all required (mutating) calls made in order with correct args |
| m13 | true | 4 | 120.989 | 1.10 | 222.962 | all required (mutating) calls made in order with correct args |
| m14 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m15 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m16 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m17 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m18 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m19 | skipped | - | - | - | - | tools12_ok=false: not solvable under the 12-tool subset |
| m20 | false | 7 | 222.958 | 0.75 | 433.472 | agent error: agent exceeded max_steps (6) without a final text response |
| m21 | false | 3 | 59.771 | 0.79 | 130.408 | no_mutation mode: mutating call pause({"group_id":"RINCON_KITCHEN01:1"}) was made |
| m22 | false | 1 | 4.380 | 0.53 | 26.881 | any_play mode: no play_* call with a valid group_id was made |
| m23 | true | 4 | 94.563 | 1.35 | 174.685 | all required (mutating) calls made in order with correct args |
| m24 | true | 3 | 40.175 | 5.82 | 50.001 | all required (mutating) calls made in order with correct args |
| m25 | true | 3 | 47.505 | 5.49 | 58.308 | all required (mutating) calls made in order with correct args |
| m26 | false | 4 | 77.250 | 5.79 | 92.844 | missing required call adjust_group_volume({"group_id":"RINCON_LIVING01:2","volume_delta":10}) in order after step 0 |

**Summary**: correct 43.3% (30 scored, 13 skipped) — mean_steps 3.43, mean_prefill_s 66.984, mean_decode_tok_s 4.56, mean_total_s 98.193

- commit: 4dd5700
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
