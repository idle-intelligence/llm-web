# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 34 tools, native, label=kernels-v2-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 34
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 8133

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 265.315 | 0.16 | 534.336 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 57.831 | 0.83 | 134.891 | matched expected sequence exactly |
| s03 | false | 2 | 33.435 | 0.55 | 158.319 | final read get_sonos_favorites({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s04 | false | 2 | 29.673 | 0.32 | 198.171 | final read get_sonos_playlists({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s05 | false | 2 | 29.573 | 1.23 | 79.916 | final read get_registered_music_services({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s06 | false | 2 | 23.062 | 1.36 | 47.396 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 20.320 | 1.69 | 39.932 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 2 | 11.039 | 2.86 | 22.273 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 2 | 10.703 | 2.84 | 20.620 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | false | 3 | 31.110 | 2.20 | 67.551 | final read get_shuffle_repeat_crossfade({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| m01 | true | 3 | 66.058 | 2.64 | 86.942 | all required (mutating) calls made in order with correct args |
| m02 | false | 3 | 65.944 | 2.54 | 93.592 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":20}) in order after step 0 |
| m03 | false | 3 | 69.047 | 1.72 | 105.329 | missing required call set_group_mute({"group_id":"RINCON_BEDROOM01:3","muted":true}) in order after step 0 |
| m04 | false | 2 | 11.814 | 2.05 | 28.064 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 69.030 | 1.84 | 113.711 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | false | 2 | 14.826 | 1.69 | 43.340 | missing required call add_players_to_group({"group_id":"RINCON_KITCHEN01:1","player_ids":["RINCON_BEDROOM01"]}) in order after step 0 |
| m07 | false | 2 | 12.729 | 2.09 | 29.615 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 65.929 | 1.74 | 106.947 | all required (mutating) calls made in order with correct args |
| m09 | false | 3 | 142.119 | 1.58 | 182.273 | missing required call set_group_mute({"group_id":"RINCON_LIVING01:2","muted":false}) in order after step 0 |
| m10 | true | 3 | 113.036 | 1.30 | 174.496 | all required (mutating) calls made in order with correct args |
| m11 | false | 3 | 117.644 | 1.24 | 174.446 | missing required call play_artist({"group_id":"RINCON_KITCHEN01:1","artist":"Nirvana","shuffle":false}) in order after step 0 |
| m12 | false | 3 | 132.177 | 0.86 | 230.860 | missing required call remove_players_from_group({"group_id":"RINCON_KITCHEN01:1","player_ids":["RINCON_KITCHEN02"]}) in order after step 0 |

**Summary**: correct 18.2% (22 scored, 0 skipped) — mean_steps 2.45, mean_prefill_s 63.292, mean_decode_tok_s 1.61, mean_total_s 121.501

- commit: 7aab8ca
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: /Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: /Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r
- NOTE: a concurrent `cargo test --release --features wgpu --test full_forward --test q4_matmul` job (another worker) began GPU-side execution (`full_forward-*` test binary, confirmed via `pgrep`) partway through this run, contending for the GPU. decode_tok_s here (mean 1.61) is well below the uncontended 34-tool baseline (2.84-2.95 tok/s, 2026-09-10-summary.md); prefill_s and total_s are inflated for the same reason. These numbers are not a clean kernels-v2 baseline for decode/prefill throughput comparisons — treat correctness (calls made, correct%) as reliable, timings as contention-degraded.
