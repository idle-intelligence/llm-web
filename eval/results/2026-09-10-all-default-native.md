# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 34 tools, native, label=default

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 34
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 8133

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 383.328 | 3.03 | 398.203 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 25.669 | 3.00 | 40.714 | matched expected sequence exactly |
| s03 | true | 3 | 53.711 | 2.82 | 87.459 | final read made with correct args |
| s04 | false | 2 | 11.129 | 2.88 | 29.903 | final read get_sonos_playlists({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s05 | false | 2 | 11.427 | 2.86 | 30.338 | final read get_registered_music_services({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s06 | false | 2 | 9.397 | 2.99 | 20.458 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 8.891 | 3.05 | 19.759 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 2 | 8.467 | 3.06 | 18.965 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 2 | 8.361 | 3.06 | 17.881 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | false | 2 | 9.476 | 3.05 | 23.939 | final read get_shuffle_repeat_crossfade({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| m01 | false | 2 | 8.695 | 3.09 | 17.469 | missing required call pause({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m02 | true | 3 | 51.863 | 2.90 | 76.378 | all required (mutating) calls made in order with correct args |
| m03 | false | 3 | 48.484 | 2.90 | 69.888 | missing required call set_group_mute({"group_id":"RINCON_BEDROOM01:3","muted":true}) in order after step 0 |
| m04 | false | 2 | 8.700 | 3.12 | 19.306 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 54.368 | 2.82 | 83.517 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | false | 2 | 9.965 | 2.98 | 24.431 | missing required call add_players_to_group({"group_id":"RINCON_KITCHEN01:1","player_ids":["RINCON_BEDROOM01"]}) in order after step 0 |
| m07 | false | 2 | 9.394 | 2.98 | 21.153 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 52.862 | 2.84 | 77.882 | all required (mutating) calls made in order with correct args |
| m09 | false | 4 | 77.687 | 2.83 | 111.954 | missing required call set_group_mute({"group_id":"RINCON_LIVING01:2","muted":false}) in order after step 0 |
| m10 | true | 3 | 54.499 | 2.81 | 83.047 | all required (mutating) calls made in order with correct args |

**Summary**: correct 25.0% (20 scored, 0 skipped) — mean_steps 2.40, mean_prefill_s 45.319, mean_decode_tok_s 2.95, mean_total_s 63.632

- commit: a664b1e
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
