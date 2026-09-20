# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 34 tools, native, label=default-listing-first

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 34
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 8133

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 393.074 | 2.98 | 407.213 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 25.393 | 2.64 | 60.618 | matched expected sequence exactly |
| s03 | false | 2 | 14.440 | 2.71 | 39.965 | final read get_sonos_favorites({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s04 | false | 2 | 11.184 | 2.82 | 30.378 | final read get_sonos_playlists({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s05 | false | 2 | 11.636 | 2.77 | 34.078 | final read get_registered_music_services({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s06 | false | 2 | 9.588 | 2.86 | 21.150 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 9.028 | 2.91 | 20.418 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 2 | 8.638 | 2.95 | 19.512 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | false | 2 | 8.423 | 3.00 | 17.782 | missing required call resume({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s10 | false | 3 | 20.552 | 2.82 | 48.951 | final read get_shuffle_repeat_crossfade({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| m01 | true | 3 | 51.338 | 2.86 | 70.603 | all required (mutating) calls made in order with correct args |
| m02 | false | 3 | 51.400 | 2.81 | 76.348 | missing required call set_group_volume({"group_id":"RINCON_LIVING01:2","volume":20}) in order after step 0 |
| m03 | false | 3 | 50.462 | 2.80 | 72.677 | missing required call set_group_mute({"group_id":"RINCON_BEDROOM01:3","muted":true}) in order after step 0 |
| m04 | false | 2 | 8.708 | 3.08 | 19.450 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 54.209 | 2.72 | 84.354 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | false | 2 | 10.273 | 2.87 | 27.040 | missing required call add_players_to_group({"group_id":"RINCON_KITCHEN01:1","player_ids":["RINCON_BEDROOM01"]}) in order after step 0 |
| m07 | false | 2 | 9.408 | 2.94 | 21.328 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 53.863 | 2.74 | 79.838 | all required (mutating) calls made in order with correct args |
| m09 | true | 3 | 51.813 | 2.78 | 75.196 | all required (mutating) calls made in order with correct args |
| m10 | true | 3 | 55.412 | 2.75 | 84.870 | all required (mutating) calls made in order with correct args |

**Summary**: correct 25.0% (20 scored, 0 skipped) — mean_steps 2.40, mean_prefill_s 45.442, mean_decode_tok_s 2.84, mean_total_s 65.588

- commit: 76ce65b
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
