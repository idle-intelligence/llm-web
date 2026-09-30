# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 34 tools, native, label=ids-from-tools

- system: "You are a helpful home assistant with access to Sonos speaker controls. Identifiers such as group_id, player_id, household_id and favorite ids must come from tool results; if you do not have the identifier you need, first call the tool that lists them."
- tool_count: 34
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 8172

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s01 | false | 2 | 397.275 | 2.95 | 412.573 | final read get_now_playing({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| s02 | true | 2 | 25.910 | 2.73 | 60.021 | matched expected sequence exactly |
| s03 | false | 2 | 14.194 | 2.75 | 39.283 | final read get_sonos_favorites({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s04 | false | 2 | 11.123 | 2.84 | 30.183 | final read get_sonos_playlists({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s05 | false | 2 | 11.659 | 2.85 | 30.614 | final read get_registered_music_services({"household_id":"Sonos_ABC123XYZ"}) never made with correct args |
| s06 | false | 2 | 9.569 | 2.99 | 20.636 | missing required call skip_to_next_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s07 | false | 2 | 9.204 | 2.98 | 20.296 | missing required call skip_to_previous_track({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s08 | false | 2 | 8.687 | 3.04 | 19.231 | missing required call pause({"group_id":"RINCON_LIVING01:2"}) in order after step 0 |
| s09 | true | 4 | 78.542 | 2.88 | 106.019 | all required (mutating) calls made in order with correct args |
| s10 | false | 2 | 9.863 | 2.93 | 29.028 | final read get_shuffle_repeat_crossfade({"group_id":"RINCON_LIVING01:2"}) never made with correct args |
| m01 | true | 3 | 52.765 | 2.79 | 72.544 | all required (mutating) calls made in order with correct args |
| m02 | true | 3 | 52.706 | 2.83 | 77.823 | all required (mutating) calls made in order with correct args |
| m03 | false | 3 | 50.268 | 2.84 | 72.116 | missing required call set_group_mute({"group_id":"RINCON_BEDROOM01:3","muted":true}) in order after step 0 |
| m04 | false | 2 | 8.724 | 3.13 | 19.313 | missing required call skip_to_next_track({"group_id":"RINCON_KITCHEN01:1"}) in order after step 0 |
| m05 | false | 3 | 53.151 | 2.76 | 82.851 | missing required call play_sonos_favorite({"group_id":"RINCON_LIVING01:2","favorite_id":"FV:1","shuffle":false}) in order after step 0 |
| m06 | false | 2 | 10.461 | 2.87 | 27.241 | missing required call add_players_to_group({"group_id":"RINCON_KITCHEN01:1","player_ids":["RINCON_BEDROOM01"]}) in order after step 0 |
| m07 | false | 2 | 9.449 | 2.93 | 21.434 | final read get_group_volume({"group_id":"RINCON_BEDROOM01:3"}) never made with correct args |
| m08 | true | 3 | 53.028 | 2.77 | 78.723 | all required (mutating) calls made in order with correct args |
| m09 | false | 4 | 77.349 | 2.80 | 112.067 | missing required call set_group_mute({"group_id":"RINCON_LIVING01:2","muted":false}) in order after step 0 |
| m10 | true | 3 | 52.796 | 2.78 | 81.627 | all required (mutating) calls made in order with correct args |

**Summary**: correct 30.0% (20 scored, 0 skipped) — mean_steps 2.50, mean_prefill_s 49.836, mean_decode_tok_s 2.87, mean_total_s 70.681

- commit: b961159
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
