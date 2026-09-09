# Sonos MCP tool schemas

Captured 2026-09-10 from the claude.ai Sonos MCP connector's tool schemas
(surfaced to this session as deferred tools named
`mcp__claude_ai_Sonos__<tool>`, fetched via `ToolSearch` — name, description,
and `parameters` copied verbatim, unedited). This is **not** a raw MCP
`tools/list` response captured off the wire; it's a byte-for-byte transcript
of the same schema data as exposed through that connector. **To be
re-verified against a live `tools/list` call made through the relay** before
being treated as ground truth for wire-format details (e.g. whether the
relay's `tools/list` wraps `inputSchema` under a different key, or adds
fields this capture doesn't show).

## Files

- `tools.json` — all 34 tools, MCP `tools/list`-shaped:
  `[{"name", "description", "inputSchema"}, ...]`.
- `tools-12.json` — the 12-tool subset used for the tool-count experiment
  (`get_households_and_groups_and_players`, `get_now_playing`, `pause`,
  `resume`, `skip_to_next_track`, `skip_to_previous_track`,
  `set_group_volume`, `adjust_group_volume`, `get_group_volume`,
  `set_group_mute`, `get_sonos_favorites`, `play_sonos_favorite`) — same
  entries as in `tools.json`, verbatim.

## Tool name → required parameters

| Tool | Required params |
|---|---|
| add_players_to_group | group_id, player_ids |
| adjust_group_volume | group_id, volume_delta |
| adjust_player_volume | player_id, volume_delta |
| get_group_volume | group_id |
| get_households_and_groups_and_players | (none) |
| get_night_sound_and_speech_enhancement | player_id |
| get_now_playing | group_id |
| get_player_volume | player_id |
| get_registered_music_services | household_id |
| get_shuffle_repeat_crossfade | group_id |
| get_sonos_favorites | household_id |
| get_sonos_playlists | household_id |
| move_audio_to_players | source_group_id, destination_player_ids |
| pause | group_id |
| play_album | group_id, album, shuffle |
| play_artist | group_id, artist, shuffle |
| play_player_line_in | group_id, source_player_id |
| play_playlist | group_id, playlist, shuffle |
| play_radio | group_id |
| play_sonos_favorite | group_id, favorite_id, shuffle |
| play_sonos_playlist | group_id, playlist_id, shuffle |
| play_station | group_id |
| play_track | group_id, track |
| remove_players_from_group | group_id, player_ids |
| resume | group_id |
| seek | group_id |
| set_group_mute | group_id, muted |
| set_group_volume | group_id, volume |
| set_night_sound_and_speech_enhancement | player_id |
| set_player_mute | player_id, muted |
| set_player_volume | player_id, volume |
| set_shuffle_repeat_crossfade | group_id |
| skip_to_next_track | group_id |
| skip_to_previous_track | group_id |

Note: `move_audio_to_players` uses `source_group_id` (not `group_id`), and
several music-service play tools (`play_radio`, `play_station`,
`play_sonos_favorite` via `favorite_id`, etc.) take extra optional params
(`music_service`, `artist`, `album`, `shuffle`, etc.) — see `tools.json` for
the full schemas.
