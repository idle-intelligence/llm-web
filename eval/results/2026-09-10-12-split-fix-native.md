# Sonos MCP agent eval — 2026-09-10

## Run: xLAM-2-3b-fc-r-q4_0.gguf, 12 tools, native, label=split-fix

- system: "You are a helpful home assistant with access to Sonos speaker controls."
- tool_count: 12
- max_new_tokens: 256
- max_steps: 6
- prefix_tokens: 2218

| id | correct | steps | prefill_s | decode_tok_s | total_s | reason |
|----|---------|-------|-----------|---------------|---------|--------|
| s09 | true | 5 | 144.852 | 5.85 | 163.328 | all required (mutating) calls made in order with correct args |
| m01 | true | 3 | 36.602 | 5.76 | 46.340 | all required (mutating) calls made in order with correct args |

**Summary**: correct 100.0% (2 scored, 0 skipped) — mean_steps 4.00, mean_prefill_s 90.727, mean_decode_tok_s 5.81, mean_total_s 104.834

- commit: 52fe03a
- machine: Apple M2, 16 GB unified memory, macOS (Darwin 25.3.0)
- gguf: <models>/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf
- model-dir: <models>/hf/xLAM-2-3b-fc-r
