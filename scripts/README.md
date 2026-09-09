# Reference export scripts

venv mon ami. Python 3.12 (torch has no 3.14 wheels yet on this machine;
`/opt/homebrew/bin/python3.12` was used, not the default `python3`).

## Setup

```
cd /Users/tc/Code/idle-intelligence/llm-web
/opt/homebrew/bin/python3.12 -m venv scripts/.venv
scripts/.venv/bin/pip install -r scripts/requirements.txt
```

## Generate the reference fixtures

```
scripts/.venv/bin/python scripts/export_reference.py
```

Reads `fixtures/reference/inputs/{01_no_tools,02_tools_single,03_tools_multiturn}.json`,
writes:

- `fixtures/reference/rendered/<name>.txt` — exact `apply_chat_template(...,
  tokenize=False)` output, no trailing newline appended.
- `fixtures/reference/rendered/<name>.tokens.json` — token ids for that
  string.
- `fixtures/reference/logits/<name>.json` — seq_len, dtype/device, argmax
  token id per position, top-5 (id, logit) at the last position, and the
  greedy-decoded (`do_sample=False`) first 32 generated tokens as text +
  ids.
- `/Users/tc/Code/idle-intelligence/models/reference/xlam-2-3b-fc-r/<name>.logits.npy`
  — full-sequence float32 `[seq_len, vocab_size]` forward-pass logits.
  Outside the repo: multi-hundred MB, not committed.

Device/dtype: MPS with bfloat16 if available, else CPU bfloat16. float32 on
MPS was tried first and thrashed into swap on this 16GB M2 (12GB of float32
weights left no headroom; measured ~1 min of actual CPU progress after 25
min wall time) — switched to bfloat16 (~6GB weights) for both MPS and CPU so
all three fixtures share one dtype. The script prints which it picked; the
same info is recorded per-fixture in `fixtures/reference/logits/<name>.json`.
