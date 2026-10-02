"""Reference token hash for crates/lean/www/backends.html.

The page runs one fixed greedy generation (the fixture's "short" prompt
ids, 64 tokens, no EOS stop) and compares the SHA-256 of the 64 ids
(u32 little-endian) against this script's output. The reference is HF
transformers' own forward pass in float32 with the weights dequantized
from the same GGUF file the page loads, decoded with a plain argmax loop
(no `generate()`, so EOS never stops the run and no logits processor
touches the scores).

  LEAN_GGUF            GGUF file the page loads
  LEAN_TOKENIZER_DIR   HF directory with config + tokenizer for that model
  LEAN_FIXTURE         fixture JSON whose "short" case supplies input_ids

venv: crates/lean/reference/.venv (gitignored, shared with gen_fixture*.py).
"""
import hashlib
import json
import os
import struct

import torch
from transformers import AutoModelForCausalLM

N_GEN = 64

gguf = os.environ["LEAN_GGUF"]
tok_dir = os.environ["LEAN_TOKENIZER_DIR"]
fixture = json.load(open(os.environ["LEAN_FIXTURE"]))
prompt_ids = next(c for c in fixture["cases"] if c["name"] == "short")["input_ids"]

model = AutoModelForCausalLM.from_pretrained(tok_dir, gguf_file=gguf, dtype=torch.float32, device_map="cpu")
model.eval()

ids = []
with torch.no_grad():
    out = model(torch.tensor([prompt_ids]), use_cache=True)
    past = out.past_key_values
    nxt = int(torch.argmax(out.logits[0, -1]))
    ids.append(nxt)
    for _ in range(N_GEN - 1):
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(torch.argmax(out.logits[0, -1]))
        ids.append(nxt)

digest = hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()
print(json.dumps({"gguf": os.path.basename(gguf), "prompt_len": len(prompt_ids), "ids": ids, "sha256": digest}))
