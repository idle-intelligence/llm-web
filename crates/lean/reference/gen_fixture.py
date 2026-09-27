"""Reference fixture for crates/lean, milestone-1 slice 1.

Loads the same Q4_0 GGUF the Rust engine will read via transformers'
built-in GGUF loader (dequantizes on load), runs prefill + 32 greedy
decode steps for a fixed prompt, and writes a small JSON fixture (no
weights) with: input token ids, top-20 (value, id) at the last prefill
position, and the 32 greedy continuation token ids.

venv: crates/lean/reference/.venv (gitignored).
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = "/Users/tc/Code/idle-intelligence/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
TOKENIZER_DIR = "/Users/tc/Code/idle-intelligence/models/hf/Qwen2.5-0.5B-Instruct"
PROMPT = "What is the capital of France?"
N_DECODE = 32
TOP_K = 20

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
messages = [{"role": "user", "content": PROMPT}]
input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=False)
if not torch.is_tensor(input_ids):
    input_ids = torch.tensor(input_ids)

model = AutoModelForCausalLM.from_pretrained(
    TOKENIZER_DIR,
    gguf_file=GGUF_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()

with torch.no_grad():
    out = model(input_ids)
    logits_last = out.logits[0, -1, :].float()
    top = torch.topk(logits_last, TOP_K)

    gen = model.generate(
        input_ids,
        max_new_tokens=N_DECODE,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    continuation = gen[0, input_ids.shape[1]:].tolist()

fixture = {
    "prompt": PROMPT,
    "chat_template_applied": True,
    "input_ids": input_ids[0].tolist(),
    "prefill_top20": {
        "ids": top.indices.tolist(),
        "values": [round(v, 6) for v in top.values.tolist()],
    },
    "greedy_continuation": continuation,
    "continuation_text": tok.decode(continuation, skip_special_tokens=True),
    "source": "transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...) dequantized on load",
    "gguf_path_note": "qwen2.5-0.5b-instruct-q4_0.gguf (not committed; see path in this script)",
}

out_path = Path(__file__).parent / "fixture.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}")
print("input_ids:", fixture["input_ids"])
print("top1:", top.indices[0].item(), top.values[0].item())
print("continuation:", fixture["continuation_text"])
