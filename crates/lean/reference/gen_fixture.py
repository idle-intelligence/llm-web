"""Reference fixtures for crates/lean, milestone-1 slices 1-3.

Loads the same Q4_0 GGUF the Rust engine reads via transformers' built-in
GGUF loader (dequantizes on load), runs prefill + greedy decode for a few
fixed chat-templated prompts, and writes a small JSON fixture (no weights)
per prompt: input token ids, top-20 (value, id) at the last prefill
position, and the greedy continuation.

Model/tokenizer paths are never hardcoded here (crates/lean is meant to be
publishable) — set them via env vars:
  LEAN_GGUF            path to the Q4_0 GGUF file
  LEAN_TOKENIZER_DIR    directory with tokenizer.json + tokenizer_config.json

venv: crates/lean/reference/.venv (gitignored).
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = os.environ.get("LEAN_GGUF")
TOKENIZER_DIR = os.environ.get("LEAN_TOKENIZER_DIR")
if not GGUF_PATH or not TOKENIZER_DIR:
    raise SystemExit("set LEAN_GGUF and LEAN_TOKENIZER_DIR before running gen_fixture.py")

N_DECODE = 32
TOP_K = 20

# Three cases: short (slice 1's original), long enough to exercise the
# tiled prefill kernel (>64 chat-templated tokens, slice 2's TM=TN=64 tile),
# and non-English (slice 3's chat-template + tokenizer path).
CASES = [
    ("short", "What is the capital of France?"),
    (
        "long",
        "Please list, in order, the first fifteen elements of the periodic "
        "table together with their atomic number, their standard atomic "
        "weight, and one short fact about each element's most common "
        "industrial use, so that a student studying for a chemistry exam "
        "next week can memorize the table more easily.",
    ),
    ("non_english", "Quelle est la capitale de l'Allemagne, et pourquoi cette ville a-t-elle été choisie ?"),
]

# Long-context cases: the Sonos MCP agent's real (messages, tools) inputs,
# already chat-templated and tokenized once (with the *same* HF tokenizer,
# `apply_chat_template(..., tools=tools)`) into
# `fixtures/reference/rendered/<name>.tokens.json` at the repo root - see
# `fixtures/reference/inputs/README.md`. Reused here (not re-rendered) so
# this script doesn't need to grow tool-calling chat-template support just
# to exercise long sequences; these cases carry no `prompt` string and
# `fixture_parity.rs`/the browser harness skip the
# tokenizer-reproduces-input_ids check for them, feeding `input_ids`
# straight into prefill instead. Chosen as the two longest real cases that
# clear the old attn_prefill.wgsl MAX_SEQ=256 / attn_decode.wgsl
# scratch[2048] caps by a wide margin (2225, 2354 tokens; `04_tools_all` at
# 8140 tokens is timing-only elsewhere, not worth this script's per-case
# 32-token greedy decode cost).
LONG_TOKEN_CASES = [
    ("long_tools_single", "fixtures/reference/rendered/02_tools_single.tokens.json"),
    ("long_tools_multiturn", "fixtures/reference/rendered/03_tools_multiturn.tokens.json"),
]
REPO_ROOT = Path(__file__).resolve().parents[3]

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForCausalLM.from_pretrained(
    TOKENIZER_DIR,
    gguf_file=GGUF_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()

def run_case(name, input_ids_list, prompt=None):
    input_ids = torch.tensor([input_ids_list])
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

    case = {
        "name": name,
        "input_ids": input_ids[0].tolist(),
        "prefill_top20": {
            "ids": top.indices.tolist(),
            "values": [round(v, 6) for v in top.values.tolist()],
        },
        "greedy_continuation": continuation,
        "continuation_text": tok.decode(continuation, skip_special_tokens=True),
    }
    if prompt is not None:
        case["prompt"] = prompt
    else:
        case["no_retokenize"] = True
    return case


cases_out = []
for name, prompt in CASES:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if torch.is_tensor(input_ids):
        input_ids = input_ids[0].tolist()
    case = run_case(name, input_ids, prompt=prompt)
    cases_out.append(case)
    print(f"[{name}] seq={len(case['input_ids'])} top1={case['prefill_top20']['ids'][0]} continuation={case['continuation_text'][:80]!r}")

for name, rel_path in LONG_TOKEN_CASES:
    tokens_path = REPO_ROOT / rel_path
    input_ids_list = json.loads(tokens_path.read_text())
    case = run_case(name, input_ids_list)
    cases_out.append(case)
    print(f"[{name}] seq={len(case['input_ids'])} top1={case['prefill_top20']['ids'][0]} continuation={case['continuation_text'][:80]!r}")

fixture = {
    "source": "transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...) dequantized on load",
    "cases": cases_out,
}

out_path = Path(__file__).parent / "fixture.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}")
