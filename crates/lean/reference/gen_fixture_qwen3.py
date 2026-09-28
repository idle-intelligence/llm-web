"""Reference fixture for crates/lean's Qwen3 port.

Same method as `gen_fixture.py` (HF transformers' own forward pass, loading
the official Q8_0 GGUF via `gguf_file=`, greedy decode, top-20 prefill
logits): a separate script, not a parametrized `gen_fixture.py`, so the two
fixtures (`fixture.json` for Qwen2.5, `fixture_qwen3.json` here) never
collide and each model's env vars stay simple.

  LEAN_GGUF            path to the Qwen3 Q8_0 GGUF file
  LEAN_TOKENIZER_DIR   directory with tokenizer.json + tokenizer_config.json

venv: crates/lean/reference/.venv (gitignored, shared with gen_fixture.py).
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = os.environ.get("LEAN_GGUF")
TOKENIZER_DIR = os.environ.get("LEAN_TOKENIZER_DIR")
if not GGUF_PATH or not TOKENIZER_DIR:
    raise SystemExit("set LEAN_GGUF and LEAN_TOKENIZER_DIR before running gen_fixture_qwen3.py")

N_DECODE = 32
TOP_K = 20

# Same three short/long/non-English cases as gen_fixture.py's CASES, run
# through Qwen3's own chat template (enable_thinking left at its default -
# no enable_thinking kwarg passed, matching what crates/lean's
# chat_template.rs renders: the template string as-is, no extra context
# keys).
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

# One long real-agent-prompt case (not two, unlike gen_fixture.py): the
# shortest of the Sonos MCP agent's real tool-calling inputs
# (fixtures/reference/rendered/02_tools_single.tokens.json, 2225 tokens),
# reused as raw token ids exactly like gen_fixture.py's LONG_TOKEN_CASES -
# same vocab (151936, confirmed by the qwen3 survey) as Qwen2.5, and well
# inside Qwen3-0.6B's 40960 max_position_embeddings, so no retokenization
# through Qwen3's own (tools-aware) chat template is needed to exercise a
# long sequence.
LONG_TOKEN_CASES = [
    ("long_tools_single", "fixtures/reference/rendered/02_tools_single.tokens.json"),
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
    "source": "transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...) dequantized on load, Qwen3-0.6B Q8_0",
    "cases": cases_out,
}

out_path = Path(__file__).parent / "fixture_qwen3.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}")
