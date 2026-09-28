"""Reference fixture for crates/lean's Llama-architecture port (SmolLM2).

Same method as `gen_fixture.py`/`gen_fixture_qwen3.py`: HF transformers'
own forward pass, loading the official GGUF via `gguf_file=`, greedy
decode, top-20 prefill logits. Parametrized (not one script per model) so
the same script covers both SmolLM2 sizes and both quant variants (Q4_0,
Q8_0) this crate's native tests and browser harness check - a separate
fixture per (size, quant) pair, since Q4_0 and Q8_0 give different
quantization noise even from the same underlying weights.

  LEAN_GGUF            path to the SmolLM2 GGUF file (Q4_0 or Q8_0)
  LEAN_TOKENIZER_DIR   directory with tokenizer.json + tokenizer_config.json
  LEAN_FIXTURE_OUT     output path for the fixture JSON

venv: crates/lean/reference/.venv (gitignored, shared with the other
gen_fixture_*.py scripts).
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = os.environ.get("LEAN_GGUF")
TOKENIZER_DIR = os.environ.get("LEAN_TOKENIZER_DIR")
FIXTURE_OUT = os.environ.get("LEAN_FIXTURE_OUT")
if not GGUF_PATH or not TOKENIZER_DIR or not FIXTURE_OUT:
    raise SystemExit("set LEAN_GGUF, LEAN_TOKENIZER_DIR, and LEAN_FIXTURE_OUT before running gen_fixture_llama.py")

N_DECODE = 32
TOP_K = 20

# Same three short/long/non-English cases as gen_fixture.py's/
# gen_fixture_qwen3.py's CASES, run through SmolLM2-Instruct's own chat
# template (ChatML-style, ported from HuggingFaceTB/SmolLM2's own
# tokenizer_config.json - not hand-written).
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

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForCausalLM.from_pretrained(
    TOKENIZER_DIR,
    gguf_file=GGUF_PATH,
    dtype=torch.float32,
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

fixture = {
    "source": f"transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...) dequantized on load, {os.path.basename(GGUF_PATH)}",
    "cases": cases_out,
}

out_path = Path(FIXTURE_OUT)
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}")
