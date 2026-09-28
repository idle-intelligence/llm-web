"""Reference fixture for crates/lean's Qwen2.5-3B-Instruct port.

Same method as `gen_fixture.py` (HF transformers' own forward pass, loading
the official "q4_0" GGUF via `gguf_file=`, greedy decode, top-20 prefill
logits) - a separate script per model, not a parametrized one, matching
`gen_fixture_qwen3_1_7b.py`'s convention. This model's official "q4_0" GGUF
carries `output.weight` at Q6_K residency (`token_embd.weight` stays Q4_0) -
see `gguf.rs`'s `GgmlDtype` doc comment - which HF's own GGUF loader
dequantizes to F32 on load exactly like every other tensor, so this script
needs no Q6_K-specific handling; the Rust side's `Q6_K` support is what lets
`fixture_parity_qwen25_3b.rs` load the same file at all.

  LEAN_GGUF_QWEN25_3B          path to the Qwen2.5-3B-Instruct "q4_0" GGUF
  LEAN_TOKENIZER_DIR_QWEN25_3B directory with tokenizer.json + tokenizer_config.json

venv: crates/lean/reference/.venv (gitignored, shared with gen_fixture.py).
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = os.environ.get("LEAN_GGUF_QWEN25_3B")
TOKENIZER_DIR = os.environ.get("LEAN_TOKENIZER_DIR_QWEN25_3B")
if not GGUF_PATH or not TOKENIZER_DIR:
    raise SystemExit("set LEAN_GGUF_QWEN25_3B and LEAN_TOKENIZER_DIR_QWEN25_3B before running gen_fixture_qwen25_3b.py")

N_DECODE = 32
TOP_K = 20

# Same three short/long/non-English cases as gen_fixture.py's CASES - no
# LONG_TOKEN_CASES here (those need `fixtures/reference/rendered/*.tokens.json`
# at the repo root and a 32-token greedy decode per case through a 3B model
# on CPU is slow; the three short cases are enough to gate Q6_K parity).
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

import json

fixture = {
    "source": "transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...) dequantized on load, Qwen2.5-3B-Instruct q4_0 (output.weight Q6_K, token_embd.weight Q4_0)",
    "cases": cases_out,
}

out_path = Path(__file__).parent / "fixture_qwen25_3b.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}")
