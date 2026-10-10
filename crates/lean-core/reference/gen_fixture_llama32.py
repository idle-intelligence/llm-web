"""Reference fixture for lean-core's Llama 3.2 support (llama3 RoPE scaling).

Same output shape as `gen_fixture_llama.py` (HF transformers forward pass,
greedy decode, top-20 prefill logits), with two differences that matter for
Llama 3.x:

1. The model is built from the official `config.json`, not from the GGUF.
   transformers' GGUF loader (`integrations/ggml.py`) maps no rope scaling
   field and ignores the `rope_freqs.weight` tensor, so
   `from_pretrained(..., gguf_file=...)` alone runs Llama 3.2 with unscaled
   RoPE. Here the GGUF is only the weight source: its dequantized state dict
   is loaded into `LlamaForCausalLM(config)`, whose rotary embedding then
   uses `_compute_llama3_parameters` (checked below before anything runs).

2. The chat template is rendered by HF with a fixed `date_string`
   ("26 Jul 2024", the template's own fallback) so the fixture does not
   depend on the day it was generated. The rendered text is stored; the
   Rust gates re-tokenize it with the `tokenizers` crate and compare ids.

One case runs past `original_max_position_embeddings` (8192), where the
scaled low frequencies are the only thing keeping positions apart.

  LEAN_GGUF            path to the Llama 3.2 GGUF (Q4_0 or Q8_0)
  LEAN_HF_DIR          directory with config.json, tokenizer.json, tokenizer_config.json
  LEAN_FIXTURE_OUT     output path for the fixture JSON
  LEAN_THREADS         optional torch.set_num_threads value

venv: crates/lean-core/reference/.venv (gitignored, shared with the other
gen_fixture_*.py scripts).
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

GGUF_PATH = os.environ.get("LEAN_GGUF")
HF_DIR = os.environ.get("LEAN_HF_DIR")
FIXTURE_OUT = os.environ.get("LEAN_FIXTURE_OUT")
if not GGUF_PATH or not HF_DIR or not FIXTURE_OUT:
    raise SystemExit("set LEAN_GGUF, LEAN_HF_DIR, and LEAN_FIXTURE_OUT before running gen_fixture_llama32.py")
if os.environ.get("LEAN_THREADS"):
    torch.set_num_threads(int(os.environ["LEAN_THREADS"]))

N_DECODE = 32
TOP_K = 20
DATE_STRING = "26 Jul 2024"


def long_context_prompt():
    """A retrieval prompt of roughly 9k tokens: a code word stated once at
    the start, a long deterministic ledger, then a question about the code
    word, so the answer depends on attending across more than 8192
    positions."""
    lines = ["Remember this: the code word is LANTERN-47. You will be asked for it at the end.", ""]
    colors = ["red", "green", "blue", "amber", "violet", "silver", "orange", "teal"]
    cities = ["Lyon", "Oslo", "Quito", "Perth", "Turin", "Accra", "Hanoi", "Leeds", "Porto", "Davao"]
    for i in range(1, 381):
        lines.append(
            f"Ledger entry {i}: crate {i * 37 % 1000} of {colors[i % len(colors)]} tiles left {cities[i % len(cities)]} "
            f"on day {i % 28 + 1} with {i * 13 % 97} units."
        )
    lines.append("")
    lines.append("What is the code word stated at the very beginning? Answer with the code word only.")
    return "\n".join(lines)


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
    ("past_8k", long_context_prompt()),
]

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(HF_DIR)
config = AutoConfig.from_pretrained(HF_DIR)
rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
assert rope and rope.get("rope_type") == "llama3", f"expected llama3 rope scaling in {HF_DIR}/config.json, got {rope}"

gguf_model = AutoModelForCausalLM.from_pretrained(HF_DIR, gguf_file=GGUF_PATH, dtype=torch.float32, device_map="cpu")
state = gguf_model.state_dict()
del gguf_model

model = LlamaForCausalLM(config).to(torch.float32)
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [k for k in missing if k != "lm_head.weight"]
assert not missing and not unexpected, f"state dict mismatch: missing={missing} unexpected={unexpected}"
model.tie_weights()
model.eval()
del state

inv_freq = model.model.rotary_emb.inv_freq
head_dim = config.head_dim
plain = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
assert not torch.allclose(inv_freq, plain), "rotary_emb.inv_freq is unscaled; llama3 scaling not applied"
assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight), "lm_head not tied to embed_tokens"


def run_case(name, prompt):
    messages = [{"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, date_string=DATE_STRING)
    input_ids_list = tok(rendered, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([input_ids_list])
    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1)
        logits_last = out.logits[0, -1, :].float()
        top = torch.topk(logits_last, TOP_K)
        gen = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=N_DECODE,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.eos_token_id,
        )
        continuation = gen[0, input_ids.shape[1]:].tolist()
    return {
        "name": name,
        "rendered": rendered,
        "input_ids": input_ids_list,
        "prefill_top20": {
            "ids": top.indices.tolist(),
            "values": [round(v, 6) for v in top.values.tolist()],
        },
        "greedy_continuation": continuation,
        "continuation_text": tok.decode(continuation, skip_special_tokens=True),
    }


cases_out = []
for name, prompt in CASES:
    case = run_case(name, prompt)
    cases_out.append(case)
    print(f"[{name}] seq={len(case['input_ids'])} top1={case['prefill_top20']['ids'][0]} continuation={case['continuation_text'][:80]!r}", flush=True)

fixture = {
    "source": (
        f"transformers LlamaForCausalLM(config.json with llama3 rope_scaling) float32 CPU, weights dequantized from "
        f"{os.path.basename(GGUF_PATH)}, chat template date_string={DATE_STRING!r}"
    ),
    "cases": cases_out,
}

out_path = Path(FIXTURE_OUT)
out_path.write_text(json.dumps(fixture, indent=1))
print(f"wrote {out_path}")
