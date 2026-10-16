"""Repetition penalty fixture: transformers' RepetitionPenaltyLogitsProcessor
on real logits, for the parity tests of lean's CPU and GPU samplers.

For each model and prompt: the chat-templated prompt plus a greedy
continuation of N_CONTINUE tokens (so the history repeats ids) is the
`input_ids`; the f32 CPU forward gives the next-token logits; the processor
is applied to them with the whole `input_ids`, as `generate` does. The
fixture keeps, for every distinct id of the history, the raw logit and the
processed one (f32 bit patterns). The processor leaves every other logit
unchanged, which this script asserts.

Models are the Q4_0 GGUFs lean reads, dequantized on load by transformers,
with the tokenizer from a directory holding tokenizer.json, via env vars:
  LEAN_GGUF_SMOLLM2_360M, LEAN_TOKENIZER_SMOLLM2_360M   SmolLM2-360M-Instruct
  LEAN_GGUF_QWEN25_0_5B,  LEAN_TOKENIZER_QWEN25_0_5B    Qwen2.5-0.5B-Instruct

venv: crates/lean-core/reference/.venv (gitignored), torch + transformers.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

N_CONTINUE = 48
PENALTIES = [1.1, 1.3]
PROMPTS = [
    "Write a short list of five fruits, one per line, each followed by its color.",
    "Repeat after me three times: the cat sat on the mat.",
]
MODELS = [("smollm2_360m", "SMOLLM2_360M"), ("qwen25_0_5b", "QWEN25_0_5B")]


def bits(x):
    return int(np.float32(x).view(np.uint32))


def main():
    cases = []
    for name, env in MODELS:
        gguf = os.environ.get(f"LEAN_GGUF_{env}")
        tok_dir = os.environ.get(f"LEAN_TOKENIZER_{env}")
        if not gguf or not tok_dir:
            raise SystemExit(f"set LEAN_GGUF_{env} and LEAN_TOKENIZER_{env}")
        tok = AutoTokenizer.from_pretrained(tok_dir)
        model = AutoModelForCausalLM.from_pretrained(os.path.dirname(gguf), gguf_file=os.path.basename(gguf), dtype=torch.float32, device_map="cpu").eval()
        for prompt in PROMPTS:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            with torch.no_grad():
                full = model.generate(ids, max_new_tokens=N_CONTINUE, do_sample=False, min_new_tokens=N_CONTINUE)
                logits = model(full).logits[:, -1, :].float()
            hist = full[0].tolist()
            distinct = sorted(set(hist))
            for penalty in PENALTIES:
                out = RepetitionPenaltyLogitsProcessor(penalty=penalty)(full, logits.clone())
                raw = logits[0].numpy()
                got = out[0].numpy()
                others = np.ones(raw.shape[0], dtype=bool)
                others[distinct] = False
                assert np.array_equal(raw[others].view(np.uint32), got[others].view(np.uint32))
                counts = {i: hist.count(i) for i in distinct}
                cases.append(
                    {
                        "model": name,
                        "prompt": prompt,
                        "penalty": penalty,
                        "vocab": int(raw.shape[0]),
                        "input_ids": hist,
                        "max_occurrences": max(counts.values()),
                        "distinct": [[i, bits(raw[i]), bits(got[i])] for i in distinct],
                    }
                )
                print(name, repr(prompt[:30]), penalty, len(hist), "ids,", len(distinct), "distinct, max count", max(counts.values()))
    out = Path(__file__).parent / "fixture_rep_penalty.json"
    out.write_text(json.dumps({"transformers": __import__("transformers").__version__, "cases": cases}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
