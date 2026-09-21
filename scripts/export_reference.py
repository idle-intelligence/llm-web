"""
Generate ground-truth chat-template renders + forward-pass logits for the
xLAM-2-3b-fc-r reference fixtures, for the Rust (Burn/wgpu) port to be
compared against byte-for-byte (rendered prompt) and logit-for-logit
(forward pass on those exact token ids).

venv mon ami: run with scripts/.venv/bin/python (see scripts/README.md).

Tensor/output naming:
- fixtures/reference/rendered/<name>.txt          exact rendered chat-template string (no trailing newline appended)
- fixtures/reference/rendered/<name>.tokens.json  token ids for that string, as encoded by the tokenizer
- fixtures/reference/logits/<name>.json           seq_len, dtype/device, argmax per position, top-5 @ last position, greedy 32-token continuation
- models/reference/xlam-2-3b-fc-r/<name>.logits.npy  float32 [seq_len, vocab_size] full forward-pass logits (outside repo, large)
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# REPO_ROOT defaults to this script's checkout; LLM_MODELS_DIR defaults to
# `./models` next to it (see scripts/README.md).
REPO_ROOT = os.environ.get(
    "LLM_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.join(REPO_ROOT, "models"))
MODEL_DIR = os.path.join(MODELS_DIR, "hf/xLAM-2-3b-fc-r")
INPUTS_DIR = os.path.join(REPO_ROOT, "fixtures/reference/inputs")
RENDERED_DIR = os.path.join(REPO_ROOT, "fixtures/reference/rendered")
LOGITS_JSON_DIR = os.path.join(REPO_ROOT, "fixtures/reference/logits")
LOGITS_NPY_DIR = os.path.join(MODELS_DIR, "reference/xlam-2-3b-fc-r")

INPUT_NAMES = ["01_no_tools", "02_tools_single", "03_tools_multiturn"]
# Rendered/token ids only, no forward pass -- 34-tool prefill would produce a
# multi-GB logits.npy ([seq_len, 151936] float32) for no benefit here.
RENDER_ONLY_NAMES = ["04_tools_all"]

MAX_NEW_TOKENS = 32
TOP_K = 5


def pick_device_dtype():
    # float32 on this 16GB M2 thrashes into swap (12GB of float32 weights leaves
    # no headroom for activations + swap pressure from the rest of the system) —
    # measured: ~1 min of actual CPU progress after 25 min wall time. bfloat16
    # keeps weights at ~6GB and MPS supports bf16 matmuls directly.
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.bfloat16


def render_only(name, tokenizer):
    """Render + tokenize only -- no model, no forward pass. For inputs whose
    full-sequence logits.npy would be multi-GB (e.g. 34-tool prefill)."""
    with open(os.path.join(INPUTS_DIR, f"{name}.json")) as f:
        spec = json.load(f)
    messages = spec["messages"]
    tools = spec["tools"] if spec["tools"] else None

    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    with open(os.path.join(RENDERED_DIR, f"{name}.txt"), "w") as f:
        f.write(rendered)

    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    with open(os.path.join(RENDERED_DIR, f"{name}.tokens.json"), "w") as f:
        json.dump(token_ids, f)

    print(f"{name}: seq_len={len(token_ids)} (render-only, no forward pass)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        metavar="NAME",
        help="render + tokenize this input name only (no model load, no "
        "forward pass, no logits.npy); defaults to names in "
        "RENDER_ONLY_NAMES if passed with no value",
        nargs="?",
        const="__ALL_RENDER_ONLY__",
        default=None,
    )
    args = parser.parse_args()

    os.makedirs(RENDERED_DIR, exist_ok=True)
    os.makedirs(LOGITS_JSON_DIR, exist_ok=True)
    os.makedirs(LOGITS_NPY_DIR, exist_ok=True)

    if args.render_only:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        names = (
            RENDER_ONLY_NAMES
            if args.render_only == "__ALL_RENDER_ONLY__"
            else [args.render_only]
        )
        for name in names:
            render_only(name, tokenizer)
        return

    device, dtype = pick_device_dtype()
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    model.to(device)
    model.eval()
    # config.json ships use_cache=false (presumably for training); force it on
    # for autoregressive generate(), otherwise every new token recomputes
    # attention over the full growing prefix with no KV cache.
    model.config.use_cache = True

    for name in INPUT_NAMES:
        with open(os.path.join(INPUTS_DIR, f"{name}.json")) as f:
            spec = json.load(f)
        messages = spec["messages"]
        tools = spec["tools"] if spec["tools"] else None

        rendered = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        with open(os.path.join(RENDERED_DIR, f"{name}.txt"), "w") as f:
            f.write(rendered)

        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        with open(os.path.join(RENDERED_DIR, f"{name}.tokens.json"), "w") as f:
            json.dump(token_ids, f)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0]  # [seq_len, vocab_size]

        logits_f32 = logits.to(torch.float32).cpu().numpy()
        np.save(os.path.join(LOGITS_NPY_DIR, f"{name}.logits.npy"), logits_f32)

        seq_len = logits_f32.shape[0]
        argmax_per_pos = logits_f32.argmax(axis=-1).tolist()

        last = logits_f32[-1]
        top5_idx = np.argsort(last)[::-1][:TOP_K]
        top5 = [(int(i), float(last[i])) for i in top5_idx]

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        new_ids = gen_ids[0, input_ids.shape[1]:]
        greedy_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        result = {
            "seq_len": seq_len,
            "dtype": str(dtype).replace("torch.", ""),
            "device": device,
            "argmax_per_position": argmax_per_pos,
            "top5_last_position": top5,
            "greedy_first_32_tokens_text": greedy_text,
            "greedy_first_32_token_ids": new_ids.tolist(),
        }
        with open(os.path.join(LOGITS_JSON_DIR, f"{name}.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"{name}: seq_len={seq_len} logits_npy={logits_f32.shape} greedy={greedy_text!r}")


if __name__ == "__main__":
    main()
