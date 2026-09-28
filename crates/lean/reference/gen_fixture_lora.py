"""Reference fixture for crates/lean's runtime LoRA + position/mask + sliced
lm-head path (consumer survey gap items 4/5), built from llm-life's own
variant A prompt shape and a real published LoRA adapter.

Loads the same Q4_0 GGUF via transformers' GGUF loader (dequantized on
load, same as gen_fixture.py), parses one LLMLIFE2 LoRA file directly
(struct-unpacking the byte format documented in crates/lean/src/lora.rs and
crates/llm-wasm/src/lora.rs - this is the format actually published on
idle-intelligence/llm-of-life-lora; no safetensors/PEFT conversion exists
yet), applies its q/k/v/o deltas via forward hooks on
`model.model.layers[i].self_attn.{q,k,v,o}_proj` (delta(x) = (x @ a) @ b *
alpha/rank, added to the base projection's output - matches lora.rs's
documented merge math exactly), and records the "0"/"1" token logits at the
prompt's last position (the "Next: " token, per llm-life's
`variant_a::cell_prompt` doc comment on why the prompt ends there).

Prompt text: `variant_a::rules_prefix(Rule::life())` + `variant_a::cell_prompt`
for one fixed (neighbors, self) case, exactly as `crates/lean/tests/
lora_parity.rs` builds it - see that file for the shared literal strings
(this script and that test must stay byte-identical since neither imports
the other's language's string).

venv: crates/lean/reference/.venv (gitignored, same one gen_fixture.py uses).

  LEAN_GGUF            path to the Q4_0 GGUF file
  LEAN_TOKENIZER_DIR    directory with tokenizer.json + tokenizer_config.json
  LEAN_LORA_BIN         path to an LLMLIFE2 .bin adapter (e.g. llm-life's
                        artifacts/lora-a-rules-300.bin)
"""
import json
import os
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_PATH = os.environ.get("LEAN_GGUF")
TOKENIZER_DIR = os.environ.get("LEAN_TOKENIZER_DIR")
LORA_BIN = os.environ.get("LEAN_LORA_BIN")
if not GGUF_PATH or not TOKENIZER_DIR or not LORA_BIN:
    raise SystemExit("set LEAN_GGUF, LEAN_TOKENIZER_DIR and LEAN_LORA_BIN before running gen_fixture_lora.py")

RULE_STRING = "B3/S23"  # life::Rule::life().to_rulestring()

RULES_PREFIX = (
    f"Cellular automaton, rule {RULE_STRING}. Each cell is 0 (dead) or 1 (alive).\n"
    "A live cell with 2 or 3 live neighbors stays 1, otherwise it becomes 0.\n"
    "A dead cell with exactly 3 live neighbors becomes 1, otherwise it stays 0.\n"
    "For each cell, answer with one digit: its next state.\n"
)


def cell_prompt(neighbors, self_state):
    s = "Neighbors:"
    for n in neighbors:
        s += " " + ("1" if n else "0")
    s += " / Self: " + ("1" if self_state else "0") + " / Next: "
    return s


# One fixed case: a birth case (dead cell, exactly 3 live neighbors) - the
# `variant_a::fewshot_examples` case most likely to separate "0" and "1"
# logits meaningfully rather than being a near-certain "0" everywhere.
NEIGHBORS = [1, 1, 1, 0, 0, 0, 0, 0]
SELF = 0
PROMPT = RULES_PREFIX + cell_prompt(NEIGHBORS, SELF)


def parse_llmlife2(path):
    data = Path(path).read_bytes()
    assert data[0:8] == b"LLMLIFE2", "bad magic"
    rank, alpha, mlp = struct.unpack_from("<IfB", data, 8)
    assert mlp == 0, "mlp adapters not supported by this script"
    (n_tensors,) = struct.unpack_from("<I", data, 17)
    off = 21
    tensors = []
    for _ in range(n_tensors):
        rows, cols = struct.unpack_from("<II", data, off)
        off += 8
        count = rows * cols
        vals = struct.unpack_from(f"<{count}f", data, off)
        off += count * 4
        tensors.append(torch.tensor(vals, dtype=torch.float32).reshape(rows, cols))
    assert off == len(data), "trailing bytes"
    return rank, alpha, tensors


def main():
    rank, alpha, tensors = parse_llmlife2(LORA_BIN)
    scale = alpha / rank
    print(f"loaded LoRA: rank={rank} alpha={alpha} n_tensors={len(tensors)}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_DIR,
        gguf_file=GGUF_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Tensor order: per layer, [q, k, v, o], each (a, b). a: [in, rank],
    # b: [rank, out] - see lora.rs's module doc.
    assert len(tensors) == num_layers * 4 * 2, f"expected {num_layers * 4 * 2} tensors, got {len(tensors)}"
    layers = []
    it = iter(tensors)
    for _ in range(num_layers):
        proj = {}
        for name in ("q", "k", "v", "o"):
            a = next(it)
            b = next(it)
            proj[name] = (a, b)
        layers.append(proj)

    dead_id = tok.encode("0", add_special_tokens=False)
    alive_id = tok.encode("1", add_special_tokens=False)
    assert len(dead_id) == 1 and len(alive_id) == 1, "'0'/'1' must be single tokens"
    dead_id, alive_id = dead_id[0], alive_id[0]

    input_ids = tok.encode(PROMPT, add_special_tokens=False)
    print(f"prompt tokenized to {len(input_ids)} ids")

    def run(with_lora):
        handles = []
        if with_lora:
            for i, proj in enumerate(layers):
                attn = model.model.layers[i].self_attn
                for name, module_attr in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
                    a, b = proj[name]

                    def hook(mod, inp, out, a=a, b=b):
                        x = inp[0]
                        delta = (x.float() @ a) @ b * scale
                        return out + delta.to(out.dtype)

                    handles.append(getattr(attn, module_attr).register_forward_hook(hook))
        try:
            with torch.no_grad():
                out = model(torch.tensor([input_ids]))
                last = out.logits[0, -1, :].float()
                return {"dead": round(last[dead_id].item(), 6), "alive": round(last[alive_id].item(), 6)}
        finally:
            for h in handles:
                h.remove()

    base_logits = run(with_lora=False)
    lora_logits = run(with_lora=True)
    print(f"base:  dead={base_logits['dead']} alive={base_logits['alive']}")
    print(f"+lora: dead={lora_logits['dead']} alive={lora_logits['alive']}")

    fixture = {
        "source": "transformers GGUF-loaded base + hand-applied LLMLIFE2 LoRA deltas (forward hooks), see this file's doc comment",
        "prompt": PROMPT,
        "input_ids": input_ids,
        "dead_token_id": dead_id,
        "alive_token_id": alive_id,
        "rank": rank,
        "alpha": alpha,
        "base_logits": base_logits,
        "lora_logits": lora_logits,
    }
    out_path = Path(__file__).parent / "fixture_lora.json"
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
