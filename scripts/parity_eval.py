"""
Parity check: for a handful of Sonos-eval cases, does the HF bf16 reference
model's FIRST generated tool call differ from the Rust (Burn+wgpu) port's
first call on the byte-identical rendered prompt? Answers "model vs port"
for the ~25%-correct eval result (eval/results/2026-09-10-summary.md):
if HF invents ids on the same prompts the port does, it's the model
(xLAM-2-3b-fc-r), not the Rust port.

venv mon ami: run with scripts/.venv/bin/python (see scripts/README.md).

Builds messages [system, user(utterance)] + the 12-tool set
(fixtures/sonos/tools-12.json, converted via make_inputs.mcp_to_function_tools)
for the DEFAULT system prompt, for 8 cases from eval/utterances.json (s01,
s02, s06, s08, s09, m01, m04, m07). Renders with apply_chat_template(...,
add_generation_prompt=True) and tokenizes exactly as export_reference.py
does (tokenizer(rendered, add_special_tokens=False)).

Writes, per case id, under OUT_DIR:
  <id>.tokens.json   -- flat token-id array, same shape llm-agent run --tokens expects
  <id>.rendered.txt  -- exact rendered prompt text
  <id>.hf.json       -- {generated_ids, generated_text} from HF bf16 MPS greedy decode

Then runs the Rust port (llm-agent run) on each <id>.tokens.json and writes:
  <id>.port.json     -- {generated_ids, generated_text}

Only one model load (HF, sequentially first) and the port binary is invoked
once per case as a separate process afterwards -- HF weights are freed
(del + gc + torch.mps.empty_cache()) before any port process starts, so the
two never share memory at once (16GB M2: PyTorch bf16 ~6GB, port ~3GB GPU,
never concurrent).
"""
import gc
import json
import os
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = "/Users/tc/Code/idle-intelligence/llm-web"
SONOS_DIR = os.path.join(REPO_ROOT, "fixtures/sonos")
UTTERANCES_PATH = os.path.join(REPO_ROOT, "eval/utterances.json")
MODEL_DIR = "/Users/tc/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r"
GGUF_PATH = "/Users/tc/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf"
LLM_AGENT_BIN = os.path.join(REPO_ROOT, "target/release/llm-agent")

OUT_DIR = "/Users/tc/.claude/jobs/ae1e446d/tmp/parity"

DEFAULT_SYSTEM = "You are a helpful home assistant with access to Sonos speaker controls."
CASE_IDS = ["s01", "s02", "s06", "s08", "s09", "m01", "m04", "m07"]
MAX_NEW_TOKENS = 64

# m01 prompt-sensitivity extra: same case, two alternate phrasings of the
# utterance, HF only.
M01_SENSITIVITY = {
    "m01_fixture02_lower": "pause the kitchen",
    "m01_period": "Pause the kitchen.",
}


def mcp_to_function_tools(mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in mcp_tools
    ]


def load_cases():
    with open(UTTERANCES_PATH) as f:
        utterances = json.load(f)
    by_id = {u["id"]: u for u in utterances}
    return [by_id[cid] for cid in CASE_IDS]


def pick_device_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.bfloat16


def build_messages(utterance_text):
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": utterance_text},
    ]


def render_and_tokenize(tokenizer, messages, tools):
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return rendered, token_ids


def run_hf_side():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(SONOS_DIR, "tools-12.json")) as f:
        tools_12_mcp = json.load(f)
    tools = mcp_to_function_tools(tools_12_mcp)

    cases = load_cases()

    device, dtype = pick_device_dtype()
    print(f"HF device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    token_counts = {}

    for case in cases:
        cid = case["id"]
        messages = build_messages(case["utterance"])
        rendered, token_ids = render_and_tokenize(tokenizer, messages, tools)
        token_counts[cid] = len(token_ids)

        with open(os.path.join(OUT_DIR, f"{cid}.rendered.txt"), "w") as f:
            f.write(rendered)
        with open(os.path.join(OUT_DIR, f"{cid}.tokens.json"), "w") as f:
            json.dump(token_ids, f)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        dt = time.time() - t0
        new_ids = gen_ids[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=False)

        with open(os.path.join(OUT_DIR, f"{cid}.hf.json"), "w") as f:
            json.dump({"generated_ids": new_ids, "generated_text": text}, f, indent=2)

        print(f"HF {cid}: seq_len={len(token_ids)} gen_dt={dt:.2f}s text={text[:120]!r}")

    # m01 prompt-sensitivity extras, HF only, same tools/system, alt utterances.
    for tag, alt_utterance in M01_SENSITIVITY.items():
        messages = build_messages(alt_utterance)
        rendered, token_ids = render_and_tokenize(tokenizer, messages, tools)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        new_ids = gen_ids[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        with open(os.path.join(OUT_DIR, f"{tag}.hf.json"), "w") as f:
            json.dump(
                {"utterance": alt_utterance, "generated_ids": new_ids, "generated_text": text},
                f, indent=2,
            )
        print(f"HF {tag} ({alt_utterance!r}): text={text[:120]!r}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    with open(os.path.join(OUT_DIR, "token_counts.json"), "w") as f:
        json.dump(token_counts, f, indent=2)

    print("HF side done, model freed.")


def run_port_side():
    tok_json_path = os.path.join(MODEL_DIR, "tokenizer.json")
    for cid in CASE_IDS:
        tokens_path = os.path.join(OUT_DIR, f"{cid}.tokens.json")
        cmd = [
            LLM_AGENT_BIN, "run",
            "--gguf", GGUF_PATH,
            "--tokens", tokens_path,
            "--max-new", str(MAX_NEW_TOKENS),
            "--tokenizer", tok_json_path,
        ]
        print(f"port {cid}: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"port {cid}: FAILED rc={proc.returncode}\nstderr:\n{proc.stderr}")
            with open(os.path.join(OUT_DIR, f"{cid}.port.json"), "w") as f:
                json.dump({"error": proc.stderr, "returncode": proc.returncode}, f, indent=2)
            continue

        stdout = proc.stdout
        gen_ids = None
        gen_text = None
        for line in stdout.splitlines():
            if line.startswith("generated ids:"):
                ids_str = line[len("generated ids:"):].strip()
                gen_ids = json.loads(ids_str)
            elif line.startswith("generated text:"):
                gen_text = line[len("generated text:"):].strip()

        with open(os.path.join(OUT_DIR, f"{cid}.port.json"), "w") as f:
            json.dump(
                {"generated_ids": gen_ids, "generated_text": gen_text, "stdout": stdout, "stderr": proc.stderr},
                f, indent=2,
            )
        print(f"port {cid}: dt={dt:.2f}s text={(gen_text or '')[:120]!r}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("hf", "all"):
        run_hf_side()
    if stage in ("port", "all"):
        run_port_side()


if __name__ == "__main__":
    main()
