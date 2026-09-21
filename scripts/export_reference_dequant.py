"""
Generate a "dequant-matched" reference: load Salesforce/xLAM-2-3b-fc-r on
CPU in float32, then overwrite every weight with the Q4_0 (and Q6_K for the
embedding table)-dequantised values pulled straight out of the local
xLAM-2-3b-fc-r-q4_0.gguf. Forward-pass logits from this model isolate
quantisation noise from bugs in the Rust port: the Rust port also runs
Q4_0 weights (dequantised at load, f32 compute), so this reference should
match it far more closely than the plain bf16 reference does.

*** MEMORY WARNING ***
This loads a full float32 copy of the 3B model (~12 GB) on CPU, plus
briefly (one tensor at a time) a dequantised copy of each GGUF tensor
being copied in. Run this alone -- nothing else memory-hungry (browsers,
other model servers, GPU workers sharing unified memory) at the same
time. See scripts/README.md for the exact command and expected duration.

venv mon ami: run with scripts/.venv/bin/python (see scripts/README.md).

Tensor naming (llama.cpp GGUF <-> HF state_dict), per docs/MODELS.md section 2:

    token_embd.weight              <-> model.embed_tokens.weight   (tied to lm_head.weight)
    blk.N.attn_q.{weight,bias}     <-> model.layers.N.self_attn.q_proj.{weight,bias}
    blk.N.attn_k.{weight,bias}     <-> model.layers.N.self_attn.k_proj.{weight,bias}
    blk.N.attn_v.{weight,bias}     <-> model.layers.N.self_attn.v_proj.{weight,bias}
    blk.N.attn_output.weight       <-> model.layers.N.self_attn.o_proj.weight
    blk.N.attn_norm.weight         <-> model.layers.N.input_layernorm.weight
    blk.N.ffn_gate.weight          <-> model.layers.N.mlp.gate_proj.weight
    blk.N.ffn_up.weight            <-> model.layers.N.mlp.up_proj.weight
    blk.N.ffn_down.weight          <-> model.layers.N.mlp.down_proj.weight
    blk.N.ffn_norm.weight          <-> model.layers.N.post_attention_layernorm.weight
    output_norm.weight             <-> model.norm.weight

`output.weight` (a separate LM head) does not exist in this GGUF --
tie_word_embeddings=true, so lm_head.weight is the *same* parameter
object as model.embed_tokens.weight in the HF model; we only overwrite
the embed_tokens copy and rely on the existing tie (asserted below,
not re-tied by hand).

Tensor orientation: verified by dry-reading the GGUF (no model load) that
gguf.quants.dequantize(tensor.data, tensor.tensor_type) already returns
numpy arrays in HF's [out_features, in_features] convention -- e.g.
token_embd.weight dequantises to shape (151936, 2048), matching HF's
model.embed_tokens.weight, and blk.0.ffn_down.weight dequantises to
(2048, 11008), matching HF's down_proj.weight ([hidden, intermediate]).
No transpose needed relative to gguf's `ne` field (which is reported
reversed, ggml-internal order).

Outputs:
- models/reference/xlam-2-3b-fc-r/<name>.dequant.logits.npy   float32 [seq_len, vocab_size]
- fixtures/reference/logits/<name>.dequant.json               argmax/top5/greedy, same shape as the bf16 *.json
- models/reference/xlam-2-3b-fc-r/<name>.dequant.hidden.npz   (--hidden only) last-position hidden states
- stdout: per-position argmax agreement + last-position cosine vs the existing bf16 reference
"""
import argparse
import json
import os

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer

# REPO_ROOT defaults to this script's checkout; LLM_MODELS_DIR defaults to
# `./models` next to it (see scripts/README.md).
REPO_ROOT = os.environ.get(
    "LLM_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.join(REPO_ROOT, "models"))
MODEL_DIR = os.path.join(MODELS_DIR, "hf/xLAM-2-3b-fc-r")
GGUF_PATH = os.path.join(MODELS_DIR, "gguf/xlam-2-3b-fc-r/xLAM-2-3b-fc-r-q4_0.gguf")
INPUTS_DIR = os.path.join(REPO_ROOT, "fixtures/reference/inputs")
LOGITS_JSON_DIR = os.path.join(REPO_ROOT, "fixtures/reference/logits")
LOGITS_NPY_DIR = os.path.join(MODELS_DIR, "reference/xlam-2-3b-fc-r")

NUM_LAYERS = 36
HIDDEN_LAYER_IDS = [0, 9, 18, 27, 35]
TOP_K = 5
MAX_NEW_TOKENS = 32


def gguf_name_to_hf_names(name):
    """Map one llama.cpp GGUF tensor name to the HF state_dict key(s) it
    fills. Returns a list (usually length 1) since some GGUF tensors are
    the sole source for a single HF parameter -- kept a list only for
    uniformity, no GGUF tensor here maps to more than one HF parameter."""
    if name == "token_embd.weight":
        return ["model.embed_tokens.weight"]
    if name == "output_norm.weight":
        return ["model.norm.weight"]
    if name.startswith("blk."):
        _, layer_str, rest = name.split(".", 2)
        layer = int(layer_str)
        prefix = f"model.layers.{layer}."
        mapping = {
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_q.bias": "self_attn.q_proj.bias",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_k.bias": "self_attn.k_proj.bias",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_v.bias": "self_attn.v_proj.bias",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        if rest not in mapping:
            raise ValueError(f"unmapped GGUF tensor {name!r}")
        return [prefix + mapping[rest]]
    raise ValueError(f"unmapped GGUF tensor {name!r}")


def load_dequant_model():
    print("Loading GGUF header + tensor list ...")
    reader = GGUFReader(GGUF_PATH)

    print(f"Loading {MODEL_DIR} in float32 on CPU (~12GB) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    params = dict(model.named_parameters())

    covered = set()
    for tensor in reader.tensors:
        hf_names = gguf_name_to_hf_names(tensor.name)
        deq = dequantize(tensor.data, tensor.tensor_type)  # numpy float32, HF orientation
        t = torch.from_numpy(np.ascontiguousarray(deq))
        for hf_name in hf_names:
            param = params[hf_name]
            assert param.shape == t.shape, (
                f"{tensor.name} -> {hf_name}: gguf {t.shape} vs hf {param.shape}"
            )
            with torch.no_grad():
                param.copy_(t)
            covered.add(hf_name)
        del deq, t

    expected = set(params.keys()) - {"lm_head.weight"}
    missing = expected - covered
    extra = covered - expected
    assert not missing, f"HF parameters never written from GGUF: {sorted(missing)}"
    assert not extra, f"GGUF tensors mapped to unexpected HF parameters: {sorted(extra)}"
    # tied lm_head.weight: not written directly -- assert it stayed the same
    # storage as model.embed_tokens.weight rather than re-tying by hand.
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
    ), "lm_head.weight is not tied to model.embed_tokens.weight as expected"

    print(f"Loaded and overwrote {len(covered)} HF parameters from {len(reader.tensors)} GGUF tensors.")
    return model


def run_one(model, tokenizer, name, want_hidden):
    with open(os.path.join(INPUTS_DIR, f"{name}.json")) as f:
        spec = json.load(f)
    messages = spec["messages"]
    tools = spec["tools"] if spec["tools"] else None

    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    hidden_capture = {}
    hook_handle = None
    if want_hidden:
        def hook(_module, _inputs, output):
            hidden_capture["layer_35_raw"] = output[0][0, -1, :].detach().clone()

        hook_handle = model.model.layers[35].register_forward_hook(hook)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=want_hidden)
        logits = out.logits[0]  # [seq_len, vocab_size]

    if hook_handle is not None:
        hook_handle.remove()

    logits_f32 = logits.to(torch.float32).numpy()
    np.save(os.path.join(LOGITS_NPY_DIR, f"{name}.dequant.logits.npy"), logits_f32)

    seq_len = logits_f32.shape[0]
    argmax_per_pos = logits_f32.argmax(axis=-1).tolist()

    last = logits_f32[-1]
    top5_idx = np.argsort(last)[::-1][:TOP_K]
    top5 = [(int(i), float(last[i])) for i in top5_idx]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True
        )
    new_ids = gen_ids[0, input_ids.shape[1]:]
    greedy_text = tokenizer.decode(new_ids, skip_special_tokens=False)

    result = {
        "seq_len": seq_len,
        "dtype": "float32",
        "device": "cpu",
        "weights": "q4_0-dequant (q6_k for token_embd)",
        "argmax_per_position": argmax_per_pos,
        "top5_last_position": top5,
        "greedy_first_32_tokens_text": greedy_text,
        "greedy_first_32_token_ids": new_ids.tolist(),
    }
    with open(os.path.join(LOGITS_JSON_DIR, f"{name}.dequant.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"{name}: seq_len={seq_len} logits_npy={logits_f32.shape} greedy={greedy_text!r}")

    if want_hidden:
        hs = out.hidden_states  # tuple, len NUM_LAYERS+1; hs[k] = output of layer k-1 (raw), hs[0] = embeddings, hs[-1] = final normed
        npz = {}
        for layer in HIDDEN_LAYER_IDS:
            if layer == 35:
                continue  # captured raw via hook below (hs[-1] is post-norm, not raw layer 35 output)
            npz[f"layer_{layer}"] = hs[layer + 1][0, -1, :].numpy()
        npz["layer_35_raw"] = hidden_capture["layer_35_raw"].numpy()
        npz["final_normed"] = hs[-1][0, -1, :].numpy()
        np.savez(os.path.join(LOGITS_NPY_DIR, f"{name}.dequant.hidden.npz"), **npz)
        print(f"{name}: hidden states saved for layers {HIDDEN_LAYER_IDS} + final_normed")

    return logits_f32, argmax_per_pos


def compare_to_bf16(name, dequant_logits, dequant_argmax):
    bf16_json_path = os.path.join(LOGITS_JSON_DIR, f"{name}.json")
    if not os.path.exists(bf16_json_path):
        print(f"{name}: no bf16 reference json at {bf16_json_path}, skipping comparison")
        return
    with open(bf16_json_path) as f:
        bf16 = json.load(f)
    bf16_argmax = bf16["argmax_per_position"]

    n = min(len(bf16_argmax), len(dequant_argmax))
    matches = sum(1 for i in range(n) if bf16_argmax[i] == dequant_argmax[i])
    print(f"{name}: bf16 vs dequant argmax agreement: {matches}/{n}")
    mismatched = [i for i in range(n) if bf16_argmax[i] != dequant_argmax[i]]
    if mismatched:
        print(f"{name}: mismatched positions: {mismatched}")

    bf16_npy_path = os.path.join(LOGITS_NPY_DIR, f"{name}.logits.npy")
    if os.path.exists(bf16_npy_path):
        bf16_logits = np.load(bf16_npy_path).astype(np.float32)
        a = bf16_logits[-1]
        b = dequant_logits[-1]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"{name}: bf16 vs dequant cosine at last position: {cosine:.6f}")
    else:
        print(f"{name}: no bf16 logits.npy at {bf16_npy_path}, skipping cosine")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["01_no_tools"],
        help="input fixture names to run (default: 01_no_tools only -- "
        "02/03 prefill in f32 on CPU is slow, ~5-10 min for 03's 2225 tokens)",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="also dump last-position hidden states at layers "
        f"{HIDDEN_LAYER_IDS} + final normed hidden, to <name>.dequant.hidden.npz",
    )
    args = parser.parse_args()

    print("*** This loads a ~12GB float32 model on CPU. Run alone. ***")

    os.makedirs(LOGITS_JSON_DIR, exist_ok=True)
    os.makedirs(LOGITS_NPY_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_dequant_model()

    for name in args.inputs:
        logits, argmax = run_one(model, tokenizer, name, args.hidden)
        compare_to_bf16(name, logits, argmax)


if __name__ == "__main__":
    main()
