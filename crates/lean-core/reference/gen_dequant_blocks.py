"""Real-block dequantization fixture for the types lean reads beyond Q6_K:
Q4_K, Q5_0, F16 and BF16. For each (GGUF, tensor) pair given in CASES, picks
a few spread-out blocks from the actual tensor, dequantizes them with
gguf-py's own reference implementation
(`gguf.quants.dequantize`), and writes a small JSON fixture (raw block bytes
+ the expected f32 values as exact bit patterns) that
`src/tests/dequant_reference.rs` checks `gguf::dequantize_*` against,
bit for bit. With `--full DIR`, also writes each tensor's complete reference
dequantization to `DIR/<type>.f32` (little-endian f32) for the ignored
whole-tensor test in the same file.

  LEAN_GGUF_Q4K    a GGUF with Q4_K tensors (Qwen3-0.6B Q4_K_M)
  LEAN_GGUF_Q5_0   a GGUF with Q5_0 tensors (SmolLM2-360M-Instruct Q4_K_M)
  LEAN_GGUF_F16    an F16 GGUF (ggml-org Qwen3-0.6B-f16)
  LEAN_GGUF_BF16   a BF16 GGUF (ggml-org Qwen3-0.6B-BF16)

venv: crates/lean-core/reference/.venv (gitignored).
"""
import json
import os
import sys
from pathlib import Path

import gguf
import numpy as np
from gguf.constants import GGMLQuantizationType as T

CASES = [
    ("Q4_K", "LEAN_GGUF_Q4K", "blk.3.ffn_down.weight", T.Q4_K, 144, 256),
    ("Q5_0", "LEAN_GGUF_Q5_0", "blk.5.attn_q.weight", T.Q5_0, 22, 32),
    ("F16", "LEAN_GGUF_F16", "blk.2.attn_q.weight", T.F16, 2, 1),
    ("BF16", "LEAN_GGUF_BF16", "blk.2.attn_q.weight", T.BF16, 2, 1),
]
N_BLOCKS = 6
HALF_BLOCK = 64  # F16/BF16 are "blocks" of 64 values in the fixture

full_dir = None
if "--full" in sys.argv:
    full_dir = Path(sys.argv[sys.argv.index("--full") + 1])
    full_dir.mkdir(parents=True, exist_ok=True)

out = {"source": "gguf.quants.dequantize on real tensor blocks", "cases": []}
for name, env, tensor_name, qtype, block_bytes, block_vals in CASES:
    path = os.environ.get(env)
    if not path:
        raise SystemExit(f"set {env}")
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.name == tensor_name)
    assert t.tensor_type == qtype, f"{tensor_name} is {t.tensor_type.name}, expected {name}"
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    if block_vals == 1:
        block_vals, block_bytes = HALF_BLOCK, HALF_BLOCK * 2
    rows = raw.reshape(-1, block_bytes)
    n = rows.shape[0]
    picks = [0, 1, n // 3, n // 2, n - 2, n - 1][:N_BLOCKS]
    blocks = []
    for bi in picks:
        deq = gguf.quants.dequantize(rows[bi : bi + 1], qtype).reshape(-1).astype(np.float32)
        assert deq.size == block_vals
        blocks.append({"block_index_in_tensor": bi, "bytes_hex": rows[bi].tobytes().hex(), "expected_bits": [int(b) for b in deq.view(np.uint32)]})
    out["cases"].append({"type": name, "tensor": tensor_name, "block_bytes": block_bytes, "block_values": block_vals, "blocks": blocks})
    if full_dir:
        full = gguf.quants.dequantize(raw.reshape(-1, block_bytes), qtype).reshape(-1).astype("<f4")
        (full_dir / f"{name}.f32").write_bytes(full.tobytes())
        print(name, tensor_name, full.size, "values")

p = Path(__file__).parent / "dequant_reference_blocks.json"
p.write_text(json.dumps(out))
print("wrote", p)
