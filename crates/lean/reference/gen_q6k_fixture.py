"""Real-block Q6_K dequantization fixture: extracts a handful of actual
Q6_K super-blocks from a GGUF's `output.weight` tensor and dequantizes them
with gguf-py's own reference implementation
(`gguf.quants.dequantize(..., GGMLQuantizationType.Q6_K)`), writing a small
JSON fixture (raw block bytes + expected f32 values, no full tensor/model
weights) that `tests/q6k_reference_dequant.rs` checks
`gguf::dequantize_q6_k` against.

  LEAN_GGUF_Q6K   path to a GGUF whose `output.weight` is Q6_K (e.g.
                  Qwen2.5-3B-Instruct's official "q4_0" GGUF - see
                  `gguf.rs::GgmlDtype`'s doc comment)

venv: crates/lean/reference/.venv (gitignored, shared with the other
gen_fixture_*.py scripts) - needs `gguf` and `numpy` installed alongside
`torch`/`transformers`.
"""
import json
import os
from pathlib import Path

import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType

GGUF_PATH = os.environ.get("LEAN_GGUF_Q6K")
if not GGUF_PATH:
    raise SystemExit("set LEAN_GGUF_Q6K before running gen_q6k_fixture.py")

N_BLOCKS = 8
BLOCK_BYTES = 210

r = gguf.GGUFReader(GGUF_PATH)
tensor = next(t for t in r.tensors if t.name == "output.weight")
assert tensor.tensor_type.name == "Q6_K", f"expected output.weight to be Q6_K, got {tensor.tensor_type.name}"

raw_bytes = tensor.data.tobytes()
n_blocks_total = len(raw_bytes) // BLOCK_BYTES

# A few blocks from spread-out offsets (start/middle/end), not just the
# first few, which could all be degenerate/zero rows.
picks = [0, 1, 2, n_blocks_total // 2, n_blocks_total // 2 + 1, n_blocks_total - 3, n_blocks_total - 2, n_blocks_total - 1][:N_BLOCKS]

blocks_out = []
for bi in picks:
    block = raw_bytes[bi * BLOCK_BYTES : (bi + 1) * BLOCK_BYTES]
    block_np = np.frombuffer(block, dtype=np.uint8).reshape(1, BLOCK_BYTES)
    deq = gguf.quants.dequantize(block_np, GGMLQuantizationType.Q6_K)
    values = deq.reshape(-1).astype(np.float64).tolist()
    assert len(values) == 256
    blocks_out.append({"block_index_in_tensor": bi, "bytes_hex": block.hex(), "expected": [round(v, 6) for v in values]})

fixture = {
    "source": "gguf.quants.dequantize(..., GGMLQuantizationType.Q6_K) on real output.weight blocks",
    "tensor": "output.weight",
    "block_bytes": BLOCK_BYTES,
    "qk_k": 256,
    "blocks": blocks_out,
}
out_path = Path(__file__).parent / "q6k_reference_blocks.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"wrote {out_path}, {len(blocks_out)} blocks")
