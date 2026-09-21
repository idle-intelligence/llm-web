# Reference fixtures

Three kinds of ground truth are generated here, from two different scripts
(`scripts/export_reference.py` and `scripts/export_reference_dequant.py`),
for comparing the Rust (Burn/wgpu) port against PyTorch at different
precision points:

1. **bf16, MPS (or CPU), original weights** — `<name>.json` in
   `logits/`, rendered/tokenized inputs in `rendered/`. The unquantised
   model's own numbers: closest to "what the model actually does," but not
   directly comparable to a Q4_0 + f32-compute Rust port, since bf16 vs.
   Q4_0-dequant-then-f32 are different sources of numerical error.

2. **Q4_0-dequant-matched, f32, CPU** — `<name>.dequant.json` in
   `logits/`. Same architecture and inputs, but every weight is the exact
   Q4_0-dequantised (Q6_K for the embedding table) value pulled from the
   local `xLAM-2-3b-fc-r-q4_0.gguf`, loaded into an f32 HF model and run in
   f32 on CPU. This is the reference that should track the Rust port
   closely — the port also dequantises Q4_0 to f32 at load and computes in
   f32. Comparing (1) vs (2) tells you how much of any port-vs-bf16
   mismatch is just quantisation noise; comparing (2) vs. the Rust port's
   own output isolates actual port bugs.

3. **Hidden states, dequant-matched** — `<name>.dequant.hidden.npz`
   (outside the repo, in `models/reference/xlam-2-3b-fc-r/`, generated with
   `--hidden`). Last-position hidden state vectors at transformer layers
   {0, 9, 18, 27, 35} plus the final post-`model.norm` hidden state, all
   from the same Q4_0-dequant-matched model as (2). For narrowing down
   *where* in the 36-layer stack the Rust port and this reference start to
   diverge, rather than only comparing final logits.

See `scripts/README.md` for exact commands, memory requirements, and the
GGUF-tensor-name <-> HF-parameter-name mapping used to build (2) and (3).
