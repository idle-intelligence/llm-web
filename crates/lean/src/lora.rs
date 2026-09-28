//! Runtime LoRA adapters applied to q/k/v/o, alongside the frozen Q4_0 base
//! (no requantize, no merge into the base weights).
//!
//! On-disk format ("LLMLIFE2"), byte-identical to `crates/llm-wasm/src/lora.rs`
//! (the Burn engine's own loader for the same adapters) and to llm-life's
//! writer (`crates/llm-life/src/train/lora_io.rs`) / reader
//! (`tools/merge/lora_io.py`). This is the format actually published today
//! (`idle-intelligence/llm-of-life-lora` on Hugging Face carries
//! `lora-a-norules-300.bin`, `lora-a-rules-300.bin`, `lora-b-16-s3-300.bin`,
//! `lora-b-32.bin`: all LLMLIFE2, not safetensors; the PEFT/safetensors
//! conversion has not landed yet):
//!
//! ```text
//! 8 bytes  magic "LLMLIFE2"
//! u32 le   rank
//! f32 le   alpha
//! u8       mlp (adapts gate/up too, in addition to q/k/v/o)
//! u32 le   n tensors
//! per tensor: u32 rows, u32 cols, rows*cols f32 le, row-major
//! ```
//!
//! Tensor order: per layer `0..num_layers`, for each of `[q, k, v, o]`
//! always, then `[gate, up]` iff `mlp`, in that order: `a` (`[in_features,
//! rank]`) then `b` (`[rank, out_features]`).
//!
//! This crate only ever applies the q/k/v/o deltas: `mlp`-adapted files are
//! rejected up front rather than silently ignoring the gate/up tensors.
//!
//! Merge math: `delta(x) = (x @ a) @ b * (alpha / rank)`, added to the base
//! projection's output: `y = W_q4 * x + scale * B * (A * x)`. The `alpha /
//! rank` scale is folded into `b` once at upload time (a scalar multiply
//! commutes into either factor of a two-matrix product), so applying a
//! LoRA delta on the GPU is exactly two plain F32 `linear()` calls plus one
//! `add_inplace`: no dedicated scale kernel needed.

use anyhow::{ensure, Context, Result};

use crate::engine::Engine;
use crate::quant::MatMulWeight;

const MAGIC: &[u8; 8] = b"LLMLIFE2";
const HEADER_LEN: usize = 21; // 8 (magic) + 4 (rank) + 4 (alpha) + 1 (mlp) + 4 (n tensors)
const PROJECTIONS: usize = 4; // q, k, v, o

#[derive(Debug)]
struct RawTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

fn read_tensor(bytes: &[u8], off: &mut usize) -> Result<RawTensor> {
    ensure!(*off + 8 <= bytes.len(), "truncated LoRA file: tensor header at {off}");
    let rows = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[*off + 4..*off + 8].try_into().unwrap()) as usize;
    *off += 8;
    let count = rows * cols;
    let need = count * 4;
    ensure!(
        *off + need <= bytes.len(),
        "truncated LoRA file: tensor [{rows}, {cols}] at {off} needs {need} bytes, {} remain",
        bytes.len() - *off
    );
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let s = *off + i * 4;
        data.push(f32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()));
    }
    *off += need;
    Ok(RawTensor { rows, cols, data })
}

#[derive(Debug)]
struct RawProj {
    a: RawTensor,
    b: RawTensor,
}

#[derive(Debug)]
struct RawLayer {
    q: RawProj,
    k: RawProj,
    v: RawProj,
    o: RawProj,
}

/// Parsed LLMLIFE2 file, before GPU upload: the part that's unit-testable
/// without a wgpu device.
#[derive(Debug)]
pub struct RawLoraAdapter {
    pub rank: u32,
    pub alpha: f32,
    pub mlp: bool,
    layers: Vec<RawLayer>,
}

impl RawLoraAdapter {
    /// Parse `bytes` as an LLMLIFE2 file for a model with `num_layers`
    /// transformer blocks. Rejects `mlp` adapters (see module docs) and any
    /// tensor-count/size mismatch against `num_layers`.
    pub fn parse(bytes: &[u8], num_layers: usize) -> Result<Self> {
        ensure!(bytes.len() >= HEADER_LEN, "LoRA file too short ({} bytes)", bytes.len());
        ensure!(&bytes[0..8] == MAGIC, "bad magic: not an LLMLIFE2 LoRA file");
        let rank = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let alpha = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let mlp = bytes[16] != 0;
        ensure!(!mlp, "LoRA file adapts gate/up (mlp=true); this runtime path only applies q/k/v/o deltas");
        let n_tensors = u32::from_le_bytes(bytes[17..21].try_into().unwrap()) as usize;
        let expected = num_layers * PROJECTIONS * 2;
        ensure!(
            n_tensors == expected,
            "LoRA file has {n_tensors} tensors, expected {expected} ({num_layers} layers x {PROJECTIONS} projections x 2 (a, b))"
        );

        let mut off = HEADER_LEN;
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut read_proj = |name: &str| -> Result<RawProj> {
                let a = read_tensor(bytes, &mut off).with_context(|| format!("layer {layer_idx} {name}.a"))?;
                let b = read_tensor(bytes, &mut off).with_context(|| format!("layer {layer_idx} {name}.b"))?;
                ensure!(
                    a.cols == b.rows,
                    "layer {layer_idx} {name}: a is [{}, {}], b is [{}, {}] (rank mismatch)",
                    a.rows,
                    a.cols,
                    b.rows,
                    b.cols
                );
                Ok(RawProj { a, b })
            };
            let q = read_proj("q")?;
            let k = read_proj("k")?;
            let v = read_proj("v")?;
            let o = read_proj("o")?;
            layers.push(RawLayer { q, k, v, o });
        }
        ensure!(off == bytes.len(), "trailing bytes in LoRA file: {}", bytes.len() - off);

        Ok(Self { rank, alpha, mlp, layers })
    }
}

/// One projection's LoRA delta, GPU-resident: `delta(x) = (x @ a) @ b`,
/// where `b`'s values already carry the `alpha / rank` scale (see module
/// doc). Both held as `MatMulWeight::F32` so `model::linear()` runs them
/// with no new kernel.
pub struct LoraProj {
    pub a: MatMulWeight, // [in_features, rank]
    pub b: MatMulWeight, // [rank, out_features], pre-scaled by alpha/rank
    pub rank: u32,
    pub out_features: u32,
}

/// `linear()`'s `MatMulWeight::F32` path expects `w` in PyTorch/GGUF
/// `[out, in]` row-major layout (`shaders/linear.wgsl`'s doc comment: "row n
/// = output channel n"). The LLMLIFE2 file stores `a` as `[in_features,
/// rank]` and `b` as `[rank, out_features]`: the natural `x @ a`, `(..) @
/// b` orientation, i.e. `[in, out]` for both: so both need transposing to
/// `[out, in]` once at upload time, not at every forward call.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

impl LoraProj {
    fn upload(engine: &Engine, label: &str, raw: &RawProj, scale: f32) -> Self {
        // a: [in_features, rank] -> transposed to [rank, in_features].
        let a_t = transpose(&raw.a.data, raw.a.rows, raw.a.cols);
        let a = engine.buf_f32(&a_t, &format!("{label}.a"));
        // b: [rank, out_features] -> transposed to [out_features, rank], scaled by alpha/rank.
        let b_t: Vec<f32> = transpose(&raw.b.data, raw.b.rows, raw.b.cols).iter().map(|&x| x * scale).collect();
        let b = engine.buf_f32(&b_t, &format!("{label}.b"));
        LoraProj {
            a: MatMulWeight::F32 { w: a },
            b: MatMulWeight::F32 { w: b },
            rank: raw.a.cols as u32,
            out_features: raw.b.cols as u32,
        }
    }
}

/// One layer's four LoRA-adapted projections.
pub struct LoraLayer {
    pub q: LoraProj,
    pub k: LoraProj,
    pub v: LoraProj,
    pub o: LoraProj,
}

/// A fully GPU-resident LoRA adapter, one [`LoraLayer`] per transformer
/// block. Applying it (`GpuModel::apply_lora`) does not touch the base
/// Q4_0 weights at all: the base stays frozen and the delta is added to
/// each projection's output on every subsequent forward call. Switching
/// adapters or clearing (`GpuModel::clear_lora`) needs no base reload.
pub struct LoraAdapter {
    pub layers: Vec<LoraLayer>,
}

impl LoraAdapter {
    pub fn from_raw(engine: &Engine, raw: &RawLoraAdapter) -> Self {
        let scale = raw.alpha / raw.rank as f32;
        let layers = raw
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LoraLayer {
                q: LoraProj::upload(engine, &format!("lora{i}.q"), &l.q, scale),
                k: LoraProj::upload(engine, &format!("lora{i}.k"), &l.k, scale),
                v: LoraProj::upload(engine, &format!("lora{i}.v"), &l.v, scale),
                o: LoraProj::upload(engine, &format!("lora{i}.o"), &l.o, scale),
            })
            .collect();
        Self { layers }
    }

    /// Parse + upload in one call: the entry point `web.rs`'s `loadLora`
    /// and `lean-cli`'s native checks use.
    pub fn from_bytes(engine: &Engine, bytes: &[u8], num_layers: usize) -> Result<Self> {
        let raw = RawLoraAdapter::parse(bytes, num_layers)?;
        Ok(Self::from_raw(engine, &raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_file(num_layers: usize, dim: usize, rank: usize, alpha: f32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(rank as u32).to_le_bytes());
        out.extend_from_slice(&alpha.to_le_bytes());
        out.push(0); // mlp = false
        let n_tensors = (num_layers * PROJECTIONS * 2) as u32;
        out.extend_from_slice(&n_tensors.to_le_bytes());

        let mut counter = 0.0f32;
        let mut push_tensor = |out: &mut Vec<u8>, rows: usize, cols: usize| {
            out.extend_from_slice(&(rows as u32).to_le_bytes());
            out.extend_from_slice(&(cols as u32).to_le_bytes());
            for _ in 0..rows * cols {
                out.extend_from_slice(&counter.to_le_bytes());
                counter += 1.0;
            }
        };
        for _ in 0..num_layers {
            for _ in 0..PROJECTIONS {
                push_tensor(&mut out, dim, rank); // a: [in, rank]
                push_tensor(&mut out, rank, dim); // b: [rank, out]
            }
        }
        out
    }

    #[test]
    fn parses_synthetic_file() {
        let bytes = synthetic_file(2, 4, 2, 16.0);
        let raw = RawLoraAdapter::parse(&bytes, 2).expect("parse");
        assert_eq!(raw.rank, 2);
        assert_eq!(raw.alpha, 16.0);
        assert!(!raw.mlp);
        assert_eq!(raw.layers.len(), 2);
        let l0 = &raw.layers[0];
        assert_eq!((l0.q.a.rows, l0.q.a.cols), (4, 2));
        assert_eq!((l0.q.b.rows, l0.q.b.cols), (2, 4));
        assert_eq!(l0.q.a.data[0], 0.0);
        assert_eq!(l0.q.a.data.len(), 8);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic_file(1, 4, 2, 16.0);
        bytes[0] = b'X';
        assert!(RawLoraAdapter::parse(&bytes, 1).is_err());
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let bytes = synthetic_file(2, 4, 2, 16.0);
        assert!(RawLoraAdapter::parse(&bytes, 3).is_err());
    }

    #[test]
    fn rejects_mlp_adapters() {
        let mut bytes = synthetic_file(1, 4, 2, 16.0);
        bytes[16] = 1; // mlp = true
        let err = RawLoraAdapter::parse(&bytes, 1).unwrap_err();
        assert!(err.to_string().contains("mlp"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = synthetic_file(1, 4, 2, 16.0);
        bytes.push(0);
        assert!(RawLoraAdapter::parse(&bytes, 1).is_err());
    }
}
