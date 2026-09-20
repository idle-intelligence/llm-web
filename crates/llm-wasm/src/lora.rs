//! Runtime LoRA adapters applied to `Q4Attention`'s q/k/v/o projections.
//!
//! On-disk format ("LLMLIFE2"), matched byte-for-byte against llm-life's
//! `crates/llm-life/src/train/lora_io.rs` (Rust writer) and
//! `tools/merge/lora_io.py` (Python reader used for the offline merge —
//! see llm-life's `docs/runs/2026-09-20-merge.md`):
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
//! This crate only ever applies the q/k/v/o deltas (`Q4Attention::forward`)
//! — `mlp`-adapted files are rejected up front rather than silently
//! ignoring the gate/up tensors, since llm-life's own `-300` adapters are
//! q/k/v/o-only and a mismatch here should be loud, not silent.
//!
//! Merge math (same convention as the offline GGUF merge): `delta(x) = (x @
//! a) @ b * (alpha / rank)`, added to the base projection's output —
//! `y = W_q4 * x + scale * B * (A * x)`.

use anyhow::{ensure, Context, Result};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};

const MAGIC: &[u8; 8] = b"LLMLIFE2";
const HEADER_LEN: usize = 21; // 8 (magic) + 4 (rank) + 4 (alpha) + 1 (mlp) + 4 (n tensors)
const PROJECTIONS: usize = 4; // q, k, v, o

/// One parsed `[rows, cols]` f32 tensor, still on CPU.
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

/// One projection's raw (still-CPU) LoRA `a`/`b` matrices.
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

/// Parsed LLMLIFE2 file, before GPU upload — the part that's unit-testable
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
        ensure!(
            !mlp,
            "LoRA file adapts gate/up (mlp=true); this runtime path only applies q/k/v/o deltas"
        );
        let n_tensors = u32::from_le_bytes(bytes[17..21].try_into().unwrap()) as usize;
        let expected = num_layers * PROJECTIONS * 2;
        ensure!(
            n_tensors == expected,
            "LoRA file has {n_tensors} tensors, expected {expected} \
             ({num_layers} layers x {PROJECTIONS} projections x 2 (a, b))"
        );

        let mut off = HEADER_LEN;
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut read_proj = |name: &str| -> Result<RawProj> {
                let a = read_tensor(bytes, &mut off)
                    .with_context(|| format!("layer {layer_idx} {name}.a"))?;
                let b = read_tensor(bytes, &mut off)
                    .with_context(|| format!("layer {layer_idx} {name}.b"))?;
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

/// One projection's LoRA delta, resident on GPU: `delta(x) = scale * (x @ a) @ b`.
#[derive(Clone)]
pub struct LoraProj {
    a: Tensor<Wgpu, 2>, // [in_features, rank]
    b: Tensor<Wgpu, 2>, // [rank, out_features]
    scale: f32,
}

impl LoraProj {
    fn upload(raw: &RawProj, scale: f32, device: &WgpuDevice) -> Self {
        let a = Tensor::from_data(TensorData::new(raw.a.data.clone(), [raw.a.rows, raw.a.cols]), device);
        let b = Tensor::from_data(TensorData::new(raw.b.data.clone(), [raw.b.rows, raw.b.cols]), device);
        Self { a, b, scale }
    }

    /// `x`: `[1, T, in_features]` -> `[1, T, out_features]`.
    pub fn delta(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let [batch, t, k] = x.dims();
        let out_features = self.b.dims()[1];
        let x2 = x.reshape([batch * t, k]);
        let xa = x2.matmul(self.a.clone());
        let xab = xa.matmul(self.b.clone());
        xab.mul_scalar(self.scale).reshape([batch, t, out_features])
    }
}

/// One layer's four LoRA-adapted projections.
#[derive(Clone)]
pub struct LoraLayer {
    pub q: LoraProj,
    pub k: LoraProj,
    pub v: LoraProj,
    pub o: LoraProj,
}

/// A fully GPU-resident LoRA adapter, one [`LoraLayer`] per transformer
/// block, ready for `LlmModel::apply_lora`.
pub struct LoraAdapter {
    pub layers: Vec<LoraLayer>,
}

impl LoraAdapter {
    pub fn from_raw(raw: &RawLoraAdapter, device: &WgpuDevice) -> Self {
        let scale = raw.alpha / raw.rank as f32;
        let layers = raw
            .layers
            .iter()
            .map(|l| LoraLayer {
                q: LoraProj::upload(&l.q, scale, device),
                k: LoraProj::upload(&l.k, scale, device),
                v: LoraProj::upload(&l.v, scale, device),
                o: LoraProj::upload(&l.o, scale, device),
            })
            .collect();
        Self { layers }
    }

    /// Parse + upload in one call — the entry point `web.rs`'s `loadAdapter`
    /// and the native verification binaries use.
    pub fn from_bytes(bytes: &[u8], num_layers: usize, device: &WgpuDevice) -> Result<Self> {
        let raw = RawLoraAdapter::parse(bytes, num_layers)?;
        Ok(Self::from_raw(&raw, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tiny synthetic LLMLIFE2 file: `num_layers` layers, q/k/v/o
    /// all `[in, out] = [dim, dim]` at the given `rank`, matching the exact
    /// byte layout `tools/merge/lora_io.py` reads.
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
        // First tensor (layer 0, q.a) starts at counter 0.0, row-major.
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
