//! lean-core: the forward pass of lean, on raw wgpu + hand-written WGSL
//! kernels and on SIMD CPU kernels, from one source for native and wasm.
//!
//! This is the layer below lean's text-in/text-out engine, for callers that
//! work on token ids and logits directly (scoring, packed batches, custom
//! masks and positions, LoRA adapters):
//!
//! - [`engine::Engine`]: the wgpu device, queue and compute pipelines.
//! - [`model::GpuModel`] / [`cpu::CpuModel`]: a GGUF model loaded on the GPU
//!   (quantized weights dequantized inside the kernels) or on the CPU, with
//!   `apply_lora` / `clear_lora` and `embed_head_sliced` (logits for a few
//!   token ids only, through the tied embedding rows).
//! - [`model::KvCache`] / [`cpu::CpuKvCache`]: the KV cache, `kv_len`
//!   positions resident out of `max_ctx`.
//! - [`model::ForwardSpec`]: explicit positions and an attention mask for a
//!   chunk of tokens; [`model::forward_chunk_spec`] /
//!   [`cpu::forward_chunk_spec`] run it and return the final hidden states.
//! - [`model::forward_prefill`], [`model::forward_decode_step`] and their
//!   CPU twins: the plain causal forward pass returning logits.
//! - [`config::Qwen2Config`]: the architecture read from the GGUF metadata.
//!
//! Supported architectures, tensor types and kernels are listed in the
//! crate README.

pub mod config;
pub mod cpu;
mod cpu_kernels;
#[cfg(feature = "threads")]
mod cpu_team;
pub mod engine;
mod gguf;
mod lora;
pub mod model;
mod pool;
pub mod resident_sampling;
pub mod profile_report;
mod quant;

#[cfg(test)]
mod tests;
