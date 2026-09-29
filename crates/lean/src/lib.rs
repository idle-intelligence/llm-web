//! Raw wgpu + hand-written WGSL Qwen2 engine: no Burn, no CubeCL, no
//! training. See this crate's `README.md`-equivalent doc comments in
//! `model.rs` (forward pass), `gguf.rs` (parsing), `engine.rs`/`pool.rs`
//! (wgpu plumbing, ported from `t0-web/crates/t0-fast`).

pub mod chat_template;
pub mod config;
pub mod cpu;
pub mod cpu_kernels;
pub mod engine;
pub mod gguf;
pub mod lora;
pub mod model;
pub mod pool;
pub mod profile_report;
pub mod quant;
#[cfg(feature = "web")]
pub mod web;
