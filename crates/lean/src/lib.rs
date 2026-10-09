//! lean: a small LLM inference engine on raw wgpu + hand-written WGSL
//! kernels (and SIMD CPU kernels), from one source for native and wasm.
//!
//! The engine is [`Lean`]: text in, text out. [`Lean::load`] picks a
//! backend by capability (WebGPU, then CPU threads, then one CPU thread),
//! [`Lean::generate`] runs one inference turn over the given text, and
//! [`Lean::token_count`] tokenizes without running anything. The engine
//! applies no chat template and keeps no conversation. It reuses the KV
//! cache of its previous call for the longest matching token prefix, which
//! serves one client per engine: give each client its own `Lean`.
//!
//! The lower modules (`model.rs` forward pass, `gguf.rs` parsing,
//! `engine.rs`/`pool.rs` wgpu plumbing, `cpu.rs` CPU forward pass) stay
//! public for callers that work on token ids directly.

pub mod chat_template;
pub mod config;
pub mod cpu;
pub mod cpu_kernels;
#[cfg(feature = "threads")]
mod cpu_team;
pub mod engine;
pub mod error;
pub mod generate;
pub mod gguf;
mod lean;
pub mod lora;
pub mod model;
pub mod pool;
pub mod profile_report;
pub mod quant;
pub mod sampling;
mod turn;
#[cfg(feature = "web")]
pub mod web;

pub use error::LeanError;
pub use lean::{AbortFlag, Backend, GgufSource, Lean, LoadOptions, ReadSeek, TokenizerSource};
pub use turn::{GenParams, TextStream};

/// wasm-bindgen-rayon's worker-pool bootstrap, re-exported so the page can
/// call `await initThreadPool(navigator.hardwareConcurrency)` once (after
/// wasm `init()`, before any `LeanEngineCpu` call) to size rayon's global
/// pool on wasm - `cpu.rs`'s `linear_threads` then sees
/// `rayon::current_num_threads() > 1` and takes the threaded path. Native
/// builds never call this: rayon's own default pool sizing (`available_parallelism`)
/// already applies without it.
#[cfg(feature = "wasm-mt")]
pub use wasm_bindgen_rayon::init_thread_pool;
