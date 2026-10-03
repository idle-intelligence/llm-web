//! Raw wgpu + hand-written WGSL Qwen2 engine: no Burn, no CubeCL, no
//! training. See this crate's `README.md`-equivalent doc comments in
//! `model.rs` (forward pass), `gguf.rs` (parsing), `engine.rs`/`pool.rs`
//! (wgpu plumbing, ported from `t0-web/crates/t0-fast`).

pub mod chat_template;
pub mod config;
pub mod cpu;
pub mod cpu_kernels;
#[cfg(feature = "threads")]
mod cpu_team;
pub mod engine;
pub mod generate;
pub mod gguf;
pub mod lora;
pub mod model;
pub mod pool;
pub mod profile_report;
pub mod quant;
pub mod sampling;
#[cfg(feature = "web")]
pub mod web;

/// wasm-bindgen-rayon's worker-pool bootstrap, re-exported so the page can
/// call `await initThreadPool(navigator.hardwareConcurrency)` once (after
/// wasm `init()`, before any `LeanEngineCpu` call) to size rayon's global
/// pool on wasm - `cpu.rs`'s `linear_threads` then sees
/// `rayon::current_num_threads() > 1` and takes the threaded path. Native
/// builds never call this: rayon's own default pool sizing (`available_parallelism`)
/// already applies without it.
#[cfg(feature = "wasm-mt")]
pub use wasm_bindgen_rayon::init_thread_pool;
