//! lean: a small LLM inference engine on raw wgpu + hand-written WGSL
//! kernels (and SIMD CPU kernels), from one source for native and wasm.
//!
//! The engine is [`Lean`]: text in, text out. [`Lean::load`] picks a
//! backend by capability (WebGPU, then CPU threads, then one CPU thread),
//! [`Lean::generate`] runs one inference turn over the given text, and
//! [`Lean::token_count`] tokenizes without running anything,
//! [`Lean::prefill`] makes a prefix resident without generating, and
//! [`Lean::save_prefix`] / [`Lean::restore_prefix`] move a prefix's KV cache
//! to and from bytes (see `prefix_image.rs` for the format). The engine
//! applies no chat template and keeps no conversation. It reuses the KV
//! cache of its previous call for the longest matching token prefix, which
//! serves one client per engine: give each client its own `Lean`.
//!
//! The forward pass underneath (GGUF loading, GPU and CPU kernels, KV
//! cache, LoRA) is the `lean-core` crate, for callers that work on token
//! ids and logits directly.

mod error;
mod generate;
mod lean;
mod prefix_image;
mod sampling;
mod turn;
#[cfg(feature = "web")]
mod web;
// Links lean-chat into the web package, so its wasm-bindgen exports
// (ChatTemplate, chatRender, chatTrimOldestExchange) ship in the same wasm.
#[cfg(feature = "web")]
extern crate lean_chat as _;
#[cfg(feature = "diag")]
mod web_diag;
#[cfg(test)]
mod tests;

pub use error::LeanError;
pub use lean::{AbortFlag, Backend, GgufSource, Lean, LoadOptions, ReadSeek, TokenizerSource};
pub use prefix_image::PrefixImageError;
pub use turn::GenParams;
#[cfg(feature = "web")]
pub use web::OpfsReader;

/// wasm-bindgen-rayon's worker-pool bootstrap, re-exported so the page can
/// call `await initThreadPool(navigator.hardwareConcurrency)` once (after
/// wasm `init()`, before any `LeanEngineCpu` call) to size rayon's global
/// pool on wasm - `cpu.rs`'s `linear_threads` then sees
/// `rayon::current_num_threads() > 1` and takes the threaded path. Native
/// builds never call this: rayon's own default pool sizing (`available_parallelism`)
/// already applies without it.
#[cfg(feature = "wasm-mt")]
pub use wasm_bindgen_rayon::init_thread_pool;
