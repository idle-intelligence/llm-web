//! [`Lean`]: the engine, text in and text out. Load once with
//! [`Lean::load`], then call [`Lean::generate`] once per turn. The same
//! surface is exported to JavaScript as the `Lean` class (`web.rs`).
//!
//! The engine applies no chat template and keeps no conversation: the text
//! a caller passes is exactly what gets tokenized. Between calls it keeps
//! the KV cache of the previous call and reuses the longest token prefix
//! the new text shares with it, so a growing conversation only prefills its
//! new suffix; the output is the same with or without that reuse. The
//! reuse serves one client per engine: two clients sharing one `Lean`
//! evict each other's prefix (nothing breaks, it costs a full prefill).
//!
//! When a prompt plus `max_new_tokens` does not fit `max_ctx`, `generate`
//! returns [`LeanError::ContextOverflow`] before touching anything, and the
//! engine stays usable. What to drop and retry is the caller's choice.
//!
//! `load` and `generate` are `async` on every target: in a browser the GPU
//! readback must be awaited. Native callers drive them with any executor,
//! e.g. `pollster::block_on`.

use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokenizers::Tokenizer;

use lean_core::cpu::{CpuKvCache, CpuModel};
use lean_core::engine::Engine;
use crate::error::LeanError;
use lean_core::model::{rope_tables, GpuModel, KvCache};
use crate::turn::{check_fits, cpu_turn, encode, gpu_turn, GenParams, PrefixCache};

#[cfg(feature = "web")]
use wasm_bindgen::prelude::wasm_bindgen;

/// Stops a running `generate`: keep one clone, pass another to the call,
/// and call [`AbortFlag::abort`] from anywhere (another thread, a signal
/// handler, a worker's message handler). Checked once per generated token.
/// JavaScript gets this same type as the `AbortFlag` class.
#[cfg_attr(feature = "web", wasm_bindgen)]
#[derive(Clone, Default)]
pub struct AbortFlag(Arc<AtomicBool>);

#[cfg_attr(feature = "web", wasm_bindgen)]
impl AbortFlag {
    #[cfg_attr(feature = "web", wasm_bindgen(constructor))]
    pub fn new() -> AbortFlag {
        AbortFlag(Arc::new(AtomicBool::new(false)))
    }

    pub fn abort(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    #[cfg_attr(feature = "web", wasm_bindgen(js_name = isAborted))]
    pub fn is_aborted(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// A handle on the same flag. In JavaScript, pass `flag.cloneFlag()`
    /// to `generate` rather than `flag` itself: wasm-bindgen consumes a
    /// class passed by value, and the caller's `flag` must stay usable to
    /// call `abort()` while the call runs.
    #[cfg_attr(feature = "web", wasm_bindgen(js_name = cloneFlag))]
    pub fn clone_flag(&self) -> AbortFlag {
        self.clone()
    }
}

/// Which backend [`Lean::load`] uses. `Auto` is the capability ladder:
/// WebGPU when an adapter is granted and the model loads on it, otherwise
/// the CPU forward pass, which runs on a thread pool when this build has
/// threads and the pool has more than one thread, and on one thread
/// otherwise. `WebGpu` and `Cpu` take only that backend and fail if it
/// cannot start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Auto,
    WebGpu,
    Cpu,
}

/// `Read + Seek` over a GGUF file, for [`GgufSource::Reader`].
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek + ?Sized> ReadSeek for T {}

/// Where [`LoadOptions::gguf`] reads the GGUF file from. `Path` needs a
/// filesystem (native only). A reader is read one tensor at a time, so the
/// whole file never has to sit in memory twice.
pub enum GgufSource {
    Path(String),
    Bytes(Vec<u8>),
    Reader(Box<dyn ReadSeek>),
}

impl GgufSource {
    /// A reader positioned at the start of the file; callable again after a
    /// failed load (the `Auto` ladder reads the file a second time for the
    /// CPU when the GPU load fails).
    fn open(&mut self) -> anyhow::Result<Box<dyn ReadSeek + '_>> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            GgufSource::Path(path) => {
                let file = std::fs::File::open(&*path).map_err(|e| anyhow::anyhow!("opening {path}: {e}"))?;
                Ok(Box::new(std::io::BufReader::new(file)))
            }
            #[cfg(target_arch = "wasm32")]
            GgufSource::Path(_) => anyhow::bail!("GgufSource::Path needs a filesystem; use Bytes or Reader here"),
            GgufSource::Bytes(bytes) => Ok(Box::new(std::io::Cursor::new(bytes.as_slice()))),
            GgufSource::Reader(reader) => {
                reader.seek(SeekFrom::Start(0))?;
                Ok(Box::new(reader.as_mut()))
            }
        }
    }
}

/// Where [`LoadOptions::tokenizer`] reads `tokenizer.json` from: a path
/// (native only) or the file's contents.
pub enum TokenizerSource {
    Path(String),
    Json(String),
}

impl TokenizerSource {
    fn load(self) -> anyhow::Result<Tokenizer> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            TokenizerSource::Path(path) => Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("loading {path}: {e}")),
            #[cfg(target_arch = "wasm32")]
            TokenizerSource::Path(_) => anyhow::bail!("TokenizerSource::Path needs a filesystem; use Json here"),
            TokenizerSource::Json(json) => Tokenizer::from_bytes(json.as_bytes()).map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}")),
        }
    }
}

/// Everything [`Lean::load`] needs. `max_ctx` sizes the KV cache: a call's
/// prompt plus its `max_new_tokens` must fit in it.
pub struct LoadOptions {
    pub gguf: GgufSource,
    pub tokenizer: TokenizerSource,
    pub max_ctx: u32,
    pub backend: Backend,
}

// One `Parts` per engine, so the size difference between variants costs
// nothing.
#[allow(clippy::large_enum_variant)]
enum Parts {
    Gpu { engine: Engine, model: GpuModel, cache: KvCache, cos: wgpu::Buffer, sin: wgpu::Buffer },
    Cpu { model: CpuModel, cache: CpuKvCache },
}

/// A loaded model on one backend. See the module doc.
pub struct Lean {
    parts: Parts,
    tokenizer: Tokenizer,
    prefix: PrefixCache,
    max_ctx: usize,
}

/// A WebGPU device, if this client has one. In a browser, `navigator.gpu`
/// is checked first, so a client without WebGPU never asks wgpu for it.
async fn webgpu_engine() -> anyhow::Result<Engine> {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    if !crate::web::has_webgpu() {
        anyhow::bail!("WebGPU is not available (no navigator.gpu)");
    }
    Engine::new_async().await
}

fn gpu_parts(engine: Engine, gguf: &mut GgufSource, max_ctx: u32) -> anyhow::Result<Parts> {
    let model = GpuModel::load_from_reader(&engine, gguf.open()?, true)?;
    let cache = KvCache::new(&engine, &model.config, max_ctx);
    let (cos, sin) = rope_tables(&model.config, max_ctx as usize);
    let cos = engine.buf_f32(&cos, "rope_cos");
    let sin = engine.buf_f32(&sin, "rope_sin");
    Ok(Parts::Gpu { engine, model, cache, cos, sin })
}

fn cpu_parts(gguf: &mut GgufSource, max_ctx: u32) -> anyhow::Result<Parts> {
    let model = CpuModel::load_from_reader(gguf.open()?)?;
    let cache = CpuKvCache::new(&model.config, max_ctx as usize);
    Ok(Parts::Cpu { model, cache })
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
async fn yield_now() {
    crate::web::yield_to_event_loop().await;
}

#[cfg(not(all(target_arch = "wasm32", feature = "web")))]
async fn yield_now() {}

impl Lean {
    /// Loads the tokenizer and the model, on the backend `opts.backend`
    /// picks (see [`Backend`]).
    pub async fn load(opts: LoadOptions) -> Result<Lean, LeanError> {
        let LoadOptions { mut gguf, tokenizer, max_ctx, backend } = opts;
        let tokenizer = tokenizer.load()?;
        let parts = match backend {
            Backend::WebGpu => gpu_parts(webgpu_engine().await?, &mut gguf, max_ctx)?,
            Backend::Cpu => cpu_parts(&mut gguf, max_ctx)?,
            Backend::Auto => match webgpu_engine().await {
                Ok(engine) => match gpu_parts(engine, &mut gguf, max_ctx) {
                    Ok(parts) => parts,
                    Err(_) => cpu_parts(&mut gguf, max_ctx)?,
                },
                Err(_) => cpu_parts(&mut gguf, max_ctx)?,
            },
        };
        Ok(Lean { parts, tokenizer, prefix: PrefixCache::default(), max_ctx: max_ctx as usize })
    }

    /// One inference turn: tokenizes `text` as-is, prefills what the KV
    /// cache does not already hold, and decodes up to
    /// `params.max_new_tokens` tokens, stopping early at an end-of-sequence
    /// token or when `abort` is set. Each piece of reply text goes to
    /// `on_token` as soon as it is decoded; the whole reply is returned.
    pub async fn generate(&mut self, text: &str, params: &GenParams, on_token: impl FnMut(&str), abort: Option<&AbortFlag>) -> Result<String, LeanError> {
        let ids = encode(&self.tokenizer, text)?;
        check_fits(&ids, params.max_new_tokens, self.max_ctx)?;
        let should_stop = || abort.is_some_and(AbortFlag::is_aborted);
        let Lean { parts, tokenizer, prefix, .. } = self;
        match parts {
            Parts::Gpu { engine, model, cache, cos, sin } => {
                let mask = params.mask(model.config.vocab_size)?.map(|bits| engine.buf_u32(bits, "mask"));
                gpu_turn(engine, model, cache, cos, sin, prefix, tokenizer, ids, params, mask.as_ref(), on_token, should_stop).await
            }
            Parts::Cpu { model, cache } => {
                let mask = params.mask(model.config.vocab_size)?;
                cpu_turn(model, cache, prefix, tokenizer, ids, params, mask, on_token, should_stop, yield_now).await
            }
        }
    }

    /// How many tokens `text` is, as `generate` would tokenize it. No
    /// forward pass.
    pub fn token_count(&self, text: &str) -> Result<usize, LeanError> {
        Ok(encode(&self.tokenizer, text)?.len())
    }

    /// The backend this engine runs on: `"webgpu"`, `"threads"` (the CPU
    /// forward pass on a thread pool) or `"single"` (one CPU thread).
    pub fn backend(&self) -> &'static str {
        match self.parts {
            Parts::Gpu { .. } => "webgpu",
            Parts::Cpu { .. } if cpu_threads() > 1 => "threads",
            Parts::Cpu { .. } => "single",
        }
    }
}

fn cpu_threads() -> usize {
    #[cfg(feature = "threads")]
    return rayon::current_num_threads();
    #[cfg(not(feature = "threads"))]
    1
}
