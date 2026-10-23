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
use lean_core::model::{build_rope_tables, GpuModel, KvCache};
use crate::prefix_image::{self, Hash128, ImageEnv, PrefixImageError, HEADER_LEN};
use crate::turn::{check_fits, cpu_turn, encode, gpu_turn, GenParams, PrefixCache};
use lean_core::cpu::forward_prefill as cpu_forward_prefill;
use lean_core::model::{forward_prefill, forward_prefill_suffix};

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
    /// The tokenizer and a fingerprint of the file's bytes.
    fn load(self) -> anyhow::Result<(Tokenizer, [u8; 16])> {
        let json = match self {
            #[cfg(not(target_arch = "wasm32"))]
            TokenizerSource::Path(path) => std::fs::read(&path).map_err(|e| anyhow::anyhow!("loading {path}: {e}"))?,
            #[cfg(target_arch = "wasm32")]
            TokenizerSource::Path(_) => anyhow::bail!("TokenizerSource::Path needs a filesystem; use Json here"),
            TokenizerSource::Json(json) => json.into_bytes(),
        };
        let tokenizer = Tokenizer::from_bytes(&json).map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
        Ok((tokenizer, prefix_image::hash(&json)))
    }
}

/// A fingerprint of the GGUF file: its length and every header byte
/// (metadata, tokenizer tables, tensor names, types, shapes and offsets).
/// Two files that differ only in tensor data under identical metadata share
/// it; read only the header, never the weights. `header_len` is the one
/// the loader found.
fn model_fingerprint(gguf: &mut GgufSource, header_len: u64) -> anyhow::Result<[u8; 16]> {
    let mut r = gguf.open()?;
    let len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let mut h = Hash128::new();
    h.update(&len.to_le_bytes());
    let mut buf = vec![0u8; 1 << 20];
    let mut left = header_len;
    while left > 0 {
        let n = left.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..n])?;
        h.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(h.finish())
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
    image_env: ImageEnv,
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
    let (cos, sin) = build_rope_tables(&model.config, max_ctx as usize);
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
        let (tokenizer, tokenizer_fp) = tokenizer.load()?;
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
        let (config, backend, header_len) = match &parts {
            Parts::Gpu { model, .. } => (&model.config, 0, model.gguf_header_len),
            Parts::Cpu { model, .. } => (&model.config, 1, model.gguf_header_len),
        };
        let model_fp = model_fingerprint(&mut gguf, header_len)?;
        let image_env = ImageEnv { num_layers: config.num_layers as u32, num_kv_heads: config.num_kv_heads as u32, head_dim: config.head_dim as u32, backend, model: model_fp, tokenizer: tokenizer_fp };
        Ok(Lean { parts, tokenizer, prefix: PrefixCache::default(), max_ctx: max_ctx as usize, image_env })
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

    /// The storage key of `prefix`'s KV image on this engine: 32 hex
    /// characters, a hash of everything the image depends on (format
    /// version, dtype, engine build, model file, tokenizer, backend, and
    /// the prefix's token ids). Any change to one of them gives a new key.
    /// Tokenizes only, no forward pass.
    pub fn prefix_key(&self, prefix: &str) -> Result<String, LeanError> {
        let ids = encode(&self.tokenizer, prefix)?;
        Ok(prefix_image::key(&prefix_image::header(&self.image_env, &ids)))
    }

    /// Makes `text` resident in the KV cache without generating: prefills
    /// the part the cache does not already hold, so a following `generate`
    /// whose text starts with `text` prefills only what comes after it
    /// (a fixed system prompt or tools block warmed up before the first
    /// request). Returns `text`'s token count.
    pub async fn prefill(&mut self, text: &str) -> Result<usize, LeanError> {
        Ok(self.prefill_ids(text).await?.len())
    }

    async fn prefill_ids(&mut self, text: &str) -> Result<Vec<u32>, LeanError> {
        let ids = encode(&self.tokenizer, text)?;
        check_fits(&ids, 0, self.max_ctx)?;
        let Lean { parts, prefix: cached, .. } = self;
        match parts {
            Parts::Gpu { engine, model, cache, cos, sin } => {
                if !cached.holds(&ids, cache.kv_len as usize) {
                    let reuse = cached.reuse(&ids, cache.kv_len as usize);
                    if reuse == 0 {
                        forward_prefill(engine, model, cache, &ids, cos, sin, None).await;
                    } else {
                        cache.kv_len = reuse as u32;
                        forward_prefill_suffix(engine, model, cache, &ids[reuse..], cos, sin, None).await;
                    }
                    cached.commit(ids.clone(), &[], cache.kv_len as usize);
                }
            }
            Parts::Cpu { model, cache } => {
                if !cached.holds(&ids, cache.kv_len) {
                    cache.kv_len = cached.reuse(&ids, cache.kv_len);
                    let start = cache.kv_len;
                    cpu_forward_prefill(model, cache, &ids[start..]);
                    cached.commit(ids.clone(), &[], cache.kv_len);
                }
            }
        }
        Ok(ids)
    }

    /// Writes the KV image of `prefix` to `out`: a 128-byte header, then
    /// each layer's K and V for the prefix's positions, one tensor per
    /// `write_all` call (at most `num_kv_heads * positions * head_dim * 4`
    /// bytes, so the whole image is never in memory at once). Calls
    /// `prefill(prefix)` first, so after a `prefill` or a `generate` whose
    /// text started with `prefix` this is only a readback.
    /// Returns the number of bytes written. The engine keeps `prefix`
    /// resident, as after a `generate`.
    pub async fn save_prefix(&mut self, prefix: &str, out: &mut impl std::io::Write) -> Result<u64, LeanError> {
        let ids = self.prefill_ids(prefix).await?;
        let header = prefix_image::header(&self.image_env, &ids);
        let n = ids.len();
        let io = |e: std::io::Error| LeanError::Other(anyhow::anyhow!("writing the prefix image: {e}"));
        match &mut self.parts {
            Parts::Gpu { engine, cache, .. } => {
                out.write_all(&header).map_err(io)?;
                for layer in 0..cache.num_layers() {
                    for value in [false, true] {
                        out.write_all(&cache.read_prefix(engine, layer, value, n as u32).await).map_err(io)?;
                    }
                }
            }
            Parts::Cpu { cache, .. } => {
                out.write_all(&header).map_err(io)?;
                for layer in 0..cache.num_layers() {
                    for value in [false, true] {
                        out.write_all(&cache.read_prefix(layer, value, n)).map_err(io)?;
                    }
                }
            }
        }
        Ok((HEADER_LEN + 2 * self.image_env.num_layers as usize * prefix_image::tensor_bytes(&self.image_env, n)) as u64)
    }

    /// Reads a KV image written by `save_prefix` for the same `prefix` and
    /// makes it resident, so the next `generate` whose text starts with
    /// `prefix` prefills only what follows it, with the same output as
    /// without the image. Reads one tensor at a time from `input`.
    ///
    /// The header is checked before anything is touched: an image written
    /// by another engine build, model file, tokenizer, backend, dtype or
    /// format, or for another prefix, is refused with
    /// [`LeanError::PrefixImage`] and the engine is unchanged. A source that
    /// ends early gives `PrefixImageError::Truncated`; the engine then holds
    /// no prefix and stays usable. `prefix` longer than `max_ctx` gives
    /// [`LeanError::ContextOverflow`].
    pub fn restore_prefix(&mut self, prefix: &str, input: &mut impl std::io::Read) -> Result<(), LeanError> {
        let ids = encode(&self.tokenizer, prefix)?;
        check_fits(&ids, 0, self.max_ctx)?;
        let expected = prefix_image::header(&self.image_env, &ids);
        let mut found = [0u8; HEADER_LEN];
        input.read_exact(&mut found).map_err(|_| LeanError::PrefixImage(PrefixImageError::NotAnImage))?;
        prefix_image::check(&found, &expected).map_err(LeanError::PrefixImage)?;
        let n = ids.len();
        let mut buf = vec![0u8; prefix_image::tensor_bytes(&self.image_env, n)];
        let truncated = |_| LeanError::PrefixImage(PrefixImageError::Truncated);
        let Lean { parts, prefix: cached, .. } = self;
        cached.set(Vec::new());
        match parts {
            Parts::Gpu { engine, cache, .. } => {
                cache.kv_len = 0;
                for layer in 0..cache.num_layers() {
                    for value in [false, true] {
                        input.read_exact(&mut buf).map_err(truncated)?;
                        cache.write_prefix(engine, layer, value, n as u32, &buf);
                    }
                }
                cache.kv_len = n as u32;
            }
            Parts::Cpu { cache, .. } => {
                cache.kv_len = 0;
                for layer in 0..cache.num_layers() {
                    for value in [false, true] {
                        input.read_exact(&mut buf).map_err(truncated)?;
                        cache.write_prefix(layer, value, n, &buf);
                    }
                }
                cache.kv_len = n;
            }
        }
        cached.set(ids);
        Ok(())
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
