//! The wasm-bindgen browser surface, for a Web Worker.
//!
//! - `Lean`: the engine, the same surface as the Rust [`crate::Lean`]:
//!   `await Lean.load({gguf, tokenizer, maxCtx, backend})`, then
//!   `await lean.generate(text, params, onToken, abort)` once per turn and
//!   `lean.tokenCount(text)`; `await lean.prefill(text)`,
//!   `lean.prefixKey(prefix)`, `await lean.savePrefix(prefix, onChunk)`,
//!   `lean.restorePrefix(prefix, source)` for prefix KV images. Text in, text out: no chat template, no
//!   conversation state. One client per engine (see `lean.rs`).
//! - `AbortFlag`: the Rust [`crate::AbortFlag`] itself.
//! - lean-chat's `ChatTemplate`, `chatRender` and `chatTrimOldestExchange`.
//! - With the `diag` feature, `LeanEngine` and `LeanEngineCpu`
//!   (`web_diag.rs`): token-level and diagnostics calls.
//!
//! WebGPU readback is async only in the browser - every GPU-reading call
//! here awaits. Never call a blocking readback from this module.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use std::cell::RefCell;
use std::io::{Read, Seek, SeekFrom};

use crate::error::LeanError;
use crate::prefix_image::PrefixImageError;
use crate::lean::{AbortFlag, Backend, GgufSource, Lean, LoadOptions, TokenizerSource};
use crate::turn::GenParams;

/// `Read + Seek` adapter over a JS-owned `js_sys::Uint8Array`, so the whole
/// GGUF file never has to be copied into a wasm-side `Vec<u8>` just to
/// parse it. `GgufReader`/`GpuModel::load_from_reader` already read the
/// file through a handful of small `tensor_data()`-sized windows (one
/// tensor at a time, dropped right after that tensor's GPU upload - see
/// `model.rs::load_from_reader`'s doc comment); this reader makes each of
/// those windows a `Uint8Array::subarray().copy_to()` straight out of the
/// JS heap, instead of reading from an in-wasm-memory mirror of the entire
/// file. `data` is a *view* over the caller's `ArrayBuffer` (no copy at
/// construction) - the only bytes that ever cross into wasm linear memory
/// are the ones `read()` actually copies, each buffer transient and freed
/// right after its caller (`tensor_data`) is done with it. This is the fix
/// for the ~2.5x wasm-memory blowup after model load documented in
/// docs/runs/2026-09-28-lean-decode-breakdown.md's Session 3: before this,
/// `LeanEngine::load`'s `Vec<u8>` parameter forced wasm-bindgen to copy the
/// entire fetched file into wasm memory up front, on top of the JS-side
/// `Uint8Array` the fetch already produced - two full-file-sized copies
/// alive at once, and wasm memory never shrinks back down after that peak.
pub(crate) struct JsBytesReader {
    data: js_sys::Uint8Array,
    pos: u64,
    len: u64,
}

impl JsBytesReader {
    pub(crate) fn new(data: js_sys::Uint8Array) -> Self {
        let len = data.length() as u64;
        Self { data, pos: 0, len }
    }
}

impl Read for JsBytesReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.len.saturating_sub(self.pos);
        let n = (buf.len() as u64).min(remaining) as usize;
        if n == 0 {
            return Ok(0);
        }
        let start = self.pos as u32;
        let end = start + n as u32;
        self.data.subarray(start, end).copy_to(&mut buf[..n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for JsBytesReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.len as i64 + p,
            SeekFrom::Current(p) => self.pos as i64 + p,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "JsBytesReader: seek before byte 0"));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

pub(crate) fn wasm_log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

fn get_f64(obj: &JsValue, key: &str, default: f64) -> f64 {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).ok().and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Converts a JS `params` object (`maxNewTokens`, `temperature`, `topK`,
/// `topP`, `repetitionPenalty`, `seed`, optional `mask`) into `GenParams`.
/// Missing fields take `GenParams::default()`'s values.
fn gen_params_from_js(params: &JsValue) -> GenParams {
    let defaults = GenParams::default();
    let mask_bits = js_sys::Reflect::get(params, &JsValue::from_str("mask"))
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null())
        .and_then(|v| v.dyn_into::<js_sys::Array>().ok())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64()).map(|f| f as u32).collect::<Vec<u32>>())
        .filter(|bits| !bits.is_empty());
    GenParams {
        max_new_tokens: get_f64(params, "maxNewTokens", defaults.max_new_tokens as f64) as u32,
        temperature: get_f64(params, "temperature", defaults.temperature as f64) as f32,
        top_k: get_f64(params, "topK", defaults.top_k as f64) as u32,
        top_p: get_f64(params, "topP", defaults.top_p as f64) as f32,
        repetition_penalty: get_f64(params, "repetitionPenalty", defaults.repetition_penalty as f64) as f32,
        seed: get_f64(params, "seed", defaults.seed as f64) as u64,
        mask_bits,
    }
}

/// `std::io::Write` over a JS `onChunk(Uint8Array)` callback; a throw
/// becomes a write error.
struct JsChunkWriter(js_sys::Function);

impl std::io::Write for JsChunkWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.call1(&JsValue::NULL, &js_sys::Uint8Array::from(buf)).map_err(|e| std::io::Error::other(format!("onChunk threw: {}", js_message(&e))))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `std::io::Read` over a JS `read(n)` callback that returns the next at
/// most `n` bytes of the image as a `Uint8Array` (an empty one at the end).
struct JsFnReader(js_sys::Function);

impl Read for JsFnReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let got = self.0.call1(&JsValue::NULL, &JsValue::from_f64(buf.len() as f64)).map_err(|e| std::io::Error::other(format!("read threw: {}", js_message(&e))))?;
        let bytes: js_sys::Uint8Array = got.dyn_into().map_err(|_| std::io::Error::other("read(n) must return a Uint8Array"))?;
        let n = bytes.length() as usize;
        if n > buf.len() {
            return Err(std::io::Error::other("read(n) returned more than n bytes"));
        }
        bytes.copy_to(&mut buf[..n]);
        Ok(n)
    }
}

fn js_message(e: &JsValue) -> String {
    e.dyn_ref::<js_sys::Error>().map(|e| String::from(e.message())).or_else(|| e.as_string()).unwrap_or_else(|| format!("{e:?}"))
}

/// Converts a `LeanError` into the JS exception `Lean`'s calls throw:
/// `ContextOverflow` becomes an `Error` with `name === "ContextOverflow"`,
/// `code === "CONTEXT_OVERFLOW"` and numeric `needed`/`available`
/// properties (so a caller branches on `e.name` or `e.code` without
/// parsing a string); anything else becomes a plain `Error`.
fn lean_error_to_js(e: LeanError) -> JsValue {
    match e {
        LeanError::ContextOverflow { needed, available } => {
            let err = js_sys::Error::new(&e.to_string());
            err.set_name("ContextOverflow");
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("code"), &JsValue::from_str("CONTEXT_OVERFLOW"));
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("needed"), &JsValue::from_f64(needed as f64));
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("available"), &JsValue::from_f64(available as f64));
            err.into()
        }
        LeanError::PrefixImage(ref kind) => {
            let err = js_sys::Error::new(&e.to_string());
            err.set_name("PrefixImage");
            let (code, field) = match kind {
                PrefixImageError::NotAnImage => ("NOT_AN_IMAGE", None),
                PrefixImageError::Stale(field) => ("STALE", Some(*field)),
                PrefixImageError::Truncated => ("TRUNCATED", None),
            };
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("code"), &JsValue::from_str(code));
            if let Some(field) = field {
                let _ = js_sys::Reflect::set(&err, &JsValue::from_str("field"), &JsValue::from_str(field));
            }
            err.into()
        }
        LeanError::Other(e) => JsValue::from(JsError::new(&format!("{e}"))),
    }
}

/// The page's `onToken` callback for `generate`, called as `onToken(text)`
/// once per piece of reply text as soon as it is produced.
///
/// The callback runs while the engine is borrowed for the generation call,
/// so it must not call methods on the same engine (wasm-bindgen throws
/// "recursive use of an object"). A throw from the callback stops
/// generation and is rethrown from the call, instead of being dropped.
struct TextSink {
    f: Option<js_sys::Function>,
    err: Option<String>,
}

impl TextSink {
    fn new(on_token: JsValue) -> RefCell<Self> {
        RefCell::new(TextSink { f: on_token.dyn_into::<js_sys::Function>().ok(), err: None })
    }

    fn text(&mut self, text: &str) {
        if self.err.is_some() || text.is_empty() {
            return;
        }
        if let Some(f) = &self.f {
            if let Err(e) = f.call1(&JsValue::NULL, &JsValue::from(text)) {
                let msg = e.dyn_ref::<js_sys::Error>().map(|e| String::from(e.message())).or_else(|| e.as_string()).unwrap_or_else(|| format!("{e:?}"));
                self.err = Some(format!("onToken callback threw: {msg}"));
            }
        }
    }

    fn stopped(&self) -> bool {
        self.err.is_some()
    }

    fn take_err(&mut self) -> Option<JsValue> {
        self.err.take().map(|e| JsValue::from(JsError::new(&e)))
    }
}

thread_local! {
    static YIELD_CHANNEL: RefCell<Option<(JsValue, JsValue, js_sys::Function)>> = const { RefCell::new(None) };
}

/// Resolves on a later task (a `MessageChannel` message, which unlike
/// `setTimeout(0)` is never clamped to 4 ms), so the worker's own message
/// handlers - a stop button's `AbortFlag.abort()` - run between CPU tokens.
/// Resolves at once where `MessageChannel` does not exist.
pub(crate) async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let posted = YIELD_CHANNEL.with(|cell| {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                let global = js_sys::global();
                let ctor = js_sys::Reflect::get(&global, &JsValue::from_str("MessageChannel")).ok().and_then(|c| c.dyn_into::<js_sys::Function>().ok());
                if let Some(ch) = ctor.and_then(|c| js_sys::Reflect::construct(&c, &js_sys::Array::new()).ok()) {
                    let port1 = js_sys::Reflect::get(&ch, &JsValue::from_str("port1")).unwrap_or(JsValue::UNDEFINED);
                    let port2 = js_sys::Reflect::get(&ch, &JsValue::from_str("port2")).unwrap_or(JsValue::UNDEFINED);
                    if let Ok(post) = js_sys::Reflect::get(&port2, &JsValue::from_str("postMessage")).and_then(|p| p.dyn_into::<js_sys::Function>()) {
                        *cell = Some((port1, port2, post));
                    }
                }
            }
            match cell.as_ref() {
                Some((port1, port2, post)) => js_sys::Reflect::set(port1, &JsValue::from_str("onmessage"), &resolve).is_ok() && post.call1(port2, &JsValue::from(0)).is_ok(),
                None => false,
            }
        });
        if !posted {
            let _ = resolve.call0(&JsValue::NULL);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}


/// Whether `navigator.gpu` exists in this global scope: the capability
/// check before asking wgpu for a WebGPU adapter.
pub(crate) fn has_webgpu() -> bool {
    let navigator = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator")).unwrap_or(JsValue::UNDEFINED);
    if navigator.is_undefined() || navigator.is_null() {
        return false;
    }
    let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu")).unwrap_or(JsValue::UNDEFINED);
    !gpu.is_undefined() && !gpu.is_null()
}

fn get_field(obj: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

/// The engine: the JS face of [`crate::Lean`], same calls and same
/// behaviour.
#[wasm_bindgen(js_name = Lean)]
pub struct LeanJs {
    inner: Lean,
}

#[wasm_bindgen(js_class = Lean)]
impl LeanJs {
    /// `opts`: `{ gguf, tokenizer, maxCtx, backend }`. `gguf` is the GGUF
    /// file as a `Uint8Array` (read one tensor at a time, never copied
    /// whole into wasm memory; drop your reference after the call so the
    /// JS heap can reclaim it), `tokenizer` the text of `tokenizer.json`,
    /// `maxCtx` the KV cache size in tokens, `backend` `"auto"` (default:
    /// WebGPU, else the CPU forward pass), `"webgpu"` or `"cpu"`. The CPU
    /// forward pass runs on a thread pool in the threads build after
    /// `initThreadPool(n)`, and on one thread otherwise.
    pub async fn load(opts: JsValue) -> Result<LeanJs, JsValue> {
        console_error_panic_hook::set_once();
        let gguf: js_sys::Uint8Array = get_field(&opts, "gguf").dyn_into().map_err(|_| JsValue::from(JsError::new("opts.gguf must be a Uint8Array")))?;
        let tokenizer = get_field(&opts, "tokenizer").as_string().ok_or_else(|| JsValue::from(JsError::new("opts.tokenizer must be the text of tokenizer.json")))?;
        let max_ctx = get_field(&opts, "maxCtx").as_f64().ok_or_else(|| JsValue::from(JsError::new("opts.maxCtx must be a number")))? as u32;
        let backend = match get_field(&opts, "backend").as_string().as_deref() {
            None | Some("auto") => Backend::Auto,
            Some("webgpu") => Backend::WebGpu,
            Some("cpu") => Backend::Cpu,
            Some(other) => return Err(JsValue::from(JsError::new(&format!("unknown backend {other:?} (expected \"auto\", \"webgpu\" or \"cpu\")")))),
        };
        let opts = LoadOptions { gguf: GgufSource::Reader(Box::new(JsBytesReader::new(gguf))), tokenizer: TokenizerSource::Json(tokenizer), max_ctx, backend };
        let inner = Lean::load(opts).await.map_err(lean_error_to_js)?;
        wasm_log(&format!("[lean] loaded on {}", inner.backend()));
        Ok(LeanJs { inner })
    }

    /// One inference turn over `text`, exactly as given. `params`:
    /// `maxNewTokens`, `temperature` (0 = greedy), `topK`, `topP`,
    /// `repetitionPenalty`, `seed`, optional `mask`. `onToken(text)` gets
    /// each piece of the reply as it is decoded; it must not call this
    /// engine, and a throw from it stops generation and is rethrown.
    /// `abort`: `flag.cloneFlag()` of an `AbortFlag`, or nothing. Resolves
    /// to the whole reply. Rejects with a `ContextOverflow` error (see
    /// `lean_error_to_js`) when `text` plus `maxNewTokens` does not fit
    /// `maxCtx`; the engine stays usable.
    pub async fn generate(&mut self, text: String, params: JsValue, on_token: JsValue, abort: Option<AbortFlag>) -> Result<String, JsValue> {
        let params = gen_params_from_js(&params);
        let sink = TextSink::new(on_token);
        // A throw from `onToken` stops generation through the same flag.
        let flag = abort.unwrap_or_default();
        let on_text = |t: &str| {
            let mut sink = sink.borrow_mut();
            sink.text(t);
            if sink.stopped() {
                flag.abort();
            }
        };
        let result = self.inner.generate(&text, &params, on_text, Some(&flag)).await;
        if let Some(e) = sink.borrow_mut().take_err() {
            return Err(e);
        }
        result.map_err(lean_error_to_js)
    }

    /// How many tokens `text` is. No forward pass.
    #[wasm_bindgen(js_name = tokenCount)]
    pub fn token_count(&self, text: &str) -> Result<u32, JsValue> {
        self.inner.token_count(text).map(|n| n as u32).map_err(lean_error_to_js)
    }

    /// Makes `text` resident without generating (see the Rust
    /// `Lean::prefill`): a later `generate` whose text starts with `text`
    /// prefills only the rest. Resolves to `{ promptTokens }`, `text`'s
    /// token count.
    pub async fn prefill(&mut self, text: String) -> Result<JsValue, JsValue> {
        let n = self.inner.prefill(&text).await.map_err(lean_error_to_js)?;
        let out = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&out, &JsValue::from_str("promptTokens"), &JsValue::from_f64(n as f64));
        Ok(out.into())
    }

    /// The storage key of `prefix`'s KV image on this engine (32 hex
    /// characters). It changes whenever the image would: another engine
    /// build, model file, tokenizer, backend or prefix text.
    #[wasm_bindgen(js_name = prefixKey)]
    pub fn prefix_key(&self, prefix: &str) -> Result<String, JsValue> {
        self.inner.prefix_key(prefix).map_err(lean_error_to_js)
    }

    /// Writes the KV image of `prefix` through `onChunk(bytes)`, called
    /// once for the 128-byte header and once per K or V tensor of each
    /// layer, in order; write each chunk at the running offset (an OPFS
    /// `FileSystemSyncAccessHandle.write(bytes, {at})` in a worker). Each
    /// chunk is a fresh `Uint8Array` the callback may keep. Prefills the
    /// part of `prefix` not already resident. Resolves to the total byte
    /// count. A throw from `onChunk` rejects the call; the engine stays
    /// usable.
    #[wasm_bindgen(js_name = savePrefix)]
    pub async fn save_prefix(&mut self, prefix: String, on_chunk: js_sys::Function) -> Result<f64, JsValue> {
        let mut out = JsChunkWriter(on_chunk);
        self.inner.save_prefix(&prefix, &mut out).await.map(|n| n as f64).map_err(lean_error_to_js)
    }

    /// Restores an image written by `savePrefix` for the same `prefix`.
    /// `source` is the whole image as a `Uint8Array` (read one tensor at a
    /// time, never copied whole into wasm memory), or a function
    /// `read(n)` returning the next at most `n` bytes as a `Uint8Array`.
    /// Throws an `Error` with `name === "PrefixImage"` and `code`
    /// `"STALE"` (with `field`: what differs), `"NOT_AN_IMAGE"` or
    /// `"TRUNCATED"` when the image cannot be used: prefill again with
    /// `savePrefix` and overwrite it.
    #[wasm_bindgen(js_name = restorePrefix)]
    pub fn restore_prefix(&mut self, prefix: &str, source: JsValue) -> Result<(), JsValue> {
        let result = if let Some(bytes) = source.dyn_ref::<js_sys::Uint8Array>() {
            self.inner.restore_prefix(prefix, &mut JsBytesReader::new(bytes.clone()))
        } else if let Ok(f) = source.dyn_into::<js_sys::Function>() {
            self.inner.restore_prefix(prefix, &mut JsFnReader(f))
        } else {
            return Err(JsValue::from(JsError::new("restorePrefix: source must be a Uint8Array or a read(n) function")));
        };
        result.map_err(lean_error_to_js)
    }

    /// `"webgpu"`, `"threads"` or `"single"`.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.inner.backend().to_string()
    }
}

