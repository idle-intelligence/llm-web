//! Minimal wasm-bindgen browser surface. One entry point, `LeanEngine`, for
//! a Web Worker: `LeanEngine::create()` (async - requests the WebGPU
//! adapter/device), `load()` (parse GGUF bytes + tokenizer + chat template,
//! upload weights to GPU), `generate()` (one inference turn: render the
//! given messages -> tokenize -> prefill the part not already cached ->
//! decode, one `on_token(text: string)` JS callback per piece of text
//! produced). No agent/tool/grammar layer here (that's llm-wasm's `web.rs`)
//! and no conversation state either - this engine serves one turn per
//! call; which messages to send next, and when to trim them, is the
//! caller's job (see `chat.rs`'s doc comment). The engine still keeps the
//! previous call's KV cache and reuses whatever prefix the new call's
//! messages agree on (`chat::PrefixCache`) - a single-client
//! optimization, invisible to the caller, not conversation state: two
//! callers sharing one `LeanEngine` evict each other's cached prefix.
//!
//! WebGPU readback is async only in the browser - every GPU-reading call
//! here is `async`/`.await`s `Engine::read_buffer`'s `into_data_async`-style
//! path (see `engine.rs`'s doc comment). Never call a blocking readback from
//! this module.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use std::cell::{Cell, RefCell};
use std::io::{Read, Seek, SeekFrom};
use std::rc::Rc;

use tokenizers::Tokenizer;

use crate::chat::{cpu_generate, gpu_generate, render_prompt as chat_render_prompt, token_count as chat_token_count, GenParams, Message, PrefixCache};
use crate::chat_template::{chat_template_from_config_json, render_user_prompt};
use crate::cpu::{forward_decode_step_argmax as cpu_decode_step_argmax, forward_prefill as cpu_forward_prefill, CpuKvCache, CpuModel};
use crate::engine::{now_ms, Engine};
use crate::error::LeanError;
use crate::model::{build_mask_bitset, build_rope_tables, decode_greedy_pipelined, forward_decode_step_argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache, KvSnapshot};

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
struct JsBytesReader {
    data: js_sys::Uint8Array,
    pos: u64,
    len: u64,
}

impl JsBytesReader {
    fn new(data: js_sys::Uint8Array) -> Self {
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

/// `mask_bits` is this crate's packed bitset format (`build_mask_bitset`/
/// `buildMaskBitset`) - an empty vec means "no mask" (unmasked path, byte-
/// identical dispatch sequence to before this feature existed). wasm-bindgen
/// doesn't need `Option<Vec<u32>>` plumbing this way, and JS call sites read
/// naturally as `engine.decodeStepArgmax(id, [])`.
/// A non-empty mask shorter than `vocab_size / 32` words is rejected, as
/// `LeanEngineCpu::generate` does.
fn mask_buf(engine: &Engine, vocab_size: usize, mask_bits: &[u32]) -> Result<Option<wgpu::Buffer>, JsError> {
    if mask_bits.is_empty() {
        Ok(None)
    } else if mask_bits.len() * 32 < vocab_size {
        Err(JsError::new("mask_bits is shorter than vocab_size / 32"))
    } else {
        Ok(Some(engine.buf_u32(mask_bits, "mask")))
    }
}

fn wasm_log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Initializes the panic hook for readable browser-console error messages.
/// Optional but recommended: call once before `LeanEngine::create()`.
#[wasm_bindgen(js_name = leanInit)]
pub fn lean_init() {
    console_error_panic_hook::set_once();
}

/// A JS-shareable abort switch for `generate`: JS holds
/// the same `AbortFlag` it passed into the call (e.g. from a "stop" button's
/// click handler) and calls `.abort()` on it whenever it likes; the decode
/// loop checks `is_aborted()` once per generated token, between GPU
/// dispatches - see `generate::decode_loop`'s `should_stop` parameter. This
/// works because every `.await` inside the decode loop actually yields to
/// the browser's event loop (real GPU-readback awaits, not a synchronous
/// spin), so a click handler queued while a `generate` promise is
/// pending still gets to run and flip the flag before the next token.
#[wasm_bindgen]
#[derive(Clone)]
pub struct AbortFlag(Rc<Cell<bool>>);

#[wasm_bindgen]
impl AbortFlag {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AbortFlag {
        AbortFlag(Rc::new(Cell::new(false)))
    }

    pub fn abort(&self) {
        self.0.set(true);
    }

    #[wasm_bindgen(js_name = isAborted)]
    pub fn is_aborted(&self) -> bool {
        self.0.get()
    }

    /// A cheap `Rc` clone sharing the same underlying flag - **the value to
    /// pass into `generate`, not the original**.
    /// wasm-bindgen destroys a by-value class argument's JS-side handle the
    /// instant it crosses into wasm (`__destroy_into_raw()` in the generated
    /// glue), so passing the caller's own `AbortFlag` directly would leave
    /// that JS object unusable (a later `flag.abort()` call would hit a
    /// freed pointer) even though the call is still in flight. Passing
    /// `flag.cloneFlag()` instead keeps the caller's original object alive
    /// and callable for the whole (possibly multi-second) generation, since
    /// both point at the same `Rc<Cell<bool>>`.
    #[wasm_bindgen(js_name = cloneFlag)]
    pub fn clone_flag(&self) -> AbortFlag {
        self.clone()
    }
}

impl Default for AbortFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a JS `messages` array (`{role, content}` objects) into this
/// crate's `chat::Message` list, the input `generate`/`renderPrompt`/
/// `tokenCount` all take.
fn messages_from_js(messages: &JsValue) -> Result<Vec<Message>, JsValue> {
    let arr: js_sys::Array = messages.clone().dyn_into().map_err(|_| JsValue::from(JsError::new("messages must be an array")))?;
    let mut out = Vec::with_capacity(arr.length() as usize);
    for v in arr.iter() {
        let role = js_sys::Reflect::get(&v, &JsValue::from_str("role")).ok().and_then(|r| r.as_string()).ok_or_else(|| JsValue::from(JsError::new("message missing 'role'")))?;
        let content = js_sys::Reflect::get(&v, &JsValue::from_str("content")).ok().and_then(|c| c.as_string()).ok_or_else(|| JsValue::from(JsError::new("message missing 'content'")))?;
        out.push(Message::new(role, content));
    }
    Ok(out)
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

/// Converts a `LeanError` into the JS exception `generate`/`renderPrompt`/
/// `tokenCount` throw: `ContextOverflow` becomes an `Error` with `name ===
/// "ContextOverflow"` and numeric `needed`/`available` properties (so a
/// caller can branch on `e.name` without parsing a string); anything else
/// becomes a plain `JsError`.
fn lean_error_to_js(e: LeanError) -> JsValue {
    match e {
        LeanError::ContextOverflow { needed, available } => {
            let err = js_sys::Error::new(&format!("context overflow: needed {needed} tokens, {available} available"));
            err.set_name("ContextOverflow");
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("needed"), &JsValue::from_f64(needed as f64));
            let _ = js_sys::Reflect::set(&err, &JsValue::from_str("available"), &JsValue::from_f64(available as f64));
            err.into()
        }
        LeanError::Other(e) => JsValue::from(JsError::new(&format!("{e}"))),
    }
}

/// The page's `onToken` callback for `generate`, called as `onToken(text)`
/// once per piece of reply text as soon as it is produced (text-in/
/// text-out: no token ids cross the wasm boundary here - `chat::gpu_generate`/
/// `chat::cpu_generate` already turn ids into text internally).
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
async fn yield_to_event_loop() {
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

#[wasm_bindgen]
pub struct LeanEngine {
    engine: Engine,
    model: Option<GpuModel>,
    cache: Option<KvCache>,
    cos_buf: Option<wgpu::Buffer>,
    sin_buf: Option<wgpu::Buffer>,
    tokenizer: Option<Tokenizer>,
    chat_template: Option<String>,
    /// The ids `generate`'s last call left resident in the KV cache
    /// (`chat::PrefixCache`). Every other call that writes the cache
    /// invalidates the cached ids, so the next turn prefills from scratch.
    prefix: PrefixCache,
    /// `(GGUF parse + weight upload calls ms, tokenizer + chat template ms)`
    /// from the last `load()`, for the diagnostics page.
    load_ms: (f64, f64),
}

#[wasm_bindgen]
impl LeanEngine {
    /// Requests a WebGPU adapter/device (the adapter's own limits, not
    /// `wgpu::Limits::default()` - see `engine.rs::Engine::new_async`'s doc
    /// comment) and builds every compute pipeline. Must be awaited before
    /// any other call.
    #[wasm_bindgen(js_name = create)]
    pub async fn create() -> Result<LeanEngine, JsError> {
        console_error_panic_hook::set_once();
        let engine = Engine::new_async().await.map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(LeanEngine::from_engine(engine))
    }

    /// Same as `create()`, but also requests WebGPU's `timestamp-query`
    /// feature when the adapter has it (feature detection only), so the
    /// diagnostics calls below can report GPU time. Nothing else differs.
    #[wasm_bindgen(js_name = createDiag)]
    pub async fn create_diag() -> Result<LeanEngine, JsError> {
        console_error_panic_hook::set_once();
        let engine = Engine::new_async_with(true).await.map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(LeanEngine::from_engine(engine))
    }

    /// JSON: device request ms, pipeline creation calls ms (all of them, in
    /// total), whether pass timestamps are available, and the last `load()`'s
    /// split.
    #[wasm_bindgen(js_name = diagInfo)]
    pub fn diag_info(&self) -> String {
        serde_json::json!({
            "deviceMs": self.engine.diag_init_ms.0,
            "pipelinesMs": self.engine.diag_init_ms.1,
            "timestampQuery": self.engine.has_pass_timestamps(),
            "loadWeightsMs": self.load_ms.0,
            "loadTokenizerMs": self.load_ms.1,
        })
        .to_string()
    }

    /// Diagnostics switches, both off by default - see `Engine::set_diag`.
    #[wasm_bindgen(js_name = diagSet)]
    pub fn diag_set(&self, timestamps: bool, split: bool) {
        self.engine.set_diag(timestamps, split);
    }

    /// JSON for the last `prefillTokens`/`decodeStepArgmax` call:
    /// `encodeMs` (recording + submit, CPU), `waitMs` (submit to result in
    /// hand), and `gpu` (`null` unless timestamps were on: `spanMs`,
    /// `passSumMs`, `segments` as `[label, ms, passes]`, `unwritten`; a
    /// span or segment with no written timestamp is `null`).
    #[wasm_bindgen(js_name = diagLast)]
    pub fn diag_last(&self) -> String {
        let gpu = self.engine.diag_last_gpu.borrow_mut().take().map(|g| {
            serde_json::json!({
                "spanMs": g.span_ms,
                "passSumMs": g.pass_sum_ms,
                "segments": g.segments.iter().map(|(l, ms, n)| serde_json::json!([l, ms, n])).collect::<Vec<_>>(),
                "unwritten": g.unwritten,
            })
        });
        serde_json::json!({
            "encodeMs": self.engine.diag_encode_ms.get(),
            "waitMs": self.engine.diag_wait_ms.get(),
            "gpu": gpu,
        })
        .to_string()
    }

    /// Milliseconds for one 4-byte copy + `mapAsync` round trip with no
    /// other work: the per-readback floor, and (right after `create`/`load`)
    /// the time for the GPU process to drain what was queued before it.
    #[wasm_bindgen(js_name = diagRoundTrip)]
    pub async fn diag_round_trip(&self) -> f64 {
        let t = now_ms();
        let buf = self.engine.buf_empty(1, "diag_round_trip");
        let _ = self.engine.read_u32(&buf).await;
        now_ms() - t
    }

    /// Parses `gguf_bytes` (the whole GGUF file, fetched by JS, passed as a
    /// `Uint8Array` view rather than a `Vec<u8>` so wasm-bindgen never has to
    /// copy it into wasm linear memory up front - see `JsBytesReader`'s doc
    /// comment) and uploads every weight to the GPU (two-phase loading: only
    /// one tensor's raw bytes are ever resident in wasm memory at a time,
    /// dropped right after that tensor's GPU upload inside
    /// `GpuModel::load_from_reader`, well before `max_ctx`'s KV cache is
    /// allocated). `tokenizer_json`/`tokenizer_config_json` are the two
    /// files' contents as strings. `max_ctx` bounds the KV cache (prompt +
    /// max_new_tokens must fit). The caller should drop its own reference to
    /// `gguf_bytes`'s backing `ArrayBuffer` right after this call returns so
    /// the JS heap can reclaim it too.
    #[wasm_bindgen(js_name = load)]
    pub fn load(&mut self, gguf_bytes: js_sys::Uint8Array, tokenizer_json: String, tokenizer_config_json: String, max_ctx: u32) -> Result<(), JsError> {
        let t_start = now_ms();
        let model = GpuModel::load_from_reader(&self.engine, JsBytesReader::new(gguf_bytes), true)
            .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        let cache = KvCache::new(&self.engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
        let cos_buf = self.engine.buf_f32(&cos, "rope_cos");
        let sin_buf = self.engine.buf_f32(&sin, "rope_sin");
        let t_weights = now_ms();

        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
        let chat_template = chat_template_from_config_json(&tokenizer_config_json).map_err(|e| JsError::new(&format!("{e}")))?;
        self.load_ms = (t_weights - t_start, now_ms() - t_weights);

        wasm_log(&format!(
            "[lean] model loaded: layers={} hidden={} vocab={}",
            model.config.num_layers, model.config.hidden_size, model.config.vocab_size
        ));
        self.model = Some(model);
        self.cache = Some(cache);
        self.cos_buf = Some(cos_buf);
        self.sin_buf = Some(sin_buf);
        self.tokenizer = Some(tokenizer);
        self.chat_template = Some(chat_template);
        self.prefix.invalidate();
        Ok(())
    }

    /// One inference turn: renders `messages` (a JS array of `{role,
    /// content}`) through the model's own chat template, tokenizes it,
    /// reuses whatever prefix of the previous call's KV cache still
    /// agrees with the new ids (`chat::PrefixCache` - invisible to the
    /// caller, see this module's doc comment), prefills the rest, then
    /// decodes up to `params.maxNewTokens` tokens. `params` is a plain
    /// object: `maxNewTokens`, `temperature` (0 = greedy), `topK`, `topP`,
    /// `repetitionPenalty`, `seed`, and an optional `mask` (this crate's
    /// mask-bitset format, same convention as `decodeStepArgmax`).
    /// `onToken(text)` is called with each piece of text as soon as it is
    /// produced (text, not token ids - this is a text-in/text-out
    /// surface). `abort`, if given, is checked once per generated token.
    /// Returns the full reply text. Conversation history (which messages
    /// to send next turn, when to drop old ones) is the caller's job -
    /// this call does not remember `messages` after it returns.
    ///
    /// On a `ContextOverflow` (messages + `maxNewTokens` would not fit
    /// this engine's `max_ctx`), the rejected `Error` has `name ===
    /// "ContextOverflow"` and numeric `needed`/`available` properties; the
    /// engine's own state is unchanged and the next call works normally -
    /// a caller drops older messages and retries.
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&mut self, messages: JsValue, params: JsValue, on_token: JsValue, abort: Option<AbortFlag>) -> Result<String, JsValue> {
        let model = self.model.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;

        let messages = messages_from_js(&messages)?;
        let params = gen_params_from_js(&params);
        let mask = mask_buf(&self.engine, model.config.vocab_size, params.mask_bits.as_deref().unwrap_or(&[])).map_err(JsValue::from)?;

        let sink = TextSink::new(on_token);
        let should_stop = || abort.as_ref().is_some_and(AbortFlag::is_aborted) || sink.borrow().stopped();
        let result = gpu_generate(&self.engine, model, cache, cos_buf, sin_buf, &mut self.prefix, tokenizer, chat_template, &messages, &params, mask.as_ref(), |text| sink.borrow_mut().text(text), should_stop).await;
        if let Some(e) = sink.borrow_mut().take_err() {
            return Err(e);
        }
        result.map_err(lean_error_to_js)
    }

    /// Renders `messages` through the model's own chat template, the same
    /// way `generate()` would - no tokenization, no forward-pass work.
    #[wasm_bindgen(js_name = renderPrompt)]
    pub fn render_prompt(&self, messages: JsValue) -> Result<String, JsValue> {
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let messages = messages_from_js(&messages)?;
        chat_render_prompt(chat_template, &messages).map_err(lean_error_to_js)
    }

    /// Renders + tokenizes `messages` the same way `generate()` does and
    /// returns the resulting token count, with no forward-pass work - lets
    /// a caller decide whether to trim before calling `generate()`.
    #[wasm_bindgen(js_name = tokenCount)]
    pub fn token_count(&self, messages: JsValue) -> Result<u32, JsValue> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let messages = messages_from_js(&messages)?;
        chat_token_count(tokenizer, chat_template, &messages).map(|n| n as u32).map_err(lean_error_to_js)
    }

    /// JSON string with basic model/device info, for a status line.
    #[wasm_bindgen(js_name = info)]
    pub fn info(&self) -> String {
        match &self.model {
            Some(model) => serde_json::json!({
                "loaded": true,
                "numLayers": model.config.num_layers,
                "hiddenSize": model.config.hidden_size,
                "vocabSize": model.config.vocab_size,
                "maxStorageBufferBindingSize": self.engine.max_storage_buffer_binding_size(),
            })
            .to_string(),
            None => serde_json::json!({ "loaded": false }).to_string(),
        }
    }

    /// GPU-resident byte counts, broken down by category (see
    /// `GpuModel::weight_gpu_bytes`/`KvCache::gpu_bytes`/
    /// `Pool::resident_bytes`'s doc comments): `weightBytes` (fixed at
    /// load, independent of context), `kvCacheBytes` (fixed at `max_ctx`,
    /// independent of `kv_len`), `poolBytes` (per-forward scratch -
    /// activations, RoPE tables, uniforms; settles to a fixed shape once a
    /// given `(rows, kv_len-bucket)` combination has been seen once). Does
    /// NOT include the wasm linear memory (`memory.buffer.byteLength`,
    /// read directly from JS) or the JS-side GGUF `Uint8Array`/`Vec<u8>`
    /// copies, which this method has no visibility into - a caller
    /// combines this with `performance.memory`/`wasm.memory` for the full
    /// picture. Added for this session's browser memory investigation
    /// (docs/runs/2026-09-28-lean-decode-breakdown.md).
    #[wasm_bindgen(js_name = gpuMemoryInfo)]
    pub fn gpu_memory_info(&self) -> Result<String, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(serde_json::json!({
            "weightBytes": model.weight_gpu_bytes(),
            "kvCacheBytes": cache.gpu_bytes(),
            "poolBytes": model.pool.resident_bytes(),
        })
        .to_string())
    }

    /// Renders + tokenizes `prompt` through the model's chat template, same
    /// as `generate()`, but returns the raw token ids instead of running
    /// generation - the low-level entry point a consumer's own prefix/suffix
    /// split (e.g. a resident tool-schema prefix) is built on top of, rather
    /// than `generate()`'s all-in-one path.
    #[wasm_bindgen(js_name = tokenize)]
    pub fn tokenize(&self, prompt: String) -> Result<Vec<u32>, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let rendered = render_user_prompt(chat_template, &prompt).map_err(|e| JsError::new(&format!("failed to render prompt: {e}")))?;
        let encoding = tokenizer.encode(rendered, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Tokenizes raw `text` with no chat-template rendering (unlike
    /// `tokenize()`) - for building a target/mask continuation from
    /// arbitrary text (e.g. a fixed string a constrained-decoding test wants
    /// to force), not a user chat turn.
    #[wasm_bindgen(js_name = encodeRaw)]
    pub fn encode_raw(&self, text: String) -> Result<Vec<u32>, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let encoding = tokenizer.encode(text, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes `token_ids` back into text - the inverse of `tokenize`/
    /// `encodeRaw`, exposed so a harness can print what a masked or restored
    /// generation actually produced.
    #[wasm_bindgen(js_name = decodeIds)]
    pub fn decode_ids(&self, token_ids: Vec<u32>) -> Result<String, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        tokenizer.decode(&token_ids, true).map_err(|e| JsError::new(&format!("tokenizer decode failed: {e}")))
    }

    /// Packs `allowed_ids` into this crate's mask-bitset format
    /// (`model::build_mask_bitset`) for `mask_bits` arguments below - a
    /// consumer's grammar/schema loop calls this once per step with that
    /// step's allowed vocabulary, then passes the result straight through.
    #[wasm_bindgen(js_name = buildMaskBitset)]
    pub fn build_mask_bitset_js(&self, allowed_ids: Vec<u32>) -> Result<Vec<u32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(build_mask_bitset(model.config.vocab_size, &allowed_ids))
    }

    /// Low-level prefill over raw token ids (no chat-template rendering -
    /// use `tokenize()` first if needed): resets the pool and starts a fresh
    /// KV cache at position 0, runs prefill, and returns the last position's
    /// logits (`vocab_size` long). `mask_bits`, if non-empty, constrains the
    /// *first* generated token the same way `decodeStepArgmax`'s mask
    /// constrains later ones (see `mask_buf`'s doc comment on the empty-vec
    /// convention).
    #[wasm_bindgen(js_name = prefillTokens)]
    pub async fn prefill_tokens(&mut self, token_ids: Vec<u32>, mask_bits: Vec<u32>) -> Result<Vec<f32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let mask = mask_buf(&self.engine, model.config.vocab_size, &mask_bits)?;
        if token_ids.len() as u32 > cache.max_ctx {
            return Err(JsError::new("token_ids exceeds max_ctx"));
        }
        cache.kv_len = 0;
        self.prefix.invalidate();
        Ok(forward_prefill(&self.engine, model, cache, &token_ids, cos_buf, sin_buf, mask.as_ref()).await)
    }

    /// Debug only: the adapter/device features and limits this engine was
    /// created with (JSON, see `Engine::adapter_report`).
    #[wasm_bindgen(js_name = debugAdapter)]
    pub fn debug_adapter(&self) -> String {
        self.engine.adapter_report.clone()
    }

    /// Debug only: record every weight upload from now on (call before
    /// `load`), so `debugVerifyUploads` can check the device copies.
    #[wasm_bindgen(js_name = debugRecordUploads)]
    pub fn debug_record_uploads(&self, on: bool) {
        self.engine.set_debug_uploads(on);
    }

    /// Debug only: reads back every recorded upload and reports the buffers
    /// whose device bytes differ from what was uploaded (JSON).
    #[wasm_bindgen(js_name = debugVerifyUploads)]
    pub async fn debug_verify_uploads(&self) -> String {
        self.engine.debug_verify_uploads().await
    }

    /// Debug only: a fresh prefill of `token_ids` (same as `prefillTokens`
    /// with no mask) with op taps on, returning one checksum row per op
    /// (JSON array, see `Engine::debug_collect`). The taps split compute
    /// passes, so use it to compare runs with each other, not for timing.
    #[wasm_bindgen(js_name = debugPrefill)]
    pub async fn debug_prefill(&mut self, token_ids: Vec<u32>) -> Result<String, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        if token_ids.len() as u32 > cache.max_ctx {
            return Err(JsError::new("token_ids exceeds max_ctx"));
        }
        model.pool.reset();
        cache.kv_len = 0;
        self.prefix.invalidate();
        self.engine.set_debug_taps(true);
        let _ = forward_prefill(&self.engine, model, cache, &token_ids, cos_buf, sin_buf, None).await;
        let rows = self.engine.debug_collect().await;
        self.engine.set_debug_taps(false);
        Ok(rows)
    }

    /// Appends `token_ids` onto the *existing* KV cache (typically just
    /// after `restoreKv` from a resident-prefix snapshot, or continuing a
    /// live session) instead of starting a fresh one - the "prefill(suffix)"
    /// half of KV snapshot/restore. Returns the last position's logits.
    #[wasm_bindgen(js_name = appendTokens)]
    pub async fn append_tokens(&mut self, token_ids: Vec<u32>, mask_bits: Vec<u32>) -> Result<Vec<f32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let mask = mask_buf(&self.engine, model.config.vocab_size, &mask_bits)?;
        if token_ids.is_empty() {
            return Err(JsError::new("appendTokens needs at least one token"));
        }
        if cache.kv_len + token_ids.len() as u32 > cache.max_ctx {
            return Err(JsError::new("appendTokens would exceed max_ctx"));
        }
        self.prefix.invalidate();
        Ok(forward_prefill_suffix(&self.engine, model, cache, &token_ids, cos_buf, sin_buf, mask.as_ref()).await)
    }

    /// Decodes one token against the existing KV cache and returns the
    /// argmax id, same fast path `generate()` uses internally, exposed for a
    /// consumer driving its own step loop (e.g. after `restoreKv`, or with a
    /// per-step mask that changes every call - a fixed mask across an entire
    /// `generate()` call would not let a grammar narrow the allowed set as
    /// it consumes each token). `mask_bits` is applied before argmax, in the
    /// same GPU submission as the rest of the step - no extra readback.
    #[wasm_bindgen(js_name = decodeStepArgmax)]
    pub async fn decode_step_argmax(&mut self, token_id: u32, mask_bits: Vec<u32>) -> Result<u32, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let mask = mask_buf(&self.engine, model.config.vocab_size, &mask_bits)?;
        if cache.kv_len >= cache.max_ctx {
            return Err(JsError::new("decodeStepArgmax would exceed max_ctx"));
        }
        self.prefix.invalidate();
        Ok(forward_decode_step_argmax(&self.engine, model, cache, token_id, cos_buf, sin_buf, mask.as_ref()).await)
    }

    /// Greedy decode of `steps` forward steps from `token_id` (no EOS stop),
    /// pipelined: each step is submitted before the previous step's id is
    /// read back (see `model::decode_greedy_pipelined`). Returns the ids the
    /// steps produced, the same as `steps` calls of `decodeStepArgmax`
    /// chained on their own outputs.
    #[wasm_bindgen(js_name = decodeGreedy)]
    pub async fn decode_greedy(&mut self, token_id: u32, steps: u32) -> Result<Vec<u32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        if cache.kv_len + steps > cache.max_ctx {
            return Err(JsError::new("decodeGreedy would exceed max_ctx"));
        }
        if steps == 0 {
            return Ok(Vec::new());
        }
        // `on_token` sees each step's input (token_id first); the last
        // step's output comes back separately.
        let (mut ids, last) = decode_greedy_pipelined(&self.engine, model, cache, token_id, steps, false, cos_buf, sin_buf, None, |_| true).await;
        ids.extend(last);
        ids.remove(0);
        Ok(ids)
    }

    /// The number of positions currently populated in the KV cache (0 right
    /// after `load()` or a fresh `prefillTokens`, grows with `appendTokens`/
    /// `decodeStepArgmax`, or is set directly by `restoreKv`).
    #[wasm_bindgen(js_name = kvLen)]
    pub fn kv_len(&self) -> Result<u32, JsError> {
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(cache.kv_len)
    }

    /// Reads back the KV cache's `[0, kvLen())` prefix and returns it as
    /// `KvSnapshot::to_bytes()` - a resident-prefix image a consumer can
    /// store keyed by its own prompt/tool-schema hash (see this crate's
    /// consumer survey, gap #1) and later hand back to `restoreKv`. Async
    /// (`Engine::read_buffer`'s `into_data_async` path) - never blocks the
    /// browser's main/worker thread.
    #[wasm_bindgen(js_name = snapshotKv)]
    pub async fn snapshot_kv(&self) -> Result<Vec<u8>, JsError> {
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(cache.snapshot(&self.engine).await.to_bytes())
    }

    /// Restores a snapshot produced by `snapshotKv` (this session's own, or
    /// one a consumer stored earlier and is handing back) into the current
    /// KV cache, and sets `kvLen()` to the snapshot's length. Queued
    /// `queue.write_buffer` calls only - no readback, safe to call
    /// synchronously. Follow with `appendTokens` for the resumed suffix, or
    /// `decodeStepArgmax` to continue decoding directly from the restored
    /// prefix's last position.
    #[wasm_bindgen(js_name = restoreKv)]
    pub fn restore_kv(&mut self, bytes: Vec<u8>) -> Result<(), JsError> {
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let snapshot = KvSnapshot::from_bytes(&bytes).map_err(|e| JsError::new(&format!("failed to parse kv snapshot: {e}")))?;
        if snapshot.kv_len > cache.max_ctx {
            return Err(JsError::new("kv snapshot's kv_len exceeds this engine's max_ctx"));
        }
        cache.restore(&self.engine, &snapshot);
        self.prefix.invalidate();
        Ok(())
    }

}

impl LeanEngine {
    fn from_engine(engine: Engine) -> LeanEngine {
        wasm_log(&format!("[lean] device ready, max_storage_buffer_binding_size={}", engine.max_storage_buffer_binding_size()));
        LeanEngine { engine, model: None, cache: None, cos_buf: None, sin_buf: None, tokenizer: None, chat_template: None, prefix: PrefixCache::new(), load_ms: (0.0, 0.0) }
    }
}

/// The CPU backend's wasm-bindgen surface (`cpu.rs`, one thread or a
/// rayon pool in the `wasm-mt` build): same method names and argument
/// shapes as `LeanEngine` wherever a CPU equivalent exists, so a page holds
/// either behind the same call sites (`create`/`load`/`generate`/
/// `renderPrompt`/`tokenCount`/`tokenize`/`decodeIds`/`prefillTokens`/
/// `decodeStepArgmax`). `generate` is async like the GPU one and yields
/// to the event loop between tokens so an `AbortFlag` can be flipped; the
/// other methods are synchronous and block the calling thread for the
/// forward pass (run them in a Web Worker). No LoRA or KV-snapshot surface
/// yet.
#[wasm_bindgen]
pub struct LeanEngineCpu {
    model: Option<CpuModel>,
    cache: Option<CpuKvCache>,
    tokenizer: Option<Tokenizer>,
    chat_template: Option<String>,
    /// Same role as `LeanEngine::chat`.
    prefix: PrefixCache,
}

#[wasm_bindgen]
impl LeanEngineCpu {
    /// No adapter/device to request (unlike `LeanEngine::create`) - kept as
    /// a function (not a plain struct literal) for API-shape symmetry with
    /// the GPU surface's `create()`.
    #[wasm_bindgen(js_name = create)]
    pub fn create() -> LeanEngineCpu {
        console_error_panic_hook::set_once();
        LeanEngineCpu { model: None, cache: None, tokenizer: None, chat_template: None, prefix: PrefixCache::new() }
    }

    /// Parses `gguf_bytes` (a `Uint8Array` view, same reasoning as
    /// `LeanEngine::load`) into a CPU-resident model (Q4_0/Q8_0/Q6_K tensor
    /// bytes held as-is - see `cpu.rs`'s doc comment, this is the one rung
    /// that legitimately needs its own resident copy of the quantized
    /// bytes, since it computes directly off them) and allocates a
    /// `CpuKvCache` sized to `max_ctx`. Same signature as `LeanEngine::load`
    /// minus the `Result` needing to report GPU-adapter failures.
    #[wasm_bindgen(js_name = load)]
    pub fn load(&mut self, gguf_bytes: js_sys::Uint8Array, tokenizer_json: String, tokenizer_config_json: String, max_ctx: u32) -> Result<(), JsError> {
        let model = CpuModel::load_from_reader(JsBytesReader::new(gguf_bytes)).map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        let cache = CpuKvCache::new(&model.config, max_ctx as usize);
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
        let chat_template = chat_template_from_config_json(&tokenizer_config_json).map_err(|e| JsError::new(&format!("{e}")))?;
        wasm_log(&format!("[lean-cpu] model loaded: layers={} hidden={} vocab={}", model.config.num_layers, model.config.hidden_size, model.config.vocab_size));
        self.model = Some(model);
        self.cache = Some(cache);
        self.tokenizer = Some(tokenizer);
        self.chat_template = Some(chat_template);
        self.prefix.invalidate();
        Ok(())
    }

    /// Same contract as `LeanEngine::generate` (one inference turn over
    /// `messages`, prefix reuse against the previous call, text streamed
    /// through `onToken(text)`), on the CPU forward pass. `mask_bits` in
    /// `params` is applied the same way `mask_logits.wgsl` applies it on
    /// the GPU.
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&mut self, messages: JsValue, params: JsValue, on_token: JsValue, abort: Option<AbortFlag>) -> Result<String, JsValue> {
        let model = self.model.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;

        let messages = messages_from_js(&messages)?;
        let params = gen_params_from_js(&params);
        if let Some(bits) = &params.mask_bits {
            if !bits.is_empty() && bits.len() * 32 < model.config.vocab_size {
                return Err(JsValue::from(JsError::new("mask is shorter than vocab_size / 32")));
            }
        }

        let sink = TextSink::new(on_token);
        let should_stop = || abort.as_ref().is_some_and(AbortFlag::is_aborted) || sink.borrow().stopped();
        let result = cpu_generate(model, cache, &mut self.prefix, tokenizer, chat_template, &messages, &params, |text| sink.borrow_mut().text(text), should_stop, yield_to_event_loop).await;
        if let Some(e) = sink.borrow_mut().take_err() {
            return Err(e);
        }
        result.map_err(lean_error_to_js)
    }

    /// Same as `LeanEngine::renderPrompt`.
    #[wasm_bindgen(js_name = renderPrompt)]
    pub fn render_prompt(&self, messages: JsValue) -> Result<String, JsValue> {
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let messages = messages_from_js(&messages)?;
        chat_render_prompt(chat_template, &messages).map_err(lean_error_to_js)
    }

    /// Same as `LeanEngine::tokenCount`.
    #[wasm_bindgen(js_name = tokenCount)]
    pub fn token_count(&self, messages: JsValue) -> Result<u32, JsValue> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsValue::from(JsError::new("load() must be called first")))?;
        let messages = messages_from_js(&messages)?;
        chat_token_count(tokenizer, chat_template, &messages).map(|n| n as u32).map_err(lean_error_to_js)
    }

    #[wasm_bindgen(js_name = info)]
    pub fn info(&self) -> String {
        match &self.model {
            Some(model) => serde_json::json!({
                "loaded": true,
                "numLayers": model.config.num_layers,
                "hiddenSize": model.config.hidden_size,
                "vocabSize": model.config.vocab_size,
            })
            .to_string(),
            None => serde_json::json!({ "loaded": false }).to_string(),
        }
    }

    #[wasm_bindgen(js_name = tokenize)]
    pub fn tokenize(&self, prompt: String) -> Result<Vec<u32>, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let rendered = render_user_prompt(chat_template, &prompt).map_err(|e| JsError::new(&format!("failed to render prompt: {e}")))?;
        let encoding = tokenizer.encode(rendered, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    #[wasm_bindgen(js_name = encodeRaw)]
    pub fn encode_raw(&self, text: String) -> Result<Vec<u32>, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let encoding = tokenizer.encode(text, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    #[wasm_bindgen(js_name = decodeIds)]
    pub fn decode_ids(&self, token_ids: Vec<u32>) -> Result<String, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        tokenizer.decode(&token_ids, true).map_err(|e| JsError::new(&format!("tokenizer decode failed: {e}")))
    }

    /// Low-level prefill over raw token ids - resets the KV cache to
    /// position 0, runs prefill, returns the last position's logits. No
    /// mask argument (unlike `LeanEngine::prefillTokens`) - `cpu.rs` has no
    /// mask support yet.
    #[wasm_bindgen(js_name = prefillTokens)]
    pub fn prefill_tokens(&mut self, token_ids: Vec<u32>) -> Result<Vec<f32>, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        if token_ids.len() > cache.max_ctx() {
            return Err(JsError::new("token_ids exceeds max_ctx"));
        }
        cache.kv_len = 0;
        self.prefix.invalidate();
        Ok(cpu_forward_prefill(model, cache, &token_ids))
    }

    #[wasm_bindgen(js_name = decodeStepArgmax)]
    pub fn decode_step_argmax(&mut self, token_id: u32) -> Result<u32, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        if cache.remaining_capacity() == 0 {
            return Err(JsError::new("decodeStepArgmax would exceed max_ctx"));
        }
        self.prefix.invalidate();
        Ok(cpu_decode_step_argmax(model, cache, token_id))
    }

    #[wasm_bindgen(js_name = kvLen)]
    pub fn kv_len(&self) -> Result<u32, JsError> {
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(cache.kv_len as u32)
    }

}
