//! Minimal wasm-bindgen browser surface. One entry point, `LeanEngine`, for
//! a Web Worker: `LeanEngine::create()` (async - requests the WebGPU
//! adapter/device), `load()` (parse GGUF bytes + tokenizer + chat template,
//! upload weights to GPU), `generate()` (render -> tokenize -> prefill ->
//! greedy decode, one `on_token(id: number)` JS callback per generated
//! token). No agent/tool/grammar layer here (that's llm-wasm's `web.rs`) -
//! this crate's browser surface is exactly what `lean-cli` does natively,
//! reused verbatim (`chat_template::render_user_prompt`,
//! `model::{forward_prefill, forward_decode_step}`).
//!
//! WebGPU readback is async only in the browser - every GPU-reading call
//! here is `async`/`.await`s `Engine::read_buffer`'s `into_data_async`-style
//! path (see `engine.rs`'s doc comment). Never call a blocking readback from
//! this module.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use std::io::Cursor;

use tokenizers::Tokenizer;

use crate::chat_template::{chat_template_from_config_json, render_user_prompt};
use crate::engine::Engine;
use crate::model::{build_rope_tables, forward_decode_step_argmax, forward_prefill, GpuModel, KvCache};

fn wasm_log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Initializes the panic hook for readable browser-console error messages.
/// Optional but recommended: call once before `LeanEngine::create()`.
#[wasm_bindgen(js_name = leanInit)]
pub fn lean_init() {
    console_error_panic_hook::set_once();
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as u32
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
        wasm_log(&format!(
            "[lean] device ready, max_storage_buffer_binding_size={}",
            engine.max_storage_buffer_binding_size()
        ));
        Ok(LeanEngine { engine, model: None, cache: None, cos_buf: None, sin_buf: None, tokenizer: None, chat_template: None })
    }

    /// Parses `gguf_bytes` (the whole GGUF file, fetched by JS) and uploads
    /// every weight to the GPU (two-phase loading: `gguf_bytes` and the
    /// `Cursor`/`GgufReader` wrapping it are dropped inside
    /// `GpuModel::load_from_reader` before this call returns, well before
    /// `max_ctx`'s KV cache is allocated). `tokenizer_json`/
    /// `tokenizer_config_json` are the two files' contents as strings.
    /// `max_ctx` bounds the KV cache (prompt + max_new_tokens must fit).
    #[wasm_bindgen(js_name = load)]
    pub fn load(&mut self, gguf_bytes: Vec<u8>, tokenizer_json: String, tokenizer_config_json: String, max_ctx: u32) -> Result<(), JsError> {
        let model = GpuModel::load_from_reader(&self.engine, Cursor::new(gguf_bytes), true)
            .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        let cache = KvCache::new(&self.engine, &model.config, max_ctx);
        let (cos, sin) = build_rope_tables(model.config.head_dim, model.config.rope_theta, max_ctx as usize);
        let cos_buf = self.engine.buf_f32(&cos, "rope_cos");
        let sin_buf = self.engine.buf_f32(&sin, "rope_sin");

        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
        let chat_template = chat_template_from_config_json(&tokenizer_config_json).map_err(|e| JsError::new(&format!("{e}")))?;

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
        Ok(())
    }

    /// Renders `prompt` through the model's own chat template (single user
    /// turn, `add_generation_prompt = true` - same shape as `lean-cli`'s
    /// `--prompt` path), tokenizes it, prefills, then greedily decodes up to
    /// `max_new_tokens` tokens (stopping early on any of the model's
    /// `eos_token_ids`). Every decoded token id is passed to `on_token`
    /// (called as `on_token(id: number)`) as soon as it's produced - a
    /// no-op if `on_token` isn't a JS function. Returns the decoded
    /// continuation text and prefill/decode timing.
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&mut self, prompt: String, max_new_tokens: u32, on_token: JsValue) -> Result<String, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cos_buf = self.cos_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let sin_buf = self.sin_buf.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;

        // A fresh generation per call: reset the pool's stale bind-group
        // cache (see pool.rs's bug note referenced in lean_cli.rs) and
        // restart the KV cache from position 0.
        model.pool.reset();
        cache.kv_len = 0;

        let rendered = render_user_prompt(chat_template, &prompt).map_err(|e| JsError::new(&format!("failed to render prompt: {e}")))?;
        let encoding = tokenizer.encode(rendered, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        let token_ids = encoding.get_ids().to_vec();
        if token_ids.len() as u32 + max_new_tokens > cache.max_ctx {
            return Err(JsError::new("prompt + max_new_tokens exceeds max_ctx"));
        }

        let on_token = on_token.dyn_into::<js_sys::Function>().ok();
        let call_on_token = |id: u32| {
            if let Some(f) = &on_token {
                let _ = f.call1(&JsValue::NULL, &JsValue::from(id));
            }
        };

        self.engine.reset_dispatch_count();
        let logits = forward_prefill(&self.engine, model, cache, &token_ids, cos_buf, sin_buf).await;
        let prefill_dispatches = self.engine.dispatch_count();
        // Only the prompt's last-position argmax runs on the CPU (off the
        // one `Vec<f32>` prefill already reads back); every decode step's
        // argmax runs on the GPU inside `forward_decode_step_argmax`, so
        // decode's readback is 4 bytes/step, not `vocab_size * 4` - see
        // model.rs's doc comment.
        let mut next_id = argmax(&logits);

        self.engine.reset_dispatch_count();
        let mut generated = Vec::with_capacity(max_new_tokens as usize);
        for _ in 0..max_new_tokens {
            if model.config.eos_token_ids.contains(&next_id) {
                break;
            }
            generated.push(next_id);
            call_on_token(next_id);
            next_id = forward_decode_step_argmax(&self.engine, model, cache, next_id, cos_buf, sin_buf).await;
        }
        let decode_steps = generated.len().max(1) as u64;
        wasm_log(&format!(
            "[lean] seq={} prefill_dispatches={} ({:.1}/token) decode_dispatches_per_step={}",
            token_ids.len(),
            prefill_dispatches,
            prefill_dispatches as f64 / token_ids.len() as f64,
            self.engine.dispatch_count() / decode_steps
        ));

        tokenizer.decode(&generated, true).map_err(|e| JsError::new(&format!("tokenizer decode failed: {e}")))
    }

    /// Renders + tokenizes `prompt` the same way `generate()` does and
    /// returns the resulting token count, with no GPU work - lets a harness
    /// log a synthetic timing-only prompt's actual length (e.g. the
    /// ~1000-token prefill case in `www/main.js`) without duplicating the
    /// chat-template/tokenizer path in JS.
    #[wasm_bindgen(js_name = tokenCount)]
    pub fn token_count(&self, prompt: String) -> Result<u32, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let chat_template = self.chat_template.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let rendered = render_user_prompt(chat_template, &prompt).map_err(|e| JsError::new(&format!("failed to render prompt: {e}")))?;
        let encoding = tokenizer.encode(rendered, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().len() as u32)
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
}
