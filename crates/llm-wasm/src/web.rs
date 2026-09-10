//! wasm-bindgen browser surface for the xLAM-2-3b-fc-r agent.
//!
//! Mirrors stt-web's `stt-wasm/src/web/bindings.rs`: a module-level
//! `initWgpuDevice()` that must be awaited once before constructing
//! `LlmEngine`, then `LlmEngine` itself as the single entry point a Web
//! Worker calls.
//!
//! ## Why the step loop is reimplemented here instead of reusing `Agent`
//!
//! `agent.rs`'s `Agent::start`/`provide_tool_results` are synchronous —
//! right for native callers (the CLI, tests), where Burn's wgpu backend can
//! block on tensor readback (`Tensor::into_data()`, used by
//! `model::logits_to_vec`). In the browser that readback is a WebGPU buffer
//! map, which is **asynchronous only** — there is no blocking equivalent
//! (see `model.rs`'s `logits_to_vec` doc comment: "WASM callers must use
//! `into_data_async().await` instead"). `Generator::generate`'s signature is
//! synchronous, so it cannot wrap an async readback; a real GPU-backed
//! `Generator` impl only exists for native.
//!
//! Rather than block the browser (impossible) or fake synchrony, `LlmEngine`
//! runs its own async decode loop directly against `model.rs`'s public
//! `forward_hidden`/`lm_head` and `kv.rs`'s `KvCache`, `await`ing
//! `into_data_async()` once per generated token, and calls the same
//! `sample::greedy` and `tools::parse_output`/`format_tool_result` functions
//! `Agent` uses natively. This duplicates `Agent::step_inner`'s
//! orchestration (render → encode → generate → parse → bookkeeping) in an
//! async form; it does **not** duplicate any model/tokenizer/template/tool
//! logic — every actual computation still goes through the same functions
//! `agent.rs` calls. If Burn ever grows a sync-over-async escape hatch for
//! wasm32 (e.g. via Atomics.wait in a cross-origin-isolated Worker), this
//! duplication could be removed by implementing `Generator` for a
//! wasm-backed type instead.
//!
//! Prefix caching here works the same way as `Agent`'s: the previous turn's
//! prompt (or step's) exact token sequence is retained in `resident_tokens`
//! (mirroring what's actually written into the `KvCache`, position for
//! position); a new prompt whose leading tokens match are served by
//! `KvCache::restore()`-ing to that length and prefilling only the new
//! suffix, instead of the whole prompt.

use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

use burn::backend::wgpu::WgpuDevice;

use crate::agent::PendingToolCall;
use crate::gguf::Q4ModelLoader;
use crate::kv::KvCache;
use crate::model::LlmModel;
use crate::sample::greedy;
use crate::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use crate::tokenizer::Tokenizer;
use crate::tools::{format_tool_result, parse_output, ParsedOutput};

/// Default context length: sized (per `kv.rs`'s doc comment) for the 34-tool
/// Sonos prompt plus conversation headroom.
const DEFAULT_MAX_CTX: usize = 12288;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_MAX_STEPS: usize = 6;

/// Device initialized by `initWgpuDevice()` — used by every `LlmEngine`.
static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn wasm_log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Initialize panic hook for readable browser-console error messages.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the WebGPU device asynchronously. **Must** be called (and
/// awaited) once before constructing any `LlmEngine`. Requests the
/// adapter's full limits, same as stt-web's `initWgpuDevice` — this model's
/// 175MB single-buffer tied lm_head needs `maxStorageBufferBindingSize`
/// above the WebGPU spec default (see docs/ENGINE.md's Browser section).
#[wasm_bindgen(js_name = initWgpuDevice)]
pub async fn init_wgpu_device() -> Result<(), JsError> {
    use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| JsError::new("No WebGPU adapter found"))?;

    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    wasm_log(&format!(
        "[llm] Adapter: {} ({:?}), backend: {:?}, max_buffer_size={}, max_storage_buffer_binding_size={}",
        info.name, info.device_type, info.backend, adapter_limits.max_buffer_size, adapter_limits.max_storage_buffer_binding_size,
    ));

    let features = adapter.features() - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;

    // Detect subgroup support for the cooperative matvec kernel's subgroup
    // variant (`gguf.rs::has_subgroup_support()`); falls back to the
    // portable kernel otherwise.
    //
    // The subgroup kernel (`shader_q4_matvec_subgroup.wgsl`) hard-codes
    // SUBGROUP_SIZE=32 — enabling it on a device whose actual subgroup size
    // is 8/16/64 (e.g. some Intel/AMD/mobile GPUs) silently produces wrong
    // sums, not an error. `wgpu::Features::SUBGROUP` alone only says the
    // *feature* is available, not what size it runs at, so gate on
    // `min_subgroup_size == max_subgroup_size == 32` as well. As of wgpu
    // 26's `BROWSER_WEBGPU` backend, `adapter.limits()` doesn't surface a
    // real subgroup size from the browser (`min_subgroup_size`/
    // `max_subgroup_size` come back `Limits::default()`, i.e. 0/0 — see
    // `wgpu-26.0.1/src/backend/webgpu.rs`), so this gate keeps the subgroup
    // kernel disabled on WebGPU today; it only activates once wgpu (or the
    // WebGPU spec's `adapter-info` subgroup extension) actually reports a
    // real size.
    let subgroup_size_confirmed_32 =
        adapter_limits.min_subgroup_size == 32 && adapter_limits.max_subgroup_size == 32;
    let subgroups_available =
        features.contains(wgpu::Features::SUBGROUP) && subgroup_size_confirmed_32;
    crate::gguf::set_subgroup_support(subgroups_available);
    wasm_log(&format!(
        "[llm] Subgroup support: {subgroups_available} (feature={}, min_subgroup_size={}, max_subgroup_size={})",
        features.contains(wgpu::Features::SUBGROUP),
        adapter_limits.min_subgroup_size,
        adapter_limits.max_subgroup_size,
    ));

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("llm-wgpu"),
            required_features: features,
            required_limits: adapter_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| JsError::new(&format!("Failed to create WebGPU device: {e}")))?;

    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: info.backend,
    };
    let wgpu_device = init_device(setup, RuntimeOptions::default());
    WGPU_DEVICE
        .set(wgpu_device)
        .map_err(|_| JsError::new("initWgpuDevice() called more than once"))?;
    Ok(())
}

/// Browser-facing agent engine. Single entry point a Web Worker calls —
/// see `web/agent/worker.js` for the message protocol built on it.
#[wasm_bindgen]
pub struct LlmEngine {
    device: WgpuDevice,
    shard_bufs: Vec<Vec<u8>>,

    model: Option<LlmModel>,
    cache: Option<KvCache>,
    /// Exact token sequence currently resident in `cache`, position for
    /// position from offset 0 — see module docs on prefix caching.
    resident_tokens: Vec<u32>,

    tokenizer: Option<Tokenizer>,
    template: Option<ChatTemplate>,

    system_prompt: String,
    max_new_tokens: usize,
    max_steps: usize,

    messages: Vec<Message>,
    tools: Vec<Tool>,
    pending_calls: Vec<PendingToolCall>,
    step_index: usize,
    /// Cached (tools set, common prefix token length) — recomputed only
    /// when `tools` changes between `start()` calls.
    prefix_cache: Option<(Vec<Tool>, usize)>,
}

#[wasm_bindgen]
impl LlmEngine {
    /// `#[wasm_bindgen(constructor)]` cannot return `Result` (no fallible
    /// JS constructor), so if `initWgpuDevice()` wasn't awaited first this
    /// silently falls back to `WgpuDevice::default()` rather than erroring
    /// — logged as a `console.warn` since a fallback device here almost
    /// certainly means every subsequent GPU call fails or targets the
    /// wrong adapter.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let device = WGPU_DEVICE.get().cloned().unwrap_or_else(|| {
            web_sys::console::warn_1(&JsValue::from_str(
                "[llm] LlmEngine::new() called before initWgpuDevice() completed — \
                 falling back to WgpuDevice::default(), which will likely fail on GPU calls",
            ));
            WgpuDevice::default()
        });
        Self {
            device,
            shard_bufs: Vec::new(),
            model: None,
            cache: None,
            resident_tokens: Vec::new(),
            tokenizer: None,
            template: None,
            system_prompt: "You are a helpful assistant with access to tools.".to_string(),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_steps: DEFAULT_MAX_STEPS,
            messages: Vec::new(),
            tools: Vec::new(),
            pending_calls: Vec::new(),
            step_index: 0,
            prefix_cache: None,
        }
    }

    /// Append one GGUF shard (any split the caller likes — a single
    /// element is fine for a non-sharded file). Call before `load()`.
    #[wasm_bindgen(js_name = appendModelShard)]
    pub fn append_model_shard(&mut self, shard: &[u8]) {
        self.shard_bufs.push(shard.to_vec());
        wasm_log(&format!(
            "[llm] shard appended ({} bytes, {} total)",
            shard.len(),
            self.shard_bufs.len()
        ));
    }

    /// Parse the GGUF (from previously `appendModelShard`-ed bytes), upload
    /// weights to GPU, and build the tokenizer + chat template.
    ///
    /// `on_progress`, if a function, is called as `on_progress(stage:
    /// string, step: number, total: number)` at each of this method's three
    /// phases (`"parsing-gguf"`, `"finalizing-gpu"`, `"ready"`) — coarse
    /// progress, since `gguf.rs`'s `load_deferred` loads all 36 transformer
    /// layers in one call with no per-layer hook. Byte-level shard fetch
    /// progress is the caller's job (see `web/agent/worker.js`'s `load`
    /// handler, which reports progress while fetching, before ever calling
    /// `appendModelShard`/`load`).
    ///
    /// GGUF bytes stay as whatever shards were appended — this crate's
    /// `Q4ModelLoader::from_shards` reads through a `ShardedCursor`
    /// (`gguf.rs`), so the 1.74GB model file need not be one contiguous
    /// buffer; the caller can split the fetch into e.g. 64-128MB chunks (or
    /// pass one element) — either fits comfortably under wasm32's 4GB
    /// address space.
    #[wasm_bindgen(js_name = load)]
    pub async fn load(
        &mut self,
        tokenizer_json: String,
        tokenizer_config_json: String,
        on_progress: JsValue,
    ) -> Result<(), JsError> {
        if self.shard_bufs.is_empty() {
            return Err(JsError::new("No shards appended. Call appendModelShard first."));
        }
        let report = |stage: &str, step: u32, total: u32| {
            if let Some(f) = on_progress.dyn_ref::<js_sys::Function>() {
                let _ = f.call3(
                    &JsValue::NULL,
                    &JsValue::from_str(stage),
                    &JsValue::from(step),
                    &JsValue::from(total),
                );
            }
        };

        report("parsing-gguf", 0, 3);
        let shards = std::mem::take(&mut self.shard_bufs);
        let parts = {
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| JsError::new(&format!("failed to open GGUF: {e}")))?;
            loader
                .load_deferred(&self.device)
                .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?
            // loader (and shard bytes) dropped here before GPU finalize.
        };

        report("finalizing-gpu", 1, 3);
        let model = parts
            .finalize(&self.device)
            .map_err(|e| JsError::new(&format!("failed to finalize model on GPU: {e}")))?;
        let cache = model.new_cache(DEFAULT_MAX_CTX);

        let tokenizer = Tokenizer::from_json(tokenizer_json.as_bytes())
            .map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
        let cfg: serde_json::Value = serde_json::from_str(&tokenizer_config_json)
            .map_err(|e| JsError::new(&format!("invalid tokenizer_config.json: {e}")))?;
        let template = ChatTemplate::from_tokenizer_config(&cfg)
            .map_err(|e| JsError::new(&format!("failed to compile chat_template: {e}")))?;

        self.model = Some(model);
        self.cache = Some(cache);
        self.resident_tokens.clear();
        self.tokenizer = Some(tokenizer);
        self.template = Some(template);

        report("ready", 3, 3);
        Ok(())
    }

    /// Set the system prompt used by subsequent `start()` calls.
    #[wasm_bindgen(js_name = setSystemPrompt)]
    pub fn set_system_prompt(&mut self, system_prompt: String) {
        self.system_prompt = system_prompt;
    }

    /// Begin a new turn. `tools_json` is a JSON array of MCP `tools/list`
    /// entries (`{"name", "description", "inputSchema"}`); `opts_json` is
    /// `{"maxNewTokens"?: number, "maxSteps"?: number}` (both optional; `{}`
    /// or `"{}"` is fine). Returns a JSON string: `{"outcome":"needTools",
    /// "calls":[{"call_id","name","arguments"}...],"step":{...}}`,
    /// `{"outcome":"final","text":...,"step":{...}}`, or
    /// `{"outcome":"error","message":...}` — see docs/ENGINE.md's Browser
    /// section for the full shape and `web/agent/worker.js` for how it's
    /// consumed. No `on_token` streaming callback: `model.rs`'s `generate`
    /// has no per-token hook at HEAD, so a step's text arrives in one piece
    /// when the step completes (see module docs).
    #[wasm_bindgen(js_name = start)]
    pub async fn start(&mut self, utterance: String, tools_json: String, opts_json: String) -> Result<String, JsError> {
        self.tools = parse_tools(&tools_json)?;
        apply_opts(self, &opts_json)?;
        self.messages = vec![Message::system(&self.system_prompt), Message::user(&utterance)];
        self.step_index = 0;
        self.pending_calls.clear();

        match self.compute_prefix_len(&utterance) {
            Ok(len) => self.prefix_cache = Some((self.tools.clone(), len)),
            Err(e) => return Ok(error_json(&format!("failed to compute prefix cache length: {e}"))),
        }

        self.run_step().await
    }

    /// Continue the current turn with tool results, keyed by `call_id` from
    /// the most recent `NeedTools` outcome. `results_json`:
    /// `[{"call_id":"call_0","result":{...}}, ...]`. Same return shape as
    /// `start()`.
    #[wasm_bindgen(js_name = provideToolResults)]
    pub async fn provide_tool_results(&mut self, results_json: String) -> Result<String, JsError> {
        let results: Vec<ToolResultIn> = serde_json::from_str(&results_json)
            .map_err(|e| JsError::new(&format!("invalid results_json: {e}")))?;
        for r in results {
            if let Some(pending) = self.pending_calls.iter().find(|c| c.call_id == r.call_id) {
                self.messages.push(format_tool_result(&pending.name, &r.result));
            }
        }
        self.pending_calls.clear();
        self.run_step().await
    }

    /// Drop the in-progress conversation (messages/tools/pending calls) and
    /// the KV cache's resident tokens, keeping the loaded model.
    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        self.messages.clear();
        self.tools.clear();
        self.pending_calls.clear();
        self.step_index = 0;
        self.prefix_cache = None;
        self.resident_tokens.clear();
        if let Some(cache) = self.cache.as_mut() {
            cache.restore(0);
        }
    }

    /// JSON string with basic model/device info, for the page's status line.
    #[wasm_bindgen(js_name = info)]
    pub fn info(&self) -> String {
        match &self.model {
            Some(model) => {
                let cfg = model.config();
                serde_json::json!({
                    "loaded": true,
                    "numLayers": cfg.num_layers,
                    "hiddenSize": cfg.hidden_size,
                    "vocabSize": cfg.vocab_size,
                    "maxCtx": DEFAULT_MAX_CTX,
                    "cacheLen": self.cache.as_ref().map(|c| c.len()).unwrap_or(0),
                })
                .to_string()
            }
            None => serde_json::json!({ "loaded": false }).to_string(),
        }
    }
}

impl Default for LlmEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(serde::Deserialize)]
struct ToolResultIn {
    call_id: String,
    result: serde_json::Value,
}

#[derive(serde::Deserialize, Default)]
struct OptsIn {
    #[serde(rename = "maxNewTokens")]
    max_new_tokens: Option<usize>,
    #[serde(rename = "maxSteps")]
    max_steps: Option<usize>,
    #[serde(rename = "systemPrompt")]
    system_prompt: Option<String>,
}

fn apply_opts(engine: &mut LlmEngine, opts_json: &str) -> Result<(), JsError> {
    let trimmed = opts_json.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let opts: OptsIn = serde_json::from_str(trimmed).map_err(|e| JsError::new(&format!("invalid opts_json: {e}")))?;
    if let Some(n) = opts.max_new_tokens {
        engine.max_new_tokens = n;
    }
    if let Some(n) = opts.max_steps {
        engine.max_steps = n;
    }
    if let Some(s) = opts.system_prompt {
        engine.system_prompt = s;
    }
    Ok(())
}

fn parse_tools(tools_json: &str) -> Result<Vec<Tool>, JsError> {
    let raw: Vec<serde_json::Value> =
        serde_json::from_str(tools_json).map_err(|e| JsError::new(&format!("invalid tools_json: {e}")))?;
    raw.into_iter()
        .map(|t| {
            let name = t
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| JsError::new("tool missing string field `name`"))?;
            let description = t.get("description").and_then(serde_json::Value::as_str).unwrap_or("");
            let schema = t.get("inputSchema").cloned().unwrap_or(serde_json::json!({}));
            Ok(Tool::from_mcp(name, description, schema))
        })
        .collect()
}

fn error_json(message: &str) -> String {
    serde_json::json!({ "outcome": "error", "message": message }).to_string()
}

impl LlmEngine {
    /// Common leading token-id run between rendering `utterance` and a
    /// throwaway probe utterance under the current `self.tools` — same
    /// method `agent.rs`'s `Agent::prefix_len_for` uses natively.
    fn compute_prefix_len(&self, utterance: &str) -> anyhow::Result<usize> {
        let template = self.template.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let render = |u: &str| -> anyhow::Result<Vec<u32>> {
            let messages = vec![Message::system(&self.system_prompt), Message::user(u)];
            let prompt = template.render_prompt(&messages, &self.tools, true)?;
            tokenizer.encode(&prompt, false)
        };
        let a = render(utterance)?;
        let b = render("\u{0}prefix-cache-probe\u{0}")?;
        Ok(a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count())
    }

    /// One render -> encode -> (restore prefix ->) prefill -> decode ->
    /// parse round. See module docs for why this can't go through
    /// `Agent::step_inner`.
    async fn run_step(&mut self) -> Result<String, JsError> {
        if self.step_index >= self.max_steps {
            return Ok(error_json(&format!(
                "agent exceeded max_steps ({}) without a final text response",
                self.max_steps
            )));
        }
        self.step_index += 1;

        let template = self.template.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;

        let prompt = template
            .render_prompt(&self.messages, &self.tools, true)
            .map_err(|e| JsError::new(&format!("failed to render prompt: {e}")))?;
        let prompt_tokens = tokenizer
            .encode(&prompt, false)
            .map_err(|e| JsError::new(&format!("failed to encode prompt: {e}")))?;

        let prefix_len = self
            .prefix_cache
            .as_ref()
            .filter(|(cached_tools, _)| cached_tools == &self.tools)
            .map(|(_, len)| (*len).min(prompt_tokens.len()))
            .unwrap_or(0);
        let effective_prefix = if prefix_len > 0
            && prefix_len <= self.resident_tokens.len()
            && self.resident_tokens[..prefix_len] == prompt_tokens[..prefix_len]
        {
            prefix_len
        } else {
            0
        };

        let model = self.model.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("model not loaded"))?;
        cache.restore(effective_prefix);
        let suffix = &prompt_tokens[effective_prefix..];
        if suffix.is_empty() {
            return Err(JsError::new("prompt fully cached with no new tokens to prefill"));
        }
        self.resident_tokens.truncate(effective_prefix);

        let prefill_start = now_ms();
        let hidden = model.forward_hidden(suffix, cache);
        self.resident_tokens.extend_from_slice(suffix);
        let last = hidden.narrow(1, suffix.len() - 1, 1);
        let mut logits = model.lm_head(last);
        let prefill_ms = now_ms() - prefill_start;

        let stop_ids = tokenizer.eos_ids();
        let mut out_ids = Vec::with_capacity(self.max_new_tokens);
        let decode_start = now_ms();
        for _ in 0..self.max_new_tokens {
            let data = logits
                .into_data_async()
                .await
                .map_err(|e| JsError::new(&format!("GPU readback failed: {e}")))?;
            let logits_vec: Vec<f32> = data.into_vec().expect("f32 logits");
            let next = greedy(&logits_vec);
            out_ids.push(next);
            if stop_ids.contains(&next) {
                break;
            }
            let hidden = model.forward_hidden(&[next], cache);
            self.resident_tokens.push(next);
            logits = model.lm_head(hidden);
        }
        let decode_ms = now_ms() - decode_start;

        let generated_text = tokenizer
            .decode(&out_ids, false)
            .map_err(|e| JsError::new(&format!("failed to decode generated tokens: {e}")))?;
        let parsed = match parse_output(&generated_text) {
            Ok(p) => p,
            Err(e) => return Ok(error_json(&format!("failed to parse model output: {e}"))),
        };

        match parsed {
            ParsedOutput::ToolCalls(calls) => {
                let pending: Vec<PendingToolCall> = calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| PendingToolCall {
                        call_id: format!("call_{i}"),
                        name: c.name.clone(),
                        arguments: c.arguments.clone(),
                    })
                    .collect();
                let entries: Vec<ToolCallEntry> = pending
                    .iter()
                    .map(|c| ToolCallEntry {
                        id: Some(c.call_id.clone()),
                        kind: "function".to_string(),
                        function: ToolCallFunction {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        },
                    })
                    .collect();
                self.messages.push(Message::assistant_tool_calls(entries));
                self.pending_calls = pending.clone();

                Ok(serde_json::json!({
                    "outcome": "needTools",
                    "calls": pending.iter().map(|c| serde_json::json!({
                        "call_id": c.call_id, "name": c.name, "arguments": c.arguments,
                    })).collect::<Vec<_>>(),
                    "step": {
                        "promptTokens": prompt_tokens.len(),
                        "text": generated_text,
                        "prefillMs": prefill_ms,
                        "decodeMs": decode_ms,
                        "tokens": out_ids.len(),
                    },
                })
                .to_string())
            }
            ParsedOutput::Text(text) => Ok(serde_json::json!({
                "outcome": "final",
                "text": text,
                "step": {
                    "promptTokens": prompt_tokens.len(),
                    "text": generated_text,
                    "prefillMs": prefill_ms,
                    "decodeMs": decode_ms,
                    "tokens": out_ids.len(),
                },
            })
            .to_string()),
        }
    }
}

/// Wall-clock milliseconds, for prefill/decode timings. `js_sys::Date::now`
/// (epoch-ms, ~1ms resolution) rather than `Performance.now()` — the latter
/// needs the `Window`/`Performance` web-sys features, which aren't in this
/// crate's (or stt-wasm's) `web-sys` feature list; `js-sys` is already a
/// dependency either way.
fn now_ms() -> f64 {
    js_sys::Date::now()
}
