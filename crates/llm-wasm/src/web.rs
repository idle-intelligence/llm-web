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
use crate::grammar::{self, Constraint, Grammar, GrammarConstraint, IdValues, TokenVocab};
use crate::kv::KvCache;
use crate::kvimg::{self, Dtype, Header, KvImage};
use crate::model::LlmModel;
use crate::sample::{greedy, greedy_masked};
use crate::schemadiet::{diet_tools, DietLevel};
use crate::template::{ChatTemplate, Message, Tool, ToolCallEntry, ToolCallFunction};
use crate::tokenizer::Tokenizer;
use crate::tools::{format_tool_error, format_tool_result, parse_output, tool_error_message, ParsedOutput};

/// Default context length: sized (per `kv.rs`'s doc comment) for the 34-tool
/// Sonos prompt plus conversation headroom.
const DEFAULT_MAX_CTX: usize = 12288;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_MAX_STEPS: usize = 6;

/// Per-step cap on malformed-output retries (empty tool-call array,
/// unparsable `[...]` JSON, or a call naming a tool outside the current
/// `tools` set) — mirrors `agent.rs::DEFAULT_MAX_RETRIES`. See
/// `docs/ENGINE.md` "Agent loop" for the retry policy `run_step` mirrors.
const MAX_RETRIES: usize = 2;

/// A generic, non-committal nudge appended as a `user` message when a model
/// output is malformed/empty/unknown-tool twice in a row — mirrors
/// `agent.rs::RETRY_NOTE`.
const RETRY_NOTE: &str = "Respond with a tool call from the list or a final answer.";

/// Number of consecutive tool-error results `run_step` will feed back to
/// the model before giving up instead of trying again — mirrors
/// `agent.rs::MAX_CONSECUTIVE_TOOL_ERRORS`.
const MAX_CONSECUTIVE_TOOL_ERRORS: usize = 2;

/// Diet level applied to raw MCP tool lists before `Tool::from_mcp` when
/// `LlmEngine`'s diet flag is on (the default) — mirrors
/// `agent.rs::AGENT_DIET_LEVEL`.
const ENGINE_DIET_LEVEL: DietLevel = DietLevel::Level1;

/// A forced-run length below this is decoded one token at a time via
/// masked argmax instead of the batched jump-forward path — mirrors
/// `model.rs::JUMP_MIN_TOKENS` (kept as a separate constant here since
/// that one isn't `pub`; see docs/ENGINE.md "Session 12 speed addendum").
const JUMP_MIN_TOKENS: usize = 8;

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

    /// sha256 of the loaded GGUF's bytes, as computed by the caller and
    /// passed to `load()` — see `prefix_key`/`import_kv_image`/
    /// `export_kv_image` and `docs/ENGINE.md` "Prefix KV images" for why
    /// this is computed in JS (`crypto.subtle.digest`) rather than in wasm.
    model_fingerprint: Option<String>,

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
    /// Consecutive tool-error results fed back so far this turn (reset by
    /// `start`/`reset`) — see `provide_tool_results`.
    consecutive_tool_errors: usize,

    /// Whether to constrain generation to `grammar.rs`'s tool-call schema
    /// (see `docs/ENGINE.md` "Schema-constrained decoding"). On by
    /// default — set via `opts.constrained`.
    constrained: bool,
    /// Whether `start()`/the KV-image entry points run raw MCP tool lists
    /// through `schemadiet::diet_tools` before `Tool::from_mcp`. On by
    /// default — set via `opts.diet`.
    diet: bool,
    /// Id-shaped strings harvested from every tool result seen so far this
    /// conversation (see `docs/ENGINE.md` "The id rule"). Reset in
    /// `start`/`reset`.
    id_values: IdValues,
    /// Built lazily on first constrained step and cached.
    token_vocab: Option<TokenVocab>,
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
            model_fingerprint: None,
            system_prompt: "You are a helpful assistant with access to tools.".to_string(),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_steps: DEFAULT_MAX_STEPS,
            messages: Vec::new(),
            tools: Vec::new(),
            pending_calls: Vec::new(),
            step_index: 0,
            prefix_cache: None,
            consecutive_tool_errors: 0,
            constrained: true,
            diet: true,
            id_values: IdValues::new(),
            token_vocab: None,
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
        let (model_fingerprint, parts) = {
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| JsError::new(&format!("failed to open GGUF: {e}")))?;
            // Model-identity fingerprint from the header bytes already
            // resident in `loader` (magic through the tensor-info table —
            // a few KB) + file size, NOT a hash of the whole (1.7GB+) GGUF
            // — see `kvimg.rs`'s "Hashing" module docs. No `crypto.subtle`
            // call from JS and no extra pass over the shard bytes needed.
            let file_len = loader.reader().file_len();
            let header_bytes = loader
                .reader_mut()
                .header_bytes()
                .map_err(|e| JsError::new(&format!("failed to read GGUF header for fingerprint: {e}")))?;
            let model_fingerprint = kvimg::gguf_header_fingerprint(file_len, &header_bytes);

            let parts = loader
                .load_deferred(&self.device)
                .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
            // loader (and shard bytes) dropped here before GPU finalize.
            (model_fingerprint, parts)
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
        self.model_fingerprint = Some(model_fingerprint);

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
        apply_opts(self, &opts_json)?;
        self.tools = parse_tools_dieted(&tools_json, self.diet)?;
        self.messages = vec![Message::system(&self.system_prompt), Message::user(&utterance)];
        self.step_index = 0;
        self.pending_calls.clear();
        self.consecutive_tool_errors = 0;
        self.id_values = IdValues::new();

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
                if let Some(error_message) = tool_error_message(&r.result) {
                    self.consecutive_tool_errors += 1;
                    if self.consecutive_tool_errors > MAX_CONSECUTIVE_TOOL_ERRORS {
                        self.pending_calls.clear();
                        return Ok(error_json(&format!(
                            "{} consecutive tool errors (giving up after {}): {error_message}",
                            self.consecutive_tool_errors, MAX_CONSECUTIVE_TOOL_ERRORS
                        )));
                    }
                    self.messages.push(format_tool_error(&pending.name, &error_message));
                } else {
                    self.consecutive_tool_errors = 0;
                    self.id_values.collect_from_result(&r.result);
                    self.messages.push(format_tool_result(&pending.name, &r.result));
                }
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
        self.consecutive_tool_errors = 0;
        self.id_values = IdValues::new();
        if let Some(cache) = self.cache.as_mut() {
            cache.restore(0);
        }
    }

    /// Debug A/B toggle for a browser-only numerical-divergence bisection
    /// (see `gguf.rs`'s `force_naive_kernel`): `"naive"` forces the naive
    /// per-element Q4 matmul kernel for every prefill matmul regardless of
    /// M; anything else (including `"pinned"`, the default) restores
    /// production `ForceKernel::Auto` routing. Not used by any production
    /// code path.
    #[wasm_bindgen(js_name = setPrefillKernel)]
    pub fn set_prefill_kernel(&self, kernel: String) {
        crate::gguf::set_force_naive_kernel(kernel == "naive");
    }

    /// Prefix-KV-image cache key for `tools_json`/`system` under the
    /// currently loaded model (`docs/ENGINE.md` "Prefix KV images"):
    /// `sha256(model_fingerprint || rendered_prefix_text)`, computed the same way
    /// `bin/llm-agent.rs`'s `kv-export` subcommand computes it when writing
    /// an image, so a worker can `fetch(<modelBase>/kv/<key>.kvimg)` before
    /// its first prefill of a given tool set. Errors if the model isn't
    /// loaded yet (no `model_fingerprint`/tokenizer/template) or `tools_json` is
    /// malformed.
    #[wasm_bindgen(js_name = prefixKey)]
    pub fn prefix_key(&self, tools_json: String, system: String) -> Result<String, JsError> {
        let tools = parse_tools_dieted(&tools_json, self.diet)?;
        let model_fingerprint = self
            .model_fingerprint
            .as_ref()
            .ok_or_else(|| JsError::new("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let prefix_tokens = self
            .render_kv_prefix_tokens(&tools, &system)
            .map_err(|e| JsError::new(&format!("failed to render prefix: {e}")))?;
        let prefix_text = tokenizer
            .decode(&prefix_tokens, false)
            .map_err(|e| JsError::new(&format!("failed to decode prefix tokens: {e}")))?;
        Ok(kvimg::prefix_key(model_fingerprint, &prefix_text))
    }

    /// Import a prefix KV image (`bytes`, a full `.kvimg` file as fetched
    /// from `<modelBase>/kv/<prefix_key>.kvimg` or OPFS) in place of
    /// running prefill for `tools_json`/`system`'s system+tools preamble.
    /// Validates `header.model_fingerprint`, `header.prefix_key`, and
    /// `header.tokens` against what this engine/model/tools/system would
    /// actually render (same match discipline `run_step`'s
    /// `effective_prefix` check already applies to `resident_tokens`) —
    /// returns `Ok(false)` on any mismatch (caller falls back to normal
    /// prefill) rather than importing a wrong prefix. Synchronous:
    /// `KvCache::import_prefix` only writes (`from_data`/`slice_assign`),
    /// no GPU readback, so no async/await is needed on this path.
    #[wasm_bindgen(js_name = importKvImage)]
    pub fn import_kv_image(&mut self, bytes: &[u8], tools_json: String, system: String) -> Result<bool, JsError> {
        let tools = parse_tools_dieted(&tools_json, self.diet)?;
        let model_fingerprint = self
            .model_fingerprint
            .as_ref()
            .ok_or_else(|| JsError::new("model not loaded"))?
            .clone();
        let model = self.model.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;

        let (header, data_offset) =
            KvImage::read_header(bytes).map_err(|e| JsError::new(&format!("bad kv image: {e}")))?;
        if header.model_fingerprint != model_fingerprint {
            wasm_log("[llm] kv image model_fingerprint mismatch — ignoring");
            return Ok(false);
        }

        let expected_tokens = self
            .render_kv_prefix_tokens(&tools, &system)
            .map_err(|e| JsError::new(&format!("failed to render prefix: {e}")))?;
        if header.tokens != expected_tokens {
            wasm_log("[llm] kv image tokens mismatch — ignoring");
            return Ok(false);
        }
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let prefix_text = tokenizer
            .decode(&expected_tokens, false)
            .map_err(|e| JsError::new(&format!("failed to decode prefix tokens: {e}")))?;
        let expected_key = kvimg::prefix_key(&model_fingerprint, &prefix_text);
        if header.prefix_key != expected_key {
            wasm_log("[llm] kv image prefix_key mismatch — ignoring");
            return Ok(false);
        }

        let cfg = model.config();
        let expected_head_dim = cfg.hidden_size / cfg.num_heads;
        if header.n_layers != cfg.num_layers || header.n_kv_heads != cfg.num_kv_heads || header.head_dim != expected_head_dim
        {
            wasm_log("[llm] kv image shape mismatch — ignoring");
            return Ok(false);
        }

        let mut layers = Vec::with_capacity(header.n_layers);
        for layer in 0..header.n_layers {
            let (k, v) = KvImage::layer_f32(bytes, &header, data_offset, layer)
                .map_err(|e| JsError::new(&format!("failed to decode kv image layer {layer}: {e}")))?;
            layers.push((k, v));
        }

        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("model not loaded"))?;
        cache.import_prefix(&layers, header.tokens.len());
        self.resident_tokens = header.tokens.clone();
        wasm_log(&format!(
            "[llm] imported kv image: {} tokens, {} bytes",
            header.tokens.len(),
            bytes.len()
        ));
        Ok(true)
    }

    /// Export the current cache's system+tools prefix (for `tools_json`/
    /// `system`) as a `.kvimg` byte buffer — the counterpart to
    /// `import_kv_image`, called on a cache *miss* after the first prefill
    /// of a tool set so the worker can save the image to OPFS for next
    /// time. Errors (rather than exporting garbage) if `resident_tokens`
    /// doesn't currently cover the rendered prefix — call this only after
    /// a `start()`/step whose prefill included the full system+tools
    /// preamble. `dtype` is always `q8_0` (the format `docs/ENGINE.md`
    /// recommends for a one-time browser download — see kvimg.rs module
    /// docs).
    #[wasm_bindgen(js_name = exportKvImage)]
    pub async fn export_kv_image(&self, tools_json: String, system: String) -> Result<Vec<u8>, JsError> {
        let tools = parse_tools_dieted(&tools_json, self.diet)?;
        let model_fingerprint = self
            .model_fingerprint
            .as_ref()
            .ok_or_else(|| JsError::new("model not loaded"))?
            .clone();
        let model = self.model.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;

        let prefix_tokens = self
            .render_kv_prefix_tokens(&tools, &system)
            .map_err(|e| JsError::new(&format!("failed to render prefix: {e}")))?;
        let n_tokens = prefix_tokens.len();
        if n_tokens == 0
            || n_tokens > self.resident_tokens.len()
            || self.resident_tokens[..n_tokens] != prefix_tokens[..]
        {
            return Err(JsError::new(
                "current resident tokens do not cover the rendered system+tools prefix — \
                 call exportKvImage only right after a prefill that included it",
            ));
        }

        let prefix_text = tokenizer
            .decode(&prefix_tokens, false)
            .map_err(|e| JsError::new(&format!("failed to decode prefix tokens: {e}")))?;
        let prefix_key = kvimg::prefix_key(&model_fingerprint, &prefix_text);

        let layers = cache.export_prefix_async(n_tokens).await;
        let cfg = model.config();
        let header = Header {
            model_fingerprint,
            prefix_key,
            tokens: prefix_tokens,
            n_layers: cfg.num_layers,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
            dtype: Dtype::Q8_0.as_str().to_string(),
            engine: format!("llm-wasm/{}", env!("CARGO_PKG_VERSION")),
            created: js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default(),
        };

        let mut buf = Vec::new();
        let layer_refs: Vec<(&[f32], &[f32])> = layers.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
        KvImage::write(&mut buf, &header, Dtype::Q8_0, layer_refs)
            .map_err(|e| JsError::new(&format!("failed to write kv image: {e}")))?;
        Ok(buf)
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
    /// Schema-constrained decoding (`docs/ENGINE.md` "Schema-constrained
    /// decoding"). Defaults to on (`LlmEngine::new`'s `constrained: true`)
    /// when omitted.
    constrained: Option<bool>,
    /// Tool-schema token diet applied before `Tool::from_mcp` (`docs/
    /// ENGINE.md` "Tool-schema token diet"). Defaults to on when omitted.
    diet: Option<bool>,
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
    if let Some(b) = opts.constrained {
        engine.constrained = b;
    }
    if let Some(b) = opts.diet {
        engine.diet = b;
    }
    Ok(())
}

/// Build `Tool`s from raw MCP `tools/list` entries, optionally running them
/// through `schemadiet::diet_tools` first — mirrors
/// `agent.rs::tools_from_mcp`. Entries missing `name`/`description`/
/// `inputSchema` are skipped rather than erroring.
fn parse_tools_dieted(tools_json: &str, diet: bool) -> Result<Vec<Tool>, JsError> {
    let raw: Vec<serde_json::Value> =
        serde_json::from_str(tools_json).map_err(|e| JsError::new(&format!("invalid tools_json: {e}")))?;
    let dieted;
    let raw = if diet {
        dieted = diet_tools(&raw, ENGINE_DIET_LEVEL);
        &dieted
    } else {
        &raw
    };
    raw.iter()
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

/// One [`LlmEngine::generate_attempt`] round's output, before `run_step`
/// decides whether it's valid or needs a retry — mirrors
/// `agent.rs::AttemptOutput`. `parsed` is `Err` for a parse failure
/// (retryable), not a hard error.
struct AttemptOutput {
    prompt_tokens: Vec<u32>,
    generated_text: String,
    parsed: Result<ParsedOutput, String>,
    prefill_ms: f64,
    decode_ms: f64,
    tokens_generated: usize,
    model_steps: usize,
    forced_tokens: usize,
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

    /// The system+tools prefix `bin/llm-agent.rs`'s `kv-export` subcommand
    /// exports images for: common leading token-id run between two
    /// content-free probe utterances rendered under `tools`/`system` (not
    /// `self.tools`/`self.system_prompt` — a caller checking a prefix
    /// image before `start()` doesn't have those set yet). **Must** use the
    /// exact same two probe strings `kv-export` uses, or the two sides
    /// compute different `prefix_key`s for the same actual prefix.
    fn render_kv_prefix_tokens(&self, tools: &[Tool], system: &str) -> anyhow::Result<Vec<u32>> {
        let template = self.template.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let render = |u: &str| -> anyhow::Result<Vec<u32>> {
            let messages = vec![Message::system(system), Message::user(u)];
            let prompt = template.render_prompt(&messages, tools, true)?;
            tokenizer.encode(&prompt, false)
        };
        let a = render("kv-export-probe-alpha")?;
        let b = render("totally-different-probe-beta")?;
        let prefix_len = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
        Ok(a[..prefix_len].to_vec())
    }

    /// One render -> encode -> (restore prefix ->) prefill -> decode ->
    /// parse round. See module docs for why this can't go through
    /// `Agent::step_inner`.
    ///
    /// Wraps [`LlmEngine::generate_attempt`] with `agent.rs::step_inner`'s
    /// malformed-output retry policy (see `docs/ENGINE.md` "Agent loop" —
    /// "what web.rs must mirror" (1)): an empty tool-call array, unparsable
    /// `[...]` JSON, or a call naming a tool outside `self.tools` does not
    /// become an assistant turn — instead attempt 1 forces
    /// schema-constrained decoding on (if not already), attempt 2 appends
    /// `RETRY_NOTE` as a `user` message, and exhausting `MAX_RETRIES` gives
    /// up with an `"outcome":"error"` payload. Retries don't consume
    /// `step_index`'s budget (already charged above, once per `run_step`
    /// call).
    async fn run_step(&mut self) -> Result<String, JsError> {
        if self.step_index >= self.max_steps {
            return Ok(error_json(&format!(
                "agent exceeded max_steps ({}) without a final text response",
                self.max_steps
            )));
        }
        self.step_index += 1;

        let mut retries = 0usize;
        loop {
            let force_constrained = retries == 1 && !self.constrained && !self.tools.is_empty();
            let attempt = self.generate_attempt(force_constrained).await?;

            let invalid = match &attempt.parsed {
                Err(_) => true,
                Ok(ParsedOutput::ToolCalls(calls)) => {
                    calls.is_empty()
                        || calls
                            .iter()
                            .any(|c| !self.tools.iter().any(|t| t.function.name == c.name))
                }
                Ok(ParsedOutput::Text(_)) => false,
            };

            if invalid {
                if retries < MAX_RETRIES {
                    retries += 1;
                    if retries == 2 {
                        self.messages.push(Message::user(RETRY_NOTE));
                    }
                    continue;
                }
                return Ok(error_json("model produced no valid call"));
            }

            let AttemptOutput {
                prompt_tokens,
                generated_text,
                parsed,
                prefill_ms,
                decode_ms,
                tokens_generated,
                model_steps,
                forced_tokens,
            } = attempt;
            let parsed = parsed.expect("checked valid above");
            let tool_errors = self.consecutive_tool_errors;

            return match parsed {
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
                            "promptTokenIds": prompt_tokens,
                            "text": generated_text,
                            "prefillMs": prefill_ms,
                            "decodeMs": decode_ms,
                            "tokens": tokens_generated,
                            "modelSteps": model_steps,
                            "forcedTokens": forced_tokens,
                            "retries": retries,
                            "toolErrors": tool_errors,
                        },
                    })
                    .to_string())
                }
                ParsedOutput::Text(text) => Ok(serde_json::json!({
                    "outcome": "final",
                    "text": text,
                    "step": {
                        "promptTokens": prompt_tokens.len(),
                        "promptTokenIds": prompt_tokens,
                        "text": generated_text,
                        "prefillMs": prefill_ms,
                        "decodeMs": decode_ms,
                        "tokens": tokens_generated,
                        "modelSteps": model_steps,
                        "forcedTokens": forced_tokens,
                        "retries": retries,
                        "toolErrors": tool_errors,
                    },
                })
                .to_string()),
            };
        }
    }

    /// One render -> encode -> (restore prefix ->) prefill -> constrained
    /// decode -> parse round — the async, WebGPU-backed shape of
    /// `agent.rs::Agent::generate_attempt` (see `docs/ENGINE.md`
    /// "Schema-constrained decoding", "web.rs's job"): builds a
    /// `Grammar::for_tools` from `self.tools` + the running `self.id_values`
    /// when constrained decoding is on (`self.constrained` or
    /// `force_constrained`), drives it through the decode loop with the
    /// same jump-forward semantics as `model.rs::LlmModel::decode_with_constraint`
    /// (batched `forward_hidden` for forced runs of at least
    /// `JUMP_MIN_TOKENS`, masked-argmax otherwise), and reports the
    /// model-step/forced-token breakdown. A parse failure is returned as
    /// `Ok(AttemptOutput{parsed: Err(_), ..})`, not `Err`, so `run_step`'s
    /// retry loop can decide whether to retry — only render/encode/
    /// GPU/model failures propagate as `Err`.
    async fn generate_attempt(&mut self, force_constrained: bool) -> Result<AttemptOutput, JsError> {
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

        // Build this step's schema constraint (if `self.constrained`, or
        // this attempt forces it on) from the current tool set + every id
        // harvested from tool results so far — mirrors
        // `agent.rs::Agent::generate_attempt`.
        let use_constrained = self.constrained || force_constrained;
        let grammar_tools: Vec<grammar::Tool> = if use_constrained {
            self.tools
                .iter()
                .map(|t| grammar::Tool::from_schema(&t.function.name, &t.function.parameters))
                .collect()
        } else {
            Vec::new()
        };
        let grammar_for_step = use_constrained.then(|| Grammar::for_tools(&grammar_tools, &self.id_values));
        if use_constrained && self.token_vocab.is_none() {
            self.token_vocab = Some(TokenVocab::from_tokenizer(tokenizer));
        }
        let mut constraint_impl: Option<GrammarConstraint> = grammar_for_step
            .as_ref()
            .map(|g| GrammarConstraint::new(g, tokenizer, self.token_vocab.as_ref().expect("built above")));

        let model = self.model.as_ref().ok_or_else(|| JsError::new("model not loaded"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("model not loaded"))?;
        cache.restore(effective_prefix);
        let suffix = &prompt_tokens[effective_prefix..];
        if suffix.is_empty() {
            return Err(JsError::new("prompt fully cached with no new tokens to prefill"));
        }
        self.resident_tokens.truncate(effective_prefix);

        let prefill_start = now_ms();
        let hidden = model
            .forward_hidden(suffix, cache)
            .map_err(|e| JsError::new(&format!("forward pass failed: {e}")))?;
        self.resident_tokens.extend_from_slice(suffix);
        let last = hidden.narrow(1, suffix.len() - 1, 1);
        let logits = model.lm_head(last);
        // P2 (docs/BENCHMARKS.md Session 4): force GPU completion here via
        // async readback, needed anyway for the first decode token, so
        // `prefill_ms` measures actual GPU completion instead of just
        // kernel-submission time (previously the sync point was the first
        // iteration of the decode loop below, silently folding prefill's
        // real GPU time into `decode_ms`).
        let data = logits
            .into_data_async()
            .await
            .map_err(|e| JsError::new(&format!("GPU readback failed: {e}")))?;
        let mut logits_vec: Vec<f32> = data
            .into_vec()
            .map_err(|e| JsError::new(&format!("failed to read back f32 logits: {e:?}")))?;
        let prefill_ms = now_ms() - prefill_start;

        // Debug aid for a browser-only numerical-divergence bisection
        // (compare against `llm-agent run`'s greedy output on the same
        // token ids natively) — top-5 (id, logit) at the first decode
        // position of this step's prefill. Cheap (one sort over vocab_size
        // once per step), left on unconditionally since it's diagnostic
        // output, not a hot loop.
        {
            let mut top: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = top.iter().take(5).map(|(id, v)| format!("({id},{v:.4})")).collect();
            wasm_log(&format!(
                "[llm] step {} prefill top5 logits: [{}]",
                self.step_index - 1,
                top5.join(", ")
            ));
        }

        let stop_ids = tokenizer.eos_ids();
        let mut out_ids = Vec::with_capacity(self.max_new_tokens);
        let mut model_steps = 0usize;
        let mut forced_tokens = 0usize;
        let decode_start = now_ms();
        while out_ids.len() < self.max_new_tokens {
            let forced = constraint_impl
                .as_ref()
                .and_then(Constraint::forced_run)
                .unwrap_or_default();

            if forced.len() >= JUMP_MIN_TOKENS {
                let take = forced.len().min(self.max_new_tokens - out_ids.len());
                let run = forced[..take].to_vec();
                let hidden = model
                    .forward_hidden(&run, cache)
                    .map_err(|e| JsError::new(&format!("forward pass failed: {e}")))?;
                for &t in &run {
                    if let Some(c) = constraint_impl.as_mut() {
                        c.advance(t);
                    }
                }
                self.resident_tokens.extend_from_slice(&run);
                out_ids.extend_from_slice(&run);
                model_steps += 1;
                forced_tokens += run.len();
                if take < forced.len() {
                    // Hit max_new_tokens mid-run: stop without sampling further.
                    break;
                }
                let last = hidden.narrow(1, run.len() - 1, 1);
                let logits = model.lm_head(last);
                let data = logits
                    .into_data_async()
                    .await
                    .map_err(|e| JsError::new(&format!("GPU readback failed: {e}")))?;
                logits_vec = data
                    .into_vec()
                    .map_err(|e| JsError::new(&format!("failed to read back f32 logits: {e:?}")))?;
                continue;
            }

            let next = match constraint_impl.as_ref().and_then(Constraint::allowed) {
                Some(mask) => greedy_masked(&logits_vec, mask),
                None => greedy(&logits_vec),
            };
            out_ids.push(next);
            model_steps += 1;
            if let Some(c) = constraint_impl.as_mut() {
                c.advance(next);
            }
            if stop_ids.contains(&next) {
                break;
            }
            let hidden = model
                .forward_hidden(&[next], cache)
                .map_err(|e| JsError::new(&format!("forward pass failed: {e}")))?;
            self.resident_tokens.push(next);
            let logits = model.lm_head(hidden);
            let data = logits
                .into_data_async()
                .await
                .map_err(|e| JsError::new(&format!("GPU readback failed: {e}")))?;
            logits_vec = data
                .into_vec()
                .map_err(|e| JsError::new(&format!("failed to read back f32 logits: {e:?}")))?;
        }
        let decode_ms = now_ms() - decode_start;

        let generated_text = tokenizer
            .decode(&out_ids, false)
            .map_err(|e| JsError::new(&format!("failed to decode generated tokens: {e}")))?;
        let parsed = parse_output(&generated_text).map_err(|e| e.to_string());

        Ok(AttemptOutput {
            prompt_tokens,
            generated_text,
            parsed,
            prefill_ms,
            decode_ms,
            tokens_generated: out_ids.len(),
            model_steps,
            forced_tokens,
        })
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
