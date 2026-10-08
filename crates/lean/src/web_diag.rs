//! Token-level and diagnostics calls for the browser, behind the `diag`
//! feature: `LeanEngine` (WebGPU) and `LeanEngineCpu` work on raw token
//! ids (prefill, decode steps, KV snapshots, masks) and report device and
//! timing details. Pages that only need text in, text out use `Lean`.

use wasm_bindgen::prelude::*;

use tokenizers::Tokenizer;

use crate::web::{wasm_log, JsBytesReader};
use lean_core::cpu::{forward_decode_step_argmax as cpu_decode_step_argmax, forward_prefill as cpu_forward_prefill, CpuKvCache, CpuModel};
use lean_core::engine::{now_ms, Engine};
use lean_core::model::{build_mask_bitset, rope_tables, decode_greedy_pipelined, forward_decode_step_argmax, forward_prefill, forward_prefill_suffix, GpuModel, KvCache, KvSnapshot};

/// `mask_bits` is this crate's packed bitset format (`build_mask_bitset`/
/// `buildMaskBitset`) - an empty vec means "no mask" (unmasked path, byte-
/// identical dispatch sequence to before this feature existed). wasm-bindgen
/// doesn't need `Option<Vec<u32>>` plumbing this way, and JS call sites read
/// naturally as `engine.decodeStepArgmax(id, [])`.
/// A non-empty mask shorter than `vocab_size / 32` words is rejected, as
/// `Lean::generate` does.
fn mask_buf(engine: &Engine, vocab_size: usize, mask_bits: &[u32]) -> Result<Option<wgpu::Buffer>, JsError> {
    if mask_bits.is_empty() {
        Ok(None)
    } else if mask_bits.len() * 32 < vocab_size {
        Err(JsError::new("mask_bits is shorter than vocab_size / 32"))
    } else {
        Ok(Some(engine.buf_u32(mask_bits, "mask")))
    }
}

/// Initializes the panic hook for readable browser-console error messages.
/// Optional but recommended: call once before `LeanEngine::create()`.
#[wasm_bindgen(js_name = leanInit)]
pub fn lean_init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct LeanEngine {
    engine: Engine,
    model: Option<GpuModel>,
    cache: Option<KvCache>,
    cos_buf: Option<wgpu::Buffer>,
    sin_buf: Option<wgpu::Buffer>,
    tokenizer: Option<Tokenizer>,
    /// `(GGUF parse + weight upload calls ms, tokenizer ms)`
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
    /// allocated). `tokenizer_json` is `tokenizer.json`'s text. `max_ctx` bounds the KV cache (prompt +
    /// max_new_tokens must fit). The caller should drop its own reference to
    /// `gguf_bytes`'s backing `ArrayBuffer` right after this call returns so
    /// the JS heap can reclaim it too.
    #[wasm_bindgen(js_name = load)]
    pub fn load(&mut self, gguf_bytes: js_sys::Uint8Array, tokenizer_json: String, max_ctx: u32) -> Result<(), JsError> {
        let t_start = now_ms();
        let model = GpuModel::load_from_reader(&self.engine, JsBytesReader::new(gguf_bytes), true)
            .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        let cache = KvCache::new(&self.engine, &model.config, max_ctx);
        let (cos, sin) = rope_tables(&model.config, max_ctx as usize);
        let cos_buf = self.engine.buf_f32(&cos, "rope_cos");
        let sin_buf = self.engine.buf_f32(&sin, "rope_sin");
        let t_weights = now_ms();

        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
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
        Ok(())
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

    /// Tokenizes `text` as-is (no special tokens added).
    #[wasm_bindgen(js_name = encodeRaw)]
    pub fn encode_raw(&self, text: String) -> Result<Vec<u32>, JsError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let encoding = tokenizer.encode(text, false).map_err(|e| JsError::new(&format!("tokenizer encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes `token_ids` back into text - the inverse of `encodeRaw`,
    /// exposed so a harness can print what a masked or restored
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

    /// Low-level prefill over raw token ids: resets the pool and starts a fresh
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
        Ok(())
    }

}

impl LeanEngine {
    fn from_engine(engine: Engine) -> LeanEngine {
        wasm_log(&format!("[lean] device ready, max_storage_buffer_binding_size={}", engine.max_storage_buffer_binding_size()));
        LeanEngine { engine, model: None, cache: None, cos_buf: None, sin_buf: None, tokenizer: None, load_ms: (0.0, 0.0) }
    }
}

/// The CPU backend's low-level wasm-bindgen surface (`cpu.rs`, one thread
/// or a rayon pool in the `wasm-mt` build): same method names and argument
/// shapes as `LeanEngine` wherever a CPU equivalent exists (`create`/
/// `load`/`encodeRaw`/`decodeIds`/`prefillTokens`/`decodeStepArgmax`).
/// The methods are synchronous and block the calling thread for the
/// forward pass (run them in a Web Worker). No LoRA or KV-snapshot surface
/// yet.
#[wasm_bindgen]
pub struct LeanEngineCpu {
    model: Option<CpuModel>,
    cache: Option<CpuKvCache>,
    tokenizer: Option<Tokenizer>,
}

#[wasm_bindgen]
impl LeanEngineCpu {
    /// No adapter/device to request (unlike `LeanEngine::create`) - kept as
    /// a function (not a plain struct literal) for API-shape symmetry with
    /// the GPU surface's `create()`.
    #[wasm_bindgen(js_name = create)]
    pub fn create() -> LeanEngineCpu {
        console_error_panic_hook::set_once();
        LeanEngineCpu { model: None, cache: None, tokenizer: None }
    }

    /// Parses `gguf_bytes` (a `Uint8Array` view, same reasoning as
    /// `LeanEngine::load`) into a CPU-resident model (Q4_0/Q8_0/Q6_K tensor
    /// bytes held as-is - see `cpu.rs`'s doc comment, this is the one rung
    /// that legitimately needs its own resident copy of the quantized
    /// bytes, since it computes directly off them) and allocates a
    /// `CpuKvCache` sized to `max_ctx`. Same signature as `LeanEngine::load`
    /// minus the `Result` needing to report GPU-adapter failures.
    #[wasm_bindgen(js_name = load)]
    pub fn load(&mut self, gguf_bytes: js_sys::Uint8Array, tokenizer_json: String, max_ctx: u32) -> Result<(), JsError> {
        let model = CpuModel::load_from_reader(JsBytesReader::new(gguf_bytes)).map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        let cache = CpuKvCache::new(&model.config, max_ctx as usize);
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(|e| JsError::new(&format!("failed to load tokenizer.json: {e}")))?;
        wasm_log(&format!("[lean-cpu] model loaded: layers={} hidden={} vocab={}", model.config.num_layers, model.config.hidden_size, model.config.vocab_size));
        self.model = Some(model);
        self.cache = Some(cache);
        self.tokenizer = Some(tokenizer);
        Ok(())
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
        Ok(cpu_forward_prefill(model, cache, &token_ids))
    }

    #[wasm_bindgen(js_name = decodeStepArgmax)]
    pub fn decode_step_argmax(&mut self, token_id: u32) -> Result<u32, JsError> {
        let model = self.model.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        let cache = self.cache.as_mut().ok_or_else(|| JsError::new("load() must be called first"))?;
        if cache.remaining_capacity() == 0 {
            return Err(JsError::new("decodeStepArgmax would exceed max_ctx"));
        }
        Ok(cpu_decode_step_argmax(model, cache, token_id))
    }

    #[wasm_bindgen(js_name = kvLen)]
    pub fn kv_len(&self) -> Result<u32, JsError> {
        let cache = self.cache.as_ref().ok_or_else(|| JsError::new("load() must be called first"))?;
        Ok(cache.kv_len as u32)
    }

}
