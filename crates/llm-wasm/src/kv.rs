//! KV cache for autoregressive decoding + prefill.
//!
//! ## Storage mode (Session 13, docs/BENCHMARKS.md)
//!
//! [`KvDtype::F32`]: one `[1, n_kv_heads, max_ctx, head_dim]` f32 tensor per
//! layer, **sequence-major** (contiguous along the time axis) so a prefill
//! of `T` tokens writes one contiguous `slice_assign` per layer instead of
//! `T` separate single-token writes. Cost at the default 12288-token
//! `max_ctx`: `num_layers(36) * 2(k+v) * n_kv_heads(2) * max_ctx(12288) *
//! head_dim(128) * 4 bytes ≈ 906 MB`.
//!
//! [`KvDtype::Q8_0`] (**not yet the default** — see [`DEFAULT_KV_DTYPE`] and
//! the correctness note below): K/V held as raw cubecl GPU buffers (not
//! Burn tensors), one absmax/127 f32 scale + 32 packed-i8 values
//! (4-per-u32 little-endian) per 32-element block along `head_dim` — same
//! on-disk convention as `kvimg.rs`'s `dtype: "q8_0"` image format (see
//! that module's doc comment), just resident on GPU instead of
//! round-tripped through a file. Per layer: `scales: [n_kv_heads, max_ctx,
//! head_dim/32]` f32, `words: [n_kv_heads, max_ctx, head_dim/4]` u32 — 36
//! bytes per 32 values (1.125 B/value) vs f32's 4 B/value, ~3.55x smaller:
//! the 906MB f32 cache above drops to ~255MB. `write` quantizes new rows
//! on the GPU (`gguf::kv_quantize_dispatch`, `wgsl/shader_kv_quantize.wgsl`)
//! — no CPU round trip. Decode-time (M=1) attention reads these buffers
//! directly (`model.rs`'s `Q4Attention::forward`, `gguf::attn_decode_q8_dispatch`,
//! `wgsl/shader_attn_decode_q8.wgsl`) with **no dequant-to-f32 step**;
//! prefill (M>1) still dequantizes the needed range into an f32 scratch
//! tensor once per layer per forward (`read_or_dequant_f32`,
//! `gguf::kv_dequant_range_dispatch`, `wgsl/shader_kv_dequant_range.wgsl`)
//! and runs the existing Burn-matmul attention path — correctness first,
//! a fused prefill kernel is a later step. Attention always accumulates in
//! f32 regardless of KV storage dtype.
//!
//! **Correctness status (Session 13, docs/BENCHMARKS.md)**: every kernel
//! is unit-tested correct in isolation — q8_0 quantize/dequant round trip
//! within its documented `absmax/127` per-block bound (`tests/kvimg.rs`'s
//! `q8_append_roundtrip_error_bound`), a bulk `T=31` write dequantizes
//! bit-identically to 31 separate `T=1` writes
//! (`q8_bulk_write_matches_row_by_row`), and the fused decode kernel
//! matches an independent f64 CPU reference to <=2.5e-4 relative error at
//! `kv_len` in `{31, 2225, 5000}` (`q8_decode_attention_matches_f64_reference`).
//! But running the *real* model end-to-end with `KvDtype::Q8_0` as the
//! cache (`tests/full_forward.rs`) diverges badly — every one of 31
//! prefill positions' argmax disagreed with the F32 reference (vs. 8/31
//! under `KvDtype::F32`, the pre-existing baseline), and the last-position
//! top-5 had zero overlap with the reference on the 2225/2354-token
//! fixtures. The per-block quantization error is small and bounded
//! (~2% mean relative, measured on synthetic K/V-shaped data), but applied
//! to *every* K/V value at *every* layer from token 0 onward, it compounds
//! across 36 residual layers into much larger output divergence than the
//! isolated-kernel tests suggested — greedy decoding amplifies this further
//! since it only takes one flipped top-1 logit to derail the rest of the
//! sequence. Root cause not yet isolated to K vs V specifically (the task
//! brief's suggested next diagnostic — quantize K only, keep V in f32 — is
//! not yet implemented; see `docs/BENCHMARKS.md` Session 13's "what's
//! left"). Until that's resolved, [`DEFAULT_KV_DTYPE`] stays `F32` and
//! `Q8_0` is opt-in via [`KvCache::new_with_dtype`] for continued
//! development/benchmarking, not production use.
//!
//! `f16` would also roughly halve the f32 cache, but Burn's wgpu backend
//! has no native f16 compute path here and adding one would mean depending
//! on WGSL's `shader-f16` extension (gaps on Firefox/Linux/NVIDIA and
//! Qualcomm — the same reason `kvimg.rs`'s image format chose q8_0 over
//! f16). int8 with a portable WGSL kernel avoids that dependency entirely.
//!
//! `snapshot()`/`restore()` are O(1) in both modes: they only save/restore
//! the fill length `len`, not any tensor/buffer data. This lets a constant
//! prefix (system prompt + tool schemas) be prefilled once, `snapshot()`-ed,
//! and then `restore()`-ed before each new user turn — new tokens simply
//! overwrite whatever was previously written past the restored length; no
//! data is copied or cleared.

use burn::backend::wgpu::{into_contiguous, CubeTensor, WgpuDevice, WgpuRuntime};
use burn::backend::Wgpu;
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use cubecl::server::Handle;
use cubecl::Runtime;

/// KV cache storage dtype. See this module's doc comment for the layout of
/// each mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    F32,
    Q8_0,
}

/// Production default — still `F32`. `Q8_0` is fully implemented and
/// kernel-correct (see this module's doc comment) but not yet safe as the
/// default: `tests/full_forward.rs` shows real end-to-end model divergence
/// under `Q8_0` that the isolated kernel tests didn't predict. Use
/// [`KvCache::new_with_dtype`] to opt into `Q8_0` for continued
/// investigation/benchmarking — see `docs/BENCHMARKS.md` Session 13.
pub const DEFAULT_KV_DTYPE: KvDtype = KvDtype::F32;

/// Per-layer K/V ring of fixed capacity `max_ctx`, append-only until reset.
pub struct KvCache {
    dtype: KvDtype,
    // KvDtype::F32 storage.
    k_f32: Vec<Tensor<Wgpu, 4>>,
    v_f32: Vec<Tensor<Wgpu, 4>>,
    // KvDtype::Q8_0 storage: raw cubecl buffers, not Burn tensors (see
    // module doc comment for the per-layer layout).
    k_scales: Vec<Handle>,
    k_words: Vec<Handle>,
    v_scales: Vec<Handle>,
    v_words: Vec<Handle>,
    len: usize,
    max_ctx: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: WgpuDevice,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        device: &WgpuDevice,
    ) -> Self {
        Self::new_with_dtype(num_layers, n_kv_heads, head_dim, max_ctx, device, DEFAULT_KV_DTYPE)
    }

    pub fn new_with_dtype(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        device: &WgpuDevice,
        dtype: KvDtype,
    ) -> Self {
        match dtype {
            KvDtype::F32 => {
                let shape = [1, n_kv_heads, max_ctx, head_dim];
                let k_f32 = (0..num_layers)
                    .map(|_| Tensor::<Wgpu, 4>::zeros(shape, device))
                    .collect();
                let v_f32 = (0..num_layers)
                    .map(|_| Tensor::<Wgpu, 4>::zeros(shape, device))
                    .collect();
                Self {
                    dtype,
                    k_f32,
                    v_f32,
                    k_scales: Vec::new(),
                    k_words: Vec::new(),
                    v_scales: Vec::new(),
                    v_words: Vec::new(),
                    len: 0,
                    max_ctx,
                    n_kv_heads,
                    head_dim,
                    device: device.clone(),
                }
            }
            KvDtype::Q8_0 => {
                assert_eq!(
                    head_dim % 32,
                    0,
                    "Q8_0 KV cache requires head_dim % 32 == 0, got {head_dim}"
                );
                let client = WgpuRuntime::client(device);
                let blocks_per_row = head_dim / 32;
                let scales_bytes = n_kv_heads * max_ctx * blocks_per_row * 4;
                let words_bytes = n_kv_heads * max_ctx * blocks_per_row * 8 * 4;
                let k_scales = (0..num_layers).map(|_| client.empty(scales_bytes)).collect();
                let k_words = (0..num_layers).map(|_| client.empty(words_bytes)).collect();
                let v_scales = (0..num_layers).map(|_| client.empty(scales_bytes)).collect();
                let v_words = (0..num_layers).map(|_| client.empty(words_bytes)).collect();
                Self {
                    dtype,
                    k_f32: Vec::new(),
                    v_f32: Vec::new(),
                    k_scales,
                    k_words,
                    v_scales,
                    v_words,
                    len: 0,
                    max_ctx,
                    n_kv_heads,
                    head_dim,
                    device: device.clone(),
                }
            }
        }
    }

    pub fn dtype(&self) -> KvDtype {
        self.dtype
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Current fill length (== next-write offset, == absolute position of
    /// the next token for RoPE).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn client(&self) -> cubecl::client::ComputeClient<WgpuRuntime> {
        WgpuRuntime::client(&self.device)
    }

    /// Extract a contiguous `CubeTensor` (client + handle) from a `[1,
    /// n_kv_heads, T, head_dim]` Burn tensor, for handing to a raw WGSL
    /// kernel dispatch.
    fn into_cube(t: Tensor<Wgpu, 4>) -> CubeTensor<WgpuRuntime> {
        into_contiguous(t.into_primitive().tensor())
    }

    fn cube_to_tensor(
        client: cubecl::client::ComputeClient<WgpuRuntime>,
        device: WgpuDevice,
        shape: [usize; 4],
        handle: Handle,
    ) -> Tensor<Wgpu, 4> {
        let cube_tensor = CubeTensor::new_contiguous(
            client,
            device,
            burn::prelude::Shape::from(shape.to_vec()),
            handle,
            DType::F32,
        );
        Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
    }

    /// Write `k`/`v` (`[1, n_kv_heads, T, head_dim]`) for `layer` at
    /// `[len, len+T)`. Does **not** advance `len` — call [`Self::advance`]
    /// once per forward step (all layers in a step share the same write
    /// range). Use [`Self::read_or_dequant_f32`] (or, in `KvDtype::Q8_0`
    /// mode, [`Self::q8_layer`]) to read the cache back afterward.
    pub fn write(&mut self, layer: usize, k: Tensor<Wgpu, 4>, v: Tensor<Wgpu, 4>) {
        let t = k.dims()[2];
        assert!(
            self.len + t <= self.max_ctx,
            "KV cache overflow: len={} + t={} > max_ctx={}",
            self.len,
            t,
            self.max_ctx
        );
        match self.dtype {
            KvDtype::F32 => {
                let ranges = [
                    0..1usize,
                    0..self.n_kv_heads,
                    self.len..self.len + t,
                    0..self.head_dim,
                ];
                // D1 (docs/BENCHMARKS.md Session 2 "next most valuable"):
                // `mem::replace`-ing the slot with a tiny placeholder
                // before `slice_assign` drops the second reference to the
                // cache's own tensor so cubecl mutates the existing
                // buffer in place instead of copying the whole
                // `max_ctx`-sized tensor — see the original version of
                // this comment (git history) for the full story.
                let placeholder_shape = [1, self.n_kv_heads, 1, self.head_dim];
                let old_k = std::mem::replace(
                    &mut self.k_f32[layer],
                    Tensor::<Wgpu, 4>::empty(placeholder_shape, &k.device()),
                );
                let old_v = std::mem::replace(
                    &mut self.v_f32[layer],
                    Tensor::<Wgpu, 4>::empty(placeholder_shape, &v.device()),
                );
                self.k_f32[layer] = old_k.slice_assign(ranges.clone(), k);
                self.v_f32[layer] = old_v.slice_assign(ranges, v);
            }
            KvDtype::Q8_0 => {
                let client = self.client();
                let k_cube = Self::into_cube(k);
                let v_cube = Self::into_cube(v);
                crate::gguf::kv_quantize_dispatch(
                    &client,
                    k_cube.handle,
                    &self.k_scales[layer],
                    &self.k_words[layer],
                    self.n_kv_heads,
                    t,
                    self.head_dim,
                    self.max_ctx,
                    self.len,
                );
                crate::gguf::kv_quantize_dispatch(
                    &client,
                    v_cube.handle,
                    &self.v_scales[layer],
                    &self.v_words[layer],
                    self.n_kv_heads,
                    t,
                    self.head_dim,
                    self.max_ctx,
                    self.len,
                );
            }
        }
    }

    /// Read back the valid prefix `[1, n_kv_heads, kv_len, head_dim]` of
    /// `layer`'s K/V as f32 tensors. In `KvDtype::F32` mode this is a plain
    /// narrow (no copy of the underlying data beyond what `narrow` already
    /// does); in `KvDtype::Q8_0` mode this dequantizes the whole
    /// `[0, kv_len)` range on the GPU into a fresh f32 tensor (the prefill
    /// fallback path — see this module's doc comment).
    pub fn read_or_dequant_f32(&self, layer: usize, kv_len: usize) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        match self.dtype {
            KvDtype::F32 => {
                let k_all = self.k_f32[layer].clone().narrow(2, 0, kv_len);
                let v_all = self.v_f32[layer].clone().narrow(2, 0, kv_len);
                (k_all, v_all)
            }
            KvDtype::Q8_0 => {
                let client = self.client();
                let k_handle = crate::gguf::kv_dequant_range_dispatch(
                    &client,
                    &self.k_scales[layer],
                    &self.k_words[layer],
                    self.n_kv_heads,
                    kv_len,
                    self.head_dim,
                    self.max_ctx,
                );
                let v_handle = crate::gguf::kv_dequant_range_dispatch(
                    &client,
                    &self.v_scales[layer],
                    &self.v_words[layer],
                    self.n_kv_heads,
                    kv_len,
                    self.head_dim,
                    self.max_ctx,
                );
                let shape = [1, self.n_kv_heads, kv_len, self.head_dim];
                let k_all = Self::cube_to_tensor(client.clone(), self.device.clone(), shape, k_handle);
                let v_all = Self::cube_to_tensor(client, self.device.clone(), shape, v_handle);
                (k_all, v_all)
            }
        }
    }

    /// `KvDtype::Q8_0`-only: raw handles for `layer`'s q8_0 buffers
    /// (`k_scales, k_words, v_scales, v_words`), for the decode-time fused
    /// attention kernel (`gguf::attn_decode_q8_dispatch`) to bind directly.
    pub fn q8_layer(&self, layer: usize) -> (&Handle, &Handle, &Handle, &Handle) {
        assert_eq!(self.dtype, KvDtype::Q8_0, "q8_layer requires KvDtype::Q8_0");
        (
            &self.k_scales[layer],
            &self.k_words[layer],
            &self.v_scales[layer],
            &self.v_words[layer],
        )
    }

    /// Advance the fill length by `t` — call once per forward step, after
    /// every layer's `write` for that step.
    pub fn advance(&mut self, t: usize) {
        self.len += t;
    }

    /// O(1): save the current fill length.
    pub fn snapshot(&self) -> usize {
        self.len
    }

    /// O(1): restore a previously `snapshot()`-ed fill length. Rows past
    /// the restored length are left in place (garbage from a longer prior
    /// turn) but are never read, since `write`/reads always operate on
    /// `[0, len)`.
    pub fn restore(&mut self, snapshot: usize) {
        assert!(snapshot <= self.max_ctx);
        self.len = snapshot;
    }

    /// Read back positions `[0, n_tokens)` of every layer's K/V from the
    /// GPU as flat `Vec<f32>` in `[n_kv_heads, n_tokens, head_dim]`
    /// row-major order (the batch=1 axis is dropped) — the layout
    /// `kvimg::KvImage::write` expects per layer. One synchronous readback
    /// per layer per tensor. **Native-only usage**: a caller running
    /// inside WASM must not use this (deadlocks the browser) and should add
    /// an `into_data_async` variant before calling this from `web.rs`.
    pub fn export_prefix(&self, n_tokens: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        assert!(n_tokens <= self.max_ctx);
        match self.dtype {
            KvDtype::F32 => self
                .k_f32
                .iter()
                .zip(self.v_f32.iter())
                .map(|(k, v)| {
                    let k_data = k.clone().narrow(2, 0, n_tokens).into_data();
                    let v_data = v.clone().narrow(2, 0, n_tokens).into_data();
                    (
                        k_data.into_vec::<f32>().expect("K tensor is f32"),
                        v_data.into_vec::<f32>().expect("V tensor is f32"),
                    )
                })
                .collect(),
            KvDtype::Q8_0 => (0..self.k_scales.len())
                .map(|layer| {
                    let (k_all, v_all) = self.read_or_dequant_f32(layer, n_tokens);
                    (
                        k_all.into_data().into_vec::<f32>().expect("dequant output is f32"),
                        v_all.into_data().into_vec::<f32>().expect("dequant output is f32"),
                    )
                })
                .collect(),
        }
    }

    /// `KvDtype::Q8_0`-only counterpart to [`Self::export_prefix`] that
    /// exports the raw quantized bytes with **no dequantization** — for
    /// writing/reading a `dtype: "q8_0"` prefix image directly (see
    /// `kvimg.rs`'s module doc comment for that on-disk format; this
    /// method emits the same per-tensor `(scales, words)` shape it
    /// expects, just re-strided from `max_ctx` down to `n_tokens` rows).
    /// One synchronous readback per layer per buffer — native-only, same
    /// constraint as `export_prefix`.
    pub fn export_prefix_q8(&self, n_tokens: usize) -> Vec<Q8LayerBytes> {
        assert_eq!(self.dtype, KvDtype::Q8_0, "export_prefix_q8 requires KvDtype::Q8_0");
        assert!(n_tokens <= self.max_ctx);
        let blocks_per_row = self.head_dim / 32;
        let words_per_row = blocks_per_row * 8;
        let client = self.client();
        (0..self.k_scales.len())
            .map(|layer| {
                let k_scales = Self::repack_prefix_f32(
                    &client,
                    &self.k_scales[layer],
                    self.n_kv_heads,
                    self.max_ctx,
                    n_tokens,
                    blocks_per_row,
                );
                let k_words = Self::repack_prefix_u32(
                    &client,
                    &self.k_words[layer],
                    self.n_kv_heads,
                    self.max_ctx,
                    n_tokens,
                    words_per_row,
                );
                let v_scales = Self::repack_prefix_f32(
                    &client,
                    &self.v_scales[layer],
                    self.n_kv_heads,
                    self.max_ctx,
                    n_tokens,
                    blocks_per_row,
                );
                let v_words = Self::repack_prefix_u32(
                    &client,
                    &self.v_words[layer],
                    self.n_kv_heads,
                    self.max_ctx,
                    n_tokens,
                    words_per_row,
                );
                Q8LayerBytes {
                    k_scales,
                    k_words,
                    v_scales,
                    v_words,
                }
            })
            .collect()
    }

    fn repack_prefix_f32(
        client: &cubecl::client::ComputeClient<WgpuRuntime>,
        handle: &Handle,
        n_kv_heads: usize,
        max_ctx: usize,
        n_tokens: usize,
        row_stride: usize,
    ) -> Vec<f32> {
        let bytes = client.read_one(handle.clone());
        let full: &[f32] = bytemuck_cast_f32(&bytes);
        let mut out = Vec::with_capacity(n_kv_heads * n_tokens * row_stride);
        for h in 0..n_kv_heads {
            let base = h * max_ctx * row_stride;
            out.extend_from_slice(&full[base..base + n_tokens * row_stride]);
        }
        out
    }

    fn repack_prefix_u32(
        client: &cubecl::client::ComputeClient<WgpuRuntime>,
        handle: &Handle,
        n_kv_heads: usize,
        max_ctx: usize,
        n_tokens: usize,
        row_stride: usize,
    ) -> Vec<u32> {
        let bytes = client.read_one(handle.clone());
        let full: &[u32] = bytemuck_cast_u32(&bytes);
        let mut out = Vec::with_capacity(n_kv_heads * n_tokens * row_stride);
        for h in 0..n_kv_heads {
            let base = h * max_ctx * row_stride;
            out.extend_from_slice(&full[base..base + n_tokens * row_stride]);
        }
        out
    }

    /// Same as [`Self::export_prefix`] but via `into_data_async().await`
    /// (a WebGPU buffer map, asynchronous-only in the browser — see
    /// `model.rs`'s `logits_to_vec` doc comment) instead of the synchronous
    /// `into_data()` readback, so `web.rs` can call this on a KV-image
    /// cache *miss* (after its first full prefill) to write its own image
    /// back to OPFS without deadlocking.
    pub async fn export_prefix_async(&self, n_tokens: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        assert!(n_tokens <= self.max_ctx);
        let mut out = Vec::with_capacity(self.k.len());
        for (k, v) in self.k.iter().zip(self.v.iter()) {
            let k_data = k.clone().narrow(2, 0, n_tokens).into_data_async().await.expect("GPU readback failed");
            let v_data = v.clone().narrow(2, 0, n_tokens).into_data_async().await.expect("GPU readback failed");
            out.push((
                k_data.into_vec::<f32>().expect("K tensor is f32"),
                v_data.into_vec::<f32>().expect("V tensor is f32"),
            ));
        }
        out
    }

    /// Upload `layers` (per-layer `(k, v)` flat `[n_kv_heads, n_tokens,
    /// head_dim]` row-major, as produced by [`Self::export_prefix`] or
    /// `kvimg::KvImage::layer_slices`/`layer_f32`) into positions `[0,
    /// n_tokens)` of this cache, and set `len = n_tokens` so `snapshot()`
    /// reflects the imported prefix immediately.
    pub fn import_prefix(&mut self, layers: &[(Vec<f32>, Vec<f32>)], n_tokens: usize) {
        assert!(n_tokens <= self.max_ctx);
        let num_layers = match self.dtype {
            KvDtype::F32 => self.k_f32.len(),
            KvDtype::Q8_0 => self.k_scales.len(),
        };
        assert_eq!(
            layers.len(),
            num_layers,
            "layers.len()={} does not match cache's num_layers={}",
            layers.len(),
            num_layers
        );
        let expect_len = self.n_kv_heads * n_tokens * self.head_dim;
        for (layer, (k_flat, v_flat)) in layers.iter().enumerate() {
            assert_eq!(k_flat.len(), expect_len, "layer {layer} k length mismatch");
            assert_eq!(v_flat.len(), expect_len, "layer {layer} v length mismatch");

            match self.dtype {
                KvDtype::F32 => {
                    let device = self.k_f32[layer].device();
                    let ranges = [0..1usize, 0..self.n_kv_heads, 0..n_tokens, 0..self.head_dim];
                    let k_new = Tensor::<Wgpu, 4>::from_data(
                        TensorData::new(k_flat.clone(), [1, self.n_kv_heads, n_tokens, self.head_dim]),
                        &device,
                    );
                    let v_new = Tensor::<Wgpu, 4>::from_data(
                        TensorData::new(v_flat.clone(), [1, self.n_kv_heads, n_tokens, self.head_dim]),
                        &device,
                    );
                    let placeholder_shape = [1, self.n_kv_heads, 1, self.head_dim];
                    let old_k =
                        std::mem::replace(&mut self.k_f32[layer], Tensor::<Wgpu, 4>::empty(placeholder_shape, &device));
                    let old_v =
                        std::mem::replace(&mut self.v_f32[layer], Tensor::<Wgpu, 4>::empty(placeholder_shape, &device));
                    self.k_f32[layer] = old_k.slice_assign(ranges.clone(), k_new);
                    self.v_f32[layer] = old_v.slice_assign(ranges, v_new);
                }
                KvDtype::Q8_0 => {
                    let client = self.client();
                    let k_handle = client.create_from_slice(&bytemuck_bytes_f32(k_flat));
                    let v_handle = client.create_from_slice(&bytemuck_bytes_f32(v_flat));
                    crate::gguf::kv_quantize_dispatch(
                        &client,
                        k_handle,
                        &self.k_scales[layer],
                        &self.k_words[layer],
                        self.n_kv_heads,
                        n_tokens,
                        self.head_dim,
                        self.max_ctx,
                        0,
                    );
                    crate::gguf::kv_quantize_dispatch(
                        &client,
                        v_handle,
                        &self.v_scales[layer],
                        &self.v_words[layer],
                        self.n_kv_heads,
                        n_tokens,
                        self.head_dim,
                        self.max_ctx,
                        0,
                    );
                }
            }
        }
        self.len = n_tokens;
    }

    /// `KvDtype::Q8_0`-only counterpart to [`Self::import_prefix`] that
    /// writes already-quantized bytes directly (no dequant/re-quantize
    /// round trip) — the inverse of [`Self::export_prefix_q8`]. Read-modify
    /// -write on the CPU (readback the full per-layer buffer, patch rows
    /// `[0, n_tokens)`, re-upload): correctness first, acceptable since
    /// this runs once per session/prefix load, not per token.
    pub fn import_prefix_q8(&mut self, layers: &[Q8LayerBytes], n_tokens: usize) {
        assert_eq!(self.dtype, KvDtype::Q8_0, "import_prefix_q8 requires KvDtype::Q8_0");
        assert!(n_tokens <= self.max_ctx);
        assert_eq!(layers.len(), self.k_scales.len());
        let blocks_per_row = self.head_dim / 32;
        let words_per_row = blocks_per_row * 8;
        let client = self.client();
        for (layer, data) in layers.iter().enumerate() {
            self.k_scales[layer] = Self::patch_prefix_f32(
                &client,
                &self.k_scales[layer],
                &data.k_scales,
                self.n_kv_heads,
                self.max_ctx,
                n_tokens,
                blocks_per_row,
            );
            self.k_words[layer] = Self::patch_prefix_u32(
                &client,
                &self.k_words[layer],
                &data.k_words,
                self.n_kv_heads,
                self.max_ctx,
                n_tokens,
                words_per_row,
            );
            self.v_scales[layer] = Self::patch_prefix_f32(
                &client,
                &self.v_scales[layer],
                &data.v_scales,
                self.n_kv_heads,
                self.max_ctx,
                n_tokens,
                blocks_per_row,
            );
            self.v_words[layer] = Self::patch_prefix_u32(
                &client,
                &self.v_words[layer],
                &data.v_words,
                self.n_kv_heads,
                self.max_ctx,
                n_tokens,
                words_per_row,
            );
        }
        self.len = n_tokens;
    }

    fn patch_prefix_f32(
        client: &cubecl::client::ComputeClient<WgpuRuntime>,
        handle: &Handle,
        new_rows: &[f32],
        n_kv_heads: usize,
        max_ctx: usize,
        n_tokens: usize,
        row_stride: usize,
    ) -> Handle {
        let bytes = client.read_one(handle.clone());
        let full: &[f32] = bytemuck_cast_f32(&bytes);
        let mut patched = full.to_vec();
        for h in 0..n_kv_heads {
            let dst = h * max_ctx * row_stride;
            let src = h * n_tokens * row_stride;
            patched[dst..dst + n_tokens * row_stride].copy_from_slice(&new_rows[src..src + n_tokens * row_stride]);
        }
        client.create_from_slice(&bytemuck_bytes_f32(&patched))
    }

    fn patch_prefix_u32(
        client: &cubecl::client::ComputeClient<WgpuRuntime>,
        handle: &Handle,
        new_rows: &[u32],
        n_kv_heads: usize,
        max_ctx: usize,
        n_tokens: usize,
        row_stride: usize,
    ) -> Handle {
        let bytes = client.read_one(handle.clone());
        let full: &[u32] = bytemuck_cast_u32(&bytes);
        let mut patched = full.to_vec();
        for h in 0..n_kv_heads {
            let dst = h * max_ctx * row_stride;
            let src = h * n_tokens * row_stride;
            patched[dst..dst + n_tokens * row_stride].copy_from_slice(&new_rows[src..src + n_tokens * row_stride]);
        }
        client.create_from_slice(&bytemuck_bytes_u32(&patched))
    }
}

/// Raw quantized bytes for one layer's K and V q8_0 buffers, re-strided to
/// exactly `n_tokens` rows (no `max_ctx` padding) — see
/// [`KvCache::export_prefix_q8`]/[`KvCache::import_prefix_q8`].
pub struct Q8LayerBytes {
    pub k_scales: Vec<f32>,
    pub k_words: Vec<u32>,
    pub v_scales: Vec<f32>,
    pub v_words: Vec<u32>,
}

fn bytemuck_cast_f32(bytes: &[u8]) -> &[f32] {
    assert_eq!(bytes.len() % 4, 0);
    // SAFETY: `bytes` comes from a GPU buffer of f32 data created via
    // `create_from_slice`/kernel writes only; length is a multiple of 4 and
    // alignment is handled by copying into a fresh Vec when misaligned.
    let (prefix, mid, suffix) = unsafe { bytes.align_to::<f32>() };
    if prefix.is_empty() && suffix.is_empty() {
        mid
    } else {
        // Fallback: shouldn't happen for GPU-readback buffers in practice,
        // but stay correct rather than panic on an alignment fluke.
        Box::leak(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        )
    }
}

fn bytemuck_cast_u32(bytes: &[u8]) -> &[u32] {
    assert_eq!(bytes.len() % 4, 0);
    let (prefix, mid, suffix) = unsafe { bytes.align_to::<u32>() };
    if prefix.is_empty() && suffix.is_empty() {
        mid
    } else {
        Box::leak(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>()
                .into_boxed_slice(),
        )
    }
}

fn bytemuck_bytes_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytemuck_bytes_u32(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(all(test, feature = "wgpu"))]
mod bench {
    use super::*;

    /// D1 isolation bench (docs/BENCHMARKS.md Session 3): times `write`
    /// alone, one layer, T=1 (decode shape), at the model's real
    /// `n_kv_heads=2, head_dim=128, max_ctx=12288` cache shape, in
    /// `KvDtype::F32` mode (the original target of this bench — Session 13
    /// added `bench_write_q8` alongside it for the new mode). Run with:
    /// `cargo test --release --features wgpu --lib kv::bench::bench_write
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_write() {
        let device = WgpuDevice::default();
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 12288;
        let iters = 200;

        let mut cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::F32);
        let kv_shape = [1, n_kv_heads, 1, head_dim];

        let prefill_len = 2225;
        cache.len = prefill_len;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let k = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            let v = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            cache.write(0, k, v);
            cache.advance(1);
            let (k_all, _v_all) = cache.read_or_dequant_f32(0, cache.len());
            let _ = k_all.into_data().into_vec::<f32>().unwrap();
        }
        let elapsed = t0.elapsed();
        println!(
            "write (F32, T=1, n_kv_heads={n_kv_heads}, head_dim={head_dim}, max_ctx={max_ctx}): \
             {:.4} ms/call over {iters} iters",
            elapsed.as_secs_f64() * 1000.0 / iters as f64
        );
    }

    /// Session 13: `KvDtype::Q8_0` counterpart to `bench_write`. Run with:
    /// `cargo test --release --features wgpu --lib kv::bench::bench_write_q8
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_write_q8() {
        let device = WgpuDevice::default();
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 12288;
        let iters = 200;

        let mut cache = KvCache::new_with_dtype(1, n_kv_heads, head_dim, max_ctx, &device, KvDtype::Q8_0);
        let kv_shape = [1, n_kv_heads, 1, head_dim];

        let prefill_len = 2225;
        cache.len = prefill_len;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let k = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            let v = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            cache.write(0, k, v);
            cache.advance(1);
            let bytes = cache.client().read_one(cache.k_words[0].clone());
            let _ = bytes.len();
        }
        let elapsed = t0.elapsed();
        println!(
            "write (Q8_0, T=1, n_kv_heads={n_kv_heads}, head_dim={head_dim}, max_ctx={max_ctx}): \
             {:.4} ms/call over {iters} iters",
            elapsed.as_secs_f64() * 1000.0 / iters as f64
        );
    }
}
