//! KV cache for autoregressive decoding + prefill.
//!
//! Layout: one `[1, n_kv_heads, max_ctx, head_dim]` f32 tensor per layer,
//! **sequence-major** (contiguous along the time axis) so a prefill of `T`
//! tokens writes one contiguous `slice_assign` per layer instead of `T`
//! separate single-token writes. Batch is always 1 (single-session MCP
//! agent — see `model.rs`'s `Q4Attention::forward` assert).
//!
//! dtype: **f32**, not f16. Burn's wgpu backend doesn't give native f16
//! compute for the tensor ops used here (softmax, matmul via cubecl), and
//! the model's own attention math runs in f32 (stt-web precedent,
//! docs/ENGINE.md §4) — keeping the cache f32 avoids per-step cast
//! overhead. Cost at the default 12288-token `max_ctx` (sized so the
//! 8140-token 34-tool Sonos prompt plus conversation fits, per the task
//! brief): `num_layers(36) * 2(k+v) * n_kv_heads(2) * max_ctx(12288) *
//! head_dim(128) * 4 bytes ≈ 906 MB`. f16 would halve this to ~453MB; left
//! as a documented future optimization, not done here.
//!
//! `snapshot()`/`restore()` are O(1): they only save/restore the fill
//! length `len`, not any tensor data. This lets a constant prefix (system
//! prompt + tool schemas) be prefilled once, `snapshot()`-ed, and then
//! `restore()`-ed before each new user turn — new tokens simply overwrite
//! whatever was previously written past the restored length; no data is
//! copied or cleared.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

/// Per-layer K/V ring of fixed capacity `max_ctx`, append-only until reset.
pub struct KvCache {
    k: Vec<Tensor<Wgpu, 4>>,
    v: Vec<Tensor<Wgpu, 4>>,
    len: usize,
    max_ctx: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        device: &WgpuDevice,
    ) -> Self {
        let shape = [1, n_kv_heads, max_ctx, head_dim];
        let k = (0..num_layers)
            .map(|_| Tensor::<Wgpu, 4>::zeros(shape, device))
            .collect();
        let v = (0..num_layers)
            .map(|_| Tensor::<Wgpu, 4>::zeros(shape, device))
            .collect();
        Self {
            k,
            v,
            len: 0,
            max_ctx,
            n_kv_heads,
            head_dim,
        }
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

    /// Write `k`/`v` (`[1, n_kv_heads, T, head_dim]`) for `layer` at
    /// `[len, len+T)` and return the full valid prefix `[1, n_kv_heads,
    /// len+T, head_dim]` (prior cache contents + the just-written rows).
    /// Does **not** advance `len` — call [`Self::advance`] once per forward
    /// step (all layers in a step share the same write range).
    pub fn append(
        &mut self,
        layer: usize,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let t = k.dims()[2];
        assert!(
            self.len + t <= self.max_ctx,
            "KV cache overflow: len={} + t={} > max_ctx={}",
            self.len,
            t,
            self.max_ctx
        );
        let ranges = [
            0..1usize,
            0..self.n_kv_heads,
            self.len..self.len + t,
            0..self.head_dim,
        ];
        // D1 (docs/BENCHMARKS.md Session 2 "next most valuable"): the old
        // code did `self.k[layer].clone().slice_assign(...)`, which left
        // `self.k[layer]` itself holding a second reference to the same
        // buffer at the moment `slice_assign` ran, guaranteeing refcount
        // >= 2 and forcing cubecl's wgpu backend to copy the *entire*
        // `[1, n_kv_heads, max_ctx, head_dim]` cache tensor (12.58MB/layer
        // at max_ctx=12288) instead of writing only the new `t` rows in
        // place. `mem::replace`-ing the slot with a tiny placeholder first
        // drops that second reference before `slice_assign` runs, so the
        // taken-out tensor is uniquely owned and cubecl can mutate its
        // buffer in place.
        let placeholder_shape = [1, self.n_kv_heads, 1, self.head_dim];
        let old_k = std::mem::replace(
            &mut self.k[layer],
            Tensor::<Wgpu, 4>::empty(placeholder_shape, &k.device()),
        );
        let old_v = std::mem::replace(
            &mut self.v[layer],
            Tensor::<Wgpu, 4>::empty(placeholder_shape, &v.device()),
        );
        self.k[layer] = old_k.slice_assign(ranges.clone(), k);
        self.v[layer] = old_v.slice_assign(ranges, v);

        let k_all = self.k[layer].clone().narrow(2, 0, self.len + t);
        let v_all = self.v[layer].clone().narrow(2, 0, self.len + t);
        (k_all, v_all)
    }

    /// Advance the fill length by `t` — call once per forward step, after
    /// every layer's `append` for that step.
    pub fn advance(&mut self, t: usize) {
        self.len += t;
    }

    /// O(1): save the current fill length.
    pub fn snapshot(&self) -> usize {
        self.len
    }

    /// O(1): restore a previously `snapshot()`-ed fill length. Rows past
    /// the restored length are left in place (garbage from a longer prior
    /// turn) but are never read, since `append`/reads always operate on
    /// `[0, len)`.
    pub fn restore(&mut self, snapshot: usize) {
        assert!(snapshot <= self.max_ctx);
        self.len = snapshot;
    }

    /// Read back positions `[0, n_tokens)` of every layer's K/V from the
    /// GPU as flat `Vec<f32>` in `[n_kv_heads, n_tokens, head_dim]`
    /// row-major order (the batch=1 axis is dropped — it's redundant with
    /// the flattened layout) — the layout `kvimg::KvImage::write` expects
    /// per layer. One `into_data()` readback per layer per tensor (not per
    /// token). **Native-only usage**: this is the build-time `kv-export`
    /// CLI path (see `docs/ENGINE.md` "Prefix KV images"); a caller
    /// running inside WASM must not use this synchronous readback (it
    /// deadlocks the browser) and should add an `into_data_async` variant
    /// before calling this from `web.rs`.
    pub fn export_prefix(&self, n_tokens: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        assert!(n_tokens <= self.max_ctx);
        self.k
            .iter()
            .zip(self.v.iter())
            .map(|(k, v)| {
                let k_data = k.clone().narrow(2, 0, n_tokens).into_data();
                let v_data = v.clone().narrow(2, 0, n_tokens).into_data();
                (
                    k_data.into_vec::<f32>().expect("K tensor is f32"),
                    v_data.into_vec::<f32>().expect("V tensor is f32"),
                )
            })
            .collect()
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
    /// `kvimg::KvImage::layer_slices`) into positions `[0, n_tokens)` of
    /// this cache's tensors, one `slice_assign` per layer per tensor (not
    /// per token — matches `append`'s bulk-write shape), and set `len =
    /// n_tokens` so `snapshot()` reflects the imported prefix immediately.
    pub fn import_prefix(&mut self, layers: &[(Vec<f32>, Vec<f32>)], n_tokens: usize) {
        assert!(n_tokens <= self.max_ctx);
        assert_eq!(
            layers.len(),
            self.k.len(),
            "layers.len()={} does not match cache's num_layers={}",
            layers.len(),
            self.k.len()
        );
        let ranges = [0..1usize, 0..self.n_kv_heads, 0..n_tokens, 0..self.head_dim];
        for (layer, (k_flat, v_flat)) in layers.iter().enumerate() {
            let expect_len = self.n_kv_heads * n_tokens * self.head_dim;
            assert_eq!(k_flat.len(), expect_len, "layer {layer} k length mismatch");
            assert_eq!(v_flat.len(), expect_len, "layer {layer} v length mismatch");

            let device = self.k[layer].device();
            let k_new = Tensor::<Wgpu, 4>::from_data(
                burn::tensor::TensorData::new(k_flat.clone(), [1, self.n_kv_heads, n_tokens, self.head_dim]),
                &device,
            );
            let v_new = Tensor::<Wgpu, 4>::from_data(
                burn::tensor::TensorData::new(v_flat.clone(), [1, self.n_kv_heads, n_tokens, self.head_dim]),
                &device,
            );

            // Same placeholder-swap discipline as `append` (see its D1
            // comment): drop the second reference to the cache slot before
            // `slice_assign` so cubecl mutates the existing buffer in
            // place instead of copying the whole `max_ctx`-sized tensor.
            let placeholder_shape = [1, self.n_kv_heads, 1, self.head_dim];
            let old_k = std::mem::replace(&mut self.k[layer], Tensor::<Wgpu, 4>::empty(placeholder_shape, &device));
            let old_v = std::mem::replace(&mut self.v[layer], Tensor::<Wgpu, 4>::empty(placeholder_shape, &device));
            self.k[layer] = old_k.slice_assign(ranges.clone(), k_new);
            self.v[layer] = old_v.slice_assign(ranges.clone(), v_new);
        }
        self.len = n_tokens;
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod bench {
    use super::*;

    /// D1 isolation bench (docs/BENCHMARKS.md Session 3): times `append`
    /// alone, one layer, T=1 (decode shape), at the model's real
    /// `n_kv_heads=2, head_dim=128, max_ctx=12288` cache shape. Run with:
    /// `cargo test --release --features wgpu --lib kv::bench::bench_append
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_append() {
        let device = WgpuDevice::default();
        let n_kv_heads = 2;
        let head_dim = 128;
        let max_ctx = 12288;
        let iters = 200;

        let mut cache = KvCache::new(1, n_kv_heads, head_dim, max_ctx, &device);
        let kv_shape = [1, n_kv_heads, 1, head_dim];

        // Warm up + advance past a synthetic prefill so append is measured
        // at a realistic mid-cache offset, not len=0.
        let prefill_len = 2225;
        cache.len = prefill_len;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let k = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            let v = Tensor::<Wgpu, 4>::zeros(kv_shape, &device);
            let (k_all, _v_all) = cache.append(0, k, v);
            cache.advance(1);
            // Force sync every iter so each append's GPU work is actually
            // retired before starting the next — otherwise wgpu queues
            // work asynchronously and the wall clock only measures
            // submission overhead, not execution time.
            let _ = k_all.into_data().into_vec::<f32>().unwrap();
        }
        let elapsed = t0.elapsed();
        println!(
            "append (T=1, n_kv_heads={n_kv_heads}, head_dim={head_dim}, max_ctx={max_ctx}): \
             {:.4} ms/call over {iters} iters",
            elapsed.as_secs_f64() * 1000.0 / iters as f64
        );
    }
}
