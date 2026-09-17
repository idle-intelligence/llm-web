//! `ForwardSpec` (caller-supplied RoPE positions + attention mask) and the
//! sliced lm-head, on a tiny random-weight Qwen2 model.
//!
//! Added for `llm-life`, which drives this engine as a cellular-automaton
//! update rule: many independent cells packed into one sequence, RoPE
//! positions that do not mean "index", attention restricted to a 2D stencil,
//! and logits read at every position over a two-token answer vocabulary.
//!
//! Every test here builds its own ~90k-parameter model from a deterministic
//! PRNG — no GGUF, no download, seconds to run. They still need a wgpu
//! adapter (this crate has no CPU backend), so they fail loudly rather than
//! skipping if none is available.
#![cfg(feature = "wgpu")]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::module::{Param, ParamId};
use burn::tensor::{Tensor, TensorData};
use llm_wasm::gguf::{EmbeddingStore, Q4Linear, Q4Tensor};
use llm_wasm::model::{ForwardSpec, LlmModel, Q4Attention, Q4FeedForward, Q4TransformerBlock, RmsNormLayer, RoPE};
use llm_wasm::LlmConfig;

const LAYERS: usize = 2;
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 96;
const MAX_SEQ: usize = 64;

struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() % 2000) as f32 / 1000.0 - 1.0
    }
}

/// f32 -> IEEE binary16 bits (normals only; the scales here are ~0.01-0.2).
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exp as u16) << 10) | mant
}

/// Random Q4_0 blob of shape `[n, k]`: 18 bytes per 32-element block
/// (f16 scale + 16 bytes of paired nibbles).
fn random_q4_bytes(n: usize, k: usize, rng: &mut Xorshift) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks = k / 32;
    let mut bytes = vec![0u8; n * blocks * 18];
    for row in 0..n {
        for b in 0..blocks {
            let off = (row * blocks + b) * 18;
            let scale = 0.01 + (rng.next_u32() % 50) as f32 * 0.002;
            bytes[off..off + 2].copy_from_slice(&half_bits(scale).to_le_bytes());
            for byte in bytes[off + 2..off + 18].iter_mut() {
                *byte = (rng.next_u32() & 0xff) as u8;
            }
        }
    }
    bytes
}

fn q4_linear(n: usize, k: usize, rng: &mut Xorshift, device: &WgpuDevice) -> Q4Linear {
    let bytes = random_q4_bytes(n, k, rng);
    Q4Linear::new(Q4Tensor::from_q4_bytes(&bytes, [n, k], device).unwrap(), None)
}

fn rms_norm(n: usize, rng: &mut Xorshift, device: &WgpuDevice) -> RmsNormLayer {
    let data: Vec<f32> = (0..n).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    let weight: Tensor<Wgpu, 1> = Tensor::from_data(TensorData::new(data, [n]), device);
    RmsNormLayer {
        inner: burn::nn::RmsNorm {
            gamma: Param::initialized(ParamId::new(), weight),
            epsilon: 1e-6,
        },
    }
}

fn tiny_config() -> LlmConfig {
    LlmConfig {
        num_layers: LAYERS,
        hidden_size: HIDDEN,
        num_heads: HEADS,
        num_kv_heads: KV_HEADS,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        rope_theta: 10_000.0,
        max_seq_len: MAX_SEQ,
        rms_norm_eps: 1e-6,
        bos_token_id: 0,
        eos_token_ids: vec![1],
    }
}

fn tiny_model(seed: u64, device: &WgpuDevice) -> LlmModel {
    let mut rng = Xorshift::new(seed);
    let config = tiny_config();
    let layers: Vec<Q4TransformerBlock> = (0..LAYERS)
        .map(|_| {
            let attn = Q4Attention::new(
                q4_linear(HEADS * HEAD_DIM, HIDDEN, &mut rng, device),
                q4_linear(KV_HEADS * HEAD_DIM, HIDDEN, &mut rng, device),
                q4_linear(KV_HEADS * HEAD_DIM, HIDDEN, &mut rng, device),
                q4_linear(HIDDEN, HEADS * HEAD_DIM, &mut rng, device),
                HEADS,
                KV_HEADS,
                HEAD_DIM,
            );
            let ffn = Q4FeedForward::new(
                q4_linear(INTERMEDIATE, HIDDEN, &mut rng, device),
                q4_linear(INTERMEDIATE, HIDDEN, &mut rng, device),
                q4_linear(HIDDEN, INTERMEDIATE, &mut rng, device),
            );
            Q4TransformerBlock::new(rms_norm(HIDDEN, &mut rng, device), attn, rms_norm(HIDDEN, &mut rng, device), ffn)
        })
        .collect();

    let embd_bytes = random_q4_bytes(VOCAB, HIDDEN, &mut rng);
    let lm_head = Q4Linear::new(
        Q4Tensor::from_q4_bytes(&embd_bytes, [VOCAB, HIDDEN], device).unwrap(),
        None,
    );
    let embed = EmbeddingStore::new(embd_bytes, VOCAB, HIDDEN);

    LlmModel::new(
        embed,
        layers,
        RoPE::new(HEAD_DIM, MAX_SEQ, 10_000.0, device),
        rms_norm(HIDDEN, &mut rng, device),
        lm_head,
        config,
        device.clone(),
    )
}

fn hidden_to_vec(t: Tensor<Wgpu, 3>) -> Vec<f32> {
    t.into_data().into_vec::<f32>().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// The default `ForwardSpec` must be the old behavior: spelling out
/// "positions are 0..T" and "the mask is causal" by hand has to reproduce it.
/// If this drifts, every llm-life measurement is against a different model
/// than the one llm-web ships.
#[test]
fn explicit_causal_spec_matches_the_default_path() {
    let device = WgpuDevice::default();
    let model = tiny_model(0xA11CE, &device);
    let ids: Vec<u32> = vec![3, 17, 42, 5, 88, 1, 9];
    let t = ids.len();

    let mut cache = model.new_cache(MAX_SEQ);
    let baseline = hidden_to_vec(model.forward_hidden(&ids, &mut cache).unwrap());

    let allowed: Vec<bool> = (0..t).flat_map(|i| (0..t).map(move |j| j <= i)).collect();
    let spec = ForwardSpec::default()
        .with_positions((0..t as u32).collect())
        .with_allowed(&allowed, t, t, &device);
    let mut cache = model.new_cache(MAX_SEQ);
    let specced = hidden_to_vec(model.forward_hidden_spec(&ids, &mut cache, &spec).unwrap());

    let d = max_abs_diff(&baseline, &specced);
    println!("explicit causal spec vs default: max_abs_diff={d:.3e}");
    assert!(d < 1e-4, "explicit causal spec diverged from the default path by {d}");
}

/// A mask that closes a key must actually change the answer — otherwise
/// `with_allowed` could be silently ignored and every test above would still
/// pass.
#[test]
fn closing_a_key_changes_the_output() {
    let device = WgpuDevice::default();
    let model = tiny_model(0xB0B, &device);
    let ids: Vec<u32> = vec![3, 17, 42, 5];
    let t = ids.len();

    let causal: Vec<bool> = (0..t).flat_map(|i| (0..t).map(move |j| j <= i)).collect();
    // Same, but the last query no longer sees token 0.
    let mut restricted = causal.clone();
    restricted[(t - 1) * t] = false;

    let run = |allowed: &[bool]| {
        let spec = ForwardSpec::default().with_allowed(allowed, t, t, &device);
        let mut cache = model.new_cache(MAX_SEQ);
        hidden_to_vec(model.forward_hidden_spec(&ids, &mut cache, &spec).unwrap())
    };
    let d = max_abs_diff(&run(&causal), &run(&restricted));
    println!("closing one key: max_abs_diff={d:.3e}");
    assert!(d > 1e-4, "closing a key left the output unchanged (mask ignored?)");
}

/// CONCEPT.md §4: Life is outer-totalistic, so the default is "no positions"
/// — every grid token gets the same position id, RoPE is relative, and the
/// neighbors become a *bag*. This asserts exactly that: with constant
/// positions and an all-visible mask, permuting the other tokens leaves a
/// given token's hidden state alone.
#[test]
fn constant_positions_make_the_context_a_bag() {
    let device = WgpuDevice::default();
    let model = tiny_model(0xBA6, &device);
    let t = 4usize;
    let allowed = vec![true; t * t];

    let run = |ids: &[u32]| {
        let spec = ForwardSpec::default()
            .with_positions(vec![0; t])
            .with_allowed(&allowed, t, t, &device);
        let mut cache = model.new_cache(MAX_SEQ);
        hidden_to_vec(model.forward_hidden_spec(ids, &mut cache, &spec).unwrap())
    };

    let a = run(&[7, 11, 23, 31]);
    let b = run(&[7, 23, 31, 11]);
    // Token 0 is in the same slot in both; the rest were shuffled.
    let d = max_abs_diff(&a[..HIDDEN], &b[..HIDDEN]);
    println!("bag-mode permutation invariance: max_abs_diff={d:.3e}");
    assert!(d < 1e-4, "bag mode is order-sensitive: {d}");

    // Sanity: with real positions the same permutation *does* move it.
    let ordered = |ids: &[u32]| {
        let spec = ForwardSpec::default()
            .with_positions((0..t as u32).collect())
            .with_allowed(&allowed, t, t, &device);
        let mut cache = model.new_cache(MAX_SEQ);
        hidden_to_vec(model.forward_hidden_spec(ids, &mut cache, &spec).unwrap())
    };
    let d2 = max_abs_diff(&ordered(&[7, 11, 23, 31])[..HIDDEN], &ordered(&[7, 23, 31, 11])[..HIDDEN]);
    println!("ordered permutation sensitivity: max_abs_diff={d2:.3e}");
    assert!(d2 > 1e-4, "positions 0..T should make order matter, got {d2}");
}

/// The sliced head must equal the corresponding columns of the full head, at
/// every position — that is what makes all-position logits affordable
/// (CONCEPT.md §1 "sliced lm-head").
#[test]
fn sliced_head_matches_the_full_head_at_every_position() {
    let device = WgpuDevice::default();
    let model = tiny_model(0x5111CE, &device);
    let ids: Vec<u32> = vec![3, 17, 42, 5, 88];
    let t = ids.len();
    let answers: Vec<u32> = vec![15, 16, 61];

    let mut cache = model.new_cache(MAX_SEQ);
    let hidden = model.forward_hidden(&ids, &mut cache).unwrap();

    let full = hidden_to_vec(model.lm_head(hidden.clone()));
    let head = model.head_slice(&answers).unwrap();
    let sliced = hidden_to_vec(model.lm_head_sliced(hidden, &head));
    assert_eq!(sliced.len(), t * answers.len());

    let mut worst = 0f32;
    for pos in 0..t {
        for (a, &tok) in answers.iter().enumerate() {
            let expected = full[pos * VOCAB + tok as usize];
            let got = sliced[pos * answers.len() + a];
            worst = worst.max((expected - got).abs() / expected.abs().max(1e-3));
        }
    }
    println!("sliced vs full head: max_rel_diff={worst:.3e}");
    assert!(worst < 1e-3, "sliced head diverged from the full head by rel {worst}");
}
