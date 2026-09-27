//! Qwen2.5-0.5B-Instruct GPU-resident forward pass: GGUF weights straight
//! onto the GPU (two-phase loading — `GgufReader`'s file handle is dropped
//! at the end of `GpuModel::load`, never held alongside the GPU-resident
//! copies), one dispatch per op recorded into a single `wgpu::CommandEncoder`
//! per prefill/decode call, one readback per call (`Engine::read_buffer`).
//! Structure (Engine/Pool split, `linear`/`rmsnorm`/`add_inplace`/`silu_mul`
//! helper-fn shape) ported from `t0-web/crates/t0-fast/src/model.rs`; the
//! attention/RoPE/embedding-gather kernels are new (t0 has no GQA, no
//! causal mask, no token embedding table — see the shaders' own doc
//! comments for exactly what's ported vs new).
//!
//! Reference for every op's numerics: `crates/lean/reference/gen_fixture.py`
//! (HF transformers' own Qwen2 forward, `AutoModelForCausalLM` loading the
//! same GGUF via `gguf_file=`) — see `modeling_qwen2.py` in that venv's
//! transformers install for `rotate_half`, GQA `repeat_kv`, RMSNorm, SwiGLU.
//!
//! Tensor names follow llama.cpp's GGUF convention
//! (`blk.N.attn_{q,k,v,output}`, `ffn_{gate,up,down}`). This GGUF carries a
//! separate `output.weight` (Q8_0) distinct from `token_embd.weight`
//! (Q4_0) despite `tie_word_embeddings: true` in `config.json` — verified
//! against the file directly (not assumed), so the lm head uses
//! `output.weight` and needs no tied-embedding sharing logic.
//!
//! KV cache layout (load-bearing, chosen here): per layer, two buffers
//! `[n_kv_heads, max_ctx, head_dim]`, head-major and contiguous per head —
//! matches `shaders/attn_decode.wgsl`'s indexing. Not a ring buffer: `kv_len`
//! only grows, capped by `max_ctx` (the CLI's fixed context, not the
//! model's `qwen2.context_length`). Writes into it are plain
//! `copy_buffer_to_buffer` calls recorded in the same encoder as the
//! dispatch that produced the source K/V — GPU-resident, no CPU readback —
//! so a freshly-computed decode step's own K/V is visible to that same
//! step's causal attention (`kv_len` passed to `attn_decode` already
//! includes the current position).

use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use wgpu::BindGroupEntry;

use crate::config::{config_from_gguf, Qwen2Config};
use crate::engine::Engine;
use crate::gguf::GgufReader;
use crate::pool::Pool;
use crate::quant::{load_matmul_weight_gguf, load_q4_embedding_gguf, MatMulWeight, Q4EmbeddingTable};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearQDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AddDims {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsDims {
    rows: u32,
    dim: u32,
    eps: f32,
    _p0: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeDims {
    rows: u32,
    heads: u32,
    head_dim: u32,
    pos_base: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluDims {
    rows: u32,
    hidden: u32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnPrefillDims {
    seq: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeDims {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    max_ctx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherDims {
    rows: u32,
    hidden: u32,
    blocks_per_row: u32,
    _p0: u32,
}

struct LayerWeights {
    attn_norm: wgpu::Buffer,
    q_w: MatMulWeight,
    q_b: wgpu::Buffer,
    k_w: MatMulWeight,
    k_b: wgpu::Buffer,
    v_w: MatMulWeight,
    v_b: wgpu::Buffer,
    o_w: MatMulWeight,
    o_b: wgpu::Buffer, // zero, attn_output has no bias
    ffn_norm: wgpu::Buffer,
    gate_w: MatMulWeight,
    gate_b: wgpu::Buffer, // zero
    up_w: MatMulWeight,
    up_b: wgpu::Buffer, // zero
    down_w: MatMulWeight,
    down_b: wgpu::Buffer, // zero
}

pub struct GpuModel {
    pub config: Qwen2Config,
    embed: Q4EmbeddingTable,
    layers: Vec<LayerWeights>,
    out_norm: wgpu::Buffer,
    lm_head: MatMulWeight,
    zero_bias_vocab: wgpu::Buffer,
    /// Selects, inside `linear()`, between the naive reference kernel
    /// (`false`, always `linear_q4.wgsl`) and the tiled/coalesced
    /// fast kernels ported in slice 2 (`true`). Set at `load()` time so a
    /// single `GpuModel` doesn't mix the two — the fixture check runs both.
    pub fast_kernels: bool,
    pub pool: Pool,
}

/// One KV cache buffer pair per layer. See this file's top doc comment for
/// the layout. `max_ctx` is the CLI's fixed context length (prompt +
/// max_new_tokens known up front), not the model's full 32768 window.
pub struct KvCache {
    pub k: Vec<wgpu::Buffer>,
    pub v: Vec<wgpu::Buffer>,
    pub max_ctx: u32,
    pub kv_len: u32,
}

impl KvCache {
    pub fn new(engine: &Engine, config: &Qwen2Config, max_ctx: u32) -> Self {
        let per_layer = (config.num_kv_heads * config.head_dim) as u32 * max_ctx;
        let k = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.k"))).collect();
        let v = (0..config.num_layers).map(|i| engine.buf_empty(per_layer as usize, &format!("kv{i}.v"))).collect();
        KvCache { k, v, max_ctx, kv_len: 0 }
    }
}

fn gguf_f32<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<wgpu::Buffer> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let n: usize = info.shape().iter().product();
    let bytes = reader.tensor_data(name)?;
    let data = crate::gguf::dequantize_for(info.dtype(), &bytes, n);
    Ok(engine.buf_f32(&data, name))
}

fn gguf_matmul<R: std::io::Read + std::io::Seek>(engine: &Engine, reader: &mut GgufReader<R>, name: &str) -> Result<MatMulWeight> {
    let info = reader.tensor_info(name).with_context(|| format!("missing tensor {name}"))?.clone();
    let shape = info.shape();
    let bytes = reader.tensor_data(name)?;
    Ok(load_matmul_weight_gguf(engine, name, &shape, info.dtype(), &bytes))
}

impl GpuModel {
    pub fn load(engine: &Engine, gguf_path: &str, fast_kernels: bool) -> Result<Self> {
        let file = File::open(gguf_path).with_context(|| format!("opening {gguf_path}"))?;
        let mut reader = GgufReader::open(BufReader::new(file))?;
        let config = config_from_gguf(&reader)?;

        let embed_info = reader.tensor_info("token_embd.weight").context("missing token_embd.weight")?.clone();
        let embed_shape = embed_info.shape();
        anyhow::ensure!(embed_info.dtype() == crate::gguf::GgmlDtype::Q4_0, "expected token_embd.weight to be Q4_0, got {:?}", embed_info.dtype());
        let embed_bytes = reader.tensor_data("token_embd.weight")?;
        let embed = load_q4_embedding_gguf(engine, "token_embd", &embed_shape, &embed_bytes);
        drop(embed_bytes);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{i}");
            layers.push(LayerWeights {
                attn_norm: gguf_f32(engine, &mut reader, &format!("{p}.attn_norm.weight"))?,
                q_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_q.weight"))?,
                q_b: gguf_f32(engine, &mut reader, &format!("{p}.attn_q.bias"))?,
                k_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_k.weight"))?,
                k_b: gguf_f32(engine, &mut reader, &format!("{p}.attn_k.bias"))?,
                v_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_v.weight"))?,
                v_b: gguf_f32(engine, &mut reader, &format!("{p}.attn_v.bias"))?,
                o_w: gguf_matmul(engine, &mut reader, &format!("{p}.attn_output.weight"))?,
                o_b: engine.buf_f32(&vec![0f32; config.hidden_size], "o_b_zero"),
                ffn_norm: gguf_f32(engine, &mut reader, &format!("{p}.ffn_norm.weight"))?,
                gate_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_gate.weight"))?,
                gate_b: engine.buf_f32(&vec![0f32; config.intermediate_size], "gate_b_zero"),
                up_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_up.weight"))?,
                up_b: engine.buf_f32(&vec![0f32; config.intermediate_size], "up_b_zero"),
                down_w: gguf_matmul(engine, &mut reader, &format!("{p}.ffn_down.weight"))?,
                down_b: engine.buf_f32(&vec![0f32; config.hidden_size], "down_b_zero"),
            });
        }

        let out_norm = gguf_f32(engine, &mut reader, "output_norm.weight")?;
        let lm_head = gguf_matmul(engine, &mut reader, "output.weight")?;
        let zero_bias_vocab = engine.buf_f32(&vec![0f32; config.vocab_size], "zero_bias_vocab");

        // `reader` (and its underlying `File`) drops here — two-phase
        // loading: no raw GGUF bytes remain in memory past this point,
        // only the GPU-resident buffers built above.
        drop(reader);

        Ok(GpuModel { config, embed, layers, out_norm, lm_head, zero_bias_vocab, fast_kernels, pool: Pool::new(engine.device.clone(), engine.queue.clone()) })
    }
}

/// RoPE cos/sin tables for absolute positions `[0, max_pos)`, `half =
/// head_dim/2` columns each. `inv_freq[j] = theta^(-2j/head_dim)` — HF
/// Qwen2RotaryEmbedding's convention (`gen_fixture.py`'s reference reads the
/// same `qwen2.rope.freq_base` metadata key).
pub fn build_rope_tables(head_dim: usize, theta: f32, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|j| theta.powf(-2.0 * j as f32 / head_dim as f32)).collect();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for j in 0..half {
            let angle = pos as f32 * inv_freq[j];
            cos[pos * half + j] = angle.cos();
            sin[pos * half + j] = angle.sin();
        }
    }
    (cos, sin)
}

#[allow(clippy::too_many_arguments)]
fn linear(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, in_dim: u32, w: &MatMulWeight, b: &wgpu::Buffer, out_dim: u32, fast: bool) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * out_dim) as usize);
    let wgs = (out_dim.div_ceil(16), rows.div_ceil(16), 1);
    match w {
        MatMulWeight::F32 { w } => {
            let dims = pool.uniform(&format!("{key}.dims"), LinearDims { m: rows, k: in_dim, n: out_dim, act: 0 });
            let bg = pool.bind_group(
                key,
                &engine.linear,
                &[
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
                ],
            );
            engine.dispatch(encoder, &engine.linear, &bg, wgs, key);
        }
        MatMulWeight::Q8_0 { qs, scales, blocks_per_row } => {
            // Q8_0 is only used for the lm head (`output.weight`) in this
            // model — one call per forward, not worth a tiled/coalesced
            // port yet (see the slice-2 report's "did not port" list).
            let dims = pool.uniform(&format!("{key}.dims"), LinearQDims { m: rows, k: in_dim, n: out_dim, act: 0, blocks_per_row: *blocks_per_row, _p0: 0, _p1: 0, _p2: 0 });
            let bg = pool.bind_group(
                key,
                &engine.linear_q8,
                &[
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: qs.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: scales.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
                ],
            );
            engine.dispatch(encoder, &engine.linear_q8, &bg, wgs, key);
        }
        MatMulWeight::Q4_0 { qs, scales, blocks_per_row } => {
            let dims = pool.uniform(&format!("{key}.dims"), LinearQDims { m: rows, k: in_dim, n: out_dim, act: 0, blocks_per_row: *blocks_per_row, _p0: 0, _p1: 0, _p2: 0 });
            let entries = [
                BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: qs.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: scales.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
            ];
            if !fast {
                // Reference/naive path — always linear_q4.wgsl regardless
                // of `rows`, kept for the fixture-parity gate and as the
                // fallback when a fast kernel's correctness is in doubt.
                let bg = pool.bind_group(key, &engine.linear_q4, &entries);
                engine.dispatch(encoder, &engine.linear_q4, &bg, wgs, key);
            } else if rows > 1 {
                // Prefill: tiled matmul (llm-wasm's shader_q4_tiled.wgsl port).
                let bg = pool.bind_group(&format!("{key}.tiled"), &engine.linear_q4_tiled, &entries);
                engine.dispatch(encoder, &engine.linear_q4_tiled, &bg, (out_dim.div_ceil(64), rows.div_ceil(64), 1), key);
            } else {
                // Decode: coalesced matvec (llm-wasm's shader_q4_matvec_coalesced.wgsl port).
                let bg = pool.bind_group(&format!("{key}.decode"), &engine.linear_q4_decode, &entries);
                engine.dispatch(encoder, &engine.linear_q4_decode, &bg, (out_dim.div_ceil(4), 1, 1), key);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, scale: &wgpu::Buffer, rows: u32, dim: u32, eps: f32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * dim) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), RmsDims { rows, dim, eps, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rmsnorm,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm, &bg, (rows.div_ceil(64), 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn rope(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, buf: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer, rows: u32, heads: u32, head_dim: u32, pos_base: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), RopeDims { rows, heads, head_dim, pos_base });
    let bg = pool.bind_group(
        key,
        &engine.rope,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(encoder, &engine.rope, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

fn add_inplace(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), AddDims { len, _p0: 0, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.add_inplace,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.add_inplace, &bg, (len.div_ceil(256), 1, 1), key);
}

#[allow(clippy::too_many_arguments)]
fn silu_mul(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, gate: &wgpu::Buffer, up: &wgpu::Buffer, rows: u32, hidden: u32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * hidden) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), SiluDims { rows, hidden, _p0: 0, _p1: 0 });
    let bg = pool.bind_group(
        key,
        &engine.silu_mul,
        &[
            BindGroupEntry { binding: 0, resource: gate.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: up.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.silu_mul, &bg, ((rows * hidden).div_ceil(256), 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn qkv_proj(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, layer: &LayerWeights, fast: bool) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let hidden = cfg.hidden_size as u32;
    let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u32;
    let q = linear(engine, pool, encoder, &format!("{key}.q"), x, rows, hidden, &layer.q_w, &layer.q_b, hidden, fast);
    let k = linear(engine, pool, encoder, &format!("{key}.k"), x, rows, hidden, &layer.k_w, &layer.k_b, kv_dim, fast);
    let v = linear(engine, pool, encoder, &format!("{key}.v"), x, rows, hidden, &layer.v_w, &layer.v_b, kv_dim, fast);
    (q, k, v)
}

#[allow(clippy::too_many_arguments)]
fn mlp(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, layer: &LayerWeights, fast: bool) -> wgpu::Buffer {
    let hidden = cfg.hidden_size as u32;
    let inter = cfg.intermediate_size as u32;
    let gate = linear(engine, pool, encoder, &format!("{key}.gate"), x, rows, hidden, &layer.gate_w, &layer.gate_b, inter, fast);
    let up = linear(engine, pool, encoder, &format!("{key}.up"), x, rows, hidden, &layer.up_w, &layer.up_b, inter, fast);
    let gated = silu_mul(engine, pool, encoder, &format!("{key}.silu"), &gate, &up, rows, inter);
    linear(engine, pool, encoder, &format!("{key}.down"), &gated, rows, inter, &layer.down_w, &layer.down_b, hidden, fast)
}

fn embed_gather(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, model: &GpuModel, token_ids: &[u32]) -> wgpu::Buffer {
    let rows = token_ids.len() as u32;
    let hidden = model.config.hidden_size as u32;
    let ids_buf = pool.upload_u32("embed.ids", token_ids);
    let out = pool.data("embed.out", (rows * hidden) as usize);
    let dims = pool.uniform("embed.dims", GatherDims { rows, hidden, blocks_per_row: model.embed.blocks_per_row, _p0: 0 });
    let bg = pool.bind_group(
        "embed",
        &engine.embed_gather_q4,
        &[
            BindGroupEntry { binding: 0, resource: ids_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: model.embed.qs.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: model.embed.scales.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.embed_gather_q4, &bg, ((rows * hidden).div_ceil(256), 1, 1), "embed");
    out
}

/// GPU-side scatter of `rows` freshly-computed K/V rows
/// (`[rows, kv_heads, head_dim]`, row-major) into the cache's
/// `[kv_head, kv_base+row, head_dim]` layout, via one `copy_buffer_to_buffer`
/// per (row, kv_head) recorded into the same encoder as the dispatch that
/// produced `src` — no CPU readback, so this can run before the attention
/// dispatch that needs to see it (decode's self-attention).
fn scatter_kv_gpu(encoder: &mut wgpu::CommandEncoder, cache_buf: &wgpu::Buffer, src: &wgpu::Buffer, rows: u32, cfg: &Qwen2Config, kv_base: u32, max_ctx: u32) {
    let kv_heads = cfg.num_kv_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let row_bytes = (head_dim * 4) as u64;
    for row in 0..rows {
        for h in 0..kv_heads {
            let src_off = (((row * kv_heads + h) * head_dim) as u64) * 4;
            let dst_pos = kv_base + row;
            let dst_off = (((h * max_ctx + dst_pos) * head_dim) as u64) * 4;
            encoder.copy_buffer_to_buffer(src, src_off, cache_buf, dst_off, row_bytes);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_prefill(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, seq: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let hidden = cfg.hidden_size as u32;
    let out = pool.data(&format!("{key}.out"), (seq * hidden) as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{key}.dims"),
        AttnPrefillDims { seq, n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, scale, _p0: 0, _p1: 0, _p2: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_prefill,
        &[
            BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: k.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.attn_prefill, &bg, (cfg.num_heads as u32, 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn attn_decode(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, q: &wgpu::Buffer, k_cache: &wgpu::Buffer, v_cache: &wgpu::Buffer, kv_len: u32, max_ctx: u32, cfg: &Qwen2Config) -> wgpu::Buffer {
    let hidden = cfg.hidden_size as u32;
    let out = pool.data(&format!("{key}.out"), hidden as usize);
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let dims = pool.uniform(
        &format!("{key}.dims"),
        AttnDecodeDims { n_heads: cfg.num_heads as u32, n_kv_heads: cfg.num_kv_heads as u32, head_dim: cfg.head_dim as u32, kv_len, max_ctx, scale, _p0: 0, _p1: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.attn_decode,
        &[
            BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.attn_decode, &bg, (cfg.num_heads as u32, 1, 1), key);
    out
}

/// Prefill: runs every layer over the whole prompt with causal attention
/// (no cache read needed — attends directly over this call's own q/k/v),
/// GPU-scatters every position's K/V into `cache`, and returns the last
/// position's logits ([vocab]).
pub async fn forward_prefill(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_ids: &[u32], cos: &wgpu::Buffer, sin: &wgpu::Buffer) -> Vec<f32> {
    let cfg = &model.config;
    let seq = token_ids.len() as u32;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill") });
    let x = embed_gather(engine, pool, &mut encoder, model, token_ids);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("layer{i}");
        let normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.norm"), &x, &layer.attn_norm, seq, hidden, cfg.rms_norm_eps);
        let (q, k, v) = qkv_proj(engine, pool, &mut encoder, &format!("{key}.qkv"), &normed, seq, cfg, layer, model.fast_kernels);
        rope(engine, pool, &mut encoder, &format!("{key}.ropeq"), &q, cos, sin, seq, cfg.num_heads as u32, cfg.head_dim as u32, 0);
        rope(engine, pool, &mut encoder, &format!("{key}.ropek"), &k, cos, sin, seq, cfg.num_kv_heads as u32, cfg.head_dim as u32, 0);
        let attn_out = attn_prefill(engine, pool, &mut encoder, &format!("{key}.attn"), &q, &k, &v, seq, cfg);
        scatter_kv_gpu(&mut encoder, &cache.k[i], &k, seq, cfg, 0, cache.max_ctx);
        scatter_kv_gpu(&mut encoder, &cache.v[i], &v, seq, cfg, 0, cache.max_ctx);

        let o = linear(engine, pool, &mut encoder, &format!("{key}.wo"), &attn_out, seq, hidden, &layer.o_w, &layer.o_b, hidden, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add1"), &x, &o, seq * hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, seq, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, &mut encoder, &format!("{key}.mlp"), &ffn_normed, seq, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add2"), &x, &mlp_out, seq * hidden);
    }

    let normed_final = rmsnorm(engine, pool, &mut encoder, "out_norm", &x, &model.out_norm, seq, hidden, cfg.rms_norm_eps);
    let logits = linear(engine, pool, &mut encoder, "lm_head", &normed_final, seq, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, false);

    engine.queue.submit(Some(encoder.finish()));
    let vocab = cfg.vocab_size;
    let all_logits = engine.read_buffer(&logits, (seq as usize) * vocab).await;
    cache.kv_len = seq;
    all_logits[(seq as usize - 1) * vocab..].to_vec()
}

/// Decode one token against the cache (already populated up to
/// `cache.kv_len`): embed, run every layer (each layer GPU-scatters this
/// step's K/V into the cache at `cache.kv_len` before its own attention
/// dispatch, so the step attends to itself too), return logits ([vocab]).
pub async fn forward_decode_step(engine: &Engine, model: &GpuModel, cache: &mut KvCache, token_id: u32, cos: &wgpu::Buffer, sin: &wgpu::Buffer) -> Vec<f32> {
    let cfg = &model.config;
    let hidden = cfg.hidden_size as u32;
    let pool = &model.pool;
    let pos = cache.kv_len;

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("decode") });
    let x = embed_gather(engine, pool, &mut encoder, model, &[token_id]);

    for (i, layer) in model.layers.iter().enumerate() {
        let key = format!("dec_layer{i}");
        let normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.norm"), &x, &layer.attn_norm, 1, hidden, cfg.rms_norm_eps);
        let (q, k, v) = qkv_proj(engine, pool, &mut encoder, &format!("{key}.qkv"), &normed, 1, cfg, layer, model.fast_kernels);
        rope(engine, pool, &mut encoder, &format!("{key}.ropeq"), &q, cos, sin, 1, cfg.num_heads as u32, cfg.head_dim as u32, pos);
        rope(engine, pool, &mut encoder, &format!("{key}.ropek"), &k, cos, sin, 1, cfg.num_kv_heads as u32, cfg.head_dim as u32, pos);

        scatter_kv_gpu(&mut encoder, &cache.k[i], &k, 1, cfg, pos, cache.max_ctx);
        scatter_kv_gpu(&mut encoder, &cache.v[i], &v, 1, cfg, pos, cache.max_ctx);
        let attn_out = attn_decode(engine, pool, &mut encoder, &format!("{key}.attn"), &q, &cache.k[i], &cache.v[i], pos + 1, cache.max_ctx, cfg);

        let o = linear(engine, pool, &mut encoder, &format!("{key}.wo"), &attn_out, 1, hidden, &layer.o_w, &layer.o_b, hidden, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add1"), &x, &o, hidden);

        let ffn_normed = rmsnorm(engine, pool, &mut encoder, &format!("{key}.ffnnorm"), &x, &layer.ffn_norm, 1, hidden, cfg.rms_norm_eps);
        let mlp_out = mlp(engine, pool, &mut encoder, &format!("{key}.mlp"), &ffn_normed, 1, cfg, layer, model.fast_kernels);
        add_inplace(engine, pool, &mut encoder, &format!("{key}.add2"), &x, &mlp_out, hidden);
    }

    let normed_final = rmsnorm(engine, pool, &mut encoder, "dec_out_norm", &x, &model.out_norm, 1, hidden, cfg.rms_norm_eps);
    let logits = linear(engine, pool, &mut encoder, "dec_lm_head", &normed_final, 1, hidden, &model.lm_head, &model.zero_bias_vocab, cfg.vocab_size as u32, false);

    engine.queue.submit(Some(encoder.finish()));
    cache.kv_len = pos + 1;
    engine.read_buffer(&logits, cfg.vocab_size).await
}
