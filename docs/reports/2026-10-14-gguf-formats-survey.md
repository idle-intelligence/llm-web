# GGUF quant formats in the wild: what small models actually ship, and what lean needs

Date: 2026-10-14. Scope: GGUF models up to about 8B parameters (LLM text models), plus what the browser engines load.
All Hub numbers were pulled from the Hugging Face API on 2026-10-14; "downloads" is the API's 30-day figure unless stated. Throwaway data and header-parsing scripts are in /tmp/gg (not part of the repo).

## Summary

- Per-file download counts are not exposed by the Hub (the API gives per-repo counts only). So "most downloaded variant" can be measured directly only where a repo holds exactly one GGUF file. What can be measured for every repo is which variants it ships.
- Shipping: among 237 multi-quant small-model repos (of the top 1000 GGUF repos by 30-day downloads), 93% ship Q4_K_M, 94% Q8_0, 85% Q6_K, 78% Q5_K_M. Q4_0 is shipped by 46%, Q5_0 by 15%, Q4_1 by 26%; Q5_1 is named in only 14 of 282 repos.
- Measured downloads (single-file repos): Q4_K_M 1.72M, Q4_0 1.12M (almost all Google's own Gemma 4 QAT q4_0 repos), Q8_0 0.83M, F16 0.26M, Q6_K 0.07M, Q5_K_M 0.03M. K-quant Q4_K_M is the default 4-bit file; Q4_0 matters only where the model publisher made it the primary release.
- A file named "Q4_K_M" is a mix, not one type. For models whose hidden size is not a multiple of 256 (SmolLM2-135M/360M, Qwen2.5-0.5B, Gemma-3-1B), llama.cpp falls back per tensor to Q5_0 (and Q8_0 in place of Q6_K). Verified from real headers: SmolLM2-360M Q4_K_M = 176 Q5_0 + 16 Q4_K + 16 Q6_K + 17 Q8_0.
- Surprising finding for lean: lean's current set (F32, F16, Q4_0, Q4_1, Q8_0, Q6_K) loads 0 of the 220 Q4_K_M files in the sample. Adding Q4_K alone makes 220/237 repos' Q4_K_M loadable for models with hidden size divisible by 256 (Qwen3 0.6B/1.7B/4B/8B, Llama-3.2 1B/3B, Qwen2.5-1.5B/3B, Phi-4-mini). Adding Q5_0 covers the odd-dimension small models.
- Qwen's own repos: Qwen3-0.6B and 1.7B ship only Q8_0; Qwen3-4B/8B ship Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0; Qwen2.5 ships Q2_K to Q8_0 including Q4_0 and Q5_0. ggml-org/Qwen3-0.6B-GGUF ships Q4_0 + Q8_0 + BF16 + F16, all of which lean already (mostly) loads.

## 1. What the Hub ships (opened 2026-10-14)

### 1.1 Method and limits
- Source: `https://huggingface.co/api/models?filter=gguf&sort=downloads&limit=1000&expand[]=gguf&expand[]=downloads&expand[]=downloadsAllTime`. Note `library=gguf` is ignored by the API (it returns the all-models ranking, MiniLM first); `filter=gguf` is the working form.
- Reduced to repos with `gguf.total` between 0.05B and 8.6B and a text-LLM architecture, minus names containing 27B/30B/35B etc.: 282 repos, 42.9M of the 100.5M downloads of that size class. `gguf.total` is taken from one file of the repo, so multi-size repos and mmproj-labelled repos are noisy. Per-repo trees fetched with `/api/models/<repo>/tree/main?recursive=true`; quant parsed from file names (regex), mmproj files dropped.
- 237 repos ship 3 or more quant variants ("ladder repos"); 45 ship fewer.
- The top-1000 is dominated this month by very new families (Qwen3.5/3.8, Gemma 4, Ornith, Bonsai); older families appear in the family searches below.

### 1.2 Variants shipped, across the 237 ladder repos
Percent of repos shipping the variant; last column is the same weighted by repo downloads (a repo-level weight, not a per-file one).

| Variant | repos | % repos | % dl-weighted |
|---|---|---|---|
| Q8_0 | 223 | 94.1 | 91.7 |
| Q4_K_M | 220 | 92.8 | 93.5 |
| Q6_K | 202 | 85.2 | 84.8 |
| Q5_K_M | 184 | 77.6 | 71.9 |
| Q3_K_M | 145 | 61.2 | 59.4 |
| Q2_K | 134 | 56.5 | 43.1 |
| Q5_K_S | 132 | 55.7 | 41.6 |
| Q4_K_S | 128 | 54.0 | 40.7 |
| Q4_0 | 109 | 46.0 | 44.9 |
| IQ4_XS | 104 | 43.9 | 38.3 |
| F16 | 88 | 37.1 | 35.4 |
| BF16 | 76 | 32.1 | 35.0 |
| Q4_1 | 62 | 26.2 | 22.8 |
| IQ4_NL | 60 | 25.3 | 22.7 |
| Q5_0 | 36 | 15.2 | 13.0 |
| UD-Q4_K_XL (unsloth) | 32 | 13.5 | 17.9 |
| UD-IQ2_M etc. | 27 | 11.4 | 15.5 |

Q5_1 is shipped as a named file by only 14 of the 282 sampled repos (it also appears inside files, see section 2). Note that BF16 is not in lean's loadable set either.

### 1.3 Measured downloads where a repo has one file
Repos with exactly one non-mmproj GGUF in the 282-repo sample (22 repos), 30-day downloads summed by quant:

| Quant | repos | downloads (30d) | notes |
|---|---|---|---|
| Q4_K_M | 9 | 1,718,365 | HyperCLOVAX-SEED 1.5B (1.10M), Agents-A1-4B, nl2sh-1.5b, Llama-3.2 1B/3B by hugging-quants |
| Q4_0 | 3 | 1,116,002 | google/gemma-4-E4B-it-qat-q4_0-gguf 771k, gemma-4-E2B-it-qat-q4_0-gguf 314k |
| Q8_0 | 5 | 833,435 | Qwen/Qwen3-0.6B-GGUF 318k, Qwen3-1.7B-GGUF 151k, Qwen3-Reranker-0.6B 99k |
| F16 | 4 | 259,579 | |
| Q6_K | 1 | 65,178 | |
| Q5_K_M | 1 | 30,710 | |

Second proxy: repos whose name ends in the quant (gguf-my-repo style, search `<QUANT>-GGUF`, up to 300 hits each, `gguf.total` <= 8.6B). 30-day / all-time downloads:
Q4_K_M 172 repos, 1.76M / 10.28M; Q4_0 56 repos, 1.12M / 4.31M; Q8_0 177 repos, 0.54M / 11.0M; Q5_K_M 138 repos, 27k / 222k; Q6_K 143 repos, 18k / 149k; Q2_K 154 repos, 15k / 147k; IQ4_NL 244 repos, 8k / 85k; Q3_K_M 113 repos, 7k / 61k; IQ4_XS 144 repos, 7k / 63k; Q5_0 95 repos, 4k / 34k; Q5_K_S 76 repos, 3.6k / 38k; Q3_K_S 89 repos, 2.8k / 28k; Q4_K_S 23 repos, 2.6k / 25k; Q5_1 1 repo, 32 / 189; Q4_1 no repos; F16 218 repos, 17k / 547k; BF16 53 repos, 36k / 164k.
Ranking by downloads: Q4_K_M, Q8_0 and Q4_0 are two to three orders of magnitude above every other type. Caveat: this proxy only sees self-quantized single-file repos; the Q4_0 figure is dominated by Google's two Gemma 4 QAT repos; the all-time Q8_0 figure includes embedding and reranker models.

### 1.4 By publisher (file lists opened)
- Qwen official (HF `Qwen/*-GGUF`): Qwen3-8B and Qwen3-4B: Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0. Qwen3-1.7B and 0.6B: Q8_0 only. Qwen2.5-0.5B/1.5B/3B-Instruct: Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0 (3B also fp16 split). Qwen2.5-Coder-7B: same ladder, files split into shards for the bigger ones.
- ggml-org: Qwen3-0.6B-GGUF: Q4_0, Q8_0, BF16, f16 (no K-quants). gemma-3-1b-it-GGUF: Q4_K_M, Q8_0 (seen in tree). 212k 30d downloads for gemma-3-1b-it.
- unsloth: widest ladder (Qwen3-4B-GGUF: 26 files): Q2_K ... Q8_0, IQ4_NL, IQ4_XS, Q4_0, Q4_1, UD-IQ1_S through UD-Q8_K_XL, BF16. Qwen3.5-4B-GGUF has 1.22M 30d downloads.
- bartowski: Llama-3.2-1B/3B and SmolLM2: IQ3_M/IQ3_XS/IQ4_XS, Q2_K to Q8_0 with `_L`/`_XL` variants (embedding at Q8_0), Q4_0, plus the older Q4_0_4_4/4_8/8_8 repack files, and f16.
- lmstudio-community: Llama-3.2-3B: Q3_K_L, Q4_K_M, Q6_K, Q8_0 (a deliberately short ladder).
- HuggingFaceTB (SmolLM2 official): SmolLM2-360M-Instruct-GGUF ships only q8_0.
- Google (Gemma 4 QAT): single file `q4_0` (see section 2).
- TheBloke (legacy, TinyLlama-1.1B-Chat): Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_0, Q5_K_S/M, Q6_K, Q8_0. 80k 30d downloads, 2.4M all-time (pre-IQ era, so no IQ files).
- LiquidAI LFM2.5-1.2B: BF16, F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0.
- MaziyarPanahi (large all-time counts, e.g. Llama-3.2-1B 12.2M all-time): Q2_K to Q6_K + fp16.

### 1.5 Family download snapshots (repo level, 30d / all-time)
- SmolLM2: unsloth 135M 62k / 565k; bartowski 135M 47k / 219k; HuggingFaceTB 360M 16k / 237k; bartowski 1.7B 12k / 821k.
- Llama-3.2: bartowski 3B 166k / 4.87M; bartowski 1B 157k / 2.90M; MaziyarPanahi 1B 127k / 12.2M; MaziyarPanahi 3B 119k / 10.0M.
- Phi: MaziyarPanahi Phi-4-mini 120k / 3.9M; unsloth Phi-4-mini 111k / 674k; Phi-3.5-mini MaziyarPanahi 126k / 12.4M; microsoft Phi-3-mini-4k-instruct-gguf 43k / 1.37M.
- Qwen: Qwen/Qwen3-8B-GGUF 555k; Qwen3-4B-GGUF 539k; unsloth Qwen3-4B 491k; Qwen/Qwen2.5-1.5B-Instruct-GGUF 324k; Qwen3-0.6B-GGUF 318k; Qwen2.5-3B-Instruct-GGUF 210k; Qwen2.5-0.5B-Instruct-GGUF 269k.
- Gemma: ggml-org gemma-3-1b-it 212k; MaziyarPanahi gemma-3-4b-it 125k / 3.6M; Google gemma-4-E4B QAT q4_0 771k, E2B 314k.

## 2. Tensor-type mix per variant

### 2.1 Rules (llama.cpp src/llama-quant.cpp at commit 097cf33d9499b1eade67d51e16d7c75427e7b267, 2026-10-12)
Permalink base: https://github.com/ggml-org/llama.cpp/blob/097cf33d9499b1eade67d51e16d7c75427e7b267/src/llama-quant.cpp
- Default type per file type, `llama_ftype_get_default_type`, lines 886-930: Q4_K_S/Q4_K_M -> Q4_K; Q5_K_S/M -> Q5_K; Q6_K -> Q6_K; Q3_K_* -> Q3_K; Q2_K -> Q2_K; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 -> same.
- Per-tensor override `llama_tensor_get_type_impl`, lines 449-715:
  - Output / tied token_embd (lines 497-516): Q8_0 if the arch is Falcon or the row length is not divisible by the block size; the IQ1/IQ2/IQ3_XXS files get Q5_K; otherwise anything not Q8_0 becomes Q6_K. In practice: output (or tied token_embd) = Q6_K for most K-quant files.
  - attn_v (lines 573-598, `use_more_bits` defined at 457): Q4_K_M and Q5_K_M use Q6_K when `i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`, i.e. about half the layers; Q4_K_S uses Q5_K for the first 4; Q3_K_M uses Q5_K for the first 2 then Q4_K; Q2_K uses Q4_K (GQA >= 4) or Q3_K.
  - ffn_down (lines 630-672): same `use_more_bits` Q6_K rule for Q4_K_M and Q5_K_M; Q3_K_M mixes Q5_K/Q4_K/Q3_K; Q2_K gives Q3_K; Q4_K_S first n/8 layers Q5_K; Q4_0/Q5_0 with an imatrix first n/8 layers use Q4_1/Q5_1 (lines 666-671).
  - attn_qkv (fused) (lines 693-699): Q4_K_M gives Q5_K, Q5_K_M gives Q6_K (this is why Phi-4-mini Q4_K_M contains Q5_K).
  - attn_output, ffn_gate, ffn_up stay at the default except for IQ3/Q3_K_M/L and Q2_K cases.
- Shape fallback `tensor_type_fallback`, lines 395-440, applied at line 774 after the type is chosen: if the row length (ne[0], the input dim) is not divisible by 256, Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0, Q2_K/Q3_K -> Q4_0, IQ4_XS and the other IQ types -> IQ4_NL; if still not divisible by 32, F16.
- Norms, biases and other 1-D tensors stay F32.

### 2.2 Verified against real file headers
GGUF headers read by HTTP range request (first 24 MB; metadata and tensor-info table only, no weights run). Type counts are tensors per file.

| File | hidden | Tensor types |
|---|---|---|
| bartowski SmolLM2-360M Q4_K_M | 960 | Q5_0 x176, Q4_K x16, Q6_K x16, Q8_0 x17 (16 attn_v + token_embd), F32 x65. ffn_down: Q4_K 16 + Q6_K 16; q/k/o/gate/up all Q5_0; attn_v 16 Q8_0 + 16 Q5_0 |
| bartowski SmolLM2-360M Q5_K_M | 960 | Q5_1 x176, Q5_K x16, Q6_K x16, Q8_0 x17 |
| bartowski SmolLM2-360M Q4_0 | 960 | Q4_0 x220, Q4_1 x4, Q8_0 x1 (token_embd), F32 |
| bartowski SmolLM2-360M IQ4_XS | 960 | IQ4_NL x192, IQ4_XS x32, Q8_0 x1 |
| bartowski SmolLM2-360M Q8_0 | 960 | Q8_0 x225 |
| Qwen2.5-0.5B-Instruct Q4_K_M (official) | 896 | Q5_0 x133, Q8_0 x13 (output.weight and 12 attn_v), Q6_K x12, Q4_K x12 (ffn_down); token_embd is Q5_0 |
| gemma-3-1b-it Q4_K_M (ggml-org) | 1152 | Q5_0 x117, Q4_K x39, Q6_K x13, Q8_0 x14 |
| Qwen3-0.6B Q4_K_M (unsloth) | 1024 | Q4_K x168, Q6_K x29 (token_embd + 14 attn_v + 14 ffn_down); tied, no output.weight |
| Qwen2.5-1.5B-Instruct Q4_K_M (official) | 1536 | Q4_K x169 (incl. token_embd), Q6_K x29 (output + attn_v/ffn_down halves) |
| Llama-3.2-1B Q4_K_M (bartowski) | 2048 | Q4_K x96, Q6_K x17 |
| Llama-3.2-3B Q4_K_M (bartowski) | 3072 | Q4_K x168, Q6_K x29 |
| Phi-4-mini Q4_K_M (bartowski) | 3072 | Q4_K x80, Q5_K x32 (fused qkv), Q6_K x17 |
| Qwen3-4B Q4_K_M (official and unsloth) | 2560 | Q4_K x216, Q6_K x37 |
| Qwen3-8B Q4_K_M (official) | 4096 | Q4_K x217 (incl. token_embd), Q6_K x37 (incl. output.weight) |
| Qwen3-8B Q5_K_M (official) | 4096 | Q5_K x217, Q6_K x37 |
| Qwen3-4B Q5_0 (official) | 2560 | Q5_0 x252, Q6_K x1 (token_embd) |
| Qwen3-4B Q4_0 (unsloth) | 2560 | Q4_0 x248, Q4_1 x4, Q6_K x1 |
| Llama-3.2-1B Q4_0 (bartowski) | 2048 | Q4_0 x110, Q4_1 x2, Q6_K x1 |
| Qwen3-4B Q8_0 | 2560 | Q8_0 x253 |
| Qwen3-4B Q4_K_S (unsloth) | 2560 | Q4_K x244, Q5_K x8, Q6_K x1 |
| Qwen3-4B IQ4_XS (unsloth) | 2560 | IQ4_XS x216, Q5_K x36, Q6_K x1 |
| Qwen3-4B Q2_K (unsloth) | 2560 | Q2_K x144, Q3_K x72, Q4_K x36, Q6_K x1 |
| Qwen3-4B UD-Q4_K_XL (unsloth) | 2560 | Q4_K x154, Q6_K x49, Q5_K x30, IQ4_XS x20 |
| Qwen3-4B UD-Q2_K_XL (unsloth) | 2560 | Q2_K 104, Q3_K 93, IQ3_S 15, IQ3_XXS 10, IQ2_XS 10, IQ2_S 10, Q4_K 5, IQ4_XS 5, Q6_K 1 |
| Gemma 4 E2B QAT q4_0 (Google) | 1536 | Q4_0 x275, Q6_K x2, F16 x1, F32 |

Reading: the 16 Q6_K + 16 Q4_K in the bartowski SmolLM2-360M Q4_K_M file are the ffn_down split of 32 layers (4 + 4 + 8 = 16 layers satisfy `use_more_bits`); the 176 Q5_0 are every 960-wide-input matrix that fell back from Q4_K; the 16 Q8_0 attn_v are the Q6_K layers that fell back.

### 2.3 Kernel requirement per variant (from the above; "odd dims" = hidden not divisible by 256)
| Variant | Needs, hidden % 256 == 0 | Needs, odd dims |
|---|---|---|
| Q8_0 | Q8_0 | Q8_0 |
| Q4_0 | Q4_0, Q4_1 (a few ffn_down), Q6_K or Q8_0 (embedding) | same |
| Q4_K_M | Q4_K, Q6_K | Q4_K (ffn_down only), Q5_0, Q6_K, Q8_0 |
| Q4_K_S | Q4_K, Q5_K, Q6_K | plus Q5_0/Q5_1 (inferred from the fallback table, not opened) |
| Q5_K_M / Q5_K_S | Q5_K, Q6_K | Q5_1, Q5_K, Q6_K, Q8_0 |
| Q5_0 | Q5_0, Q6_K | Q5_0, Q8_0 |
| Q6_K | Q6_K (plus Q8_0 embedding in some) | Q8_0 |
| Q3_K_* / Q2_K | Q3_K, Q4_K, Q5_K, Q6_K (Q2_K adds Q2_K) | plus Q4_0 fallbacks |
| IQ4_XS / IQ4_NL | IQ4_XS or IQ4_NL, Q5_K, Q6_K | IQ4_NL, Q8_0 |
| UD-*_XL (unsloth) | Q4_K/Q5_K/Q6_K plus IQ4_XS or lower IQ types | not opened |
| Q4_0_4_4 / _4_8 / _8_8 (bartowski) | ARM-repacked Q4_0 layouts, not standard Q4_0 | unverified |
| Gemma 4 QAT q4_0 | Q4_0, Q6_K, F16 | n/a |

Hidden sizes checked from headers: not divisible by 256: SmolLM2-360M (960), Qwen2.5-0.5B (896), Gemma-3-1B (1152). Divisible: Qwen3 0.6B (1024), 4B (2560), 8B (4096), Qwen2.5-1.5B (1536), Llama-3.2 (2048, 3072), Phi-4-mini (3072), Gemma-4-E2B (1536). SmolLM2-135M (576) and Qwen3-1.7B / Qwen2.5-3B (2048) were not opened; 576 is not divisible by 256 (arithmetic), 2048 is.

## 3. Browser engines

| Engine | Format | Default / catalogue | Evidence |
|---|---|---|---|
| WebLLM (MLC) | Own MLC weights + compiled WebGPU kernels, not GGUF | `prebuiltAppConfig` has 169 model entries: q4f16_1 x77, q4f32_1 x69, q0f32 x11, q0f16 x8. Catalogue includes Qwen3 0.6B/1.7B/4B/8B, Llama-3.2 1B/3B, SmolLM2 135M/360M/1.7B, Phi-3.5/Phi-4-mini, Gemma 2/3/4 | mlc-ai/web-llm main (c42c366, 2026-10-11) `src/config.ts`, counted by `model_id` |
| transformers.js | ONNX Runtime Web; dtype enum fp32, fp16, q8, int8, uint8, q4, bnb4, q4f16, plus q2/q2f16/q1/q1f16; WASM default q8, otherwise fp32 | e.g. onnx-community/Qwen3-0.6B-ONNX ships model.onnx (fp32, 315 MB + 2.09 GB data), fp16 1.20 GB, q4 919 MB, q4f16 570 MB, int8/uint8/quantized 618 MB, bnb4 892 MB | huggingface/transformers.js main (511bb61, 2026-10-06) `packages/transformers/src/utils/dtypes.js`; Hub tree of the ONNX repo, 2026-10-14 |
| wllama | llama.cpp compiled to wasm, any GGUF; v3 adds WebGPU (via llama.cpp's WebGPU backend), multimodal, tool calling | README recommends Q4, Q5, Q6 quants; says IQ quants (with imatrix) are not recommended (slow, low quality); supports split GGUF shards | ngxson/wllama master (1a5687e, 2026-10-11) README lines 90-108 |
| llama.cpp WebGPU backend (reference for a WGSL kernel set) | n/a | `ggml-webgpu.cpp` mul_mat supports F32, F16, BF16, Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, all IQ types, MXFP4, NVFP4 | ggml-org/llama.cpp @097cf33, `ggml/src/ggml-webgpu/ggml-webgpu.cpp` lines 4401-4422 and 4506-4527 |
| Ollama (not a browser engine, shows what users pull) | GGUF | qwen3 0.6b/1.7b tags: q4_K_M, q8_0, fp16; llama3.2 1b: q4_0, q4_K_M, q8_0, fp16 | ollama.com/library/qwen3/tags and llama3.2/tags, 2026-10-14 |

Which llama.cpp revision wllama's bundled wasm uses was not checked (unknown; would need the submodule pointer in wllama).

## 4. Options for lean (facts, then trade-offs)

### 4.1 Coverage by added kernel
Computed over the 2,939 (repo, variant) pairs in the 237 ladder repos. A pair counts as loadable when every tensor type in its requirement set (table 2.3, hidden % 256 == 0 column) is in lean's kernel set. 663 pairs (22.6%) are variants I did not model (IQ1/IQ2/IQ3, UD-*, BF16, MXFP4, Q4_0_x_x), which stay in the denominator as not loadable. First figure is unweighted share of pairs, second is weighted by repo downloads.

| Kernel set | pairs loadable | dl-weighted | repos with >=1 4-5 bit file loadable | repos with Q4_K_M loadable |
|---|---|---|---|---|
| Current lean (F32, F16, Q4_0, Q4_1, Q8_0, Q6_K) | 25.0% | 26.3% | 110 / 237 | 0 |
| + Q4_K | 33.7% | 35.5% | 232 / 237 | 220 |
| + Q5_0 | 34.9% | 36.6% | 232 / 237 | 220 (odd-dim models now also load) |
| + Q5_K | 51.2% | 51.3% | 232 / 237 | 220 |
| + Q3_K | 65.0% | 63.4% | | |
| + Q2_K | 71.4% | 68.7% | | |
| + Q5_1 | 71.9% | 69.0% | | |
| + IQ4_XS | 75.4% | 72.4% | | |
| + IQ4_NL | 77.4% | 74.5% | | |

Order matters for the middle rows (Q5_K only helps because Q4_K is already in); Q5_0 and Q5_1 gain little in the pair metric because the pair metric is computed for hidden % 256 == 0 and Q5_0 matters only for odd dimensions. Pair coverage is a poor proxy for demand: one file per repo is what a user downloads, and the measured single-file downloads (1.3) are almost entirely Q4_K_M, Q4_0 and Q8_0. By that evidence the one kernel with the highest value per kernel is Q4_K (it unlocks Q4_K_M for all models with hidden % 256 == 0), the next is Q5_0 (it unlocks Q4_K_M for the small odd-dimension models, and official Qwen Q5_0 files), and Q5_K plus Q5_1 round out Q5_K_M and Q4_K_S. Q4_K + Q5_0 + Q6_K + Q8_0 + Q4_0 is the 5-type set from the question; adding Q4_1 (already in lean) covers Q4_0 files, and Q5_K covers Q4_K_S, Q5_K_M and Phi's fused-qkv Q4_K_M.

Skip candidates by the evidence: IQ* (wllama itself recommends against; shipped by 44% of repos for IQ4_XS but measured downloads in the 1e3 to 1e4 range per 30 days against 1.7M for Q4_K_M), Q2_K and Q3_K_* (low-quality, 1e3 to 1e4 range), Q5_1 (named Q5_1 files exist in only 14 of 282 sampled repos; otherwise it appears as a fallback tensor inside odd-dim Q5_K_M files), BF16 files (can be converted when read, as F16 already is).

### 4.2 Alternative: publish lean-friendly GGUFs on the idle-intelligence HF org
Facts for the comparison:
- Demo models are few and chosen by the demo pages. Existing community files already match lean for several: ggml-org/Qwen3-0.6B-GGUF has Q4_0 and Q8_0; Qwen3-0.6B/1.7B official are Q8_0 only; Google's Gemma 4 QAT q4_0 is Q4_0 + Q6_K + F16.
- A pure Q4_0 produced by llama.cpp still contains a few Q4_1 and one Q6_K or Q8_0 embedding (section 2.2), all of which lean already loads. `llama-quantize` with `--pure` or `--tensor-type` overrides can force a single type per file (options exist in the source at lines 723-780; the exact CLI use was not tested).
- Pros: no new kernels, smaller wasm/WGSL surface, exact control of file layout and shard sizes for the 2 GB ArrayBuffer limit, tokenizer and chat template checked once, files hosted with the demo.
- Cons: users cannot drop in an arbitrary community GGUF (the most popular file, Q4_K_M, is rejected); Q4_0 is lower quality than Q4_K_M at similar size, and a Q8_0 file is about 1.9x larger than Q4_K_M; the org must keep re-publishing as new model families appear; a published file is a separate artifact to maintain and license-check per model.
- Both are compatible: a small kernel set for load-anything-common plus own files for the demos.

## Implementation notes
- A GGUF's per-tensor type is in the tensor-info table; a loader decides support per tensor, not per file name. The file-level `general.file_type` (15 = Q4_K_M, 17 = Q5_K_M, 2 = Q4_0, 7 = Q8_0, 8 = Q5_0, 14 = Q4_K_S, 10 = Q2_K, 30 = IQ4_XS) only names the recipe.
- Row-length (ne[0]) divisibility is what triggers the fallback, so ffn_gate/ffn_up and attn q/k/v/o (input dim = hidden) fall back, while ffn_down (input dim = ffn size) often does not. That is why odd-hidden models show a Q5_0 majority with a K-quant minority.
- Tied embeddings: token_embd is Q6_K (Qwen3-0.6B/4B, Llama), Q8_0 (SmolLM2, Gemma-3), or Q4_K/Q5_0 with a separate Q6_K/Q8_0 output.weight in some Qwen2.5 files; a loader needs to accept both token_embd and output.weight in each type.
- Split shards (`-00001-of-0000N.gguf`) are used by Qwen2.5-Coder-7B and fp16 files.

## Open questions
- Per-file downloads for multi-file repos: not available from the Hub API. Hub-side per-file counts, or an HF dataset of download logs, would resolve it.
- Hidden sizes of the Qwen3.5/3.8, Gemma 4, LFM2, Ornith families in the top-1000 were not checked; those archs might require tensor types or ops lean does not have regardless of quant (architecture support is a separate question).
- Whether `llama-quantize --pure` yields exactly one type including the embedding: the source suggests so (`!params->pure` guard at line 771) but no file was produced.
- Which llama.cpp revision wllama pins.
- Q4_K_S and UD-* odd-dim mixes were inferred from the fallback table or unopened, not read from headers.

## References
- llama-quant.cpp: https://github.com/ggml-org/llama.cpp/blob/097cf33d9499b1eade67d51e16d7c75427e7b267/src/llama-quant.cpp (lines 395-440 fallback, 449-715 per-tensor rules, 774 fallback call, 886-930 default types)
- llama.cpp WebGPU backend: https://github.com/ggml-org/llama.cpp/blob/097cf33d9499b1eade67d51e16d7c75427e7b267/ggml/src/ggml-webgpu/ggml-webgpu.cpp
- HF API queries as listed in 1.1; repos: https://huggingface.co/Qwen/Qwen3-4B-GGUF, https://huggingface.co/Qwen/Qwen3-0.6B-GGUF, https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF, https://huggingface.co/unsloth/Qwen3-4B-GGUF, https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF, https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF, https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf, https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF, https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX
- WebLLM config: https://github.com/mlc-ai/web-llm/blob/main/src/config.ts
- transformers.js dtypes: https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/utils/dtypes.js
- wllama: https://github.com/ngxson/wllama
- Ollama tags: https://ollama.com/library/qwen3/tags, https://ollama.com/library/llama3.2/tags
