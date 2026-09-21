# Model reference: xLAM-2-3b-fc-r (+ 8B candidates for later)

Sources: local checkout `~/Code/idle-intelligence/models/hf/xLAM-2-3b-fc-r/`
(config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
special_tokens_map.json, README.md, xlam_tool_call_parser.py) and
`~/Code/idle-intelligence/models/gguf/xlam-2-3b-fc-r/` (README.md,
`xLAM-2-3B-fc-r-Q4_0.gguf`, header parsed locally with the `gguf` pip
package). HF Hub API (`/api/models/<repo>?blobs=true`, `/raw/main/config.json`)
queried live for quant listings and 8B configs. `docs.rs/minijinja` queried
live for feature-flag semantics. `~/Code/idle-intelligence/trucs.ai/.claude/worktrees/sonos-mcp/sonos/{NOTES,PLAN}.md`
read for project context.

## 1. xLAM-2-3b-fc-r — base model, license, config

- Base: `Salesforce/xLAM-2-3b-fc-r`, architecture `Qwen2ForCausalLM`
  (README doesn't name the exact Qwen2.5 checkpoint fine-tuned from, but
  `model_type: qwen2`, hidden/layer/vocab numbers match the Qwen2.5-3B
  family; README's own long-context note points at "Qwen-2.5-based models").
- License: `cc-by-nc-4.0` (README frontmatter). README also states
  **research-purposes-only** release, and flags that any Llama-based sibling
  model in the family additionally carries the Meta Llama 3 Community
  License — irrelevant to this 3B (Qwen2-based) but relevant to the 8B
  Llama-xLAM candidate below.

| field | value |
|---|---|
| layers (`num_hidden_layers`) | 36 |
| attention heads | 16 |
| KV heads (GQA) | 2 |
| head dim | 2048/16 = 128 |
| hidden size | 2048 |
| intermediate size | 11008 |
| tied embeddings | `true` (config `tie_word_embeddings`; confirmed in GGUF — no `output.weight` tensor, see §2) |
| rope theta | 1,000,000.0 |
| rms norm eps | 1e-6 (config) / GGUF stores 9.999999974752427e-07, same value in f32 |
| max position embeddings | 32768 (config) — tokenizer_config `model_max_length` is 16384; GGUF `qwen2.context_length` is 32768 |
| bos/eos id (config) | bos 151643, eos 151645 |
| generation_config eos | `[151645, 151643]` — i.e. generation stops on **either** `<|im_end|>` (151645) or `<|endoftext|>` (151643, also the pad id) |
| pad id | 151643 (`<|endoftext|>`) |
| stop tokens in practice | `<|im_end|>` (turn end, chat-template delimiter) and `<|endoftext|>` (pad/eos fallback) |
| generation_config sampling defaults | `do_sample: true`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, `repetition_penalty: 1.05` |
| torch dtype | bfloat16 |

### Vocab: three different numbers, and why

- `config.json` `vocab_size`: **151936** — the padded embedding-matrix size.
- `tokenizer.json` BPE vocab dict: **151643** entries, plus a separate
  `added_tokens` list of **22** entries (special/control tokens such as
  `<|im_start|>`, `<|im_end|>`, vision placeholders, etc.) → **151665**
  tokens actually addressable by the tokenizer.
- GGUF `tokenizer.ggml.tokens` array (embedded, parsed from the file):
  **151936** entries — matches `config.json`, not the tokenizer's own
  151665.
- Gap: `151936 - 151665 = 271` unused embedding rows. This is the known
  Qwen2/2.5 pattern of padding vocab_size up (for hardware-friendly
  matmul tiling / reserved future special tokens) beyond what the shipped
  tokenizer uses. The GGUF `token_embd.weight` tensor has shape
  `[2048, 151936]` (see §2) — i.e. it carries rows for all 151936 ids, most
  of the trailing 271 presumably near-zero/untrained. The engine's
  tokenizer (tokenizer.json-driven) will never emit ids in the unused
  range, so this only matters for embedding-table sizing, not decode
  correctness.

## 2. GGUF

### Official Salesforce GGUF repo quants (`Salesforce/xLAM-2-3b-fc-r-gguf`, via HF API `blobs=true`)

| file | size |
|---|---|
| F16 | 6,178,316,864 |
| Q8_0 | 3,285,475,904 |
| Q6_K | 2,538,158,656 |
| Q5_K_M | 2,224,814,656 |
| Q5_0 | 2,169,666,112 |
| Q5_K_S | 2,169,666,112 |
| Q4_K_M | 1,929,902,656 |
| Q4_K_S | 1,834,383,936 |
| **Q4_0** | **1,822,849,600** |
| Q3_K_L | 1,707,391,552 |
| Q3_K_S | 1,454,357,056 |
| Q2_K | 1,274,755,648 |

Local file `xLAM-2-3B-fc-r-Q4_0.gguf` on disk: 1,822,849,600 bytes — matches
the hub listing exactly.

### Local Q4_0 file: parsed header (via `gguf` Python package, `GGUFReader`)

- `GGUF.version` = 3, `GGUF.tensor_count` = 434, `GGUF.kv_count` = 27.
- `general.architecture` = `qwen2`; `general.name` = "xLAM 2 3b Fc R";
  `general.organization` = Salesforce; `general.basename` = xLAM-2;
  `general.size_label` = 3B; `general.file_type` = 2 (mostly-Q4_0);
  `general.quantization_version` = 2.
- Model hyperparameters embedded and cross-checked against config.json —
  all match: `qwen2.block_count`=36, `qwen2.context_length`=32768,
  `qwen2.embedding_length`=2048, `qwen2.feed_forward_length`=11008,
  `qwen2.attention.head_count`=16, `qwen2.attention.head_count_kv`=2,
  `qwen2.rope.freq_base`=1000000.0,
  `qwen2.attention.layer_norm_rms_epsilon`≈9.9999999748e-07.
- Tokenizer metadata: `tokenizer.ggml.model`=`gpt2` (byte-level BPE family
  tag used by llama.cpp), `tokenizer.ggml.pre`=`qwen2` (selects Qwen2's
  pre-tokenizer regex inside llama.cpp's built-in table),
  `tokenizer.ggml.tokens` (151936 strings), `tokenizer.ggml.merges`
  (151387 BPE merge rules — matches tokenizer.json's merge count exactly),
  `tokenizer.ggml.token_type` (151936 entries), `tokenizer.ggml.bos_token_id`=151643,
  `tokenizer.ggml.eos_token_id`=151645, `tokenizer.ggml.padding_token_id`=151643,
  `tokenizer.ggml.add_bos_token`=false.
- `tokenizer.chat_template`: the **full Jinja2 template string is embedded
  verbatim** in the GGUF metadata (byte-identical to
  `tokenizer_config.json`'s `chat_template`, see §3).

### Tensor inventory (434 tensors = 36 layers × 12 + 2 top-level)

| tensor (per-layer pattern, ×36) | shape | quant |
|---|---|---|
| `blk.N.attn_q.weight` | [2048, 2048] | **Q4_0** |
| `blk.N.attn_q.bias` | [2048] | F32 |
| `blk.N.attn_k.weight` | [2048, 256] | **Q4_0** |
| `blk.N.attn_k.bias` | [256] | F32 |
| `blk.N.attn_v.weight` | [2048, 256] | **Q4_0** |
| `blk.N.attn_v.bias` | [256] | F32 |
| `blk.N.attn_output.weight` | [2048, 2048] | **Q4_0** |
| `blk.N.attn_norm.weight` | [2048] | F32 |
| `blk.N.ffn_gate.weight` | [2048, 11008] | **Q4_0** |
| `blk.N.ffn_up.weight` | [2048, 11008] | **Q4_0** |
| `blk.N.ffn_down.weight` | [11008, 2048] | **Q4_0** |
| `blk.N.ffn_norm.weight` | [2048] | F32 |

Top-level (×1 each):

| tensor | shape | quant |
|---|---|---|
| `token_embd.weight` | [2048, 151936] | **Q6_K** (not Q4_0 — llama.cpp keeps the embedding table at a higher-precision quant by default even in an otherwise-Q4_0 file) |
| `output_norm.weight` | [2048] | F32 |

**`output.weight` (the separate LM head) does not exist in this file.**
Consistent with `tie_word_embeddings: true` in config.json: the engine
must reuse `token_embd.weight` (transposed) as the LM head projection
rather than expect a dedicated output tensor.

QKV bias tensors (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) exist and
are F32 — Qwen2 uses biased QKV projections (unlike most Llama-family
models), the engine's attention block must add these biases before RoPE.
There are no `attn_output.bias` / `ffn_*.bias` tensors — only QKV carries
bias.

`general.file_type = 2` reflects that this is a mixed-precision file:
Q4_0 for all large 2D weight matrices, F32 for norms and QKV biases, Q6_K
for the embedding table (llama.cpp's typical "mostly Q4_0" convention).

## 3. Chat template

Full template as embedded in `tokenizer_config.json` (`chat_template`,
2726 chars) and byte-identical in the GGUF's `tokenizer.chat_template` KV:

```jinja
{# System message #}
{{- "<|im_start|>system\n" }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] | trim %}
    {%- set messages = messages[1:] %}
    {{- system_message + "\n" }}
{%- else %}
    {%- set system_message = "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team." %}
    {% set format_instruction %}You have access to a set of tools. When using tools, make calls in a single JSON array: 

[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]

If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification. Do not interpret or respond until tool results are returned. Once they are available, process them or make additional calls if needed. For tasks that don't require tools, such as casual conversation or general advice, respond directly in plain text. The available tools are:{% endset %}
    {{- system_message + "\n" }}
    {%- if tools is not none %}
        {{- format_instruction + "\n\n" }}
    {%- endif %}
{%- endif %}

{%- if tools is not none %}
    {%- for func in tools %}
        {{- func | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- "<|im_end|>" }}
{%- for message in messages %}
    {%- if message['role'] == 'tool' %}
        {{- "<|im_start|>tool\n" }}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {%- if content is mapping or content is iterable and content is not string %}
            {{- content | tojson }}
        {%- else %}
            {{- content }}
        {%- endif %}
        {{- "<|im_end|>" }}
    {%- elif 'tool_calls' in message %}
        {{- "<|im_start|>assistant\n" }}
        {%- if message['tool_calls'] %}
            {{- "[" }}
            {%- for tool_call in message.tool_calls %}
                {%- set out = tool_call.function | tojson %}
                {{- out }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- "]"}}
        {%- elif message['content'] %}
            {{- message['content'] | trim }}
        {%- else %}
            {{- "[]\n" }}
        {%- endif %}
        {{- "<|im_end|>" }}
    {%- else %}
        {{- "<|im_start|>" + message['role'] + "\n" + message['content'] | trim + "<|im_end|>" }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- "<|im_start|>assistant\n" }}
{%- endif %}
```

Notes: there is **no** dedicated `{% generation %}`/thinking block and no
`{%- set ns = namespace(...) %}` in this template (unlike many Qwen3
templates) — see §4.

### Tool syntax on input

- `tools` is a top-level Jinja variable — a list of raw JSON-Schema
  function objects (README's example passes plain `{"name", "description",
  "parameters": {...}}` dicts to `tokenizer.apply_chat_template(...,
  tools=tools)`; note this is **not** the OpenAI-style
  `{"type":"function","function":{...}}` wrapper — the vLLM/OpenAI example
  in the README wraps it that way for the server API, but the template
  itself just `tojson`-dumps whatever object is in each `tools` list entry).
- Each tool is serialized with `func | tojson(indent=4)` — pretty-printed,
  4-space indent, one tool per block, separated by a blank line.
- The system prompt is always the **first** block in the rendered prompt,
  wrapped in `<|im_start|>system ... <|im_end|>`: either the caller's own
  `messages[0]` (role `system`) content, or (if absent) a hardcoded
  Salesforce default ("You are a helpful assistant that can use tools...").
  The tool-use format instructions and the serialized tool list are
  appended inside that same system block, **only when `tools is not
  none`**.

### Tool-call OUTPUT format the model emits

Contrary to the generic Qwen `<tool_call>{...}</tool_call>` XML-ish
convention, xLAM-2 emits **a bare JSON array, no wrapper tags**:

```
[{"name": "get_weather", "arguments": {"location": "London"}}, {"name": "...", "arguments": {...}}]
```

This is confirmed both by the chat template's own assistant-turn rendering
(`{{- "[" }} ... tool_call.function | tojson ... {{- "]" }}`) and by
`xlam_tool_call_parser.py` (the vLLM tool-call parser plugin for this model
family), whose `extract_tool_calls` does:

```python
if not model_output.strip().startswith('['):
    return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
tool_calls_data = json.loads(model_output)
for idx, call in enumerate(tool_calls_data):
    tool_call = ToolCall(..., function=FunctionCall(name=call["name"], arguments=json.dumps(call["arguments"])))
```

I.e.: the engine's tool-call detector should treat "output starts with
`[`" as the trigger, parse the whole output as a JSON array, and each
array element as `{"name": ..., "arguments": {...}}`. **Parallel tool
calls are just multiple objects in that one array** — there's no separate
delimiter or multiple `<tool_call>` blocks, unlike the Hermes/Qwen3
convention. If the model instead emits plain prose (no tools needed), the
array-detection short-circuits to "not a tool call" and the raw text is
the content. `xlam_tool_call_parser.py` also implements incremental/
streaming extraction via `partial_json_parser`, for the same array format.

### Tool RESULTS fed back

Role `"tool"` messages, rendered as `<|im_start|>tool\n{content}<|im_end|>`.
Content handling is defensive/nested: the template first checks for
`message.content.content` (an object-with-a-content-field shape), else
uses `message.content` directly; if the resulting value `is mapping` or is
`iterable and not string`, it's `tojson`-dumped (compact, no indent arg),
otherwise emitted as a raw string. In practice: pass the tool's result as
either a JSON-serializable object/list (auto-dumped) or a plain string.

### Jinja features used (for minijinja mapping)

| feature | used how | minijinja support |
|---|---|---|
| `tojson` filter, with `indent=N` kwarg | `func \| tojson(indent=4)`, `content \| tojson`, `tool_call.function \| tojson` | needs the `json` crate feature (adds `tojson` builtin filter incl. `indent` kwarg) |
| `trim` filter | `system_message \| trim`, `message['content'] \| trim` | builtin (`builtins` feature) |
| `loop.last` | tool-call array comma-joining | builtin — no `adjacent_loop_items` needed (that's only for `previtem`/`nextitem`; `loop.last`/`loop.first`/`loop.index` are core) |
| `{% set ... %}` and block-form `{% set var %}...{% endset %}` | `system_message`, `format_instruction` (block form) | core parser feature, no flag |
| `{%- ... -%}` whitespace control | throughout | core parser behavior, no flag |
| tests: `is not none`, `is mapping`, `is iterable`, `is not string`, `is defined` | tool/content branching | builtin tests (`builtins` feature) |
| slicing (`messages[1:]`) | system-message stripping | core, no flag |
| `+` string concatenation, `in` operator (`'tool_calls' in message`) | throughout | core |
| **not used**: `namespace()`, `loop.cycle`/`loop.previtem`/`loop.nextitem`, custom delimiters, `{% break %}`/`{% continue %}`, macros/imports/extends | — | so `loop_controls`, `custom_syntax`, `adjacent_loop_items`, `macros`, `multi_template` are all **not required** for this template |

Net minijinja feature set needed: default `builtins` (filters/tests) +
`json` (for `tojson` incl. `indent`). No custom functions/filters are
required beyond what minijinja ships — everything the template calls
(`tojson`, `trim`, `mapping`/`iterable`/string tests, `loop.last`) is a
stock builtin once `json` is enabled. (Checked live against
docs.rs/minijinja's "Optional Features" section, current release.)

## 4. Thinking toggle

**Not supported by this template.** There is no `enable_thinking`
variable, no `<think>`/`</think>` handling, and no conditional branch on a
thinking flag anywhere in the 2726-char template — unlike Qwen3's own chat
template (which has an explicit thinking/no-thinking switch). xLAM-2-3b-fc-r
is fine-tuned from a Qwen2.5-class base (no native reasoning mode) and its
own template does not add one. Default/only behavior: direct response
(plain text) or the bare-JSON tool-call array described in §3 — no
intermediate reasoning trace is part of the protocol.

## 5. Tokenizer

- Byte-level BPE (`tokenizer.json` `model.type` = `BPE`, GGUF
  `tokenizer.ggml.model` = `gpt2` — llama.cpp's tag for this same
  byte-level-BPE family).
- Pre-tokenizer: `Sequence` of `Split` (regex, `Isolated` behavior) +
  `ByteLevel` (`add_prefix_space: false`, `trim_offsets: false`,
  `use_regex: false`). The split regex:
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  — the standard GPT-2/Qwen2 pre-tokenizer pattern. GGUF's
  `tokenizer.ggml.pre = "qwen2"` selects the equivalent built-in regex
  inside llama.cpp rather than embedding the pattern as a string.
- Decoder: `ByteLevel`. Normalizer: `NFC`.
- Vocab: 151,643 base BPE tokens + 151,387 merges + 22 added/special
  tokens = 151,665 addressable tokens (see §1 for the 151,936 padding
  discrepancy vs. config/GGUF).
- `special_tokens_map.json`: 13 `additional_special_tokens` (`<|im_start|>`,
  `<|im_end|>`, plus 11 vision/object/box/quad placeholder tokens inherited
  from the Qwen2-VL family, unused by this text-only checkpoint), plus
  `eos_token` = `<|im_end|>` and `pad_token` = `<|endoftext|>`.
- `tokenizer.json` alone is sufficient for encode+decode: it embeds vocab,
  merges, pre-tokenizer, decoder, normalizer, post-processor, and all
  added tokens — no external vocab.json/merges.txt needed. (The GGUF file
  is independently self-sufficient too, carrying its own copy of
  tokens/merges/token_type arrays.)

## 6. The two 8B candidates ("for later")

`sonos/PLAN.md` and `sonos/NOTES.md` (in the sonos-mcp worktree) mention an
8B tier as a later experiment but do not name two specific 8B repos to
evaluate — `PLAN.md` line 101-103 says only "An 8B (~5 GB in GPU buffers on
a 16 GB unified M2) is plausible later, not in v1." Per fallback
instruction, using **Salesforce/Llama-xLAM-2-8b-fc-r** (same family, next
size up) and **Qwen/Qwen3-8B** (NOTES.md's own "smart lane" 8B reference,
line 106).

| | Llama-xLAM-2-8b-fc-r | Qwen3-8B |
|---|---|---|
| architecture | `LlamaForCausalLM` | `Qwen3ForCausalLM` |
| layers | 32 | 36 |
| heads / KV heads | 32 / 8 | 32 / 8 |
| head dim | 128 | 128 |
| hidden / intermediate | 4096 / 14336 | 4096 / 12288 |
| vocab_size | 128256 | 151936 |
| max position embeddings | 131072 | 40960 |
| rope theta | 500000.0 | 1000000 |
| tied embeddings | false | false |
| license | cc-by-nc-4.0 (+ Meta Llama 3 Community License per xLAM README's model-license note, since this is a Llama-architecture fine-tune) | apache-2.0 |
| official GGUF repo | `Salesforce/Llama-xLAM-2-8b-fc-r-gguf`: F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q5_K_S, Q4_K_M, Q4_K_S, **Q4_0** (4,661,214,432 B), Q3_K_L, Q3_K_S, Q2_K | `Qwen/Qwen3-8B-GGUF`: Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M only — **no Q4_0** quant published officially |
| chat template tool syntax | same xLAM-2 family template/parser as the 3B — bare JSON array output, `xlam_tool_call_parser.py` applies to "all xLAM series models" per that model's README | Qwen3's own template: uses `<tool_call>{...}</tool_call>` XML-ish wrapper (Hermes-style, different from xLAM's bare-array format) and **does** support a thinking/no-thinking switch (`enable_thinking`, default true in most Qwen3 deployments) — not verified against this specific model's tokenizer_config.json locally, flagged as uncertain |

Uncertain / not verified for the 8B row: Qwen3-8B's exact chat_template
text and its `enable_thinking` default were not fetched and diffed here
(only the config.json and GGUF quant listing were pulled) — confirm from
`Qwen/Qwen3-8B`'s `tokenizer_config.json` before relying on it.

## 7. What the engine must implement (checklist, from §1-3)

- [ ] GQA attention: 16 query heads / 2 KV heads, head dim 128, with
      **QKV biases** (F32) added before RoPE — not just weight matmuls.
- [ ] RoPE theta 1,000,000; RMSNorm eps ≈1e-6, pre-norm (attn_norm,
      ffn_norm) around attention and SwiGLU FFN.
- [ ] SwiGLU FFN: `ffn_gate`, `ffn_up` (2048→11008), `ffn_down`
      (11008→2048), all Q4_0.
- [ ] LM head = **reuse `token_embd.weight` transposed** (tied embeddings;
      no `output.weight` tensor exists in the GGUF to fall back to).
- [ ] Mixed quant dequant: Q4_0 for attn/ffn weight matrices, F32 for
      norms and QKV biases, **Q6_K for `token_embd.weight`** — loader must
      support both Q4_0 and Q6_K block formats, not just Q4_0.
- [ ] Vocab/embedding table sized to 151,936 rows (config/GGUF), while the
      tokenizer only ever emits ids in [0, 151,665) — don't assume the two
      numbers are the same when computing buffer sizes vs. valid-id ranges.
- [ ] Stop condition: terminate on **either** token id 151645
      (`<|im_end|>`) or 151643 (`<|endoftext|>`), matching
      generation_config's two-element `eos_token_id` list.
- [ ] Chat template rendering via minijinja with `json` feature enabled
      (for `tojson`, incl. `indent` kwarg) — default `builtins` covers
      everything else used (`trim`, `loop.last`, mapping/iterable/string
      tests, block-form `{% set %}`).
- [ ] Tool-call detection: check if generated text (post-decode, stripped)
      starts with `[`; if so, `json.parse` the whole thing as an array of
      `{name, arguments}` objects — **no `<tool_call>` tags to strip**,
      unlike other Qwen/Hermes-style models the engine may also target.
- [ ] No thinking-mode handling needed for this specific model (template
      has no such branch) — don't wire up a thinking toggle for xLAM-2-3b;
      reserve that logic path for when/if Qwen3-8B is added later.
