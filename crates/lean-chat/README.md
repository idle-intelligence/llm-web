# lean-chat

lean-chat turns chat messages into the prompt text a model expects, with the model's own chat template (the `chat_template` string of its `tokenizer_config.json`, with its `bos_token` and `eos_token`). The text it returns is byte for byte what transformers' `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=..., tokenize=False)` returns, tools included. It runs no model: the text goes to an engine such as lean, whose `generate` takes text and returns text.

## Rust

```rust
use lean_chat::{read_chat_template, render, trim_oldest_exchange, Message};

// The template and its special tokens.

let template = read_chat_template("tokenizer_config.json")?;
let mut messages = vec![Message::system("Answer briefly."), Message::user("What is the capital of France?")];
let prompt = render(&template, &messages, None, true)?;
// lean.generate(&prompt, ...); on ContextOverflow: trim_oldest_exchange(&mut messages) and render again.
```

- `ChatTemplate { source, bos_token, eos_token }`: the template plus the special tokens transformers passes into the template context, read by `template_from_tokenizer_config(json)` (or, natively, `read_chat_template(path)`) from `tokenizer_config.json`. `ChatTemplate::new(source)` makes one from the source alone, with no special tokens.
- `render(&template, messages, tools, add_generation_prompt) -> String`. `tools` is `None` or a list of JSON tool definitions (`{"type": "function", "function": {"name", "description", "parameters"}}`). With `add_generation_prompt`, the text ends with the opening of an assistant turn. A template that calls `strftime_now` (Llama 3.x writes today's date with it) gets the local date and time.
- `render_at(&template, messages, tools, add_generation_prompt, now)` renders with `strftime_now` formatting `now` (a `chrono::NaiveDateTime`) instead, for a reproducible prompt.
- `Message { role, content, tool_calls, name, tool_call_id }`, with `Message::system/user/assistant/tool` and `Message::assistant_tool_calls(content, calls)` for an assistant turn that called tools (`ToolCall::new(name, arguments)`).
- `trim_oldest_exchange(&mut messages) -> bool` drops the oldest exchange: the first user message and everything up to the next one (the reply, tool calls and tool results). The system message and the latest user turn stay; it returns `false` when there is nothing older to drop. Whether and when to trim is the caller's choice.

## JavaScript

With the `web` feature, lean's wasm package exports:

- `ChatTemplate.fromTokenizerConfig(json)` and `new ChatTemplate(source)`.
- `chatRender(template, messages, tools, addGenerationPrompt)`: `template` is a `ChatTemplate`, `messages` is an array of `{role, content, tool_calls?, name?, tool_call_id?}`, `tools` an array or `null`.
- `chatTrimOldestExchange(messages)`: a new array without the oldest exchange, or `undefined` when there is nothing older to drop.

```js
const template = mod.ChatTemplate.fromTokenizerConfig(tokenizerConfigJson);
const reply = await lean.generate(mod.chatRender(template, messages, null, true), params, onToken, abort.cloneFlag());
```

## Exactness

`tests/hf_exactness.rs` renders the conversations in `tests/fixtures/*.json` (Qwen2.5, Qwen3, SmolLM2, Mistral v0.3 and Llama 3.2 templates; Mistral writes `bos_token` and `eos_token` into the text, Llama 3.2 the date from `strftime_now`, pinned in the fixture: single and multi-turn chat, system messages, non-ASCII and HTML-like text, tools, tool calls with object and string arguments, parallel calls, tool results) and requires the same bytes transformers produced. `scripts/gen_chat_fixtures.py` writes the fixtures.

To match transformers, the template is rendered with `trim_blocks` and `lstrip_blocks`, `tojson` is Python's `json.dumps` (`", "` and `": "` separators, no HTML escaping, keys in insertion order), Python string methods come from minijinja-contrib's `pycompat`, `raise_exception` fails the render, `tools` is `none` when absent, `bos_token` and `eos_token` are in the context, and `strftime_now(format)` formats the local time like Python's `strftime`.

Not covered: other special tokens (`pad_token`, `unk_token`, ...); a template that reads them renders differently.
