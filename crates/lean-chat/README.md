# lean-chat

lean-chat turns chat messages into the prompt text a model expects, with the model's own chat template (the `chat_template` string of its `tokenizer_config.json`). The text it returns is byte for byte what transformers' `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=..., tokenize=False)` returns, tools included. It runs no model: the text goes to an engine such as lean, whose `generate` takes text and returns text.

## Rust

```rust
use lean_chat::{read_chat_template, render, trim_oldest_exchange, Message};

let template = read_chat_template("tokenizer_config.json")?;
let mut messages = vec![Message::system("Answer briefly."), Message::user("What is the capital of France?")];
let prompt = render(&template, &messages, None, true)?;
// lean.generate(&prompt, ...); on ContextOverflow: trim_oldest_exchange(&mut messages) and render again.
```

- `render(template, messages, tools, add_generation_prompt) -> String`. `tools` is `None` or a list of JSON tool definitions (`{"type": "function", "function": {"name", "description", "parameters"}}`). With `add_generation_prompt`, the text ends with the opening of an assistant turn.
- `Message { role, content, tool_calls, name, tool_call_id }`, with `Message::system/user/assistant/tool` and `Message::assistant_tool_calls(content, calls)` for an assistant turn that called tools (`ToolCall::new(name, arguments)`).
- `trim_oldest_exchange(&mut messages) -> bool` drops the oldest exchange: the first user message and everything up to the next one (the reply, tool calls and tool results). The system message and the latest user turn stay; it returns `false` when there is nothing older to drop. Whether and when to trim is the caller's choice.
- `template_from_tokenizer_config(json)` and, natively, `read_chat_template(path)` read the template out of `tokenizer_config.json`.

## JavaScript

With the `web` feature, lean's wasm package exports:

- `chatRender(template, messages, tools, addGenerationPrompt)`: `messages` is an array of `{role, content, tool_calls?, name?, tool_call_id?}`, `tools` an array or `null`.
- `chatTrimOldestExchange(messages)`: a new array without the oldest exchange, or `undefined` when there is nothing older to drop.

```js
const template = JSON.parse(tokenizerConfigJson).chat_template;
const reply = await lean.generate(mod.chatRender(template, messages, null, true), params, onToken, abort.cloneFlag());
```

## Exactness

`tests/hf_exactness.rs` renders the conversations in `tests/fixtures/*.json` (Qwen2.5, Qwen3 and SmolLM2 templates: single and multi-turn chat, system messages, non-ASCII and HTML-like text, tools, tool calls with object and string arguments, parallel calls, tool results) and requires the same bytes transformers produced. `scripts/gen_chat_fixtures.py` writes the fixtures.

To match transformers, the template is rendered with `trim_blocks` and `lstrip_blocks`, `tojson` is Python's `json.dumps` (`", "` and `": "` separators, no HTML escaping, keys in insertion order), Python string methods come from minijinja-contrib's `pycompat`, `raise_exception` fails the render, and `tools` is `none` when absent.

Templates that read `bos_token` or other special tokens from the render context get nothing for them; none of the templates above does.
