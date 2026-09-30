//! Renders Qwen2.5-Instruct's own Jinja2 `chat_template` (read straight out
//! of `tokenizer_config.json`, never hand-copied) via minijinja, for a
//! single-user-turn prompt with no tools - the scope this slice needs.
//! Unlike llm-wasm's `template.rs` (347 lines: Qwen2.5-3B-Instruct's tool-calling
//! message model, custom `py_tojson` filter for byte-exact `tojson`
//! fidelity against Python's `json.dumps`), Qwen2.5's own template never
//! calls `tojson` on the no-tools path this crate exercises, so none of
//! that machinery is needed here - this is intentionally the minimal
//! renderer for the plain-chat path, not a port of llm-wasm's tool-calling
//! one.
//!
//! `apply_chat_template` in `gen_fixture.py`'s reference is HF
//! transformers' own; matching it exactly means rendering the *same*
//! template string (read from `tokenizer_config.json`, not re-typed) with
//! the same `messages`/`add_generation_prompt` context.

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::Serialize;

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Renders `chat_template` for a single user message, `add_generation_prompt
/// = true` - the same call shape `gen_fixture.py` makes via
/// `tok.apply_chat_template([{"role": "user", "content": prompt}],
/// add_generation_prompt=True)`.
pub fn render_user_prompt(chat_template: &str, prompt: &str) -> Result<String> {
    let mut env = Environment::new();
    // Qwen3's chat_template calls Python string methods (`.startswith`,
    // `.endswith`, `.split`, `.strip`, `.lstrip`, `.rstrip`) that plain
    // minijinja doesn't implement - minijinja-contrib's pycompat callback
    // adds them (and Python-style negative-step slicing) without changing
    // how a template with none of these calls (Qwen2.5's) renders: this is
    // additive, so Qwen2.5's `render_user_prompt` output is unaffected -
    // see `renders_default_system_plus_user_turn` below, unchanged.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", chat_template).context("parsing chat_template")?;
    let tmpl = env.get_template("chat").unwrap();
    let messages = vec![ChatMessage { role: "user", content: prompt }];
    // `tools` is left out of the context entirely: minijinja's Undefined is
    // falsy in the template's `{%- if tools %}` check, matching Python
    // Jinja2's behavior for a variable that was never passed.
    let rendered = tmpl.render(context! { messages => messages, add_generation_prompt => true }).context("rendering chat_template")?;
    Ok(rendered)
}

/// Extracts the `chat_template` string out of a HF `tokenizer_config.json`'s
/// already-read text - the wasm32-safe half of [`read_chat_template`] (no
/// filesystem access), shared with `web.rs`, which gets this text from JS
/// (`fetch(...).text()`) instead of a path.
pub fn chat_template_from_config_json(tokenizer_config_json: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(tokenizer_config_json)?;
    value
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .context("tokenizer_config.json has no string 'chat_template' field")
}

/// Reads the `chat_template` string out of a HF `tokenizer_config.json` file
/// on disk. Native only (`std::fs`) - see [`chat_template_from_config_json`]
/// for the wasm32 path.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_chat_template(tokenizer_config_path: &str) -> Result<String> {
    let text = std::fs::read_to_string(tokenizer_config_path).with_context(|| format!("reading {tokenizer_config_path}"))?;
    chat_template_from_config_json(&text).with_context(|| format!("in {tokenizer_config_path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen2.5-Instruct's template, no-tools/no-system-message branch -
    /// copied from the model's own `tokenizer_config.json` (not
    /// hand-written) so this test exercises the exact string
    /// `read_chat_template` would load, without needing the model files on
    /// disk.
    const QWEN25_TEMPLATE: &str = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n";

    #[test]
    fn renders_default_system_plus_user_turn() {
        let out = render_user_prompt(QWEN25_TEMPLATE, "What is the capital of France?").unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }
}
