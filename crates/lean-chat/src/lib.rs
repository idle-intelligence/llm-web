//! lean-chat: chat messages to prompt text, with the model's own chat
//! template.
//!
//! [`render`] renders a conversation (and optionally a tool list) through a
//! model's [`ChatTemplate`] (read from its `tokenizer_config.json`) and returns the same text, byte for byte, as
//! transformers' `tokenizer.apply_chat_template(messages, tools=tools,
//! add_generation_prompt=..., tokenize=False)`. The text then goes to an
//! inference engine as-is, e.g. lean's `Lean::generate`.
//!
//! [`trim_oldest_exchange`] is an optional helper for a caller whose
//! conversation no longer fits the model's context: it drops the oldest
//! exchange and keeps the system message. When to call it is the caller's
//! policy.
//!
//! What makes the output match transformers:
//! - the template is rendered with `trim_blocks` and `lstrip_blocks` on, as
//!   transformers' Jinja environment does;
//! - `tojson` is replaced with the `json.dumps` transformers uses (see
//!   `tojson.rs`), and object keys keep their insertion order;
//! - Python string methods some templates call (`.startswith`, `.split`,
//!   `.strip`, ...) come from minijinja-contrib's `pycompat`;
//! - `raise_exception(msg)` fails the render with `msg`, and `tools` is
//!   `none` when no tools are given, as in transformers;
//! - the tokenizer's `bos_token` and `eos_token` are in the context, as
//!   transformers passes them; [`template_from_tokenizer_config`] reads
//!   them along with the template into a [`ChatTemplate`].

use std::ops::Range;

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

mod tojson;
#[cfg(feature = "web")]
pub mod web;

/// One chat message, with the fields chat templates read: `role`,
/// `content`, and for tool use `tool_calls` (on an assistant message),
/// `name` and `tool_call_id` (on a tool result). `content` is always a
/// string; a JSON `null` content reads as `""`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

fn null_as_empty<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message { role: role.into(), content: content.into(), tool_calls: Vec::new(), name: None, tool_call_id: None }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// An assistant turn that called tools.
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Message { tool_calls, ..Self::new("assistant", content) }
    }

    /// A tool's result, fed back to the model.
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new("tool", content)
    }
}

/// One element of an assistant message's `tool_calls`, in the
/// OpenAI/transformers shape `{"type": "function", "function": {"name",
/// "arguments"}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: ToolCallFunction,
}

fn function_kind() -> String {
    "function".to_string()
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        ToolCall { id: None, kind: function_kind(), function: ToolCallFunction { name: name.into(), arguments } }
    }
}

/// `arguments` is an object, or a JSON-encoded string as some APIs send
/// it; templates render either form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: Value,
}

/// A model's chat template: the Jinja2 source plus the special tokens
/// transformers passes into the template context (`bos_token`,
/// `eos_token`). Templates that start the text with `{{ bos_token }}` or end
/// turns with `{{ eos_token }}` need them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplate {
    pub source: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

impl ChatTemplate {
    /// A template from its source alone, with no special tokens.
    pub fn new(source: impl Into<String>) -> Self {
        ChatTemplate { source: source.into(), bos_token: None, eos_token: None }
    }
}

/// Renders `messages` (and `tools`, each one a JSON tool definition such as
/// `{"type": "function", "function": {"name", "description",
/// "parameters"}}`) through `template`. With `add_generation_prompt`, the
/// text ends with the opening of an assistant turn, ready for generation.
pub fn render(template: &ChatTemplate, messages: &[Message], tools: Option<&[Value]>, add_generation_prompt: bool) -> Result<String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson::py_tojson);
    env.add_function("raise_exception", raise_exception);
    env.add_template("chat", &template.source).context("parsing chat_template")?;
    let tmpl = env.get_template("chat")?;
    // A special token the tokenizer does not define stays undefined, as in
    // transformers (which only passes the tokens it has).
    let mut ctx = context! { messages => messages, tools => tools, add_generation_prompt => add_generation_prompt };
    if let Some(bos) = &template.bos_token {
        ctx = context! { bos_token => bos, ..ctx };
    }
    if let Some(eos) = &template.eos_token {
        ctx = context! { eos_token => eos, ..ctx };
    }
    let rendered = tmpl.render(ctx).context("rendering chat_template")?;
    Ok(rendered)
}

fn raise_exception(msg: String) -> std::result::Result<String, minijinja::Error> {
    Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
}

/// The index range of the oldest exchange in a conversation given by its
/// roles: from the first `user` message up to (not including) the next
/// `user` message, so the assistant reply and any tool calls and results
/// go with it. `None` when no later `user` message follows (the latest
/// exchange, the one being answered, is never dropped). Messages before
/// the first `user` message (the system message) are never part of it.
pub(crate) fn oldest_exchange<'a>(roles: impl IntoIterator<Item = &'a str>) -> Option<Range<usize>> {
    let roles: Vec<&str> = roles.into_iter().collect();
    let start = roles.iter().position(|r| *r == "user")?;
    let end = start + 1 + roles[start + 1..].iter().position(|r| *r == "user")?;
    Some(start..end)
}

/// Drops the oldest exchange: the first `user` message and everything up
/// to the next `user` message (its assistant reply, tool calls and tool
/// results). The system message stays, and so does the latest user turn.
/// Returns `false`, changing nothing, when there is no older exchange left
/// to drop.
pub fn trim_oldest_exchange(messages: &mut Vec<Message>) -> bool {
    match oldest_exchange(messages.iter().map(|m| m.role.as_str())) {
        Some(range) => {
            messages.drain(range);
            true
        }
        None => false,
    }
}

/// The chat template of a `tokenizer_config.json`'s text: its
/// `chat_template` string and its `bos_token` and `eos_token` (written
/// either as a string or as an added-token object with a `content` field).
pub fn template_from_tokenizer_config(tokenizer_config_json: &str) -> Result<ChatTemplate> {
    let value: Value = serde_json::from_str(tokenizer_config_json)?;
    let source = value.get("chat_template").and_then(Value::as_str).context("tokenizer_config.json has no string 'chat_template' field")?;
    let token = |key: &str| match value.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    };
    Ok(ChatTemplate { source: source.to_string(), bos_token: token("bos_token"), eos_token: token("eos_token") })
}

/// [`template_from_tokenizer_config`] for a file on disk (native only).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_chat_template(tokenizer_config_path: &str) -> Result<ChatTemplate> {
    let text = std::fs::read_to_string(tokenizer_config_path).with_context(|| format!("reading {tokenizer_config_path}"))?;
    template_from_tokenizer_config(&text).with_context(|| format!("in {tokenizer_config_path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(messages: &[Message]) -> Vec<&str> {
        messages.iter().map(|m| m.role.as_str()).collect()
    }

    #[test]
    fn trim_drops_the_oldest_exchange_with_its_tool_messages() {
        let mut m = vec![
            Message::system("s"),
            Message::user("u1"),
            Message::assistant_tool_calls("", vec![ToolCall::new("f", serde_json::json!({}))]),
            Message::tool("r"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
            Message::user("u3"),
        ];
        assert!(trim_oldest_exchange(&mut m));
        assert_eq!(roles(&m), ["system", "user", "assistant", "user"]);
        assert_eq!(m[1].content, "u2");
        assert!(trim_oldest_exchange(&mut m));
        assert_eq!(roles(&m), ["system", "user"]);
        assert_eq!(m[1].content, "u3");
        assert!(!trim_oldest_exchange(&mut m), "the latest user turn is never dropped");
        assert_eq!(roles(&m), ["system", "user"]);
    }

    #[test]
    fn trim_without_system_message() {
        let mut m = vec![Message::user("u1"), Message::assistant("a1"), Message::user("u2")];
        assert!(trim_oldest_exchange(&mut m));
        assert_eq!(roles(&m), ["user"]);
        let mut empty: Vec<Message> = Vec::new();
        assert!(!trim_oldest_exchange(&mut empty));
    }

    #[test]
    fn null_content_reads_as_empty() {
        let m: Message = serde_json::from_str(r#"{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"f","arguments":{"a":1}}}]}"#).unwrap();
        assert_eq!(m.content, "");
        assert_eq!(m.tool_calls[0].function.name, "f");
    }

    #[test]
    fn special_tokens_from_tokenizer_config() {
        let t = template_from_tokenizer_config(r#"{"chat_template": "{{ bos_token }}x{{ eos_token }}", "bos_token": {"content": "<s>", "lstrip": false}, "eos_token": "</s>"}"#).unwrap();
        assert_eq!(render(&t, &[], None, false).unwrap(), "<s>x</s>");
        assert_eq!(render(&ChatTemplate::new("{{ bos_token }}x"), &[], None, false).unwrap(), "x");
    }

    #[test]
    fn raise_exception_fails_the_render() {
        let err = render(&ChatTemplate::new("{{ raise_exception('no system role') }}"), &[], None, false).unwrap_err();
        assert!(format!("{err:#}").contains("no system role"), "{err:#}");
    }
}
