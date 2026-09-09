//! Chat template rendering (minijinja) for xLAM-2-3b-fc-r's embedded Jinja2
//! `chat_template`. Owned by phase 1a. See `docs/MODELS.md` §3 for the
//! template source and the wire shapes it expects.
//!
//! ## Closing the `tojson` fidelity gap vs. Python/transformers
//!
//! `transformers` patches Jinja's `tojson` filter (see
//! `transformers/utils/chat_template_utils.py`) to:
//!
//! ```python
//! def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
//!     return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
//! ```
//!
//! Python's `json.dumps` defaults `separators` to `(", ", ": ")` when
//! `indent is None`, and to `(",", ": ")` (with a newline after each comma)
//! when an indent is given. minijinja's *built-in* `tojson` filter differs
//! from this in two ways that break byte-for-byte fidelity:
//!
//! 1. it HTML/JS-escapes `<`, `>`, `&`, `'` to `<` etc. so the output is
//!    safe to embed in a `<script>` tag. The chat template isn't HTML and
//!    Python's `json.dumps` never does this rewrite, so our replacement
//!    filter (`py_tojson`) skips that post-processing pass entirely.
//! 2. in *compact* (no-indent) mode it calls `serde_json::to_string`, whose
//!    `CompactFormatter` uses `(",", ":")` with **no spaces** — not
//!    Python's `(", ", ": ")`. We fix this with `PySeparatorsFormatter`, a
//!    `serde_json::ser::Formatter` that overrides just the three separator
//!    hooks (`begin_array_value`, `begin_object_key`, `begin_object_value`)
//!    to insert Python's spacing; everything else (string/number escaping)
//!    already matches serde_json's defaults, which — like Python's
//!    `ensure_ascii=False` — leave non-ASCII text unescaped and don't
//!    special-case `/`.
//!
//! The `indent=N` (pretty) case needs no such rewrite: minijinja's built-in
//! filter already drives `serde_json::ser::PrettyFormatter` with the given
//! indent width, whose comma/brace placement matches
//! `json.dumps(x, indent=N)` byte-for-byte (verified against the reference
//! fixtures) — we reuse that formatter as-is, only dropping the HTML-escape
//! post-pass from point 1.
//!
//! ## Object key order
//!
//! `Message`, `Tool`, `ToolFunction`, `ToolCallEntry` and `ToolCallFunction`
//! below are plain Rust structs, not dynamic maps. `#[derive(Serialize)]`
//! on a struct always emits fields in declaration order — that's a property
//! of serde's `serialize_struct`, unrelated to the `preserve_order` cargo
//! feature (which only affects *dynamic* map types like
//! `serde_json::Value::Object`, `HashMap`, etc.). minijinja mirrors this:
//! its `Value::from_serialize` routes `serialize_struct` calls into a
//! `Vec<(&'static str, Value)>`-backed object (`StaticKeyMap`), preserving
//! field order regardless of features (see
//! `minijinja::value::serialize::SerializeStruct`), while `serialize_map`
//! calls (i.e. genuinely dynamic maps, including every `serde_json::Value`
//! we embed) go through minijinja's `ValueMap`, which is a `BTreeMap`
//! (sorted by key) unless minijinja's own `preserve_order` feature is
//! enabled — and neither that nor `serde_json`'s `preserve_order` feature
//! is enabled here (`Cargo.toml` is frozen for this crate, so we didn't add
//! either). Net effect: the *wrapper* objects we control the shape of
//! (`{"type": ..., "function": {...}}`, `{"name": ..., "arguments": ...}`,
//! etc.) render in exactly the declared field order; the one place we hold
//! genuinely freeform JSON — the JSON-Schema `parameters`/`inputSchema`
//! blob, and `tool_calls[].arguments` — renders in *sorted* key order.
//! This is a real limitation for arbitrary JSON, but every Sonos tool
//! schema in `fixtures/sonos/tools.json` (all 34 tools, checked
//! programmatically) already declares its JSON-Schema properties in
//! alphabetical order, so sorted-key rendering happens to reproduce the
//! fixtures' actual byte layout with no further work.

use anyhow::{anyhow, Context, Result};
use minijinja::value::{Kwargs, Value as MjValue};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::io;

/// A single chat message, matching the fields the xLAM-2 chat template
/// inspects (`role`, `content`, `tool_calls`, `name`, `tool_call_id`).
///
/// Field order here has no bearing on template output — the template only
/// ever accesses fields by name (`message['role']`, `message.tool_calls`,
/// …), never dumps the whole message object with `tojson`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallEntry>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(JsonValue::String(content.into())),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(JsonValue::String(content.into())),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(JsonValue::String(content.into())),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCallEntry>) -> Self {
        Self {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(tool_calls),
            name: None,
            tool_call_id: None,
        }
    }
}

/// One element of `message.tool_calls` (assistant messages that already
/// made tool calls earlier in the conversation, fed back in on replay).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_function_kind")]
    pub kind: String,
    pub function: ToolCallFunction,
}

fn default_function_kind() -> String {
    "function".to_string()
}

/// `tool_call.function` — rendered as `{"name": ..., "arguments": ...}` via
/// `tojson` (no indent). Field order (`name` then `arguments`) matches the
/// reference fixtures because this is a plain struct (see module docs).
///
/// `arguments` is a `serde_json::Value` so it can carry either shape seen
/// in the wild: an OpenAI-style JSON-*encoded string* (e.g. `"{}"`, as in
/// `fixtures/reference/inputs/03_tools_multiturn.json`) when replaying a
/// conversation captured from an OpenAI-shaped transcript, or a real JSON
/// *object* (e.g. `{"group_id": "RINCON_KITCHEN01:1"}`) as the model itself
/// emits and as the agent loop in `agent.rs` appends to history. The
/// template renders whichever it's given (a string renders quoted, an
/// object renders unquoted) — see `tests/template.rs` for both cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: JsonValue,
}

/// One entry of the `tools` list, transformers/OpenAI shape:
/// `{"type": "function", "function": {"name", "description", "parameters"}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_function_kind")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    /// JSON-Schema object, opaque to us — see module docs for the ordering
    /// caveat.
    pub parameters: JsonValue,
}

impl Tool {
    /// Build a `Tool` from an MCP `tools/list` entry
    /// (`{"name", "description", "inputSchema"}`).
    pub fn from_mcp(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonValue,
    ) -> Self {
        Tool {
            kind: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters: input_schema,
            },
        }
    }
}

/// A compiled xLAM-2 chat template, ready to render prompts.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    /// Build from the raw Jinja2 template string plus the special tokens it
    /// may reference (this particular xLAM-2 template doesn't use
    /// `bos_token`/`eos_token` as Jinja globals — see `docs/MODELS.md` §3 —
    /// but other chat templates do, so we still thread them through).
    pub fn new(
        template_source: impl Into<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self> {
        let mut env = Environment::new();
        env.add_filter("tojson", py_tojson);
        env.add_template_owned("chat", template_source.into())
            .context("failed to compile chat_template")?;
        Ok(Self {
            env,
            bos_token,
            eos_token,
        })
    }

    /// Build from a parsed `tokenizer_config.json`.
    pub fn from_tokenizer_config(cfg: &JsonValue) -> Result<Self> {
        let template_source = cfg
            .get("chat_template")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| anyhow!("tokenizer_config.json missing string field `chat_template`"))?
            .to_string();
        let bos_token = cfg
            .get("bos_token")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        let eos_token = cfg
            .get("eos_token")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        Self::new(template_source, bos_token, eos_token)
    }

    /// Render a full prompt. `tools` is treated as Jinja `none` when empty
    /// (matching how `transformers.apply_chat_template` is invoked for the
    /// no-tools case in `scripts/export_reference.py` — see
    /// `fixtures/reference/rendered/01_no_tools.txt`, which has no tool
    /// preamble even though its `tools` array is `[]`, not omitted).
    pub fn render_prompt(
        &self,
        messages: &[Message],
        tools: &[Tool],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let tools_ctx = if tools.is_empty() { None } else { Some(tools) };
        let rendered = tmpl.render(context! {
            messages => messages,
            tools => tools_ctx,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        })?;
        Ok(rendered)
    }
}

/// `serde_json::ser::Formatter` that reproduces Python's default (no
/// `indent`) `json.dumps` separators: `", "` between array/object entries
/// and `": "` between an object key and its value — `serde_json`'s own
/// `CompactFormatter` uses `","`/`":"` with no spaces.
#[derive(Default)]
struct PySeparatorsFormatter;

impl serde_json::ser::Formatter for PySeparatorsFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(if first { b"" } else { b", " })
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(if first { b"" } else { b", " })
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Serialize `value` the way Python's `json.dumps(value, ensure_ascii=False)`
/// would with no `indent` — see module docs.
pub(crate) fn to_json_compact_py<T: ?Sized + Serialize>(value: &T) -> Result<String> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PySeparatorsFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(out).expect("serde_json only emits valid UTF-8"))
}

fn json_error(err: impl std::fmt::Display) -> minijinja::Error {
    minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("cannot serialize to JSON: {err}"),
    )
}

/// Drop-in replacement for minijinja's built-in `tojson` filter that
/// matches Python/transformers's patched `tojson` exactly — see module
/// docs for the two gaps this closes.
fn py_tojson(value: MjValue, kwargs: Kwargs) -> std::result::Result<MjValue, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent").ok();
    kwargs.assert_all_used()?;

    let rendered = match indent {
        Some(indent) => {
            let mut out = Vec::new();
            let indentation = " ".repeat(indent);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
            serde::Serialize::serialize(&value, &mut ser).map_err(json_error)?;
            String::from_utf8(out).expect("serde_json only emits valid UTF-8")
        }
        None => to_json_compact_py(&value).map_err(json_error)?,
    };

    Ok(MjValue::from_safe_string(rendered))
}
