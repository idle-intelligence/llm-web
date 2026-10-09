//! The `tojson` filter transformers installs in its chat-template
//! environment (`transformers/utils/chat_template_utils.py`):
//!
//! ```python
//! def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
//!     return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
//! ```
//!
//! minijinja's built-in `tojson` differs in two ways that change the bytes:
//!
//! 1. it HTML-escapes `<`, `>`, `&` and `'` so the output is safe inside a
//!    `<script>` tag. `json.dumps` never does, so [`py_tojson`] skips that
//!    pass.
//! 2. without an indent it uses `serde_json`'s compact separators `","` and
//!    `":"`, while `json.dumps` uses `", "` and `": "`.
//!    [`PySeparatorsFormatter`] writes Python's separators; string and number
//!    escaping already match (`ensure_ascii=False` leaves non-ASCII text as
//!    is, and `/` is not escaped).
//!
//! With `indent=N`, `serde_json`'s `PrettyFormatter` already matches
//! `json.dumps(x, indent=N)` and is used as is.
//!
//! Object keys keep their insertion order: `serde_json` and minijinja are
//! both built with `preserve_order` (root `Cargo.toml`).
//!
//! This filter and the formatter come from this project's llm-wasm crate
//! (`crates/llm-wasm/src/template.rs`), where they were written for
//! Qwen2.5's tool-calling template.

use std::io;

use minijinja::value::{Kwargs, Value};
use serde::Serialize;

/// Python's default `json.dumps` separators without an indent: `", "`
/// between entries and `": "` between a key and its value.
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

/// `json.dumps(value, ensure_ascii=False)` with no indent.
fn to_json_compact_py<T: ?Sized + Serialize>(value: &T) -> serde_json::Result<String> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PySeparatorsFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(out).expect("serde_json only emits valid UTF-8"))
}

fn json_error(err: impl std::fmt::Display) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("cannot serialize to JSON: {err}"))
}

/// The `tojson` filter: transformers' `json.dumps`, see the module doc.
pub(crate) fn py_tojson(value: Value, kwargs: Kwargs) -> Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;

    let rendered = match indent {
        Some(indent) => {
            let mut out = Vec::new();
            let indentation = " ".repeat(indent);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
            value.serialize(&mut ser).map_err(json_error)?;
            String::from_utf8(out).expect("serde_json only emits valid UTF-8")
        }
        None => to_json_compact_py(&value).map_err(json_error)?,
    };

    Ok(Value::from_safe_string(rendered))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_json_dumps() {
        let v = serde_json::json!({"b": [1, 2.5, true, null], "a": "<tag> & 'q' café 東京 \n"});
        assert_eq!(to_json_compact_py(&v).unwrap(), r#"{"b": [1, 2.5, true, null], "a": "<tag> & 'q' café 東京 \n"}"#);
    }
}
