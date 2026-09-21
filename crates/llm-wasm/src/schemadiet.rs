//! Token diet for MCP `tools/list` payloads.
//!
//! MCP tool objects (`{"name", "description", "inputSchema"}`) carry
//! fields a JSON-Schema-authoring tool cares about but a model consuming
//! the schema via the chat template does not: `annotations`, a `title`
//! that duplicates the `description`, `$schema`, `additionalProperties:
//! false`, an empty `required: []`, `minLength: 1` on a string (the
//! default — the model can't tell "at least 1 char" from "no
//! constraint"), and `minimum`/`maximum` set to the i64 extremes (a
//! schema author's "effectively unbounded" spelling, not a real bound).
//! None of that changes what counts as a valid call, so stripping it is
//! lossless in meaning — see `docs/ENGINE.md` §"Tool-schema token diet"
//! for the measured savings and where this hooks into the two loops that
//! own `Tool::from_mcp` (`agent.rs`, `web.rs` — not touched here).
//!
//! Deliberately generic: nothing here is keyed on a Sonos tool/property
//! name, so it degrades gracefully (a no-op) against any other MCP
//! server's `tools/list` shape.

use serde_json::Value;

/// How aggressively to diet a tool list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DietLevel {
    /// No changes — pass the MCP payload through untouched.
    Level0,
    /// Structural-only trimming: drops fields that don't affect what a
    /// valid call is (see module docs). Safe to enable unconditionally.
    Level1,
    /// Level 1 plus description whitespace normalisation and cross-tool
    /// exact-duplicate-sentence dedup. The dedup step is meaning-affecting
    /// (a tool's description literally loses a sentence a human reader
    /// would see on that tool) even though the *information* survives
    /// elsewhere in the tool list — default OFF, opt in explicitly.
    Level2,
}

const I64_EXTREME_MAX: i64 = 9_007_199_254_740_991;
const I64_EXTREME_MIN: i64 = -9_007_199_254_740_991;

/// Apply `level` to a list of MCP tool objects
/// (`[{"name", "description", "inputSchema"}, ...]`), returning a new
/// list. Unknown/malformed entries are passed through unchanged rather
/// than dropped — this is a size optimization, not a validator.
pub fn diet_tools(tools: &[Value], level: DietLevel) -> Vec<Value> {
    if level == DietLevel::Level0 {
        return tools.to_vec();
    }

    let mut seen_sentences: std::collections::HashSet<String> = std::collections::HashSet::new();
    tools
        .iter()
        .map(|t| diet_one(t, level, &mut seen_sentences))
        .collect()
}

fn diet_one(
    tool: &Value,
    level: DietLevel,
    seen_sentences: &mut std::collections::HashSet<String>,
) -> Value {
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };
    let mut out = obj.clone();

    if level == DietLevel::Level2 {
        if let Some(Value::String(desc)) = out.get("description").cloned() {
            out.insert(
                "description".to_string(),
                Value::String(diet_description(&desc, seen_sentences)),
            );
        }
    }

    // `title` only when it's redundant with `description` being present
    // at all (a bare structural drop, per the level-1 spec) — MCP tool
    // objects don't usually carry a top-level `title`, but some servers
    // echo the schema's own `title` up a level.
    if out.contains_key("description") {
        out.remove("title");
    }

    if let Some(schema) = out.get("inputSchema") {
        let dieted = diet_schema(schema);
        out.insert("inputSchema".to_string(), dieted);
    }

    Value::Object(out)
}

/// Diet one JSON-Schema object (an `inputSchema`, or recursively a nested
/// `items`/`properties` entry).
fn diet_schema(schema: &Value) -> Value {
    let Some(obj) = schema.as_object() else {
        return schema.clone();
    };
    let mut out = obj.clone();

    out.remove("annotations");
    out.remove("$schema");

    if out.get("title").is_some() && out.get("description").is_some() {
        out.remove("title");
    }

    if matches!(out.get("additionalProperties"), Some(Value::Bool(false))) {
        out.remove("additionalProperties");
    }

    if matches!(out.get("required"), Some(Value::Array(a)) if a.is_empty()) {
        out.remove("required");
    }

    if matches!(out.get("minLength"), Some(Value::Number(n)) if n.as_u64() == Some(1)) {
        out.remove("minLength");
    }

    if let Some(Value::Number(n)) = out.get("minimum") {
        if n.as_i64() == Some(I64_EXTREME_MIN) {
            out.remove("minimum");
        }
    }
    if let Some(Value::Number(n)) = out.get("maximum") {
        if n.as_i64() == Some(I64_EXTREME_MAX) {
            out.remove("maximum");
        }
    }

    if let Some(props) = out.get("properties").and_then(|p| p.as_object()) {
        let mut new_props = serde_json::Map::new();
        for (k, v) in props {
            new_props.insert(k.clone(), diet_schema(v));
        }
        out.insert("properties".to_string(), Value::Object(new_props));
    }

    if let Some(items) = out.get("items") {
        out.insert("items".to_string(), diet_schema(items));
    }

    Value::Object(out)
}

/// Collapse whitespace runs, then drop any sentence that has already
/// appeared verbatim in an earlier tool's description this call
/// (first-seen tool keeps it, later tools lose it).
fn diet_description(desc: &str, seen_sentences: &mut std::collections::HashSet<String>) -> String {
    let collapsed = collapse_whitespace(desc);
    let sentences = split_sentences(&collapsed);

    let mut kept: Vec<&str> = Vec::new();
    for s in &sentences {
        let key = s.trim().to_string();
        if key.is_empty() {
            continue;
        }
        if seen_sentences.contains(&key) {
            continue;
        }
        seen_sentences.insert(key);
        kept.push(s);
    }

    kept.join(" ").trim().to_string()
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split on whitespace after `.`/`!`/`?`, keeping the terminator attached
/// to the sentence it ends. A `.` is NOT treated as a sentence end when
/// the word it closes is a single lowercase letter (`e.`, `g.`, `i.`,
/// `a.`, `p.`, `m.`, ...) — the common shape of abbreviations like
/// "e.g.", "i.e.", "a.m." this corpus actually contains; missing an
/// abbreviation boundary only costs a dedup opportunity, but wrongly
/// splitting inside one produces a bogus fragment (e.g. a stray "g.")
/// that can spuriously collide with unrelated text in another tool and
/// delete it — worse than leaving a duplicate sentence in place.
fn split_sentences(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut word = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if c.is_whitespace() {
            word.clear();
        } else if c != '.' && c != '!' && c != '?' {
            word.push(c);
        }

        let is_abbreviation = (c == '.') && word.len() == 1 && word.chars().all(|c| c.is_lowercase());
        if (c == '.' || c == '!' || c == '?')
            && !is_abbreviation
            && chars.get(i + 1).map(|c| c.is_whitespace()).unwrap_or(true)
        {
            out.push(cur.trim().to_string());
            cur = String::new();
            word.clear();
        }
        i += 1;
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}
