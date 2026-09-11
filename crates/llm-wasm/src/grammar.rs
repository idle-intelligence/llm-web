//! Schema-constrained decoding for xLAM-2 tool calls.
//!
//! A pure state machine (no GPU, no model) that decides, at every decoding
//! step, which vocab tokens are legal continuations of the output so far —
//! see `docs/ENGINE.md` §"Schema-constrained decoding" for the design and
//! how this is meant to hook into `sample.rs` (not wired up yet — another
//! worker owns the generate loop).
//!
//! xLAM-2 emits either a bare JSON array of tool calls
//! (`[{"name": ..., "arguments": {...}}, ...]`, see `tools.rs`) or plain
//! text. `Grammar::for_tools` builds the constraint for one turn (given the
//! tool schemas and the `IdValues` known so far); `GrammarState` walks a
//! generation through it token by token.
//!
//! Scope, intentionally: flat `object` schemas only (`properties` of
//! string/integer/boolean/array-of-string) — no nested objects, no arrays
//! of anything but strings. That matches every schema in
//! `fixtures/sonos/tools.json`.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------
// Tool schema (input side) — parsed from MCP `inputSchema` JSON Schema.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PropKind {
    String {
        enum_values: Option<Vec<String>>,
        min_length: usize,
    },
    Integer {
        minimum: Option<i64>,
        maximum: Option<i64>,
    },
    Boolean,
    ArrayOfString {
        min_items: usize,
    },
}

#[derive(Debug, Clone)]
pub struct Prop {
    pub name: String,
    pub required: bool,
    /// Name ends with `_id`/`_ids`, or description contains "ID" — see
    /// `docs/ENGINE.md` for why this is generic rather than Sonos-specific.
    pub is_id: bool,
    pub kind: PropKind,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub properties: Vec<Prop>,
}

impl Tool {
    /// Parse one MCP tool's `{"name": ..., "inputSchema": {...}}` shape
    /// (the `inputSchema` value only — `name` is passed separately since
    /// callers usually already have both).
    pub fn from_schema(name: &str, input_schema: &Value) -> Self {
        let required: BTreeSet<String> = input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut properties = Vec::new();
        if let Some(props) = input_schema.get("properties").and_then(|p| p.as_object()) {
            for (pname, pschema) in props {
                let ty = pschema.get("type").and_then(|t| t.as_str()).unwrap_or("string");
                let description = pschema
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                let is_id =
                    pname.ends_with("_id") || pname.ends_with("_ids") || description.contains("ID");
                let kind = match ty {
                    "integer" => PropKind::Integer {
                        minimum: pschema.get("minimum").and_then(|v| v.as_i64()),
                        maximum: pschema.get("maximum").and_then(|v| v.as_i64()),
                    },
                    "boolean" => PropKind::Boolean,
                    "array" => PropKind::ArrayOfString {
                        min_items: pschema
                            .get("minItems")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                    },
                    _ => PropKind::String {
                        enum_values: pschema.get("enum").and_then(|e| e.as_array()).map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        }),
                        min_length: pschema
                            .get("minLength")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                    },
                };
                properties.push(Prop {
                    name: pname.clone(),
                    required: required.contains(pname),
                    is_id,
                    kind,
                });
            }
        }
        Self {
            name: name.to_string(),
            properties,
        }
    }
}

/// Parse every entry of `fixtures/sonos/tools.json`-shaped
/// `[{"name": ..., "inputSchema": {...}}, ...]` into `Tool`s.
pub fn tools_from_json(list: &[Value]) -> Vec<Tool> {
    list.iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let schema = t.get("inputSchema")?;
            Some(Tool::from_schema(name, schema))
        })
        .collect()
}

// ---------------------------------------------------------------------
// IdValues — strings harvested from earlier tool results in the
// conversation, the only values a `*_id`/`*_ids` property may take.
//
// Values are bucketed by a "type token" derived from the key they were
// harvested under (`normalize_type_token`) so that, e.g., a `groupId` and
// a `playerId` never land in the same pool — see docs/ENGINE.md "Typed
// ids" for the regression (a player id satisfying a `group_id` property)
// this guards against. Values harvested from a key that isn't itself
// id-shaped (`is_id_key`) fall into the untyped `""` bucket, exactly like
// the old flat-set behaviour.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct IdValues(BTreeMap<String, BTreeSet<String>>);

impl IdValues {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Insert into the untyped (`""`) bucket — the same as harvesting a
    /// value from a key that isn't id-shaped. Test/debug convenience: a
    /// property whose typed bucket is empty falls back to this one (see
    /// `candidates_for`), so building an `IdValues` this way still works
    /// against `Grammar::for_tools` exactly as before typed buckets
    /// existed.
    pub fn insert(&mut self, s: impl Into<String>) {
        self.insert_typed("", s);
    }

    pub fn insert_typed(&mut self, token: &str, s: impl Into<String>) {
        self.0.entry(token.to_string()).or_default().insert(s.into());
    }

    pub fn is_empty(&self) -> bool {
        self.0.values().all(|set| set.is_empty())
    }

    /// Every value harvested so far, across every type bucket — test/debug
    /// helper; production code should go through `candidates_for` instead,
    /// which respects typing.
    pub fn sorted_vec(&self) -> Vec<String> {
        let mut all: BTreeSet<String> = BTreeSet::new();
        for set in self.0.values() {
            all.extend(set.iter().cloned());
        }
        all.into_iter().collect()
    }

    /// Candidate values for a schema property whose name normalises to
    /// `token` (see `normalize_type_token`): the typed bucket for `token`,
    /// or — only when that bucket is empty — the untyped `""` bucket. A
    /// property's typed bucket taking priority (even a non-empty one) is
    /// exactly what keeps a `player` id from ever satisfying a `group_id`
    /// property once *any* group id has been harvested.
    pub fn candidates_for(&self, token: &str) -> Vec<String> {
        let typed = self.0.get(token);
        match typed {
            Some(set) if !set.is_empty() => set.iter().cloned().collect(),
            _ => self.0.get("").map(|set| set.iter().cloned().collect()).unwrap_or_default(),
        }
    }

    /// Harvest id-shaped strings from one tool result: values of keys
    /// that look id-shaped (`is_id_key` — case-insensitive `id`/`ids`
    /// suffix, covering `Id`/`_id`/`ID`/`Ids`/`_ids`) go into the type
    /// bucket named by that key (`normalize_type_token`), unconditionally;
    /// any other string that generically *looks* like an id token
    /// (`looks_like_id`), regardless of its key, goes into the untyped
    /// `""` bucket exactly as before typed buckets existed.
    pub fn collect_from_result(&mut self, json: &Value) {
        self.walk(json);
    }

    fn walk(&mut self, v: &Value) {
        match v {
            Value::Object(map) => {
                for (k, val) in map {
                    if is_id_key(k) {
                        self.walk_id_value(val, &normalize_type_token(k));
                    } else {
                        self.walk(val);
                    }
                }
            }
            Value::Array(items) => {
                for it in items {
                    self.walk(it);
                }
            }
            Value::String(s) => {
                // MCP tool results commonly carry their payload as
                // `{"content":[{"type":"text","text":"<pretty-printed JSON
                // string>"}]}` (see `docs/ENGINE.md`'s "The id rule" and
                // the real-MCP-shape regression this guards against) —
                // mirrors `tools.rs::compact_embedded_json`'s detection of
                // the same shape. Try parsing every string value as JSON
                // first and recurse into it if it is one; only fall back to
                // treating the raw string itself as a candidate id when it
                // isn't (a real id string like `RINCON_KITCHEN01:1` never
                // parses as JSON, so this adds no false negatives).
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    self.walk(&parsed);
                } else if looks_like_id(s) {
                    self.insert_typed("", s.clone());
                }
            }
            _ => {}
        }
    }

    /// Walk a value known to sit directly under an id-shaped key
    /// (`is_id_key`): every string reached — directly, or as an element of
    /// an array (`player_ids: [...]`) — goes into `token`'s bucket
    /// unconditionally, no `looks_like_id` shape check (the key already
    /// told us it's an id). A string that turns out to be embedded JSON
    /// (the MCP `content`/`text` wrapper) is parsed and handed to the
    /// normal `walk`, which re-derives typing from its own keys — an
    /// id-shaped key never actually wraps a JSON-string payload in
    /// practice, but this keeps the two walks consistent if it ever did.
    fn walk_id_value(&mut self, v: &Value, token: &str) {
        match v {
            Value::String(s) => {
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    self.walk(&parsed);
                } else {
                    self.insert_typed(token, s.clone());
                }
            }
            Value::Array(items) => {
                for it in items {
                    self.walk_id_value(it, token);
                }
            }
            _ => self.walk(v),
        }
    }
}

/// Whether `key` looks id-shaped enough to harvest its value(s)
/// unconditionally (rather than only when the value itself
/// `looks_like_id`) — case-insensitive `id`/`ids` suffix, covering
/// `Id`/`_id`/`ID` and, for arrays of ids, `Ids`/`_ids`. Same
/// false-positive shape as any suffix heuristic (a key literally named
/// `valid` or `kids` would also match) — pre-existing for the singular
/// case, accepted here as generic and not Sonos-specific.
fn is_id_key(k: &str) -> bool {
    let lower = k.to_ascii_lowercase();
    lower.ends_with("id") || lower.ends_with("ids")
}

/// Normalises an id-shaped key (harvest side) or schema property name
/// (constraint side) to a short "type token" grouping ids of the same
/// kind — `groupId`/`group_id` and `player_ids`/`playerIds` normalise to
/// `"group"`/`"player"` respectively, so a `group_id` property is
/// restricted to values harvested from `groupId`-shaped keys, never
/// `playerId`-shaped ones (docs/ENGINE.md "Typed ids" — the `bloupblip`
/// regression this fixes: a player id satisfying `get_now_playing`'s
/// `group_id`).
///
/// Rule, in order: lowercase; strip the longest matching trailing suffix
/// among `_ids`, `ids`, `_id`, `id`; take the last `_`-separated segment
/// of what remains (drops a leading qualifier — `source_player_id` and
/// `destination_player_ids` both give `"player"`, `group_id` gives
/// `"group"`); strip any leftover non-alphanumeric characters; singularise
/// a final trailing `s`.
///
/// Known limits (documented, not fixed here — generic, not Sonos-specific):
/// a qualifier joined by camelCase rather than `_` (`sourcePlayerId`) isn't
/// split — the whole remainder becomes the token (`"sourceplayer"`), so it
/// won't line up with a plain `player_id`/`playerId` property. A name that
/// *is* only the suffix (`"id"`, `"ids"`) normalises to the empty string —
/// the untyped bucket. Singularisation is a bare trailing-`s` strip, not
/// real pluralisation, so it won't undo an irregular plural.
fn normalize_type_token(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let stripped = ["_ids", "ids", "_id", "id"]
        .iter()
        .find_map(|suffix| lower.strip_suffix(suffix))
        .unwrap_or(lower.as_str());
    let last_seg = stripped.rsplit('_').next().unwrap_or(stripped);
    let alnum: String = last_seg.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if alnum.len() > 1 && alnum.ends_with('s') {
        alnum[..alnum.len() - 1].to_string()
    } else {
        alnum
    }
}

/// Generic id-shape heuristic: no whitespace, reasonably long, and has the
/// "token with a separator" look of an opaque identifier (contains a digit
/// and one of `_`/`:`/`-`) — matches things like `RINCON_KITCHEN01:1`
/// without hardcoding any Sonos-specific pattern.
fn looks_like_id(s: &str) -> bool {
    s.len() >= 6
        && !s.contains(' ')
        && s.chars().any(|c| c.is_ascii_digit())
        && s.chars().any(|c| c == '_' || c == ':' || c == '-')
}

// ---------------------------------------------------------------------
// Grammar — tool schemas resolved against a fixed IdValues snapshot.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ResolvedKind {
    StringEnum(Vec<String>),
    StringFree(usize),
    Integer(Option<i64>, Option<i64>),
    Boolean,
    Array {
        elem_candidates: Option<Vec<String>>,
        min_items: usize,
    },
}

#[derive(Debug, Clone)]
struct ResolvedProp {
    required: bool,
    kind: ResolvedKind,
}

#[derive(Debug, Clone)]
struct CallableTool {
    props: Vec<ResolvedProp>,
    prop_names: Vec<String>,
}

/// The decoding-time constraint for one turn: which tool calls are
/// syntactically and semantically legal, given a fixed `IdValues` snapshot.
///
/// A tool whose required `*_id`/`*_ids` property has no known value is
/// dropped entirely — it cannot be called, so its name never appears in the
/// `"name"` choice at all (not merely rejected once `arguments` is wrong).
pub struct Grammar {
    tools: Vec<CallableTool>,
    tool_names: Vec<String>,
    name_lit: Vec<String>,
    arguments_lit: Vec<String>,
    /// When set, `Pos::Start` never enters the tool-call-array branch (a
    /// leading `[` is rejected outright) — every generation is forced into
    /// `Pos::FreeText` instead. Built by [`Grammar::text_only`] for the
    /// agent loops' forced-final-answer fallback (`docs/ENGINE.md` "Agent
    /// loop"): once retries are exhausted on a repeated or empty tool call,
    /// the model must produce a text answer, not another array.
    text_only: bool,
    /// When set, `Pos::Start` never enters the free-text branch (a leading
    /// non-`[` byte is rejected outright) — every generation is forced into
    /// the tool-call-array branch instead. Mirror image of `text_only`; set
    /// by [`Grammar::tools_only`] for both agent loops'
    /// `require_tool_call_first_step` (`docs/ENGINE.md` "Agent loop"): the
    /// first generation of a turn must at least attempt a tool call rather
    /// than refuse in prose (the observed `bloupblip` failure this guards
    /// against never looked anything up).
    tools_only: bool,
}

fn resolve_plain(k: &PropKind) -> ResolvedKind {
    match k {
        PropKind::String {
            enum_values,
            min_length,
        } => match enum_values {
            Some(v) => ResolvedKind::StringEnum(v.clone()),
            None => ResolvedKind::StringFree(*min_length),
        },
        PropKind::Integer { minimum, maximum } => ResolvedKind::Integer(*minimum, *maximum),
        PropKind::Boolean => ResolvedKind::Boolean,
        PropKind::ArrayOfString { min_items } => ResolvedKind::Array {
            elem_candidates: None,
            min_items: *min_items,
        },
    }
}

impl Grammar {
    pub fn for_tools(tools: &[Tool], id_values: &IdValues) -> Self {
        Self::build(tools, id_values, true)
    }

    /// Like [`Grammar::for_tools`], but never drops a property or tool for
    /// having no known id value — every `*_id`/`*_ids` property resolves
    /// as an ordinary free string instead of being restricted to (or
    /// dropped for lack of) `id_values`. Fail-open escape hatch for when
    /// the id-restricted grammar would leave the model nothing new to
    /// call (`docs/ENGINE.md` "Agent loop" — "fail-open"): both agent
    /// loops rebuild with this instead of `for_tools` for one step when
    /// every still-callable tool is a read tool already called this turn.
    pub fn for_tools_unrestricted_ids(tools: &[Tool]) -> Self {
        Self::build(tools, &IdValues::new(), false)
    }

    /// A grammar with no callable tools at all — `Pos::Start` rejects a
    /// leading `[` outright, so the only legal output is free text. Used by
    /// both agent loops' forced-final-answer fallback once retries are
    /// exhausted on a repeated or empty tool call (`docs/ENGINE.md` "Agent
    /// loop"): the model must answer, not emit another tool-call array.
    pub fn text_only() -> Self {
        Self {
            tools: Vec::new(),
            tool_names: Vec::new(),
            name_lit: vec!["name".to_string()],
            arguments_lit: vec!["arguments".to_string()],
            text_only: true,
            tools_only: false,
        }
    }

    /// Same schema as `self`, but forced into tool-call-only mode:
    /// `Pos::Start` never enters the free-text branch (a leading non-`[`
    /// byte is rejected), mirroring how [`Grammar::text_only`] forces the
    /// opposite. Unlike `text_only`, the tool set built by `for_tools` /
    /// `for_tools_unrestricted_ids` is kept — only the "answer in prose
    /// instead" escape hatch is removed. See `tools_only`'s field doc.
    pub fn tools_only(mut self) -> Self {
        self.tools_only = true;
        self
    }

    fn build(tools: &[Tool], id_values: &IdValues, restrict_ids: bool) -> Self {
        let mut resolved_tools = Vec::new();
        let mut tool_names = Vec::new();

        for t in tools {
            let mut callable = true;
            let mut props = Vec::new();
            let mut prop_names = Vec::new();
            for p in &t.properties {
                if p.is_id && restrict_ids {
                    // Candidates are typed by the PROPERTY NAME's own type
                    // token (`normalize_type_token`), not a flat pool of
                    // every id ever seen — see `IdValues::candidates_for`
                    // and docs/ENGINE.md "Typed ids".
                    let candidates = id_values.candidates_for(&normalize_type_token(&p.name));
                    if candidates.is_empty() {
                        // Property cannot be emitted at all. If it was
                        // required, the whole tool is uncallable.
                        if p.required {
                            callable = false;
                        }
                        continue;
                    }
                    let kind = match &p.kind {
                        PropKind::String { .. } => ResolvedKind::StringEnum(candidates),
                        PropKind::ArrayOfString { min_items } => ResolvedKind::Array {
                            elem_candidates: Some(candidates),
                            min_items: *min_items,
                        },
                        other => resolve_plain(other),
                    };
                    props.push(ResolvedProp {
                        required: p.required,
                        kind,
                    });
                    prop_names.push(p.name.clone());
                } else {
                    props.push(ResolvedProp {
                        required: p.required,
                        kind: resolve_plain(&p.kind),
                    });
                    prop_names.push(p.name.clone());
                }
            }
            if !callable {
                continue;
            }
            resolved_tools.push(CallableTool { props, prop_names });
            tool_names.push(t.name.clone());
        }

        Self {
            tools: resolved_tools,
            tool_names,
            name_lit: vec!["name".to_string()],
            arguments_lit: vec!["arguments".to_string()],
            text_only: false,
            tools_only: false,
        }
    }

    /// Whether `name` survived id-availability filtering (test/debug
    /// helper).
    pub fn can_call(&self, name: &str) -> bool {
        self.tool_names.iter().any(|n| n == name)
    }

    /// Every tool name that survived id-availability filtering — the set
    /// the model can actually start typing (`Grammar::can_call`'s data,
    /// exposed as a slice for the fail-open "is the model stuck" check —
    /// see `docs/ENGINE.md` "Agent loop").
    pub fn callable_tool_names(&self) -> &[String] {
        &self.tool_names
    }
}

// ---------------------------------------------------------------------
// Byte-level matcher primitives.
// ---------------------------------------------------------------------

/// Small fixed-capacity byte buffer, `Copy` so state transitions never
/// allocate — capacity comfortably covers tool names, property names,
/// enum values and id strings seen in `fixtures/sonos/tools.json`.
#[derive(Clone, Copy)]
struct SmallBuf {
    data: [u8; 64],
    len: u8,
}

impl SmallBuf {
    fn new() -> Self {
        Self { data: [0; 64], len: 0 }
    }
    fn push(self, b: u8) -> Option<Self> {
        if self.len as usize >= self.data.len() {
            return None;
        }
        let mut d = self;
        d.data[d.len as usize] = b;
        d.len += 1;
        Some(d)
    }
    fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }
}

/// Matches a quoted JSON string against a fixed candidate list (tool
/// names, property names, enum values, or id values) via a byte-by-byte
/// prefix walk — no allocation, no tries built up front (candidate counts
/// here are always tiny: at most 34 tool names).
#[derive(Clone, Copy)]
struct QuotedChoice<'g> {
    candidates: &'g [String],
    phase: QPhase,
}

#[derive(Clone, Copy)]
enum QPhase {
    BeforeQuote,
    InContent(SmallBuf),
}

enum QStep<'g> {
    More(QuotedChoice<'g>),
    Done(&'g str),
    Reject,
}

impl<'g> QuotedChoice<'g> {
    fn start(candidates: &'g [String]) -> Self {
        Self {
            candidates,
            phase: QPhase::BeforeQuote,
        }
    }

    fn step(self, b: u8) -> QStep<'g> {
        match self.phase {
            QPhase::BeforeQuote => {
                if b == b'"' {
                    QStep::More(Self {
                        candidates: self.candidates,
                        phase: QPhase::InContent(SmallBuf::new()),
                    })
                } else {
                    QStep::Reject
                }
            }
            QPhase::InContent(buf) => {
                if b == b'"' {
                    match self.candidates.iter().find(|c| c.as_bytes() == buf.as_bytes()) {
                        Some(c) => QStep::Done(c.as_str()),
                        None => QStep::Reject,
                    }
                } else {
                    match buf.push(b) {
                        Some(nb) => {
                            let prefix = nb.as_bytes();
                            let ok = self.candidates.iter().any(|c| {
                                let cb = c.as_bytes();
                                cb.len() >= prefix.len() && &cb[..prefix.len()] == prefix
                            });
                            if ok {
                                QStep::More(Self {
                                    candidates: self.candidates,
                                    phase: QPhase::InContent(nb),
                                })
                            } else {
                                QStep::Reject
                            }
                        }
                        None => QStep::Reject,
                    }
                }
            }
        }
    }
}

fn bool_advance(buf: SmallBuf, b: u8) -> Option<(SmallBuf, bool)> {
    let nb = buf.push(b)?;
    let s = nb.as_bytes();
    let matches_true = s.len() <= 4 && &b"true"[..s.len()] == s;
    let matches_false = s.len() <= 5 && &b"false"[..s.len()] == s;
    if matches_true {
        Some((nb, s.len() == 4))
    } else if matches_false {
        Some((nb, s.len() == 5))
    } else {
        None
    }
}

#[derive(Clone, Copy)]
enum ElemState<'g> {
    Enum(QuotedChoice<'g>),
    Free(u16),
}

// ---------------------------------------------------------------------
// Pos — the parser position. `Copy` throughout so `GrammarState::allowed`
// can trial-walk every vocab token's bytes from a cheap stack copy of the
// current position, without touching the real state.
// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Pos<'g> {
    Start,
    FreeText,

    ArrWs,
    NameKey(QuotedChoice<'g>),
    NameColon,
    NameValueStart(bool),
    ToolNameVal(QuotedChoice<'g>),
    AfterName(usize),
    ArgsKeyStart(usize, bool),
    ArgsKey(usize, QuotedChoice<'g>),
    ArgsColon(usize),
    ArgsValueStart(usize, bool),

    PropKeyOrClose(usize, u32),
    PropKeyStart(usize, u32, bool),
    PropKey(usize, u32, QuotedChoice<'g>),
    KeyColon(usize, u32, usize),
    ValueStart(usize, u32, usize, bool),

    ValStrEnum(usize, u32, usize, QuotedChoice<'g>),
    ValStrFree(usize, u32, usize, usize, u16, bool),
    ValBool(usize, u32, usize, SmallBuf),
    ValInt(usize, u32, usize, SmallBuf),

    ValArrElemOrClose(usize, u32, usize, u16),
    ValArrElem(usize, u32, usize, u16, ElemState<'g>),
    ValArrAfterElem(usize, u32, usize, u16),
    ValArrElemCommaWs(usize, u32, usize, u16, bool),

    AfterVal(usize, u32),
    CallClose,
    AfterCall,
    AfterCallCommaWs(bool),
    ArrClosed,
}

const WS: [u8; 4] = [b' ', b'\t', b'\n', b'\r'];

impl<'g> Pos<'g> {
    /// The `QuotedChoice` this position is inside, if any — every `Pos`
    /// variant that carries one wraps it directly (tool name, key names,
    /// enum/id values, enum-array elements). Used by `forced_bytes` to
    /// tell "genuinely only one candidate left" apart from "several
    /// candidates share this prefix and haven't diverged yet" (see that
    /// function's doc comment).
    fn quoted_choice(&self) -> Option<QuotedChoice<'g>> {
        match *self {
            Pos::NameKey(qc)
            | Pos::ToolNameVal(qc)
            | Pos::ArgsKey(_, qc)
            | Pos::PropKey(_, _, qc)
            | Pos::ValStrEnum(_, _, _, qc) => Some(qc),
            Pos::ValArrElem(_, _, _, _, ElemState::Enum(qc)) => Some(qc),
            _ => None,
        }
    }
}

impl Grammar {
    fn step<'g>(&'g self, pos: Pos<'g>, b: u8) -> Option<Pos<'g>> {
        match pos {
            Pos::Start => {
                if WS.contains(&b) {
                    Some(Pos::Start)
                } else if b == b'[' {
                    if self.text_only {
                        None
                    } else {
                        Some(Pos::ArrWs)
                    }
                } else if self.tools_only {
                    None
                } else {
                    Some(Pos::FreeText)
                }
            }
            Pos::FreeText => Some(Pos::FreeText),

            Pos::ArrWs => {
                if WS.contains(&b) {
                    Some(Pos::ArrWs)
                } else if b == b'{' {
                    Some(Pos::NameKey(QuotedChoice::start(&self.name_lit)))
                } else {
                    None
                }
            }
            Pos::NameKey(qc) => match qc.step(b) {
                QStep::More(nqc) => Some(Pos::NameKey(nqc)),
                QStep::Done(_) => Some(Pos::NameColon),
                QStep::Reject => None,
            },
            Pos::NameColon => {
                if b == b':' {
                    Some(Pos::NameValueStart(false))
                } else {
                    None
                }
            }
            Pos::NameValueStart(space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::NameValueStart(true));
                }
                match QuotedChoice::start(&self.tool_names).step(b) {
                    QStep::More(qc) => Some(Pos::ToolNameVal(qc)),
                    _ => None,
                }
            }
            Pos::ToolNameVal(qc) => match qc.step(b) {
                QStep::More(nqc) => Some(Pos::ToolNameVal(nqc)),
                QStep::Done(matched) => {
                    let idx = self.tool_names.iter().position(|n| n == matched)?;
                    Some(Pos::AfterName(idx))
                }
                QStep::Reject => None,
            },
            Pos::AfterName(tool) => {
                if b == b',' {
                    Some(Pos::ArgsKeyStart(tool, false))
                } else {
                    None
                }
            }
            Pos::ArgsKeyStart(tool, space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::ArgsKeyStart(tool, true));
                }
                match QuotedChoice::start(&self.arguments_lit).step(b) {
                    QStep::More(qc) => Some(Pos::ArgsKey(tool, qc)),
                    _ => None,
                }
            }
            Pos::ArgsKey(tool, qc) => match qc.step(b) {
                QStep::More(nqc) => Some(Pos::ArgsKey(tool, nqc)),
                QStep::Done(_) => Some(Pos::ArgsColon(tool)),
                QStep::Reject => None,
            },
            Pos::ArgsColon(tool) => {
                if b == b':' {
                    Some(Pos::ArgsValueStart(tool, false))
                } else {
                    None
                }
            }
            Pos::ArgsValueStart(tool, space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::ArgsValueStart(tool, true));
                }
                if b == b'{' {
                    Some(Pos::PropKeyOrClose(tool, 0))
                } else {
                    None
                }
            }

            Pos::PropKeyOrClose(tool, seen) => {
                if b == b'}' {
                    if self.required_satisfied(tool, seen) {
                        Some(Pos::CallClose)
                    } else {
                        None
                    }
                } else if b == b'"' && self.has_unseen_prop(tool, seen) {
                    Some(Pos::PropKey(
                        tool,
                        seen,
                        QuotedChoice {
                            candidates: &self.tools[tool].prop_names,
                            phase: QPhase::InContent(SmallBuf::new()),
                        },
                    ))
                } else {
                    None
                }
            }
            Pos::PropKeyStart(tool, seen, space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::PropKeyStart(tool, seen, true));
                }
                if b == b'"' {
                    Some(Pos::PropKey(
                        tool,
                        seen,
                        QuotedChoice {
                            candidates: &self.tools[tool].prop_names,
                            phase: QPhase::InContent(SmallBuf::new()),
                        },
                    ))
                } else {
                    None
                }
            }
            Pos::PropKey(tool, seen, qc) => match qc.step(b) {
                QStep::More(nqc) => Some(Pos::PropKey(tool, seen, nqc)),
                QStep::Done(matched) => {
                    let idx = self.tools[tool].prop_names.iter().position(|n| n == matched)?;
                    if seen & (1 << idx) != 0 {
                        return None; // duplicate key
                    }
                    Some(Pos::KeyColon(tool, seen, idx))
                }
                QStep::Reject => None,
            },
            Pos::KeyColon(tool, seen, prop) => {
                if b == b':' {
                    Some(Pos::ValueStart(tool, seen, prop, false))
                } else {
                    None
                }
            }
            Pos::ValueStart(tool, seen, prop, space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::ValueStart(tool, seen, prop, true));
                }
                match &self.tools[tool].props[prop].kind {
                    ResolvedKind::StringEnum(cands) => {
                        match QuotedChoice::start(cands).step(b) {
                            QStep::More(qc) => Some(Pos::ValStrEnum(tool, seen, prop, qc)),
                            _ => None,
                        }
                    }
                    ResolvedKind::StringFree(min_len) => {
                        if b == b'"' {
                            Some(Pos::ValStrFree(tool, seen, prop, *min_len, 0, false))
                        } else {
                            None
                        }
                    }
                    ResolvedKind::Boolean => {
                        let (nb, done) = bool_advance(SmallBuf::new(), b)?;
                        if done {
                            Some(Pos::AfterVal(tool, seen | (1 << prop)))
                        } else {
                            Some(Pos::ValBool(tool, seen, prop, nb))
                        }
                    }
                    ResolvedKind::Integer(..) => {
                        if b == b'-' || b.is_ascii_digit() {
                            let buf = SmallBuf::new().push(b)?;
                            Some(Pos::ValInt(tool, seen, prop, buf))
                        } else {
                            None
                        }
                    }
                    ResolvedKind::Array { .. } => {
                        if b == b'[' {
                            Some(Pos::ValArrElemOrClose(tool, seen, prop, 0))
                        } else {
                            None
                        }
                    }
                }
            }

            Pos::ValStrEnum(tool, seen, prop, qc) => match qc.step(b) {
                QStep::More(nqc) => Some(Pos::ValStrEnum(tool, seen, prop, nqc)),
                QStep::Done(_) => Some(Pos::AfterVal(tool, seen | (1 << prop))),
                QStep::Reject => None,
            },
            Pos::ValStrFree(tool, seen, prop, min_len, len, escaped) => {
                if escaped {
                    return Some(Pos::ValStrFree(tool, seen, prop, min_len, len + 1, false));
                }
                if b == b'\\' {
                    return Some(Pos::ValStrFree(tool, seen, prop, min_len, len, true));
                }
                if b == b'"' {
                    return if (len as usize) >= min_len {
                        Some(Pos::AfterVal(tool, seen | (1 << prop)))
                    } else {
                        None
                    };
                }
                if b < 0x20 {
                    return None;
                }
                Some(Pos::ValStrFree(tool, seen, prop, min_len, len + 1, false))
            }
            Pos::ValBool(tool, seen, prop, buf) => {
                let (nb, done) = bool_advance(buf, b)?;
                if done {
                    Some(Pos::AfterVal(tool, seen | (1 << prop)))
                } else {
                    Some(Pos::ValBool(tool, seen, prop, nb))
                }
            }
            Pos::ValInt(tool, seen, prop, buf) => {
                if b.is_ascii_digit() {
                    return buf.push(b).map(|nb| Pos::ValInt(tool, seen, prop, nb));
                }
                let s = std::str::from_utf8(buf.as_bytes()).ok()?;
                let n: i64 = s.parse().ok()?;
                let ResolvedKind::Integer(min, max) = &self.tools[tool].props[prop].kind else {
                    return None;
                };
                if let Some(mn) = min {
                    if n < *mn {
                        return None;
                    }
                }
                if let Some(mx) = max {
                    if n > *mx {
                        return None;
                    }
                }
                self.step(Pos::AfterVal(tool, seen | (1 << prop)), b)
            }

            Pos::ValArrElemOrClose(tool, seen, prop, count) => {
                let ResolvedKind::Array {
                    elem_candidates,
                    min_items,
                } = &self.tools[tool].props[prop].kind
                else {
                    return None;
                };
                if b == b']' {
                    return if count as usize >= *min_items {
                        Some(Pos::AfterVal(tool, seen | (1 << prop)))
                    } else {
                        None
                    };
                }
                if b == b'"' {
                    let es = match elem_candidates {
                        Some(cands) => ElemState::Enum(QuotedChoice {
                            candidates: cands,
                            phase: QPhase::InContent(SmallBuf::new()),
                        }),
                        None => ElemState::Free(0),
                    };
                    return Some(Pos::ValArrElem(tool, seen, prop, count, es));
                }
                None
            }
            Pos::ValArrElem(tool, seen, prop, count, es) => match es {
                ElemState::Enum(qc) => match qc.step(b) {
                    QStep::More(nqc) => {
                        Some(Pos::ValArrElem(tool, seen, prop, count, ElemState::Enum(nqc)))
                    }
                    QStep::Done(_) => Some(Pos::ValArrAfterElem(tool, seen, prop, count + 1)),
                    QStep::Reject => None,
                },
                ElemState::Free(len) => {
                    if b == b'"' {
                        return Some(Pos::ValArrAfterElem(tool, seen, prop, count + 1));
                    }
                    if b < 0x20 {
                        return None;
                    }
                    Some(Pos::ValArrElem(tool, seen, prop, count, ElemState::Free(len + 1)))
                }
            },
            Pos::ValArrAfterElem(tool, seen, prop, count) => {
                if b == b',' {
                    return Some(Pos::ValArrElemCommaWs(tool, seen, prop, count, false));
                }
                if b == b']' {
                    let ResolvedKind::Array { min_items, .. } = &self.tools[tool].props[prop].kind
                    else {
                        return None;
                    };
                    return if count as usize >= *min_items {
                        Some(Pos::AfterVal(tool, seen | (1 << prop)))
                    } else {
                        None
                    };
                }
                None
            }
            Pos::ValArrElemCommaWs(tool, seen, prop, count, space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::ValArrElemCommaWs(tool, seen, prop, count, true));
                }
                if b == b'"' {
                    let ResolvedKind::Array { elem_candidates, .. } =
                        &self.tools[tool].props[prop].kind
                    else {
                        return None;
                    };
                    let es = match elem_candidates {
                        Some(cands) => ElemState::Enum(QuotedChoice {
                            candidates: cands,
                            phase: QPhase::InContent(SmallBuf::new()),
                        }),
                        None => ElemState::Free(0),
                    };
                    return Some(Pos::ValArrElem(tool, seen, prop, count, es));
                }
                None
            }

            Pos::AfterVal(tool, seen) => match b {
                b',' if self.has_unseen_prop(tool, seen) => Some(Pos::PropKeyStart(tool, seen, false)),
                b'}' => {
                    if self.required_satisfied(tool, seen) {
                        Some(Pos::CallClose)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            Pos::CallClose => {
                if b == b'}' {
                    Some(Pos::AfterCall)
                } else {
                    None
                }
            }
            Pos::AfterCall => {
                if b == b',' {
                    Some(Pos::AfterCallCommaWs(false))
                } else if b == b']' {
                    Some(Pos::ArrClosed)
                } else {
                    None
                }
            }
            Pos::AfterCallCommaWs(space_used) => {
                if !space_used && b == b' ' {
                    return Some(Pos::AfterCallCommaWs(true));
                }
                if b == b'{' {
                    Some(Pos::NameKey(QuotedChoice::start(&self.name_lit)))
                } else {
                    None
                }
            }
            Pos::ArrClosed => {
                if WS.contains(&b) {
                    Some(Pos::ArrClosed)
                } else {
                    None
                }
            }
        }
    }

    fn required_satisfied(&self, tool: usize, seen: u32) -> bool {
        self.tools[tool]
            .props
            .iter()
            .enumerate()
            .all(|(i, p)| !p.required || (seen & (1 << i)) != 0)
    }

    /// Whether `tool` has at least one property not yet in `seen` — i.e.
    /// whether a `,` at `Pos::AfterVal` could ever lead anywhere. Without
    /// this check `,` was structurally always a legal one-byte transition
    /// at `AfterVal` (deferred rejection: the next property-key attempt
    /// would find no candidate names left and fail then), which is
    /// harmless for `TokenMask`-based sampling (a token spelling a bare
    /// `,` there just never wins against `}` in practice) but poisons
    /// `GrammarConstraint::forced_bytes`'s "exactly one legal next byte"
    /// test — it would see `,` as a live second option and refuse to force
    /// through `}` even when `}` is the only byte that can ever complete
    /// the grammar (`docs/ENGINE.md` "Schema-constrained decoding" /
    /// jump-forward).
    fn has_unseen_prop(&self, tool: usize, seen: u32) -> bool {
        self.tools[tool].props.len() > seen.count_ones() as usize
    }
}

// ---------------------------------------------------------------------
// Token-level API.
// ---------------------------------------------------------------------

/// Precomputed per-token byte strings for the whole vocab, plus the eos
/// ids — built once per generation (not once per step) from the
/// tokenizer.
pub struct TokenVocab {
    bytes: Vec<Vec<u8>>,
    eos_ids: Vec<u32>,
    all_mask: TokenMask,
}

impl TokenVocab {
    pub fn from_tokenizer(t: &crate::tokenizer::Tokenizer) -> Self {
        let n = t.vocab_size();
        let bytes: Vec<Vec<u8>> = (0..n as u32).map(|id| t.token_bytes(id)).collect();
        let eos_ids = t.eos_ids().to_vec();
        let all_mask = TokenMask::all_ones(n);
        Self {
            bytes,
            eos_ids,
            all_mask,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A bitset over vocab ids — the token mask a sampler would AND into its
/// logits before argmax/top-k (see `docs/ENGINE.md`; not wired into
/// `sample.rs` by this change).
#[derive(Clone)]
pub struct TokenMask {
    bits: Vec<u64>,
    len: usize,
}

impl TokenMask {
    fn new(len: usize) -> Self {
        Self {
            bits: vec![0u64; len.div_ceil(64)],
            len,
        }
    }

    /// Test/debug constructor: a mask over `len` ids with exactly
    /// `allowed` set. `pub(crate)` — `sample.rs`'s masked-sampling unit
    /// tests build fixture masks with this rather than going through a
    /// full `Grammar`/`GrammarState`.
    #[cfg(test)]
    pub(crate) fn from_allowed(len: usize, allowed: &[usize]) -> Self {
        let mut m = Self::new(len);
        for &i in allowed {
            m.set(i);
        }
        m
    }

    fn all_ones(len: usize) -> Self {
        let mut bits = vec![u64::MAX; len.div_ceil(64)];
        let rem = len % 64;
        if rem != 0 {
            if let Some(last) = bits.last_mut() {
                *last = (1u64 << rem) - 1;
            }
        }
        Self { bits, len }
    }

    fn set(&mut self, i: usize) {
        if i < self.len {
            self.bits[i / 64] |= 1 << (i % 64);
        }
    }

    pub fn is_allowed(&self, i: usize) -> bool {
        i < self.len && (self.bits[i / 64] >> (i % 64)) & 1 == 1
    }

    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Walks one generation through a `Grammar`, token by token. `Clone` so
/// `GrammarConstraint::forced_run` can trial-walk a scratch copy without
/// disturbing the real state.
#[derive(Clone)]
pub struct GrammarState<'g> {
    grammar: &'g Grammar,
    pos: Pos<'g>,
}

impl<'g> GrammarState<'g> {
    pub fn new(grammar: &'g Grammar) -> Self {
        Self {
            grammar,
            pos: Pos::Start,
        }
    }

    /// The set of vocab tokens that are valid continuations right now.
    ///
    /// Inside free text every token is valid (the constraint only kicks in
    /// for the tool-call-array branch), so that case returns the shared
    /// precomputed all-ones mask instead of re-walking the whole vocab.
    pub fn allowed(&self, vocab: &TokenVocab) -> TokenMask {
        if matches!(self.pos, Pos::FreeText) {
            return vocab.all_mask.clone();
        }
        let mut mask = TokenMask::new(vocab.len());
        for (id, bytes) in vocab.bytes.iter().enumerate() {
            let mut p = self.pos;
            let mut ok = true;
            for &b in bytes {
                match self.grammar.step(p, b) {
                    Some(np) => p = np,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                mask.set(id);
            }
        }
        if self.is_complete() {
            for &e in &vocab.eos_ids {
                mask.set(e as usize);
            }
        }
        mask
    }

    /// Advance the real state through one accepted token's bytes. Callers
    /// are expected to only pass tokens that `allowed()` marked valid; an
    /// out-of-grammar token is a no-op (state left unchanged) rather than a
    /// panic.
    pub fn advance(&mut self, token_id: u32, vocab: &TokenVocab) {
        if vocab.eos_ids.contains(&token_id) {
            return;
        }
        if let Some(bytes) = vocab.bytes.get(token_id as usize) {
            let mut p = self.pos;
            for &b in bytes {
                match self.grammar.step(p, b) {
                    Some(np) => p = np,
                    None => return,
                }
            }
            self.pos = p;
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.pos, Pos::FreeText | Pos::ArrClosed)
    }

    /// Feed raw bytes through the grammar directly, bypassing token
    /// boundaries. Used by tests (and debugging) to check acceptance or
    /// rejection at an exact character position, independent of how the
    /// tokenizer happens to chunk that text into BPE tokens. Stops and
    /// returns `false` at the first rejected byte, leaving `self`
    /// positioned at the last byte that *was* accepted.
    pub fn feed_bytes(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            match self.grammar.step(self.pos, b) {
                Some(np) => self.pos = np,
                None => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------
// Constraint — the trait `model.rs::generate` and `agent.rs::Agent` drive
// the decode loop through (see docs/ENGINE.md "Schema-constrained
// decoding" for the jump-forward wiring). `GrammarConstraint` is the only
// implementation today; the trait exists so `generate`'s signature doesn't
// hard-code `GrammarState`/`TokenVocab`.
// ---------------------------------------------------------------------

pub trait Constraint {
    /// The current token mask, or `None` if this constraint imposes no
    /// restriction at all right now (a `GrammarConstraint` always returns
    /// `Some` — even free text has a mask, just an all-ones one).
    fn allowed(&self) -> Option<&TokenMask>;

    /// Advance past one *accepted* token (same no-op-on-rejected-token
    /// contract as `GrammarState::advance`).
    fn advance(&mut self, token: u32);

    /// The maximal run of tokens, starting from the current position,
    /// where the mask allows exactly one token at each step — computed by
    /// walking a scratch clone of the state forward, never touching the
    /// real one. Excludes any trailing EOS (a single-EOS-allowed mask ends
    /// the run without EOS in it, since EOS is the decode loop's own stop
    /// signal, not a token to jump-forward over). `None`/empty means there
    /// is no such run right now (the caller falls back to a normal masked
    /// sampling step).
    fn forced_run(&self) -> Option<Vec<u32>>;

    fn is_complete(&self) -> bool;
}

/// `Constraint` adapter over `GrammarState` + `TokenVocab`. Owns an
/// eagerly-computed mask for the current position (recomputed on every
/// `advance`) so `allowed()` can be a cheap `&self` reference return
/// instead of re-walking the vocab per call.
pub struct GrammarConstraint<'g> {
    state: GrammarState<'g>,
    tokenizer: &'g crate::tokenizer::Tokenizer,
    vocab: &'g TokenVocab,
    mask: TokenMask,
}

impl<'g> GrammarConstraint<'g> {
    pub fn new(grammar: &'g Grammar, tokenizer: &'g crate::tokenizer::Tokenizer, vocab: &'g TokenVocab) -> Self {
        let state = GrammarState::new(grammar);
        let mask = state.allowed(vocab);
        Self {
            state,
            tokenizer,
            vocab,
            mask,
        }
    }

    /// The longest span of bytes, starting from the current position,
    /// where the grammar's byte-level DFA (`Grammar::step`) accepts
    /// exactly one continuation byte at each step — i.e. the literal text
    /// the grammar has already fully committed to, independent of how the
    /// tokenizer happens to chunk it into BPE pieces (see `forced_run`'s
    /// doc comment on why token-level "exactly one legal token" is far too
    /// strict: most positions inside a forced literal have *several*
    /// legal vocab tokens simultaneously, one per differently-lengthed BPE
    /// segmentation of the same forced substring).
    fn forced_bytes(&self) -> Vec<u8> {
        let grammar = self.state.grammar;
        let mut pos = self.state.pos;
        let mut out = Vec::new();
        loop {
            // Never force byte-level literal *into* a `QuotedChoice`'s
            // content while more than one candidate is still live for it
            // — even when the byte-DFA below finds "exactly one legal next
            // byte" at every position of the shared prefix (e.g. the
            // `RINCON_` seven bytes common to every device id), forcing
            // that prefix re-encodes it in isolation
            // (`forced_run`/`Tokenizer::encode("RINCON_", ...)`), which can
            // land on a token boundary the model never actually produces
            // mid-generation (its natural tokenization of the *full*
            // candidate string may split differently — BPE merges aren't
            // prefix-invariant). That off-distribution KV state then
            // biases the very next masked-decode step toward whichever
            // candidate the tokenizer's own artifacts happen to favor,
            // independent of what the model actually meant — this is what
            // broke the four id-choice cases in `constrained43` (session 12
            // eval addendum, `docs/BENCHMARKS.md`). So: stop the forced
            // run right at the opening quote of any multi-candidate
            // `QuotedChoice` and hand off to ordinary per-token masked
            // decoding (`GrammarState::allowed`), which already restricts
            // to exactly the tokens that are prefix-compatible with *some*
            // live candidate — no token-boundary distortion, since the
            // model is choosing tokens the same way it always does.
            if let Some(qc) = pos.quoted_choice() {
                if qc.candidates.len() > 1 && matches!(qc.phase, QPhase::InContent(_)) {
                    break;
                }
            }
            let mut found: Option<(u8, Pos<'g>)> = None;
            let mut count = 0u32;
            for b in 0u16..256 {
                if let Some(np) = grammar.step(pos, b as u8) {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                    found = Some((b as u8, np));
                }
            }
            let Some((b, np)) = (if count == 1 { found } else { None }) else {
                break;
            };
            out.push(b);
            pos = np;
        }
        out
    }
}

impl<'g> Constraint for GrammarConstraint<'g> {
    fn allowed(&self) -> Option<&TokenMask> {
        Some(&self.mask)
    }

    fn advance(&mut self, token: u32) {
        self.state.advance(token, self.vocab);
        self.mask = self.state.allowed(self.vocab);
    }

    /// See `forced_bytes`: finds the unambiguous forced literal (byte
    /// level, tokenizer-independent), then re-encodes it with the real
    /// tokenizer to get the token ids the model's own BPE would produce
    /// for that text — the ids `model.rs`'s jump-forward prefill actually
    /// feeds through the KV cache. Byte-level BPE tokenization is a pure
    /// function of the input bytes (no surrounding-context dependence), so
    /// re-encoding the forced literal in isolation reproduces exactly what
    /// the tokenizer would have chunked it into inline — verified
    /// defensively by decoding the result back and comparing bytes; a
    /// mismatch (tokenizer round-trip surprise) falls back to `None`
    /// (normal masked per-token sampling for this step) rather than risk
    /// feeding the model text it didn't actually mean to commit to.
    fn forced_run(&self) -> Option<Vec<u32>> {
        let bytes = self.forced_bytes();
        if bytes.is_empty() {
            return None;
        }
        let text = std::str::from_utf8(&bytes).ok()?;
        let tokens = self.tokenizer.encode(text, false).ok()?;
        if tokens.is_empty() {
            return None;
        }
        let round_trip = self.tokenizer.decode(&tokens, false).ok()?;
        if round_trip.as_bytes() != bytes.as_slice() {
            return None;
        }
        Some(tokens)
    }

    fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
}
