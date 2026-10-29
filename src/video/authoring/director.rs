//! Director's Story messages, byte for byte: comfyui-helto-director
//! `shared/prompt_studio.py` (`_inventory_user_prompt`,
//! `_story_outline_user_prompt`, `_story_scene_user_prompt`),
//! `shared/contracts/story_outline.py` (the outline schema) and the browser
//! helpers in `web/timeline/prompt_studio*.js` that shape their input.
//! `tests/fixtures/director-story-messages.json` is the source oracle's output.
use crate::video::text::PythonText;
use std::fmt::Write;

/// A JSON value as Python holds it: keys in insertion order, ints and floats apart.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    /// An integer beyond i64, kept as its digits as Python keeps it.
    BigInt(String),
    Str(String),
    List(Vec<Py>),
    Obj(Vec<(String, Py)>),
}
enum Style {
    Default,
    Compact,
    Indent(usize),
}
impl Py {
    pub fn obj<const N: usize>(rows: [(&str, Py); N]) -> Self {
        Self::Obj(rows.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    pub fn str(text: impl Into<String>) -> Self {
        Self::Str(text.into())
    }
    pub fn strs<S: AsRef<str>>(items: impl IntoIterator<Item = S>) -> Self {
        Self::List(items.into_iter().map(|s| Self::str(s.as_ref())).collect())
    }
    /// A number that passed through a browser's JSON, where integral values
    /// lose their fraction.
    pub fn js(n: f64) -> Self {
        if n.fract() == 0. && n.abs() < 9_007_199_254_740_992. {
            Self::Int(n as i64)
        } else {
            Self::Float(n)
        }
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Py> {
        match self {
            Self::Obj(rows) => rows.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn push(&mut self, key: &str, value: Py) {
        if let Self::Obj(rows) = self {
            rows.push((key.into(), value));
        }
    }
    /// `json.dumps(value, ensure_ascii=False)`.
    pub fn dumps(&self) -> String {
        self.render(Style::Default)
    }
    /// `separators=(",", ":")`; also the browser's `JSON.stringify` for values
    /// without floats.
    pub fn compact(&self) -> String {
        self.render(Style::Compact)
    }
    /// `indent=2`.
    pub fn indented(&self) -> String {
        self.render(Style::Indent(0))
    }
    pub fn get(&self, key: &str) -> Option<&Py> {
        match self {
            Self::Obj(rows) => rows.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// `value[key]` of a value Imagine built itself.
    pub fn at(&self, key: &str) -> &Py {
        self.get(key).unwrap_or(&Py::Null)
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[Py]> {
        match self {
            Self::List(items) => Some(items),
            _ => None,
        }
    }
    pub fn keys(&self) -> Option<Vec<&str>> {
        match self {
            Self::Obj(rows) => Some(rows.iter().map(|(k, _)| k.as_str()).collect()),
            _ => None,
        }
    }
    /// Python's `str(value or "")` for the strings Director reads.
    pub fn text(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Int(n) if *n != 0 => n.to_string(),
            Self::Float(f) if *f != 0. => py_float(*f),
            Self::Bool(true) => "True".into(),
            _ => String::new(),
        }
    }
    /// Python's `float(value)` for a JSON number; `None` for anything else.
    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Int(n) => Some(*n as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }
    /// `json.loads`, keeping key order and Python's int/float distinction.
    pub fn parse(text: &str) -> Option<Py> {
        let mut parser = Parser {
            text: text.as_bytes(),
            pos: 0,
        };
        let value = parser.value()?;
        parser.space();
        (parser.pos == parser.text.len()).then_some(value)
    }
    fn render(&self, style: Style) -> String {
        let mut out = String::new();
        self.write(&mut out, &style);
        out
    }
    fn write(&self, out: &mut String, style: &Style) {
        let (item, key) = match style {
            Style::Default => (", ", ": "),
            Style::Compact => (",", ":"),
            Style::Indent(_) => (",", ": "),
        };
        let open = |out: &mut String, level: usize| {
            if let Style::Indent(_) = style {
                out.push('\n');
                out.push_str(&"  ".repeat(level));
            }
        };
        let level = match style {
            Style::Indent(level) => *level,
            _ => 0,
        };
        let inner = match style {
            Style::Indent(level) => Style::Indent(level + 1),
            Style::Default => Style::Default,
            Style::Compact => Style::Compact,
        };
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(n) => write!(out, "{n}").unwrap(),
            Self::BigInt(digits) => out.push_str(digits),
            Self::Float(f) => out.push_str(&py_float(*f)),
            Self::Str(s) => escape(out, s),
            Self::List(items) if items.is_empty() => out.push_str("[]"),
            Self::Obj(rows) if rows.is_empty() => out.push_str("{}"),
            Self::List(items) => {
                out.push('[');
                for (i, value) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    open(out, level + 1);
                    value.write(out, &inner);
                }
                open(out, level);
                out.push(']');
            }
            Self::Obj(rows) => {
                out.push('{');
                for (i, (k, value)) in rows.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    open(out, level + 1);
                    escape(out, k);
                    out.push_str(key);
                    value.write(out, &inner);
                }
                open(out, level);
                out.push('}');
            }
        }
    }
}
struct Parser<'a> {
    text: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while self
            .text
            .get(self.pos)
            .is_some_and(|b| b" \t\n\r".contains(b))
        {
            self.pos += 1;
        }
    }
    fn eat(&mut self, literal: &str) -> bool {
        let found = self.text[self.pos..].starts_with(literal.as_bytes());
        if found {
            self.pos += literal.len();
        }
        found
    }
    fn value(&mut self) -> Option<Py> {
        self.space();
        match *self.text.get(self.pos)? {
            b'{' => {
                self.pos += 1;
                let mut rows = vec![];
                self.space();
                if self.eat("}") {
                    return Some(Py::Obj(rows));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    if !self.eat(":") {
                        return None;
                    }
                    // As a Python dict: a repeated key keeps its place, the last value wins.
                    let value = self.value()?;
                    match rows.iter_mut().find(|(k, _)| *k == key) {
                        Some(row) => row.1 = value,
                        None => rows.push((key, value)),
                    }
                    self.space();
                    if self.eat("}") {
                        return Some(Py::Obj(rows));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                let mut items = vec![];
                self.space();
                if self.eat("]") {
                    return Some(Py::List(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    if self.eat("]") {
                        return Some(Py::List(items));
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
            }
            b'"' => self.string().map(Py::Str),
            b't' => self.eat("true").then_some(Py::Bool(true)),
            b'f' => self.eat("false").then_some(Py::Bool(false)),
            b'n' => self.eat("null").then_some(Py::Null),
            b'N' => self.eat("NaN").then_some(Py::Float(f64::NAN)),
            b'I' => self.eat("Infinity").then_some(Py::Float(f64::INFINITY)),
            _ if self.eat("-Infinity") => Some(Py::Float(f64::NEG_INFINITY)),
            _ => {
                // Python's `-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`.
                let start = self.pos;
                let digits = |p: &mut Self| {
                    let from = p.pos;
                    while p.text.get(p.pos).is_some_and(u8::is_ascii_digit) {
                        p.pos += 1;
                    }
                    p.pos > from
                };
                self.eat("-");
                let leading = self
                    .text
                    .get(self.pos)
                    .is_some_and(|b| (b'1'..=b'9').contains(b));
                if !(self.eat("0") || leading && digits(self)) {
                    return None;
                }
                let mut float = false;
                let mark = self.pos;
                if self.eat(".") {
                    if digits(self) {
                        float = true;
                    } else {
                        self.pos = mark;
                    }
                }
                let mark = self.pos;
                if self
                    .text
                    .get(self.pos)
                    .is_some_and(|b| *b == b'e' || *b == b'E')
                {
                    self.pos += 1;
                    if !self.eat("-") {
                        self.eat("+");
                    }
                    if digits(self) {
                        float = true;
                    } else {
                        self.pos = mark;
                    }
                }
                let number = std::str::from_utf8(&self.text[start..self.pos]).ok()?;
                if float {
                    number.parse().ok().map(Py::Float)
                } else {
                    Some(
                        number
                            .parse()
                            .map_or_else(|_| Py::BigInt(number.into()), Py::Int),
                    )
                }
            }
        }
    }
    fn string(&mut self) -> Option<String> {
        let start = self.pos;
        if !self.eat("\"") {
            return None;
        }
        let mut escaped = false;
        while let Some(&b) = self.text.get(self.pos) {
            self.pos += 1;
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => {
                    return serde_json::from_slice(&self.text[start..self.pos]).ok();
                }
                _ => {}
            }
        }
        None
    }
}
/// A JSON string without ASCII escaping, as Python and browsers write it.
fn escape(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}
/// Python's `repr(float)`: the shortest round-trip digits, positional from
/// 1e-4 up to 1e16.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0. { "Infinity" } else { "-Infinity" }.into();
    }
    let scientific = format!("{f:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits = mantissa.replace('.', "");
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let text = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(-point as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{text}")
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        )
    }
}
/// Python's `re.IGNORECASE` match of the fence language names, where `ſ`
/// matches `s` and the Kelvin sign `k`.
fn fence_language(rest: &str) -> usize {
    for word in ["text", "json", "markdown"] {
        let mut chars = rest.char_indices();
        let mut matched = true;
        for expected in word.chars() {
            match chars.next() {
                Some((_, c))
                    if c.to_ascii_lowercase() == expected
                        || (expected == 's' && c == 'ſ')
                        || (expected == 'k' && c == '\u{212a}') => {}
                _ => {
                    matched = false;
                    break;
                }
            }
        }
        if matched {
            return chars.next().map_or(rest.len(), |(i, _)| i);
        }
    }
    0
}
/// `_strip_outer_markdown`: a whole-text code fence, then a whole-text
/// inline code span, each unwrapped and stripped.
pub fn strip_outer_markdown(value: &str) -> String {
    let mut text = value.pytrim().to_owned();
    if text.len() >= 6 && text.starts_with("```") && text.ends_with("```") {
        let mut rest = &text[3..text.len() - 3];
        rest = &rest[fence_language(rest)..];
        rest = rest.trim_start_matches([' ', '\t']);
        rest = rest
            .strip_prefix("\r\n")
            .or_else(|| rest.strip_prefix('\n'))
            .unwrap_or(rest);
        rest = rest
            .strip_suffix("\r\n")
            .or_else(|| rest.strip_suffix('\n'))
            .unwrap_or(rest);
        text = rest.pytrim().to_owned();
    }
    if text.len() >= 2
        && text.starts_with('`')
        && text.ends_with('`')
        && !text[1..text.len() - 1].contains('`')
    {
        text = text[1..text.len() - 1].pytrim().to_owned();
    }
    text
}
/// `clean_text_response`: the unwrapped text without any backtick; `None`
/// when nothing remains.
pub fn clean_text_response(value: &str) -> Option<String> {
    let text = strip_outer_markdown(value)
        .replace('`', "")
        .pytrim()
        .to_owned();
    (!text.is_empty()).then_some(text)
}
/// `parse_json_object`: the unwrapped text as JSON, else its span from the
/// first `{` to the last `}`; the root must be an object.
pub fn parse_json_object(value: &str) -> Option<Py> {
    let text = strip_outer_markdown(value);
    let parsed = Py::parse(&text).or_else(|| {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        (end > start).then(|| Py::parse(&text[start..=end]))?
    })?;
    matches!(parsed, Py::Obj(_)).then_some(parsed)
}
/// JavaScript's `String.prototype.trim`.
pub fn js_trim(text: &str) -> &str {
    text.trim_matches(|c: char| {
        matches!(
            c,
            '\u{9}'..='\u{d}'
                | ' '
                | '\u{a0}'
                | '\u{1680}'
                | '\u{2000}'..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
        )
    })
}
/// Director's FNV-1a revision hash over UTF-16 code units, in base 36.
pub fn stable_hash(text: &str) -> String {
    let mut hash: u32 = 2166136261;
    for unit in text.encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(16777619);
    }
    let mut digits = vec![];
    loop {
        digits.push(char::from_digit(hash % 36, 36).unwrap());
        hash /= 36;
        if hash == 0 {
            break;
        }
    }
    digits.iter().rev().collect()
}
/// `localeCompare` for reference tags: punctuation before digits before
/// letters, letters ignoring case; then lower case first.
pub fn locale_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let primary = |c: char| -> (u8, char) {
        match c {
            '_' => (0, c),
            '-' => (1, c),
            '.' => (2, c),
            '@' => (3, c),
            '0'..='9' => (4, c),
            c if c.is_alphabetic() => (5, c.to_lowercase().next().unwrap_or(c)),
            c => (6, c),
        }
    };
    let key = |s: &str| s.chars().map(primary).collect::<Vec<_>>();
    key(a)
        .cmp(&key(b))
        .then_with(|| {
            let tertiary = |s: &str| s.chars().map(|c| c.is_uppercase()).collect::<Vec<_>>();
            tertiary(a).cmp(&tertiary(b))
        })
        .then_with(|| a.cmp(b))
}

pub const MAX_SCENE_SECONDS: i64 = 15;
pub const MAX_CAST: usize = 64;
const MAX_CAST_FIELD_CHARS: usize = 4000;
const MAX_STORY_NAMES: usize = 31;
const MAX_STORY_NAME_CHARS: usize = 200;
pub const MAX_CONTINUATION_FIELD_CHARS: usize = 16000;
pub const MAX_CONTINUATION_SECTIONS: usize = 128;
pub const MAX_CONTINUATION_JSON_CHARS: usize = 100000;
pub const DEPICTED_MOMENT: &str = "depicted_moment";
pub const STATE_TRANSITION: &str = "state_transition";

/// One visual beat the outline must place, in Director's order.
#[derive(Debug, Clone, PartialEq)]
pub struct Beat {
    pub key: String,
    pub image_ids: Vec<String>,
    pub role: &'static str,
}
/// The request-specific facts of Director's `StoryOutlineModelContract`, for
/// its default Scene policy.
#[derive(Debug, Clone, Default)]
pub struct Contract {
    pub setting_ids: Vec<String>,
    pub beats: Vec<Beat>,
    pub dialogue: bool,
    /// The roster whose opening appearance the outline must resolve.
    pub opening: Option<Vec<OpeningCharacter>>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct OpeningCharacter {
    pub character_id: String,
    pub reference_tag: String,
    pub fallback_wardrobe: String,
}
pub const OPENING_SOURCES: [&str; 5] = [
    "user_instruction",
    "latest_moment",
    "earlier_continuity",
    "default_wardrobe",
    "not_stated",
];
pub const EXPOSURES: [&str; 4] = ["not_stated", "clothed", "partially_clothed", "nude"];
/// Appearance lists and their item limits.
pub const APPEARANCE_LISTS: [(&str, usize); 4] = [
    ("garments", 12),
    ("worn_items", 12),
    ("temporary_traits", 20),
    ("visible_body_state", 20),
];
pub const APPEARANCE_FACT_CHARS: usize = 4000;
const APPEARANCE_CHARACTER_CHARS: usize = 16000;
const FALLBACK_WARDROBE_CHARS: usize = 2000;
const CHARACTER_ID_CHARS: usize = 300;
pub const MAX_PROTECTED_FACTS: usize = 128;
const PROTECTED_FACT_CHARS: usize = 16000;
const GROUNDING_BYTES: usize = 2097152;
/// `_opening_appearance_schema`.
fn opening_schema(count: usize) -> Py {
    let fact = || {
        Py::obj([
            ("type", Py::str("string")),
            ("minLength", Py::Int(1)),
            ("maxLength", Py::Int(APPEARANCE_FACT_CHARS as i64)),
        ])
    };
    let mut properties = Py::obj([
        (
            "character_id",
            Py::obj([("type", Py::str("string")), ("minLength", Py::Int(1))]),
        ),
        (
            "reference_tag",
            Py::obj([("type", Py::str("string")), ("minLength", Py::Int(1))]),
        ),
        ("source", Py::obj([("enum", Py::strs(OPENING_SOURCES))])),
        ("exposure", Py::obj([("enum", Py::strs(EXPOSURES))])),
    ]);
    for (field, maximum) in APPEARANCE_LISTS {
        properties.push(
            field,
            Py::obj([
                ("type", Py::str("array")),
                ("maxItems", Py::Int(maximum as i64)),
                ("items", fact()),
            ]),
        );
    }
    Py::obj([
        ("type", Py::str("object")),
        ("additionalProperties", Py::Bool(false)),
        ("required", required(&["schema_version", "characters"])),
        (
            "properties",
            Py::obj([
                ("schema_version", Py::obj([("const", Py::str("1.0"))])),
                (
                    "characters",
                    Py::obj([
                        ("type", Py::str("array")),
                        ("minItems", Py::Int(count as i64)),
                        ("maxItems", Py::Int(count as i64)),
                        (
                            "items",
                            Py::obj([
                                ("type", Py::str("object")),
                                ("additionalProperties", Py::Bool(false)),
                                (
                                    "required",
                                    required(&[
                                        "character_id",
                                        "reference_tag",
                                        "source",
                                        "exposure",
                                        "garments",
                                        "worn_items",
                                        "temporary_traits",
                                        "visible_body_state",
                                    ]),
                                ),
                                ("properties", properties),
                            ]),
                        ),
                    ]),
                ),
            ]),
        ),
    ])
}
/// Where Director found a context or answer invalid, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Invalid {
    pub path: String,
    pub reason: &'static str,
}
fn invalid<T>(path: impl Into<String>, reason: &'static str) -> Result<T, Invalid> {
    Err(Invalid {
        path: path.into(),
        reason,
    })
}
/// Director's `_REFERENCE_TAG_RE` for grounding rosters.
fn roster_tag(tag: &str) -> bool {
    reference_tag(tag) && tag.chars().count() <= 129
}
fn digest(value: &Py) -> bool {
    value.as_str().is_some_and(|d| {
        d.strip_prefix("sha256:").is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    })
}
fn sorted_keys(value: &Py) -> Py {
    match value {
        Py::Obj(rows) => {
            let mut rows = rows
                .iter()
                .map(|(k, v)| (k.clone(), sorted_keys(v)))
                .collect::<Vec<_>>();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            Py::Obj(rows)
        }
        Py::List(items) => Py::List(items.iter().map(sorted_keys).collect()),
        other => other.clone(),
    }
}
/// `_bounded_string_list`.
fn bounded_list(value: &Py, path: &str, maximum: usize) -> Result<Vec<String>, Invalid> {
    let Some(items) = value.as_list() else {
        return invalid(path, "must be a bounded array");
    };
    let mut out: Vec<String> = vec![];
    for (i, item) in items.iter().enumerate() {
        let Some(text) = item.as_str() else {
            return invalid(format!("{path}[{i}]"), "must be a string");
        };
        let text = text.pytrim();
        if text.is_empty() {
            continue;
        }
        if text.chars().count() > APPEARANCE_FACT_CHARS {
            return invalid(format!("{path}[{i}]"), "is too long");
        }
        if !out.iter().any(|t| t == text) {
            out.push(text.into());
        }
    }
    if out.len() > maximum {
        return invalid(path, "must be a bounded array");
    }
    Ok(out)
}
/// `_normalize_appearance`.
fn appearance(value: &Py, path: &str) -> Result<Py, Invalid> {
    let keys = [
        "exposure",
        "garments",
        "worn_items",
        "temporary_traits",
        "visible_body_state",
    ];
    if value.keys().is_none() {
        return invalid(path, "must be an object");
    }
    if !exact_keys(value, &keys) {
        return invalid(path, "must use the exact appearance shape");
    }
    let exposure = value.at("exposure").text();
    if !EXPOSURES.contains(&exposure.as_str()) {
        return invalid(format!("{path}.exposure"), "uses an unsupported value");
    }
    let mut out = Py::obj([("exposure", Py::Str(exposure))]);
    let mut total = 0;
    for (field, maximum) in APPEARANCE_LISTS {
        let items = bounded_list(value.at(field), &format!("{path}.{field}"), maximum)?;
        total += items.iter().map(|i| i.chars().count()).sum::<usize>();
        out.push(field, Py::strs(&items));
    }
    if total > APPEARANCE_CHARACTER_CHARS {
        return invalid(path, "exceeds the published per-character appearance limit");
    }
    Ok(out)
}
/// `normalize_grounding_context`; paths are relative to the context (`$`).
pub fn normalize_grounding(value: &Py) -> Result<Py, Invalid> {
    if *value == Py::Null {
        return Ok(Py::Null);
    }
    if value.keys().is_none() {
        return invalid("$", "must be an object");
    }
    if sorted_keys(value).compact().len() > GROUNDING_BYTES {
        return invalid("$", "exceeds the published UTF-8 payload limit");
    }
    let allowed = [
        "schema_version",
        "source_digest",
        "characters",
        "protected_facts",
    ];
    if value.keys().unwrap().iter().any(|k| !allowed.contains(k)) {
        return invalid("$", "contains unsupported fields");
    }
    if value.at("schema_version").text() != "1.0" {
        return invalid("$.schema_version", "must be 1.0");
    }
    if !digest(value.at("source_digest")) {
        return invalid("$.source_digest", "must be a canonical SHA-256 digest");
    }
    let rows = match value.at("characters").as_list() {
        Some(rows) if rows.len() <= MAX_CAST => rows,
        _ => return invalid("$.characters", "must be a bounded array"),
    };
    let mut ids = std::collections::BTreeSet::new();
    let mut aliases = std::collections::BTreeSet::new();
    let mut characters = vec![];
    for (i, row) in rows.iter().enumerate() {
        let path = format!("$.characters[{i}]");
        if row.keys().is_none() {
            return invalid(path, "must be an object");
        }
        if !exact_keys(
            row,
            &["character_id", "reference_tag", "current_appearance"],
        ) {
            return invalid(path, "must use the exact character shape");
        }
        let id = row.at("character_id").text();
        let tag = row.at("reference_tag").text();
        if !(1..=CHARACTER_ID_CHARS).contains(&id.chars().count()) {
            return invalid(format!("{path}.character_id"), "is invalid");
        }
        if !roster_tag(&tag) {
            return invalid(format!("{path}.reference_tag"), "is invalid");
        }
        if !ids.insert(id.clone()) {
            return invalid(format!("{path}.character_id"), "must be unique");
        }
        if !aliases.insert(tag.to_lowercase()) {
            return invalid(format!("{path}.reference_tag"), "must be unique");
        }
        characters.push(Py::obj([
            ("character_id", Py::Str(id)),
            ("reference_tag", Py::Str(tag)),
            (
                "current_appearance",
                appearance(
                    row.at("current_appearance"),
                    &format!("{path}.current_appearance"),
                )?,
            ),
        ]));
    }
    let facts = match value.get("protected_facts") {
        None => &[][..],
        Some(Py::List(facts)) if facts.len() <= MAX_PROTECTED_FACTS => facts,
        Some(_) => return invalid("$.protected_facts", "must be a bounded array"),
    };
    let mut protected_facts = vec![];
    for (i, fact) in facts.iter().enumerate() {
        let text = fact.text().pytrim().to_owned();
        if text.is_empty() || text.chars().count() > PROTECTED_FACT_CHARS {
            return invalid(format!("$.protected_facts[{i}]"), "is empty or too long");
        }
        protected_facts.push(text);
    }
    Ok(Py::obj([
        ("schema_version", Py::str("1.0")),
        ("source_digest", value.at("source_digest").clone()),
        ("characters", Py::List(characters)),
        ("protected_facts", Py::strs(&protected_facts)),
    ]))
}
/// `normalize_opening_appearance_context` against the normalized grounding.
pub fn normalize_opening(value: &Py, grounding: &Py) -> Result<Py, Invalid> {
    if *value == Py::Null {
        return Ok(Py::Null);
    }
    if value.keys().is_none() {
        return invalid("$", "must be an object");
    }
    if !exact_keys(value, &["schema_version", "source_digest", "characters"]) {
        return invalid("$", "must use the exact context shape");
    }
    if value.at("schema_version").text() != "1.0" {
        return invalid("$.schema_version", "must be 1.0");
    }
    if !digest(value.at("source_digest")) {
        return invalid("$.source_digest", "must be a canonical SHA-256 digest");
    }
    let grounded = *grounding != Py::Null;
    if grounded && value.at("source_digest") != grounding.at("source_digest") {
        return invalid(
            "$.source_digest",
            "must match the structured grounding source digest",
        );
    }
    let rows = match value.at("characters").as_list() {
        Some(rows) if rows.len() <= MAX_CAST => rows,
        _ => return invalid("$.characters", "must be a bounded array"),
    };
    let roster = grounding.at("characters").as_list().unwrap_or_default();
    if grounded && rows.len() != roster.len() {
        return invalid("$.characters", "must match the structured grounding roster");
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut aliases = std::collections::BTreeSet::new();
    let mut characters = vec![];
    for (i, row) in rows.iter().enumerate() {
        let path = format!("$.characters[{i}]");
        if row.keys().is_none() {
            return invalid(path, "must be an object");
        }
        if !exact_keys(row, &["character_id", "reference_tag", "fallback_wardrobe"]) {
            return invalid(path, "must use the exact character shape");
        }
        let id = row.at("character_id").text();
        let tag = row.at("reference_tag").text();
        if !(1..=CHARACTER_ID_CHARS).contains(&id.chars().count()) {
            return invalid(format!("{path}.character_id"), "is invalid");
        }
        if !roster_tag(&tag) {
            return invalid(format!("{path}.reference_tag"), "is invalid");
        }
        let Some(wardrobe) = row.at("fallback_wardrobe").as_str() else {
            return invalid(format!("{path}.fallback_wardrobe"), "must be a string");
        };
        let wardrobe = wardrobe.pytrim();
        if wardrobe.chars().count() > FALLBACK_WARDROBE_CHARS {
            return invalid(
                format!("{path}.fallback_wardrobe"),
                "exceeds the published default wardrobe limit",
            );
        }
        if !ids.insert(id.clone()) {
            return invalid(format!("{path}.character_id"), "must be unique");
        }
        if !aliases.insert(tag.to_lowercase()) {
            return invalid(format!("{path}.reference_tag"), "must be unique");
        }
        if grounded && roster[i].at("character_id").text() != id {
            return invalid(
                format!("{path}.character_id"),
                "must match the structured grounding roster and order",
            );
        }
        if grounded && roster[i].at("reference_tag").text() != tag {
            return invalid(
                format!("{path}.reference_tag"),
                "must match the structured grounding roster and order",
            );
        }
        characters.push(Py::obj([
            ("character_id", Py::Str(id)),
            ("reference_tag", Py::Str(tag)),
            ("fallback_wardrobe", Py::str(wardrobe)),
        ]));
    }
    Ok(Py::obj([
        ("schema_version", Py::str("1.0")),
        ("source_digest", value.at("source_digest").clone()),
        ("characters", Py::List(characters)),
    ]))
}
/// The outline answer's `opening_appearance` against its schema and roster
/// (`validate_story_outline_contract`), before references are restored.
pub fn validate_opening(value: &Py, roster: &[OpeningCharacter]) -> Result<(), Invalid> {
    let path = "$.opening_appearance";
    if value.keys().is_none() {
        return invalid(path, "must be object");
    }
    let keys = value.keys().unwrap();
    for key in ["schema_version", "characters"] {
        if !keys.contains(&key) {
            return invalid(format!("{path}.{key}"), "is required");
        }
    }
    if keys
        .iter()
        .any(|k| !["schema_version", "characters"].contains(k))
    {
        return invalid(path, "has properties that are not allowed");
    }
    if *value.at("schema_version") != Py::str("1.0") {
        return invalid(format!("{path}.schema_version"), "must equal 1.0");
    }
    let Some(rows) = value.at("characters").as_list() else {
        return invalid(format!("{path}.characters"), "must be array");
    };
    if rows.len() != roster.len() {
        return invalid(format!("{path}.characters"), "must contain every character");
    }
    let fields = [
        "character_id",
        "reference_tag",
        "source",
        "exposure",
        "garments",
        "worn_items",
        "temporary_traits",
        "visible_body_state",
    ];
    for (i, (row, expected)) in rows.iter().zip(roster).enumerate() {
        let path = format!("{path}.characters[{i}]");
        let Some(keys) = row.keys() else {
            return invalid(path, "must be object");
        };
        if let Some(key) = fields.iter().find(|k| !keys.contains(k)) {
            return invalid(format!("{path}.{key}"), "is required");
        }
        if keys.iter().any(|k| !fields.contains(k)) {
            return invalid(path, "has properties that are not allowed");
        }
        for key in ["character_id", "reference_tag"] {
            if row.at(key).as_str().is_none_or(str::is_empty) {
                return invalid(format!("{path}.{key}"), "must be a non-empty string");
            }
        }
        if !row
            .at("source")
            .as_str()
            .is_some_and(|s| OPENING_SOURCES.contains(&s))
        {
            return invalid(format!("{path}.source"), "must use an allowed value");
        }
        if !row
            .at("exposure")
            .as_str()
            .is_some_and(|s| EXPOSURES.contains(&s))
        {
            return invalid(format!("{path}.exposure"), "must use an allowed value");
        }
        for (field, maximum) in APPEARANCE_LISTS {
            let Some(items) = row.at(field).as_list() else {
                return invalid(format!("{path}.{field}"), "must be array");
            };
            if items.len() > maximum {
                return invalid(format!("{path}.{field}"), "has too many items");
            }
            for (j, item) in items.iter().enumerate() {
                if !item
                    .as_str()
                    .is_some_and(|s| (1..=APPEARANCE_FACT_CHARS).contains(&s.chars().count()))
                {
                    return invalid(
                        format!("{path}.{field}[{j}]"),
                        "must be a string of allowed length",
                    );
                }
            }
        }
        if row.at("character_id").text() != expected.character_id {
            return invalid(
                format!("{path}.character_id"),
                "must preserve the supplied character ID and order",
            );
        }
        if row.at("reference_tag").text() != expected.reference_tag {
            return invalid(
                format!("{path}.reference_tag"),
                "must preserve the supplied reference tag and order",
            );
        }
        if row.at("source").as_str() == Some("default_wardrobe")
            && expected.fallback_wardrobe.is_empty()
        {
            return invalid(
                format!("{path}.source"),
                "cannot use default_wardrobe without a supplied fallback",
            );
        }
    }
    Ok(())
}
fn string() -> Py {
    Py::obj([("type", Py::str("string"))])
}
fn required(keys: &[&str]) -> Py {
    Py::strs(keys)
}
/// `story_outline_json_schema(contract)`.
pub fn outline_schema(contract: &Contract) -> Py {
    let mut section = Py::obj([
        ("type", Py::str("object")),
        ("additionalProperties", Py::Bool(false)),
        ("required", required(&["prompt", "weight"])),
        (
            "properties",
            Py::obj([
                (
                    "prompt",
                    Py::obj([("type", Py::str("string")), ("minLength", Py::Int(1))]),
                ),
                (
                    "weight",
                    Py::obj([
                        ("type", Py::str("number")),
                        ("exclusiveMinimum", Py::Int(0)),
                    ]),
                ),
            ]),
        ),
    ]);
    if contract.dialogue {
        section.get_mut("properties").unwrap().push(
            "dialogue_intent",
            Py::obj([
                ("type", Py::str("object")),
                ("additionalProperties", Py::Bool(false)),
                ("required", required(&["participants", "purpose"])),
                (
                    "properties",
                    Py::obj([
                        (
                            "participants",
                            Py::obj([
                                ("type", Py::str("array")),
                                ("minItems", Py::Int(2)),
                                ("maxItems", Py::Int(2)),
                                (
                                    "items",
                                    Py::obj([
                                        ("type", Py::str("string")),
                                        ("minLength", Py::Int(1)),
                                    ]),
                                ),
                            ]),
                        ),
                        (
                            "purpose",
                            Py::obj([("type", Py::str("string")), ("minLength", Py::Int(1))]),
                        ),
                    ]),
                ),
            ]),
        );
    }
    let scene = Py::obj([
        ("type", Py::str("object")),
        ("additionalProperties", Py::Bool(false)),
        (
            "required",
            required(&[
                "title",
                "summary",
                "description",
                "ambient_sound",
                "non_diegetic_music",
                "weight",
                "boundary_from_previous",
                "setting_reference_matches",
                "sections",
            ]),
        ),
        (
            "properties",
            Py::obj([
                ("title", string()),
                ("summary", string()),
                ("description", string()),
                ("ambient_sound", string()),
                ("non_diegetic_music", string()),
                (
                    "weight",
                    Py::obj([
                        ("type", Py::str("number")),
                        ("exclusiveMinimum", Py::Int(0)),
                    ]),
                ),
                (
                    "boundary_from_previous",
                    Py::obj([("enum", Py::strs(["hard_cut", "continuous_shot"]))]),
                ),
                (
                    "setting_reference_matches",
                    Py::obj([
                        ("type", Py::str("object")),
                        ("additionalProperties", Py::Bool(false)),
                        ("required", Py::strs(&contract.setting_ids)),
                        (
                            "properties",
                            Py::Obj(
                                contract
                                    .setting_ids
                                    .iter()
                                    .map(|id| (id.clone(), Py::obj([("type", Py::str("boolean"))])))
                                    .collect(),
                            ),
                        ),
                    ]),
                ),
                (
                    "sections",
                    Py::obj([
                        ("type", Py::str("array")),
                        ("minItems", Py::Int(1)),
                        ("items", section),
                    ]),
                ),
            ]),
        ),
    ]);
    let mut beat_key = string();
    if !contract.beats.is_empty() {
        beat_key.push("enum", Py::strs(contract.beats.iter().map(|b| &b.key)));
    }
    let placement = Py::obj([
        ("type", Py::str("object")),
        ("additionalProperties", Py::Bool(false)),
        (
            "required",
            required(&[
                "beat_key",
                "scene_ordinal",
                "section_ordinal",
                "semantic_role",
            ]),
        ),
        (
            "properties",
            Py::obj([
                ("beat_key", beat_key),
                (
                    "scene_ordinal",
                    Py::obj([("type", Py::str("integer")), ("minimum", Py::Int(1))]),
                ),
                (
                    "section_ordinal",
                    Py::obj([("type", Py::str("integer")), ("minimum", Py::Int(1))]),
                ),
                (
                    "semantic_role",
                    Py::obj([("enum", Py::strs([DEPICTED_MOMENT, STATE_TRANSITION]))]),
                ),
            ]),
        ),
    ]);
    let beats = contract.beats.len() as i64;
    let mut schema = Py::obj([
        ("type", Py::str("object")),
        ("additionalProperties", Py::Bool(false)),
        ("required", required(&["scenes", "visual_placements"])),
        (
            "properties",
            Py::obj([
                (
                    "scenes",
                    Py::obj([
                        ("type", Py::str("array")),
                        ("minItems", Py::Int(1)),
                        ("items", scene),
                    ]),
                ),
                (
                    "visual_placements",
                    Py::obj([
                        ("type", Py::str("array")),
                        ("minItems", Py::Int(beats)),
                        ("maxItems", Py::Int(beats)),
                        ("items", placement),
                    ]),
                ),
            ]),
        ),
    ]);
    if let Some(roster) = &contract.opening {
        if let Some(Py::List(keys)) = schema.get_mut("required") {
            keys.push(Py::str("opening_appearance"));
        }
        schema
            .get_mut("properties")
            .unwrap()
            .push("opening_appearance", opening_schema(roster.len()));
    }
    schema
}
/// `story_outline_json_schema_text(contract)`, also the system prompt's
/// `{{STORY_OUTLINE_JSON_SCHEMA}}`.
pub fn schema_text(contract: &Contract) -> String {
    outline_schema(contract).compact()
}
fn sections(rows: &[(&str, String)]) -> String {
    rows.iter()
        .map(|(label, value)| format!("{label}:\n{value}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}
/// One row of `_story_prompt`: a labelled section or an XML block.
enum Row {
    Labelled(&'static str, String),
    Block(&'static str, String),
}
fn story_prompt(rows: Vec<Row>) -> String {
    rows.into_iter()
        .map(|row| match row {
            Row::Labelled(label, value) => format!("{label}:\n{value}"),
            Row::Block(tag, value) => format!("<{tag}>\n{value}\n</{tag}>"),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
/// `_inventory_user_prompt` for one visual request descriptor.
pub fn inventory_user(id: &str, note: &str, label: &str, usage: &str, instruction: &str) -> String {
    let or = |value: &str, fallback: &str| {
        if value.is_empty() {
            fallback.to_owned()
        } else {
            value.to_owned()
        }
    };
    sections(&[
        (
            "Task",
            "Write one short factual planning note for this image. Return only the note.".into(),
        ),
        ("Stable image ID", or(id, "image")),
        ("User label or cue", or(note, &or(label, "None"))),
        ("Usage", or(usage, "story_reference")),
        ("Additional user instruction", or(instruction, "None")),
    ])
}
/// The outline message's inputs, after reference protection.
pub struct Outline<'a> {
    pub story: &'a str,
    pub instruction: &'a str,
    pub cast: &'a Py,
    pub contract: &'a Contract,
    pub continuation: &'a Py,
    pub duration_seconds: f64,
    pub music: bool,
    pub inventory: &'a Py,
    pub beats: &'a Py,
    pub grounding: &'a Py,
    pub opening: &'a Py,
}
/// `_story_outline_user_prompt` without a one-Scene policy.
pub fn outline_user(input: &Outline) -> String {
    let contract = input.contract;
    let mut scene = Py::obj([
        ("title", Py::str("Short title")),
        ("summary", Py::str("Story purpose")),
        ("description", Py::str("Brief visual and continuity plan")),
        ("ambient_sound", Py::str("Brief diegetic sound direction")),
        (
            "non_diegetic_music",
            Py::str("Brief score direction, or empty when music is disabled"),
        ),
        ("weight", Py::Int(1)),
        ("boundary_from_previous", Py::str("hard_cut")),
        (
            "setting_reference_matches",
            Py::Obj(
                contract
                    .setting_ids
                    .iter()
                    .map(|id| (id.clone(), Py::Bool(true)))
                    .collect(),
            ),
        ),
        (
            "sections",
            Py::List(vec![Py::obj([
                (
                    "prompt",
                    Py::str("Generic chronological source description"),
                ),
                ("weight", Py::Int(1)),
            ])]),
        ),
    ]);
    let tags = match input.cast {
        Py::List(rows) => rows
            .iter()
            .filter_map(|row| match row {
                Py::Obj(fields) => fields.iter().find(|(k, _)| k == "reference_tag"),
                _ => None,
            })
            .filter_map(|(_, v)| match v {
                Py::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => vec![],
    };
    if let Some(first) = tags.first() {
        let (summary, description, prompt) = if let Some(second) = tags.get(1) {
            (
                format!("{first} and {second} complete the Story's central interaction."),
                format!(
                    "{first} performs the planned action involving {second}, followed by the Story's resulting state."
                ),
                format!("{first} performs one chronological Story action involving {second}."),
            )
        } else {
            (
                format!("{first} completes the Story's central action."),
                format!(
                    "{first} performs the planned action, followed by the Story's resulting state."
                ),
                format!("{first} performs one chronological Story action."),
            )
        };
        *scene.get_mut("summary").unwrap() = Py::Str(summary);
        *scene.get_mut("description").unwrap() = Py::Str(description);
        if let Some(Py::List(rows)) = scene.get_mut("sections") {
            *rows[0].get_mut("prompt").unwrap() = Py::Str(prompt);
        }
    }
    let duration = input.duration_seconds.clamp(0.25, 86400.);
    let mut scenes = vec![scene.clone()];
    if duration > MAX_SCENE_SECONDS as f64 {
        let mut second = scene;
        *second.get_mut("title").unwrap() = Py::str("Another short title");
        *second.get_mut("boundary_from_previous").unwrap() = Py::str("hard_cut");
        *second.get_mut("setting_reference_matches").unwrap() = Py::Obj(
            contract
                .setting_ids
                .iter()
                .map(|id| (id.clone(), Py::Bool(false)))
                .collect(),
        );
        scenes.push(second);
    }
    if contract.dialogue
        && let Some(Py::List(rows)) = scenes[0].get_mut("sections")
    {
        rows[0].push(
            "dialogue_intent",
            Py::obj([
                (
                    "participants",
                    Py::strs([
                        "exact protected cast marker 1",
                        "exact protected cast marker 2",
                    ]),
                ),
                (
                    "purpose",
                    Py::str(
                        "Brief dramatic purpose, or omit dialogue_intent when silence is better",
                    ),
                ),
            ]),
        );
    }
    let mut shape = Py::obj([
        ("scenes", Py::List(scenes)),
        (
            "visual_placements",
            Py::List(
                contract
                    .beats
                    .iter()
                    .map(|beat| {
                        Py::obj([
                            ("beat_key", Py::str(&beat.key)),
                            ("scene_ordinal", Py::Int(1)),
                            ("section_ordinal", Py::Int(1)),
                            ("semantic_role", Py::str(beat.role)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    if let Some(roster) = &contract.opening {
        shape.push(
            "opening_appearance",
            Py::obj([
                ("schema_version", Py::str("1.0")),
                (
                    "characters",
                    Py::List(
                        roster
                            .iter()
                            .map(|c| {
                                Py::obj([
                                    ("character_id", Py::str(&c.character_id)),
                                    ("reference_tag", Py::str(&c.reference_tag)),
                                    ("source", Py::str("not_stated")),
                                    ("exposure", Py::str("not_stated")),
                                    ("garments", Py::List(vec![])),
                                    ("worn_items", Py::List(vec![])),
                                    ("temporary_traits", Py::List(vec![])),
                                    ("visible_body_state", Py::List(vec![])),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        );
    }
    let minimum = ((duration / MAX_SCENE_SECONDS as f64).ceil() as i64).max(1);
    let top = if contract.opening.is_some() {
        "The only top-level properties are \"scenes\", \"visual_placements\", and \"opening_appearance\". "
    } else {
        "The only top-level properties are \"scenes\" and \"visual_placements\". "
    };
    let mut rows = vec![
        Row::Labelled(
            "Task",
            "Plan as many Scenes and Sections as the story needs. Return one JSON object and no markdown.".into(),
        ),
        Row::Labelled(
            "Array placement rule",
            format!("{top}Put every Scene object inside the single scenes array in playback order. Do not generate internal Scene or Section IDs, raw image IDs, frame bindings, or max_turns."),
        ),
    ];
    if !tags.is_empty() {
        rows.push(Row::Labelled(
            "Character references in generated prose",
            "The example below illustrates reference syntax only; use the supplied Story for actual actions and only the characters involved. When naming a selected character in Scene summary, Scene description, or any Section prompt, copy that character's exact selected_cast.reference_tag instead of a story_names value. Correct reference_tag values in opening_appearance do not replace the required markers in Scene and Section prose. Names inside authored quoted dialogue may remain unchanged.".into(),
        ));
    }
    if contract.opening.is_some() {
        rows.push(Row::Labelled(
            "Opening appearance example",
            "The opening_appearance values in the example illustrate structure only. They are not an assessment of the characters. Resolve every character's actual opening appearance from the supplied Story and Additional instruction. Use not_stated only when the evidence leaves exposure unknown. A supplied fallback wardrobe does not establish that its garments are currently worn and must never override established nudity or partial clothing.".into(),
        ));
    }
    rows.extend([
        Row::Labelled("Minimal JSON shape", shape.dumps()),
        Row::Labelled("Exact JSON Schema", schema_text(contract)),
        // `min(86400, number)` keeps Python's int bound when it applies.
        Row::Labelled(
            "Total duration seconds",
            if input.duration_seconds >= 86400. {
                "86400".into()
            } else {
                py_float(duration)
            },
        ),
        Row::Labelled(
            "Maximum Scene duration seconds",
            MAX_SCENE_SECONDS.to_string(),
        ),
        Row::Labelled("Minimum Scene count", minimum.to_string()),
        Row::Labelled(
            "Include non-diegetic music",
            if input.music { "Yes" } else { "No" }.into(),
        ),
        Row::Labelled("Image inventory", input.inventory.dumps()),
        Row::Labelled("Required image order and visual beats", input.beats.dumps()),
        Row::Block("selected_cast", input.cast.dumps()),
        Row::Block("continuation_context", input.continuation.dumps()),
    ]);
    if *input.opening != Py::Null {
        rows.push(Row::Block(
            "opening_appearance_context",
            input.opening.dumps(),
        ));
    }
    if *input.grounding != Py::Null {
        rows.push(Row::Block("grounding_context", input.grounding.dumps()));
    }
    rows.extend([
        Row::Block("additional_instruction", input.instruction.into()),
        Row::Block("story", input.story.into()),
    ]);
    story_prompt(rows)
}
/// The Scene expansion message's inputs, after reference protection; JSON
/// blocks are already rendered so that an agent's bundle can hold slots.
pub struct Expansion<'a> {
    pub scene_json: &'a str,
    /// The visual context rule.
    pub rule: &'a str,
    pub instruction: &'a str,
    pub cast: &'a str,
    pub previous_summary: &'a str,
    pub previous: &'a str,
    pub next_summary: &'a str,
    pub context_cast: &'a str,
    /// With dialogue, the Scene's `dialogue_timing`.
    pub timing: Option<&'a str>,
    pub grounding: Option<&'a str>,
}
pub const SCENE_IMAGES: &str = "Actual assigned images are supplied. Treat them as visual truth; focus on action, camera, timing, interaction, and change instead of detailed environment description.";
pub const SCENE_NO_IMAGES: &str = "No images are supplied. Provide detailed environment, spatial layout, appearance, lighting, and atmosphere.";
/// `_story_scene_user_prompt` without a visual-binding sidecar.
pub fn scene_user(input: &Expansion) -> String {
    let mut rows = vec![
        Row::Labelled(
            "Task",
            "Expand this one Scene and its Sections. Return one JSON object matching the supplied Scene shape and no markdown.".into(),
        ),
        Row::Labelled("Visual context rule", input.rule.into()),
        Row::Block("previous_scene_summary", input.previous_summary.into()),
        Row::Block("previous_scene_context", input.previous.into()),
        Row::Block("next_scene_summary", input.next_summary.into()),
        Row::Block("selected_cast", input.cast.into()),
        Row::Block("continuity_cast", input.context_cast.into()),
    ];
    if let Some(timing) = input.timing {
        rows.push(Row::Block("dialogue_timing", timing.into()));
    }
    if let Some(grounding) = input.grounding {
        rows.push(Row::Block("grounding_context", grounding.into()));
    }
    rows.extend([
        Row::Block("additional_instruction", input.instruction.into()),
        Row::Block("scene_json", input.scene_json.into()),
    ]);
    story_prompt(rows)
}
/// One selected cast member as the browser holds it.
pub struct CastMember<'a> {
    pub title: &'a str,
    pub alias: &'a str,
    pub description: &'a str,
    pub language: &'a str,
    pub voice: &'a str,
}
/// `storyCastPromptRecords`.
pub fn cast_records(members: &[CastMember]) -> Py {
    let mut aliases = std::collections::BTreeSet::new();
    let mut rows = vec![];
    for member in members {
        let alias = js_trim(member.alias).to_lowercase();
        if alias.is_empty() || !aliases.insert(alias) {
            continue;
        }
        let mut names: Vec<&str> = vec![];
        for name in [js_trim(member.title), js_trim(member.alias)] {
            if !name.is_empty() && !names.contains(&name) {
                names.push(name);
            }
        }
        rows.push(Py::obj([
            ("story_names", Py::strs(names)),
            (
                "reference_tag",
                Py::Str(format!(
                    "@{}",
                    imagine_video_contract::creative::reference_alias(member.alias)
                )),
            ),
            ("description", Py::str(js_trim(member.description))),
            ("default_language", Py::str(js_trim(member.language))),
            ("voice_guidance", Py::str(js_trim(member.voice))),
            ("required", Py::Bool(true)),
        ]));
    }
    Py::List(rows)
}
/// `storyCastPromptRecordsForScene`: the records a Scene's JSON names.
pub fn scene_cast(records: &Py, scene: &Py) -> Py {
    let used = imagine_video_contract::references::aliases(&scene.compact());
    Py::List(
        records
            .as_list()
            .unwrap_or_default()
            .iter()
            .filter(|r| {
                imagine_video_contract::references::aliases(&r.at("reference_tag").text())
                    .first()
                    .is_some_and(|a| used.contains(a))
            })
            .cloned()
            .collect(),
    )
}
fn cast_field(value: &Py) -> String {
    value
        .text()
        .pytrim()
        .chars()
        .take(MAX_CAST_FIELD_CHARS)
        .collect()
}
/// Director's `REFERENCE_RE.fullmatch`.
fn reference_tag(tag: &str) -> bool {
    let mut chars = tag.chars();
    chars.next() == Some('@')
        && chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
}
fn casefold(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            'ß' => vec!['s', 's'],
            'ς' => vec!['σ'],
            c => c.to_lowercase().collect(),
        })
        .collect()
}
/// `normalize_story_cast`.
pub fn normalize_cast(value: &Py) -> Vec<Py> {
    let mut aliases = std::collections::BTreeSet::new();
    let mut out = vec![];
    for row in value.as_list().unwrap_or_default().iter().take(MAX_CAST) {
        if row.keys().is_none() {
            continue;
        }
        let tag = row.at("reference_tag").text().pytrim().to_owned();
        if !reference_tag(&tag) || !aliases.insert(tag.to_lowercase()) {
            continue;
        }
        let mut names: Vec<String> = vec![];
        for item in row
            .at("story_names")
            .as_list()
            .unwrap_or_default()
            .iter()
            .take(MAX_STORY_NAMES)
        {
            let Some(name) = item.as_str() else { continue };
            let name = name.pytrim();
            if name.is_empty()
                || name.chars().count() > MAX_STORY_NAME_CHARS
                || names.iter().any(|n| casefold(n) == casefold(name))
            {
                continue;
            }
            names.push(name.into());
        }
        if names.is_empty() {
            let fallback = tag.trim_start_matches('@').replace('_', " ");
            if !fallback.pytrim().is_empty() {
                names.push(fallback.pytrim().into());
            }
        }
        out.push(Py::obj([
            ("story_names", Py::strs(&names)),
            ("reference_tag", Py::Str(tag)),
            ("description", Py::Str(cast_field(row.at("description")))),
            (
                "default_language",
                Py::Str(cast_field(row.at("default_language"))),
            ),
            (
                "voice_guidance",
                Py::Str(cast_field(row.at("voice_guidance"))),
            ),
            ("required", Py::Bool(*row.at("required") != Py::Bool(false))),
        ]));
    }
    out
}
/// `story_cast_for_prompt`.
pub fn cast_for_prompt(cast: &[Py], dialogue: bool) -> Py {
    Py::List(
        cast.iter()
            .map(|row| {
                if dialogue {
                    return row.clone();
                }
                Py::obj([
                    ("story_names", row.at("story_names").clone()),
                    ("reference_tag", row.at("reference_tag").clone()),
                    ("description", row.at("description").clone()),
                    ("required", row.at("required").clone()),
                ])
            })
            .collect(),
    )
}
/// A continuation error: past a limit, or malformed.
fn continuation_text(value: &Py, required: bool) -> Result<String, &'static str> {
    let Py::Str(value) = value else {
        return Err("story_continuation_context_invalid");
    };
    if value.chars().count() > MAX_CONTINUATION_FIELD_CHARS {
        return Err("story_continuation_context_limit");
    }
    let value = value.pytrim();
    if required && value.is_empty() {
        return Err("story_continuation_context_invalid");
    }
    Ok(value.into())
}
fn exact_keys(value: &Py, keys: &[&str]) -> bool {
    value.keys().is_some_and(|mut found| {
        found.sort();
        let mut keys = keys.to_vec();
        keys.sort();
        found == keys
    })
}
/// `normalize_story_previous_scene_context`.
pub fn normalize_previous(value: &Py, required: bool) -> Result<Py, &'static str> {
    if *value == Py::Null && !required {
        return Ok(Py::Null);
    }
    let invalid = "story_continuation_context_invalid";
    if !exact_keys(
        value,
        &[
            "title",
            "summary",
            "description",
            "ambient_sound",
            "non_diegetic_music",
            "sections",
        ],
    ) {
        return Err(invalid);
    }
    let rows = value.at("sections").as_list().ok_or(invalid)?;
    if rows.len() > MAX_CONTINUATION_SECTIONS {
        return Err("story_continuation_context_limit");
    }
    let mut sections = vec![];
    for row in rows {
        if !exact_keys(row, &["prompt"]) {
            return Err(invalid);
        }
        sections.push(continuation_text(row.at("prompt"), true)?);
    }
    let text = |key| continuation_text(value.at(key), false);
    let normalized = [
        "title",
        "summary",
        "description",
        "ambient_sound",
        "non_diegetic_music",
    ]
    .map(|key| text(key).map(|t| (key, t)));
    let mut out = Py::Obj(vec![]);
    for field in normalized {
        let (key, value) = field?;
        out.push(key, Py::Str(value));
    }
    if [
        "summary",
        "description",
        "ambient_sound",
        "non_diegetic_music",
    ]
    .iter()
    .all(|k| out.at(k).text().is_empty())
        && sections.is_empty()
    {
        return Err(invalid);
    }
    out.push(
        "sections",
        Py::List(
            sections
                .into_iter()
                .map(|p| Py::obj([("prompt", Py::Str(p))]))
                .collect(),
        ),
    );
    Ok(out)
}
/// `normalize_story_context_cast`.
pub fn normalize_context_cast(value: &Py) -> Result<Py, &'static str> {
    let invalid = "story_continuation_context_invalid";
    if *value == Py::Null {
        return Ok(Py::List(vec![]));
    }
    let rows = value.as_list().ok_or(invalid)?;
    if rows.len() > MAX_CAST {
        return Err("story_continuation_context_limit");
    }
    let mut aliases = std::collections::BTreeSet::new();
    let mut out = vec![];
    for row in rows {
        if !exact_keys(row, &["reference_tag", "description", "required"]) {
            return Err(invalid);
        }
        let tag = continuation_text(row.at("reference_tag"), true)?;
        if !reference_tag(&tag) || *row.at("required") != Py::Bool(false) {
            return Err(invalid);
        }
        if !aliases.insert(tag.to_lowercase()) {
            return Err(invalid);
        }
        out.push(Py::obj([
            ("reference_tag", Py::Str(tag)),
            (
                "description",
                Py::Str(continuation_text(row.at("description"), false)?),
            ),
            ("required", Py::Bool(false)),
        ]));
    }
    Ok(Py::List(out))
}
/// `normalize_story_continuation_context`.
pub fn normalize_continuation(value: &Py) -> Result<Py, &'static str> {
    let invalid = "story_continuation_context_invalid";
    if *value == Py::Null {
        return Ok(Py::Null);
    }
    if !exact_keys(
        value,
        &[
            "schema_version",
            "source_scene_id",
            "source_revision",
            "recent_scene_summaries",
            "previous_scene",
            "context_cast",
        ],
    ) || *value.at("schema_version") != Py::Int(1)
    {
        return Err(invalid);
    }
    let recent = value
        .at("recent_scene_summaries")
        .as_list()
        .filter(|r| r.len() <= 2)
        .ok_or(invalid)?;
    let mut summaries = vec![];
    for row in recent {
        if !exact_keys(row, &["title", "summary"]) {
            return Err(invalid);
        }
        summaries.push(Py::obj([
            ("title", Py::Str(continuation_text(row.at("title"), false)?)),
            (
                "summary",
                Py::Str(continuation_text(row.at("summary"), true)?),
            ),
        ]));
    }
    let normalized = Py::obj([
        ("schema_version", Py::Int(1)),
        (
            "source_scene_id",
            Py::Str(continuation_text(value.at("source_scene_id"), true)?),
        ),
        (
            "source_revision",
            Py::Str(continuation_text(value.at("source_revision"), true)?),
        ),
        ("recent_scene_summaries", Py::List(summaries)),
        (
            "previous_scene",
            normalize_previous(value.at("previous_scene"), true)?,
        ),
        (
            "context_cast",
            normalize_context_cast(value.at("context_cast"))?,
        ),
    ]);
    if normalized.dumps().chars().count() > MAX_CONTINUATION_JSON_CHARS {
        return Err("story_continuation_context_limit");
    }
    Ok(normalized)
}
/// `build_story_outline_model_contract` for Director's default Scene policy.
pub fn contract(inventory: &Py, beats: &Py, dialogue: bool) -> Result<Contract, &'static str> {
    let invalid = "story_visual_input_invalid";
    let mut ids = std::collections::BTreeSet::new();
    let mut setting_ids = vec![];
    for row in inventory.as_list().unwrap_or_default() {
        row.keys().ok_or(invalid)?;
        let id = row.at("id").text().pytrim().to_owned();
        if id.is_empty() || !ids.insert(id.clone()) {
            return Err(invalid);
        }
        if row.at("usage").as_str() == Some("story_reference")
            && *row.at("reuse") != Py::Bool(false)
        {
            setting_ids.push(id);
        }
    }
    let mut keys = std::collections::BTreeSet::new();
    let mut counts = [0; 2];
    let mut out = vec![];
    for row in beats.as_list().unwrap_or_default() {
        row.keys().ok_or(invalid)?;
        let kind = row.at("type").text().pytrim().to_owned();
        let moment = match kind.as_str() {
            "required_moment" => true,
            "start_end_pair" => false,
            _ => return Err(invalid),
        };
        counts[usize::from(!moment)] += 1;
        let mut key = row.at("beat_key").text().pytrim().to_owned();
        if key.is_empty() {
            key = format!("{kind}_{:03}", counts[usize::from(!moment)]);
        }
        if !keys.insert(key.clone()) {
            return Err(invalid);
        }
        let image_ids = if moment {
            vec![row.at("image_id").text().pytrim().to_owned()]
        } else {
            vec![
                row.at("start_image_id").text().pytrim().to_owned(),
                row.at("end_image_id").text().pytrim().to_owned(),
            ]
        };
        if image_ids
            .iter()
            .any(|id| id.is_empty() || !ids.contains(id))
        {
            return Err(invalid);
        }
        out.push(Beat {
            key,
            image_ids,
            role: if moment {
                DEPICTED_MOMENT
            } else {
                STATE_TRANSITION
            },
        });
    }
    Ok(Contract {
        setting_ids,
        beats: out,
        dialogue,
        opening: None,
    })
}
/// Director's reference markers in every string of `value`, as
/// `protect_references` leaves them in the dumped payload.
fn protect(value: &Py, markers: &[(String, String)]) -> Py {
    match value {
        Py::Str(s) => Py::Str(
            serde_json::from_str(&super::references::apply(&Py::str(s).dumps(), markers)).unwrap(),
        ),
        Py::List(items) => Py::List(items.iter().map(|v| protect(v, markers)).collect()),
        Py::Obj(rows) => Py::Obj(
            rows.iter()
                .map(|(k, v)| (k.clone(), protect(v, markers)))
                .collect(),
        ),
        other => other.clone(),
    }
}
/// `protect_references(json.dumps(payload))` and the protected payload.
pub fn protected(payload: &Py) -> (Vec<(String, String)>, Py) {
    let markers = super::references::Protected::new(&payload.dumps()).references;
    (markers.clone(), protect(payload, &markers))
}
/// A built provider message and the markers its answer is restored with.
#[derive(Debug, Clone, Default)]
pub struct Built {
    pub user: String,
    pub references: Vec<(String, String)>,
    pub contract: Contract,
}
fn dialogue(request: &Py) -> bool {
    *request.at("add_dialogue") == Py::Bool(true)
}
/// The grounding and opening appearance contexts a request carries, as
/// `prepare_story_request` normalizes them for the Story bridge.
fn contexts(request: &Py) -> Result<(Py, Py), &'static str> {
    let grounding = normalize_grounding(request.at("grounding_context"))
        .map_err(|_| "story_context_invalid")?;
    let opening = normalize_opening(request.at("opening_appearance_context"), &grounding)
        .map_err(|_| "story_context_invalid")?;
    Ok((grounding, opening))
}
/// `_build_story_outline` without a one-Scene policy; a request may carry a
/// Story bridge's `grounding_context` and `opening_appearance_context`.
pub fn build_outline(request: &Py) -> Result<Built, &'static str> {
    let story = request.at("story").text().pytrim().to_owned();
    if story.is_empty() {
        return Err("story_brief_required");
    }
    let (grounding, opening) = contexts(request)?;
    let dialogue = dialogue(request);
    let mut contract = contract(
        request.at("image_inventory"),
        request.at("visual_beats"),
        dialogue,
    )?;
    let cast = normalize_cast(request.at("cast"));
    let continuation = normalize_continuation(request.at("continuation_context"))?;
    if dialogue && cast.len() < 2 {
        return Err("story_dialogue_cast_required");
    }
    let mut payload = Py::obj([
        ("story", Py::Str(story)),
        ("cast", cast_for_prompt(&cast, dialogue)),
        ("continuation_context", continuation),
        ("grounding_context", grounding),
    ]);
    if opening != Py::Null {
        payload.push("opening_appearance_context", opening);
    }
    let (references, payload) = protected(&payload);
    if let Some(opening) = payload.get("opening_appearance_context") {
        contract.opening = Some(
            opening
                .at("characters")
                .as_list()
                .unwrap()
                .iter()
                .map(|c| OpeningCharacter {
                    character_id: c.at("character_id").text(),
                    reference_tag: c.at("reference_tag").text(),
                    fallback_wardrobe: c.at("fallback_wardrobe").text(),
                })
                .collect(),
        );
    }
    let duration = match request.at("duration_seconds") {
        Py::Bool(_) => 5.,
        value => value.number().unwrap_or(5.),
    };
    let user = outline_user(&Outline {
        story: payload.at("story").as_str().unwrap(),
        instruction: request.at("user_instruction").text().pytrim(),
        cast: payload.at("cast"),
        contract: &contract,
        continuation: payload.at("continuation_context"),
        duration_seconds: duration,
        music: !matches!(
            request.at("include_non_diegetic_music"),
            Py::Null | Py::Bool(false)
        ),
        inventory: request.at("image_inventory"),
        beats: request.at("visual_beats"),
        grounding: payload.at("grounding_context"),
        opening: payload.at("opening_appearance_context"),
    });
    Ok(Built {
        user,
        references,
        contract,
    })
}
/// `_run_story_scene`'s user message for a request without visual bindings;
/// `images` when its images load.
pub fn build_scene(request: &Py, images: bool) -> Result<Built, &'static str> {
    let (grounding, _) = contexts(request)?;
    let dialogue = dialogue(request);
    let cast = normalize_cast(request.at("cast"));
    let previous = normalize_previous(request.at("previous_scene_context"), false)?;
    let context_cast = normalize_context_cast(request.at("context_cast"))?;
    if Py::obj([
        ("previous_scene_context", previous.clone()),
        ("context_cast", context_cast.clone()),
    ])
    .dumps()
    .chars()
    .count()
        > MAX_CONTINUATION_JSON_CHARS
    {
        return Err("story_continuation_context_limit");
    }
    let (references, payload) = protected(&Py::obj([
        ("scene", request.at("scene").clone()),
        ("cast", cast_for_prompt(&cast, dialogue)),
        ("previous_scene_context", previous),
        ("context_cast", context_cast),
        ("grounding_context", grounding),
    ]));
    let grounding = match payload.at("grounding_context") {
        Py::Null => None,
        grounding => Some(grounding.dumps()),
    };
    let timing = dialogue.then(|| match request.at("dialogue_timing") {
        timing @ Py::Obj(_) => timing.dumps(),
        _ => "{}".into(),
    });
    let user = scene_user(&Expansion {
        scene_json: &payload.at("scene").dumps(),
        rule: if images {
            SCENE_IMAGES
        } else {
            SCENE_NO_IMAGES
        },
        instruction: request.at("user_instruction").text().pytrim(),
        cast: &payload.at("cast").dumps(),
        previous_summary: &request.at("previous_summary").text(),
        previous: &payload.at("previous_scene_context").dumps(),
        next_summary: &request.at("next_summary").text(),
        context_cast: &payload.at("context_cast").dumps(),
        timing: timing.as_deref(),
        grounding: grounding.as_deref(),
    });
    Ok(Built {
        user,
        references,
        contract: Contract::default(),
    })
}
/// `_run_animate_image`'s user message for a validated `animate_image` request.
pub fn build_animate(request: &Py) -> String {
    let lines = request
        .at("scene_context")
        .at("messages")
        .as_list()
        .unwrap_or_default()
        .iter()
        .map(|m| format!("[{}] {}", m.at("name").text(), m.at("text").text()))
        .collect::<Vec<_>>()
        .join("\n");
    let mut blocks = vec![format!(
        "<scene_context>\n<latest_moment>\n{lines}\n</latest_moment>\n</scene_context>"
    )];
    for key in [
        "active_characters",
        "user_instruction",
        "audio_request",
        "generation_context",
    ] {
        blocks.push(format!("<{key}>\n{}\n</{key}>", request.at(key).indented()));
    }
    blocks.join("\n")
}
/// `contracts.animate_image.validate_content`: exactly three strings.
pub fn animate_content(value: &Py) -> Result<[String; 3], Invalid> {
    let fields = ["description", "prompt", "ambient_sound"];
    if !exact_keys(value, &fields) {
        return invalid("$", "must contain exactly the expected fields");
    }
    let text = |key: &str| match value.at(key) {
        Py::Str(s) => Ok(s.clone()),
        _ => invalid(format!("$.{key}"), "must be a string"),
    };
    Ok([text(fields[0])?, text(fields[1])?, text(fields[2])?])
}
/// `_inventory_user_prompt` for a browser visual descriptor.
pub fn build_inventory(item: &Py, instruction: &str) -> String {
    inventory_user(
        &item.at("id").text(),
        &item.at("note").text(),
        &item.at("label").text(),
        &item.at("usage").text(),
        instruction.pytrim(),
    )
}
/// `allocateWeightedRanges`: each item's start and end within `duration`.
pub fn ranges(weights: &[f64], duration: f64, start: f64, cap: f64) -> Vec<(f64, f64)> {
    if weights.is_empty() {
        return vec![];
    }
    let n = weights.len() as f64;
    let duration = if duration.is_finite() && duration != 0. {
        duration.max(0.001)
    } else {
        0.001
    };
    let minimum = if duration >= n * 0.25 {
        0.25
    } else {
        duration / n
    };
    let weights = weights
        .iter()
        .map(|w| {
            let w = if w.is_finite() && *w != 0. { *w } else { 1. };
            w.max(0.001)
        })
        .collect::<Vec<_>>();
    let mut durations = vec![minimum; weights.len()];
    let cap = if cap.is_finite() {
        minimum.max(cap)
    } else {
        f64::INFINITY
    };
    let mut remaining = (duration - minimum * n).max(0.);
    let mut active = (0..weights.len()).collect::<Vec<_>>();
    while remaining > 1e-12 && !active.is_empty() {
        let total = active.iter().fold(0., |sum, &i| sum + weights[i]);
        let saturated = active
            .iter()
            .copied()
            .filter(|&i| durations[i] + remaining * weights[i] / total > cap + 1e-12)
            .collect::<Vec<_>>();
        if saturated.is_empty() {
            for &i in &active {
                durations[i] += remaining * weights[i] / total;
            }
            break;
        }
        let mut spent = 0.;
        for i in saturated {
            let addition = (cap - durations[i]).max(0.);
            durations[i] += addition;
            spent += addition;
            active.retain(|&j| j != i);
        }
        remaining = (remaining - spent).max(0.);
        if spent <= 1e-12 && active.is_empty() {
            break;
        }
    }
    let allocated = durations.iter().fold(0., |sum, d| sum + d);
    let residual = duration - allocated;
    if residual.abs() > 1e-12
        && let Some(i) = durations
            .iter()
            .position(|&d| d + residual <= cap + 1e-9 && d + residual > 0.)
    {
        durations[i] += residual;
    }
    let mut cursor = start;
    let last = durations.len() - 1;
    let mut out = durations
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let length = if i == last {
                start + duration - cursor
            } else {
                d
            };
            let begin = cursor;
            cursor += length;
            (begin, cursor)
        })
        .collect::<Vec<_>>();
    out[last].1 = start + duration;
    out
}
/// `allocateStoryTimings` then `promptStudioDialogueTiming` for every Scene:
/// `weights` holds each Scene's weight and its Sections' (id, weight).
pub fn dialogue_timings(scenes: &[(f64, Vec<(String, f64)>)], duration: f64) -> Vec<Py> {
    let duration = if duration.is_finite() && duration != 0. {
        duration.max(0.001)
    } else {
        0.001
    };
    let mut scene_ranges = ranges(
        &scenes.iter().map(|s| s.0).collect::<Vec<_>>(),
        duration,
        0.,
        MAX_SCENE_SECONDS as f64,
    );
    let mut section_ranges = scenes
        .iter()
        .zip(&scene_ranges)
        .map(|((_, sections), &(start, end))| {
            ranges(
                &sections.iter().map(|s| s.1).collect::<Vec<_>>(),
                end - start,
                start,
                f64::INFINITY,
            )
        })
        .collect::<Vec<_>>();
    if let Some(first) = scene_ranges.first_mut() {
        first.0 = 0.;
    }
    if let Some(last) = scene_ranges.last_mut() {
        last.1 = duration;
    }
    if let Some(last) = section_ranges.last_mut().and_then(|s| s.last_mut()) {
        last.1 = duration;
    }
    let length = |(start, end): (f64, f64)| (end - start).max(0.);
    scenes
        .iter()
        .zip(scene_ranges)
        .zip(section_ranges)
        .map(|(((_, sections), scene), section_ranges)| {
            Py::obj([
                ("scene_duration_seconds", Py::js(length(scene))),
                (
                    "sections",
                    Py::List(
                        sections
                            .iter()
                            .zip(section_ranges)
                            .map(|((id, _), range)| {
                                Py::obj([
                                    ("id", Py::str(id)),
                                    ("duration_seconds", Py::js(length(range))),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Director's own builders on the same browser requests
    /// (`tools/director-story-messages-oracle.py`).
    #[test]
    fn messages_match_the_director_source_oracle() {
        let fixture = Py::parse(include_str!(
            "../../../tests/fixtures/director-story-messages.json"
        ))
        .unwrap();
        let cases = fixture.at("cases").as_list().unwrap();
        assert_eq!(cases.len(), 18);
        let answers = fixture.at("answers").as_list().unwrap();
        assert_eq!(answers.len(), 25);
        for answer in answers {
            let raw = answer.at("raw").text();
            assert_eq!(
                strip_outer_markdown(&raw),
                answer.at("stripped").text(),
                "{raw:?}"
            );
            let object = match answer.at("object") {
                Py::Null => None,
                object => Some(object.text()),
            };
            assert_eq!(
                parse_json_object(&raw).map(|v| v.dumps()),
                object,
                "{raw:?}"
            );
            let text = match answer.at("text") {
                Py::Null => None,
                text => Some(text.text()),
            };
            assert_eq!(clean_text_response(&raw), text, "{raw:?}");
        }
        for case in cases {
            let name = case.at("name").text();
            let request = case.at("request");
            let user = match case.at("kind").as_str().unwrap() {
                "outline" => {
                    let built = build_outline(request).unwrap();
                    assert_eq!(
                        schema_text(&built.contract),
                        case.at("schema").text(),
                        "{name}"
                    );
                    let references = case
                        .at("references")
                        .as_list()
                        .unwrap()
                        .iter()
                        .map(|r| {
                            let r = r.as_list().unwrap();
                            (r[0].text(), r[1].text())
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(built.references, references, "{name}");
                    for answer in case.at("answers").as_list().unwrap_or_default() {
                        let result = validate_opening(
                            answer.at("answer").at("opening_appearance"),
                            built.contract.opening.as_ref().unwrap(),
                        );
                        match answer.at("path") {
                            Py::Null => assert_eq!(result, Ok(()), "{name}"),
                            path => assert_eq!(result.unwrap_err().path, path.text(), "{name}"),
                        }
                    }
                    built.user
                }
                "animate" => {
                    assert_eq!(*case.at("images"), Py::Int(1), "{name}");
                    for content in case.at("contents").as_list().unwrap() {
                        let result = animate_content(content.at("content"));
                        match content.at("path") {
                            Py::Null => assert!(result.is_ok(), "{name}"),
                            path => assert_eq!(result.unwrap_err().path, path.text(), "{name}"),
                        }
                    }
                    build_animate(request)
                }
                "scene" => {
                    let images = !request.at("images").as_list().unwrap().is_empty();
                    assert_eq!(*case.at("images"), Py::Int(images.into()), "{name}");
                    build_scene(request, images).unwrap().user
                }
                _ => build_inventory(request.at("item"), &request.at("user_instruction").text()),
            };
            assert_eq!(user, case.at("user").text(), "{name}");
        }
    }
    #[test]
    fn python_float_repr_and_json_spacing() {
        for (value, text) in [
            (10., "10.0"),
            (0.25, "0.25"),
            (6.583333333333334, "6.583333333333334"),
            (1e16, "1e+16"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (123456789012345.6, "123456789012345.6"),
            (-2.5, "-2.5"),
        ] {
            assert_eq!(py_float(value), text);
        }
        let value = Py::obj([
            ("a", Py::List(vec![Py::Int(1), Py::Float(1.)])),
            ("b", Py::str("ö\"\n\u{1}")),
            ("c", Py::List(vec![])),
        ]);
        assert_eq!(
            value.dumps(),
            "{\"a\": [1, 1.0], \"b\": \"ö\\\"\\n\\u0001\", \"c\": []}"
        );
        assert_eq!(
            value.compact(),
            "{\"a\":[1,1.0],\"b\":\"ö\\\"\\n\\u0001\",\"c\":[]}"
        );
        assert_eq!(
            value.indented(),
            "{\n  \"a\": [\n    1,\n    1.0\n  ],\n  \"b\": \"ö\\\"\\n\\u0001\",\n  \"c\": []\n}"
        );
    }
    #[test]
    fn browser_hash_and_reference_order() {
        assert_eq!(stable_hash(""), "ztntfp");
        assert_eq!(stable_hash("a"), "1r9wi7g");
        let mut tags = vec![
            "@b", "@A", "@a_1", "@a-1", "@a1", "@a", "@a.b", "@ab", "@aB",
        ];
        tags.sort_by(|a, b| locale_cmp(a, b));
        assert_eq!(
            tags,
            [
                "@a", "@A", "@a_1", "@a-1", "@a.b", "@a1", "@ab", "@aB", "@b"
            ]
        );
    }
}
