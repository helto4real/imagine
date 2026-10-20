//! CPython default JSON metadata semantics for the frozen checkpoint helpers.
//! Ordered objects, duplicate-key overwrite, escaped surrogate code points and
//! arbitrary decimal integer text survive decode/encode without a serde map.
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Int(String),
    Float(f64),
    Text(Vec<u32>),
    List(Vec<Json>),
    Dict(Vec<(Vec<u32>, Json)>),
}
pub(super) fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap(message: &str) -> Error {
    Error::BindingGap {
        backend: "checkpoint_metadata".into(),
        operation: message.into(),
    }
}
pub(super) fn points(text: &str) -> Vec<u32> {
    text.chars().map(u32::from).collect()
}
pub(super) fn string(points: &[u32]) -> Result<String, Error> {
    points
        .iter()
        .map(|p| {
            char::from_u32(*p).ok_or_else(|| {
                gap("surrogate code point in a native checkpoint key/value requires admission")
            })
        })
        .collect()
}
impl Json {
    pub(super) fn get(&self, key: &str) -> Result<Self, Error> {
        match self {
            Self::Dict(values) => values
                .iter()
                .find(|(k, _)| k == &points(key))
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    error(
                        "KeyError",
                        crate::triton_specialization::python_str_repr(key),
                    )
                }),
            Self::List(_) => Err(error(
                "TypeError",
                "list indices must be integers or slices, not str",
            )),
            Self::Text(_) => Err(error(
                "TypeError",
                "string indices must be integers, not 'str'",
            )),
            _ => Err(error(
                "TypeError",
                format!("'{}' object is not subscriptable", self.typename()),
            )),
        }
    }
    pub(super) fn items(&self) -> Result<&[(Vec<u32>, Json)], Error> {
        if let Self::Dict(values) = self {
            return Ok(values);
        }
        Err(error(
            "AttributeError",
            format!("'{}' object has no attribute 'items'", self.typename()),
        ))
    }
    fn typename(&self) -> &'static str {
        match self {
            Self::Null => "NoneType",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Text(_) => "str",
            Self::List(_) => "list",
            Self::Dict(_) => "dict",
        }
    }
    pub(super) fn options(&self) -> Result<Options, Error> {
        self.options_impl(false)
    }
    /// Metadata at the native object boundary retains Unicode object owners;
    /// cloning a field handle must not allocate a different string object.
    pub(super) fn source_options(&self) -> Result<Options, Error> {
        self.options_impl(true)
    }
    fn options_impl(&self, source_strings: bool) -> Result<Options, Error> {
        Ok(match self {
            Self::Null => Options::Atom(V::None),
            Self::Bool(v) => Options::Atom(V::Bool(*v)),
            Self::Float(v) => Options::Atom(V::Float(*v)),
            Self::Int(v) => Options::Atom(if let Ok(n) = v.parse::<i64>() {
                V::Int(n)
            } else if let Ok(n) = v.parse::<u64>() {
                V::UInt(n)
            } else {
                return Err(gap("JSON integer exceeds native Options admission"));
            }),
            Self::Text(v) if source_strings => {
                crate::patching::options::SourceText::from_codepoints(v.clone())?
            }
            Self::Text(v) => Options::Atom(V::Text(string(v)?)),
            Self::List(v) => Options::List(Rc::new(RefCell::new(
                v.iter()
                    .map(|value| value.options_impl(source_strings))
                    .collect::<Result<_, _>>()?,
            ))),
            Self::Dict(v) => Options::Dict(Rc::new(RefCell::new(
                v.iter()
                    .map(|(k, v)| Ok((string(k)?, v.options_impl(source_strings)?)))
                    .collect::<Result<_, Error>>()?,
            ))),
        })
    }
    pub(super) fn dumps(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(v) => v.to_string(),
            Self::Int(v) => v.clone(),
            Self::Float(v) => {
                if v.is_nan() {
                    "NaN".into()
                } else if *v == f64::INFINITY {
                    "Infinity".into()
                } else if *v == f64::NEG_INFINITY {
                    "-Infinity".into()
                } else {
                    crate::registry::python_float(*v)
                }
            }
            Self::Text(v) => quote(v),
            Self::List(v) => format!(
                "[{}]",
                v.iter().map(Self::dumps).collect::<Vec<_>>().join(", ")
            ),
            Self::Dict(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!("{}: {}", quote(k), v.dumps()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
fn quote(value: &[u32]) -> String {
    use std::fmt::Write;
    let mut out = String::from("\"");
    for &p in value {
        match p {
            34 => out.push_str("\\\""),
            92 => out.push_str("\\\\"),
            8 => out.push_str("\\b"),
            9 => out.push_str("\\t"),
            10 => out.push_str("\\n"),
            12 => out.push_str("\\f"),
            13 => out.push_str("\\r"),
            32..=126 => out.push(char::from_u32(p).unwrap()),
            0..=65535 => write!(out, "\\u{p:04x}").unwrap(),
            _ => {
                let p = p - 65536;
                write!(
                    out,
                    "\\u{:04x}\\u{:04x}",
                    0xd800 + (p >> 10),
                    0xdc00 + (p & 1023)
                )
                .unwrap();
            }
        }
    }
    out.push('"');
    out
}
pub(super) fn loads(value: &Options) -> Result<Json, Error> {
    let chars = if let Some(text) = crate::patching::options::SourceText::get(value) {
        text.codepoints().to_vec()
    } else if let Options::Atom(V::Text(text)) = value {
        points(text)
    } else {
        return Err(error(
            "TypeError",
            format!(
                "the JSON object must be str, bytes or bytearray, not {}",
                value.typename()
            ),
        ));
    };
    let mut p = Parser { chars, pos: 0 };
    if p.chars.first() == Some(&0xfeff) {
        return Err(p.err("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    p.ws();
    let value = p.value(0)?;
    p.ws();
    if p.pos != p.chars.len() {
        return Err(p.err("Extra data", p.pos));
    }
    Ok(value)
}
pub(super) fn load_options(value: &Options) -> Result<Options, Error> {
    loads(value)?.options()
}
struct Parser {
    chars: Vec<u32>,
    pos: usize,
}
impl Parser {
    fn peek(&self) -> Option<u32> {
        self.chars.get(self.pos).copied()
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(32 | 9 | 10 | 13)) {
            self.pos += 1;
        }
    }
    fn err(&self, message: &str, pos: usize) -> Error {
        let line = 1 + self.chars[..pos.min(self.chars.len())]
            .iter()
            .filter(|p| **p == 10)
            .count();
        let col = pos
            - self.chars[..pos.min(self.chars.len())]
                .iter()
                .rposition(|p| *p == 10)
                .map_or(0, |i| i + 1)
            + 1;
        error(
            "JSONDecodeError",
            format!("{message}: line {line} column {col} (char {pos})"),
        )
    }
    fn begins(&self, word: &str) -> bool {
        self.chars[self.pos..].starts_with(&points(word))
    }
    fn value(&mut self, depth: usize) -> Result<Json, Error> {
        // Do not silently impose a different CPython recursion-limit policy.
        // Deep unadmitted metadata fails closed before Rust stack exhaustion.
        if depth > 128 {
            return Err(gap(
                "deep JSON nesting requires original recursion-limit admission",
            ));
        }
        for (word, value) in [
            ("null", Json::Null),
            ("true", Json::Bool(true)),
            ("false", Json::Bool(false)),
            ("NaN", Json::Float(f64::NAN)),
            ("Infinity", Json::Float(f64::INFINITY)),
            ("-Infinity", Json::Float(f64::NEG_INFINITY)),
        ] {
            if self.begins(word) {
                self.pos += word.len();
                return Ok(value);
            }
        }
        match self.peek() {
            Some(34) => self.text().map(Json::Text),
            Some(91) => {
                self.pos += 1;
                self.ws();
                let mut out = vec![];
                if self.peek() == Some(93) {
                    self.pos += 1;
                    return Ok(Json::List(out));
                }
                loop {
                    out.push(self.value(depth + 1)?);
                    self.ws();
                    if self.peek() == Some(93) {
                        self.pos += 1;
                        break;
                    }
                    if self.peek() != Some(44) {
                        return Err(self.err("Expecting ',' delimiter", self.pos));
                    }
                    self.pos += 1;
                    self.ws();
                }
                Ok(Json::List(out))
            }
            Some(123) => {
                self.pos += 1;
                self.ws();
                let mut out: Vec<(Vec<u32>, Json)> = vec![];
                if self.peek() == Some(125) {
                    self.pos += 1;
                    return Ok(Json::Dict(out));
                }
                loop {
                    if self.peek() != Some(34) {
                        return Err(self.err(
                            "Expecting property name enclosed in double quotes",
                            self.pos,
                        ));
                    }
                    let key = self.text()?;
                    self.ws();
                    if self.peek() != Some(58) {
                        return Err(self.err("Expecting ':' delimiter", self.pos));
                    }
                    self.pos += 1;
                    self.ws();
                    let value = self.value(depth + 1)?;
                    if let Some((_, v)) = out.iter_mut().find(|(k, _)| k == &key) {
                        *v = value;
                    } else {
                        out.push((key, value));
                    }
                    self.ws();
                    if self.peek() == Some(125) {
                        self.pos += 1;
                        break;
                    }
                    if self.peek() != Some(44) {
                        return Err(self.err("Expecting ',' delimiter", self.pos));
                    }
                    self.pos += 1;
                    self.ws();
                }
                Ok(Json::Dict(out))
            }
            Some(45 | 48..=57) => self.number(),
            _ => Err(self.err("Expecting value", self.pos)),
        }
    }
    fn hex(&self, slash: usize) -> Result<u32, Error> {
        let mut out = 0;
        for i in slash + 2..slash + 6 {
            let Some(p) = self.chars.get(i).copied() else {
                return Err(self.err("Invalid \\uXXXX escape", slash + 1));
            };
            let Some(d) = char::from_u32(p).and_then(|p| p.to_digit(16)) else {
                return Err(self.err("Invalid \\uXXXX escape", slash + 1));
            };
            out = out * 16 + d;
        }
        Ok(out)
    }
    fn text(&mut self) -> Result<Vec<u32>, Error> {
        let begin = self.pos;
        self.pos += 1;
        let mut out = vec![];
        while let Some(p) = self.peek() {
            if p == 34 {
                self.pos += 1;
                return Ok(out);
            }
            if p < 32 {
                return Err(self.err("Invalid control character at", self.pos));
            }
            if p != 92 {
                out.push(p);
                self.pos += 1;
                continue;
            }
            let slash = self.pos;
            self.pos += 1;
            let Some(escaped) = self.peek() else {
                return Err(self.err("Unterminated string starting at", begin));
            };
            let c = match escaped {
                34 | 92 | 47 => escaped,
                98 => 8,
                102 => 12,
                110 => 10,
                114 => 13,
                116 => 9,
                117 => {
                    let mut c = self.hex(slash)?;
                    self.pos = slash + 6;
                    if (0xd800..=0xdbff).contains(&c) && self.begins("\\u") {
                        let second = self.hex(self.pos)?;
                        if (0xdc00..=0xdfff).contains(&second) {
                            c = 65536 + ((c - 0xd800) << 10) + (second - 0xdc00);
                            self.pos += 6;
                        }
                    }
                    out.push(c);
                    continue;
                }
                _ => return Err(self.err("Invalid \\escape", slash)),
            };
            out.push(c);
            self.pos += 1;
        }
        Err(self.err("Unterminated string starting at", begin))
    }
    fn number(&mut self) -> Result<Json, Error> {
        let start = self.pos;
        if self.peek() == Some(45) {
            self.pos += 1;
        }
        match self.peek() {
            Some(48) => self.pos += 1,
            Some(49..=57) => {
                while matches!(self.peek(), Some(48..=57)) {
                    self.pos += 1;
                }
            }
            _ => {
                self.pos = start;
                return Err(self.err("Expecting value", start));
            }
        }
        let mut floating = false;
        if self.peek() == Some(46) && matches!(self.chars.get(self.pos + 1), Some(48..=57)) {
            floating = true;
            self.pos += 1;
            while matches!(self.peek(), Some(48..=57)) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(69 | 101)) {
            let mut next = self.pos + 1;
            if matches!(self.chars.get(next), Some(43 | 45)) {
                next += 1;
            }
            if matches!(self.chars.get(next), Some(48..=57)) {
                floating = true;
                self.pos = next + 1;
                while matches!(self.peek(), Some(48..=57)) {
                    self.pos += 1;
                }
            }
        }
        let text = string(&self.chars[start..self.pos])?;
        if floating {
            return Ok(Json::Float(text.parse().map_err(|_| {
                gap("JSON decimal float conversion requires admission")
            })?));
        }
        let digits = text.len() - usize::from(text.starts_with('-'));
        if digits > 4300 {
            return Err(error(
                "ValueError",
                format!(
                    "Exceeds the limit (4300 digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
                ),
            ));
        }
        Ok(Json::Int(if text == "-0" { "0".into() } else { text }))
    }
}
