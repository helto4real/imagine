//! Director reads catalogs with PyYAML's SafeLoader (YAML 1.1). This reader
//! reproduces it on libyaml: the token scan that rejects anchors, aliases and
//! tags; composing every document; then construction in PyYAML's breadth-first
//! order with YAML 1.1 implicit types, merge-key and duplicate-key rejection.
use std::{collections::VecDeque, mem::MaybeUninit, slice};
use unsafe_libyaml as unsafe_sys;

/// A constructed PyYAML value. Only strings, lists and mappings can be valid
/// catalog content; other scalars keep enough to compare mapping keys.
#[derive(Debug, Clone)]
pub(crate) enum Value {
    Str(String),
    Null,
    Bool(bool),
    Int(Number),
    Float(f64),
    Timestamp(String),
    List(Vec<Node>),
    Map(Vec<(Node, Node)>),
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Number {
    Small(i128),
    /// Beyond i128: the sign and decimal digits.
    Big(String),
}
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub value: Value,
    /// One-based start position of the composed node.
    pub line: usize,
    pub column: usize,
    /// A scalar's source text, used for mapping-key locations.
    pub scalar: Option<String>,
}
impl Node {
    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Error {
    /// PROMPT_CATALOG_YAML_FEATURE_UNSUPPORTED or PROMPT_CATALOG_YAML_INVALID.
    pub code: &'static str,
    pub line: Option<u32>,
    pub column: Option<u32>,
}
fn invalid(mark: Option<unsafe_sys::yaml_mark_t>) -> Error {
    Error {
        code: "PROMPT_CATALOG_YAML_INVALID",
        line: mark.map(|m| m.line as u32 + 1),
        column: mark.map(|m| m.column as u32 + 1),
    }
}

/// libyaml parser state; boxed so its address never changes while in use.
struct Parser {
    sys: Box<MaybeUninit<unsafe_sys::yaml_parser_t>>,
}
impl Parser {
    fn new(input: &[u8]) -> Self {
        let mut sys = Box::new(MaybeUninit::<unsafe_sys::yaml_parser_t>::uninit());
        unsafe {
            let parser = sys.as_mut_ptr();
            if unsafe_sys::yaml_parser_initialize(parser).fail {
                panic!("libyaml parser allocation failed");
            }
            unsafe_sys::yaml_parser_set_encoding(parser, unsafe_sys::YAML_UTF8_ENCODING);
            // The input outlives the parser: both live only inside one call below.
            unsafe_sys::yaml_parser_set_input_string(parser, input.as_ptr(), input.len() as u64);
        }
        Self { sys }
    }
    fn ptr(&mut self) -> *mut unsafe_sys::yaml_parser_t {
        self.sys.as_mut_ptr()
    }
    /// PyYAML's reader errors carry no position; scanner and parser errors do.
    fn error(&mut self) -> Error {
        // The prefix fields are read through the parser's Deref.
        let parser: &unsafe_sys::yaml_parser_t = unsafe { &*self.ptr() };
        invalid((parser.error != unsafe_sys::YAML_READER_ERROR).then_some(parser.problem_mark))
    }
}
impl Drop for Parser {
    fn drop(&mut self) {
        unsafe { unsafe_sys::yaml_parser_delete(self.ptr()) }
    }
}

/// A double-quoted `\uD8xx\uDCxx` pair that libyaml rejects, by byte offset of
/// its first hex digit and its zero-based line and column.
struct Surrogate {
    index: usize,
    line: usize,
    column: usize,
}
/// PyYAML's `yaml.scan`: the first anchor, alias or tag token, or a scanner error.
fn scan(input: &[u8]) -> Result<(), std::result::Result<Error, Surrogate>> {
    let mut parser = Parser::new(input);
    loop {
        let mut token = MaybeUninit::<unsafe_sys::yaml_token_t>::uninit();
        unsafe {
            let token = token.as_mut_ptr();
            if unsafe_sys::yaml_parser_scan(parser.ptr(), token).fail {
                let p: &unsafe_sys::yaml_parser_t = &*parser.ptr();
                let problem = std::ffi::CStr::from_ptr(p.problem.cast());
                if problem.to_bytes() == b"found invalid Unicode character escape code" {
                    return Err(Err(Surrogate {
                        index: p.problem_mark.index as usize,
                        line: p.problem_mark.line as usize,
                        column: p.problem_mark.column as usize,
                    }));
                }
                return Err(Ok(parser.error()));
            }
            let kind = (*token).type_;
            let mark = (*token).start_mark;
            unsafe_sys::yaml_token_delete(token);
            match kind {
                unsafe_sys::YAML_ALIAS_TOKEN
                | unsafe_sys::YAML_ANCHOR_TOKEN
                | unsafe_sys::YAML_TAG_TOKEN => {
                    return Err(Ok(Error {
                        code: "PROMPT_CATALOG_YAML_FEATURE_UNSUPPORTED",
                        line: Some(mark.line as u32 + 1),
                        column: Some(mark.column as u32 + 1),
                    }));
                }
                unsafe_sys::YAML_STREAM_END_TOKEN | unsafe_sys::YAML_NO_TOKEN => return Ok(()),
                _ => {}
            }
        }
    }
}

/// A composed node before construction.
enum Raw {
    Scalar { value: String, plain: bool },
    Sequence(Vec<Composed>),
    Mapping(Vec<(Composed, Composed)>),
}
struct Composed {
    raw: Raw,
    mark: unsafe_sys::yaml_mark_t,
}
enum Event {
    DocumentStart,
    DocumentEnd,
    StreamEnd,
    Scalar(String, bool),
    SequenceStart,
    SequenceEnd,
    MappingStart,
    MappingEnd,
    Other,
}
fn next(parser: &mut Parser) -> Result<(Event, unsafe_sys::yaml_mark_t), Error> {
    let mut event = MaybeUninit::<unsafe_sys::yaml_event_t>::uninit();
    unsafe {
        let event = event.as_mut_ptr();
        if unsafe_sys::yaml_parser_parse(parser.ptr(), event).fail {
            return Err(parser.error());
        }
        let mark = (*event).start_mark;
        let converted = match (*event).type_ {
            unsafe_sys::YAML_DOCUMENT_START_EVENT => Event::DocumentStart,
            unsafe_sys::YAML_DOCUMENT_END_EVENT => Event::DocumentEnd,
            unsafe_sys::YAML_STREAM_END_EVENT => Event::StreamEnd,
            unsafe_sys::YAML_SCALAR_EVENT => {
                let scalar = &(*event).data.scalar;
                let bytes = slice::from_raw_parts(scalar.value, scalar.length as usize);
                let plain = scalar.style == unsafe_sys::YAML_PLAIN_SCALAR_STYLE;
                match std::str::from_utf8(bytes) {
                    Ok(text) => Event::Scalar(text.to_owned(), plain),
                    Err(_) => {
                        unsafe_sys::yaml_event_delete(event);
                        return Err(invalid(Some(mark)));
                    }
                }
            }
            unsafe_sys::YAML_SEQUENCE_START_EVENT => Event::SequenceStart,
            unsafe_sys::YAML_SEQUENCE_END_EVENT => Event::SequenceEnd,
            unsafe_sys::YAML_MAPPING_START_EVENT => Event::MappingStart,
            unsafe_sys::YAML_MAPPING_END_EVENT => Event::MappingEnd,
            _ => Event::Other,
        };
        unsafe_sys::yaml_event_delete(event);
        Ok((converted, mark))
    }
}
fn compose_node(
    parser: &mut Parser,
    first: (Event, unsafe_sys::yaml_mark_t),
) -> Result<Composed, Error> {
    let (event, mark) = first;
    let raw = match event {
        Event::Scalar(value, plain) => Raw::Scalar { value, plain },
        Event::SequenceStart => {
            let mut items = vec![];
            loop {
                let e = next(parser)?;
                if matches!(e.0, Event::SequenceEnd) {
                    break;
                }
                items.push(compose_node(parser, e)?);
            }
            Raw::Sequence(items)
        }
        Event::MappingStart => {
            let mut pairs = vec![];
            loop {
                let e = next(parser)?;
                if matches!(e.0, Event::MappingEnd) {
                    break;
                }
                let key = compose_node(parser, e)?;
                let value = next(parser).and_then(|e| compose_node(parser, e))?;
                pairs.push((key, value));
            }
            Raw::Mapping(pairs)
        }
        _ => return Err(invalid(Some(mark))),
    };
    Ok(Composed { raw, mark })
}
/// PyYAML's `compose_all`: every document of the stream.
fn compose_all(input: &[u8]) -> Result<Vec<Composed>, Error> {
    let mut parser = Parser::new(input);
    let mut documents = vec![];
    loop {
        match next(&mut parser)? {
            (Event::StreamEnd, _) => return Ok(documents),
            (Event::DocumentStart, _) => {
                let first = next(&mut parser)?;
                documents.push(compose_node(&mut parser, first)?);
                if !matches!(next(&mut parser)?.0, Event::DocumentEnd) {
                    return Err(invalid(None));
                }
            }
            _ => {}
        }
    }
}

/// PyYAML's YAML 1.1 implicit resolution of a plain scalar.
enum Implicit {
    Str,
    Null,
    Bool(bool),
    Int,
    Float,
    Merge,
    Timestamp,
    /// The `=` value tag, which SafeLoader cannot construct.
    Value,
}
fn all(s: &str, f: impl Fn(u8) -> bool) -> bool {
    s.bytes().all(f)
}
fn digits(s: &str) -> bool {
    !s.is_empty() && all(s, |b| b.is_ascii_digit())
}
/// `[0-9][0-9_]*`
fn decimal_run(s: &str) -> bool {
    s.as_bytes().first().is_some_and(u8::is_ascii_digit)
        && all(s, |b| b.is_ascii_digit() || b == b'_')
}
/// `(?::[0-5]?[0-9])+`
fn sexagesimal_tail(s: &str) -> bool {
    s.starts_with(':')
        && s[1..].split(':').all(|p| {
            let p = p.as_bytes();
            matches!(p, [d] if d.is_ascii_digit())
                || matches!(p, [a, b] if (b'0'..=b'5').contains(a) && b.is_ascii_digit())
        })
}
fn strip_sign(s: &str) -> &str {
    s.strip_prefix(['-', '+']).unwrap_or(s)
}
/// `[eE][-+][0-9]+` or nothing.
fn exponent(s: &str) -> bool {
    s.is_empty() || (s.starts_with(['e', 'E']) && s[1..].starts_with(['-', '+']) && digits(&s[2..]))
}
fn is_float(s: &str) -> bool {
    let unsigned = strip_sign(s);
    // [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
    if let Some((whole, rest)) = unsigned.split_once('.')
        && decimal_run(whole)
    {
        let fraction_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        if exponent(&rest[fraction_end..]) {
            return true;
        }
    }
    // \.[0-9][0-9_]*(?:[eE][-+][0-9]+)?   (no sign)
    if let Some(rest) = s.strip_prefix('.')
        && rest.as_bytes().first().is_some_and(u8::is_ascii_digit)
    {
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        if exponent(&rest[end..]) {
            return true;
        }
    }
    // [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    if let Some((head, fraction)) = unsigned.rsplit_once('.')
        && all(fraction, |b| b.is_ascii_digit() || b == b'_')
        && let Some(colon) = head.find(':')
        && decimal_run(&head[..colon])
        && sexagesimal_tail(&head[colon..])
    {
        return true;
    }
    // [-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)
    matches!(unsigned, ".inf" | ".Inf" | ".INF") || matches!(s, ".nan" | ".NaN" | ".NAN")
}
fn is_int(s: &str) -> bool {
    let u = strip_sign(s);
    if let Some(b) = u.strip_prefix("0b") {
        return !b.is_empty() && all(b, |c| matches!(c, b'0' | b'1' | b'_'));
    }
    if let Some(h) = u.strip_prefix("0x") {
        return !h.is_empty() && all(h, |c| c.is_ascii_hexdigit() || c == b'_');
    }
    // 0[0-7_]+
    if u.len() > 1
        && u.starts_with('0')
        && all(&u[1..], |c| (b'0'..=b'7').contains(&c) || c == b'_')
    {
        return true;
    }
    // 0|[1-9][0-9_]*
    if u == "0"
        || (u
            .as_bytes()
            .first()
            .is_some_and(|c| (b'1'..=b'9').contains(c))
            && decimal_run(u))
    {
        return true;
    }
    // [1-9][0-9_]*(?::[0-5]?[0-9])+
    if let Some(colon) = u.find(':')
        && u.as_bytes()
            .first()
            .is_some_and(|c| (b'1'..=b'9').contains(c))
        && decimal_run(&u[..colon])
        && sexagesimal_tail(&u[colon..])
    {
        return true;
    }
    false
}
/// The timestamp groups: (year, month, day, time) where time is
/// (hour, minute, second, fraction digits, timezone).
type Time<'a> = (u32, u32, u32, Option<&'a str>, Option<&'a str>);
fn timestamp(s: &str) -> Option<(u32, u32, u32, Option<Time<'_>>)> {
    let b = s.as_bytes();
    let take = |from: usize, min: usize, max: usize| -> Option<(u32, usize)> {
        let n = b[from..]
            .iter()
            .take(max)
            .take_while(|c| c.is_ascii_digit())
            .count();
        (n >= min).then(|| (s[from..from + n].parse().unwrap(), from + n))
    };
    if b.len() < 8 {
        return None;
    }
    let (year, at) = take(0, 4, 4)?;
    if b.get(at) != Some(&b'-') {
        return None;
    }
    // Date only: exactly two-digit month and day.
    if s.len() == 10 && b[7] == b'-' && digits(&s[5..7]) && digits(&s[8..10]) {
        return Some((
            year,
            s[5..7].parse().unwrap(),
            s[8..10].parse().unwrap(),
            None,
        ));
    }
    let (month, at) = take(at + 1, 1, 2)?;
    if b.get(at) != Some(&b'-') {
        return None;
    }
    let (day, mut at) = take(at + 1, 1, 2)?;
    // (?:[Tt]|[ \t]+)
    match b.get(at) {
        Some(b'T' | b't') => at += 1,
        Some(b' ' | b'\t') => {
            while matches!(b.get(at), Some(b' ' | b'\t')) {
                at += 1;
            }
        }
        _ => return None,
    }
    let (hour, at) = take(at, 1, 2)?;
    if b.get(at) != Some(&b':') {
        return None;
    }
    let (minute, at) = take(at + 1, 2, 2)?;
    if b.get(at) != Some(&b':') {
        return None;
    }
    let (second, mut at) = take(at + 1, 2, 2)?;
    let mut fraction = None;
    if b.get(at) == Some(&b'.') {
        let n = b[at + 1..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count();
        fraction = Some(&s[at + 1..at + 1 + n]);
        at += 1 + n;
    }
    let rest = &s[at..];
    let zone = rest.trim_start_matches([' ', '\t']);
    let tz = if zone.is_empty() {
        if !rest.is_empty() {
            return None;
        }
        None
    } else if zone == "Z" {
        Some("Z")
    } else {
        let z = zone.strip_prefix(['-', '+'])?;
        let (h, m) = z.split_once(':').unwrap_or((z, ""));
        let ok_h = !h.is_empty() && h.len() <= 2 && digits(h);
        let ok_m = if z.contains(':') {
            m.len() == 2 && digits(m)
        } else {
            true
        };
        if !(ok_h && ok_m) {
            return None;
        }
        Some(zone)
    };
    let _ = (hour, minute, second);
    Some((year, month, day, Some((hour, minute, second, fraction, tz))))
}
fn resolve(value: &str) -> Implicit {
    let first = value.chars().next();
    let Some(first) = first else {
        return Implicit::Null;
    };
    if "yYnNtTfFoO".contains(first)
        && let Some(b) = match value {
            "yes" | "Yes" | "YES" | "true" | "True" | "TRUE" | "on" | "On" | "ON" => Some(true),
            "no" | "No" | "NO" | "false" | "False" | "FALSE" | "off" | "Off" | "OFF" => Some(false),
            _ => None,
        }
    {
        return Implicit::Bool(b);
    }
    if "-+0123456789.".contains(first) && is_float(value) {
        return Implicit::Float;
    }
    if "-+0123456789".contains(first) && is_int(value) {
        return Implicit::Int;
    }
    if first == '<' && value == "<<" {
        return Implicit::Merge;
    }
    if "~nN".contains(first) && matches!(value, "~" | "null" | "Null" | "NULL") {
        return Implicit::Null;
    }
    if first.is_ascii_digit() && timestamp(value).is_some() {
        return Implicit::Timestamp;
    }
    if value == "=" {
        return Implicit::Value;
    }
    Implicit::Str
}
fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400) => {
            29
        }
        2 => 28,
        _ => 0,
    }
}
fn int_value(text: &str) -> Number {
    // SafeConstructor.construct_yaml_int
    let cleaned = text.replace('_', "");
    let (negative, body) = match cleaned.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, cleaned.strip_prefix('+').unwrap_or(&cleaned)),
    };
    let parsed: Option<i128> = if body == "0" {
        Some(0)
    } else if let Some(b) = body.strip_prefix("0b") {
        i128::from_str_radix(b, 2).ok()
    } else if let Some(h) = body.strip_prefix("0x") {
        i128::from_str_radix(h, 16).ok()
    } else if body.starts_with('0') {
        i128::from_str_radix(body, 8).ok()
    } else if body.contains(':') {
        body.split(':').try_fold(0i128, |acc, part| {
            acc.checked_mul(60)?.checked_add(part.parse::<i128>().ok()?)
        })
    } else {
        body.parse().ok()
    };
    match parsed {
        Some(n) => Number::Small(if negative { -n } else { n }),
        None => Number::Big(format!("{}{body}", if negative { "-" } else { "" })),
    }
}
fn float_value(text: &str) -> f64 {
    let cleaned = text.replace('_', "").to_lowercase();
    let (sign, body) = match cleaned.strip_prefix('-') {
        Some(rest) => (-1.0, rest.to_owned()),
        None => (
            1.0,
            cleaned.strip_prefix('+').unwrap_or(&cleaned).to_owned(),
        ),
    };
    if body == ".inf" {
        return sign * f64::INFINITY;
    }
    if body == ".nan" {
        return f64::NAN;
    }
    if body.contains(':') {
        let mut value = 0.0;
        for part in body.split(':') {
            value = value * 60.0 + part.parse::<f64>().unwrap_or(0.0);
        }
        return sign * value;
    }
    sign * body.parse::<f64>().unwrap_or(f64::NAN)
}
/// A scalar constructed the way SafeConstructor does, or the reason it fails.
fn scalar(value: &str, plain: bool) -> Result<Value, ()> {
    if !plain {
        return Ok(Value::Str(value.into()));
    }
    Ok(match resolve(value) {
        Implicit::Str => Value::Str(value.into()),
        Implicit::Null => Value::Null,
        Implicit::Bool(b) => Value::Bool(b),
        Implicit::Int => Value::Int(int_value(value)),
        Implicit::Float => Value::Float(float_value(value)),
        // SafeConstructor has no constructor for these tags.
        Implicit::Merge | Implicit::Value => return Err(()),
        Implicit::Timestamp => {
            // datetime.date/datetime raise for impossible dates and times.
            let (year, month, day, time) = timestamp(value).unwrap();
            if year < 1 || !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month)
            {
                return Err(());
            }
            if let Some((hour, minute, second, _, _)) = time
                && (hour > 23 || minute > 59 || second > 59)
            {
                return Err(());
            }
            Value::Timestamp(value.into())
        }
    })
}
/// Python dict-key equality for constructed scalars (True == 1 == 1.0).
fn same_key(a: &Value, b: &Value) -> bool {
    fn number(v: &Value) -> Option<Number> {
        match v {
            Value::Bool(b) => Some(Number::Small(i128::from(*b))),
            Value::Int(n) => Some(n.clone()),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Float(x), other) | (other, Value::Float(x)) => match number(other) {
            Some(Number::Small(n)) => *x == n as f64,
            _ => false,
        },
        _ => match (number(a), number(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}
fn mark_node(value: Value, composed: &Composed) -> Node {
    Node {
        value,
        line: composed.mark.line as usize + 1,
        column: composed.mark.column as usize + 1,
        scalar: match &composed.raw {
            Raw::Scalar { value, .. } => Some(value.clone()),
            _ => None,
        },
    }
}
/// Construction errors in PyYAML's breadth-first order: a container's own
/// keys and scalar values are checked before the containers it holds.
fn check(document: &Composed) -> Result<(), Error> {
    let fail = |c: &Composed| invalid(Some(c.mark));
    let mut queue: VecDeque<&Composed> = VecDeque::new();
    match &document.raw {
        Raw::Scalar { value, plain } => {
            scalar(value, *plain).map_err(|_| fail(document))?;
        }
        _ => queue.push_back(document),
    }
    while let Some(node) = queue.pop_front() {
        match &node.raw {
            Raw::Sequence(items) => {
                for item in items {
                    match &item.raw {
                        Raw::Scalar { value, plain } => {
                            scalar(value, *plain).map_err(|_| fail(item))?;
                        }
                        _ => queue.push_back(item),
                    }
                }
            }
            Raw::Mapping(pairs) => {
                let mut keys: Vec<Value> = vec![];
                for (key, value) in pairs {
                    // A merge key, plain or quoted, is rejected before construction.
                    if matches!(&key.raw, Raw::Scalar { value, .. } if value == "<<") {
                        return Err(fail(key));
                    }
                    let constructed = match &key.raw {
                        Raw::Scalar { value, plain } => {
                            scalar(value, *plain).map_err(|_| fail(key))?
                        }
                        // Lists and mappings are unhashable keys.
                        _ => return Err(fail(key)),
                    };
                    if keys.iter().any(|k| same_key(k, &constructed)) {
                        return Err(fail(key));
                    }
                    keys.push(constructed);
                    match &value.raw {
                        Raw::Scalar { value: v, plain } => {
                            scalar(v, *plain).map_err(|_| fail(value))?;
                        }
                        _ => queue.push_back(value),
                    }
                }
            }
            Raw::Scalar { .. } => {}
        }
    }
    Ok(())
}
fn construct(composed: &Composed) -> Node {
    let value = match &composed.raw {
        Raw::Scalar { value, plain } => scalar(value, *plain).expect("checked scalar"),
        Raw::Sequence(items) => Value::List(items.iter().map(construct).collect()),
        Raw::Mapping(pairs) => Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (construct(k), construct(v)))
                .collect(),
        ),
    };
    mark_node(value, composed)
}

/// A UTF-16 surrogate pair escape at byte `index` (just after `\u`) as one
/// `\U` escape of the same character: PyYAML reads the pair as two code units,
/// which Director's frontend receives as that character.
fn pair_escape(text: &str, index: usize) -> Option<(std::ops::Range<usize>, String)> {
    let start = index.checked_sub(2)?;
    let escape = text.get(start..start + 12)?;
    let unit = |s: &str| u32::from_str_radix(s, 16).ok();
    let (high, low) = (escape.strip_prefix("\\u")?, escape.get(6..)?);
    let high = unit(&high[..4])?;
    let low = unit(low.strip_prefix("\\u")?)?;
    if !(0xD800..=0xDBFF).contains(&high) || !(0xDC00..=0xDFFF).contains(&low) {
        return None;
    }
    let code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
    Some((start..start + 12, format!("\\U{code:08X}")))
}
/// Every document of `text`, constructed as Director's loader does.
pub(crate) fn load(text: &str) -> Result<Vec<Node>, Error> {
    let mut text = std::borrow::Cow::Borrowed(text);
    // Each rewrite shortens its line by two characters after (line, column).
    let mut shifts: Vec<(usize, usize)> = vec![];
    loop {
        match scan(text.as_bytes()) {
            Ok(()) => break,
            Err(Ok(error)) => return Err(unshift_error(error, &shifts)),
            Err(Err(found)) => {
                let Some((range, escape)) = pair_escape(&text, found.index) else {
                    return Err(unshift_error(invalid_at(found.line, found.column), &shifts));
                };
                text.to_mut().replace_range(range, &escape);
                shifts.push((found.line, found.column - 2 + escape.len()));
            }
        }
    }
    let input = text.as_bytes();
    let documents = compose_all(input).map_err(|e| unshift_error(e, &shifts))?;
    for document in &documents {
        check(document).map_err(|e| unshift_error(e, &shifts))?;
    }
    Ok(documents
        .iter()
        .map(|d| {
            let mut node = construct(d);
            unshift_node(&mut node, &shifts);
            node
        })
        .collect())
}
fn invalid_at(line: usize, column: usize) -> Error {
    Error {
        code: "PROMPT_CATALOG_YAML_INVALID",
        line: Some(line as u32 + 1),
        column: Some(column as u32 + 1),
    }
}
/// A one-based column of the rewritten text in the original text.
fn unshift(line: usize, mut column: usize, shifts: &[(usize, usize)]) -> usize {
    for &(at_line, after) in shifts.iter().rev() {
        if line == at_line + 1 && column > after {
            column += 2;
        }
    }
    column
}
fn unshift_error(mut error: Error, shifts: &[(usize, usize)]) -> Error {
    if let (Some(line), Some(column)) = (error.line, error.column) {
        error.column = Some(unshift(line as usize, column as usize, shifts) as u32);
    }
    error
}
fn unshift_node(node: &mut Node, shifts: &[(usize, usize)]) {
    if shifts.is_empty() {
        return;
    }
    node.column = unshift(node.line, node.column, shifts);
    match &mut node.value {
        Value::List(items) => items.iter_mut().for_each(|n| unshift_node(n, shifts)),
        Value::Map(pairs) => pairs.iter_mut().for_each(|(k, v)| {
            unshift_node(k, shifts);
            unshift_node(v, shifts);
        }),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kind(text: &str) -> &'static str {
        match &load(&format!("v: {text}\n")).unwrap()[0].value {
            Value::Map(pairs) => match &pairs[0].1.value {
                Value::Str(_) => "str",
                Value::Null => "null",
                Value::Bool(_) => "bool",
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::Timestamp(_) => "timestamp",
                _ => "other",
            },
            _ => unreachable!(),
        }
    }
    #[test]
    fn plain_scalars_resolve_as_yaml_1_1() {
        for (text, expected) in [
            ("on", "bool"),
            ("No", "bool"),
            ("y", "str"),
            ("1e3", "str"),
            ("1.0e+3", "float"),
            ("0o17", "str"),
            ("017", "int"),
            ("1_000", "int"),
            ("12:30", "int"),
            ("1:30.5", "float"),
            (".5", "float"),
            ("-.inf", "float"),
            ("2001-12-14", "timestamp"),
            ("2001-12-14 21:59:43.10 -5", "timestamp"),
            ("~", "null"),
            ("", "null"),
            ("'on'", "str"),
            ("\"12\"", "str"),
            ("Prompt text", "str"),
        ] {
            assert_eq!(kind(text), expected, "{text}");
        }
    }
    #[test]
    fn unsupported_yaml_is_rejected_like_the_strict_loader() {
        let code = |t: &str| load(t).err().map(|e| e.code);
        assert_eq!(
            code("a: &x 1\nb: *x\n"),
            Some("PROMPT_CATALOG_YAML_FEATURE_UNSUPPORTED")
        );
        assert_eq!(
            code("a: !!str x\n"),
            Some("PROMPT_CATALOG_YAML_FEATURE_UNSUPPORTED")
        );
        assert_eq!(code("a: 1\na: 2\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(code("yes: 1\n1: 2\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(code("'<<': 1\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(code("a: 2001-02-30\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(code("? [a]\n: 1\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(code("a: [1\n"), Some("PROMPT_CATALOG_YAML_INVALID"));
        assert_eq!(load("a: 1\n---\nb: 2\n").unwrap().len(), 2);
        assert_eq!(load("# only a comment\n").unwrap().len(), 0);
        // A JSON-style surrogate pair escape is the character it encodes.
        let pair = load("a: \"x \\ud83d\\ude42 y\"\nb: 1\n").unwrap();
        let Value::Map(pairs) = &pair[0].value else {
            unreachable!()
        };
        assert_eq!(pairs[0].1.as_str(), Some("x 🙂 y"));
        assert_eq!(
            code("a: \"\\ude42\"\n"),
            Some("PROMPT_CATALOG_YAML_INVALID")
        );
    }
}
