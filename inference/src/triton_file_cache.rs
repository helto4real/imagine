//! Local FileCacheManager mechanics from the frozen Triton runtime/cache.py.
//! Executable compilation, custom cache classes and dynamic Python values are
//! separate components; this adapter stores bytes and ordered string-path groups.
use crate::{native::Error, triton_specialization::python_str_repr};
use std::{
    ffi::CStr,
    fmt, fs, io,
    path::{Path, PathBuf},
};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn os_error(e: io::Error, path: &Path, other: Option<&Path>) -> Error {
    let code = e.raw_os_error().unwrap_or(libc::EIO);
    let text = unsafe { CStr::from_ptr(libc::strerror(code)) }.to_string_lossy();
    let class = match code {
        libc::ENOENT => "FileNotFoundError",
        libc::EACCES | libc::EPERM => "PermissionError",
        libc::EISDIR => "IsADirectoryError",
        libc::ENOTDIR => "NotADirectoryError",
        libc::EEXIST => "FileExistsError",
        _ => "OSError",
    };
    let mut message = format!(
        "[Errno {code}] {text}: {}",
        python_str_repr(&path.to_string_lossy())
    );
    if let Some(other) = other {
        message.push_str(&format!(
            " -> {}",
            python_str_repr(&other.to_string_lossy())
        ));
    }
    error(class, message)
}
/// Python cache strings may contain unpaired surrogates. Keep code points
/// losslessly instead of forcing them through Rust's Unicode-scalar String.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheText(Vec<u32>);
impl CacheText {
    pub fn code_points(&self) -> &[u32] {
        &self.0
    }
    pub fn from_code_points(points: Vec<u32>) -> Option<Self> {
        points
            .iter()
            .all(|p| *p <= 0x10ffff)
            .then_some(Self(points))
    }
    pub fn to_utf8(&self) -> Option<String> {
        self.0.iter().map(|p| char::from_u32(*p)).collect()
    }
    fn equals(&self, text: &str) -> bool {
        self.0.iter().copied().eq(text.chars().map(|c| c as u32))
    }
    /// Frozen Linux UTF-8/surrogateescape filesystem encoding. Other unpaired
    /// surrogates fail os.path.exists through its original ValueError catch.
    pub(crate) fn filesystem_path(&self) -> Option<PathBuf> {
        use std::os::unix::ffi::OsStringExt;
        let mut out = vec![];
        for &cp in &self.0 {
            if (0xdc80..=0xdcff).contains(&cp) {
                out.push((cp - 0xdc00) as u8);
            } else {
                let c = char::from_u32(cp)?;
                let mut bytes = [0; 4];
                out.extend_from_slice(c.encode_utf8(&mut bytes).as_bytes());
            }
        }
        Some(PathBuf::from(std::ffi::OsString::from_vec(out)))
    }
}
impl From<&str> for CacheText {
    fn from(s: &str) -> Self {
        Self(s.chars().map(|c| c as u32).collect())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Cache,
    Dump,
    Override,
}
pub struct FileCache {
    directory: PathBuf,
    lock: Option<PathBuf>,
}
impl FileCache {
    pub fn new(base: &Path, key: &str, mode: Mode) -> Result<Self, Error> {
        if mode == Mode::Cache && base.as_os_str().is_empty() {
            return Err(error(
                "RuntimeError",
                "Could not create or locate cache dir",
            ));
        }
        let directory = base.join(key);
        let lock = if mode == Mode::Override {
            None
        } else {
            fs::create_dir_all(&directory).map_err(|e| os_error(e, &directory, None))?;
            Some(directory.join("lock"))
        };
        Ok(Self { directory, lock })
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn lock_path(&self) -> Option<&Path> {
        self.lock.as_deref()
    }
    pub fn get_file(&self, filename: &str) -> Option<PathBuf> {
        let path = self.directory.join(filename);
        path.exists().then_some(path)
    }
    /// Missing/unreadable/malformed group files are ignored by the source.
    /// Missing child files are omitted without discarding the remaining group.
    pub fn get_group(&self, filename: &str) -> Result<Option<Vec<(CacheText, JsonPath)>>, Error> {
        let Some(path) = self.get_file(&format!("__grp__{filename}")) else {
            return Ok(None);
        };
        let Ok(text) = fs::read_to_string(path) else {
            return Ok(None);
        };
        let Some(root) = JsonParser::parse(&text) else {
            return Ok(None);
        };
        let Json::Object(root) = root else {
            return Err(error(
                "AttributeError",
                format!("'{}' object has no attribute 'get'", root.name()),
            ));
        };
        let Some((_, children)) = root.into_iter().find(|(key, _)| key.equals("child_paths"))
        else {
            return Ok(None);
        };
        if matches!(children, Json::Null) {
            return Ok(None);
        }
        let Json::Object(children) = children else {
            return Err(error(
                "AttributeError",
                format!("'{}' object has no attribute 'items'", children.name()),
            ));
        };
        let mut out = vec![];
        for (name, value) in children {
            let value = JsonPath::from_json(value)?;
            if value.exists()? {
                out.push((name, value));
            }
        }
        Ok(Some(out))
    }
    pub fn put_group(&self, filename: &str, group: &[(String, String)]) -> Result<PathBuf, Error> {
        let group = group
            .iter()
            .map(|(k, v)| (CacheText::from(k.as_str()), CacheText::from(v.as_str())))
            .collect::<Vec<_>>();
        self.put_group_text(filename, &group)
    }
    pub fn put_group_text(
        &self,
        filename: &str,
        group: &[(CacheText, CacheText)],
    ) -> Result<PathBuf, Error> {
        let contents = format!(
            r#"{{"child_paths": {{{}}}}}"#,
            group
                .iter()
                .map(|(k, v)| format!("{}: {}", json_text(k), json_text(v)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.put(contents.as_bytes(), &format!("__grp__{filename}"))
    }
    pub fn put_group_values(
        &self,
        filename: &str,
        group: &[(CacheText, JsonPath)],
    ) -> Result<PathBuf, Error> {
        let value = |p: &JsonPath| match p {
            JsonPath::Text(s) => json_text(s),
            JsonPath::Signed(n) => n.to_string(),
            JsonPath::Unsigned(n) => n.to_string(),
            JsonPath::Bool(b) => b.to_string(),
        };
        let entries = group
            .iter()
            .map(|(k, v)| format!("{}: {}", json_text(k), value(v)))
            .collect::<Vec<_>>()
            .join(", ");
        let contents = format!("{{\"child_paths\": {{{entries}}}}}");
        self.put(contents.as_bytes(), &format!("__grp__{filename}"))
    }
    /// Original UUID/PID temporary directory + POSIX atomic replacement. No
    /// fsync, lock acquisition, eviction or cleanup-on-failure is added.
    pub fn put(&self, data: &[u8], filename: &str) -> Result<PathBuf, Error> {
        if self.lock.is_none() {
            return Err(error("AssertionError", ""));
        }
        let path = self.directory.join(filename);
        let temporary = self
            .directory
            .join(format!("tmp.pid_{}_{}", std::process::id(), uuid4()?));
        fs::create_dir_all(&temporary).map_err(|e| os_error(e, &temporary, None))?;
        let input = temporary.join(filename);
        fs::write(&input, data).map_err(|e| os_error(e, &input, None))?;
        fs::rename(&input, &path).map_err(|e| os_error(e, &input, Some(&path)))?;
        fs::remove_dir(&temporary).map_err(|e| os_error(e, &temporary, None))?;
        let mut parent = temporary.parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() || fs::remove_dir(path).is_err() {
                break;
            }
            parent = path.parent();
        }
        Ok(path)
    }
}
pub(crate) fn uuid4() -> Result<String, Error> {
    let mut bytes = [0u8; 16];
    let mut filled = 0;
    while filled < bytes.len() {
        let n = unsafe {
            libc::getrandom(bytes[filled..].as_mut_ptr().cast(), bytes.len() - filled, 0)
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(Error::Native(e.to_string()));
        }
        filled += n as usize;
    }
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    let text = bytes.iter().map(|x| format!("{x:02x}")).collect::<String>();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &text[..8],
        &text[8..12],
        &text[12..16],
        &text[16..20],
        &text[20..]
    ))
}
pub(crate) fn json_text(s: &CacheText) -> String {
    use fmt::Write;
    let mut out = String::new();
    out.push('"');
    for &cp in &s.0 {
        match cp {
            34 => out.push_str(r#"\""#),
            92 => out.push_str(r#"\\"#),
            10 => out.push_str(r#"\n"#),
            13 => out.push_str(r#"\r"#),
            9 => out.push_str(r#"\t"#),
            8 => out.push_str(r#"\b"#),
            12 => out.push_str(r#"\f"#),
            32..=126 => out.push(char::from_u32(cp).unwrap()),
            0..=0xffff => write!(out, r#"\u{cp:04x}"#).unwrap(),
            _ => {
                let cp = cp - 0x10000;
                write!(
                    out,
                    r#"\u{:04x}\u{:04x}"#,
                    0xd800 + (cp >> 10),
                    0xdc00 + (cp & 1023)
                )
                .unwrap();
            }
        }
    }
    out.push('"');
    out
}
#[derive(Debug, PartialEq)]
pub enum JsonPath {
    Text(CacheText),
    Signed(i64),
    Unsigned(u64),
    Bool(bool),
}
impl JsonPath {
    fn from_json(value: Json) -> Result<Self, Error> {
        Ok(match value {
            Json::Text(s) => Self::Text(s),
            Json::Signed(n) => Self::Signed(n),
            Json::Unsigned(n) => Self::Unsigned(n),
            Json::LargeInteger(negative) => {
                return Err(error(
                    "OverflowError",
                    if negative {
                        "fd is less than minimum"
                    } else {
                        "fd is greater than maximum"
                    },
                ));
            }
            Json::Bool(b) => Self::Bool(b),
            other => {
                return Err(error(
                    "TypeError",
                    format!(
                        "stat: path should be string, bytes, os.PathLike or integer, not {}",
                        other.name()
                    ),
                ));
            }
        })
    }
    fn exists(&self) -> Result<bool, Error> {
        let n = match self {
            Self::Text(s) => return Ok(s.filesystem_path().is_some_and(|p| p.exists())),
            Self::Signed(n) => *n,
            Self::Unsigned(n) => {
                if *n > i32::MAX as u64 {
                    return Err(error("OverflowError", "fd is greater than maximum"));
                } else {
                    *n as i64
                }
            }
            Self::Bool(b) => i64::from(*b),
        };
        if n > i32::MAX as i64 {
            return Err(error("OverflowError", "fd is greater than maximum"));
        }
        if n < i32::MIN as i64 {
            return Err(error("OverflowError", "fd is less than minimum"));
        }
        let mut info = std::mem::MaybeUninit::<libc::stat>::uninit();
        Ok(unsafe { libc::fstat(n as i32, info.as_mut_ptr()) } == 0)
    }
}
#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float,
    LargeInteger(bool),
    Text(CacheText),
    Array,
    Object(Vec<(CacheText, Json)>),
}
impl Json {
    fn name(&self) -> &'static str {
        match self {
            Self::Null => "NoneType",
            Self::Bool(_) => "bool",
            Self::Signed(_) | Self::Unsigned(_) | Self::LargeInteger(_) => "int",
            Self::Float => "float",
            Self::Text(_) => "str",
            Self::Array => "list",
            Self::Object(_) => "dict",
        }
    }
}
// JSONDecodeError is ignored by the source cache. Python's default decoder
// also accepts NaN/Infinity and arbitrary-size integers; retain those types
// instead of converting parseable-but-wrong cache structures to a cache miss.
struct JsonParser<'a> {
    text: &'a str,
    at: usize,
}
impl<'a> JsonParser<'a> {
    fn parse(text: &'a str) -> Option<Json> {
        let mut parser = Self { text, at: 0 };
        let result = parser.value(0)?;
        parser.whitespace();
        (parser.at == text.len()).then_some(result)
    }
    fn whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.at += 1;
        }
    }
    fn take(&mut self, token: &str) -> bool {
        if self.text[self.at..].starts_with(token) {
            self.at += token.len();
            true
        } else {
            false
        }
    }
    fn hex4(&self, at: usize) -> Option<u32> {
        u32::from_str_radix(self.text.get(at..at + 4)?, 16).ok()
    }
    fn string(&mut self) -> Option<CacheText> {
        if self.text.as_bytes().get(self.at) != Some(&b'"') {
            return None;
        }
        self.at += 1;
        let mut out = vec![];
        loop {
            let &b = self.text.as_bytes().get(self.at)?;
            if b == b'"' {
                self.at += 1;
                return Some(CacheText(out));
            }
            if b < 32 {
                return None;
            }
            if b != 92 {
                let c = self.text[self.at..].chars().next()?;
                self.at += c.len_utf8();
                out.push(c as u32);
                continue;
            }
            self.at += 1;
            let &escape = self.text.as_bytes().get(self.at)?;
            self.at += 1;
            let cp = match escape {
                b'"' => 34,
                92 => 92,
                b'/' => 47,
                b'b' => 8,
                b'f' => 12,
                b'n' => 10,
                b'r' => 13,
                b't' => 9,
                b'u' => {
                    let mut cp = self.hex4(self.at)?;
                    self.at += 4;
                    if (0xd800..=0xdbff).contains(&cp)
                        && self.text.as_bytes().get(self.at..self.at + 2) == Some(&[92, b'u'])
                        && let Some(low) = self.hex4(self.at + 2)
                        && (0xdc00..=0xdfff).contains(&low)
                    {
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                        self.at += 6;
                    }
                    cp
                }
                _ => return None,
            };
            out.push(cp);
        }
    }
    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth >= 1000 {
            return None;
        }
        self.whitespace();
        if self.take("null") {
            return Some(Json::Null);
        }
        if self.take("true") {
            return Some(Json::Bool(true));
        }
        if self.take("false") {
            return Some(Json::Bool(false));
        }
        if self.take("NaN") || self.take("Infinity") || self.take("-Infinity") {
            return Some(Json::Float);
        }
        match self.text.as_bytes().get(self.at)? {
            b'"' => self.string().map(Json::Text),
            b'[' => {
                self.at += 1;
                self.whitespace();
                if self.take("]") {
                    return Some(Json::Array);
                }
                loop {
                    self.value(depth + 1)?;
                    self.whitespace();
                    if self.take("]") {
                        break;
                    }
                    if !self.take(",") {
                        return None;
                    }
                }
                Some(Json::Array)
            }
            b'{' => {
                self.at += 1;
                self.whitespace();
                let mut entries: Vec<(CacheText, Json)> = vec![];
                if self.take("}") {
                    return Some(Json::Object(entries));
                }
                loop {
                    self.whitespace();
                    let key = self.string()?;
                    self.whitespace();
                    if !self.take(":") {
                        return None;
                    }
                    let value = self.value(depth + 1)?;
                    if let Some((_, current)) = entries.iter_mut().find(|(k, _)| k == &key) {
                        *current = value;
                    } else {
                        entries.push((key, value));
                    }
                    self.whitespace();
                    if self.take("}") {
                        break;
                    }
                    if !self.take(",") {
                        return None;
                    }
                }
                Some(Json::Object(entries))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }
    fn digits(&mut self) -> bool {
        let start = self.at;
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(u8::is_ascii_digit)
        {
            self.at += 1;
        }
        self.at > start
    }
    fn number(&mut self) -> Option<Json> {
        let start = self.at;
        self.take("-");
        if !self.take("0") && !self.digits() {
            return None;
        }
        let mut float = false;
        if self.take(".") {
            float = true;
            if !self.digits() {
                return None;
            }
        }
        if self.take("e") || self.take("E") {
            float = true;
            if !self.take("+") {
                self.take("-");
            }
            if !self.digits() {
                return None;
            }
        }
        if float {
            return Some(Json::Float);
        }
        let n = &self.text[start..self.at];
        if let Ok(n) = n.parse::<i64>() {
            Some(Json::Signed(n))
        } else if let Ok(n) = n.parse::<u64>() {
            Some(Json::Unsigned(n))
        } else {
            Some(Json::LargeInteger(n.starts_with('-')))
        }
    }
}
