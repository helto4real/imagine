use fancy_regex::{Captures, Regex};
use serde_json::Value;
use std::{
    io::{self, IsTerminal, Write},
    sync::{
        LazyLock,
        atomic::{AtomicI32, Ordering},
    },
};

static SPEC: LazyLock<Value> =
    LazyLock::new(|| serde_json::from_str(include_str!("../reference/logging.json")).unwrap());
static TOKENS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(SPEC["_TOKENS"].as_str().unwrap()).unwrap());
static STATUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(SPEC["_STATUS"].as_str().unwrap()).unwrap());
static STRATA: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(SPEC["_STRATA_TOKENS"].as_str().unwrap()).unwrap());
static LEVEL: AtomicI32 = AtomicI32::new(20);
fn color(key: &str) -> &'static str {
    SPEC[key].as_str().unwrap()
}
pub fn colors_enabled(tty: bool, no_color: Option<&str>, term: Option<&str>) -> bool {
    tty && no_color.is_none_or(str::is_empty) && term != Some("dumb")
}
pub fn configure(level: &str) {
    LEVEL.store(
        match level.to_uppercase().as_str() {
            "DEBUG" => 10,
            "WARNING" | "WARN" => 30,
            "ERROR" => 40,
            "CRITICAL" | "FATAL" => 50,
            "NOTSET" => 0,
            _ => 20,
        },
        Ordering::Relaxed,
    );
}
pub fn message(message: &str, level: i32) -> String {
    let successful = level < 30
        && STATUS
            .captures_iter(message)
            .all(|c| c.unwrap()[1].parse::<u16>().unwrap() < 400);
    TOKENS
        .replace_all(message, |c: &Captures<'_>| {
            let field = c.name("field").map(|m| m.as_str());
            let v = if field.is_some() {
                c.name("value").unwrap().as_str()
            } else {
                c.get(0).unwrap().as_str()
            };
            let col = match field {
                Some("status") if v.len() == 3 && v.bytes().all(|b| b.is_ascii_digit()) => {
                    match v.as_bytes()[0] {
                        b'2' => Some("_GREEN"),
                        b'4' => Some("_YELLOW"),
                        b'5' => Some("_RED"),
                        _ => None,
                    }
                }
                Some("duration") => Some("_TEAL"),
                Some("state" | "expected") => Some(if v == "error" { "_RED" } else { "_PEACH" }),
                Some("reason" | "error") => Some("_RED"),
                _ if c.name("transition").is_some() => {
                    Some(if v == "error" { "_RED" } else { "_PEACH" })
                }
                _ if c.name("detail").is_some() => Some("_RED"),
                _ if c.name("backend").is_some() => Some("_MAUVE"),
                _ if c.name("completed").is_some() && successful => Some("_GREEN"),
                _ if c.name("failure").is_some() => {
                    Some(if level >= 40 { "_RED" } else { "_YELLOW" })
                }
                _ => None,
            };
            match col {
                Some(col) => format!(
                    "{}{}{v}{}",
                    field.map(|f| format!("{f}=")).unwrap_or_default(),
                    color(col),
                    color("_TEXT")
                ),
                None => c[0].to_string(),
            }
        })
        .into_owned()
}
pub fn format(timestamp: &str, level: i32, name: &str, text: &str, colored: bool) -> String {
    let label = match level {
        10 => "DEBUG".into(),
        20 => "INFO".into(),
        30 => "WARNING".into(),
        40 => "ERROR".into(),
        50 => "CRITICAL".into(),
        0 => "NOTSET".into(),
        _ => format!("Level {level}"),
    };
    if !colored {
        return format!("{timestamp} {label} {name}: {text}");
    }
    let lc = color(if level >= 40 {
        "_RED"
    } else if level >= 30 {
        "_YELLOW"
    } else if level >= 20 {
        "_BLUE"
    } else {
        "_GRAY"
    });
    let gray = color("_GRAY");
    let reset = color("_RESET");
    let base = color("_TEXT");
    format!(
        "{gray}{timestamp}{reset} {lc}{label}{reset} {gray}{name}{reset}: {base}{}{reset}",
        message(text, level)
    )
}
pub fn strata_line(line: &str) -> String {
    let Some(body) = line.strip_prefix("[strata] ") else {
        return line.into();
    };
    if line.contains("\x1b[") {
        return line.into();
    }
    let text = body.trim_end_matches(['\r', '\n']);
    let ending = &body[text.len()..];
    let highlighted = STRATA.replace_all(text, |c: &Captures<'_>| {
        let v = c.get(0).unwrap().as_str();
        let col = if c.name("phase").is_some() {
            if v == "done"
                && (Regex::new(r"\((?:error|failed)\b")
                    .unwrap()
                    .is_match(text)
                    .unwrap())
            {
                color("_RED")
            } else if v == "done" && text.contains("cancel=True") {
                color("_YELLOW")
            } else {
                SPEC["_STRATA_PHASES"][v].as_str().unwrap()
            }
        } else {
            color(if c.name("cancel").is_some() {
                "_YELLOW"
            } else {
                "_TEAL"
            })
        };
        format!("{col}{v}{}", color("_TEXT"))
    });
    format!(
        "{}[strata]{} {highlighted}{}{ending}",
        color("_MAUVE"),
        color("_TEXT"),
        color("_RESET")
    )
}
pub fn log(level: i32, name: &str, text: impl AsRef<str>) {
    if level < LEVEL.load(Ordering::Relaxed) {
        return;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let seconds = now.as_secs() as libc::time_t;
    let mut tm = std::mem::MaybeUninit::<libc::tm>::uninit();
    // localtime_r writes a thread-local result into our owned storage.
    let tm = unsafe {
        libc::localtime_r(&seconds, tm.as_mut_ptr());
        tm.assume_init()
    };
    let stamp = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02},{:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        now.subsec_millis()
    );
    let colored = colors_enabled(
        io::stderr().is_terminal(),
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("TERM").ok().as_deref(),
    );
    let _ = writeln!(
        io::stderr().lock(),
        "{}",
        format(&stamp, level, name, text.as_ref(), colored)
    );
}
