use crate::video::text::{self, PythonText};
use std::collections::BTreeMap;
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Maps {
    pub subject: BTreeMap<String, String>,
    pub speaker: BTreeMap<String, String>,
    pub language: BTreeMap<String, String>,
    pub voice_audio: BTreeMap<String, String>,
    pub voice_profile: BTreeMap<String, String>,
}
#[derive(Clone, serde::Serialize)]
pub struct Cue {
    pub alias: String,
    pub token: String,
    pub verb: String,
    pub context: Option<String>,
    pub language: Option<String>,
    #[serde(skip)]
    raw_language: Option<String>,
    pub dialogue: String,
    pub start: usize,
    pub end: usize,
    #[serde(skip)]
    from: usize,
    #[serde(skip)]
    to: usize,
}
fn ws(text: &str, pos: &mut usize) -> bool {
    let before = *pos;
    while let Some(c) = text[*pos..].chars().next() {
        if !text::space(c) {
            break;
        }
        *pos += c.len_utf8();
    }
    *pos > before
}
fn alias(value: &str) -> String {
    let mut result = String::new();
    let mut invalid = false;
    for c in value.to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-') {
            result.push(c);
            invalid = false;
        } else if !invalid {
            result.push('_');
            invalid = true;
        }
    }
    let result = result.trim_matches(['_', '-']);
    if result.starts_with(|c: char| c.is_ascii_lowercase()) {
        result.into()
    } else {
        "ref1".into()
    }
}
fn verbs() -> Vec<(&'static str, &'static str)> {
    let mut list = vec![];
    for group in [
        "say says saying",
        "speak speaks speaking",
        "ask asks asking",
        "reply replies replying",
        "exclaim exclaims exclaiming",
        "whisper whispers whispering",
        "murmur murmurs murmuring",
        "mutter mutters muttering",
        "shout shouts shouting",
        "yell yells yelling",
        "scream screams screaming",
        "sing sings singing",
    ] {
        let parts = group.split(' ').collect::<Vec<_>>();
        for p in &parts {
            list.push((*p, parts[1]));
        }
    }
    list.sort_by_key(|(v, _)| std::cmp::Reverse(v.len()));
    list
}
pub fn parse(text: &str) -> Vec<Cue> {
    let mut result = vec![];
    let mut consumed = 0;
    let verbs = verbs();
    for (from, c) in text.char_indices() {
        if from < consumed
            || c != '@'
            || text[..from].chars().next_back().is_some_and(|c| {
                text::ascii_case(c).is_ascii_alphanumeric() || matches!(c, '_' | '@')
            })
        {
            continue;
        }
        let mut p = from + 1;
        if !text[p..]
            .chars()
            .next()
            .is_some_and(|c| text::ascii_case(c).is_ascii_alphabetic())
        {
            continue;
        }
        p += text[p..].chars().next().unwrap().len_utf8();
        while let Some(c) = text[p..].chars().next() {
            if !text::ascii_case(c).is_ascii_alphanumeric() && !matches!(c, '_' | '-') {
                break;
            }
            p += c.len_utf8();
        }
        let alias = alias(&text[from + 1..p]);
        if !ws(text, &mut p) {
            continue;
        }
        let Some(length) = verbs.iter().find_map(|(verb, _)| {
            let length = text::prefix_case(&text[p..], verb)?;
            (!text[p + length..].chars().next().is_some_and(text::word)).then_some(length)
        }) else {
            continue;
        };
        let raw_verb = text[p..p + length].to_lowercase();
        let canonical = verbs
            .iter()
            .find(|(v, _)| *v == raw_verb)
            .map_or(raw_verb.as_str(), |(_, v)| *v)
            .to_owned();
        p += length;
        let mut context = None;
        let after_verb = p;
        let mut matched_context = false;
        if ws(text, &mut p) {
            let begin = p;
            if let Some(length) = text::prefix_case(&text[p..], "off-screen") {
                p += length;
                matched_context = true;
            } else if let Some(length) = text::prefix_case(&text[p..], "in") {
                p += length;
                if ws(text, &mut p)
                    && let Some(length) = text::prefix_case(&text[p..], "voiceover")
                {
                    p += length;
                    matched_context = true;
                }
            }
            if matched_context {
                context = match text::single(&text[begin..p].to_lowercase()).as_str() {
                    "off-screen" => Some("off-screen".into()),
                    "in voiceover" => Some("voiceover".into()),
                    _ => None,
                };
            }
        }
        if !matched_context {
            p = after_verb;
        }
        let mut language = None;
        let after_context = p;
        if ws(text, &mut p) && text.as_bytes().get(p) == Some(&b'[') {
            let begin = p + 1;
            if let Some(end) = text[begin..].find(']') {
                let value = &text[begin..begin + end];
                if !value.is_empty() && !value.contains(['\r', '\n']) {
                    language = Some(value.into());
                    p = begin + end + 1;
                } else {
                    p = after_context;
                }
            } else {
                p = after_context;
            }
        } else {
            p = after_context;
        }
        if !ws(text, &mut p) || text.as_bytes().get(p) != Some(&b'"') {
            continue;
        }
        p += 1;
        let begin = p;
        let mut end = None;
        while p < text.len() {
            let c = text[p..].chars().next().unwrap();
            if c == '"' {
                end = Some(p);
                p += 1;
                break;
            }
            if c == '\r' || c == '\n' {
                break;
            }
            if c == '\\' {
                p += 1;
                let Some(escaped) = text[p..].chars().next() else {
                    break;
                };
                if escaped == '\n' {
                    break;
                }
                p += escaped.len_utf8();
            } else {
                p += c.len_utf8();
            }
        }
        let Some(end) = end else {
            continue;
        };
        consumed = p;
        result.push(Cue {
            alias,
            token: text[from..p].into(),
            verb: canonical,
            context,
            raw_language: language.clone(),
            language: language
                .map(|s| s.pytrim().to_owned())
                .filter(|s| !s.is_empty()),
            dialogue: text[begin..end].replace("\\\"", "\"").replace("\\\\", "\\"),
            start: text[..from].chars().count(),
            end: text[..p].chars().count(),
            from,
            to: p,
        });
    }
    result
}
pub fn compile(text: &str, maps: &Maps) -> String {
    let mut out = String::new();
    let mut cursor = 0;
    for cue in parse(text) {
        out.push_str(&text[cursor..cue.from]);
        cursor = cue.to;
        let Some(subject) = maps.subject.get(&cue.alias).filter(|s| !s.is_empty()) else {
            out.push_str(&cue.token);
            continue;
        };
        let Some(speaker) = maps.speaker.get(&cue.alias).filter(|s| !s.is_empty()) else {
            out.push_str(&cue.token);
            continue;
        };
        let language = cue
            .raw_language
            .as_deref()
            .or_else(|| {
                maps.language
                    .get(&cue.alias)
                    .filter(|s| !s.is_empty())
                    .map(String::as_str)
            })
            .unwrap_or("English")
            .pytrim();
        let language = if language.is_empty() {
            "English"
        } else {
            language
        };
        let profile = maps
            .voice_profile
            .get(&cue.alias)
            .map_or("", String::as_str)
            .pytrim()
            .trim_end_matches('.');
        let audio = maps.voice_audio.get(&cue.alias).map_or("", String::as_str);
        let prefix = format!("{subject} ({speaker})");
        if cue.context.as_deref() == Some("voiceover") {
            let mut modifiers = vec![];
            if !profile.is_empty() {
                modifiers.push(format!("speaking in {profile}"));
            }
            if !audio.is_empty() {
                modifiers.push(format!("using the voice timbre referenced from {audio}"));
            }
            let delivery = match cue.verb.as_str() {
                "asks" => "as a question",
                "replies" => "as a reply",
                "exclaims" => "as an exclamation",
                "whispers" => "in a whisper",
                "murmurs" => "in a murmur",
                "mutters" => "in a mutter",
                "shouts" => "as a shout",
                "yells" => "as a yell",
                "screams" => "as a scream",
                "sings" => "as sung vocals",
                _ => "",
            };
            if !delivery.is_empty() {
                modifiers.push(format!("delivering the line {delivery}"));
            }
            let prefix = if modifiers.is_empty() {
                prefix
            } else {
                format!("{prefix}, {},", super::common::english(&modifiers))
            };
            out.push_str(&format!("{prefix} says in an off-screen voiceover: <d>[{language}] {}</d> while their lips remain completely closed",cue.dialogue));
        } else {
            let speaker = if audio.is_empty() {
                prefix
            } else {
                format!("{prefix}, using the voice timbre referenced from {audio},")
            };
            let context = if cue.context.as_deref() == Some("off-screen") {
                " off-screen"
            } else {
                ""
            };
            let delivery = if profile.is_empty() {
                String::new()
            } else {
                format!(" in {profile}")
            };
            out.push_str(&format!(
                "{speaker} {}{context}{delivery}: <d>[{language}] {}</d>",
                cue.verb, cue.dialogue
            ));
        }
    }
    out.push_str(&text[cursor..]);
    out
}
