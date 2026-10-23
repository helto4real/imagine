use crate::video::text::{self, PythonText};
use std::collections::BTreeMap;
pub(super) const AMBIENT: &str = "Ambient sounds that fit the scene action and background sounds";
pub(super) fn single(s: &str) -> String {
    text::single(s)
}
pub(super) fn dedupe(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out = vec![];
    for s in values {
        let s = s.pytrim().to_owned();
        if !s.is_empty() && !out.contains(&s) {
            out.push(s);
        }
    }
    out
}
pub(super) fn english(v: &[String]) -> String {
    match v.len() {
        0 => String::new(),
        1 => v[0].clone(),
        2 => format!("{} and {}", v[0], v[1]),
        n => format!("{}, and {}", v[..n - 1].join(", "), v[n - 1]),
    }
}
pub(super) fn join(a: &str, b: &str) -> String {
    let (a, b) = (a.pytrim(), b.pytrim());
    if a.is_empty() {
        return b.into();
    }
    if b.is_empty() {
        return a.into();
    }
    format!(
        "{a}{}{b}",
        if a.ends_with(['.', '!', '?', ':', ';']) {
            " "
        } else {
            ". "
        }
    )
}
pub(super) fn values(v: Vec<String>) -> String {
    let s = dedupe(v)
        .into_iter()
        .fold(String::new(), |a, b| join(&a, &b));
    if s.is_empty() { "N/A".into() } else { s }
}
pub(super) fn timestamp(seconds: f64) -> String {
    let n = (seconds * 1000.).round_ties_even().max(0.) as u64;
    format!("{:02}:{:02}.{:03}", n / 60000, n / 1000 % 60, n % 1000)
}
pub(super) fn replace(text: &str, maps: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    let mut cursor = 0;
    for token in imagine_video_contract::references::tokens(text) {
        out.push_str(&text[cursor..token.start]);
        out.push_str(
            maps.get(&token.alias)
                .filter(|s| !s.is_empty())
                .map(String::as_str)
                .unwrap_or(&text[token.start..token.end]),
        );
        cursor = token.end;
    }
    out.push_str(&text[cursor..]);
    out
}
pub(super) fn channels(text: &str) -> (String, Vec<String>, Vec<String>) {
    let (mut visual, mut audio, mut music): (Vec<String>, Vec<String>, Vec<String>) =
        (vec![], vec![], vec![]);
    for line in text::lines(text)
        .map(PythonText::pytrim)
        .filter(|s| !s.is_empty())
    {
        let mut matches = vec![];
        for (start, _) in line.char_indices() {
            if line[..start].chars().next_back().is_some_and(text::word) {
                continue;
            }
            for label in [
                "audio",
                "soundscape",
                "sound",
                "sfx",
                "music",
                "score",
                "bgm",
            ] {
                let Some(length) = text::prefix_case(&line[start..], label) else {
                    continue;
                };
                let mut end = start + length;
                let actual_label = line[start..end].to_lowercase();
                while line[end..].chars().next().is_some_and(text::space) {
                    end += line[end..].chars().next().unwrap().len_utf8();
                }
                if line.as_bytes().get(end) != Some(&b':') {
                    continue;
                }
                end += 1;
                while line[end..].chars().next().is_some_and(text::space) {
                    end += line[end..].chars().next().unwrap().len_utf8();
                }
                matches.push((
                    start,
                    end,
                    matches!(actual_label.as_str(), "music" | "score" | "bgm"),
                ));
                break;
            }
        }
        if matches.is_empty() {
            visual.push(line.into());
            continue;
        }
        let prefix = line[..matches[0].0].trim_matches([' ', ';']);
        if !prefix.is_empty() {
            visual.push(prefix.into());
        }
        for (i, (_, start, score)) in matches.iter().enumerate() {
            let end = matches.get(i + 1).map_or(line.len(), |m| m.0);
            if *start > end {
                continue;
            }
            let s = line[*start..end].trim_matches([' ', ';']);
            if !s.is_empty() {
                if *score {
                    music.push(s.into());
                } else {
                    audio.push(s.into());
                }
            }
        }
    }
    (visual.join(" "), audio, music)
}
