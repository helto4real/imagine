//! Director's ordinary-text/@identity composer. @@ escapes a literal at sign.
use super::identifier;
use serde_json::{Value, json};
pub fn completion(text: &str, caret: usize) -> Option<(usize, String)> {
    let chars = text.chars().take(caret).collect::<Vec<_>>();
    let start = chars.iter().rposition(|c| *c == '@')?;
    if start > 0
        && (chars[start - 1].is_ascii_alphanumeric() || matches!(chars[start - 1], '_' | '@'))
    {
        return None;
    }
    let query = chars[start + 1..].iter().collect::<String>();
    query
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .then_some((start, query.to_ascii_lowercase()))
}
pub fn duplicate(container: &mut Value, index: usize) -> Option<usize> {
    let mut copy = container.get("choices")?.get(index)?.clone();
    let mut used = std::collections::BTreeSet::new();
    for c in container["choices"].as_array().into_iter().flatten() {
        if let Some(k) = c["key"].as_str() {
            used.insert(k.to_owned());
        }
        used.extend(
            c["aliases"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned),
        );
    }
    used.extend(
        container["groups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|g| g["key"].as_str())
            .map(str::to_owned),
    );
    fn numbered(raw: &str, used: &std::collections::BTreeSet<String>) -> String {
        let raw = raw.trim();
        let raw = if raw.len() <= 80 && identifier(raw) {
            raw
        } else {
            "phrase"
        };
        let suffix = raw
            .rsplit_once(['_', '-'])
            .filter(|(_, s)| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()));
        let (root, sep, mut n) = suffix.map_or((raw, '_', 2), |(r, s)| {
            (
                r,
                raw.as_bytes()[r.len()] as char,
                s.parse::<u64>()
                    .ok()
                    .and_then(|v| v.checked_add(1))
                    .unwrap_or(2),
            )
        });
        loop {
            let tail = format!("{sep}{n}");
            let base = root
                .chars()
                .take(80usize.saturating_sub(tail.len()).max(1))
                .collect::<String>();
            let base = base.trim_end_matches(['_', '-']);
            let candidate = format!("{}{tail}", if base.is_empty() { "phrase" } else { base });
            if !used.contains(&candidate) {
                return candidate;
            }
            n += 1;
        }
    }
    let key = numbered(copy["key"].as_str().unwrap_or("phrase"), &used);
    used.insert(key.clone());
    copy["key"] = json!(key);
    let mut aliases = vec![];
    for a in copy["aliases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        let a = numbered(a, &used);
        used.insert(a.clone());
        aliases.push(a);
    }
    copy["aliases"] = json!(aliases);
    let label = copy["label"].as_str().unwrap_or("Phrase").trim();
    let label = if label.is_empty() { "Phrase" } else { label };
    let suffix = label
        .rsplit_once(char::is_whitespace)
        .filter(|(_, s)| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()));
    let (base, mut n) = suffix.map_or((label, 2), |(r, s)| {
        (
            r,
            s.parse::<u64>()
                .ok()
                .and_then(|v| v.checked_add(1))
                .unwrap_or(2),
        )
    });
    loop {
        let tail = format!(" {n}");
        let candidate = format!(
            "{}{tail}",
            base.chars()
                .take(100usize.saturating_sub(tail.len()).max(1))
                .collect::<String>()
                .trim_end()
        );
        if !container["choices"].as_array()?.iter().any(|c| {
            c["label"]
                .as_str()
                .is_some_and(|l| l.to_lowercase() == candidate.to_lowercase())
        }) {
            copy["label"] = json!(candidate);
            break;
        }
        n += 1;
    }
    container
        .get_mut("choices")?
        .as_array_mut()?
        .insert(index + 1, copy);
    Some(index + 1)
}
pub fn text(choice: &Value) -> String {
    choice["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            s["text"]
                .as_str()
                .map(|s| s.replace('@', "@@"))
                .unwrap_or_else(|| {
                    s["placeholder_ref"]
                        .as_str()
                        .map(|id| format!("@{id}"))
                        .unwrap_or_default()
                })
        })
        .collect()
}
pub fn parse(text: &str, previous: &Value) -> Value {
    let chars = text.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut plain = String::new();
    let mut segments = vec![];
    let mut definitions = serde_json::Map::new();
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    while index < chars.len() {
        if chars[index] != '@' {
            plain.push(chars[index]);
            index += 1;
            continue;
        }
        if chars.get(index + 1) == Some(&'@') {
            plain.push('@');
            index += 2;
            continue;
        }
        if index > 0 && (chars[index - 1].is_ascii_alphanumeric() || chars[index - 1] == '_') {
            plain.push('@');
            index += 1;
            continue;
        }
        let mut end = index + 1;
        while end < chars.len()
            && (chars[end].is_ascii_alphanumeric() || matches!(chars[end], '_' | '-'))
        {
            end += 1;
        }
        let raw = chars[index + 1..end].iter().collect::<String>();
        let id = raw.to_ascii_lowercase();
        if !identifier(&id) {
            plain.extend(chars[index..end.max(index + 1)].iter());
            index = end.max(index + 1);
            continue;
        }
        if !plain.is_empty() {
            segments.push(json!({"text":std::mem::take(&mut plain)}));
        }
        definitions.entry(id.clone()).or_insert_with(||{
            let old=&previous[&id];let label=id.split(['-','_']).map(|w|format!("{}{}",w[..1].to_ascii_uppercase(),&w[1..])).collect::<Vec<_>>().join(" ");
            json!({"label":old["label"].as_str().unwrap_or(&label),"initial_text":old["initial_text"].as_str().map(str::to_owned).unwrap_or_else(||format!("@{raw}")),"hint":old["hint"].as_str().unwrap_or("Free text or an optional Director @reference")})
        });
        segments.push(json!({"placeholder_ref":id}));
        *counts.entry(id).or_default() += 1;
        index = end;
    }
    if !plain.is_empty() {
        segments.push(json!({"text":plain}));
    }
    json!({"placeholders":definitions,"segments":segments,"occurrences":counts})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_uses_sibling_keys_aliases_groups_and_numbered_labels() {
        let mut c = json!({"choices":[{"key":"meet","aliases":["greet"],"label":"Meeting","append":"none","segments":[{"text":"Meet"}]},{"key":"meet_2","aliases":["greet_2"],"label":"Meeting 2","append":"none","segments":[{"text":"Other"}]}],"groups":[{"key":"meet_3"}]});
        assert_eq!(duplicate(&mut c, 0), Some(1));
        assert_eq!(c["choices"][1]["key"], "meet_4");
        assert_eq!(c["choices"][1]["aliases"], json!(["greet_3"]));
        assert_eq!(c["choices"][1]["label"], "Meeting 3");
        assert_eq!(completion("🙂 @WHO", 6), Some((2, "who".into())));
        assert!(completion("@@who", 5).is_none());
        assert!(completion("mail@who", 8).is_none());
    }
    #[test]
    fn identity_links_escape_email_and_existing_metadata() {
        let old = json!({"who":{"label":"Own person","initial_text":"@keeper","hint":"Own hint"}});
        let parsed = parse("Å🙂 @WHO meets @who, @@literal email@host @-bad", &old);
        assert_eq!(parsed["occurrences"]["who"], 2);
        assert_eq!(parsed["placeholders"]["who"], old["who"]);
        assert_eq!(
            text(&parsed),
            "Å🙂 @who meets @who, @@literal email@@host @@-bad"
        );
        assert_eq!(parse(&text(&parsed), &parsed["placeholders"]), parsed);
    }
}
