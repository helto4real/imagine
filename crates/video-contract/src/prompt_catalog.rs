//! Director catalog grammar, replace/merge and model-neutral snippet expansion.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub mod composer;
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub schema_version: String,
    pub namespaces: Vec<Container>,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Container {
    pub key: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub groups: Vec<Container>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub key: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub label: String,
    #[serde(default)]
    pub description: String,
    pub append: String,
    pub segments: Vec<Segment>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Segment {
    Text { text: String },
    Placeholder { placeholder: Placeholder },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placeholder {
    pub id: String,
    pub label: String,
    pub initial_text: String,
    pub hint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Issue {
    pub source: String,
    pub path: String,
    pub message: String,
}
impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.source, self.path, self.message)
    }
}
impl std::error::Error for Issue {}
type Result<T> = std::result::Result<T, Issue>;
fn issue(source: &str, path: &str, message: &str) -> Issue {
    Issue {
        source: source.into(),
        path: path.into(),
        message: message.into(),
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema_version: String,
    mode: String,
    namespaces: Vec<RawContainer>,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RawContainer {
    key: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    choices: Vec<RawChoice>,
    #[serde(default)]
    groups: Vec<RawContainer>,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RawChoice {
    key: String,
    #[serde(default)]
    aliases: Vec<String>,
    label: String,
    #[serde(default)]
    description: Option<String>,
    append: String,
    segments: Vec<serde_json::Value>,
    #[serde(default)]
    placeholders: BTreeMap<String, Definition>,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Definition {
    label: String,
    initial_text: String,
    hint: String,
}
#[derive(Clone)]
struct Parsed {
    mode: String,
    namespaces: Vec<Container>,
    metadata: BTreeMap<Vec<String>, (bool, bool)>,
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value.split(['-', '_']).all(|p| {
            !p.is_empty()
                && p.bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        })
}
fn line(value: &str) -> bool {
    !value.is_empty()
        && value.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
            == value
        && !value.contains(['\r', '\n'])
}
fn fragment(value: &str) -> bool {
    !value.is_empty() && !value.contains(['\r', '\n', '#'])
}
fn choice(raw: RawChoice, version: &str, source: &str, path: &str) -> Result<Choice> {
    if !identifier(&raw.key)
        || !line(&raw.label)
        || !matches!(raw.append.as_str(), "none" | "space")
        || raw.segments.is_empty()
    {
        return Err(issue(
            source,
            path,
            "Choice requires a key, label, append none/space and segments.",
        ));
    }
    let mut aliases = BTreeSet::new();
    for alias in &raw.aliases {
        if !identifier(alias) || !aliases.insert(alias) {
            return Err(issue(
                source,
                path,
                "Choice aliases must be distinct identifiers.",
            ));
        }
    }
    if raw.description.as_ref().is_some_and(|v| !line(v))
        || version == "1.0" && raw.description.is_some()
    {
        return Err(issue(
            source,
            path,
            "Description must be a trimmed single line.",
        ));
    }
    if version == "1.0" && !raw.placeholders.is_empty() {
        return Err(issue(
            source,
            path,
            "Named placeholder definitions require schema 1.1.",
        ));
    }
    let mut definitions = BTreeMap::new();
    for (id, d) in raw.placeholders {
        if !identifier(&id) || !line(&d.label) || !fragment(&d.initial_text) || !line(&d.hint) {
            return Err(issue(source, path, "Invalid named placeholder definition."));
        }
        definitions.insert(
            id.clone(),
            Placeholder {
                id,
                label: d.label,
                initial_text: d.initial_text,
                hint: d.hint,
            },
        );
    }
    let mut segments = vec![];
    let mut ids = BTreeSet::new();
    for (i, s) in raw.segments.into_iter().enumerate() {
        let at = format!("{path}/segments/{i}");
        let map = s.as_object().filter(|m| m.len() == 1).ok_or_else(|| {
            issue(
                source,
                &at,
                "Each segment has exactly one text or placeholder value.",
            )
        })?;
        if let Some(text) = map.get("text").and_then(|s| s.as_str()) {
            if !fragment(text) {
                return Err(issue(
                    source,
                    &at,
                    "Prompt fragments must be nonempty single lines without hash directives.",
                ));
            }
            segments.push(Segment::Text { text: text.into() });
        } else if version == "1.1" {
            let id = map
                .get("placeholder_ref")
                .and_then(|v| v.as_str())
                .ok_or_else(|| issue(source, &at, "Schema 1.1 uses placeholder_ref."))?;
            let placeholder = definitions
                .get(id)
                .ok_or_else(|| issue(source, &at, "Unknown placeholder_ref in this choice."))?;
            segments.push(Segment::Placeholder {
                placeholder: placeholder.clone(),
            });
        } else {
            let placeholder: Placeholder = serde_json::from_value(
                map.get("placeholder")
                    .cloned()
                    .ok_or_else(|| issue(source, &at, "Schema 1.0 uses inline placeholder."))?,
            )
            .map_err(|_| {
                issue(
                    source,
                    &at,
                    "Placeholder requires id, label, initial_text and hint.",
                )
            })?;
            if !identifier(&placeholder.id)
                || !ids.insert(placeholder.id.clone())
                || !line(&placeholder.label)
                || !fragment(&placeholder.initial_text)
                || !line(&placeholder.hint)
            {
                return Err(issue(source, &at, "Invalid or duplicate placeholder."));
            }
            segments.push(Segment::Placeholder { placeholder });
        }
    }
    Ok(Choice {
        key: raw.key,
        aliases: raw.aliases,
        label: raw.label,
        description: raw.description.unwrap_or_default(),
        append: raw.append,
        segments,
    })
}
fn container(
    raw: RawContainer,
    version: &str,
    mode: &str,
    source: &str,
    path: &str,
    keys: Vec<String>,
    meta: &mut BTreeMap<Vec<String>, (bool, bool)>,
) -> Result<Container> {
    if !identifier(&raw.key) || raw.choices.is_empty() && raw.groups.is_empty() {
        return Err(issue(
            source,
            path,
            "Container requires an identifier and choices or groups.",
        ));
    }
    if raw.label.as_ref().is_some_and(|v| !line(v))
        || raw.description.as_ref().is_some_and(|v| !line(v))
    {
        return Err(issue(
            source,
            path,
            "Container metadata must be trimmed single lines.",
        ));
    }
    if mode == "replace" && (raw.label.is_none() || (keys.len() == 1 && raw.description.is_none()))
    {
        return Err(issue(
            source,
            path,
            "Replace namespaces require label/description; groups require label.",
        ));
    }
    meta.insert(
        keys.clone(),
        (raw.label.is_some(), raw.description.is_some()),
    );
    let mut choices = vec![];
    let mut seen = BTreeSet::new();
    for (i, c) in raw.choices.into_iter().enumerate() {
        if !seen.insert(c.key.clone()) {
            return Err(issue(source, path, "Duplicate choice key."));
        }
        choices.push(choice(c, version, source, &format!("{path}/choices/{i}"))?);
    }
    let mut groups = vec![];
    let mut seen = BTreeSet::new();
    for (i, g) in raw.groups.into_iter().enumerate() {
        if !seen.insert(g.key.clone()) {
            return Err(issue(source, path, "Duplicate group key."));
        }
        let mut next = keys.clone();
        next.push(g.key.clone());
        groups.push(container(
            g,
            version,
            mode,
            source,
            &format!("{path}/groups/{i}"),
            next,
            meta,
        )?);
    }
    Ok(Container {
        key: raw.key,
        label: raw.label.unwrap_or_default(),
        description: raw.description.unwrap_or_default(),
        choices,
        groups,
    })
}
fn parse(source: &str, text: &str) -> Result<Parsed> {
    let raw: Document =
        serde_yaml::from_str(text).map_err(|e| issue(source, "/", &e.to_string()))?;
    if !matches!(raw.schema_version.as_str(), "1.0" | "1.1")
        || !matches!(raw.mode.as_str(), "replace" | "merge")
        || raw.namespaces.is_empty()
    {
        return Err(issue(
            source,
            "/",
            "Catalog requires schema 1.0/1.1, mode replace/merge and namespaces.",
        ));
    }
    let mut metadata = BTreeMap::new();
    let mut namespaces = vec![];
    let mut seen = BTreeSet::new();
    for (i, n) in raw.namespaces.into_iter().enumerate() {
        if !seen.insert(n.key.clone()) {
            return Err(issue(source, "/namespaces", "Duplicate namespace key."));
        }
        let keys = vec![n.key.clone()];
        namespaces.push(container(
            n,
            &raw.schema_version,
            &raw.mode,
            source,
            &format!("/namespaces/{i}"),
            keys,
            &mut metadata,
        )?);
    }
    Ok(Parsed {
        mode: raw.mode,
        namespaces,
        metadata,
    })
}
fn merge(
    target: &mut Vec<Container>,
    incoming: Vec<Container>,
    keys: Vec<String>,
    meta: &BTreeMap<Vec<String>, (bool, bool)>,
) {
    for next in incoming {
        if let Some(current) = target.iter_mut().find(|c| c.key == next.key) {
            let mut path = keys.clone();
            path.push(next.key.clone());
            let (label, description) = meta[&path];
            if label {
                current.label = next.label;
            }
            if description {
                current.description = next.description;
            }
            for choice in next.choices {
                if let Some(old) = current.choices.iter_mut().find(|c| c.key == choice.key) {
                    *old = choice;
                } else {
                    current.choices.push(choice);
                }
            }
            merge(&mut current.groups, next.groups, path, meta);
        } else {
            target.push(next);
        }
    }
}
fn validate(container: &Container, path: &str) -> Result<()> {
    if container.label.is_empty() {
        return Err(issue(
            "merged",
            path,
            "Every resolved container needs a label.",
        ));
    }
    let mut owners = BTreeMap::new();
    for c in &container.choices {
        for id in std::iter::once(&c.key).chain(&c.aliases) {
            if owners.insert(id, &c.key).is_some_and(|old| old != &c.key) {
                return Err(issue(
                    "merged",
                    path,
                    "Choice aliases must resolve to one item.",
                ));
            }
        }
    }
    for g in &container.groups {
        if owners.contains_key(&g.key) {
            return Err(issue(
                "merged",
                path,
                "Group collides with a choice or alias.",
            ));
        }
        validate(g, &format!("{path}.{}", g.key))?;
    }
    Ok(())
}
pub fn load(sources: &BTreeMap<String, String>) -> Result<Catalog> {
    if sources.is_empty() {
        return Err(issue(
            "catalog",
            "/",
            "Choose at least one YAML catalog file.",
        ));
    }
    let mut parsed = sources
        .iter()
        .map(|(n, t)| parse(n, t))
        .collect::<Result<Vec<_>>>()?;
    if parsed.iter().filter(|p| p.mode == "replace").count() > 1 {
        return Err(issue(
            "catalog",
            "/",
            "Only one source may use replace mode.",
        ));
    }
    parsed.sort_by_key(|p| p.mode != "replace");
    let mut namespaces = vec![];
    for p in parsed {
        merge(&mut namespaces, p.namespaces, vec![], &p.metadata);
    }
    for n in &namespaces {
        if n.description.is_empty() {
            return Err(issue(
                "merged",
                &n.key,
                "Every namespace needs a description.",
            ));
        }
        validate(n, &n.key)?;
    }
    Ok(Catalog {
        schema_version: "1.1".into(),
        namespaces,
    })
}
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub start: usize,
    pub end: usize,
    pub namespace: String,
    pub groups: Vec<String>,
    pub query: String,
    pub namespace_stage: bool,
}
pub fn context(text: &str, start: usize, end: usize) -> Option<Context> {
    if start != end {
        return None;
    }
    let chars = text.chars().take(start).collect::<Vec<_>>();
    let at = chars.iter().rposition(|c| *c == '#')?;
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '#';
    if at > 0 && is_word(chars[at - 1]) {
        return None;
    }
    let token = chars[at + 1..].iter().collect::<String>();
    if !token
        .chars()
        .all(|c| c.is_alphanumeric() || "_-.".contains(c))
    {
        return None;
    }
    let bits = token.split('.').collect::<Vec<_>>();
    let namespace_stage = bits.len() == 1;
    Some(Context {
        start: at,
        end,
        namespace: bits[0].into(),
        groups: if bits.len() > 2 {
            bits[1..bits.len() - 1]
                .iter()
                .map(|s| (*s).into())
                .collect()
        } else {
            vec![]
        },
        query: if namespace_stage {
            String::new()
        } else {
            bits.last().unwrap().to_string()
        },
        namespace_stage,
    })
}
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub label: String,
    pub detail: String,
    pub path: Vec<String>,
    pub choice: Option<Choice>,
}
impl Catalog {
    pub fn completions(&self, context: &Context) -> Vec<Completion> {
        fn search(text: &str) -> String {
            text.to_lowercase()
                .chars()
                .filter(|c| !c.is_whitespace() && !matches!(c, '_' | '-'))
                .collect()
        }
        let mut out = vec![];
        for n in self.namespaces.iter().filter(|n| {
            search(&n.key).starts_with(&search(&context.namespace))
                || search(&n.label).starts_with(&search(&context.namespace))
        }) {
            if context.namespace_stage {
                out.push(Completion {
                    label: n.label.clone(),
                    detail: n.description.clone(),
                    path: vec![n.key.clone()],
                    choice: None,
                });
                continue;
            }
            let mut parent = n;
            let mut valid = true;
            for key in &context.groups {
                if let Some(g) = parent.groups.iter().find(|g| &g.key == key) {
                    parent = g;
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }
            let mut path = vec![n.key.clone()];
            path.extend(context.groups.clone());
            let query = search(&context.query);
            for g in &parent.groups {
                if [&g.key, &g.label, &g.description]
                    .iter()
                    .any(|s| search(s).starts_with(&query))
                {
                    let mut path = path.clone();
                    path.push(g.key.clone());
                    out.push(Completion {
                        label: g.label.clone(),
                        detail: g.description.clone(),
                        path,
                        choice: None,
                    });
                }
            }
            for c in &parent.choices {
                if std::iter::once(&c.key)
                    .chain(std::iter::once(&c.label))
                    .chain(&c.aliases)
                    .any(|s| search(s).starts_with(&query))
                {
                    let mut path = path.clone();
                    path.push(c.key.clone());
                    out.push(Completion {
                        label: c.label.clone(),
                        detail: c.preview(),
                        path,
                        choice: Some(c.clone()),
                    });
                }
            }
        }
        out.sort_by(|a, b| natural_cmp(&a.label, &b.label).then(a.path.cmp(&b.path)));
        out.truncate(50);
        out
    }
}
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    let mut a = a.chars().peekable();
    let mut b = b.chars().peekable();
    while let (Some(x), Some(y)) = (a.peek().copied(), b.peek().copied()) {
        if x.is_ascii_digit() && y.is_ascii_digit() {
            let mut aa = String::new();
            let mut bb = String::new();
            while a.peek().is_some_and(char::is_ascii_digit) {
                aa.push(a.next().unwrap());
            }
            while b.peek().is_some_and(char::is_ascii_digit) {
                bb.push(b.next().unwrap());
            }
            let aa = aa.trim_start_matches('0');
            let bb = bb.trim_start_matches('0');
            let order = aa.len().cmp(&bb.len()).then(aa.cmp(bb));
            if order.is_ne() {
                return order;
            }
        } else {
            a.next();
            b.next();
            if x != y {
                return x.cmp(&y);
            }
        }
    }
    a.count().cmp(&b.count())
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Occurrence {
    pub start: usize,
    pub end: usize,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    pub id: String,
    pub label: String,
    pub hint: String,
    pub occurrences: Vec<Occurrence>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expansion {
    pub text: String,
    pub slots: Vec<Slot>,
}
impl Choice {
    pub fn preview(&self) -> String {
        self.expand(&BTreeMap::new()).text
    }
    pub fn expand(&self, values: &BTreeMap<String, String>) -> Expansion {
        let mut text = String::new();
        let mut slots: Vec<Slot> = vec![];
        let mut keys = vec![];
        for segment in &self.segments {
            match segment {
                Segment::Text { text: part } => text.push_str(part),
                Segment::Placeholder { placeholder: p } => {
                    let identity = p.initial_text.strip_prefix('@').filter(|s| {
                        !s.is_empty()
                            && s.chars()
                                .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
                    });
                    let key = identity.map_or_else(
                        || format!("id:{}", p.id),
                        |s| format!("identity:{}", s.to_lowercase()),
                    );
                    let index = keys.iter().position(|k| k == &key).unwrap_or_else(|| {
                        keys.push(key);
                        slots.push(Slot {
                            id: p.id.clone(),
                            label: p.label.clone(),
                            hint: p.hint.clone(),
                            occurrences: vec![],
                        });
                        slots.len() - 1
                    });
                    let value = values
                        .get(&slots[index].id)
                        .map_or(p.initial_text.as_str(), String::as_str);
                    let start = text.chars().count();
                    text.push_str(value);
                    slots[index].occurrences.push(Occurrence {
                        start,
                        end: text.chars().count(),
                    });
                }
            }
        }
        if self.append == "space" {
            text.push(' ');
        }
        Expansion { text, slots }
    }
}

pub fn source_value(text: &str) -> std::result::Result<serde_json::Value, String> {
    parse("draft", text).map_err(|e| e.to_string())?;
    serde_yaml::from_str(text).map_err(|e| e.to_string())
}
pub fn source_text(value: &serde_json::Value) -> std::result::Result<String, String> {
    serde_yaml::to_string(value).map_err(|e| e.to_string())
}

#[derive(Clone, Debug)]
pub struct Session {
    pub slots: Vec<Slot>,
    pub active: usize,
    pub occurrence: usize,
}
impl Session {
    pub fn new(mut slots: Vec<Slot>, offset: usize) -> Self {
        for s in &mut slots {
            for o in &mut s.occurrences {
                o.start += offset;
                o.end += offset;
            }
        }
        Self {
            slots,
            active: 0,
            occurrence: 0,
        }
    }
    pub fn selection(&self) -> Option<Occurrence> {
        self.slots.get(self.active)?.occurrences.first().cloned()
    }
    pub fn activate(&mut self, start: usize, end: usize) -> bool {
        for (i, s) in self.slots.iter().enumerate() {
            if let Some(j) = s
                .occurrences
                .iter()
                .position(|o| start >= o.start && end <= o.end)
            {
                self.active = i;
                self.occurrence = j;
                return true;
            }
        }
        false
    }
    fn change(&mut self, start: usize, end: usize, length: usize, edited: (usize, usize)) {
        let delta = length as isize - (end - start) as isize;
        for (i, s) in self.slots.iter_mut().enumerate() {
            for (j, o) in s.occurrences.iter_mut().enumerate() {
                if (i, j) == edited {
                    o.end = o.end.saturating_add_signed(delta);
                } else if o.start >= end {
                    o.start = o.start.saturating_add_signed(delta);
                    o.end = o.end.saturating_add_signed(delta);
                }
            }
        }
    }
    pub fn track(&mut self, old: &str, new: &str) -> bool {
        if old == new {
            return true;
        }
        let old = old.chars().collect::<Vec<_>>();
        let new = new.chars().collect::<Vec<_>>();
        let start = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
        let suffix = old[start..]
            .iter()
            .rev()
            .zip(new[start..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let end = old.len() - suffix;
        let length = new.len() - suffix - start;
        let Some(j) = self.slots.get(self.active).and_then(|s| {
            s.occurrences
                .iter()
                .position(|o| start >= o.start && end <= o.end)
        }) else {
            return false;
        };
        self.occurrence = j;
        self.change(start, end, length, (self.active, j));
        true
    }
    pub fn commit(&mut self, text: &mut String) {
        let Some(active) = self.slots.get(self.active) else {
            return;
        };
        let Some(edited) = active.occurrences.get(self.occurrence).cloned() else {
            return;
        };
        let value = text
            .chars()
            .skip(edited.start)
            .take(edited.end - edited.start)
            .collect::<String>();
        let mut mirrors = active
            .occurrences
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != self.occurrence)
            .map(|(j, o)| (j, o.clone()))
            .collect::<Vec<_>>();
        mirrors.sort_by_key(|(_, o)| std::cmp::Reverse(o.start));
        for (j, o) in mirrors {
            let before = text.chars().take(o.start).collect::<String>();
            let after = text.chars().skip(o.end).collect::<String>();
            *text = format!("{before}{value}{after}");
            self.change(o.start, o.end, value.chars().count(), (self.active, j));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sources() -> BTreeMap<String, String> {
        BTreeMap::from([(
            "own.yaml".into(),
            r#"schema_version: '1.1'
mode: replace
namespaces:
  - key: own
    label: Own
    description: Own synthetic phrases
    choices:
      - key: meet
        aliases: [greet]
        label: Meeting
        append: space
        placeholders:
          who: {label: Person, initial_text: '@keeper', hint: Choose person}
          place: {label: Place, initial_text: pier, hint: Choose place}
        segments:
          - placeholder_ref: who
          - {text: ' greets '}
          - placeholder_ref: who
          - {text: ' at '}
          - placeholder_ref: place
"#
            .into(),
        )])
    }
    #[test]
    fn alias_completion_and_repeated_unicode_snippet_edit() {
        let c = load(&sources()).unwrap();
        let context = context("å🙂 #own.gre", 11, 11).unwrap();
        let entries = c.completions(&context);
        assert_eq!(entries.len(), 1);
        let expansion = entries[0].choice.as_ref().unwrap().expand(&BTreeMap::new());
        assert_eq!(expansion.text, "@keeper greets @keeper at pier ");
        let mut session = Session::new(expansion.slots, 0);
        let mut text = expansion.text.replace_range_for_test(0..7, "Å🙂");
        assert!(session.track("@keeper greets @keeper at pier ", &text));
        session.commit(&mut text);
        assert_eq!(text, "Å🙂 greets Å🙂 at pier ");
        session.active = 1;
        let p = session.selection().unwrap();
        assert_eq!(
            text.chars()
                .skip(p.start)
                .take(p.end - p.start)
                .collect::<String>(),
            "pier"
        );
    }
    trait Replaced {
        fn replace_range_for_test(self, range: std::ops::Range<usize>, text: &str) -> String;
    }
    impl Replaced for String {
        fn replace_range_for_test(mut self, range: std::ops::Range<usize>, text: &str) -> String {
            self.replace_range(range, text);
            self
        }
    }
    #[test]
    fn replace_precedes_sorted_merges_and_alias_collisions_fail() {
        let mut s = sources();
        s.insert("00-merge.yaml".into(),"schema_version: '1.1'\nmode: merge\nnamespaces:\n  - key: own\n    choices:\n      - key: meet\n        label: Updated\n        append: none\n        segments: [{text: Updated}]\n".into());
        assert_eq!(load(&s).unwrap().namespaces[0].choices[0].label, "Updated");
        s.insert("a.yaml".into(),"schema_version: '1.1'\nmode: merge\nnamespaces:\n  - key: own\n    choices:\n      - key: other\n        aliases: [meet]\n        label: Conflict\n        append: none\n        segments: [{text: Conflict}]\n".into());
        assert!(load(&s).is_err());
    }
    #[test]
    fn selected_text_and_embedded_hash_do_not_open_catalog() {
        assert!(context("a#own", 5, 5).is_none());
        assert!(context("#own", 0, 4).is_none());
        assert!(context("🙂 #own", 6, 6).is_some());
    }
}
