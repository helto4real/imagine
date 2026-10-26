//! Director's Prompt IntelliSense catalogs (comfyui-helto-director
//! `shared/prompt_intellisense.py` and `web/timeline/prompt_intellisense.js`):
//! the same YAML reading, validation, limits, error codes, replace/merge
//! result, completion and snippet expansion.
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
pub mod composer;
pub mod editor;
mod yaml;
use yaml::{Node, Value as Y};

pub const SCHEMA_VERSION: &str = "1.1";
const MAX_FILES: usize = 64;
const MAX_FILE_BYTES: usize = 256 * 1024;
const MAX_NAMESPACES: usize = 64;
const MAX_CHOICES_PER_CONTAINER: usize = 256;
const MAX_GROUPS_PER_CONTAINER: usize = 256;
const MAX_TOTAL_CHOICES: usize = 2048;
const MAX_TOTAL_GROUPS: usize = 2048;
const MAX_GROUP_DEPTH: usize = 16;
const MAX_SEGMENTS_PER_CHOICE: usize = 64;
const MAX_OUTPUT_LENGTH: usize = 4000;

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

/// Director's `PromptCatalogError`: a stable code and safe message, with the
/// catalog file, structural path and one-based YAML position when known.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Issue {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub line: Option<u32>,
    #[serde(default)]
    pub column: Option<u32>,
}
impl Issue {
    fn new(code: &str, message: &str, source: &str, path: &str) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            source: source.into(),
            path: path.into(),
            line: None,
            column: None,
        }
    }
    /// The file name only, as Director shows it.
    pub fn safe_source(&self) -> Option<String> {
        let name = self.source.rsplit(['/', '\\']).next().unwrap_or("");
        let name = name.replace(['\r', '\n', '\t'], " ");
        let name = name.trim();
        (!name.is_empty()).then(|| name.chars().take(255).collect())
    }
    /// The structural path when it is a plain JSON-pointer-like path.
    pub fn safe_path(&self) -> Option<&str> {
        let segment = |s: &str| {
            (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                || s.chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        };
        let rest = self.path.strip_prefix('/')?;
        rest.split('/').all(segment).then_some(self.path.as_str())
    }
}
/// Director's `public_summary`.
impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut message = self
            .message
            .lines()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();
        if message.is_empty() {
            message = "The Prompt IntelliSense catalog is invalid.".into();
        }
        if !message.ends_with(['.', '!', '?']) {
            message.push('.');
        }
        let mut location = vec![];
        if let Some(source) = self.safe_source() {
            location.push(format!("File: {source}"));
        }
        match (self.line, self.column) {
            (Some(line), Some(column)) => location.push(format!("line {line}, column {column}")),
            (Some(line), None) => location.push(format!("line {line}")),
            (None, Some(column)) => location.push(format!("column {column}")),
            (None, None) => {}
        }
        if let Some(path) = self.safe_path() {
            location.push(format!("path: {path}"));
        }
        if !location.is_empty() {
            message = format!("{message} {}.", location.join("; "));
        }
        write!(f, "{message} Code: {}.", self.code)
    }
}
impl std::error::Error for Issue {}
type Result<T> = std::result::Result<T, Issue>;
fn invalid<T>(code: &str, message: &str, source: &str, path: &str) -> Result<T> {
    Err(Issue::new(code, message, source, path))
}

/// Python's `str.isspace`: Unicode White_Space plus the separators U+001C..U+001F.
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(python_space)
}
fn length(value: &str) -> usize {
    value.chars().count()
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.split(['-', '_']).all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        })
}
fn identifier_field(value: Option<&Node>, source: &str, path: &str) -> Result<String> {
    match value.and_then(Node::as_str) {
        Some(v) if length(v) <= 80 && identifier(v) => Ok(v.into()),
        _ => invalid(
            "PROMPT_CATALOG_IDENTIFIER_INVALID",
            "Catalog identifiers must start and end with a lowercase letter or number and may contain only lowercase letters, numbers, hyphens, and underscores; spaces, dots, uppercase letters, and leading or trailing separators are not allowed.",
            source,
            path,
        ),
    }
}
fn bounded_string(
    value: Option<&Node>,
    maximum: usize,
    source: &str,
    path: &str,
) -> Result<String> {
    match value.and_then(Node::as_str) {
        Some(v)
            if !python_strip(v).is_empty()
                && v == python_strip(v)
                && length(v) <= maximum
                && !v.contains(['\r', '\n']) =>
        {
            Ok(v.into())
        }
        _ => invalid(
            "PROMPT_CATALOG_STRING_INVALID",
            "Catalog text must be one trimmed, bounded line.",
            source,
            path,
        ),
    }
}
fn prompt_fragment(
    value: Option<&Node>,
    maximum: usize,
    source: &str,
    path: &str,
) -> Result<String> {
    match value.and_then(Node::as_str) {
        Some(v) if !v.is_empty() && length(v) <= maximum && !v.contains(['\r', '\n', '#']) => {
            Ok(v.into())
        }
        _ => invalid(
            "PROMPT_CATALOG_PROMPT_FRAGMENT_INVALID",
            "Inserted prompt fragments must be bounded ordinary one-line text without hash directives.",
            source,
            path,
        ),
    }
}
/// A mapping's string keys, in order; `None` when it is not a mapping.
type Mapping<'a> = Vec<(&'a str, &'a Node)>;
fn expect_mapping<'a>(value: Option<&'a Node>, source: &str, path: &str) -> Result<Mapping<'a>> {
    let Some(Node {
        value: Y::Map(pairs),
        ..
    }) = value
    else {
        return invalid(
            "PROMPT_CATALOG_EXPECTED_MAPPING",
            "Catalog structure requires a mapping.",
            source,
            path,
        );
    };
    pairs
        .iter()
        .map(|(k, v)| k.as_str().map(|k| (k, v)))
        .collect::<Option<Vec<_>>>()
        .map_or_else(
            || {
                invalid(
                    "PROMPT_CATALOG_NON_STRING_KEY",
                    "Catalog mapping keys must be strings.",
                    source,
                    path,
                )
            },
            Ok,
        )
}
fn get<'a>(mapping: &Mapping<'a>, key: &str) -> Option<&'a Node> {
    mapping.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}
fn has(mapping: &Mapping<'_>, key: &str) -> bool {
    mapping.iter().any(|(k, _)| *k == key)
}
fn expect_list<'a>(
    value: Option<&'a Node>,
    source: &str,
    path: &str,
    minimum: usize,
    maximum: usize,
) -> Result<&'a [Node]> {
    match value {
        Some(Node {
            value: Y::List(items),
            ..
        }) if (minimum..=maximum).contains(&items.len()) => Ok(items),
        _ => invalid(
            "PROMPT_CATALOG_LIST_BOUNDS",
            "Catalog list length is outside the supported bounds.",
            source,
            path,
        ),
    }
}
fn expect_only_keys(
    mapping: &Mapping<'_>,
    allowed: &[&str],
    source: &str,
    path: &str,
) -> Result<()> {
    if mapping.iter().any(|(k, _)| !allowed.contains(k)) {
        return invalid(
            "PROMPT_CATALOG_UNKNOWN_FIELD",
            "Catalog structure contains an unknown field.",
            source,
            path,
        );
    }
    Ok(())
}
fn json_pointer_segment(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

/// One file's catalog before merging: metadata is optional until resolved.
#[derive(Clone, Debug)]
struct Draft {
    key: String,
    label: Option<String>,
    description: Option<String>,
    choices: Vec<Choice>,
    groups: Vec<Draft>,
}
struct Parsed {
    mode: String,
    namespaces: Vec<Draft>,
}
fn validate_document(value: Option<&Node>, source: &str) -> Result<Parsed> {
    let root = expect_mapping(value, source, "/")?;
    expect_only_keys(
        &root,
        &["schema_version", "mode", "namespaces"],
        source,
        "/",
    )?;
    let schema_version = match get(&root, "schema_version").and_then(Node::as_str) {
        Some(v @ ("1.0" | "1.1")) => v,
        _ => {
            return invalid(
                "PROMPT_CATALOG_SCHEMA_VERSION",
                "Unsupported Prompt IntelliSense schema version.",
                source,
                "/schema_version",
            );
        }
    };
    let mode = match get(&root, "mode").and_then(Node::as_str) {
        Some(v @ ("replace" | "merge")) => v,
        _ => {
            return invalid(
                "PROMPT_CATALOG_MODE_INVALID",
                "Prompt IntelliSense mode must be replace or merge.",
                source,
                "/mode",
            );
        }
    };
    let raw = expect_list(
        get(&root, "namespaces"),
        source,
        "/namespaces",
        1,
        MAX_NAMESPACES,
    )?;
    let mut namespaces = vec![];
    let mut seen = std::collections::HashSet::new();
    for (index, raw) in raw.iter().enumerate() {
        let path = format!("/namespaces/{index}");
        let namespace = expect_mapping(Some(raw), source, &path)?;
        expect_only_keys(
            &namespace,
            &["key", "label", "description", "choices", "groups"],
            source,
            &path,
        )?;
        let key = identifier_field(get(&namespace, "key"), source, &format!("{path}/key"))?;
        if !seen.insert(key.clone()) {
            return invalid(
                "PROMPT_CATALOG_DUPLICATE_NAMESPACE",
                "A namespace key is duplicated in one catalog file.",
                source,
                &format!("{path}/key"),
            );
        }
        let label = has(&namespace, "label")
            .then(|| {
                bounded_string(
                    get(&namespace, "label"),
                    100,
                    source,
                    &format!("{path}/label"),
                )
            })
            .transpose()?;
        let description = has(&namespace, "description")
            .then(|| {
                bounded_string(
                    get(&namespace, "description"),
                    300,
                    source,
                    &format!("{path}/description"),
                )
            })
            .transpose()?;
        if mode == "replace" && (label.is_none() || description.is_none()) {
            return invalid(
                "PROMPT_CATALOG_NAMESPACE_METADATA_REQUIRED",
                "Replace-mode namespaces require label and description.",
                source,
                &path,
            );
        }
        let (choices, groups) =
            container_content(&namespace, source, &path, mode, 0, schema_version)?;
        namespaces.push(Draft {
            key,
            label,
            description,
            choices,
            groups,
        });
    }
    Ok(Parsed {
        mode: mode.into(),
        namespaces,
    })
}
fn container_content(
    container: &Mapping<'_>,
    source: &str,
    path: &str,
    mode: &str,
    depth: usize,
    schema_version: &str,
) -> Result<(Vec<Choice>, Vec<Draft>)> {
    let mut choices = vec![];
    let mut groups = vec![];
    if has(container, "choices") {
        let at = format!("{path}/choices");
        let raw = expect_list(
            get(container, "choices"),
            source,
            &at,
            1,
            MAX_CHOICES_PER_CONTAINER,
        )?;
        choices = validate_choices(raw, source, &at, schema_version)?;
    }
    if has(container, "groups") {
        let at = format!("{path}/groups");
        let raw = expect_list(
            get(container, "groups"),
            source,
            &at,
            1,
            MAX_GROUPS_PER_CONTAINER,
        )?;
        groups = validate_groups(raw, source, &at, mode, depth + 1, schema_version)?;
    }
    if !has(container, "choices") && !has(container, "groups") {
        return invalid(
            "PROMPT_CATALOG_CONTAINER_EMPTY",
            "Namespaces and groups require choices, groups, or both.",
            source,
            path,
        );
    }
    Ok((choices, groups))
}
fn validate_groups(
    raw: &[Node],
    source: &str,
    base: &str,
    mode: &str,
    depth: usize,
    schema_version: &str,
) -> Result<Vec<Draft>> {
    if depth > MAX_GROUP_DEPTH {
        return invalid(
            "PROMPT_CATALOG_GROUP_DEPTH_EXCEEDED",
            "Prompt IntelliSense group nesting exceeds the supported depth.",
            source,
            base,
        );
    }
    let mut groups = vec![];
    let mut seen = std::collections::HashSet::new();
    for (index, raw) in raw.iter().enumerate() {
        let path = format!("{base}/{index}");
        let group = expect_mapping(Some(raw), source, &path)?;
        expect_only_keys(
            &group,
            &["key", "label", "description", "choices", "groups"],
            source,
            &path,
        )?;
        let key = identifier_field(get(&group, "key"), source, &format!("{path}/key"))?;
        if !seen.insert(key.clone()) {
            return invalid(
                "PROMPT_CATALOG_DUPLICATE_GROUP",
                "A group key is duplicated in one catalog container.",
                source,
                &format!("{path}/key"),
            );
        }
        let label = has(&group, "label")
            .then(|| bounded_string(get(&group, "label"), 100, source, &format!("{path}/label")))
            .transpose()?;
        let description = has(&group, "description")
            .then(|| {
                bounded_string(
                    get(&group, "description"),
                    300,
                    source,
                    &format!("{path}/description"),
                )
            })
            .transpose()?;
        if mode == "replace" && label.is_none() {
            return invalid(
                "PROMPT_CATALOG_GROUP_METADATA_REQUIRED",
                "Replace-mode groups require a label.",
                source,
                &path,
            );
        }
        let (choices, nested) =
            container_content(&group, source, &path, mode, depth, schema_version)?;
        groups.push(Draft {
            key,
            label,
            description,
            choices,
            groups: nested,
        });
    }
    Ok(groups)
}
fn validate_choices(
    raw: &[Node],
    source: &str,
    base: &str,
    schema_version: &str,
) -> Result<Vec<Choice>> {
    let mut choices = vec![];
    let mut seen = std::collections::HashSet::new();
    for (index, raw) in raw.iter().enumerate() {
        let path = format!("{base}/{index}");
        let choice = expect_mapping(Some(raw), source, &path)?;
        let mut allowed = vec!["key", "aliases", "label", "append", "segments"];
        if schema_version == "1.1" {
            allowed.extend(["description", "placeholders"]);
        }
        expect_only_keys(&choice, &allowed, source, &path)?;
        let key = identifier_field(get(&choice, "key"), source, &format!("{path}/key"))?;
        if !seen.insert(key.clone()) {
            return invalid(
                "PROMPT_CATALOG_DUPLICATE_CHOICE",
                "A choice key is duplicated in one catalog container.",
                source,
                &format!("{path}/key"),
            );
        }
        let empty = Node {
            value: Y::List(vec![]),
            line: 0,
            column: 0,
            scalar: None,
        };
        let raw_aliases = expect_list(
            if has(&choice, "aliases") {
                get(&choice, "aliases")
            } else {
                Some(&empty)
            },
            source,
            &format!("{path}/aliases"),
            0,
            20,
        )?;
        let mut aliases = vec![];
        for (i, alias) in raw_aliases.iter().enumerate() {
            let alias = identifier_field(Some(alias), source, &format!("{path}/aliases/{i}"))?;
            if aliases.contains(&alias) {
                return invalid(
                    "PROMPT_CATALOG_DUPLICATE_ALIAS",
                    "A choice repeats an alias.",
                    source,
                    &format!("{path}/aliases/{i}"),
                );
            }
            aliases.push(alias);
        }
        let label = bounded_string(get(&choice, "label"), 100, source, &format!("{path}/label"))?;
        let description = has(&choice, "description")
            .then(|| {
                bounded_string(
                    get(&choice, "description"),
                    300,
                    source,
                    &format!("{path}/description"),
                )
            })
            .transpose()?;
        let append = match get(&choice, "append").and_then(Node::as_str) {
            Some(v @ ("space" | "none")) => v.to_owned(),
            _ => {
                return invalid(
                    "PROMPT_CATALOG_APPEND_INVALID",
                    "Choice append must be space or none.",
                    source,
                    &format!("{path}/append"),
                );
            }
        };
        let raw_segments = expect_list(
            get(&choice, "segments"),
            source,
            &format!("{path}/segments"),
            1,
            MAX_SEGMENTS_PER_CHOICE,
        )?;
        let definitions = placeholder_definitions(&choice, source, &path, schema_version)?;
        let mut segments = vec![];
        let mut ids = std::collections::HashSet::new();
        let mut output = usize::from(append == "space");
        for (i, raw) in raw_segments.iter().enumerate() {
            let at = format!("{path}/segments/{i}");
            let segment = expect_mapping(Some(raw), source, &at)?;
            let fields: &[&str] = if schema_version == "1.0" {
                &["text", "placeholder"]
            } else {
                &["text", "placeholder_ref"]
            };
            expect_only_keys(&segment, fields, source, &at)?;
            let keys: Vec<&str> = segment.iter().map(|(k, _)| *k).collect();
            if keys == ["text"] {
                let text =
                    prompt_fragment(get(&segment, "text"), 1000, source, &format!("{at}/text"))?;
                output += length(&text);
                segments.push(Segment::Text { text });
                continue;
            }
            if schema_version == "1.1" {
                if keys != ["placeholder_ref"] {
                    return invalid(
                        "PROMPT_CATALOG_SEGMENT_INVALID",
                        "Each schema 1.1 segment must contain exactly text or placeholder_ref.",
                        source,
                        &at,
                    );
                }
                let id = identifier_field(
                    get(&segment, "placeholder_ref"),
                    source,
                    &format!("{at}/placeholder_ref"),
                )?;
                let Some(placeholder) = definitions.get(&id) else {
                    return invalid(
                        "PROMPT_CATALOG_PLACEHOLDER_REFERENCE_UNKNOWN",
                        "Every placeholder_ref must name a placeholder defined by the same choice.",
                        source,
                        &format!("{at}/placeholder_ref"),
                    );
                };
                output += length(&placeholder.initial_text);
                segments.push(Segment::Placeholder {
                    placeholder: placeholder.clone(),
                });
                continue;
            }
            if keys != ["placeholder"] {
                return invalid(
                    "PROMPT_CATALOG_SEGMENT_INVALID",
                    "Each segment must contain exactly text or placeholder.",
                    source,
                    &at,
                );
            }
            let at = format!("{at}/placeholder");
            let placeholder = expect_mapping(get(&segment, "placeholder"), source, &at)?;
            expect_only_keys(
                &placeholder,
                &["id", "label", "initial_text", "hint"],
                source,
                &at,
            )?;
            if !["id", "label", "initial_text", "hint"]
                .iter()
                .all(|k| has(&placeholder, k))
            {
                return invalid(
                    "PROMPT_CATALOG_PLACEHOLDER_FIELDS_REQUIRED",
                    "Placeholders require id, label, initial_text, and hint.",
                    source,
                    &at,
                );
            }
            let id = identifier_field(get(&placeholder, "id"), source, &format!("{at}/id"))?;
            if !ids.insert(id.clone()) {
                return invalid(
                    "PROMPT_CATALOG_DUPLICATE_PLACEHOLDER",
                    "Placeholder ids must be unique inside a choice.",
                    source,
                    &format!("{at}/id"),
                );
            }
            let initial_text = prompt_fragment(
                get(&placeholder, "initial_text"),
                500,
                source,
                &format!("{at}/initial_text"),
            )?;
            let label = bounded_string(
                get(&placeholder, "label"),
                100,
                source,
                &format!("{at}/label"),
            )?;
            let hint = bounded_string(
                get(&placeholder, "hint"),
                300,
                source,
                &format!("{at}/hint"),
            )?;
            output += length(&initial_text);
            segments.push(Segment::Placeholder {
                placeholder: Placeholder {
                    id,
                    label,
                    initial_text,
                    hint,
                },
            });
        }
        if output > MAX_OUTPUT_LENGTH {
            return invalid(
                "PROMPT_CATALOG_OUTPUT_TOO_LONG",
                "A choice expands beyond the output length limit.",
                source,
                &path,
            );
        }
        choices.push(Choice {
            key,
            aliases,
            label,
            description: description.unwrap_or_default(),
            append,
            segments,
        });
    }
    Ok(choices)
}
fn placeholder_definitions(
    choice: &Mapping<'_>,
    source: &str,
    path: &str,
    schema_version: &str,
) -> Result<BTreeMap<String, Placeholder>> {
    let mut definitions = BTreeMap::new();
    if schema_version == "1.0" || !has(choice, "placeholders") {
        return Ok(definitions);
    }
    let base = format!("{path}/placeholders");
    let raw = match get(choice, "placeholders") {
        // Mapping keys must be strings before the field check.
        Some(
            node @ Node {
                value: Y::Map(pairs),
                ..
            },
        ) => {
            let mapping = expect_mapping(Some(node), source, &base)?;
            if pairs.len() > MAX_SEGMENTS_PER_CHOICE {
                return invalid(
                    "PROMPT_CATALOG_LIST_BOUNDS",
                    "A choice contains too many placeholder definitions.",
                    source,
                    &base,
                );
            }
            mapping
        }
        other => expect_mapping(other, source, &base)?,
    };
    for (raw_id, raw_placeholder) in raw {
        let at = format!("{base}/{}", json_pointer_segment(raw_id));
        let id_node = Node {
            value: Y::Str(raw_id.into()),
            line: 0,
            column: 0,
            scalar: None,
        };
        let id = identifier_field(Some(&id_node), source, &at)?;
        let placeholder = expect_mapping(Some(raw_placeholder), source, &at)?;
        expect_only_keys(
            &placeholder,
            &["label", "initial_text", "hint"],
            source,
            &at,
        )?;
        if !["label", "initial_text", "hint"]
            .iter()
            .all(|k| has(&placeholder, k))
        {
            return invalid(
                "PROMPT_CATALOG_PLACEHOLDER_FIELDS_REQUIRED",
                "Schema 1.1 placeholders require label, initial_text, and hint.",
                source,
                &at,
            );
        }
        let label = bounded_string(
            get(&placeholder, "label"),
            100,
            source,
            &format!("{at}/label"),
        )?;
        let initial_text = prompt_fragment(
            get(&placeholder, "initial_text"),
            500,
            source,
            &format!("{at}/initial_text"),
        )?;
        let hint = bounded_string(
            get(&placeholder, "hint"),
            300,
            source,
            &format!("{at}/hint"),
        )?;
        definitions.insert(
            id.clone(),
            Placeholder {
                id,
                label,
                initial_text,
                hint,
            },
        );
    }
    Ok(definitions)
}

/// One-based positions of composed nodes by structural path.
fn locations(node: &Node, path: &str, out: &mut BTreeMap<String, (u32, u32)>) {
    out.insert(path.into(), (node.line as u32, node.column as u32));
    let prefix = if path == "/" { "" } else { path };
    match &node.value {
        Y::Map(pairs) => {
            for (key, value) in pairs {
                if let Some(text) = &key.scalar {
                    locations(
                        value,
                        &format!("{prefix}/{}", json_pointer_segment(text)),
                        out,
                    );
                }
            }
        }
        Y::List(items) => {
            for (i, item) in items.iter().enumerate() {
                locations(item, &format!("{prefix}/{i}"), out);
            }
        }
        _ => {}
    }
}
fn attach_location(issue: &mut Issue, locations: &BTreeMap<String, (u32, u32)>) {
    if issue.line.is_some() || issue.path.is_empty() {
        return;
    }
    let mut candidate = issue.path.clone();
    loop {
        if let Some(&(line, column)) = locations.get(&candidate) {
            issue.line = Some(line);
            issue.column = Some(column);
            return;
        }
        if candidate == "/" {
            return;
        }
        candidate = match candidate.rsplit_once('/') {
            Some(("", _)) | None => "/".into(),
            Some((head, _)) => head.into(),
        };
    }
}
fn parse_text(text: &str, source: &str) -> Result<Parsed> {
    if text.len() > MAX_FILE_BYTES {
        return invalid(
            "PROMPT_CATALOG_FILE_TOO_LARGE",
            "A Prompt IntelliSense catalog file exceeds the size limit.",
            source,
            "",
        );
    }
    if python_strip(text).is_empty() {
        return invalid(
            "PROMPT_CATALOG_EMPTY_FILE",
            "A Prompt IntelliSense catalog file is empty.",
            source,
            "",
        );
    }
    let documents = yaml::load(text).map_err(|e| Issue {
        code: e.code.into(),
        message: if e.code == "PROMPT_CATALOG_YAML_FEATURE_UNSUPPORTED" {
            "YAML aliases, anchors, and explicit tags are not supported.".into()
        } else {
            "A Prompt IntelliSense catalog file contains malformed YAML.".into()
        },
        source: source.into(),
        path: String::new(),
        line: e.line,
        column: e.column,
    })?;
    if documents.len() != 1 {
        return invalid(
            "PROMPT_CATALOG_MULTIPLE_DOCUMENTS",
            "Each Prompt IntelliSense YAML file must contain exactly one document.",
            source,
            "",
        );
    }
    validate_document(documents.first(), source).map_err(|mut issue| {
        let mut map = BTreeMap::new();
        locations(&documents[0], "/", &mut map);
        attach_location(&mut issue, &map);
        issue
    })
}
fn merge_container(current: &mut Draft, incoming: Draft) {
    if incoming.label.is_some() {
        current.label = incoming.label;
    }
    if incoming.description.is_some() {
        current.description = incoming.description;
    }
    for choice in incoming.choices {
        match current.choices.iter_mut().find(|c| c.key == choice.key) {
            Some(old) => *old = choice,
            None => current.choices.push(choice),
        }
    }
    merge_named(&mut current.groups, incoming.groups);
}
fn merge_named(target: &mut Vec<Draft>, incoming: Vec<Draft>) {
    for container in incoming {
        match target.iter_mut().find(|c| c.key == container.key) {
            Some(current) => merge_container(current, container),
            None => target.push(container),
        }
    }
}
fn resolved_container(
    container: Draft,
    path: &str,
    depth: usize,
) -> Result<(Container, usize, usize)> {
    let empty = |code: &str, message: &str| invalid(code, message, "", path);
    if container.choices.is_empty() && container.groups.is_empty() {
        return empty(
            "PROMPT_CATALOG_CONTAINER_EMPTY",
            "Every resolved namespace and group requires choices, groups, or both.",
        );
    }
    if container.choices.len() > MAX_CHOICES_PER_CONTAINER {
        return empty(
            "PROMPT_CATALOG_TOO_MANY_CHOICES_IN_CONTAINER",
            "A Prompt IntelliSense namespace or group contains too many direct choices.",
        );
    }
    if container.groups.len() > MAX_GROUPS_PER_CONTAINER {
        return empty(
            "PROMPT_CATALOG_TOO_MANY_GROUPS_IN_CONTAINER",
            "A Prompt IntelliSense namespace or group contains too many direct groups.",
        );
    }
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    for (i, choice) in container.choices.iter().enumerate() {
        let owner = format!("choice:{}", choice.key);
        for id in std::iter::once(&choice.key).chain(&choice.aliases) {
            if owners.get(id).is_some_and(|o| *o != owner) {
                return invalid(
                    "PROMPT_CATALOG_ALIAS_COLLISION",
                    "Group, choice, and alias identifiers must resolve to exactly one item per level.",
                    "",
                    &format!("{path}/choices/{i}"),
                );
            }
            owners.insert(id.clone(), owner.clone());
        }
    }
    let mut choices_total = container.choices.len();
    let mut groups_total = container.groups.len();
    let mut groups = vec![];
    for (i, group) in container.groups.into_iter().enumerate() {
        let at = format!("{path}/groups/{i}");
        if depth + 1 > MAX_GROUP_DEPTH {
            return invalid(
                "PROMPT_CATALOG_GROUP_DEPTH_EXCEEDED",
                "Prompt IntelliSense group nesting exceeds the supported depth.",
                "",
                &at,
            );
        }
        if group.label.as_deref().is_none_or(str::is_empty) {
            return invalid(
                "PROMPT_CATALOG_GROUP_METADATA_REQUIRED",
                "Every resolved group requires a label.",
                "",
                &at,
            );
        }
        if owners.contains_key(&group.key) {
            return invalid(
                "PROMPT_CATALOG_GROUP_IDENTIFIER_COLLISION",
                "Group, choice, and alias identifiers must resolve to exactly one item per level.",
                "",
                &at,
            );
        }
        owners.insert(group.key.clone(), format!("group:{}", group.key));
        let (resolved, c, g) = resolved_container(group, &at, depth + 1)?;
        choices_total += c;
        groups_total += g;
        groups.push(resolved);
    }
    Ok((
        Container {
            key: container.key,
            label: container.label.unwrap_or_default(),
            description: container.description.unwrap_or_default(),
            choices: container.choices,
            groups,
        },
        choices_total,
        groups_total,
    ))
}
fn merge_parsed(parsed: Vec<Parsed>) -> Result<Catalog> {
    if parsed.iter().filter(|p| p.mode == "replace").count() > 1 {
        return invalid(
            "PROMPT_CATALOG_MULTIPLE_REPLACE",
            "Only one Prompt IntelliSense catalog file may use replace mode.",
            "",
            "",
        );
    }
    let (replace, merge): (Vec<_>, Vec<_>) = parsed.into_iter().partition(|p| p.mode == "replace");
    let mut merged = vec![];
    for p in replace.into_iter().chain(merge) {
        merge_named(&mut merged, p.namespaces);
    }
    if merged.is_empty() {
        return invalid(
            "PROMPT_CATALOG_EMPTY_RESULT",
            "Prompt IntelliSense catalog merging produced no namespaces.",
            "",
            "",
        );
    }
    if merged.len() > MAX_NAMESPACES {
        return invalid(
            "PROMPT_CATALOG_TOO_MANY_NAMESPACES",
            "Prompt IntelliSense catalog merging produced too many namespaces.",
            "",
            "",
        );
    }
    let (mut choices, mut groups) = (0, 0);
    let mut namespaces = vec![];
    for (i, namespace) in merged.into_iter().enumerate() {
        let path = format!("/namespaces/{i}");
        if namespace.label.as_deref().is_none_or(str::is_empty)
            || namespace.description.as_deref().is_none_or(str::is_empty)
        {
            return invalid(
                "PROMPT_CATALOG_NAMESPACE_METADATA_REQUIRED",
                "Every resolved namespace requires label and description.",
                "",
                &path,
            );
        }
        let (resolved, c, g) = resolved_container(namespace, &path, 0)?;
        choices += c;
        groups += g;
        namespaces.push(resolved);
    }
    if choices > MAX_TOTAL_CHOICES {
        return invalid(
            "PROMPT_CATALOG_TOO_MANY_CHOICES",
            "Prompt IntelliSense catalog merging produced too many choices.",
            "",
            "",
        );
    }
    if groups > MAX_TOTAL_GROUPS {
        return invalid(
            "PROMPT_CATALOG_TOO_MANY_GROUPS",
            "Prompt IntelliSense catalog merging produced too many groups.",
            "",
            "",
        );
    }
    Ok(Catalog {
        schema_version: SCHEMA_VERSION.into(),
        namespaces,
    })
}
/// Director's `validate_prompt_intellisense_catalog_sources`: files by name,
/// the replace file first, then merges in filename order.
pub fn load(sources: &BTreeMap<String, String>) -> Result<Catalog> {
    if sources.is_empty() {
        return invalid(
            "PROMPT_CATALOG_NO_FILES",
            "Prompt IntelliSense requires at least one YAML catalog file.",
            "",
            "",
        );
    }
    if sources.len() > MAX_FILES {
        return invalid(
            "PROMPT_CATALOG_TOO_MANY_FILES",
            "Prompt IntelliSense has too many YAML catalog files.",
            "",
            "",
        );
    }
    let parsed = sources
        .iter()
        .map(|(name, text)| parse_text(text, name))
        .collect::<Result<Vec<_>>>()?;
    merge_parsed(parsed)
}
/// One file on its own, as Director validates an editor draft's syntax.
pub fn check(text: &str, source: &str) -> Result<()> {
    parse_text(text, source).map(|_| ())
}
/// The catalog files of a folder, by Director's rules: every `.yaml`/`.yml`
/// entry directly inside, by name; at most 64, each a regular file in that
/// folder. Symlinks and other entries fail the whole folder.
fn folder_files(folder: &Path) -> Result<Vec<(String, PathBuf)>> {
    let missing = || {
        invalid(
            "PROMPT_CATALOG_DIRECTORY_MISSING",
            "Prompt IntelliSense configuration directory is unavailable.",
            "",
            "",
        )
    };
    let Ok(meta) = std::fs::symlink_metadata(folder) else {
        return missing();
    };
    if !meta.is_dir() {
        return missing();
    }
    let Ok(entries) = std::fs::read_dir(folder) else {
        return missing();
    };
    let mut files: Vec<(String, PathBuf)> = vec![];
    for entry in entries.flatten() {
        let path = entry.path();
        let yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"));
        if yaml {
            files.push((entry.file_name().to_string_lossy().into_owned(), path));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    if files.is_empty() {
        return invalid(
            "PROMPT_CATALOG_NO_FILES",
            "Prompt IntelliSense requires at least one YAML catalog file.",
            "",
            "",
        );
    }
    if files.len() > MAX_FILES {
        return invalid(
            "PROMPT_CATALOG_TOO_MANY_FILES",
            "Prompt IntelliSense has too many YAML catalog files.",
            "",
            "",
        );
    }
    let resolved = std::fs::canonicalize(folder).ok();
    for (name, path) in &files {
        let regular = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
        let inside = std::fs::canonicalize(path)
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            == resolved;
        if !regular || !inside {
            return invalid(
                "PROMPT_CATALOG_UNSAFE_FILE",
                "Prompt IntelliSense catalog files must be regular files inside the configuration directory.",
                name,
                "",
            );
        }
    }
    Ok(files)
}
/// One catalog file: at most 256 KiB, UTF-8, read with Python's universal
/// newlines (CRLF and CR become LF).
fn read_file(name: &str, path: &Path) -> Result<String> {
    let unreadable = || {
        Issue::new(
            "PROMPT_CATALOG_FILE_UNREADABLE",
            "A Prompt IntelliSense catalog file could not be read as UTF-8.",
            name,
            "",
        )
    };
    let size = std::fs::metadata(path).map_err(|_| unreadable())?.len();
    if size > MAX_FILE_BYTES as u64 {
        return invalid(
            "PROMPT_CATALOG_FILE_TOO_LARGE",
            "A Prompt IntelliSense catalog file exceeds the size limit.",
            name,
            "",
        );
    }
    let text = std::fs::read(path)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .ok_or_else(unreadable)?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}
/// A folder's catalog files for editing, without parsing them.
pub fn read_folder(folder: &Path) -> Result<BTreeMap<String, String>> {
    folder_files(folder)?
        .into_iter()
        .map(|(name, path)| read_file(&name, &path).map(|text| (name, text)))
        .collect()
}
/// Director's catalog load: the folder's files are checked, then each is read
/// and parsed in name order, then merged.
pub fn load_folder(folder: &Path) -> Result<(BTreeMap<String, String>, Catalog)> {
    let mut sources = BTreeMap::new();
    let mut parsed = vec![];
    for (name, path) in folder_files(folder)? {
        let text = read_file(&name, &path)?;
        parsed.push(parse_text(&text, &name)?);
        sources.insert(name, text);
    }
    Ok((sources, merge_parsed(parsed)?))
}

/// Sorted, merged ranges of a Unicode class written in regex syntax.
fn class(pattern: &str) -> Vec<(char, char)> {
    use regex_syntax::hir::{Class, HirKind};
    let hir = regex_syntax::parse(pattern).expect("valid class");
    match hir.kind() {
        HirKind::Class(Class::Unicode(class)) => class
            .ranges()
            .iter()
            .map(|r| (r.start(), r.end()))
            .collect(),
        _ => unreachable!("character class"),
    }
}
fn in_class(ranges: &[(char, char)], c: char) -> bool {
    ranges
        .binary_search_by(|&(start, end)| {
            if end < c {
                std::cmp::Ordering::Less
            } else if start > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}
/// JavaScript's `[\p{L}\p{N}]`.
fn letter_or_number(c: char) -> bool {
    static RANGES: std::sync::OnceLock<Vec<(char, char)>> = std::sync::OnceLock::new();
    in_class(RANGES.get_or_init(|| class(r"[\p{L}\p{N}]")), c)
}
/// JavaScript's `\s` with the `u` flag.
fn js_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
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
/// Director's `promptCatalogCompletionContext`, in character positions:
/// `(^|[^\p{L}\p{N}_#])(#[\p{L}\p{N}_-]*(?:\.[\p{L}\p{N}_-]*)*)$` before the caret.
pub fn context(text: &str, start: usize, end: usize) -> Option<Context> {
    if start != end {
        return None;
    }
    let chars = text.chars().take(start).collect::<Vec<_>>();
    let at = chars.iter().rposition(|c| *c == '#')?;
    if at > 0 && (letter_or_number(chars[at - 1]) || matches!(chars[at - 1], '_' | '#')) {
        return None;
    }
    let token = chars[at + 1..].iter().collect::<String>();
    if !token
        .chars()
        .all(|c| letter_or_number(c) || matches!(c, '_' | '-' | '.'))
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
/// Director's `normalizeSearch`.
fn search(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| !js_space(*c) && !matches!(c, '_' | '-'))
        .collect()
}
impl Catalog {
    /// Director's `filterPromptCatalogCompletions`: at most 50, sorted by name.
    pub fn completions(&self, context: &Context) -> Vec<Completion> {
        let needle = search(&context.namespace);
        let namespaces = self.namespaces.iter().filter(|n| {
            search(&n.key).starts_with(&needle) || search(&n.label).starts_with(&needle)
        });
        // (completion, Director's item id for the tie-break)
        let mut out: Vec<(Completion, String)> = vec![];
        for n in namespaces {
            if context.namespace_stage {
                out.push((
                    Completion {
                        label: n.label.clone(),
                        detail: format!("#{}. — {}", n.key, n.description),
                        path: vec![n.key.clone()],
                        choice: None,
                    },
                    format!("namespace-{}", n.key),
                ));
                continue;
            }
            let mut parent = n;
            let mut valid = true;
            for key in &context.groups {
                match parent
                    .groups
                    .iter()
                    .find(|g| !key.is_empty() && &g.key == key)
                {
                    Some(g) => parent = g,
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            let query = search(&context.query);
            for g in &parent.groups {
                if !query.is_empty()
                    && ![&g.key, &g.label, &g.description]
                        .iter()
                        .any(|s| search(s).starts_with(&query))
                {
                    continue;
                }
                let mut group_path = context.groups.clone();
                group_path.push(g.key.clone());
                let description = if g.description.is_empty() {
                    "Prompt category"
                } else {
                    &g.description
                };
                out.push((
                    Completion {
                        label: g.label.clone(),
                        detail: format!("#{}.{}. — {description}", n.key, group_path.join(".")),
                        path: std::iter::once(n.key.clone())
                            .chain(group_path.clone())
                            .collect(),
                        choice: None,
                    },
                    format!("group-{}-{}", n.key, group_path.join("-")),
                ));
            }
            for c in &parent.choices {
                if !query.is_empty()
                    && !std::iter::once(&c.key)
                        .chain(std::iter::once(&c.label))
                        .chain(&c.aliases)
                        .any(|s| search(s).starts_with(&query))
                {
                    continue;
                }
                let mut choice_path = context.groups.clone();
                choice_path.push(c.key.clone());
                out.push((
                    Completion {
                        label: c.label.clone(),
                        detail: format!("#{}.{} — {}", n.key, choice_path.join("."), c.preview()),
                        path: std::iter::once(n.key.clone())
                            .chain(choice_path.clone())
                            .collect(),
                        choice: Some(c.clone()),
                    },
                    format!("choice-{}-{}", n.key, choice_path.join("-")),
                ));
            }
        }
        out.sort_by(|(a, x), (b, y)| {
            let name = |c: &Completion| {
                if c.label.is_empty() {
                    c.path[0].clone()
                } else {
                    c.label.clone()
                }
            };
            collate(&name(a), &name(b)).then_with(|| collate(x, y))
        });
        out.truncate(50);
        out.into_iter().map(|(c, _)| c).collect()
    }
}
/// Director sorts with `Intl.Collator(undefined, {sensitivity: "base",
/// numeric: true})`, whose order follows the browser's locale. This follows
/// its root order: case and accents are ignored, digit runs compare as
/// numbers, and punctuation precedes digits, which precede letters.
fn collate(a: &str, b: &str) -> std::cmp::Ordering {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum Key {
        Other(char),
        Number(usize, String),
        Letter(char),
    }
    fn keys(text: &str) -> Vec<Key> {
        static MARKS: std::sync::OnceLock<Vec<(char, char)>> = std::sync::OnceLock::new();
        let marks = MARKS.get_or_init(|| class(r"\p{M}"));
        let decomposed = icu_normalizer::DecomposingNormalizerBorrowed::new_nfd().normalize(text);
        let base = decomposed
            .chars()
            .filter(|c| !in_class(marks, *c))
            .collect::<String>()
            .to_lowercase();
        let mut out = vec![];
        let mut chars = base.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_ascii_digit() {
                let mut digits = String::from(c);
                while let Some(d) = chars.next_if(char::is_ascii_digit) {
                    digits.push(d);
                }
                let digits = digits.trim_start_matches('0').to_owned();
                out.push(Key::Number(digits.len(), digits));
            } else if letter_or_number(c) {
                out.push(Key::Letter(c));
            } else {
                out.push(Key::Other(c));
            }
        }
        out
    }
    keys(a).cmp(&keys(b))
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
/// Director's `promptPlaceholderLinkKey`: placeholders whose whole initial
/// text is one `@identity` are one logical placeholder.
fn link_key(p: &Placeholder) -> String {
    let identity = p.initial_text.strip_prefix('@').filter(|s| {
        s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    });
    identity.map_or_else(
        || format!("id:{}", p.id),
        |s| format!("identity:{}", s.to_lowercase()),
    )
}
impl Choice {
    /// Director's `promptChoicePreview`: placeholders show as `[label]`.
    pub fn preview(&self) -> String {
        let mut text = self
            .segments
            .iter()
            .map(|s| match s {
                Segment::Text { text } => text.clone(),
                Segment::Placeholder { placeholder: p } => format!(
                    "[{}]",
                    if !p.label.is_empty() {
                        &p.label
                    } else if !p.id.is_empty() {
                        &p.id
                    } else {
                        "text"
                    }
                ),
            })
            .collect::<String>();
        if self.append == "space" {
            text.push(' ');
        }
        text
    }
    /// Director's `expandPromptCatalogChoice`, with optional slot values.
    pub fn expand(&self, values: &BTreeMap<String, String>) -> Expansion {
        let mut text = String::new();
        let mut slots: Vec<Slot> = vec![];
        let mut keys = vec![];
        for segment in &self.segments {
            match segment {
                Segment::Text { text: part } => text.push_str(part),
                Segment::Placeholder { placeholder: p } => {
                    let key = link_key(p);
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
fn json(node: &Node) -> serde_json::Value {
    use serde_json::Value as J;
    match &node.value {
        Y::Str(s) | Y::Timestamp(s) => J::String(s.clone()),
        Y::Null => J::Null,
        Y::Bool(b) => J::Bool(*b),
        Y::Int(yaml::Number::Small(n)) => {
            i64::try_from(*n).map_or_else(|_| J::String(n.to_string()), J::from)
        }
        Y::Int(yaml::Number::Big(n)) => J::String(n.clone()),
        Y::Float(f) => serde_json::Number::from_f64(*f).map_or(J::Null, J::Number),
        Y::List(items) => J::Array(items.iter().map(json).collect()),
        Y::Map(pairs) => J::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.as_str().unwrap_or_default().to_owned(), json(v)))
                .collect(),
        ),
    }
}
/// A valid single catalog file as its document value, for the form editor.
pub fn source_value(text: &str) -> std::result::Result<serde_json::Value, String> {
    check(text, "draft").map_err(|e| e.to_string())?;
    let documents = yaml::load(text).map_err(|_| "Malformed YAML.".to_owned())?;
    Ok(json(&documents[0]))
}
/// Canonical schema 1.1 YAML exactly as Director's catalog editor writes it.
pub fn source_text(value: &serde_json::Value) -> std::result::Result<String, String> {
    Ok(editor::serialize(value))
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
