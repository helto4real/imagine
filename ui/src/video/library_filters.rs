//! Director's Library filtering: a search over title, description and tags,
//! the tab's filters, one tag, and four sort orders.
use serde_json::Value;
use std::collections::BTreeSet;

use super::VideoProject;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Sort {
    #[default]
    Recent,
    Oldest,
    NameAz,
    NameZa,
}
impl Sort {
    pub(super) const ALL: [Self; 4] = [Self::Recent, Self::Oldest, Self::NameAz, Self::NameZa];
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Recent => "Recently updated",
            Self::Oldest => "Oldest first",
            Self::NameAz => "Name A–Z",
            Self::NameZa => "Name Z–A",
        }
    }
}
/// A character keeps its tags and description in its own document; other
/// entries keep them on the entry.
pub(super) fn tags(entry: &Value) -> Vec<String> {
    let source = if entry["content"]["kind"] == "character" {
        &entry["content"]["document"]["tags"]
    } else {
        &entry["tags"]
    };
    source
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t.as_str().map(str::to_owned))
        .collect()
}
pub(super) fn description(entry: &Value) -> &str {
    if entry["content"]["kind"] == "character" {
        entry["content"]["document"]["description"].as_str()
    } else {
        entry["description"].as_str()
    }
    .unwrap_or("")
}
/// The filters a tab offers, as (key, label).
pub(super) fn offered(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "project" => &[("references", "Has references"), ("audio", "Has audio")],
        "character" => &[("loaded", "Loaded in project"), ("tagged", "Has tags")],
        _ => &[
            ("image", "Images"),
            ("video", "Videos"),
            ("audio", "Audio"),
            ("in_project", "In project"),
        ],
    }
}
/// Whether the open project already holds a copy of `entry`.
pub(super) fn in_project(entry: &Value, project: Option<&VideoProject>) -> bool {
    let (Some(project), Some(id)) = (project, entry["id"].as_str()) else {
        return false;
    };
    match entry["content"]["kind"].as_str() {
        Some("asset") => project
            .assets
            .values()
            .any(|a| a.library_source.as_ref().is_some_and(|s| s.entry_id == id)),
        Some("character") => project
            .creative
            .references
            .iter()
            .any(|r| r.origin.as_ref().is_some_and(|o| o.entry_id == id)),
        _ => false,
    }
}
fn asset_kind(entry: &Value) -> &str {
    entry["content"]["document"]["asset"]["kind"]
        .as_str()
        .unwrap_or("")
}
fn matches(filter: &str, entry: &Value, project: Option<&VideoProject>) -> bool {
    let doc = &entry["content"]["document"];
    match filter {
        "references" => doc["project"]["creative"]["references"]
            .as_array()
            .is_some_and(|r| !r.is_empty()),
        "audio" if entry["content"]["kind"] == "project" => doc["project"]["assets"]
            .as_object()
            .is_some_and(|a| a.values().any(|a| a["kind"] == "audio")),
        "tagged" => !tags(entry).is_empty(),
        "loaded" | "in_project" => in_project(entry, project),
        kind => asset_kind(entry) == kind,
    }
}
/// The tab's entries for the search, filters and tag, in `sort` order, with
/// their index in `entries`. Media type filters widen each other; the others narrow.
pub(super) fn rows(
    entries: &[Value],
    kind: &str,
    search: &str,
    filters: &BTreeSet<&'static str>,
    tag: Option<&str>,
    sort: Sort,
    project: Option<&VideoProject>,
) -> Vec<(usize, Value)> {
    let needle = search.trim().to_lowercase();
    let types = ["image", "video", "audio"];
    let chosen_types: Vec<_> = if kind == "asset" {
        filters.iter().filter(|f| types.contains(f)).collect()
    } else {
        vec![]
    };
    let mut out: Vec<(usize, Value)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e["content"]["kind"] == kind)
        .filter(|(_, e)| {
            needle.is_empty()
                || e["title"]
                    .as_str()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains(&needle)
                || description(e).to_lowercase().contains(&needle)
                || tags(e).iter().any(|t| t.to_lowercase().contains(&needle))
        })
        .filter(|(_, e)| {
            (chosen_types.is_empty() || chosen_types.iter().any(|f| matches(f, e, project)))
                && filters
                    .iter()
                    .filter(|f| !chosen_types.contains(f))
                    .all(|f| matches(f, e, project))
        })
        .filter(|(_, e)| tag.is_none_or(|t| tags(e).iter().any(|x| x == t)))
        .map(|(i, e)| (i, e.clone()))
        .collect();
    let name = |e: &Value| e["title"].as_str().unwrap_or("").to_lowercase();
    // Entries saved before dates were kept stay in saved order: later is newer.
    let when = |(i, e): &(usize, Value)| (e["updated_ms"].as_u64().unwrap_or(0), *i);
    match sort {
        Sort::Recent => out.sort_by_key(|row| std::cmp::Reverse(when(row))),
        Sort::Oldest => out.sort_by_key(when),
        Sort::NameAz => out.sort_by_key(|(_, e)| name(e)),
        Sort::NameZa => out.sort_by_key(|(_, e)| std::cmp::Reverse(name(e))),
    }
    out
}
/// The tags used on the tab, each with how many entries carry it, by name.
pub(super) fn tag_counts(entries: &[Value], kind: &str) -> Vec<(String, usize)> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for entry in entries.iter().filter(|e| e["content"]["kind"] == kind) {
        for tag in tags(entry) {
            *counts.entry(tag).or_default() += 1;
        }
    }
    counts.into_iter().collect()
}
/// A card's second line: what a project holds, or an asset's type.
pub(super) fn meta(entry: &Value) -> String {
    let doc = &entry["content"]["document"];
    match entry["content"]["kind"].as_str() {
        Some("project") => {
            let count = |v: &Value| v.as_array().map_or(0, Vec::len);
            let audio = doc["project"]["assets"]
                .as_object()
                .map_or(0, |a| a.values().filter(|a| a["kind"] == "audio").count());
            format!(
                "{} scenes · {} refs · {audio} audio",
                count(&doc["project"]["scenes"]),
                count(&doc["project"]["creative"]["references"])
            )
        }
        Some("character") => tags(entry).join(", "),
        _ => match asset_kind(entry) {
            "image" => "Image".into(),
            "video" => "Video".into(),
            "audio" => "Audio".into(),
            _ => String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn asset(id: &str, title: &str, kind: &str, updated: u64, tags: &[&str]) -> Value {
        json!({"id":id,"title":title,"revision":1,"updated_ms":updated,"tags":tags,
            "content":{"kind":"asset","document":{"asset":{"kind":kind}}}})
    }
    #[test]
    fn rows_search_filter_tag_and_sort_as_director_does() {
        let entries = vec![
            asset("a", "Own harbour", "image", 30, &["Dusk"]),
            asset("b", "Own beacon", "video", 10, &["Night", "Dusk"]),
            asset("c", "Own waves", "audio", 20, &[]),
            json!({"id":"p","title":"Own project","revision":1,"content":{"kind":"project","document":{}}}),
        ];
        let none = BTreeSet::new();
        let ids = |rows: Vec<(usize, Value)>| {
            rows.iter()
                .map(|(_, e)| e["id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        let rows = |filters: &BTreeSet<&'static str>, tag, sort, search| {
            ids(super::rows(
                &entries, "asset", search, filters, tag, sort, None,
            ))
        };
        assert_eq!(rows(&none, None, Sort::Recent, ""), ["a", "c", "b"]);
        assert_eq!(rows(&none, None, Sort::Oldest, ""), ["b", "c", "a"]);
        assert_eq!(rows(&none, None, Sort::NameAz, ""), ["b", "a", "c"]);
        assert_eq!(rows(&none, None, Sort::NameZa, ""), ["c", "a", "b"]);
        // The search covers tags too.
        assert_eq!(rows(&none, None, Sort::NameAz, "night"), ["b"]);
        // Media types widen each other.
        let types = BTreeSet::from(["image", "video"]);
        assert_eq!(rows(&types, None, Sort::NameAz, ""), ["b", "a"]);
        assert_eq!(rows(&none, Some("Dusk"), Sort::NameAz, ""), ["b", "a"]);
        // Nothing is in a project without one.
        assert!(rows(&BTreeSet::from(["in_project"]), None, Sort::NameAz, "").is_empty());
        assert_eq!(
            tag_counts(&entries, "asset"),
            [("Dusk".to_owned(), 2), ("Night".to_owned(), 1)]
        );
        assert_eq!(meta(&entries[1]), "Video");
    }
}
