//! Director Take review selection, retaining attachment order rather than dates.
pub(super) fn normalize(
    ids: &[String],
    accepted: Option<&str>,
    selection: &[Option<String>; 2],
) -> [Option<String>; 2] {
    if ids.len() < 2 {
        return [ids.first().cloned(), None];
    }
    let latest = ids.last().unwrap();
    let accepted = accepted.and_then(|id| ids.iter().find(|candidate| candidate.as_str() == id));
    let fallback = match accepted {
        Some(id) if id != latest => [id.clone(), latest.clone()],
        _ => [ids[ids.len() - 2].clone(), latest.clone()],
    };
    let a = selection[0]
        .as_ref()
        .filter(|id| ids.contains(id))
        .cloned()
        .unwrap_or_else(|| fallback[0].clone());
    let b = selection[1]
        .as_ref()
        .filter(|id| ids.contains(id))
        .cloned()
        .unwrap_or_else(|| fallback[1].clone());
    let b = if a == b {
        ids.iter().find(|id| *id != &a).cloned()
    } else {
        Some(b)
    };
    [Some(a), b]
}

pub(super) fn update(
    ids: &[String],
    accepted: Option<&str>,
    selection: &[Option<String>; 2],
    side: usize,
    id: &str,
) -> [Option<String>; 2] {
    let mut next = normalize(ids, accepted, selection);
    if side > 1 || !ids.iter().any(|candidate| candidate == id) {
        return next;
    }
    let other = 1 - side;
    if next[other].as_deref() == Some(id) {
        next[other] = next[side].clone();
    }
    next[side] = Some(id.into());
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_matches_executed_director_source() {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("take-selection-director.json")).unwrap();
        let cases = oracle["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for row in cases {
            let name = row["name"].as_str().unwrap();
            let ids: Vec<String> = serde_json::from_value(row["ids"].clone()).unwrap();
            let accepted = row["accepted"].as_str();
            let selection = serde_json::from_value(row["selection"].clone()).unwrap();
            let expected: [Option<String>; 2] =
                serde_json::from_value(row["normalized"].clone()).unwrap();
            assert_eq!(normalize(&ids, accepted, &selection), expected, "{name}");
            let expected: [Option<String>; 2] =
                serde_json::from_value(row["default"].clone()).unwrap();
            assert_eq!(
                normalize(&ids, accepted, &[None, None]),
                expected,
                "{name} default"
            );
            if let Some(change) = row["change"].as_object() {
                let expected: [Option<String>; 2] =
                    serde_json::from_value(row["updated"].clone()).unwrap();
                assert_eq!(
                    update(
                        &ids,
                        accepted,
                        &selection,
                        change["side"].as_u64().unwrap() as usize,
                        change["id"].as_str().unwrap()
                    ),
                    expected,
                    "{name} update"
                );
            }
        }
    }
}
