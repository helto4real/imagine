//! Prompt Studio protection has distinct lexical rules from timeline reference tags.
use crate::video::text::{self, PythonText};
use std::collections::BTreeSet;
pub fn enhancement_user_prompt(
    section: bool,
    context: &str,
    instruction: &str,
    source: &str,
    images: bool,
    opening: bool,
) -> String {
    let context = context.pytrim();
    let task = if section {
        "section prompt (prompt)"
    } else {
        "scene field (description)"
    };
    let rule = if images && opening {
        "Images are supplied. Treat them as visual truth and do not describe the visible environment in detail. The first image is the Scene's opening frame: the video starts on it, so describe what moves and changes from that frame on rather than a different opening."
    } else if images {
        "Images are supplied. Treat them as visual truth and do not describe the visible environment in detail."
    } else {
        "No images are supplied. Add the environmental, spatial, lighting, and atmospheric detail needed for visual grounding."
    };
    format!(
        "Task:\nEnhance one {task}. Return only the enhanced text.\n\nVisual context rule:\n{rule}\n\nScene context:\n{}\n\nAdditional user instruction:\n{}\n\nSource text:\n{}",
        if context.is_empty() { "None" } else { context },
        if instruction.is_empty() {
            "None"
        } else {
            instruction
        },
        if source.is_empty() {
            "Create suitable text for this empty field."
        } else {
            source
        }
    )
}
pub struct Protected {
    pub source: String,
    pub text: String,
    pub references: Vec<(String, String)>,
}
fn tokens(text: &str) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut consumed = 0;
    for (start, c) in text.char_indices() {
        if start < consumed
            || c != '@'
            || text[..start]
                .chars()
                .next_back()
                .is_some_and(|c| text::word(c) || c == '@')
        {
            continue;
        }
        let mut end = start + 1;
        if !text
            .as_bytes()
            .get(end)
            .is_some_and(u8::is_ascii_alphanumeric)
        {
            continue;
        }
        end += 1;
        while text
            .as_bytes()
            .get(end)
            .is_some_and(|b| b.is_ascii_alphanumeric() || b"_.-".contains(b))
        {
            end += 1;
        }
        out.push((start, end));
        consumed = end;
    }
    out
}
/// Director's markers for every reference in `text`, given the markers
/// `protect_references` assigned to the whole payload `text` belongs to.
pub fn apply(text: &str, references: &[(String, String)]) -> String {
    let mut out = String::new();
    let mut cursor = 0;
    for (start, end) in tokens(text) {
        out.push_str(&text[cursor..start]);
        let reference = &text[start..end];
        match references.iter().find(|(_, r)| r == reference) {
            Some((marker, _)) => out.push_str(marker),
            None => out.push_str(reference),
        }
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}
/// `restore_references_in_value` on JSON text: markers never need escaping.
pub fn restore_all(text: &str, references: &[(String, String)]) -> String {
    references
        .iter()
        .fold(text.to_owned(), |text, (marker, reference)| {
            text.replace(marker, reference)
        })
}
impl Protected {
    pub fn new(source: &str) -> Self {
        let mut references: Vec<(String, String)> = vec![];
        let mut text = String::new();
        let mut cursor = 0;
        for (start, end) in tokens(source) {
            text.push_str(&source[cursor..start]);
            let reference = &source[start..end];
            let marker = if let Some((marker, _)) = references.iter().find(|(_, r)| r == reference)
            {
                marker.clone()
            } else {
                let marker = format!("[[DIRECTOR_REF_{:04}]]", references.len() + 1);
                references.push((marker.clone(), reference.into()));
                marker
            };
            text.push_str(&marker);
            cursor = end;
        }
        text.push_str(&source[cursor..]);
        Self {
            source: source.into(),
            text,
            references,
        }
    }
    pub fn restore(&self, text: &str) -> Result<(String, bool), &'static str> {
        let mut restored = text.to_owned();
        for (marker, reference) in &self.references {
            if !restored.contains(marker) {
                return Err("authoring_reference_marker_missing");
            }
            restored = restored.replace(marker, reference);
        }
        let before = tokens(&self.source)
            .into_iter()
            .map(|(a, b)| &self.source[a..b])
            .collect::<BTreeSet<_>>();
        let introduced = tokens(&restored)
            .into_iter()
            .any(|(a, b)| !before.contains(&restored[a..b]));
        Ok((restored, introduced))
    }
}
