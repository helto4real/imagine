//! Director reference-manifest definitions and retention wording.
use super::{common::*, f, s};
use crate::video::text::PythonText;
use serde_json::Value;
use std::collections::BTreeMap;
pub(super) fn list<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v[key].as_array().map_or(&[], Vec::as_slice)
}
fn label(v: &Value) -> &str {
    s(v, "prompt_label")
}
fn marker(v: &Value, default: &str) -> String {
    let m = s(v, "marker");
    if m.is_empty() {
        default.into()
    } else {
        m.into()
    }
}
fn target_note(r: &Value, subjects: &BTreeMap<&str, &Value>) -> String {
    let note = s(r, "note").pytrim();
    if s(r, "marker") != "attribute_transfer" {
        return note.into();
    }
    let target = subjects
        .get(s(r, "target_reference_id"))
        .map_or("", |s| label(s));
    if target.is_empty() {
        note.into()
    } else if note.is_empty() {
        format!("The defined attributes are transferred to {target}.")
    } else if !note.contains(target) {
        format!(
            "{} The transfer target is {target}.",
            note.trim_end_matches('.')
        )
    } else {
        note.into()
    }
}
pub(super) fn picture_instruction(p: &Value, b: &Value) -> String {
    if s(p, "role") == "boundary_first_frame" {
        return String::new();
    }
    let l = label(p);
    let custom = s(b, "custom_role").pytrim().trim_end_matches('.');
    match s(b, "resolved_role") {
        "first_frame" => format!("The shot begins from {l}."),
        "last_frame" => format!("The shot ends at {l}."),
        "keyframe" => format!("The shot's keyframe corresponds to {l}."),
        "custom_influence" if !custom.is_empty() => format!("{l} provides {custom}."),
        _ => format!(
            "{l} influences the shot's visual interpretation without fixing its first or last frame."
        ),
    }
}
fn picture_definition(p: &Value, b: &Value) -> String {
    let l = label(p);
    let shot = format!("[Shot {}]", f(b, "internal_shot_index") as i64 + 1);
    let custom = s(b, "custom_role").pytrim().trim_end_matches('.');
    match s(b, "resolved_role") {
        "first_frame" => {
            format!("{l} is the first frame of {shot}, defining its opening composition.")
        }
        "last_frame" => {
            format!("{l} is the last frame of {shot}, defining its closing composition.")
        }
        "keyframe" => format!("{l} is the keyframe of {shot}, defining its composition."),
        "custom_influence" if !custom.is_empty() => format!("{l} provides {custom} for {shot}."),
        _ => format!("{l} is a visual influence reference for {shot}."),
    }
}
fn video_definition(v: &Value) -> String {
    let l = label(v);
    let custom = s(v, "custom_role").pytrim().trim_end_matches('.');
    if s(v, "role") == "boundary_continuation" {
        return if v["motion_only"] == true {
            format!(
                "{l} is the motion and timing reference for [Shot 1], providing subject motion, camera trajectory, and framing development after the opening frame."
            )
        } else {
            format!("{l} is the source video whose ending is continued by the target video.")
        };
    }
    match s(v, "role") {
        "source_video_edit" => format!("{l} is the source video for the target video edit."),
        "custom_video_reference" if !custom.is_empty() => format!(
            "{l} provides {custom} for [Shot {}].",
            f(v, "internal_shot_index") as i64 + 1
        ),
        _ => format!(
            "{l} is the motion, camera, rhythm, and temporal-structure reference for [Shot {}].",
            f(v, "internal_shot_index") as i64 + 1
        ),
    }
}
pub(super) fn definitions(manifest: &Value) -> Vec<String> {
    let mut out = vec![];
    let restore = list(manifest, "images")
        .iter()
        .any(|p| s(p, "role") == "boundary_first_frame");
    if !restore {
        out.extend(
            list(manifest, "videos")
                .iter()
                .filter(|v| s(v, "role") == "boundary_continuation")
                .map(video_definition),
        );
    }
    for subject in list(manifest, "subjects") {
        let l = label(subject);
        let mut desc = s(subject, "description")
            .pytrim()
            .trim_end_matches('.')
            .to_owned();
        if desc.is_empty() {
            desc = format!(
                "the referenced {}",
                match s(subject, "reference_kind") {
                    "character" => "character",
                    "object" => "object",
                    "scene" => "scene",
                    "appearance" => "appearance or visual element",
                    "style_action" => "style, action, expression, or pose",
                    _ => "visible subject",
                }
            );
        }
        let images = list(subject, "source_images");
        let labels = images
            .iter()
            .filter(|p| !label(p).is_empty())
            .map(|p| label(p).to_owned())
            .collect::<Vec<_>>();
        let mut line = if labels.is_empty() {
            format!("{l} is {desc}.")
        } else {
            format!("{l} is {desc} in {}.", english(&labels))
        };
        let contribution = images
            .iter()
            .filter(|p| !label(p).is_empty() && !s(p, "contribution").pytrim().is_empty())
            .map(|p| {
                format!(
                    "{} provides {}",
                    label(p),
                    s(p, "contribution").pytrim().trim_end_matches('.')
                )
            })
            .collect::<Vec<_>>();
        if !contribution.is_empty() {
            line = format!("{line} {}.", english(&contribution));
        }
        out.push(line);
    }
    for p in list(manifest, "images") {
        if s(p, "role") == "boundary_first_frame" {
            if !label(p).is_empty() {
                out.push(format!("{} is the first frame of [Shot 1] and defines its opening composition, framing, camera position, subject placement, lighting, color, and object state.",label(p)));
            }
            continue;
        }
        if !label(p).is_empty() {
            out.extend(dedupe(
                list(p, "section_bindings")
                    .iter()
                    .map(|b| picture_definition(p, b)),
            ));
        }
    }
    for v in list(manifest, "videos") {
        if s(v, "role") != "boundary_continuation" || restore {
            out.push(video_definition(v));
        }
    }
    for a in list(manifest, "audios") {
        let (l, subject, speaker, purpose) = (
            label(a),
            s(a, "target_subject_label"),
            s(a, "target_speaker_id"),
            s(a, "purpose"),
        );
        if purpose == "voice_timbre" && !subject.is_empty() && !speaker.is_empty() {
            let voice = if s(a, "voice_description").is_empty() {
                s(a, "description")
            } else {
                s(a, "voice_description")
            };
            let details = dedupe([
                voice.pytrim().trim_end_matches('.').into(),
                s(a, "contribution").pytrim().trim_end_matches('.').into(),
            ]);
            let mut line = format!("{l} is the voice-timbre reference for {subject} ({speaker})");
            if !details.is_empty() {
                line = format!("{line}, providing {}", english(&details));
            }
            out.push(format!("{line}."));
        } else if !s(a, "paired_video_label").is_empty() {
            out.push(format!(
                "{l} is the synchronized audio reference for {}.",
                s(a, "paired_video_label")
            ));
        } else {
            let description = s(a, "description").pytrim().trim_end_matches('.');
            if purpose == "custom" {
                out.push(format!(
                    "{l} is a custom audio reference that {}.",
                    if description.is_empty() {
                        "provides authored guidance"
                    } else {
                        description
                    }
                ));
                continue;
            }
            let role = match purpose {
                "dialogue_lyrics" => "dialogue and lyric-content reference",
                "ambience_sfx" => "ambience and physical-sound reference",
                "music_style" => "music-style reference",
                "beat_rhythm" => "beat and rhythm reference",
                "continuity" => "audio-continuity reference",
                _ => "standalone audio reference",
            };
            out.push(format!(
                "{l} is the {role}{}.",
                if description.is_empty() {
                    String::new()
                } else {
                    format!(", providing {description}")
                }
            ));
        }
    }
    out
}
pub(super) fn retention(manifest: &Value) -> Vec<String> {
    let subjects = list(manifest, "subjects")
        .iter()
        .filter(|s| !s["reference_id"].is_null())
        .map(|v| (s(v, "reference_id"), v))
        .collect::<BTreeMap<_, _>>();
    let restore = list(manifest, "images")
        .iter()
        .any(|p| s(p, "role") == "boundary_first_frame");
    let mut out = vec![];
    let video_line = |v: &Value| {
        let l = label(v);
        if l.is_empty() {
            return String::new();
        }
        let m = if s(v, "role") != "boundary_continuation" || v["motion_only"] == true {
            marker(&v["relationship"], "fully_preserved")
        } else {
            "fully_preserved".into()
        };
        let note = target_note(&v["relationship"], &subjects);
        format!(
            "{l}{}: {m}{}",
            if v["motion_only"] == true {
                " (motion timing, camera trajectory, and subsequent framing evolution)"
            } else {
                ""
            },
            if note.is_empty() {
                String::new()
            } else {
                format!(" - {note}")
            }
        )
    };
    if !restore {
        out.extend(
            list(manifest, "videos")
                .iter()
                .filter(|v| s(v, "role") == "boundary_continuation")
                .map(&video_line),
        );
    }
    for subject in list(manifest, "subjects") {
        let l = label(subject);
        if l.is_empty() {
            continue;
        }
        let r = &subject["relationship"];
        let m = marker(r, "fully_preserved");
        let mut shots = list(subject, "internal_shot_indices")
            .iter()
            .filter_map(Value::as_i64)
            .collect::<Vec<_>>();
        shots.sort();
        shots.dedup();
        let scope = if shots.is_empty() {
            String::new()
        } else {
            format!(
                " (appears in {})",
                english(
                    &shots
                        .iter()
                        .map(|n| format!("[Shot {}]", n + 1))
                        .collect::<Vec<_>>()
                )
            )
        };
        let note = target_note(r, &subjects);
        out.push(format!(
            "{l}{scope}: {m}{}",
            if note.is_empty() {
                String::new()
            } else {
                format!(" - {note}")
            }
        ));
    }
    for p in list(manifest, "images") {
        let l = label(p);
        if l.is_empty() {
            continue;
        }
        for b in list(p, "section_bindings") {
            let m = marker(&b["relationship"], "fully_preserved");
            let note = target_note(&b["relationship"], &subjects);
            let qualifier = match s(b, "resolved_role") {
                "first_frame" => "first frame",
                "last_frame" => "last frame",
                "keyframe" => "keyframe",
                _ => "influence",
            };
            out.push(format!(
                "{l} ([Shot {}] {qualifier}): {m}{}",
                f(b, "internal_shot_index") as i64 + 1,
                if note.is_empty() {
                    String::new()
                } else {
                    format!(" - {note}")
                }
            ));
        }
    }
    out.extend(
        list(manifest, "videos")
            .iter()
            .filter(|v| s(v, "role") != "boundary_continuation" || restore)
            .map(video_line),
    );
    for a in list(manifest, "audios") {
        let l = label(a);
        if !l.is_empty() {
            let m = if s(a, "retention_marker").is_empty() {
                "reference"
            } else {
                s(a, "retention_marker")
            };
            let note = s(a, "retention_note").pytrim();
            out.push(format!(
                "{l}: {m}{}",
                if note.is_empty() {
                    String::new()
                } else {
                    format!(" - {note}")
                }
            ));
        }
    }
    dedupe(out)
}
