//! Director semantic placement -> code-owned image bindings and dialogue limits.
use super::director::{Contract, DEPICTED_MOMENT, Py, STATE_TRANSITION, outline_schema};
use super::engine::*;
use serde_json::{Value, json};
pub fn validate_cast(board: &Storyboard, cast: &Value, dialogue: bool) -> Result<(), &'static str> {
    let tags = cast
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["reference_tag"].as_str())
        .collect::<Vec<_>>();
    for scene in &board.scenes {
        for section in &scene.sections {
            if let Some(intent) = &section.dialogue_intent
                && (!dialogue
                    || intent.participants.len() != 2
                    || intent
                        .participants
                        .iter()
                        .any(|p| !tags.contains(&p.as_str()))
                    || intent.participants[0] == intent.participants[1])
            {
                return Err("story_dialogue_cast_missing");
            }
        }
    }
    Ok(())
}
pub use super::director::Beat;
/// Director's visual beats: every Required moment in image order, then each
/// start/end pair as a directed beat `beat_<n>`.
pub fn beats(images: &[StoryImage]) -> Result<Vec<Beat>, &'static str> {
    let mut out = images
        .iter()
        .filter(|i| i.usage == ImageUse::RequiredMoment)
        .map(|image| Beat {
            key: format!("required_moment:{}", image.id),
            image_ids: vec![image.id.clone()],
            role: DEPICTED_MOMENT,
        })
        .collect::<Vec<_>>();
    let starts = images
        .iter()
        .filter(|i| i.usage == ImageUse::StartFrame)
        .collect::<Vec<_>>();
    let ends = images
        .iter()
        .filter(|i| i.usage == ImageUse::EndFrame)
        .collect::<Vec<_>>();
    let mut used = std::collections::BTreeSet::new();
    for (n, image) in starts.iter().enumerate() {
        let target = image
            .pair_with
            .as_deref()
            .or_else(|| (starts.len() == 1 && ends.len() == 1).then(|| ends[0].id.as_str()))
            .ok_or("story_image_pair_incomplete")?;
        let end = images
            .iter()
            .find(|end| end.id == target)
            .ok_or("story_image_pair_incomplete")?;
        used.insert(end.id.as_str());
        out.push(Beat {
            key: format!("start_end_pair:beat_{}", n + 1),
            image_ids: vec![image.id.clone(), end.id.clone()],
            role: STATE_TRANSITION,
        });
    }
    if ends.iter().any(|end| !used.contains(end.id.as_str())) {
        return Err("story_image_pair_incomplete");
    }
    Ok(out)
}
/// The outline contract for `images`: setting references may recur; other
/// story images are placed once.
pub fn contract(images: &[StoryImage], dialogue: bool) -> Result<Contract, &'static str> {
    Ok(Contract {
        setting_ids: images
            .iter()
            .filter(|i| i.usage == ImageUse::SettingReference)
            .map(|i| i.id.clone())
            .collect(),
        beats: beats(images)?,
        dialogue,
        opening: None,
    })
}
/// The outline answer's JSON Schema.
pub fn schema(images: &[StoryImage], dialogue: bool) -> Result<Value, &'static str> {
    Ok(serde_json::from_str(&outline_schema(&contract(images, dialogue)?).compact()).unwrap())
}
/// Director's `image_inventory` rows: a setting reference is reusable, every
/// other story image is placed once. Imagine keeps no separate user cue.
pub fn inventory_rows(images: &[StoryImage], notes: &dyn Fn(&str) -> String) -> Py {
    Py::List(
        images
            .iter()
            .map(|image| {
                Py::obj([
                    ("id", Py::str(&image.id)),
                    ("usage", Py::str(usage(image.usage))),
                    ("label", Py::str(&image.label)),
                    ("user_note", Py::str("")),
                    ("inventory_note", Py::Str(notes(&image.id))),
                    ("reuse", Py::Bool(image.usage == ImageUse::SettingReference)),
                ])
            })
            .collect(),
    )
}
/// Director's visual usage for a story image.
pub fn usage(usage: ImageUse) -> &'static str {
    if usage == ImageUse::RequiredMoment {
        "required_moment"
    } else {
        "story_reference"
    }
}
/// `storyVisualBeatPromptRecords`.
pub fn beat_records(beats: &[Beat]) -> Py {
    let mut moment = 0;
    let mut pair = 0;
    Py::List(
        beats
            .iter()
            .map(|beat| {
                if beat.role == DEPICTED_MOMENT {
                    moment += 1;
                    Py::obj([
                        ("beat_key", Py::str(&beat.key)),
                        ("type", Py::str("required_moment")),
                        ("order", Py::Int(moment)),
                        ("image_id", Py::str(&beat.image_ids[0])),
                        ("cue", Py::str("")),
                    ])
                } else {
                    pair += 1;
                    Py::obj([
                        ("id", Py::Str(format!("beat_{pair}"))),
                        ("start_image_id", Py::str(&beat.image_ids[0])),
                        ("end_image_id", Py::str(&beat.image_ids[1])),
                        ("cue", Py::str("")),
                        ("beat_key", Py::str(&beat.key)),
                        ("type", Py::str("start_end_pair")),
                        ("order", Py::Int(pair)),
                    ])
                }
            })
            .collect(),
    )
}
pub fn project(
    mut value: Value,
    images: &[StoryImage],
    dialogue: bool,
) -> Result<Storyboard, &'static str> {
    let beats = beats(images)?;
    let placements = value
        .as_object_mut()
        .and_then(|v| v.remove("visual_placements"))
        .unwrap_or(json!([]));
    let placements = placements
        .as_array()
        .ok_or("invalid_story_visual_placements")?;
    if placements.len() != beats.len() {
        return Err("invalid_story_visual_placements");
    }
    let scenes = value["scenes"].as_array_mut().ok_or("invalid_storyboard")?;
    for (i, scene) in scenes.iter_mut().enumerate() {
        let matches = scene
            .as_object_mut()
            .and_then(|s| s.remove("setting_reference_matches"))
            .unwrap_or(json!({}));
        let settings = images
            .iter()
            .filter(|im| im.usage == ImageUse::SettingReference)
            .collect::<Vec<_>>();
        if matches
            .as_object()
            .is_none_or(|m| m.len() != settings.len())
            || settings.iter().any(|im| !matches[&im.id].is_boolean())
        {
            return Err("invalid_story_setting_matches");
        }
        scene["image_ids"] = json!(
            settings
                .iter()
                .filter(|im| matches[&im.id] == true)
                .map(|im| &im.id)
                .collect::<Vec<_>>()
        );
        scene["id"] = json!(format!("scene_{:03}", i + 1));
        for (j, s) in scene["sections"]
            .as_array_mut()
            .ok_or("invalid_storyboard")?
            .iter_mut()
            .enumerate()
        {
            s["id"] = json!(format!("scene_{:03}_section_{:03}", i + 1, j + 1));
            s["image_ids"] = json!([]);
            s["first_frame_id"] = json!("");
            s["last_frame_id"] = json!("");
            if let Some(intent) = s.get_mut("dialogue_intent") {
                if !dialogue || intent.get("max_turns").is_some() {
                    return Err("invalid_dialogue_intent");
                }
                intent["max_turns"] = json!(2);
            }
        }
    }
    for (placement, beat) in placements.iter().zip(beats) {
        if placement["beat_key"] != beat.key || placement["semantic_role"] != beat.role {
            return Err("invalid_story_visual_placements");
        }
        let i = placement["scene_ordinal"]
            .as_u64()
            .and_then(|n| n.checked_sub(1))
            .ok_or("invalid_story_visual_placements")? as usize;
        let j = placement["section_ordinal"]
            .as_u64()
            .and_then(|n| n.checked_sub(1))
            .ok_or("invalid_story_visual_placements")? as usize;
        let section = scenes
            .get_mut(i)
            .and_then(|s| s["sections"].as_array_mut())
            .and_then(|v| v.get_mut(j))
            .ok_or("invalid_story_visual_placements")?;
        if beat.role == DEPICTED_MOMENT {
            let ids = section["image_ids"].as_array_mut().unwrap();
            if ids.contains(&json!(beat.image_ids[0])) {
                return Err("invalid_story_visual_placements");
            }
            ids.push(json!(beat.image_ids[0]));
        } else {
            if section["first_frame_id"] != "" || section["last_frame_id"] != "" {
                return Err("invalid_story_visual_placements");
            }
            section["first_frame_id"] = json!(beat.image_ids[0]);
            section["last_frame_id"] = json!(beat.image_ids[1]);
        }
    }
    let board: Storyboard = serde_json::from_value(value).map_err(|_| "invalid_storyboard")?;
    validate(&board)?;
    Ok(board)
}

#[cfg(test)]
mod pair_tests {
    use super::*;
    #[test]
    fn directed_pair_is_independent_of_id_sorting_and_explicit_pairs_do_not_cross() {
        let mut images: Vec<StoryImage> = serde_json::from_value(json!([
            {"id":"a_end","asset":"own_end","label":"End","usage":"end_frame"},
            {"id":"z_start","asset":"own_start","label":"Start","usage":"start_frame"}
        ]))
        .unwrap();
        assert_eq!(
            beats(&images).unwrap()[0].image_ids,
            vec!["z_start", "a_end"]
        );
        images.reverse();
        assert_eq!(
            beats(&images).unwrap()[0].image_ids,
            vec!["z_start", "a_end"]
        );
        images[0].pair_with = Some("a_end".into());
        images.push(serde_json::from_value(json!({"id":"second_start","asset":"own_two","label":"Second","usage":"start_frame","pair_with":"a_end"})).unwrap());
        let paired = beats(&images).unwrap();
        assert_eq!(paired[0].image_ids, vec!["z_start", "a_end"]);
        assert_eq!(paired[1].image_ids, vec!["second_start", "a_end"]);
        images[2].pair_with = Some("second_start".into());
        assert_eq!(
            beats(&images).unwrap()[1].image_ids,
            vec!["second_start", "second_start"]
        );
    }
}
