//! Director semantic placement -> code-owned image bindings and dialogue limits.
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
pub struct Beat {
    pub key: String,
    pub ids: Vec<String>,
    pub role: &'static str,
}
pub fn beats(images: &[StoryImage]) -> Result<Vec<Beat>, &'static str> {
    let mut out = vec![];
    let mut start = None;
    for image in images {
        match image.usage {
            ImageUse::SettingReference => {}
            ImageUse::RequiredMoment => out.push(Beat {
                key: format!("moment_{}", image.id),
                ids: vec![image.id.clone()],
                role: "depicted_moment",
            }),
            ImageUse::StartFrame => {
                if start.is_some() {
                    return Err("story_image_pair_incomplete");
                }
                start = Some(image.id.clone());
            }
            ImageUse::EndFrame => {
                let first = start.take().ok_or("story_image_pair_incomplete")?;
                out.push(Beat {
                    key: format!("pair_{first}"),
                    ids: vec![first, image.id.clone()],
                    role: "state_transition",
                });
            }
        }
    }
    if start.is_some() {
        return Err("story_image_pair_incomplete");
    }
    Ok(out)
}
pub fn schema(images: &[StoryImage], dialogue: bool) -> Result<Value, &'static str> {
    let beats = beats(images)?;
    let settings = images
        .iter()
        .filter(|i| i.usage == ImageUse::SettingReference)
        .map(|i| (i.id.clone(), json!({"type":"boolean"})))
        .collect::<serde_json::Map<_, _>>();
    let mut section = json!({"type":"object","required":["prompt","weight"],"additionalProperties":false,"properties":{"prompt":{"type":"string","minLength":1},"weight":{"type":"number","exclusiveMinimum":0}}});
    if dialogue {
        section["properties"]["dialogue_intent"] = json!({"type":"object","required":["participants","purpose"],"additionalProperties":false,"properties":{"participants":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}},"purpose":{"type":"string","minLength":1}}});
    }
    Ok(
        json!({"type":"object","required":["scenes","visual_placements"],"additionalProperties":false,"properties":{"scenes":{"type":"array","minItems":1,"maxItems":128,"items":{"type":"object","required":["title","summary","description","ambient_sound","non_diegetic_music","weight","boundary_from_previous","setting_reference_matches","sections"],"additionalProperties":false,"properties":{"title":{"type":"string"},"summary":{"type":"string"},"description":{"type":"string"},"ambient_sound":{"type":"string"},"non_diegetic_music":{"type":"string"},"weight":{"type":"number","exclusiveMinimum":0},"boundary_from_previous":{"enum":["hard_cut","continuous_shot"]},"setting_reference_matches":{"type":"object","additionalProperties":false,"required":settings.keys().collect::<Vec<_>>(),"properties":settings},"sections":{"type":"array","minItems":1,"maxItems":12,"items":section}}}},"visual_placements":{"type":"array","minItems":beats.len(),"maxItems":beats.len(),"items":{"type":"object","additionalProperties":false,"required":["beat_key","scene_ordinal","section_ordinal","semantic_role"],"properties":{"beat_key":{"enum":beats.iter().map(|b|&b.key).collect::<Vec<_>>()},"scene_ordinal":{"type":"integer","minimum":1},"section_ordinal":{"type":"integer","minimum":1},"semantic_role":{"enum":["depicted_moment","state_transition"]}}}}}}),
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
        if beat.role == "depicted_moment" {
            let ids = section["image_ids"].as_array_mut().unwrap();
            if ids.contains(&json!(beat.ids[0])) {
                return Err("invalid_story_visual_placements");
            }
            ids.push(json!(beat.ids[0]));
        } else {
            if section["first_frame_id"] != "" || section["last_frame_id"] != "" {
                return Err("invalid_story_visual_placements");
            }
            section["first_frame_id"] = json!(beat.ids[0]);
            section["last_frame_id"] = json!(beat.ids[1]);
        }
    }
    let board: Storyboard = serde_json::from_value(value).map_err(|_| "invalid_storyboard")?;
    validate(&board)?;
    Ok(board)
}
