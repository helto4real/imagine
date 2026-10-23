//! Pure Story validation, timing and revision-bound edit planning.
use super::profiles;
use crate::video::{
    contract::*,
    editing::{Edit, EditRequest},
};
pub use imagine_video_contract::authoring::*;
type Result<T> = std::result::Result<T, &'static str>;

pub fn validate(board: &Storyboard) -> Result<()> {
    if board.scenes.is_empty() || board.scenes.len() > 128 {
        return Err("invalid_storyboard");
    }
    let mut ids = std::collections::BTreeSet::new();
    for scene in &board.scenes {
        if scene.id.is_empty()
            || !ids.insert(scene.id.clone())
            || !scene.weight.is_finite()
            || scene.weight <= 0.
            || scene.sections.is_empty()
            || scene.sections.len() > 12
            || !["hard_cut", "continuous_shot"].contains(&scene.boundary_from_previous.as_str())
        {
            return Err("invalid_storyboard");
        }
        for text in [
            &scene.title,
            &scene.summary,
            &scene.description,
            &scene.ambient_sound,
            &scene.non_diegetic_music,
        ] {
            if text.len() > 65536 {
                return Err("story_text_limit");
            }
        }
        for section in &scene.sections {
            if section.dialogue_intent.as_ref().is_some_and(|d| {
                d.participants.len() != 2
                    || d.participants.iter().any(String::is_empty)
                    || d.purpose.trim().is_empty()
                    || d.max_turns != 2
            }) {
                return Err("invalid_dialogue_intent");
            }
            if section.id.is_empty()
                || !ids.insert(section.id.clone())
                || !section.weight.is_finite()
                || section.weight <= 0.
                || section.prompt.trim().is_empty()
                || section.prompt.len() > 65536
            {
                return Err("invalid_story_section");
            }
        }
    }
    Ok(())
}
pub fn outline(text: &str) -> Result<Storyboard> {
    outline_with_images(text, &[], false)
}
pub fn outline_with_images(
    text: &str,
    images: &[StoryImage],
    dialogue: bool,
) -> Result<Storyboard> {
    super::visuals::project(
        serde_json::from_str(text).map_err(|_| "authoring_invalid_json")?,
        images,
        dialogue,
    )
}
pub fn expansion(text: &str, source: &StoryScene) -> Result<StoryScene> {
    let scene: StoryScene = serde_json::from_str(text).map_err(|_| "authoring_invalid_json")?;
    validate(&Storyboard {
        scenes: vec![scene.clone()],
    })?;
    if scene.id != source.id
        || scene.weight != source.weight
        || scene.boundary_from_previous != source.boundary_from_previous
        || scene.sections.len() != source.sections.len()
        || scene.image_ids != source.image_ids
        || scene.sections.iter().zip(&source.sections).any(|(a, b)| {
            a.id != b.id
                || a.weight != b.weight
                || a.image_ids != b.image_ids
                || a.first_frame_id != b.first_frame_id
                || a.last_frame_id != b.last_frame_id
                || a.dialogue_intent.as_ref().map(|d| &d.participants)
                    != b.dialogue_intent.as_ref().map(|d| &d.participants)
        })
    {
        return Err("story_expansion_changed_structure");
    }
    Ok(scene)
}
/// Director ordinary-UI allocator: 250ms minimum when possible; weighted remainder and cap redistribution.
pub fn durations(weights: &[f64], duration: f64, cap: f64) -> Result<Vec<f64>> {
    if weights.is_empty()
        || !duration.is_finite()
        || duration <= 0.
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.)
    {
        return Err("invalid_story_duration");
    }
    if duration > weights.len() as f64 * cap + 1e-9 {
        return Err("story_scene_duration_capacity");
    }
    let minimum = if duration >= weights.len() as f64 * 0.25 {
        0.25
    } else {
        duration / weights.len() as f64
    };
    let mut out = vec![minimum; weights.len()];
    let mut active = (0..weights.len()).collect::<Vec<_>>();
    let mut remaining = (duration - minimum * weights.len() as f64).max(0.);
    while remaining > 1e-12 && !active.is_empty() {
        let sum = active.iter().map(|&i| weights[i].max(0.001)).sum::<f64>();
        let saturated = active
            .iter()
            .copied()
            .filter(|&i| out[i] + remaining * weights[i].max(0.001) / sum > cap + 1e-12)
            .collect::<Vec<_>>();
        if saturated.is_empty() {
            for &i in &active {
                out[i] += remaining * weights[i].max(0.001) / sum;
            }
            remaining = 0.;
        } else {
            for i in saturated {
                remaining -= (cap - out[i]).max(0.);
                out[i] = cap;
                active.retain(|&j| j != i);
            }
            remaining = remaining.max(0.);
        }
    }
    if remaining > 1e-9 {
        return Err("story_scene_duration_capacity");
    }
    let last = out.len() - 1;
    out[last] = duration - out[..last].iter().sum::<f64>();
    Ok(out)
}
fn time(seconds: f64) -> Time {
    Time {
        num: (seconds * 1_000_000.).round() as u64,
        den: 1_000_000,
    }
    .normalized()
}
fn seconds(t: Time) -> f64 {
    t.num as f64 / t.den as f64
}
pub fn apply(base: &VideoProject, draft: &Draft) -> Result<EditRequest> {
    if draft.phase != Phase::Review
        || base.id != draft.project_id
        || base.project_revision != draft.request.expected_project_revision
    {
        return Err("authoring_revision_conflict");
    }
    let request = &draft.request;
    let mut commands = vec![];
    let mut creative = base.creative.clone();
    match request.operation {
        Operation::EnhanceScene => {
            let scene_id = request.scene_id.clone().ok_or("scene_not_found")?;
            base.scene(&scene_id).map_err(|_| "scene_not_found")?;
            let text = draft
                .text
                .clone()
                .filter(|t| !t.trim().is_empty())
                .ok_or("authoring_empty_draft")?;
            match request.scene_field.unwrap_or_default() {
                SceneField::Description => commands.push(Edit::SetSceneDescription {
                    scene_id,
                    description: text,
                }),
                field => {
                    let source = creative.scene_prompts.entry(scene_id).or_default();
                    match field {
                        SceneField::Summary => source.summary = text,
                        SceneField::AmbientSound => source.ambient_sound = text,
                        SceneField::NonDiegeticMusic => source.non_diegetic_music = text,
                        SceneField::Description => unreachable!(),
                    }
                    commands.push(Edit::SetCreative { creative });
                }
            }
        }
        Operation::EnhanceSection => {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or("scene_not_found")?)
                .map_err(|_| "scene_not_found")?;
            let mut section = scene
                .sections
                .iter()
                .find(|s| Some(&s.id) == request.section_id.as_ref())
                .cloned()
                .ok_or("section_not_found")?;
            section.prompt = draft
                .text
                .clone()
                .filter(|t| !t.trim().is_empty())
                .ok_or("authoring_empty_draft")?;
            commands.push(Edit::UpdateSection {
                scene_id: scene.id.clone(),
                section,
            });
        }
        Operation::Outline => {
            let scenes = draft
                .scenes
                .iter()
                .cloned()
                .collect::<Option<Vec<_>>>()
                .ok_or("story_not_expanded")?;
            validate(&Storyboard {
                scenes: scenes.clone(),
            })?;
            let lengths = durations(
                &scenes.iter().map(|s| s.weight).collect::<Vec<_>>(),
                request.duration_seconds,
                15.,
            )?;
            if request.destination == Destination::Replace {
                creative.sections.clear();
                creative.scene_prompts.clear();
                creative.scene_loras.clear();
                creative.scene_kinds.clear();
                commands.push(Edit::SetAudioTracks { tracks: vec![] });
                for s in &base.scenes {
                    commands.push(Edit::RemoveScene {
                        scene_id: s.id.clone(),
                    });
                }
            }
            let mut cursor = if request.destination == Destination::Replace {
                0.
            } else {
                base.scenes
                    .iter()
                    .map(|s| seconds(s.start) + seconds(s.duration))
                    .fold(0., f64::max)
            };
            let mut previous = if request.destination == Destination::Continue {
                base.scenes
                    .iter()
                    .max_by(|a, b| {
                        (seconds(a.start) + seconds(a.duration))
                            .total_cmp(&(seconds(b.start) + seconds(b.duration)))
                    })
                    .map(|s| s.id.clone())
            } else {
                None
            };
            for (i, (scene, len)) in scenes.into_iter().zip(lengths).enumerate() {
                let id = SceneId(format!("scene_{}_{}", draft.id, i));
                let start = time(cursor);
                let end = time(cursor + len);
                cursor += len;
                let actual_length = seconds(end) - seconds(start);
                let section_lengths = durations(
                    &scene.sections.iter().map(|s| s.weight).collect::<Vec<_>>(),
                    actual_length,
                    f64::INFINITY,
                )?;
                let mut local = 0.;
                let sections: Vec<Section> = scene
                    .sections
                    .iter()
                    .zip(section_lengths)
                    .enumerate()
                    .map(|(j, (s, length))| {
                        let from = time(local);
                        local += length;
                        let to = time(local);
                        Section {
                            id: SectionId(format!("section_{}_{}_{}", draft.id, i, j)),
                            start: from,
                            duration: Time {
                                num: to.num * from.den - from.num * to.den,
                                den: to.den * from.den,
                            }
                            .normalized(),
                            prompt: s.prompt.clone(),
                        }
                    })
                    .collect();
                let asset_for = |image_id: &str| {
                    request
                        .images
                        .iter()
                        .find(|im| im.id == image_id)
                        .map(|im| im.asset.clone())
                        .ok_or("story_image_missing")
                };
                let mut bindings = vec![];
                for (j, (section, source)) in sections.iter().zip(&scene.sections).enumerate() {
                    use imagine_video_contract::creative::{ImageBinding, SectionSource};
                    let mut images = vec![];
                    for image_id in scene.image_ids.iter().chain(&source.image_ids) {
                        images.push(ImageBinding {
                            anchor: None,
                            image_id: image_id.clone(),
                            asset: asset_for(image_id)?,
                            role: InputRole::VisualReference,
                        });
                    }
                    for (image_id, role) in [
                        (&source.first_frame_id, InputRole::FirstFrame),
                        (&source.last_frame_id, InputRole::LastFrame),
                    ] {
                        if !image_id.is_empty() {
                            let asset = asset_for(image_id)?;
                            images.push(ImageBinding {
                                anchor: None,
                                image_id: image_id.clone(),
                                asset: asset.clone(),
                                role: role.clone(),
                            });
                            if j == 0 && role == InputRole::FirstFrame {
                                bindings.push(Binding {
                                    id: BindingId(format!("binding_{}_first", section.id.0)),
                                    asset,
                                    role,
                                    enabled: true,
                                    video: None,
                                });
                            }
                        }
                    }
                    if !images.is_empty() || source.dialogue_intent.is_some() {
                        creative.sections.insert(
                            section.id.clone(),
                            SectionSource {
                                videos: vec![],
                                media_order: vec![],
                                images,
                                dialogue_intent: source.dialogue_intent.clone(),
                            },
                        );
                    }
                }
                let continuity = if scene.boundary_from_previous == "continuous_shot" {
                    Some(Continuity {
                        source_scene: previous.clone().ok_or("story_continuity_source_missing")?,
                        method: ContinuityMethod::LastFrameFirstFrame,
                        include_audio: false,
                    })
                } else {
                    None
                };
                creative.scene_prompts.insert(
                    id.clone(),
                    imagine_video_contract::creative::ScenePrompt {
                        title: scene.title.clone(),
                        summary: scene.summary.clone(),
                        ambient_sound: scene.ambient_sound.clone(),
                        non_diegetic_music: scene.non_diegetic_music.clone(),
                    },
                );
                let description = scene.description.clone();
                commands.push(Edit::AddScene {
                    scene: Box::new(Scene {
                        id: id.clone(),
                        start,
                        duration: Time {
                            num: end.num * start.den - start.num * end.den,
                            den: end.den * start.den,
                        }
                        .normalized(),
                        description,
                        model: ModelChoice::ProjectDefault,
                        settings: base
                            .scenes
                            .iter()
                            .find(|s| Some(&s.id) == request.scene_id.as_ref())
                            .or_else(|| base.scenes.first())
                            .map(|s| {
                                s.settings
                                    .iter()
                                    .filter(|slot| slot.model == base.default_model)
                                    .cloned()
                                    .collect()
                            })
                            .unwrap_or_default(),
                        sections,
                        bindings,
                        continuity,
                        accepted_take: None,
                    }),
                });
                previous = Some(id);
            }
            commands.push(Edit::SetCreative { creative });
        }
    }
    Ok(EditRequest {
        schema_version: 1,
        expected_project_revision: request.expected_project_revision,
        commands,
    })
}
pub fn system(snapshot: &profiles::Snapshot, key: &str) -> Result<String> {
    system_with_images(snapshot, key, &[], false)
}
pub fn system_with_images(
    snapshot: &profiles::Snapshot,
    key: &str,
    images: &[StoryImage],
    dialogue: bool,
) -> Result<String> {
    let text = snapshot
        .prompts
        .get(key)
        .ok_or("prompt_profile_unavailable")?;
    Ok(text.replace(
        "{{STORY_OUTLINE_JSON_SCHEMA}}",
        &super::visuals::schema(images, dialogue)?.to_string(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn director_weighted_minimum_and_cap_are_preserved() {
        assert_eq!(
            durations(&[1., 2.], 10., 15.).unwrap(),
            vec![3.4166666666666665, 6.583333333333334]
        );
        assert_eq!(durations(&[100., 1.], 20., 15.).unwrap(), vec![15., 5.]);
        assert_eq!(durations(&[1., 2.], 0.3, 15.).unwrap(), vec![0.15, 0.15]);
        assert!(durations(&[1., 1.], 31., 15.).is_err());
        assert!(durations(&[f64::NAN], 1., 15.).is_err());
    }
    #[test]
    fn expansion_rejects_reordered_or_reweighted_structure() {
        let text = r#"{"scenes":[{"title":"a","summary":"b","description":"c","ambient_sound":"","non_diegetic_music":"","weight":1,"boundary_from_previous":"hard_cut","setting_reference_matches":{},"sections":[{"prompt":"walk","weight":1}]}],"visual_placements":[]}"#;
        let board = outline(text).unwrap();
        let mut edited = board.scenes[0].clone();
        edited.sections[0].weight = 2.;
        assert!(expansion(&serde_json::to_string(&edited).unwrap(), &board.scenes[0]).is_err());
        edited = board.scenes[0].clone();
        edited.sections[0].prompt = "Own exact dialogue Hej!".into();
        assert!(expansion(&serde_json::to_string(&edited).unwrap(), &board.scenes[0]).is_ok());
    }
}
