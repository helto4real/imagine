//! Pure Story validation, timing and revision-bound edit planning.
use super::profiles;
use crate::video::{
    contract::*,
    editing::{Edit, EditRequest},
};
pub use imagine_video_contract::authoring::*;
type Result<T> = std::result::Result<T, &'static str>;
/// The longest Scene a Story may plan; the outline message states it too.
pub const MAX_SCENE_SECONDS: f64 = 15.;
pub const MAX_SCENES: usize = 128;
pub const MAX_SECTIONS_PER_SCENE: usize = 12;
pub const BOUNDARIES: [&str; 2] = ["hard_cut", "continuous_shot"];
/// What `auto` continuity resolves to for a target variant; `reference`
/// keeps the Story's motion carry-over.
pub fn auto_continuity(variant: &str) -> Option<ContinuityMethod> {
    match variant {
        "i2v" => Some(ContinuityMethod::LastFrameFirstFrame),
        "reference" => Some(ContinuityMethod::MotionContextExp),
        _ => None,
    }
}

pub fn validate(board: &Storyboard) -> Result<()> {
    if board.scenes.is_empty() || board.scenes.len() > MAX_SCENES {
        return Err("invalid_storyboard");
    }
    let mut ids = std::collections::BTreeSet::new();
    for scene in &board.scenes {
        if scene.id.is_empty()
            || !ids.insert(scene.id.clone())
            || !scene.weight.is_finite()
            || scene.weight <= 0.
            || scene.sections.is_empty()
            || scene.sections.len() > MAX_SECTIONS_PER_SCENE
            || !BOUNDARIES.contains(&scene.boundary_from_previous.as_str())
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
/// A storyboard Scene as edited in review. Unlike a model's expansion, the cut
/// and the image assignments may change, as in Director; the Scene's and
/// Sections' identity, weights and dialogue participants may not.
pub fn reviewed(scene: &StoryScene, source: &StoryScene, images: &[StoryImage]) -> Result<()> {
    validate(&Storyboard {
        scenes: vec![scene.clone()],
    })?;
    if scene.id != source.id
        || scene.weight != source.weight
        || scene.sections.len() != source.sections.len()
        || scene.sections.iter().zip(&source.sections).any(|(a, b)| {
            a.id != b.id
                || a.weight != b.weight
                || a.dialogue_intent.as_ref().map(|d| &d.participants)
                    != b.dialogue_intent.as_ref().map(|d| &d.participants)
        })
    {
        return Err("story_expansion_changed_structure");
    }
    let known = |id: &String| id.is_empty() || images.iter().any(|i| &i.id == id);
    if !scene.image_ids.iter().all(known)
        || !scene.sections.iter().all(|s| {
            s.image_ids.iter().all(known) && known(&s.first_frame_id) && known(&s.last_frame_id)
        })
    {
        return Err("story_image_missing");
    }
    Ok(())
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
/// One Scene's production choice, resolved against the target variant.
pub struct Planned {
    /// None keeps the project default.
    pub model: Option<ModelSelection>,
    /// The scene's preset; none on the provider path without a production block.
    pub preset: Option<crate::video::presets::Preset>,
    pub i2v: bool,
    pub continuity: Option<(ContinuityMethod, bool)>,
}
/// A production error and the Scene and Section it concerns (0-based).
#[derive(Debug, PartialEq)]
pub struct Misplaced {
    pub code: &'static str,
    pub scene: Option<usize>,
    pub section: Option<usize>,
}
/// Resolves the Draft's production block for its expanded Scenes. Without
/// one, Scenes keep the project default and continuous shots carry motion.
pub fn production(
    base: &VideoProject,
    draft: &Draft,
    presets: &crate::video::presets::Catalog,
) -> std::result::Result<Vec<Planned>, Misplaced> {
    let misplaced = |code, scene, section| Misplaced {
        code,
        scene,
        section,
    };
    let scenes = draft
        .scenes
        .iter()
        .cloned()
        .collect::<Option<Vec<_>>>()
        .ok_or(misplaced("story_not_expanded", None, None))?;
    // Only a continued timeline gives the first Scene a shot to continue.
    let source = draft.request.destination == Destination::Continue && !base.scenes.is_empty();
    if !source
        && scenes
            .first()
            .is_some_and(|s| s.boundary_from_previous == "continuous_shot")
    {
        return Err(misplaced("story_continuity_source_missing", None, None));
    }
    let Some(production) = &draft.production else {
        return Ok(scenes
            .iter()
            .map(|s| Planned {
                model: None,
                preset: None,
                i2v: false,
                continuity: (s.boundary_from_previous == "continuous_shot")
                    .then_some((ContinuityMethod::MotionContextExp, false)),
            })
            .collect());
    };
    if production.scenes.len() != scenes.len() {
        return Err(misplaced("production_scene_count_mismatch", None, None));
    }
    let variants = crate::video::adapters::h3::variants();
    let mut out = vec![];
    for (i, (scene, choice)) in scenes.iter().zip(&production.scenes).enumerate() {
        let at = |code, section| misplaced(code, Some(i), section);
        let preset = match (&choice.preset, choice.model) {
            (Some(key), model) => {
                let preset = presets.get(key).ok_or(at("preset_unknown", None))?;
                if let Some(code) = preset.problem {
                    return Err(at(code, None));
                }
                if model.is_some_and(|m| m.variant() != preset.variant) {
                    return Err(at("production_preset_variant_mismatch", None));
                }
                preset
            }
            (None, Some(model)) => presets
                .get(crate::video::presets::default_key(model.variant()))
                .ok_or(at("preset_unknown", None))?,
            (None, None) => return Err(at("production_model_required", None)),
        };
        let (model, capabilities) = variants
            .iter()
            .find(|(m, _)| m.variant == preset.variant)
            .ok_or(at("scene_runtime_unsupported", None))?;
        let i2v = model.variant == "i2v";
        let continuity = match (&choice.continuity, scene.boundary_from_previous.as_str()) {
            (choice, "continuous_shot") => {
                let method = choice
                    .as_ref()
                    .and_then(|c| c.method.clone())
                    .or_else(|| auto_continuity(&model.variant))
                    .filter(|m| capabilities.continuity.contains(m))
                    .ok_or(at("continuity_method_unsupported", None))?;
                let include_audio = choice.as_ref().is_some_and(|c| c.include_audio);
                if method == ContinuityMethod::LastFrameFirstFrame {
                    if include_audio {
                        return Err(at("continuity_endpoint_incompatible", None));
                    }
                    if !scene.sections[0].first_frame_id.is_empty() {
                        return Err(at("continuity_first_frame_conflict", Some(0)));
                    }
                }
                Some((method, include_audio))
            }
            (Some(_), _) => return Err(at("continuity_requires_continuous_shot", None)),
            (None, _) => None,
        };
        if i2v {
            // One opening First Frame is I2V's only image input; a start
            // frame paired with itself is that frame alone.
            if !scene.image_ids.is_empty() {
                return Err(at("i2v_input_role_unsupported", Some(0)));
            }
            for (j, s) in scene.sections.iter().enumerate() {
                let opening = j == 0 && s.last_frame_id == s.first_frame_id;
                if !s.image_ids.is_empty()
                    || (j > 0 && !s.first_frame_id.is_empty())
                    || (!s.last_frame_id.is_empty() && !opening)
                {
                    return Err(at("i2v_input_role_unsupported", Some(j)));
                }
            }
            if scene.sections[0].first_frame_id.is_empty()
                && continuity
                    .as_ref()
                    .is_none_or(|(m, _)| *m != ContinuityMethod::LastFrameFirstFrame)
            {
                return Err(at("i2v_start_frame_required", None));
            }
        }
        out.push(Planned {
            model: (model != &base.default_model).then(|| model.clone()),
            preset: Some(preset.clone()),
            i2v,
            continuity,
        });
    }
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
pub fn apply(
    base: &VideoProject,
    draft: &Draft,
    presets: &crate::video::presets::Catalog,
) -> Result<EditRequest> {
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
        Operation::EnhanceSceneAll => {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or("scene_not_found")?)
                .map_err(|_| "scene_not_found")?;
            let mut prompts = false;
            for row in draft.fields.iter().filter(|f| f.include) {
                let Some(text) = row.text.clone().filter(|t| !t.trim().is_empty()) else {
                    continue;
                };
                match &row.target {
                    FieldTarget::Scene {
                        field: SceneField::Description,
                    } => commands.push(Edit::SetSceneDescription {
                        scene_id: scene.id.clone(),
                        description: text,
                    }),
                    FieldTarget::Scene { field } => {
                        let source = creative.scene_prompts.entry(scene.id.clone()).or_default();
                        match field {
                            SceneField::Summary => source.summary = text,
                            SceneField::AmbientSound => source.ambient_sound = text,
                            SceneField::NonDiegeticMusic => source.non_diegetic_music = text,
                            SceneField::Description => unreachable!(),
                        }
                        prompts = true;
                    }
                    FieldTarget::Section { section_id } => {
                        let mut section = scene
                            .sections
                            .iter()
                            .find(|s| &s.id == section_id)
                            .cloned()
                            .ok_or("section_not_found")?;
                        section.prompt = text;
                        commands.push(Edit::UpdateSection {
                            scene_id: scene.id.clone(),
                            section,
                        });
                    }
                }
            }
            if prompts {
                commands.push(Edit::SetCreative { creative });
            }
            if commands.is_empty() {
                return Err("authoring_empty_draft");
            }
        }
        Operation::Animate => {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or("scene_not_found")?)
                .map_err(|_| "scene_not_found")?;
            let proposal = draft.animate.clone().ok_or("authoring_empty_draft")?;
            let mut section = scene
                .sections
                .first()
                .cloned()
                .ok_or("animate_scene_invalid")?;
            section.prompt = proposal.prompt;
            commands.push(Edit::SetSceneDescription {
                scene_id: scene.id.clone(),
                description: proposal.description,
            });
            commands.push(Edit::UpdateSection {
                scene_id: scene.id.clone(),
                section,
            });
            creative
                .scene_prompts
                .entry(scene.id.clone())
                .or_default()
                .ambient_sound = proposal.ambient_sound;
            commands.push(Edit::SetCreative { creative });
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
            let planned = production(base, draft, presets).map_err(|m| m.code)?;
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
                MAX_SCENE_SECONDS,
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
            // Cuts between the new Scenes, added once all of them exist.
            let mut commands_after = vec![];
            for (i, ((scene, len), plan)) in
                scenes.into_iter().zip(lengths).zip(planned).enumerate()
            {
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
                        // I2V opens on the frame alone.
                        let opening = plan.i2v
                            && role == InputRole::LastFrame
                            && image_id == &source.first_frame_id;
                        if !image_id.is_empty() && !opening {
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
                // As a Continuous shot chosen on the Timeline: the cut is
                // continuous and the previous Scene's motion carries over.
                let continuous = scene.boundary_from_previous == "continuous_shot";
                let continuity = if continuous {
                    let source_scene = previous.clone().ok_or("story_continuity_source_missing")?;
                    commands_after.push(Edit::SetAssemblyBoundary {
                        boundary: imagine_video_contract::assembly::Boundary {
                            mode: imagine_video_contract::assembly::BoundaryMode::ContinuousShot,
                            ..imagine_video_contract::assembly::Boundary::new(
                                source_scene.clone(),
                                id.clone(),
                            )
                        },
                    });
                    let (method, include_audio) = plan
                        .continuity
                        .clone()
                        .ok_or("story_continuity_source_missing")?;
                    Some(Continuity {
                        source_scene,
                        method,
                        include_audio,
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
                // The same scene change the UI applies; recorded for Modified detection.
                let change = plan
                    .preset
                    .as_ref()
                    .map(|preset| preset.scene_change(base))
                    .transpose()?;
                let preset_settings = change.as_ref().and_then(|c| c.settings.clone());
                if let (Some(change), Some(preset)) = (&change, &plan.preset) {
                    if let Some(loras) = &change.loras {
                        creative.scene_loras.insert(id.clone(), loras.clone());
                    }
                    creative
                        .scene_presets
                        .insert(id.clone(), change.applied(&preset.key));
                }
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
                        model: plan
                            .model
                            .clone()
                            .map_or(ModelChoice::ProjectDefault, ModelChoice::Override),
                        settings: match (&preset_settings, &change) {
                            (Some(slot), _) => vec![slot.clone()],
                            // A preset without settings inherits the project's settings.
                            (None, Some(_)) => vec![],
                            (None, None) => base
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
                        },
                        sections,
                        bindings,
                        continuity,
                        accepted_take: None,
                    }),
                });
                if let Some(model) = plan.model.filter(|_| preset_settings.is_none()) {
                    commands.push(Edit::InheritModelSettings {
                        scene_id: id.clone(),
                        model,
                    });
                }
                previous = Some(id);
            }
            commands.append(&mut commands_after);
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
/// A system prompt whose outline schema follows `contract`.
pub fn system_for(
    snapshot: &profiles::Snapshot,
    key: &str,
    contract: &super::director::Contract,
) -> Result<String> {
    let text = snapshot
        .prompts
        .get(key)
        .ok_or("prompt_profile_unavailable")?;
    Ok(text.replace(
        "{{STORY_OUTLINE_JSON_SCHEMA}}",
        &super::director::schema_text(contract),
    ))
}
pub fn system_with_images(
    snapshot: &profiles::Snapshot,
    key: &str,
    images: &[StoryImage],
    dialogue: bool,
) -> Result<String> {
    system_for(snapshot, key, &super::visuals::contract(images, dialogue)?)
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
