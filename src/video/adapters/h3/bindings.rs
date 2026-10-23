//! Project-owned source projection. Compiling a binding does not claim media conditioning.
use super::prompt::{self, dialogue};
use crate::video::text::PythonText;
use crate::video::{contract::*, planning};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
#[path = "bindings/video.rs"]
mod video;
#[derive(Serialize)]
pub struct Plan {
    pub project_id: ProjectId,
    pub project_revision: u64,
    pub scene_id: SceneId,
    pub input_revision: String,
    pub prompt: String,
    pub metadata: Value,
    pub reference_manifest: Value,
    pub text_references: Vec<Value>,
    pub runtime_issues: Vec<String>,
    pub requested_duration_seconds: f64,
    pub effective_shape: crate::h3_protocol::Shape,
}
fn image(manifest: &mut Vec<Value>, asset: &AssetId) -> usize {
    if let Some(i) = manifest.iter().position(|v| v["asset_id"] == asset.0) {
        return i;
    }
    let i = manifest.len();
    manifest.push(json!({"kind":"image","native_input":"ref_images","native_key":format!("ref_image_{i}"),"prompt_label":format!("<Picture {}>",i+1),"asset_id":asset,"roles":[],"subject_labels":[],"section_ids":[],"internal_shot_indices":[],"crop_mode":"Project Default","source_crop_policy":null}));
    i
}
fn add_unique(v: &mut Value, key: &str, value: Value) {
    let a = v[key].as_array_mut().unwrap();
    if !a.contains(&value) {
        a.push(value);
    }
}
pub fn compile(project: &VideoProject, scene: &Scene) -> Result<Plan, Issue> {
    let mut ordered = scene.sections.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        (a.start.num as f64 / a.start.den as f64)
            .total_cmp(&(b.start.num as f64 / b.start.den as f64))
    });
    let seconds = |t: Time| t.num as f64 / t.den as f64;
    let requested_duration_seconds = seconds(scene.duration);
    let mut effective_shape = crate::h3_protocol::Selection {
        seconds: requested_duration_seconds,
        ..Default::default()
    }
    .resolve()
    .map_err(|code| Issue::new(code, "scene.duration"))?;
    effective_shape.width = project.delivery.width;
    effective_shape.height = project.delivery.height;
    let sections=ordered.iter().map(|s|json!({"item_id":s.id,"start_time":seconds(s.start),"end_time":seconds(s.start)+seconds(s.duration),"prompt":s.prompt})).collect::<Vec<_>>();
    let source = project
        .creative
        .scene_prompts
        .get(&scene.id)
        .cloned()
        .unwrap_or_default();
    let scene_prompt = json!({"summary":source.summary,"description":scene.description,"ambient_sound":source.ambient_sound,"non_diegetic_music":source.non_diegetic_music});
    let options = project.creative.global_prompt.as_ref();
    let global_enabled = options.map_or(!project.global_prompt.is_empty(), |o| o.enabled);
    let global = json!({"enabled":global_enabled,"position":options.map_or("Prefix",|o|o.position.as_str()),"prompt":project.global_prompt});
    let mut usage: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut usage_order = vec![];
    let mut remember = |text: &str, shots: &[usize]| {
        for token in imagine_video_contract::references::tokens(text) {
            if !usage.contains_key(&token.alias) {
                usage_order.push(token.alias.clone());
            }
            let entry = usage.entry(token.alias).or_default();
            for shot in shots {
                if !entry.contains(shot) {
                    entry.push(*shot);
                }
            }
        }
    };
    let mut speakers = BTreeMap::new();
    let mut speaker = |text: &str| {
        for cue in dialogue::parse(text) {
            if !speakers.contains_key(&cue.alias) {
                let id = format!("S{}", speakers.len() + 1);
                speakers.insert(cue.alias, id);
            }
        }
    };
    for (i, s) in ordered.iter().enumerate() {
        remember(&s.prompt, &[i]);
        speaker(&s.prompt);
    }
    remember(
        if global_enabled {
            &project.global_prompt
        } else {
            ""
        },
        &(0..ordered.len()).collect::<Vec<_>>(),
    );
    for field in [
        "summary",
        "description",
        "ambient_sound",
        "non_diegetic_music",
    ] {
        remember(scene_prompt[field].as_str().unwrap_or(""), &[]);
    }
    let i2v = project.effective_model(scene).variant == "i2v";
    if i2v {
        speaker(&project.global_prompt);
        for field in [
            "summary",
            "description",
            "ambient_sound",
            "non_diegetic_music",
        ] {
            speaker(scene_prompt[field].as_str().unwrap_or(""));
        }
    }
    let mut pictures = vec![];
    let mut subjects = vec![];
    let mut audios = vec![];
    let mut text_references = vec![];
    let mut issues = BTreeSet::new();
    if i2v && !imagine_video_contract::creative::video_bindings(project, scene).is_empty() {
        issues.insert("h3_i2v_video_reference_unavailable".into());
    }
    let refs = if i2v {
        usage_order
            .iter()
            .filter_map(|alias| {
                project
                    .creative
                    .references
                    .iter()
                    .rev()
                    .find(|r| r.alias.trim_start_matches('@').eq_ignore_ascii_case(alias))
            })
            .filter(|r| r.enabled && project.creative.visual_references_enabled != Some(false))
            .collect::<Vec<_>>()
    } else {
        imagine_video_contract::creative::selected_references(
            &project.creative,
            &usage.keys().cloned().collect(),
        )
    };
    for r in &refs {
        let alias = r.alias.trim_start_matches('@').to_ascii_lowercase();
        let n = subjects.len() + 1;
        let label = format!("<Subject {n}>");
        let mut images = vec![];
        if !i2v {
            for im in r.images.iter().filter(|im| im.enabled) {
                let i = image(&mut pictures, &im.asset);
                add_unique(&mut pictures[i], "roles", json!("subject_source"));
                add_unique(&mut pictures[i], "subject_labels", json!(label));
                images.push(json!({"prompt_label":pictures[i]["prompt_label"],"native_key":pictures[i]["native_key"],"asset_id":im.asset,"contribution":im.contribution.pytrim()}));
            }
        }
        let voice = r.voice.as_ref();
        let language = voice
            .map(|v| v.language.pytrim())
            .filter(|s| !s.is_empty())
            .unwrap_or("English");
        let profile = voice.map(|v| v.description.pytrim()).unwrap_or("");
        let shots = usage.get(&alias).cloned().unwrap_or_default();
        let relationship = r.relationship.as_ref().map(|v|json!(v)).unwrap_or_else(||json!({"marker":if r.retention.is_empty(){"fully_preserved"}else{&r.retention},"note":"","target_reference_id":null}));
        subjects.push(json!({"kind":"subject","prompt_label":label,"reference_id":r.id,"source_alias":alias,"reference_kind":r.kind.to_ascii_lowercase(),"description":r.description.pytrim(),"default_language":language,"voice_profile":profile,"speaker_id":speakers.get(&alias),"relationship":relationship,"source_images":images,"section_ids":shots.iter().map(|i|&ordered[*i].id).collect::<Vec<_>>(),"internal_shot_indices":shots}));
        text_references.push(json!({"reference_id":r.id,"alias":alias,"description":r.description.pytrim(),"reference_kind":r.kind.to_ascii_lowercase(),"default_language":language,"voice_profile":profile,"speaker_id":speakers.get(&alias),"section_ids":shots.iter().map(|i|&ordered[*i].id).collect::<Vec<_>>(),"internal_shot_indices":shots,"attached_image_count":r.images.iter().filter(|im|im.enabled).count(),"attached_voice_audio_count":voice.map_or(0,|v|v.assets.iter().filter(|a|v.reference_settings.get(*a).is_none_or(|s|s.enabled)).count()),"conditioning":"description_only"}));
        if !i2v && speakers.contains_key(&alias) {
            for asset in voice.into_iter().flat_map(|v| &v.assets) {
                let settings = voice
                    .and_then(|v| v.reference_settings.get(asset))
                    .cloned()
                    .unwrap_or_default();
                if !settings.enabled {
                    continue;
                }
                let i = audios.len();
                audios.push(json!({"kind":"audio","role":"character_voice_timbre_reference","purpose":"voice_timbre","native_input":"ref_audios","native_key":format!("ref_audio_{i}"),"prompt_label":format!("<Audio {}>",i+1),"asset_id":asset,"target_reference_id":r.id,"target_subject_label":label,"target_speaker_id":speakers[&alias],"paired_video_label":null,"voice_description":settings.voice_description.pytrim(),"contribution":settings.contribution.pytrim(),"strength":settings.strength,"retention_marker":settings.relationship}));
            }
        }
    }
    let videos = if i2v {
        vec![]
    } else {
        video::entries(project, scene)
    };
    audios.extend(video::paired(&videos, audios.len()));
    if !i2v {
        let mode = scene
            .settings
            .iter()
            .find(|slot| &slot.model == project.effective_model(scene))
            .and_then(|slot| slot.values.get("audio_reference_mode"))
            .and_then(Value::as_str)
            .unwrap_or("Timeline Audio References");
        // Director's _choice normalizes unknown values to Timeline Audio References.
        if mode != "Disabled" {
            for mut audio in imagine_video_contract::audio::references::entries(project, scene) {
                let index = audios
                    .iter()
                    .filter(|a| a["paired_video_key"].is_null())
                    .count();
                audio["native_key"] = json!(format!("ref_audio_{index}"));
                audio["prompt_label"] = json!(format!("<Audio {}>", index + 1));
                if let Some(target) = subjects
                    .iter()
                    .find(|s| s["reference_id"] == audio["target_reference_id"])
                {
                    audio["target_subject_label"] = target["prompt_label"].clone();
                    audio["target_speaker_id"] = target["speaker_id"].clone();
                }
                audios.push(audio);
            }
            if imagine_video_contract::audio::generation_clips(project, scene)
                .iter()
                .any(|c| c.asset.is_none())
            {
                issues.insert("audio_reference_missing".into());
            }
        }
    }
    text_references.sort_by_key(|r| {
        usage_order
            .iter()
            .position(|a| r["alias"].as_str() == Some(a))
            .unwrap_or(usize::MAX)
    });
    let mut keyframes = json!({"first":null,"last":null});
    for (i, section) in ordered.iter().enumerate() {
        let mut bindings = project
            .creative
            .sections
            .get(&section.id)
            .map(|s| {
                s.images
                    .iter()
                    .map(|b| (b.asset.clone(), b.role.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if i == 0 {
            for b in scene.bindings.iter().filter(|b| b.enabled) {
                if !bindings.contains(&(b.asset.clone(), b.role.clone())) {
                    bindings.push((b.asset.clone(), b.role.clone()));
                }
            }
        }
        for (asset, role) in bindings {
            if role == InputRole::VideoReference
                && !i2v
                && project
                    .assets
                    .get(&asset)
                    .is_some_and(|a| a.kind == MediaKind::Video)
            {
                continue;
            }
            if !matches!(
                role,
                InputRole::FirstFrame | InputRole::LastFrame | InputRole::VisualReference
            ) {
                issues.insert("h3_section_reference_unavailable".into());
                continue;
            }
            if i == 0 && role == InputRole::FirstFrame {
                keyframes["first"] = json!({"asset_id":asset});
            }
            if role == InputRole::LastFrame {
                keyframes["last"] = json!({"asset_id":asset});
            }
            if i2v && (i != 0 || role != InputRole::FirstFrame) {
                issues.insert("h3_section_reference_unavailable".into());
            }
            let p = image(&mut pictures, &asset);
            let role = match role {
                InputRole::FirstFrame => "first_frame",
                InputRole::LastFrame => "last_frame",
                _ => "influence",
            };
            add_unique(&mut pictures[p], "roles", json!("section_picture"));
            if pictures[p].get("section_bindings").is_none() {
                pictures[p]["section_bindings"] = json!([]);
            }
            pictures[p]["section_bindings"].as_array_mut().unwrap().push(json!({"binding_id":null,"section_id":section.id,"internal_shot_index":i,"authored_role":role,"resolved_role":role,"custom_role":"","relationship":{"marker":"fully_preserved","note":"","target_reference_id":null},"source_reference_id":null,"continuation_conflict":false}));
            add_unique(&mut pictures[p], "section_ids", json!(section.id));
            add_unique(&mut pictures[p], "internal_shot_indices", json!(i));
        }
    }
    if let Some(c) = &scene.continuity
        && c.method == ContinuityMethod::LastFrameFirstFrame
    {
        keyframes["first"] = json!({"continuity_from":c.source_scene});
    }
    let motion = scene.continuity.as_ref().filter(|c| {
        matches!(
            c.method,
            ContinuityMethod::MotionContext | ContinuityMethod::MotionContextExp
        )
    });
    let dialogue_cues = ordered
        .iter()
        .enumerate()
        .flat_map(|(index, section)| {
            dialogue::parse(&section.prompt)
                .into_iter()
                .map(move |cue| {
                    let mut value = serde_json::to_value(&cue).unwrap();
                    value["section_id"] = json!(section.id);
                    value["internal_shot_index"] = json!(index);
                    value
                })
        })
        .map(|mut cue| {
            cue["speaker_id"] = json!(speakers.get(cue["alias"].as_str().unwrap_or("")));
            cue
        })
        .collect::<Vec<_>>();
    let mut audio_index = 1;
    for video in &videos {
        if let Some(audio) = audios
            .iter_mut()
            .find(|a| a["paired_video_key"] == video["native_key"])
        {
            audio["prompt_label"] = json!(format!("<Audio {audio_index}>"));
            audio_index += 1;
        }
    }
    for audio in audios
        .iter_mut()
        .filter(|a| a["paired_video_key"].is_null())
    {
        audio["prompt_label"] = json!(format!("<Audio {audio_index}>"));
        audio_index += 1;
    }
    let mut presentation_order = pictures
        .iter()
        .map(|p| json!({"kind":"image","label":p["prompt_label"]}))
        .collect::<Vec<_>>();
    for video in &videos {
        if let Some(audio) = audios
            .iter()
            .find(|a| a["paired_video_key"] == video["native_key"])
        {
            presentation_order.push(json!({"kind":"audio","label":audio["prompt_label"]}));
        }
        presentation_order.push(json!({"kind":"video","label":video["prompt_label"]}));
    }
    presentation_order.extend(
        audios
            .iter()
            .filter(|a| a["paired_video_key"].is_null())
            .map(|a| json!({"kind":"audio","label":a["prompt_label"]})),
    );
    let paired_count = audios
        .iter()
        .filter(|a| !a["paired_video_key"].is_null())
        .count();
    let manifest = json!({"mode":"Reference","frame0_guide":{"enabled":false},"subjects":subjects,"images":pictures,"videos":videos,"audios":audios,
        "motion_context":if motion.is_some(){json!({"enabled":true,"include_audio":motion.is_some_and(|c|c.include_audio),"context_frames":0})}else{json!({"enabled":false})},
        "presentation_order":presentation_order,"dialogue":{"cues":dialogue_cues,"speaker_by_alias":speakers},
        "counts":{"images":pictures.len(),"subjects":subjects.len(),"videos":videos.len(),"paired_video_audios":paired_count,"standalone_audios":audios.len()-paired_count,"audios":audios.len(),"files":pictures.len()+videos.len()+audios.len()},
        "helto_limits":{"images":9,"videos":null,"audios":null,"files":null}});

    let input = prompt::Input {
        sections,
        reference_manifest: manifest.clone(),
        global_prompt: global,
        scene_prompt,
        text_references: text_references.clone(),
        keyframes,
        effective_duration_seconds: effective_shape.seconds(),
        ..Default::default()
    };
    let output = if i2v {
        prompt::compile_i2v(&input)
    } else {
        prompt::compile_reference(&input)
    };
    Ok(Plan {
        project_id: project.id.clone(),
        project_revision: project.project_revision,
        scene_id: scene.id.clone(),
        input_revision: planning::input_revision(project, &scene.id)?,
        prompt: output.prompt,
        metadata: output.metadata,
        reference_manifest: manifest,
        text_references,
        runtime_issues: issues.into_iter().collect(),
        requested_duration_seconds,
        effective_shape,
    })
}
