use super::{Input, Output, common::*, dialogue, f, manifest_text::*, s, truth};
use crate::video::text::PythonText;
use serde_json::{Value, json};
fn label(v: &Value) -> &str {
    s(v, "prompt_label")
}
fn fallback<'a>(s: &'a str, default: &'a str) -> &'a str {
    if s.is_empty() { default } else { s }
}
fn boundary(video: Option<&Value>, picture: Option<&Value>, guide: &Value) -> String {
    let empty = Value::Null;
    let v = video.unwrap_or(&empty);
    let p = picture.unwrap_or(&empty);
    let l = fallback(label(v), "<Video 1>");
    let mode = fallback(
        s(v, "boundary_mode"),
        fallback(
            s(p, "boundary_mode"),
            fallback(s(guide, "boundary_mode"), "Hard Cut"),
        ),
    );
    if truth(&guide["enabled"]) {
        let mut opening = vec![
            "The first frame of [Shot 1] is anchored to the previous Scene's final frame.".into(),
        ];
        opening.push(if video.is_some(){format!("{l} provides subject motion, motion timing, camera trajectory, and framing development after the anchored opening frame.")}else{"Continue forward from that anchored state without replaying earlier motion.".into()});
        let t = fallback(s(v, "transition_prompt"), s(guide, "transition_prompt")).pytrim();
        if mode == "Transition" && !t.is_empty() {
            opening.push(t.into());
        }
        return opening.join(" ");
    }
    if picture.is_some() {
        let mut opening = vec![format!(
            "The shot opens from {}.",
            fallback(label(p), "<Picture 1>")
        )];
        if video.is_some() {
            opening.push(format!("{l} provides subject motion, motion timing, camera trajectory, and framing development after the opening."));
            let t = s(v, "transition_prompt").pytrim();
            if mode == "Transition" && !t.is_empty() {
                opening.push(t.into());
            }
        }
        return opening.join(" ");
    }
    if mode == "Continuous Shot" {
        return format!(
            "The target video is a seamless continuation of {l}. The first frame of [Shot 1] is the last frame of {l}."
        );
    }
    let opening = match mode {
        "Blend Seam" => format!(
            "The opening continues the final composition, lighting, subject state, and motion of {l}."
        ),
        "Transition" => {
            let t = s(v, "transition_prompt").pytrim();
            let mut o = format!("Continuing from the ending of {l}");
            if !t.is_empty() {
                o = format!("{o}, {t}");
            }
            if !o.ends_with(['.', '!', '?']) {
                o.push('.');
            }
            o
        }
        _ => format!("The target video continues directly from the ending of {l}."),
    };
    let mut parts = vec![opening];
    if v["reuse_visual_references"] != false {
        parts.push("Visible subject identities, appearance, object state, and spatial relationships remain consistent at the seam until the authored action changes them.".into());
    }
    if v["reuse_style"] != false {
        parts.push("The opening preserves the established lighting, color, camera language, and visual style.".into());
    }
    parts.join(" ")
}
pub(super) fn compile(input: &Input) -> Output {
    let manifest = &input.reference_manifest;
    let subjects = list(manifest, "subjects");
    let pictures = list(manifest, "images");
    let videos = list(manifest, "videos");
    let audios = list(manifest, "audios");
    let mut maps = dialogue::Maps::default();
    for subject in subjects {
        let a = s(subject, "source_alias").to_lowercase();
        if !label(subject).is_empty() {
            maps.subject.insert(a.clone(), label(subject).into());
        }
        if !s(subject, "speaker_id").is_empty() {
            maps.speaker
                .insert(a.clone(), s(subject, "speaker_id").into());
        }
        maps.language.insert(
            a.clone(),
            fallback(s(subject, "default_language"), "English").into(),
        );
        if !s(subject, "voice_profile").pytrim().is_empty() {
            maps.voice_profile
                .insert(a, s(subject, "voice_profile").pytrim().into());
        }
    }
    for subject in subjects {
        let labels = dedupe(
            audios
                .iter()
                .filter(|a| {
                    s(a, "purpose") == "voice_timbre"
                        && a["target_reference_id"] == subject["reference_id"]
                        && !label(a).is_empty()
                })
                .map(|a| label(a).into()),
        );
        if !labels.is_empty() {
            maps.voice_audio
                .insert(s(subject, "source_alias").to_lowercase(), english(&labels));
        }
    }
    let boundary_video = videos
        .iter()
        .find(|v| s(v, "role") == "boundary_continuation");
    let boundary_picture = pictures
        .iter()
        .find(|p| s(p, "role") == "boundary_first_frame");
    let boundary_audio = audios.iter().find(|a| truth(&a["boundary_continuation"]));
    let motion = &manifest["motion_context"];
    let guide = &manifest["frame0_guide"];
    let motion_enabled = truth(&motion["enabled"]);
    let guide_enabled = truth(&guide["enabled"]);
    let replace = |v: &str| replace(v, &maps.subject);
    let summary = replace(&single(s(&input.scene_prompt, "summary")));
    let description = replace(&single(s(&input.scene_prompt, "description")));
    let ambient = replace(&single(s(&input.scene_prompt, "ambient_sound")));
    let authored_music = replace(&single(s(&input.scene_prompt, "non_diegetic_music")));
    let mut sound = vec![if ambient.is_empty() {
        AMBIENT.into()
    } else {
        ambient.clone()
    }];
    let mut music = if authored_music.is_empty() {
        vec![]
    } else {
        vec![authored_music.clone()]
    };
    if let Some(audio) = boundary_audio.filter(|a| !label(a).is_empty()) {
        sound.push(format!(
            "The opening continues the ambience and physical sound character of {}.",
            label(audio)
        ));
    } else if motion_enabled && truth(&motion["include_audio"]) {
        sound.push(
            "The opening continues the established ambience and physical sound character.".into(),
        );
    }
    for a in audios {
        let l = label(a);
        if l.is_empty() || truth(&a["boundary_continuation"]) {
            continue;
        }
        let verb = if matches!(s(a, "retention_marker"), "fully_copy" | "partially_copy") {
            "reuses"
        } else {
            "references"
        };
        match s(a, "purpose") {
            "ambience_sfx" | "beat_rhythm" | "continuity" => sound.push(format!(
                "The target video's ambience and physical sound character {verb} {l}."
            )),
            "music_style" => music.push(format!(
                "The audience-only score {verb} the music style of {l}."
            )),
            _ => {}
        }
    }
    let (global_visual, global_audio, global_music) =
        channels(if truth(&input.global_prompt["enabled"]) {
            s(&input.global_prompt, "prompt").pytrim()
        } else {
            ""
        });
    let global_visual = replace(&global_visual);
    sound.extend(global_audio.iter().map(|v| replace(v)));
    music.extend(global_music.iter().map(|v| replace(v)));
    let suffix = s(&input.global_prompt, "position") == "Suffix";
    let mut detail = vec![];
    for (index, section) in input.sections.iter().enumerate() {
        let (v, a, m) = channels(s(section, "prompt").pytrim());
        let mut visual = replace(&dialogue::compile(&v, &maps));
        if !global_visual.is_empty() && index == 0 && !suffix {
            visual = join(&global_visual, &visual);
        }
        if !global_visual.is_empty() && index + 1 == input.sections.len() && suffix {
            visual = join(&visual, &global_visual);
        }
        for p in pictures {
            for b in list(p, "section_bindings")
                .iter()
                .filter(|b| !b["section_id"].is_null() && b["section_id"] == section["item_id"])
            {
                visual = join(&picture_instruction(p, b), &visual);
            }
        }
        for v in videos
            .iter()
            .filter(|v| truth(&v["section_id"]) && v["section_id"] == section["item_id"])
        {
            let custom = s(v, "custom_role").pytrim().trim_end_matches('.');
            let instruction = match s(v, "role") {
                "source_video_edit" => format!(
                    "Directly edit {} according to the authored changes in this shot.",
                    label(v)
                ),
                "custom_video_reference" if !custom.is_empty() => {
                    format!("{} provides {custom}.", label(v))
                }
                _ => format!(
                    "{} provides the motion, camera, rhythm, and temporal-structure reference.",
                    label(v)
                ),
            };
            visual = join(&instruction, &visual);
        }
        for a in audios.iter().filter(|a| {
            s(a, "role") == "standalone_timeline_audio_reference"
                && f(a, "end_time") > f(section, "start_time")
                && f(a, "start_time") < f(section, "end_time")
        }) {
            let l = label(a);
            let description = s(a, "description").pytrim().trim_end_matches('.');
            if s(a, "purpose") == "dialogue_lyrics" && !l.is_empty() {
                visual = join(
                    &visual,
                    &format!(
                        "The dialogue or lyrics during this shot reference the content of {l}."
                    ),
                );
            } else if s(a, "purpose") == "custom" && !l.is_empty() && !description.is_empty() {
                visual = join(
                    &visual,
                    &format!("During this shot, {l} provides {description}."),
                );
            }
        }
        if visual.is_empty() {
            visual =
                "Continue the established action, composition, lighting, and spatial continuity."
                    .into();
        }
        if index == 0
            && (boundary_video.is_some()
                || boundary_picture.is_some()
                || motion_enabled
                || guide_enabled)
        {
            visual = join(
                &if motion_enabled {
                    "The opening seamlessly continues the established motion, camera trajectory, subject state, and action.".into()
                } else {
                    boundary(boundary_video, boundary_picture, guide)
                },
                &visual,
            );
        }
        let marker = if index == 0 {
            "[Shot 1] Starting shot:".into()
        } else {
            format!(
                "[Shot {}] At {},",
                index + 1,
                timestamp(f(section, "start_time"))
            )
        };
        detail.push(format!("{marker} {visual}"));
        sound.extend(a.iter().map(|v| replace(v)));
        music.extend(m.iter().map(|v| replace(v)));
    }
    if detail.is_empty() {
        detail.push(format!(
            "[Shot 1] Starting shot: {}",
            fallback(&global_visual, "Continue the established visual direction.")
        ));
    }
    let edits = videos
        .iter()
        .filter(|v| s(v, "role") == "source_video_edit")
        .collect::<Vec<_>>();
    let mut tasks = vec![if edits.is_empty() {
        "reference generation"
    } else {
        "video editing"
    }];
    let beyond = !subjects.iter().all(|s| label(s).is_empty())
        || videos.iter().any(|v| {
            !label(v).is_empty()
                && !matches!(s(v, "role"), "source_video_edit" | "boundary_continuation")
        })
        || pictures.iter().any(|p| {
            !label(p).is_empty()
                && (truth(&p["subject_labels"])
                    || list(p, "section_bindings")
                        .iter()
                        .any(|b| matches!(s(b, "resolved_role"), "influence" | "custom_influence")))
        });
    if !edits.is_empty() && beyond {
        tasks.push("reference generation");
    }
    if pictures.iter().any(|p| {
        !label(p).is_empty()
            && list(p, "section_bindings").iter().any(|b| {
                matches!(
                    s(b, "resolved_role"),
                    "first_frame" | "last_frame" | "keyframe"
                )
            })
    }) || guide_enabled
    {
        tasks.push("keyframe completion");
    }
    if boundary_video.is_some() || motion_enabled {
        tasks.insert(0, "video continuation");
    }
    if audios
        .iter()
        .any(|a| matches!(s(a, "retention_marker"), "fully_copy" | "partially_copy"))
    {
        tasks.push("audio reuse");
    }
    if boundary_audio.is_some()
        || audios.iter().any(|a| {
            matches!(
                fallback(s(a, "retention_marker"), "reference"),
                "reference" | "weak_reference"
            )
        })
    {
        tasks.push("audio reference");
    }
    let mut summary_line = format!("[{}]", tasks.join(" + "));
    if let Some(edit) = edits.first() {
        summary_line = format!(
            "{summary_line} The target video is an edited version of {}.",
            label(edit)
        );
    }
    if !summary.is_empty() {
        summary_line = format!("{summary_line} {summary}");
    }
    let mut lines = vec!["subject_definitions:".into()];
    lines.extend(definitions(manifest));
    lines.extend([
        String::new(),
        "summary:".into(),
        summary_line,
        String::new(),
        "retention_analysis:".into(),
    ]);
    lines.extend(retention(manifest));
    lines.extend([String::new(), "detailed_description:".into()]);
    if !description.is_empty() {
        lines.extend([description.clone(), String::new()]);
    }
    lines.extend(detail);
    let score = dedupe(music).join(" ");
    lines.extend([
        String::new(),
        "overall_soundscape:".into(),
        values(sound),
        String::new(),
        "non_diegetic_music:".into(),
        fallback(&score, "N/A").into(),
    ]);
    let mode = boundary_video
        .or(boundary_picture)
        .map(|v| v["boundary_mode"].clone())
        .or_else(|| guide_enabled.then(|| guide["boundary_mode"].clone()))
        .unwrap_or(Value::Null);
    Output {
        prompt: lines.join("\n").pytrim().into(),
        metadata: json!({"format":"full_reference_six_sections","internal_shot_count":input.sections.len(),"subject_count":subjects.len(),"picture_count":pictures.len(),"video_count":videos.len(),"audio_count":audios.len(),"scene_summary_authored":!summary.is_empty(),"scene_description_authored":!description.is_empty(),"ambient_sound_source":if ambient.is_empty(){"default"}else{"authored"},"non_diegetic_music_source":if authored_music.is_empty(){"default_na"}else{"authored"},"boundary_continuation":boundary_video.is_some(),"motion_context":motion_enabled,"frame0_guide":guide_enabled,"motion_context_frames":if motion_enabled{f(motion,"context_frames")as i64}else{0},"source_video_edit":!edits.is_empty(),"source_video_edit_count":edits.len(),"boundary_audio_reference":boundary_audio.is_some(),"boundary_mode":mode,"previous_end_frame_reference":boundary_picture.is_some()||guide_enabled,"boundary_first_frame_derived":guide_enabled||boundary_picture.is_some()}),
    }
}
