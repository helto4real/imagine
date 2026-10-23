use super::{Input, Output, common::*, dialogue, f, s, truth};
use crate::video::text::PythonText;
use serde_json::json;
fn authored(text: &str, maps: &dialogue::Maps) -> String {
    replace(&dialogue::compile(&single(text), maps), &maps.subject)
}
pub(super) fn compile(input: &Input) -> Output {
    let mut maps = dialogue::Maps::default();
    for r in &input.text_references {
        let alias = s(r, "alias").to_lowercase();
        let description = s(r, "description").pytrim();
        if !description.is_empty() {
            maps.subject.insert(alias.clone(), description.into());
        }
        let speaker = s(r, "speaker_id");
        if !speaker.is_empty() {
            maps.speaker.insert(alias.clone(), speaker.into());
        }
        maps.language.insert(
            alias.clone(),
            if s(r, "default_language").is_empty() {
                "English"
            } else {
                s(r, "default_language")
            }
            .into(),
        );
        let profile = s(r, "voice_profile").pytrim();
        if !profile.is_empty() {
            maps.voice_profile.insert(alias, profile.into());
        }
    }
    let first = truth(&input.keyframes["first"]);
    let last = truth(&input.keyframes["last"]);
    let task = match (first, last) {
        (true, true) => "FL2VA",
        (true, false) => "I2VA",
        (false, true) => "L2VA",
        _ => "T2VA",
    };
    let final_shot = input.sections.len().max(1);
    let duration = input.effective_duration_seconds;
    let instruction=match task{
        "I2VA"=>"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.".into(),
        "FL2VA"=>format!("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {final_shot}) aligns with the {duration:.2}-second mark of the target video."),
        "L2VA"=>format!("How the reference pictures align with the target video — <Picture 1> (from [Shot {final_shot}]) aligns with the {duration:.2}-second mark of the target video."),_=>String::new(),
    };
    let opening = match task {
        "I2VA" => "The opening composition and visible state are established by <Picture 1>.",
        "FL2VA" => {
            "The scene begins in the composition established by Picture 1 and develops continuously toward the composition established by Picture 2 at the end."
        }
        "L2VA" => "The action develops toward the final composition established by <Picture 1>.",
        _ => "",
    };
    let summary = authored(s(&input.scene_prompt, "summary"), &maps);
    let description = authored(s(&input.scene_prompt, "description"), &maps);
    let ambient = authored(s(&input.scene_prompt, "ambient_sound"), &maps);
    let score = authored(s(&input.scene_prompt, "non_diegetic_music"), &maps);
    let boundary = authored(&input.boundary_prompt, &maps);
    let global = if truth(&input.global_prompt["enabled"]) {
        s(&input.global_prompt, "prompt").pytrim()
    } else {
        ""
    };
    let (visual, audio, music) = channels(global);
    let global_visual = authored(&visual, &maps);
    let suffix = s(&input.global_prompt, "position") == "Suffix";
    let mut sound = vec![if ambient.is_empty() {
        AMBIENT.into()
    } else {
        ambient.clone()
    }];
    sound.extend(audio.iter().map(|s| authored(s, &maps)));
    let mut score_lines = if score.is_empty() {
        vec![]
    } else {
        vec![score.clone()]
    };
    score_lines.extend(music.iter().map(|s| authored(s, &maps)));
    let mut shots = vec![];
    for (index, section) in input.sections.iter().enumerate() {
        let (visual, audio, music) = channels(s(section, "prompt").pytrim());
        let section_visual = authored(&visual, &maps);
        let mut visual = if index == 0 {
            let v = join(opening, &summary);
            let v = join(&v, &description);
            let v = join(&v, &boundary);
            if !suffix { join(&v, &global_visual) } else { v }
        } else {
            String::new()
        };
        if suffix && index == input.sections.len() - 1 {
            visual = join(&visual, &global_visual);
        }
        visual = join(&visual, &section_visual);
        let marker = if index == 0 {
            "[Shot 1]".into()
        } else {
            format!(
                "[Shot {}] At {},",
                index + 1,
                timestamp(f(section, "start_time"))
            )
        };
        shots.push(format!("{marker} {visual}").trim_end().to_owned());
        sound.extend(audio.iter().map(|s| authored(s, &maps)));
        score_lines.extend(music.iter().map(|s| authored(s, &maps)));
    }
    if shots.is_empty() {
        let v = join(opening, &summary);
        let v = join(&v, &description);
        let v = join(&v, &boundary);
        let v = join(&v, &global_visual);
        shots.push(format!("[Shot 1] {v}").trim_end().into());
    }
    let score_text = dedupe(score_lines).join(" ");
    let core = format!(
        "integrated_multimodal_description: {}\n\noverall_soundscape: {}\n\nnon_diegetic_music: {}",
        shots.join(" "),
        values(sound),
        if score_text.is_empty() {
            "N/A"
        } else {
            &score_text
        }
    );
    Output {
        prompt: if instruction.is_empty() {
            core
        } else {
            format!("{instruction}\n\n{core}")
        }
        .pytrim()
        .into(),
        metadata: json!({"format":"base_three_fields","task_type":task,"alignment_instruction":!instruction.is_empty(),"internal_shot_count":final_shot,"text_reference_count":input.text_references.len(),"first_frame":first,"last_frame":last,"scene_summary_authored":!summary.is_empty(),"scene_description_authored":!description.is_empty(),"boundary_prompt_authored":!boundary.is_empty(),"ambient_sound_source":if ambient.is_empty(){"default"}else{"authored"},"non_diegetic_music_source":if score.is_empty(){"default_na"}else{"authored"}}),
    }
}
