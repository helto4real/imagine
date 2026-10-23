//! Project boundary intent projected into the existing ordered H3 manifest.
use crate::video::contract::*;
use imagine_video_contract::{
    assembly::BoundaryMode,
    boundary::{self, Incoming, Source},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn incoming(project: &VideoProject, scene: &Scene) -> Result<Option<Incoming>, Issue> {
    Ok(boundary::resolve(project, &scene.id)?.filter(|b| {
        b.policy != boundary::Policy::None
            && b.explicit_continuity.as_ref().is_none_or(|c| {
                matches!(
                    c.method,
                    ContinuityMethod::Combo | ContinuityMethod::Influence
                )
            })
    }))
}
fn rename(value: &mut Value, replacements: &BTreeMap<String, String>) {
    match value {
        Value::String(s) => {
            if let Some(next) = replacements.get(s) {
                *s = next.clone();
            }
        }
        Value::Array(items) => {
            for item in items {
                rename(item, replacements);
            }
        }
        Value::Object(items) => {
            for item in items.values_mut() {
                rename(item, replacements);
            }
        }
        _ => {}
    }
}
pub(super) fn apply(
    incoming: &Incoming,
    i2v: bool,
    manifest: &mut Value,
    keyframes: &mut Value,
) -> Result<String, Issue> {
    let source = incoming
        .source
        .as_ref()
        .ok_or_else(|| Issue::new("continuity_source_missing", "scene.boundary"))?;
    let influence = !i2v
        && incoming
            .explicit_continuity
            .as_ref()
            .is_some_and(|c| c.method == ContinuityMethod::Influence);
    let include_audio = incoming.audio_continuity
        == imagine_video_contract::assembly::AudioContinuity::ReferencePreviousAudio;
    if i2v && include_audio {
        return Err(Issue::new(
            "h3_i2v_audio_reference_unavailable",
            "scene.boundary.audio",
        ));
    }
    let mode = match incoming.mode {
        BoundaryMode::HardCut => "Hard Cut",
        BoundaryMode::ContinuousShot => "Continuous Shot",
        BoundaryMode::BlendSeam => "Blend Seam",
        BoundaryMode::Transition => "Transition",
    };
    let mut video = json!({"kind":"video","role":"boundary_continuation",
        "native_input":"ref_videos","native_key":"ref_video_0","native_audio_key":"ref_video_audio_0",
        "prompt_label":"<Video 1>","source_scene_id":incoming.source_scene,
        "source_in":null,"source_out":null,"requested_tail_frames":incoming.requested_tail_frames,
        "include_paired_audio":include_audio,"boundary_source_only":i2v,"boundary_mode":mode,
        "transition_prompt":incoming.transition_prompt,"reuse_visual_references":incoming.reuse_visual_references,
        "reuse_style":incoming.reuse_style});
    match source {
        Source::AcceptedTake { take_id, .. } => video["take_id"] = json!(take_id),
        Source::ImportedClip {
            asset_id,
            source_in,
            source_out,
            ..
        } => {
            video["asset_id"] = json!(asset_id);
            video["source_in"] = json!(source_in.num as f64 / source_in.den as f64);
            video["source_out"] = json!(source_out.map(|t| t.num as f64 / t.den as f64));
        }
    }
    if let Some(last) = &incoming.last_frame {
        video["endpoint_asset_id"] = json!(last.asset_id);
    }
    // Relabel the original presentation atomically, including subject source-image
    // pointers and paired-video keys. Newly prepended boundary labels are untouched.
    let mut names = BTreeMap::new();
    for (index, _) in manifest["videos"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        names.insert(
            format!("ref_video_{index}"),
            format!("ref_video_{}", index + 1),
        );
        names.insert(
            format!("ref_video_audio_{index}"),
            format!("ref_video_audio_{}", index + 1),
        );
        names.insert(
            format!("<Video {}>", index + 1),
            format!("<Video {}>", index + 2),
        );
    }
    if influence {
        for (index, _) in manifest["images"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            names.insert(
                format!("ref_image_{index}"),
                format!("ref_image_{}", index + 1),
            );
            names.insert(
                format!("<Picture {}>", index + 1),
                format!("<Picture {}>", index + 2),
            );
        }
    }
    rename(manifest, &names);
    if influence {
        manifest["images"].as_array_mut().unwrap().insert(
            0,
            json!({"kind":"image",
            "role":"boundary_first_frame","from_boundary_video":true,"native_input":"ref_images",
            "native_key":"ref_image_0","prompt_label":"<Picture 1>","boundary_mode":mode,
            "source_scene_id":incoming.source_scene}),
        );
    } else {
        manifest["frame0_guide"] = json!({"enabled":true,"boundary_mode":mode,
            "source_scene_id":incoming.source_scene,"transition_prompt":incoming.transition_prompt});
        keyframes["first"] = json!({"boundary_from":incoming.source_scene});
    }
    manifest["videos"].as_array_mut().unwrap().insert(0, video);
    if include_audio {
        manifest["audios"].as_array_mut().unwrap().insert(0,json!({"kind":"audio",
            "role":"boundary_audio_reference","boundary_continuation":true,"purpose":"continuity",
            "native_input":"ref_video_audios","native_key":"ref_video_audio_0",
            "paired_video_key":"ref_video_0","paired_video_label":"<Video 1>","prompt_label":"<Audio 1>"}));
    }
    let mut audio_index = 1;
    let videos = manifest["videos"].as_array().unwrap().clone();
    for video in &videos {
        if let Some(audio) = manifest["audios"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|a| a["paired_video_key"] == video["native_key"])
        {
            audio["prompt_label"] = json!(format!("<Audio {audio_index}>"));
            audio_index += 1;
        }
    }
    for audio in manifest["audios"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .filter(|a| a["paired_video_key"].is_null())
    {
        audio["prompt_label"] = json!(format!("<Audio {audio_index}>"));
        audio_index += 1;
    }
    let images = manifest["images"].as_array().unwrap();
    let audios = manifest["audios"].as_array().unwrap();
    let paired = audios
        .iter()
        .filter(|a| !a["paired_video_key"].is_null())
        .count();
    let mut order = images
        .iter()
        .map(|p| json!({"kind":"image","label":p["prompt_label"]}))
        .collect::<Vec<_>>();
    for video in &videos {
        if let Some(audio) = audios
            .iter()
            .find(|a| a["paired_video_key"] == video["native_key"])
        {
            order.push(json!({"kind":"audio","label":audio["prompt_label"]}));
        }
        order.push(json!({"kind":"video","label":video["prompt_label"]}));
    }
    order.extend(
        audios
            .iter()
            .filter(|a| a["paired_video_key"].is_null())
            .map(|a| json!({"kind":"audio","label":a["prompt_label"]})),
    );
    manifest["counts"] = json!({"images":images.len(),"subjects":manifest["subjects"].as_array().map_or(0,Vec::len),
        "videos":videos.len(),"paired_video_audios":paired,"standalone_audios":audios.len()-paired,
        "audios":audios.len(),"files":images.len()+videos.len()+audios.len()});
    manifest["presentation_order"] = json!(order);
    manifest["boundary_intent"] = json!(incoming);
    manifest["boundary_dependency"] = boundary::dependency(incoming, true);
    Ok(if incoming.mode == BoundaryMode::Transition {
        incoming.transition_prompt.clone()
    } else {
        String::new()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepended_boundary_keeps_subject_picture_and_paired_audio_pointers_exact() {
        let incoming = Incoming {
            boundary_left: Some(SceneId("source".into())),
            target_scene: SceneId("target".into()),
            source_scene: SceneId("source".into()),
            explicit_continuity: Some(Continuity {
                source_scene: SceneId("source".into()),
                method: ContinuityMethod::Influence,
                include_audio: false,
            }),
            mode: BoundaryMode::Transition,
            policy: boundary::Policy::Transition,
            requested_tail_frames: 72,
            blend_frames: 3,
            reuse_visual_references: true,
            reuse_style: true,
            audio_continuity:
                imagine_video_contract::assembly::AudioContinuity::ReferencePreviousAudio,
            transition_prompt: "The camera turns.".into(),
            source: Some(Source::AcceptedTake {
                take_id: TakeId("take".into()),
                source_fingerprint: "content".into(),
                model: None,
            }),
            last_frame: None,
        };
        let mut manifest = json!({"subjects":[{"source_images":[{"prompt_label":"<Picture 1>","native_key":"ref_image_0"}]}],
            "images":[{"prompt_label":"<Picture 1>","native_key":"ref_image_0"}],
            "videos":[{"prompt_label":"<Video 1>","native_key":"ref_video_0","native_audio_key":"ref_video_audio_0"}],
            "audios":[{"prompt_label":"<Audio 1>","native_key":"ref_video_audio_0","paired_video_key":"ref_video_0","paired_video_label":"<Video 1>"}],
            "frame0_guide":{"enabled":false}});
        let prompt = apply(&incoming, false, &mut manifest, &mut Value::Null).unwrap();
        assert_eq!(prompt, "The camera turns.");
        assert_eq!(
            manifest["subjects"][0]["source_images"][0]["prompt_label"],
            "<Picture 2>"
        );
        assert_eq!(
            manifest["subjects"][0]["source_images"][0]["native_key"],
            "ref_image_1"
        );
        assert_eq!(manifest["videos"][1]["native_key"], "ref_video_1");
        assert_eq!(manifest["audios"][1]["paired_video_key"], "ref_video_1");
        assert_eq!(manifest["audios"][1]["prompt_label"], "<Audio 2>");
        assert_eq!(
            manifest["presentation_order"],
            json!([
            {"kind":"image","label":"<Picture 1>"},{"kind":"image","label":"<Picture 2>"},
            {"kind":"audio","label":"<Audio 1>"},{"kind":"video","label":"<Video 1>"},
            {"kind":"audio","label":"<Audio 2>"},{"kind":"video","label":"<Video 2>"}])
        );
    }
}
