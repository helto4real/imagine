//! Director shot-local audio extraction and H3 standalone-reference metadata.
use crate::audio::*;
use serde_json::{Value, json};
fn seconds(t: Time) -> f64 {
    t.num as f64 / t.den as f64
}
/// Section video soundtracks use the same mix timing as Director's audio-plan entries.
pub fn soundtrack_clips(project: &VideoProject) -> Vec<AudioClip> {
    let mut clips = vec![];
    for scene in &project.scenes {
        for placement in crate::creative::video_bindings(project, scene) {
            let Some(options) = placement
                .binding
                .video
                .as_ref()
                .filter(|v| matches!(v.soundtrack_usage.as_str(), "both" | "final_mix"))
            else {
                continue;
            };
            let section = placement
                .section_id
                .and_then(|id| scene.sections.iter().find(|s| &s.id == id));
            let start = crate::assembly::time(
                seconds(scene.start) + section.map_or(0., |s| seconds(s.start)),
            );
            let mut clip = AudioClip::new(
                AudioClipId(placement.binding.id.0.clone()),
                start,
                section.map_or(scene.duration, |s| s.duration),
                &Default::default(),
            );
            clip.asset = Some(placement.binding.asset.clone());
            clip.source_in = options.source_in.unwrap_or(Time { num: 0, den: 1 });
            clip.source_out = options.source_out;
            clips.push(clip);
        }
    }
    clips
}
pub fn entries(project: &VideoProject, scene: &Scene) -> Vec<Value> {
    let scene_start = seconds(scene.start);
    let duration = seconds(scene.duration);
    let scene_end = scene_start + duration;
    let tracks = project
        .audio_tracks
        .iter()
        .map(|track| {
            let mut clips = track
                .clips
                .iter()
                .filter_map(|clip| {
                    let start = seconds(clip.start);
                    let end = start + seconds(clip.duration);
                    let retained_start = start.max(scene_start);
                    let retained_end = end.min(scene_end);
                    if retained_end <= retained_start {
                        return None;
                    }
                    let source_in = seconds(clip.source_in) + (retained_start - start);
                    Some((
                        clip,
                        retained_start - scene_start,
                        retained_end - scene_start,
                        source_in,
                        clip.source_out.map(|out| {
                            seconds(out)
                                .clamp(source_in, source_in + (retained_end - retained_start))
                        }),
                        (clip.fade_in - (retained_start - start)).max(0.),
                        (clip.fade_out - (end - retained_end)).max(0.),
                    ))
                })
                .collect::<Vec<_>>();
            clips.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.id.cmp(&b.0.id)));
            (track, clips)
        })
        .collect::<Vec<_>>();
    let mix_sources = tracks
        .iter()
        .flat_map(|(_, clips)| clips)
        .filter(|(c, ..)| c.enabled && c.usage.final_mix)
        .count()
        + crate::creative::video_bindings(project, scene)
            .iter()
            .filter(|p| {
                p.binding
                    .video
                    .as_ref()
                    .is_some_and(|v| matches!(v.soundtrack_usage.as_str(), "both" | "final_mix"))
            })
            .count();
    let mut entries = vec![];
    for (track_index, (track, clips)) in tracks.into_iter().enumerate() {
        for (clip, start, end, source_in, source_out, fade_in, fade_out) in clips {
            if !clip.enabled || !clip.usage.generation_reference {
                continue;
            }
            let Some(asset) = clip.asset.as_ref() else {
                continue;
            };
            let exact = mix_sources == 1
                && start.abs() < 1e-6
                && (end - duration).abs() < 1e-6
                && source_in.abs() < 1e-6
                && source_out.is_none()
                && ((if clip.volume == 0. { 100. } else { clip.volume }) - 100.).abs() < 1e-6
                && fade_in.abs() < 1e-6
                && fade_out.abs() < 1e-6;
            let purpose = serde_json::to_value(clip.reference.purpose).unwrap();
            let retention = if clip.usage.final_mix {
                if exact {
                    "fully_copy"
                } else {
                    "partially_copy"
                }
            } else {
                match clip.reference.relationship {
                    AudioRelationship::Reference => "reference",
                    AudioRelationship::WeakReference => "weak_reference",
                }
            };
            let note = if clip.usage.final_mix {
                format!(
                    "The source signal is included in the final mix and also provides {} guidance.",
                    purpose.as_str().unwrap().replace('_', " ")
                )
            } else {
                String::new()
            };
            let i = entries.len();
            entries.push(json!({"kind":"audio","role":"standalone_timeline_audio_reference","native_input":"ref_audios","native_key":format!("ref_audio_{i}"),"prompt_label":format!("<Audio {}>",i+1),"paired_video_label":null,"track_id":track.id,"track_index":track_index,"item_id":clip.id,"asset_id":asset,"start_time":start,"end_time":end,"source_in":source_in,"source_out":source_out,"purpose":purpose,"target_reference_id":clip.reference.target_reference_id,"description":clip.reference.description.trim_matches(crate::references::python_space),"strength":clip.reference.strength,"retention_marker":retention,"retention_note":note}));
        }
    }
    entries
}
