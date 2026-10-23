use crate::video::{contract::*, text::PythonText};
use serde_json::{Value, json};
pub fn entries(project: &VideoProject, scene: &Scene) -> Vec<Value> {
    imagine_video_contract::creative::video_bindings(project,scene).into_iter().enumerate().map(|(i,placement)| {
        let binding=placement.binding;
        let options=binding.video.clone().unwrap_or_default();
        let seconds=|time:Time|time.num as f64/time.den as f64;
        json!({"kind":"video","role":if options.role.is_empty(){"video_reference"}else{&options.role},"custom_role":options.custom_role.pytrim(),"relationship":options.relationship.map(|r|json!(r)).unwrap_or_else(||json!({"marker":"fully_preserved","note":"","target_reference_id":null})),"native_input":"ref_videos","native_key":format!("ref_video_{i}"),"native_audio_key":format!("ref_video_audio_{i}"),"prompt_label":format!("<Video {}>",i+1),"asset_id":binding.asset,"section_id":placement.section_id,"binding_id":binding.id,"internal_shot_index":placement.shot_index,"source_in":options.source_in.map(seconds),"source_out":options.source_out.map(seconds),"soundtrack_usage":if options.soundtrack_usage.is_empty(){"ignore"}else{&options.soundtrack_usage}})
    }).collect()
}
pub fn paired(videos: &[Value], audio_offset: usize) -> Vec<Value> {
    videos.iter().filter(|v|matches!(v["soundtrack_usage"].as_str(),Some("reference"|"both"))).enumerate().map(|(i,v)|json!({"kind":"audio","role":"paired_video_soundtrack_reference","native_input":"ref_video_audios","native_key":v["native_audio_key"],"prompt_label":format!("<Audio {}>",audio_offset+i+1),"paired_video_label":v["prompt_label"],"paired_video_key":v["native_key"],"item_id":v["section_id"],"source_in":v["source_in"],"source_out":v["source_out"],"boundary_continuation":false,"purpose":"ambience_sfx","retention_marker":"reference"})).collect()
}
