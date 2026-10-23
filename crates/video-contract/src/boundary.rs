//! Pure incoming-boundary intent. Resolving media resources belongs to the server.
use crate::{
    assembly::{AudioContinuity, Boundary, BoundaryMode},
    contract::*,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    None,
    Continuous,
    Blend,
    Transition,
}
impl From<BoundaryMode> for Policy {
    fn from(mode: BoundaryMode) -> Self {
        match mode {
            BoundaryMode::HardCut => Self::None,
            BoundaryMode::ContinuousShot => Self::Continuous,
            BoundaryMode::BlendSeam => Self::Blend,
            BoundaryMode::Transition => Self::Transition,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    AcceptedTake {
        take_id: TakeId,
        source_fingerprint: String,
        model: Option<ModelSelection>,
    },
    ImportedClip {
        asset_id: AssetId,
        revision: u64,
        source_in: Time,
        source_out: Option<Time>,
        speed: f64,
        include_audio: bool,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LastFrame {
    pub asset_id: AssetId,
    pub revision: u64,
    pub section_id: SectionId,
    pub binding_id: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Incoming {
    pub target_scene: SceneId,
    pub source_scene: SceneId,
    /// The authored physical incoming edge is independent of a selected source.
    /// None means no authored chronological edge (also the legacy wire default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_left: Option<SceneId>,
    /// Preserves both the selected algorithm and its independently authored audio choice.
    pub explicit_continuity: Option<Continuity>,
    pub mode: BoundaryMode,
    pub policy: Policy,
    pub requested_tail_frames: u64,
    pub blend_frames: u32,
    pub reuse_visual_references: bool,
    pub reuse_style: bool,
    pub audio_continuity: AudioContinuity,
    pub transition_prompt: String,
    /// None means the requested source currently has no usable clip identity.
    pub source: Option<Source>,
    pub last_frame: Option<LastFrame>,
}

/// Director's effective generation-dependency projection, without assembly-only
/// fields or model metadata. The caller records whether the chosen conditioning
/// actually references the previous clip; this cannot be inferred model-neutrally.
///
/// Port identity mapping: Director's authored boundary UUID is represented here
/// by its stable physical `{left, right}` Scene pair, independent of an explicitly
/// selected nonadjacent conditioning source. No authored edge means null identity.
/// Accepted Take IDs identify immutable
/// Take media in this port; there is no project Asset ID for that resource, so
/// `asset_id` remains null for Accepted Takes. Imported clips retain their Asset
/// ID and seconds-based trim. These mappings are explicit, not fabricated UUIDs
/// or claims that the port's JSON/hash bytes equal Director's UUID-based values.
///
/// `target_shot_id` is informational (Director includes it in the dependency but
/// excludes it from its SHA projection). Compare receipts for the same target.
pub fn dependency(incoming: &Incoming, reference_previous_clip: bool) -> serde_json::Value {
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    let policy = Policy::from(incoming.mode);
    let active = policy != Policy::None;
    let mut value = json!({
        "schema_version": 2,
        "mode": match incoming.mode {
            BoundaryMode::HardCut => "Hard Cut",
            BoundaryMode::ContinuousShot => "Continuous Shot",
            BoundaryMode::BlendSeam => "Blend Seam",
            BoundaryMode::Transition => "Transition",
        },
        "policy": policy,
        "boundary_id": null,
        "source_shot_id": null,
        "target_shot_id": incoming.target_scene,
        "source_kind": null,
        "take_id": null,
        "asset_id": null,
        "source_in": null,
        "source_out": null,
        "reference_previous_clip": reference_previous_clip,
        "requested_tail_frames": 0,
        "reuse_visual_references": false,
        "reuse_style": false,
        "audio_continuity": "New Scene Audio",
        "transition_prompt_sha256": "",
    });
    if !active && !reference_previous_clip {
        return value;
    }
    value["boundary_id"] = incoming.boundary_left.as_ref().map_or(
        Value::Null,
        |left| json!({"left": left, "right": incoming.target_scene}),
    );
    value["source_shot_id"] = json!(incoming.source_scene);
    match &incoming.source {
        Some(Source::AcceptedTake { take_id, .. }) => {
            value["source_kind"] = json!("accepted_take");
            value["take_id"] = json!(take_id);
        }
        Some(Source::ImportedClip {
            asset_id,
            source_in,
            source_out,
            ..
        }) => {
            value["source_kind"] = json!("clip_instance");
            value["asset_id"] = json!(asset_id);
            value["source_in"] = json!(crate::assembly::seconds(*source_in));
            value["source_out"] = source_out
                .map(crate::assembly::seconds)
                .map_or(Value::Null, |t| json!(t));
        }
        None => {}
    }
    if active {
        value["requested_tail_frames"] = json!(incoming.requested_tail_frames);
        value["reuse_visual_references"] = json!(incoming.reuse_visual_references);
        value["reuse_style"] = json!(incoming.reuse_style);
        value["audio_continuity"] = json!(match incoming.audio_continuity {
            AudioContinuity::NewSceneAudio => "New Scene Audio",
            AudioContinuity::ReferencePreviousAudio => "Reference Previous Audio",
        });
    }
    if policy == Policy::Transition {
        let prompt = incoming
            .transition_prompt
            .trim_matches(crate::references::python_space);
        if !prompt.is_empty() {
            value["transition_prompt_sha256"] =
                json!(format!("{:x}", Sha256::digest(prompt.as_bytes())));
        }
    }
    value
}

/// Resolve against a document whose ordinary structural validation is performed
/// at its existing boundary. This function neither loads media nor invents a Take.
/// Hard Cut alone does not request conditioning. Explicit continuity retains its
/// chosen source, including a nonadjacent source; it is never silently redirected.
/// Boundary options still belong to the chronological previous-to-target edge.
pub fn resolve(project: &VideoProject, target: &SceneId) -> Result<Option<Incoming>, Issue> {
    let target_scene = project.scene(target)?;
    let mut ordered: Vec<_> = project.scenes.iter().collect();
    // Director compareScenes: start, end, id. Time comparison stays exact here.
    ordered.sort_by(|left, right| {
        crate::audio::compare(left.start, right.start)
            .then_with(|| {
                let left = left.start.checked_add(left.duration);
                let right = right.start.checked_add(right.duration);
                match (left, right) {
                    (Some(left), Some(right)) => crate::audio::compare(left, right),
                    _ => std::cmp::Ordering::Equal,
                }
            })
            .then_with(|| left.id.cmp(&right.id))
    });
    let previous_id = ordered
        .iter()
        .position(|scene| &scene.id == target)
        .and_then(|index| index.checked_sub(1))
        .map(|previous| &ordered[previous].id);
    let source_id = target_scene
        .continuity
        .as_ref()
        .map(|c| &c.source_scene)
        .or(previous_id);
    let Some(source_id) = source_id else {
        return Ok(None);
    };
    let source_scene = project.scene(source_id)?;
    let default = Boundary::new(source_id.clone(), target.clone());
    let authored_boundary = previous_id.and_then(|previous| {
        project
            .assembly
            .boundaries
            .iter()
            .find(|boundary| &boundary.left == previous && &boundary.right == target)
    });
    let boundary = authored_boundary.unwrap_or(&default);
    let policy = Policy::from(boundary.mode);
    if policy == Policy::None && target_scene.continuity.is_none() {
        return Ok(None);
    }
    let active = policy != Policy::None;
    let last_frame_only = target_scene
        .continuity
        .as_ref()
        .is_some_and(|c| c.method == ContinuityMethod::LastFrameFirstFrame);
    Ok(Some(Incoming {
        target_scene: target.clone(),
        source_scene: source_id.clone(),
        boundary_left: authored_boundary.map(|boundary| boundary.left.clone()),
        explicit_continuity: target_scene.continuity.clone(),
        mode: boundary.mode,
        policy,
        requested_tail_frames: if active && !last_frame_only {
            boundary.requested_tail_frames(project.delivery.fps)?
        } else {
            0
        },
        blend_frames: boundary.blend_frames,
        reuse_visual_references: active && boundary.reuse_visual_references,
        reuse_style: active && boundary.reuse_style,
        audio_continuity: if active {
            boundary.audio_continuity
        } else {
            AudioContinuity::NewSceneAudio
        },
        transition_prompt: if policy == Policy::Transition {
            boundary
                .transition_prompt
                .trim_matches(crate::references::python_space)
                .to_owned()
        } else {
            String::new()
        },
        source: clip_source(project, source_scene),
        last_frame: last_frame(project, source_scene),
    }))
}

fn clip_source(project: &VideoProject, scene: &Scene) -> Option<Source> {
    if let Some(take) = &scene.accepted_take {
        return Some(Source::AcceptedTake {
            take_id: take.id.clone(),
            source_fingerprint: take.source_fingerprint.clone(),
            model: take.model.clone(),
        });
    }
    let clip = project.scene_clip(&scene.id)?;
    let asset = project.assets.get(&clip.asset)?;
    (asset.kind == MediaKind::Video).then(|| Source::ImportedClip {
        asset_id: asset.id.clone(),
        revision: asset.revision,
        source_in: clip.source_in,
        source_out: clip.source_out,
        speed: clip.speed,
        include_audio: clip.include_audio,
    })
}

/// Director inspects Sections in reverse attachment order, stopping at the first
/// Section with Last Frame bindings, even if that Section is ambiguous/unresolved.
fn last_frame(project: &VideoProject, scene: &Scene) -> Option<LastFrame> {
    for section in scene.sections.iter().rev() {
        let Some(source) = project.creative.sections.get(&section.id) else {
            continue;
        };
        let candidates: Vec<_> = source
            .images
            .iter()
            .filter(|binding| binding.role == InputRole::LastFrame)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let mut resolved = candidates.into_iter().filter_map(|binding| {
            let asset = project.assets.get(&binding.asset)?;
            (asset.kind == MediaKind::Image).then(|| LastFrame {
                asset_id: asset.id.clone(),
                revision: asset.revision,
                section_id: section.id.clone(),
                binding_id: binding.image_id.clone(),
            })
        });
        let first = resolved.next()?;
        return resolved.next().is_none().then_some(first);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::ClipInstance,
        creative::{ImageBinding, SectionSource},
    };

    fn time(num: u64) -> Time {
        Time { num, den: 1 }
    }
    fn project() -> VideoProject {
        let mut project = VideoProject::create(CreateProject {
            generation: None,
            schema_version: 1,
            default_model: ModelSelection {
                adapter: "synthetic".into(),
                model: "synthetic".into(),
                variant: "synthetic".into(),
                profile: "synthetic".into(),
                profile_revision: "1".into(),
            },
            delivery: DeliverySettings {
                fps: time(24),
                width: 64,
                height: 48,
            },
        })
        .unwrap();
        for (id, start) in [("first", 0), ("middle", 4), ("target", 8)] {
            project.scenes.push(Scene {
                id: SceneId(id.into()),
                start: time(start),
                duration: time(4),
                description: String::new(),
                model: ModelChoice::ProjectDefault,
                settings: vec![],
                sections: vec![],
                bindings: vec![],
                continuity: None,
                accepted_take: None,
            });
        }
        project
    }
    fn target() -> SceneId {
        SceneId("target".into())
    }
    fn activate(project: &mut VideoProject, mode: BoundaryMode) {
        project.assembly.boundaries.push(Boundary {
            mode,
            ..Boundary::new(SceneId("middle".into()), target())
        });
    }
    fn asset(project: &mut VideoProject, id: &str, kind: MediaKind) -> AssetId {
        let id = AssetId(id.into());
        project.assets.insert(
            id.clone(),
            Asset {
                library_source: None,
                name: String::new(),
                id: id.clone(),
                revision: 7,
                kind,
            },
        );
        id
    }

    #[test]
    fn dependency_omits_assembly_model_and_endpoint_metadata() {
        let mut project = project();
        activate(&mut project, BoundaryMode::BlendSeam);
        project.scenes[1].accepted_take = Some(AcceptedTake {
            id: TakeId("own-take".into()),
            model: None,
            source_fingerprint: "first".into(),
        });
        let mut incoming = resolve(&project, &target()).unwrap().unwrap();
        let expected = dependency(&incoming, true);
        incoming.blend_frames = 1000;
        incoming.last_frame = Some(LastFrame {
            asset_id: AssetId("endpoint".into()),
            revision: 2,
            section_id: SectionId("section".into()),
            binding_id: "binding".into(),
        });
        incoming.transition_prompt = "ignored outside Transition".into();
        if let Some(Source::AcceptedTake {
            source_fingerprint,
            model,
            ..
        }) = &mut incoming.source
        {
            *source_fingerprint = "second".into();
            *model = Some(project.default_model.clone());
        }
        assert_eq!(expected, dependency(&incoming, true));
        assert_eq!(
            expected["boundary_id"],
            serde_json::json!({"left":"middle","right":"target"})
        );
        assert_eq!(expected["source_kind"], "accepted_take");
        assert!(expected["asset_id"].is_null());
        if let Some(Source::AcceptedTake { take_id, .. }) = &mut incoming.source {
            *take_id = TakeId("replacement".into());
        }
        assert_ne!(expected, dependency(&incoming, true));
    }

    #[test]
    fn dependency_imported_clip_tracks_asset_and_trim_not_playback_metadata() {
        let mut project = project();
        activate(&mut project, BoundaryMode::ContinuousShot);
        let mut incoming = resolve(&project, &target()).unwrap().unwrap();
        incoming.source = Some(Source::ImportedClip {
            asset_id: AssetId("clip".into()),
            revision: 1,
            source_in: time(2),
            source_out: Some(time(5)),
            speed: 1.,
            include_audio: true,
        });
        let expected = dependency(&incoming, false);
        assert_eq!(expected["source_kind"], "clip_instance");
        assert_eq!(expected["source_in"], 2.);
        assert_eq!(expected["source_out"], 5.);
        if let Some(Source::ImportedClip {
            revision,
            speed,
            include_audio,
            source_in,
            ..
        }) = &mut incoming.source
        {
            *revision = 7;
            *speed = 0.5;
            *include_audio = false;
            *source_in = Time { num: 4, den: 2 };
        }
        assert_eq!(expected, dependency(&incoming, false));
        if let Some(Source::ImportedClip { source_out, .. }) = &mut incoming.source {
            *source_out = Some(time(6));
        }
        assert_ne!(expected, dependency(&incoming, false));
    }

    #[test]
    fn dependency_hard_cut_tracks_source_only_when_reference_was_requested() {
        let mut project = project();
        activate(&mut project, BoundaryMode::Transition);
        let mut incoming = resolve(&project, &target()).unwrap().unwrap();
        incoming.mode = BoundaryMode::HardCut;
        let expected = dependency(&incoming, false);
        incoming.source_scene = SceneId("replacement".into());
        incoming.source = Some(Source::AcceptedTake {
            take_id: TakeId("replacement".into()),
            source_fingerprint: String::new(),
            model: None,
        });
        incoming.transition_prompt = "unused".into();
        incoming.requested_tail_frames = 999;
        incoming.audio_continuity = AudioContinuity::ReferencePreviousAudio;
        incoming.reuse_style = false;
        assert_eq!(expected, dependency(&incoming, false));
        assert!(expected["source_shot_id"].is_null());
        assert!(expected["boundary_id"].is_null());
        let requested = dependency(&incoming, true);
        assert_eq!(requested["take_id"], "replacement");
        assert_eq!(requested["requested_tail_frames"], 0);
        assert_eq!(requested["reuse_visual_references"], false);
        assert_eq!(requested["audio_continuity"], "New Scene Audio");
        assert_eq!(requested["transition_prompt_sha256"], "");
    }

    #[test]
    fn dependency_tracks_effective_tail_reuse_audio_and_trimmed_transition_prompt() {
        let mut project = project();
        activate(&mut project, BoundaryMode::Transition);
        let mut incoming = resolve(&project, &target()).unwrap().unwrap();
        incoming.transition_prompt = "\u{1c}abc\u{85}".into();
        let expected = dependency(&incoming, true);
        assert_eq!(
            expected["transition_prompt_sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        incoming.transition_prompt = "abc".into();
        assert_eq!(expected, dependency(&incoming, true));
        incoming.requested_tail_frames += 1;
        assert_ne!(expected, dependency(&incoming, true));
        incoming.requested_tail_frames -= 1;
        incoming.reuse_style = false;
        assert_ne!(expected, dependency(&incoming, true));
        incoming.reuse_style = true;
        incoming.audio_continuity = AudioContinuity::ReferencePreviousAudio;
        assert_ne!(expected, dependency(&incoming, true));
    }

    #[test]
    fn legacy_boundary_is_sparse_and_authored_values_roundtrip_unchanged() {
        let legacy = serde_json::json!({"left":"middle","right":"target","mode":"blend_seam","blend_frames":11});
        let mut boundary: Boundary = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(boundary.requested_tail_frames(time(24)).unwrap(), 72);
        assert_eq!(
            boundary
                .requested_tail_frames(Time {
                    num: 24000,
                    den: 1001
                })
                .unwrap(),
            72
        );
        assert_eq!(
            boundary
                .requested_tail_frames(Time { num: 3, den: 2 })
                .unwrap(),
            5
        );
        assert_eq!(serde_json::to_value(&boundary).unwrap(), legacy);
        boundary.tail_frames = Some(0);
        boundary.transition_prompt = "  authored transition  ".into();
        boundary.reuse_style = false;
        boundary.audio_continuity = AudioContinuity::ReferencePreviousAudio;
        let restored: Boundary =
            serde_json::from_slice(&serde_json::to_vec(&boundary).unwrap()).unwrap();
        assert_eq!(restored, boundary);
        assert_eq!(restored.requested_tail_frames(time(60)).unwrap(), 0);
        assert_eq!(restored.blend_frames, 11);
    }

    #[test]
    fn chronological_predecessor_and_accepted_take_precede_imported_clip() {
        let mut project = project();
        let video = asset(&mut project, "own-video", MediaKind::Video);
        project.assembly.clips.insert(
            SceneId("middle".into()),
            ClipInstance {
                asset: video,
                source_in: time(2),
                source_out: Some(time(5)),
                speed: 0.5,
                include_audio: true,
            },
        );
        project.scenes[1].accepted_take = Some(AcceptedTake {
            id: TakeId("own-take".into()),
            model: None,
            source_fingerprint: "source-v1".into(),
        });
        activate(&mut project, BoundaryMode::BlendSeam);
        project.scenes.swap(0, 2);
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(incoming.source_scene, SceneId("middle".into()));
        assert_eq!(incoming.requested_tail_frames, 72);
        assert_eq!(
            incoming.source,
            Some(Source::AcceptedTake {
                take_id: TakeId("own-take".into()),
                model: None,
                source_fingerprint: "source-v1".into(),
            })
        );
        let restored: Incoming =
            serde_json::from_slice(&serde_json::to_vec(&incoming).unwrap()).unwrap();
        assert_eq!(incoming, restored);
    }

    #[test]
    fn imported_clip_preserves_identity_revision_trim_speed_and_audio() {
        let mut project = project();
        let video = asset(&mut project, "own-video", MediaKind::Video);
        project.assembly.clips.insert(
            SceneId("middle".into()),
            ClipInstance {
                asset: video.clone(),
                source_in: time(2),
                source_out: Some(time(5)),
                speed: 0.5,
                include_audio: false,
            },
        );
        activate(&mut project, BoundaryMode::ContinuousShot);
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(
            incoming.source,
            Some(Source::ImportedClip {
                asset_id: video,
                revision: 7,
                source_in: time(2),
                source_out: Some(time(5)),
                speed: 0.5,
                include_audio: false,
            })
        );
        project.creative.scene_kinds.insert(
            SceneId("middle".into()),
            crate::creative::SceneKind::Generated,
        );
        assert!(
            resolve(&project, &target())
                .unwrap()
                .unwrap()
                .source
                .is_none()
        );
    }

    #[test]
    fn hard_cut_is_inactive_and_explicit_nonadjacent_source_is_never_redirected() {
        let mut project = project();
        assert!(resolve(&project, &target()).unwrap().is_none());
        activate(&mut project, BoundaryMode::HardCut);
        assert!(resolve(&project, &target()).unwrap().is_none());
        project.scenes[2].continuity = Some(Continuity {
            source_scene: SceneId("first".into()),
            method: ContinuityMethod::Influence,
            include_audio: true,
        });
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(incoming.source_scene, SceneId("first".into()));
        assert_eq!(incoming.explicit_continuity, project.scenes[2].continuity);
        assert_eq!(incoming.policy, Policy::None);
        assert_eq!(incoming.requested_tail_frames, 0);
        assert!(!incoming.reuse_style);
        assert_eq!(incoming.audio_continuity, AudioContinuity::NewSceneAudio);
        assert!(incoming.source.is_none());
        project.scenes[2].continuity.as_mut().unwrap().source_scene = SceneId("missing".into());
        assert_eq!(
            resolve(&project, &target()).unwrap_err().code,
            "scene_not_found"
        );
    }

    #[test]
    fn nonadjacent_combo_and_influence_keep_physical_boundary_options_and_identity() {
        let mut project = project();
        activate(&mut project, BoundaryMode::Transition);
        let physical = &mut project.assembly.boundaries[0];
        physical.tail_frames = Some(17);
        physical.transition_prompt = "  physical transition  ".into();
        physical.reuse_visual_references = false;
        physical.audio_continuity = AudioContinuity::ReferencePreviousAudio;
        // A stored nonadjacent edge cannot replace the actual incoming UI edge.
        project.assembly.boundaries.push(Boundary {
            mode: BoundaryMode::BlendSeam,
            tail_frames: Some(999),
            ..Boundary::new(SceneId("first".into()), target())
        });
        project.scenes[0].accepted_take = Some(AcceptedTake {
            id: TakeId("selected-source-take".into()),
            model: None,
            source_fingerprint: "selected-source".into(),
        });
        for method in [ContinuityMethod::Combo, ContinuityMethod::Influence] {
            project.scenes[2].continuity = Some(Continuity {
                source_scene: SceneId("first".into()),
                method,
                include_audio: true,
            });
            let incoming = resolve(&project, &target()).unwrap().unwrap();
            assert_eq!(incoming.source_scene, SceneId("first".into()));
            assert_eq!(incoming.boundary_left, Some(SceneId("middle".into())));
            assert_eq!(incoming.mode, BoundaryMode::Transition);
            assert_eq!(incoming.requested_tail_frames, 17);
            assert_eq!(incoming.transition_prompt, "physical transition");
            assert!(!incoming.reuse_visual_references);
            assert_eq!(
                incoming.audio_continuity,
                AudioContinuity::ReferencePreviousAudio
            );
            let dependency = dependency(&incoming, true);
            assert_eq!(
                dependency["boundary_id"],
                serde_json::json!({"left":"middle","right":"target"})
            );
            assert_eq!(dependency["source_shot_id"], "first");
            assert_eq!(dependency["take_id"], "selected-source-take");
        }
        project.assembly.boundaries.remove(0);
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(incoming.mode, BoundaryMode::HardCut);
        assert_eq!(incoming.boundary_left, None);
        assert!(dependency(&incoming, true)["boundary_id"].is_null());
        // Old recorded intent can deserialize; it never invents a physical edge.
        let mut legacy = serde_json::to_value(&incoming).unwrap();
        legacy.as_object_mut().unwrap().remove("boundary_left");
        let legacy: Incoming = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.boundary_left, None);
        assert!(dependency(&legacy, true)["boundary_id"].is_null());
    }

    #[test]
    fn transition_intent_and_last_frame_only_keep_their_distinct_semantics() {
        let mut project = project();
        activate(&mut project, BoundaryMode::Transition);
        let boundary = &mut project.assembly.boundaries[0];
        boundary.tail_frames = Some(123);
        boundary.transition_prompt = "\u{1c}cross dissolve\u{85}".into();
        boundary.reuse_visual_references = false;
        boundary.audio_continuity = AudioContinuity::ReferencePreviousAudio;
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(incoming.transition_prompt, "cross dissolve");
        assert_eq!(incoming.requested_tail_frames, 123);
        assert!(!incoming.reuse_visual_references);
        assert!(incoming.reuse_style);
        assert_eq!(
            incoming.audio_continuity,
            AudioContinuity::ReferencePreviousAudio
        );
        project.scenes[2].continuity = Some(Continuity {
            source_scene: SceneId("middle".into()),
            method: ContinuityMethod::LastFrameFirstFrame,
            include_audio: false,
        });
        assert_eq!(
            resolve(&project, &target())
                .unwrap()
                .unwrap()
                .requested_tail_frames,
            0
        );
        // Resolution cannot rewrite authored values when the method changes later.
        assert_eq!(project.assembly.boundaries[0].tail_frames, Some(123));
    }

    #[test]
    fn endpoint_uses_last_attached_section_and_does_not_skip_ambiguity() {
        let mut project = project();
        activate(&mut project, BoundaryMode::ContinuousShot);
        let image = asset(&mut project, "own-image", MediaKind::Image);
        for id in ["early", "late"] {
            let id = SectionId(id.into());
            project.scenes[1].sections.push(Section {
                id: id.clone(),
                start: time(0),
                duration: time(1),
                prompt: String::new(),
            });
            project.creative.sections.insert(
                id,
                SectionSource {
                    images: vec![ImageBinding {
                        anchor: None,
                        image_id: format!(
                            "binding-{id}",
                            id = project.scenes[1].sections.last().unwrap().id.0
                        ),
                        asset: image.clone(),
                        role: InputRole::LastFrame,
                    }],
                    ..Default::default()
                },
            );
        }
        let incoming = resolve(&project, &target()).unwrap().unwrap();
        assert_eq!(
            incoming.last_frame,
            Some(LastFrame {
                asset_id: image,
                revision: 7,
                section_id: SectionId("late".into()),
                binding_id: "binding-late".into(),
            })
        );
        let late = project
            .creative
            .sections
            .get_mut(&SectionId("late".into()))
            .unwrap();
        let mut duplicate = late.images[0].clone();
        duplicate.image_id = "other".into();
        late.images.push(duplicate);
        assert!(
            resolve(&project, &target())
                .unwrap()
                .unwrap()
                .last_frame
                .is_none()
        );
        late_cleanup(&mut project);
        assert_eq!(
            resolve(&project, &target())
                .unwrap()
                .unwrap()
                .last_frame
                .unwrap()
                .section_id,
            SectionId("early".into())
        );
    }
    fn late_cleanup(project: &mut VideoProject) {
        project
            .creative
            .sections
            .get_mut(&SectionId("late".into()))
            .unwrap()
            .images
            .clear();
    }
}
