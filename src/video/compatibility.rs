//! Explicit older-format copies. Normal saves always retain the complete document.
use super::{contract::*, persistence::*};
use serde::Serialize;
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub retained_private_layout_in_original: bool,
    pub retained_timeline_duration_in_original: bool,
    pub project_id: ProjectId,
    pub materialized_settings: usize,
    pub omitted_asset_labels: usize,
    pub omitted_library_links: usize,
    pub omitted_image_annotations: usize,
    pub omitted_section_media_orders: usize,
    pub omitted_capture_provenance: usize,
    pub retained_newer_authoring_drafts_in_original: usize,
    pub retained_sampling_drafts_in_original: usize,
    pub retained_scene_types_in_original: usize,
    pub retained_inactive_clips_in_original: usize,
}
pub fn project(source: &ProjectDocument) -> Result<(ProjectDocument, Report), StorageError> {
    if source.resources.iter().any(|r| r.source.is_some()) {
        return Err(StorageError::new(
            "older_format_external_reference_not_representable",
        ));
    }
    if source.takes.iter().any(|take| take.provenance.is_none()) {
        return Err(StorageError::new(
            "older_format_imported_take_not_representable",
        ));
    }
    // The frozen reader has no LoRA/upscale schema. Do not label a changed
    // generation recipe or stripped Take provenance as an equivalent old copy.
    let active_loras = source.project.scenes.iter().any(|scene| {
        !imagine_video_contract::loras::resolve(
            &source.project.creative.h3_loras.clone().unwrap_or_default(),
            source.project.creative.scene_loras.get(&scene.id),
        )
        .active()
        .is_empty()
    });
    let upscale = source
        .project
        .scenes
        .iter()
        .filter_map(|scene| source.project.effective_settings(scene))
        .any(|slot| {
            slot.values
                .pointer("/stage_two/handoff_upscale_factor")
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|factor| factor != 1.)
        });
    let retained_upscale_tail = source.resources.iter().any(|resource| {
        matches!(&resource.kind, ResourceKind::H3Motion { descriptor, .. } if descriptor.handoff_upscale.is_some())
    });
    if active_loras
        || upscale
        || retained_upscale_tail
        || source.takes.iter().any(|take| {
            take.provenance
                .as_ref()
                .is_some_and(|p| p.resolved_loras.is_some())
        })
    {
        return Err(StorageError::new(
            "older_format_generation_not_representable",
        ));
    }
    // The frozen reader rejects the newer boundary fields and has no applied
    // conditioning receipt. An explicit older-format copy must not erase those
    // authored choices or claim an unsupported recipe. Normal Save is unchanged.
    let authored_boundary = source.project.assembly.boundaries.iter().any(|b| {
        b.tail_frames.is_some()
            || !b.transition_prompt.is_empty()
            || !b.reuse_visual_references
            || !b.reuse_style
            || b.audio_continuity
                != imagine_video_contract::assembly::AudioContinuity::NewSceneAudio
    });
    if authored_boundary
        || source.takes.iter().any(|take| {
            take.provenance
                .as_ref()
                .is_some_and(|p| p.boundary.is_some())
        })
    {
        return Err(StorageError::new("older_format_boundary_not_representable"));
    }
    if source.takes.iter().any(|take| {
        take.provenance
            .as_ref()
            .is_some_and(|p| p.seed > MAX_INTEGER)
    }) {
        return Err(StorageError::new("older_format_seed_out_of_range"));
    }
    let mut copy = source.clone();
    copy.storage_layout = None;
    // The old reader interprets every receipt as an attached Take. Refuse that
    // explicit export when it would reattach removed results; normal Save is free.
    let detached: std::collections::BTreeSet<_> = copy
        .captures
        .iter()
        .filter(|(_, c)| c.detached || c.removed)
        .map(|(id, _)| id.clone())
        .collect();
    if !detached.is_empty() {
        return Err(StorageError::new("older_format_captures_not_representable"));
    }
    copy.captures.clear();
    copy.project.id = ProjectId::generate();
    let mut report = Report {
        retained_private_layout_in_original: source.storage_layout.is_some(),
        retained_timeline_duration_in_original: source.project.creative.timeline_duration.is_some(),
        project_id: copy.project.id.clone(),
        materialized_settings: 0,
        omitted_asset_labels: 0,
        omitted_library_links: 0,
        omitted_image_annotations: 0,
        omitted_section_media_orders: 0,
        omitted_capture_provenance: source
            .captures
            .values()
            .filter(|c| c.source_capture.is_some())
            .count(),
        retained_newer_authoring_drafts_in_original: 0,
        retained_sampling_drafts_in_original: 0,
        retained_scene_types_in_original: source.project.creative.scene_kinds.len(),
        retained_inactive_clips_in_original: 0,
    };
    for scene in &mut copy.project.scenes {
        let original = source.project.scene(&scene.id)?;
        let model = source.project.effective_model(original);
        if !scene.settings.iter().any(|s| &s.model == model)
            && let Some(settings) = source.project.effective_settings(original)
        {
            scene.settings.push(settings.clone());
            report.materialized_settings += 1;
        }
    }
    copy.project.generation = None;
    // The complete original keeps the explicit authoring duration. The older
    // reader derives extent from its Scenes and has no field for this choice.
    copy.project.creative.timeline_duration = None;
    copy.project.assembly.clips.retain(|id, _| {
        let active =
            source.project.scene_kind(id) == imagine_video_contract::creative::SceneKind::Imported;
        report.retained_inactive_clips_in_original += usize::from(!active);
        active
    });
    copy.project.creative.scene_kinds.clear();
    // Preset provenance is a newer field; the scene settings it describes remain.
    copy.project.creative.scene_presets.clear();
    copy.project.creative.h3_loras = None;
    copy.project.creative.scene_loras.clear();
    for scene in &mut copy.project.scenes {
        for settings in &mut scene.settings {
            if settings.model.adapter == "h3"
                && let Some(values) = settings.values.as_object_mut()
            {
                report.retained_sampling_drafts_in_original +=
                    usize::from(values.remove("stage_two_draft").is_some());
                if let Some(second) = values
                    .get_mut("stage_two")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    second.remove("handoff_upscale_factor");
                }
            }
        }
    }
    for asset in copy.project.assets.values_mut() {
        report.omitted_asset_labels += usize::from(!asset.name.is_empty());
        asset.name.clear();
        report.omitted_library_links += usize::from(asset.library_source.take().is_some());
    }
    for section in copy.project.creative.sections.values_mut() {
        report.omitted_section_media_orders += usize::from(!section.media_order.is_empty());
        section.media_order.clear();
        for image in &mut section.images {
            report.omitted_image_annotations += usize::from(image.anchor.take().is_some());
        }
    }
    copy.authoring.retain_mut(|draft| {
        if draft
            .request
            .scene_field
            .is_some_and(|f| f != imagine_video_contract::authoring::SceneField::Description)
        {
            report.retained_newer_authoring_drafts_in_original += 1;
            return false;
        }
        draft.request.scene_field = None;
        true
    });
    Ok((copy, report))
}
#[cfg(test)]
use crate::task_fixtures;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compatible_export_is_independent_and_failure_preserves_active_project_and_disk() {
        let root = tempfile::tempdir().unwrap();
        assert!(task_fixtures::is_owned(root.path()));
        let g = imagine_video_contract::generation::Defaults::default();
        let mut p = VideoProject::create(CreateProject {
            generation: Some(g),
            schema_version: 1,
            default_model: imagine_video_contract::generation::h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        p.scenes.push(Scene {
            id: SceneId("own".into()),
            start: Time { num: 0, den: 1 },
            duration: Time { num: 5, den: 1 },
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        let mut session = ProjectSession::new(p).unwrap();
        let original = root.path().join("original");
        std::fs::create_dir(&original).unwrap();
        session.save_to(&original).unwrap();
        let document = session.document().clone();
        let status = session.status().clone();
        let bytes = std::fs::read(original.join("project.json")).unwrap();
        let occupied = root.path().join("occupied");
        std::fs::create_dir(&occupied).unwrap();
        std::fs::write(occupied.join("marker"), b"own fixture").unwrap();
        assert_eq!(
            session.export_compatible(&occupied).unwrap_err().code,
            "storage_not_empty"
        );
        let target = root.path().join("compatible");
        std::fs::create_dir(&target).unwrap();
        let report = session.export_compatible(&target).unwrap();
        assert_eq!(report.materialized_settings, 1);
        assert_eq!(session.document(), &document);
        let mut retained_tail = document.clone();
        let shape = crate::h3_protocol::Shape {
            width: 832,
            height: 448,
            frames: 124,
            fps: 24,
        };
        retained_tail.resources.push(Resource {
            id: "own-upscaled-tail".into(),
            source: None,
            sha256: "1".repeat(64),
            bytes: 1,
            kind: ResourceKind::H3Motion {
                take_id: TakeId("own-tail-take".into()),
                model: document.project.default_model.clone(),
                descriptor: crate::h3_protocol::continuation::Descriptor {
                    schema_version: 1,
                    shape,
                    sampled_frames: 124,
                    visible_end_frame: 124,
                    selected_end_frame: 124,
                    handoff_upscale: Some(crate::h3_protocol::continuation::HandoffUpscale {
                        base_width: 416,
                        base_height: 224,
                        factor: 2.,
                    }),
                },
            },
        });
        let before = serde_json::to_value(&retained_tail).unwrap();
        assert_eq!(
            project(&retained_tail).unwrap_err().code,
            "older_format_generation_not_representable"
        );
        assert_eq!(serde_json::to_value(&retained_tail).unwrap(), before);
        assert_eq!(session.status(), &status);
        assert_eq!(std::fs::read(original.join("project.json")).unwrap(), bytes);
        let copy = ProjectSession::open(&target).unwrap();
        assert!(copy.document().project.generation.is_none());
        assert_ne!(copy.document().project.id, document.project.id);
        assert_eq!(
            copy.document()
                .project
                .effective_settings(&copy.document().project.scenes[0]),
            document
                .project
                .effective_settings(&document.project.scenes[0])
        );
        let mut boundary_copy = document.clone();
        let boundary = imagine_video_contract::assembly::Boundary::new(
            SceneId("own".into()),
            SceneId("other".into()),
        );
        boundary_copy.project.assembly.boundaries.push(boundary);
        assert!(project(&boundary_copy).is_ok());
        boundary_copy.project.assembly.boundaries[0].tail_frames = Some(23);
        let authored = boundary_copy.clone();
        assert_eq!(
            project(&boundary_copy).unwrap_err().code,
            "older_format_boundary_not_representable"
        );
        assert_eq!(boundary_copy, authored);
        boundary_copy.project.assembly.boundaries.clear();
        boundary_copy.takes.push(StoredTake {
            id: TakeId("own-generated".into()),
            scene_id: SceneId("own".into()),
            status: TakeStatus::Candidate,
            provenance: Some(TakeProvenance {
                boundary: Some(serde_json::json!({"applied":true})),
                resolved_loras: None,
                model: document.project.default_model.clone(),
                seed: 42,
                input_revision: String::new(),
                compiled_prompt: String::new(),
                settings: document
                    .project
                    .effective_settings(&document.project.scenes[0])
                    .unwrap()
                    .clone(),
                references: vec![],
                source_take: None,
                continuity: None,
                authored: None,
            }),
        });
        assert_eq!(
            project(&boundary_copy).unwrap_err().code,
            "older_format_boundary_not_representable"
        );
        let mut newer = document.clone();
        newer.project.creative.h3_loras = Some(imagine_video_contract::loras::Stack {
            loras: vec![imagine_video_contract::loras::Row {
                name: "/own/compatible.weights".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(
            project(&newer).unwrap_err().code,
            "older_format_generation_not_representable"
        );
        assert_eq!(session.document(), &document);
    }
}
