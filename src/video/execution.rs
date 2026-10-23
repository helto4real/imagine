//! Frozen scene admission. Project intent is compiled once; queued edits cannot alter inputs.
use super::{contract::*, persistence::*, planning, service::Service};
use crate::{engine::Backend, generation::Generated, h3_protocol as h3, storage::Store};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    VideoScene,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub kind: Kind,
    pub schema_version: u32,
    pub project_id: ProjectId,
    pub expected_project_revision: u64,
    pub scene_id: SceneId,
    pub seed: u64,
}
pub struct Target {
    pub project_id: ProjectId,
    pub take: StoredTake,
    pub references: Vec<(Resource, File)>,
}
pub struct Frozen {
    pub target: Target,
    pub request: h3::Request,
    pub first_frame: Option<super::reference_inputs::FrozenFile>,
    pub reference_inputs: Vec<super::reference_inputs::Media>,
}
fn issue(code: &str) -> StorageError {
    StorageError::new(code)
}
pub fn freeze(
    service: &Service,
    request: &Admission,
    staging: &SessionStaging,
) -> Result<Frozen, StorageError> {
    if request.schema_version != 1 {
        return Err(issue("unsupported_schema"));
    }
    let session = service.projects.session(&request.project_id)?;
    let project = &session.document().project;
    if project.project_revision != request.expected_project_revision {
        return Err(issue("revision_conflict"));
    }
    let scene = project.scene(&request.scene_id)?;
    let compiled = super::adapters::h3::bindings::compile(project, scene)?;
    let reference_mode = project.effective_model(scene).variant == "reference";
    if let Some(code) = compiled.runtime_issues.first() {
        return Err(issue(code));
    }
    for (index, section) in scene
        .sections
        .iter()
        .enumerate()
        .filter(|_| !reference_mode)
    {
        if let Some(source) = project.creative.sections.get(&section.id) {
            for image in &source.images {
                if index != 0
                    || image.role != InputRole::FirstFrame
                    || !scene.bindings.iter().any(|b| {
                        b.enabled && b.asset == image.asset && b.role == InputRole::FirstFrame
                    })
                {
                    return Err(issue("h3_section_reference_unavailable"));
                }
            }
        }
    }
    if project.scene_kind(&scene.id) == imagine_video_contract::creative::SceneKind::Imported {
        return Err(issue("imported_scene_not_generatable"));
    }
    if !reference_mode
        && !imagine_video_contract::audio::generation_clips(project, scene).is_empty()
    {
        return Err(issue("h3_i2v_audio_reference_unavailable"));
    }
    let model = project.effective_model(scene).clone();
    let profile = super::adapters::h3::profile(&model)?;
    let mut settings = project
        .effective_settings(scene)
        .ok_or_else(|| issue("model_settings_required"))?
        .clone();
    if settings.schema_version != 1 {
        return Err(issue("settings_schema_unsupported"));
    }
    let mut sampling: planning::H3Settings =
        serde_json::from_value(settings.values.clone()).map_err(|_| issue("invalid_sampling"))?;
    sampling.stage_two_draft = None;
    let reference_quality = sampling.ref_image_size.clone();
    let upscale = sampling
        .stage_two
        .as_ref()
        .filter(|s| {
            s.handoff_upscale_factor > 1.
                && s.handoff == planning::H3Handoff::ReNoiseVideoPrediction
        })
        .map(|s| {
            (
                s.handoff_upscale_factor,
                s.trained_latent_upscale_model.clone(),
            )
        });
    let mut options = super::adapters::h3::options(&sampling);
    let resolved_loras = imagine_video_contract::loras::resolve(
        &project.creative.h3_loras.clone().unwrap_or_default(),
        project.creative.scene_loras.get(&scene.id),
    );
    options.loras = resolved_loras
        .active()
        .into_iter()
        .map(|row| h3::options::Lora {
            name: row.name,
            strength: row.strength_model,
        })
        .collect();
    let inherited_stage_two_model = planning::inherited_stage_two_model(project, scene, &sampling)?;
    if let Some(second) = &mut sampling.stage_two {
        let selected = if second.model.variant == "i2v" {
            &mut second.i2v_model_file
        } else {
            &mut second.reference_model_file
        };
        if selected.as_ref().is_none_or(|s| s.trim().is_empty()) {
            *selected = inherited_stage_two_model.clone();
        }
    }
    settings.values = serde_json::to_value(&sampling).map_err(|_| issue("invalid_sampling"))?;
    let mut sampling = super::adapters::h3::sampling(&sampling)?;
    if let Some(second) = &mut sampling.second
        && second.model_file.is_none()
    {
        second.model_file = inherited_stage_two_model;
    }
    let selection = h3::Selection {
        seconds: scene.duration.num as f64 / scene.duration.den as f64,
        ..Default::default()
    };
    let frames = selection.resolve().map_err(issue)?.frames;
    let shape = h3::Shape {
        width: project.delivery.width,
        height: project.delivery.height,
        frames,
        fps: 24,
    };
    if !shape.valid() {
        return Err(issue("unsupported_h3_geometry"));
    }
    let mut output_shape = shape;
    if let Some((factor, model)) = upscale {
        let settings = h3::sampling::Upscale {
            factor,
            model,
            base_width: shape.width,
            base_height: shape.height,
        };
        output_shape = settings
            .shape(shape)
            .ok_or_else(|| issue("unsupported_h3_geometry"))?;
        sampling
            .second
            .as_mut()
            .ok_or_else(|| issue("invalid_sampling"))?
            .upscale = Some(settings);
    }
    let mut references = vec![];
    let mut versions = vec![];
    let endpoint = super::artifacts::freeze_endpoint(session, scene, staging)?;
    let source = endpoint.as_ref().map(|endpoint| endpoint.source.clone());
    let mut motion = None;
    let mut motion_bytes = vec![];
    let mut first_frame = if let Some(endpoint) = endpoint {
        let frame = endpoint
            .file
            .try_clone()
            .map_err(|_| issue("storage_io_failed"))?;
        if let Some(input) = endpoint.motion {
            if !input.valid_sampling(output_shape, Some(&sampling)) || !scene.bindings.is_empty() {
                return Err(issue("continuity_endpoint_incompatible"));
            }
            super::bundle::stream(&frame, &mut motion_bytes)?;
            motion = Some(input);
            references.push((endpoint.resource, endpoint.file));
            None
        } else {
            references.push((endpoint.resource, endpoint.file));
            Some(super::reference_inputs::FrozenFile::owned(frame)?)
        }
    } else {
        None
    };
    if first_frame.is_some()
        && scene
            .bindings
            .iter()
            .any(|binding| binding.enabled && binding.role == InputRole::FirstFrame)
    {
        return Err(issue("continuity_first_frame_conflict"));
    }
    for binding in scene
        .bindings
        .iter()
        .filter(|b| b.enabled && !reference_mode)
    {
        let asset = &project.assets[&binding.asset];
        if binding.role != InputRole::FirstFrame
            || asset.kind != MediaKind::Image
            || first_frame.is_some()
        {
            return Err(issue("scene_runtime_unsupported"));
        }
        let resource = session.document().resources.iter().find(|r| matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id==&asset.id && *revision==asset.revision)).ok_or_else(||issue("reference_resource_missing"))?;
        let frozen = freeze_first_frame(session, resource, staging)?;
        references.push((
            resource.clone(),
            frozen
                .file()
                .try_clone()
                .map_err(|_| issue("storage_io_failed"))?,
        ));
        first_frame = Some(frozen);
        versions.push(AssetVersion {
            asset_id: asset.id.clone(),
            revision: asset.revision,
        });
    }
    let has_boundary = compiled.reference_manifest["boundary_dependency"].is_object();
    let reference_inputs = if reference_mode || has_boundary {
        super::reference_inputs::freeze(
            session,
            &compiled.reference_manifest,
            staging,
            &mut references,
            &mut versions,
        )?
    } else {
        vec![]
    };
    let expected = if reference_mode {
        super::job::PROFILE
    } else {
        super::job::I2V_PROFILE
    };
    if !reference_mode && first_frame.is_none() && !has_boundary {
        return Err(issue("scene_runtime_unsupported"));
    }
    if model.profile != expected {
        return Err(issue("scene_profile_mismatch"));
    }
    let prompt = compiled.prompt;
    let input = h3::Request {
        preview: None,
        options,
        capture_continuation: shape.frames >= 22,
        continuation: motion,
        continuation_bytes: motion_bytes,
        references: if reference_mode || has_boundary {
            h3::references::Input {
                quality: reference_quality,
                manifest: compiled.reference_manifest,
                ..Default::default()
            }
        } else {
            Default::default()
        },
        protocol: h3::VERSION,
        kind: h3::RequestKind::GenerateVideo,
        id: uuid::Uuid::new_v4().to_string(),
        profile,
        shape: output_shape,
        image: None,
        image_rgb: vec![],
        prompt: prompt.clone(),
        seed: request.seed,
        sampling: Some(sampling),
    };
    if !input.valid_settings() {
        return Err(issue("invalid_h3_request"));
    }
    Ok(Frozen {
        target: Target {
            project_id: project.id.clone(),
            take: StoredTake {
                id: TakeId(format!("take_{}", uuid::Uuid::new_v4().simple())),
                scene_id: scene.id.clone(),
                status: TakeStatus::Candidate,
                provenance: Some(TakeProvenance {
                    boundary: has_boundary.then(||serde_json::json!({"dependency":input.references.manifest["boundary_dependency"],"applied":false})),
                    resolved_loras: (!input.options.loras.is_empty()).then_some(resolved_loras),
                    model,
                    seed: request.seed,
                    input_revision: planning::input_revision(project, &scene.id)?,
                    compiled_prompt: prompt,
                    settings,
                    references: versions,
                    source_take: source.as_ref().map(|source| source.take_id.clone()).or_else(||
                        has_boundary.then(||input.references.manifest["videos"][0]["take_id"].as_str()
                            .map(|id|TakeId(id.into()))).flatten()),
                    continuity: source,
                }),
            },
            references,
        },
        request: input,
        first_frame,
        reference_inputs,
    })
}
impl Frozen {
    pub fn summary(&self) -> serde_json::Value {
        let provenance = self
            .target
            .take
            .provenance
            .as_ref()
            .expect("frozen generation recipe");
        serde_json::json!({"kind":"video_scene","schema_version":1,"project_id":self.target.project_id,"scene_id":self.target.take.scene_id,"take_id":self.target.take.id,"input_revision":provenance.input_revision,"model":provenance.model,"seed":self.request.seed,"shape":self.request.shape})
    }
}
fn freeze_first_frame(
    session: &ProjectSession,
    resource: &Resource,
    staging: &SessionStaging,
) -> Result<super::reference_inputs::FrozenFile, StorageError> {
    let mut source = session.resource_file(&resource.id)?;
    let file = if resource.source.is_some() {
        source
    } else {
        staging.copy(&mut source)?
    };
    super::reference_inputs::FrozenFile::resource(resource, file)
}
fn decode_first_frame(
    file: &super::reference_inputs::FrozenFile,
) -> anyhow::Result<(h3::ImageInput, Vec<u8>)> {
    file.decode(|file| {
        anyhow::ensure!(
            file.metadata()?.len() <= h3::MAX_INPUT_IMAGE as u64,
            crate::engine::EngineError("invalid_h3_start_image")
        );
        let mut bytes = vec![];
        // Independent of the descriptor cursor, including shared staged handles.
        super::bundle::stream(file, &mut bytes)?;
        super::image::decode(&bytes)
    })
}
pub async fn generate(
    engine: &mut dyn Backend,
    mut input: h3::Request,
    first_frame: Option<super::reference_inputs::FrozenFile>,
    reference_inputs: Vec<super::reference_inputs::Media>,
    proof: serde_json::Value,
    store: &Store,
) -> anyhow::Result<Generated> {
    let pending = store
        .destination_for_proof(&proof)
        .and_then(|d| d.reserve(&proof, input.shape.raw_bytes() + 64 * 1024 * 1024))
        .map_err(crate::engine::EngineError)?;
    if let Some(file) = first_frame {
        let (descriptor, rgb) =
            tokio::task::spawn_blocking(move || decode_first_frame(&file)).await??;
        input.image = Some(descriptor);
        input.image_rgb = rgb;
    }
    let prepared = super::reference_inputs::prepare(
        input.references,
        reference_inputs,
        input
            .sampling
            .as_ref()
            .map_or(input.shape, |s| s.input_shape(input.shape))
            .frames,
    )
    .await?;
    input.references = prepared.input;
    if let Some(frame) = prepared.frame0 {
        input.image = Some(h3::ImageInput {
            width: frame.width,
            height: frame.height,
            sha256: crate::storage::hash(&frame.rgb),
        });
        input.image_rgb = frame.rgb;
    }
    let mut av = engine.generate_video(&input).await?;
    anyhow::ensure!(
        av.shape == input.shape,
        crate::engine::EngineError("h3_invalid_output")
    );
    engine.stage(Some("encoding_video"));
    let mut take_artifacts = vec![super::artifacts::Produced::endpoint(&av)?];
    if let Some(receipt) = input.references.boundary_receipt() {
        take_artifacts.push(super::artifacts::Produced::BoundaryApplied(receipt));
    }
    if let Some(artifact) = av.continuation.take() {
        take_artifacts.push(super::artifacts::Produced::H3Motion(artifact));
    }
    let bytes = crate::generation::encode_video(av).await?;
    Ok(Generated::Video {
        bytes,
        take_artifacts,
        pending: Box::new(pending),
    })
}
pub fn staging(store: &Store) -> Result<SessionStaging, StorageError> {
    SessionStaging::select(&store.staging_directory().map_err(issue)?)
}
pub fn media_file(staging: &SessionStaging, bytes: &[u8]) -> Result<File, StorageError> {
    let mut file = staging.temporary()?;
    file.write_all(bytes)
        .map_err(|_| issue("storage_io_failed"))?;
    Ok(file)
}

#[cfg(test)]
mod external_first_frame_tests {
    use super::*;
    use std::os::unix::fs::{FileExt, MetadataExt};

    #[test]
    fn linked_first_frame_keeps_original_and_decodes_identical_rgb_then_rejects_queued_change() {
        let root =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let path = root.path().join("own-first-frame.png");
        image::RgbImage::from_pixel(16, 16, image::Rgb([50, 80, 120]))
            .save(&path)
            .unwrap();
        let mut file = File::open(&path).unwrap();
        let asset = AssetId("own-first".into());
        let resource = Resource::external(
            "own-first-resource".into(),
            ResourceKind::Reference {
                asset_id: asset.clone(),
                revision: 1,
            },
            &path,
            &mut file,
        )
        .unwrap();
        let project = VideoProject::create(
            serde_json::from_str(include_str!(
                "../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut session = ProjectSession::new(project).unwrap();
        let mut document = session.document().clone();
        document.project.assets.insert(
            asset.clone(),
            Asset {
                id: asset,
                revision: 1,
                kind: MediaKind::Image,
                name: String::new(),
                library_source: None,
            },
        );
        document.resources.push(resource.clone());
        let staging = SessionStaging::select(root.path()).unwrap();
        session
            .update(0, document, std::collections::BTreeMap::new(), &staging)
            .unwrap();
        let before = std::fs::read_dir(root.path()).unwrap().count();
        let frozen = freeze_first_frame(&session, &resource, &staging).unwrap();
        let original = std::fs::metadata(&path).unwrap();
        let selected = frozen.file().metadata().unwrap();
        assert_eq!(
            (selected.dev(), selected.ino()),
            (original.dev(), original.ino())
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), before);
        assert_eq!(
            session.document().resources[0].source.as_deref(),
            Some(path.as_path())
        );
        let (descriptor, rgb) = decode_first_frame(&frozen).unwrap();
        assert_eq!((descriptor.width, descriptor.height), (16, 16));
        assert_eq!(rgb, [50, 80, 120].repeat(256));
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(b"x", 0)
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), original.len());
        assert_eq!(
            decode_first_frame(&frozen).err().unwrap().to_string(),
            "source_changed"
        );
    }
}
