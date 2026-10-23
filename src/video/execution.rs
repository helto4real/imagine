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
    pub first_frame: Option<File>,
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
    if project.assembly.clips.contains_key(&scene.id) {
        return Err(issue("imported_scene_not_generatable"));
    }
    if !reference_mode
        && !imagine_video_contract::audio::generation_clips(project, scene).is_empty()
    {
        return Err(issue("h3_i2v_audio_reference_unavailable"));
    }
    let model = project.effective_model(scene).clone();
    let profile = super::adapters::h3::profile(&model)?;
    if request.seed > MAX_INTEGER {
        return Err(issue("scene_runtime_unsupported"));
    }
    let settings = scene
        .settings
        .iter()
        .find(|s| s.model == model)
        .ok_or_else(|| issue("model_settings_required"))?
        .clone();
    if settings.schema_version != 1 {
        return Err(issue("settings_schema_unsupported"));
    }
    let sampling: planning::H3Settings =
        serde_json::from_value(settings.values.clone()).map_err(|_| issue("invalid_sampling"))?;
    let reference_quality = sampling.ref_image_size.clone();
    let sampling = super::adapters::h3::sampling(&sampling)?;
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
    if !project.delivery.fps.equivalent(Time { num: 24, den: 1 }) || !shape.valid() {
        return Err(issue("unsupported_h3_geometry"));
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
            if !input.valid(shape) || !scene.bindings.is_empty() {
                return Err(issue("continuity_endpoint_incompatible"));
            }
            super::bundle::stream(&frame, &mut motion_bytes)?;
            motion = Some(input);
            references.push((endpoint.resource, endpoint.file));
            None
        } else {
            references.push((endpoint.resource, endpoint.file));
            Some(frame)
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
        let mut source = session.resource_file(&resource.id)?;
        let mut copy = staging.copy(&mut source)?;
        if Resource::inspect(resource.id.clone(), resource.kind.clone(), &mut copy)? != *resource {
            return Err(issue("resource_integrity_failed"));
        }
        first_frame = Some(copy.try_clone().map_err(|_| issue("storage_io_failed"))?);
        references.push((resource.clone(), copy));
        versions.push(AssetVersion {
            asset_id: asset.id.clone(),
            revision: asset.revision,
        });
    }
    let reference_inputs = if reference_mode {
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
    if !reference_mode && first_frame.is_none() {
        return Err(issue("scene_runtime_unsupported"));
    }
    if model.profile != expected {
        return Err(issue("scene_profile_mismatch"));
    }
    let prompt = compiled.prompt;
    let input = h3::Request {
        capture_continuation: shape.frames >= 22,
        continuation: motion,
        continuation_bytes: motion_bytes,
        references: if reference_mode {
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
        shape,
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
                provenance: TakeProvenance {
                    model,
                    seed: request.seed,
                    input_revision: planning::input_revision(project, &scene.id)?,
                    compiled_prompt: prompt,
                    settings,
                    references: versions,
                    source_take: source.as_ref().map(|source| source.take_id.clone()),
                    continuity: source,
                },
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
        serde_json::json!({"kind":"video_scene","schema_version":1,"project_id":self.target.project_id,"scene_id":self.target.take.scene_id,"take_id":self.target.take.id,"input_revision":self.target.take.provenance.input_revision,"model":self.target.take.provenance.model,"seed":self.request.seed,"shape":self.request.shape})
    }
}
pub async fn generate(
    engine: &mut dyn Backend,
    mut input: h3::Request,
    first_frame: Option<File>,
    reference_inputs: Vec<super::reference_inputs::Media>,
    proof: serde_json::Value,
    store: &Store,
) -> anyhow::Result<Generated> {
    let pending = store
        .videos
        .destination()
        .and_then(|d| d.reserve(&proof, input.shape.raw_bytes() + 64 * 1024 * 1024))
        .map_err(crate::engine::EngineError)?;
    if let Some(file) = first_frame {
        let (descriptor, rgb) = tokio::task::spawn_blocking(move || {
            anyhow::ensure!(
                file.metadata()?.len() <= h3::MAX_INPUT_IMAGE as u64,
                crate::engine::EngineError("invalid_h3_start_image")
            );
            let mut bytes = vec![];
            // Staging writes leave the descriptor at EOF. Read independently of its
            // cursor, as persistence/media delivery already do for shared handles.
            super::bundle::stream(&file, &mut bytes)?;
            super::image::decode(&bytes)
        })
        .await??;
        input.image = Some(descriptor);
        input.image_rgb = rgb;
    }
    input.references = super::reference_inputs::prepare(input.references, reference_inputs).await?;
    let mut av = engine.generate_video(&input).await?;
    anyhow::ensure!(
        av.shape == input.shape,
        crate::engine::EngineError("h3_invalid_output")
    );
    engine.stage(Some("encoding_video"));
    let mut take_artifacts = vec![super::artifacts::Produced::endpoint(&av)?];
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
    let proof = store
        .videos
        .destination()
        .and_then(|d| d.proof())
        .map_err(issue)?;
    SessionStaging::select(&proof.config.root)
}
pub fn media_file(staging: &SessionStaging, bytes: &[u8]) -> Result<File, StorageError> {
    let mut file = staging.temporary()?;
    file.write_all(bytes)
        .map_err(|_| issue("storage_io_failed"))?;
    Ok(file)
}
