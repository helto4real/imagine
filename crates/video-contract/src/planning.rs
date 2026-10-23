//! Authoring validation and input fingerprints; executable H3 plans belong to P05.
use super::contract::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Unvalidated,
    SyntheticVerified,
    RuntimeVerified,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub revision: String,
    pub runtime_available: bool,
    pub experimental: bool,
    pub validation_status: ValidationStatus,
    pub input_roles: BTreeMap<InputRole, BTreeSet<MediaKind>>,
    pub continuity: BTreeSet<ContinuityMethod>,
    pub incompatible_roles: Vec<(InputRole, InputRole)>,
}
/// Registered Rust adapters, with no inference or queue dependency.
pub trait VideoModelAdapter: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self, model: &ModelSelection) -> Option<Capabilities>;
    fn validate_settings(&self, model: &ModelSelection, settings: &SettingsSlot) -> Vec<Issue>;
}
#[derive(Default)]
pub struct AdapterRegistry {
    adapters: BTreeMap<String, Box<dyn VideoModelAdapter>>,
}
impl AdapterRegistry {
    pub fn register(&mut self, adapter: Box<dyn VideoModelAdapter>) -> Result<(), Issue> {
        let id = adapter.id().to_owned();
        if id.is_empty() || self.adapters.contains_key(&id) {
            return Err(Issue::new("adapter_registration_conflict", "adapter"));
        }
        self.adapters.insert(id, adapter);
        Ok(())
    }
    pub fn get(&self, id: &str) -> Option<&dyn VideoModelAdapter> {
        self.adapters.get(id).map(Box::as_ref)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    pub input_revision: String,
    pub model: ModelSelection,
    pub capability_revision: Option<String>,
    pub runtime_available: bool,
    pub blockers: Vec<Issue>,
}
impl ValidationReport {
    pub fn runnable(&self) -> bool {
        self.blockers.is_empty()
    }
}

pub fn validate_scene(
    project: &VideoProject,
    id: &SceneId,
    registry: &AdapterRegistry,
) -> Result<ValidationReport, Issue> {
    project.validate()?;
    let scene = project.scene(id)?;
    let model = project.effective_model(scene).clone();
    let mut report = ValidationReport {
        input_revision: input_revision(project, id)?,
        model: model.clone(),
        capability_revision: None,
        runtime_available: false,
        blockers: vec![],
    };
    let mut block =
        |code: &str, field: &str| report.blockers.push(Issue::new(code, field).scene(id));
    let Some(adapter) = registry.get(&model.adapter) else {
        block("adapter_missing", "model.adapter");
        return Ok(report);
    };
    let Some(caps) = adapter.capabilities(&model) else {
        block("model_unsupported", "model");
        return Ok(report);
    };
    report.capability_revision = Some(caps.revision);
    report.runtime_available = caps.runtime_available;
    if !caps.runtime_available {
        report
            .blockers
            .push(Issue::new("runtime_unavailable", "model").scene(id));
    }
    if let Some(settings) = scene.settings.iter().find(|s| s.model == model) {
        report.blockers.extend(
            adapter
                .validate_settings(&model, settings)
                .into_iter()
                .map(|i| i.scene(id)),
        );
    } else {
        report
            .blockers
            .push(Issue::new("model_settings_required", "settings").scene(id));
    }
    let active: Vec<_> = scene.bindings.iter().filter(|b| b.enabled).collect();
    for binding in &active {
        let asset = &project.assets[&binding.asset];
        if !caps
            .input_roles
            .get(&binding.role)
            .is_some_and(|kinds| kinds.contains(&asset.kind))
        {
            report.blockers.push(
                Issue::new(
                    "input_role_unsupported",
                    &format!("bindings.{}", binding.id.0),
                )
                .scene(id),
            );
        }
    }
    let mut active_roles = active
        .iter()
        .map(|b| b.role.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for clip in crate::audio::generation_clips(project, scene) {
        let role = if clip.reference.purpose == AudioPurpose::VoiceTimbre {
            InputRole::VoiceReference
        } else {
            InputRole::AudioReference
        };
        active_roles.insert(role.clone());
        if clip.asset.is_none() {
            report
                .blockers
                .push(Issue::new("audio_reference_missing", "audio_tracks").scene(id));
        }
        if !caps
            .input_roles
            .get(&role)
            .is_some_and(|kinds| kinds.contains(&MediaKind::Audio))
        {
            report
                .blockers
                .push(Issue::new("input_role_unsupported", "audio_tracks").scene(id));
        }
    }
    for (a, b) in &caps.incompatible_roles {
        if active_roles.contains(a) && active_roles.contains(b) {
            report
                .blockers
                .push(Issue::new("input_combination_unsupported", "bindings").scene(id));
        }
    }
    if let Some(c) = &scene.continuity {
        if !caps.continuity.contains(&c.method) {
            report
                .blockers
                .push(Issue::new("continuity_unsupported", "continuity.method").scene(id));
        }
        if project.scene(&c.source_scene)?.accepted_take.is_none() {
            report
                .blockers
                .push(Issue::new("continuity_source_missing", "continuity.source_scene").scene(id));
        } else if matches!(
            c.method,
            ContinuityMethod::MotionContext | ContinuityMethod::MotionContextExp
        ) {
            // A model match alone cannot prove latent compatibility. P05 must verify artifacts.
            report
                .blockers
                .push(Issue::new("continuity_artifact_unverified", "continuity").scene(id));
        }
    }
    Ok(report)
}

pub fn input_revision(project: &VideoProject, id: &SceneId) -> Result<String, Issue> {
    project.validate()?;
    let scene = project.scene(id)?;
    let model = project.effective_model(scene);
    let settings = scene.settings.iter().find(|s| &s.model == model);
    let bindings: Vec<_> = scene
        .bindings
        .iter()
        .filter(|b| b.enabled)
        .map(|b| (b, &project.assets[&b.asset]))
        .collect();
    let source = scene
        .continuity
        .as_ref()
        .map(|c| project.scene(&c.source_scene).map(|s| &s.accepted_take))
        .transpose()?;
    let mut sections = scene.sections.clone();
    for section in &mut sections {
        section.start = section.start.normalized();
        section.duration = section.duration.normalized();
    }
    // Include effective intent, not project revision, inactive model slots or own Take status.
    let audio_references=crate::audio::generation_clips(project,scene).into_iter().map(|clip|serde_json::json!({"id":clip.id,"asset":clip.asset.as_ref().and_then(|id|project.assets.get(id)),"start":clip.start.normalized(),"duration":clip.duration.normalized(),"source_in":clip.source_in.normalized(),"source_out":clip.source_out.map(Time::normalized),"reference":clip.reference})).collect::<Vec<_>>();
    let mut value = serde_json::json!({
        "schema":project.schema_version,"global_prompt":project.global_prompt,
        "model":model,"settings":settings,"description":scene.description,
        "start":scene.start.normalized(),"duration":scene.duration.normalized(),"sections":sections,
        "bindings":bindings,"continuity":scene.continuity,"source":source,
        "delivery":project.delivery,"audio_references":audio_references,
    });
    let audio_manifest = crate::audio::references::entries(project, scene);
    if !audio_manifest.is_empty() {
        value["audio_reference_manifest"] = serde_json::json!(audio_manifest);
    }
    if let Some(clip) = project.assembly.clips.get(id) {
        value["imported_clip"] = serde_json::json!(clip);
    }
    let creative = scene
        .sections
        .iter()
        .filter_map(|s| {
            project
                .creative
                .sections
                .get(&s.id)
                .map(|source| (&s.id, source))
        })
        .collect::<BTreeMap<_, _>>();
    if !creative.is_empty() {
        value["creative_sections"] = serde_json::json!(creative);
    }
    if let Some(enabled) = project.creative.visual_references_enabled {
        value["visual_references_enabled"] = serde_json::json!(enabled);
    }
    if let Some(global) = &project.creative.global_prompt {
        value["global_prompt_options"] = serde_json::json!(global);
    }
    if let Some(source) = project.creative.scene_prompts.get(id)
        && [
            &source.summary,
            &source.ambient_sound,
            &source.non_diegetic_music,
        ]
        .iter()
        .any(|s| !s.is_empty())
    {
        value["scene_prompt"] = serde_json::json!({"summary":source.summary,"ambient_sound":source.ambient_sound,"non_diegetic_music":source.non_diegetic_music});
    }
    let source = std::iter::once(scene.description.as_str())
        .chain(scene.sections.iter().map(|s| s.prompt.as_str()))
        .chain(std::iter::once(project.global_prompt.as_str()))
        .chain(
            project
                .creative
                .scene_prompts
                .get(id)
                .into_iter()
                .flat_map(|s| {
                    [
                        s.summary.as_str(),
                        s.ambient_sound.as_str(),
                        s.non_diegetic_music.as_str(),
                    ]
                }),
        )
        .collect::<Vec<_>>()
        .join("\n");
    let aliases = crate::references::aliases(&source);
    let cast = crate::creative::selected_references(&project.creative, &aliases);
    if !cast.is_empty() {
        value["cast"] = serde_json::json!(&cast);
    }
    let mut creative_assets = BTreeSet::new();
    for source in creative.values() {
        creative_assets.extend(source.images.iter().map(|b| &b.asset));
        creative_assets.extend(source.videos.iter().filter(|b| b.enabled).map(|b| &b.asset));
    }
    if model.variant != "i2v" {
        for reference in &cast {
            creative_assets.extend(
                reference
                    .images
                    .iter()
                    .filter(|i| i.enabled)
                    .map(|i| &i.asset),
            );
            if let Some(voice) = &reference.voice {
                creative_assets.extend(
                    voice
                        .assets
                        .iter()
                        .filter(|id| voice.reference_settings.get(*id).is_none_or(|s| s.enabled)),
                );
            }
        }
    }
    if !creative_assets.is_empty() {
        value["creative_assets"] = serde_json::json!(
            creative_assets
                .into_iter()
                .filter_map(|id| project.assets.get(id).map(|asset| (id, asset)))
                .collect::<BTreeMap<_, _>>()
        );
    }
    canonicalize(&mut value);
    let bytes =
        serde_json::to_vec(&value).map_err(|_| Issue::new("invalid_document", "project"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum H3Handoff {
    ContinueLatent,
    ReNoiseVideoPrediction,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct H3Stage {
    pub steps: u32,
    pub sampler: String,
    pub scheduler: String,
    #[serde(default = "full_denoise")]
    pub denoise: f64,
    #[serde(default)]
    pub sigmas: Vec<f32>,
}
fn full_denoise() -> f64 {
    1.
}
fn base_steps() -> u32 {
    4
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct H3SecondStage {
    pub sampling: H3Stage,
    pub model: ModelSelection,
    pub handoff: H3Handoff,
    pub trained_latent_upscale_model: Option<String>,
}
/// No image CFG/negative-conditioning defaults. Exact runtime constraints belong to P05.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct H3Settings {
    #[serde(default = "audio_reference_mode")]
    pub audio_reference_mode: String,
    #[serde(default = "reference_quality")]
    pub ref_image_size: String,
    pub stage_one: H3Stage,
    pub stage_two: Option<H3SecondStage>,
    #[serde(default = "base_steps")]
    pub base_steps_to_run: u32,
}
fn reference_quality() -> String {
    "Standard".into()
}
fn audio_reference_mode() -> String {
    "Timeline Audio References".into()
}
impl H3Settings {
    pub fn validate(&self) -> Result<(), Issue> {
        let valid = |stage: &H3Stage| {
            stage.steps > 0
                && stage.steps <= 10_000
                && stage.denoise.is_finite()
                && stage.denoise > 0.
                && stage.denoise <= 1.
                && stage
                    .sigmas
                    .iter()
                    .all(|sigma| sigma.is_finite() && *sigma >= 0. && *sigma <= 1.)
                && !stage.sampler.trim().is_empty()
                && !stage.scheduler.trim().is_empty()
        };
        if !valid(&self.stage_one) || self.stage_two.as_ref().is_some_and(|s| !valid(&s.sampling)) {
            return Err(Issue::new("invalid_sampling", "settings"));
        }
        if let Some(s) = &self.stage_two
            && s.trained_latent_upscale_model
                .as_ref()
                .is_some_and(|m| m.trim().is_empty())
        {
            return Err(Issue::new("invalid_upscale_model", "settings.stage_two"));
        }
        Ok(())
    }
}

/// Schemas generated from the same DTOs used at the service boundary.
pub fn schemas() -> serde_json::Value {
    serde_json::json!({
        "create_project": schemars::schema_for!(CreateProject),
        "edit_request": schemars::schema_for!(super::editing::EditRequest),
        "review_request": schemars::schema_for!(super::review::ReviewRequest),
        "take_state": schemars::schema_for!(super::review::TakeState),
        "run_request": schemars::schema_for!(super::runs::Request),
        "run": schemars::schema_for!(super::runs::Run),
        "assembly_request":schemars::schema_for!(super::assembly::Request),
        "assembly_plan":schemars::schema_for!(super::assembly::Plan),
        "project": schemars::schema_for!(VideoProject),
        "capabilities": schemars::schema_for!(Capabilities),
        "h3_settings": schemars::schema_for!(H3Settings),
        "validation_report": schemars::schema_for!(ValidationReport),
    })
}

fn canonicalize(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.sort_keys();
            for child in map.values_mut() {
                canonicalize(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                canonicalize(child);
            }
        }
        _ => {}
    }
}
