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
    for (a, b) in &caps.incompatible_roles {
        if active.iter().any(|v| &v.role == a) && active.iter().any(|v| &v.role == b) {
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
    let mut value = serde_json::json!({
        "schema":project.schema_version,"global_prompt":project.global_prompt,
        "model":model,"settings":settings,"description":scene.description,
        "start":scene.start.normalized(),"duration":scene.duration.normalized(),"sections":sections,
        "bindings":bindings,"continuity":scene.continuity,"source":source,
    });
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
    pub stage_one: H3Stage,
    pub stage_two: Option<H3SecondStage>,
}
impl H3Settings {
    pub fn validate(&self) -> Result<(), Issue> {
        let valid = |stage: &H3Stage| {
            stage.steps > 0
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
