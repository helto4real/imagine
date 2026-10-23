use super::contract::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn check(condition: bool, code: &str, field: &str) -> Result<(), Issue> {
    if condition {
        Ok(())
    } else {
        Err(Issue::new(code, field))
    }
}
fn name(value: &str) -> bool {
    !value.trim().is_empty()
}
fn model(value: &ModelSelection) -> bool {
    [
        &value.adapter,
        &value.model,
        &value.variant,
        &value.profile,
        &value.profile_revision,
    ]
    .into_iter()
    .all(|s| name(s))
}
impl VideoProject {
    pub fn create(request: CreateProject) -> Result<Self, Issue> {
        let project = Self {
            schema_version: request.schema_version,
            id: ProjectId(format!("project_{}", uuid::Uuid::new_v4().simple())),
            project_revision: 0,
            default_model: request.default_model,
            delivery: request.delivery,
            global_prompt: String::new(),
            creative: Default::default(),
            assets: BTreeMap::new(),
            scenes: vec![],
            audio: Default::default(),
            audio_tracks: vec![],
            assembly: Default::default(),
        };
        project.validate()?;
        Ok(project)
    }
    pub fn effective_model<'a>(&'a self, scene: &'a Scene) -> &'a ModelSelection {
        match &scene.model {
            ModelChoice::ProjectDefault => &self.default_model,
            ModelChoice::Override(m) => m,
        }
    }
    pub fn scene(&self, id: &SceneId) -> Result<&Scene, Issue> {
        self.scenes
            .iter()
            .find(|s| &s.id == id)
            .ok_or_else(|| Issue::new("scene_not_found", "scene_id"))
    }
    pub fn validate(&self) -> Result<(), Issue> {
        crate::creative::validate(self)?;
        crate::assembly::validate(self)?;
        check(
            self.schema_version == SCHEMA_VERSION,
            "unsupported_schema",
            "schema_version",
        )?;
        check(name(&self.id.0), "invalid_id", "id")?;
        check(
            self.project_revision <= MAX_INTEGER,
            "invalid_revision",
            "project_revision",
        )?;
        check(model(&self.default_model), "invalid_model", "default_model")?;
        check(
            self.delivery.fps.positive() && self.delivery.width > 0 && self.delivery.height > 0,
            "invalid_delivery",
            "delivery",
        )?;
        for (id, asset) in &self.assets {
            check(
                id == &asset.id && name(&id.0) && asset.revision <= MAX_INTEGER,
                "invalid_asset",
                "assets",
            )?;
        }
        let mut ids = BTreeSet::new();
        let mut sections = BTreeSet::new();
        let mut bindings = BTreeSet::new();
        for scene in &self.scenes {
            check(
                name(&scene.id.0) && ids.insert(&scene.id),
                "duplicate_or_invalid_id",
                "scenes.id",
            )?;
            check(
                scene.start.valid() && scene.duration.positive(),
                "invalid_time",
                "scenes.time",
            )?;
            check(
                model(self.effective_model(scene)),
                "invalid_model",
                "scenes.model",
            )?;
            let mut slots = BTreeSet::new();
            for slot in &scene.settings {
                check(
                    model(&slot.model)
                        && slot.schema_version > 0
                        && slot.values.is_object()
                        && slots.insert(&slot.model),
                    "invalid_settings_slot",
                    "scenes.settings",
                )?;
            }
            for section in &scene.sections {
                check(
                    name(&section.id.0) && sections.insert(&section.id),
                    "duplicate_or_invalid_id",
                    "sections.id",
                )?;
                check(
                    section.start.valid()
                        && section.duration.positive()
                        && section.start.fits(section.duration, scene.duration),
                    "invalid_time",
                    "sections.time",
                )?;
            }
            for binding in &scene.bindings {
                check(
                    binding.video.as_ref().is_none_or(|v| v.valid_ranges()),
                    "invalid_video_source_range",
                    "bindings.video",
                )?;
                check(
                    name(&binding.id.0) && bindings.insert(&binding.id),
                    "duplicate_or_invalid_id",
                    "bindings.id",
                )?;
                check(
                    self.assets.contains_key(&binding.asset),
                    "asset_not_found",
                    "bindings.asset",
                )?;
            }
            if let Some(take) = &scene.accepted_take {
                check(
                    name(&take.id.0) && model(&take.model) && name(&take.source_fingerprint),
                    "invalid_take",
                    "accepted_take",
                )?;
            }
        }
        crate::audio::validate(self)?;
        // Continuity sources must precede their target in sequence order: no cycles.
        let mut previous = BTreeSet::new();
        for scene in &self.scenes {
            if let Some(c) = &scene.continuity {
                check(
                    previous.contains(&c.source_scene),
                    "invalid_continuity_source",
                    "continuity.source_scene",
                )?;
            }
            previous.insert(scene.id.clone());
        }
        Ok(())
    }
}
