//! Session-only proposals; provider dispatch holds no project lock or GPU lease.
use super::{engine::*, profiles};
use crate::{
    enhancement::{ProviderSnapshot, Service as ProviderService},
    scheduler::Fault,
    video::contract::*,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;
#[derive(Default)]
pub struct Service {
    jobs: Mutex<BTreeMap<String, Job>>,
}
struct Job {
    clock: std::time::Instant,
    clock_completed: usize,
    clock_elapsed: Option<f64>,
    draft: Draft,
    base: VideoProject,
    profile: profiles::Snapshot,
    provider: ProviderSnapshot,
    cancel: CancellationToken,
    generation: u64,
    grounding: super::grounding::Grounding,
    /// The Scene a single-Scene regeneration replaces, put back if it fails
    /// or is cancelled.
    restore: Option<(usize, Option<StoryScene>)>,
    /// A whole-Scene enhancement row is being retried alone.
    single_row: bool,
}
/// Review when every Scene is expanded; otherwise the outline is still open.
fn settled(draft: &Draft) -> Phase {
    if draft.scenes.iter().all(Option::is_some) {
        Phase::Review
    } else {
        Phase::Outline
    }
}
fn fault(code: &'static str) -> Fault {
    Fault::new(422, code)
}
impl Service {
    pub(crate) fn start(
        self: &Arc<Self>,
        base: VideoProject,
        request: Request,
        provider: ProviderSnapshot,
        transport: Arc<ProviderService>,
        grounding: super::grounding::Grounding,
    ) -> Result<Draft, Fault> {
        if request.schema_version != 1 || request.expected_project_revision != base.project_revision
        {
            return Err(Fault::new(409, "authoring_revision_conflict"));
        }
        if !profiles::PROVIDERS.contains(&request.provider.as_str())
            || request.brief.len() + request.instruction.len() > 65536
            || !request.duration_seconds.is_finite()
            || !(0.25..=3600.).contains(&request.duration_seconds)
        {
            return Err(fault("invalid_authoring_request"));
        }
        if request.operation == Operation::Outline && request.brief.trim().is_empty() {
            return Err(fault("story_brief_required"));
        }
        if request.add_dialogue && grounding.cast.as_array().is_none_or(|v| v.len() < 2) {
            return Err(fault("story_dialogue_cast_required"));
        }
        if request.scene_field.is_some() && request.operation != Operation::EnhanceScene {
            return Err(fault("invalid_enhancement_target"));
        }
        if request.operation != Operation::Outline {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or(fault("scene_not_found"))?)
                .map_err(|_| fault("scene_not_found"))?;
            if request.operation == Operation::EnhanceSection
                && !scene
                    .sections
                    .iter()
                    .any(|s| Some(&s.id) == request.section_id.as_ref())
            {
                return Err(fault("section_not_found"));
            }
        }
        if request.destination == Destination::Continue && base.scenes.is_empty() {
            return Err(fault("story_continuation_source_missing"));
        }
        if request.destination == Destination::Continue {
            let scene = last_scene(&base).ok_or(fault("story_continuation_source_missing"))?;
            if scene.description.chars().count() > 16000
                || scene.sections.len() > 128
                || scene
                    .sections
                    .iter()
                    .any(|s| s.prompt.chars().count() > 16000)
                || serde_json::to_string(scene).unwrap().chars().count() > 100000
            {
                return Err(fault("story_continuation_context_limit"));
            }
        }
        let catalog = profiles::Catalog::load(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/video/prompt-studio/profiles"
        )))
        .map_err(fault)?;
        let profile = catalog
            .snapshot(&request.prompt_profile_id)
            .map_err(fault)?;
        let id = format!("authoring_{}", uuid::Uuid::new_v4().simple());
        let request_rows = request.operation == Operation::EnhanceSceneAll;
        let scene_rows = request.scene_id.clone();
        super::visuals::beats(&request.images).map_err(fault)?;
        let generation_parameters = profiles::OPERATIONS
            .into_iter()
            .map(|operation| {
                (
                    operation,
                    profiles::resolve(&profile.profile.generation, &request.provider, operation)
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let draft = Draft {
            schema_version: 1,
            id: id.clone(),
            project_id: base.id.clone(),
            request,
            phase: Phase::Running,
            outline: None,
            scenes: vec![],
            text: None,
            failure: None,
            failed_index: None,
            prompt_profile_id: profile.resolved_id.clone(),
            prompt_profile_revision: profile.profile.revision.clone(),
            profile_warning: profile.warning.clone(),
            provider_snapshot: json!({"profile_id":provider.profile.id,"name":provider.profile.name,"base_url":provider.profile.base_url,"model":provider.profile.model,"generation":generation_parameters}),
            planned_durations: vec![],
            inventory: vec![],
            selected_cast: grounding.cast.clone(),
            warnings: vec![],
            fields: if request_rows {
                FieldProposal::rows(&base, base.scene(scene_rows.as_ref().unwrap()).unwrap())
            } else {
                vec![]
            },
        };
        let mut jobs = self.jobs.lock().unwrap();
        if jobs.values().any(|j| j.draft.phase.busy()) {
            return Err(Fault::new(409, "authoring_busy"));
        }
        jobs.insert(
            id.clone(),
            Job {
                clock: std::time::Instant::now(),
                clock_completed: 0,
                clock_elapsed: None,
                draft: draft.clone(),
                base,
                profile,
                provider,
                cancel: CancellationToken::new(),
                generation: 0,
                grounding,
                restore: None,
                single_row: false,
            },
        );
        drop(jobs);
        let service = self.clone();
        tokio::spawn(async move {
            service.generate(id, 0, transport, false, 0).await;
        });
        Ok(draft)
    }
    pub fn read(&self, id: &str) -> Result<Draft, Fault> {
        self.jobs
            .lock()
            .unwrap()
            .get(id)
            .map(|j| j.draft.clone())
            .ok_or(Fault::new(404, "authoring_not_found"))
    }
    pub fn progress(&self, id: &str) -> Result<serde_json::Value, Fault> {
        let jobs = self.jobs.lock().unwrap();
        let job = jobs.get(id).ok_or(Fault::new(404, "authoring_not_found"))?;
        let draft = &job.draft;
        let (completed, total) = match draft.phase {
            _ if !draft.fields.is_empty() => (
                draft
                    .fields
                    .iter()
                    .filter(|f| f.text.is_some() || f.failure.is_some())
                    .count(),
                draft.fields.len(),
            ),
            Phase::Inventory => (draft.inventory.len(), draft.request.images.len().max(1)),
            Phase::Expanding
            | Phase::Review
            | Phase::Failed
            | Phase::Cancelled
            | Phase::Applied
                if draft.outline.is_some() =>
            {
                (
                    draft.scenes.iter().filter(|s| s.is_some()).count(),
                    draft.scenes.len().max(1),
                )
            }
            Phase::Outline | Phase::Review | Phase::Applied => (1, 1),
            _ => (0, 1),
        };
        let elapsed = job
            .clock_elapsed
            .unwrap_or_else(|| job.clock.elapsed().as_secs_f64());
        let new = completed.saturating_sub(job.clock_completed);
        let eta = (new > 0 && draft.phase.busy())
            .then(|| elapsed / new as f64 * total.saturating_sub(completed) as f64);
        Ok(
            json!({"draft_id":id,"phase":draft.phase,"completed":completed,"total":total,"elapsed_seconds":elapsed,"eta_seconds":eta}),
        )
    }
    pub fn retry(
        self: &Arc<Self>,
        id: &str,
        transport: Arc<ProviderService>,
    ) -> Result<Draft, Fault> {
        let mut jobs = self.jobs.lock().unwrap();
        if jobs.values().any(|j| j.draft.phase.busy()) {
            return Err(Fault::new(409, "authoring_busy"));
        }
        let j = jobs
            .get_mut(id)
            .ok_or(Fault::new(404, "authoring_not_found"))?;
        if !matches!(j.draft.phase, Phase::Failed | Phase::Cancelled) || j.draft.outline.is_some() {
            return Err(fault("authoring_not_retryable"));
        }
        j.cancel = CancellationToken::new();
        j.generation += 1;
        j.draft.phase = Phase::Running;
        j.clock = std::time::Instant::now();
        j.clock_elapsed = None;
        j.clock_completed = j.draft.inventory.len();
        j.draft.failure = None;
        let generation = j.generation;
        let draft = j.draft.clone();
        drop(jobs);
        let service = self.clone();
        let id = id.to_owned();
        tokio::spawn(async move {
            service.generate(id, 0, transport, false, generation).await;
        });
        Ok(draft)
    }
    pub fn cancel(&self, id: &str) -> Result<Draft, Fault> {
        let mut jobs = self.jobs.lock().unwrap();
        let j = jobs
            .get_mut(id)
            .ok_or(Fault::new(404, "authoring_not_found"))?;
        if j.draft.phase != Phase::Applied {
            j.cancel.cancel();
            j.generation += 1;
            j.draft.phase = Phase::Cancelled;
            j.clock_elapsed = Some(j.clock.elapsed().as_secs_f64());
            if let Some((index, scene)) = j.restore.take() {
                j.draft.scenes[index] = scene;
                j.draft.phase = settled(&j.draft);
            }
        }
        Ok(j.draft.clone())
    }
    pub fn review(&self, id: &str, review: Review) -> Result<Draft, Fault> {
        let mut jobs = self.jobs.lock().unwrap();
        let j = jobs
            .get_mut(id)
            .ok_or(Fault::new(404, "authoring_not_found"))?;
        if review.schema_version != 1 || j.draft.phase.busy() || j.draft.phase == Phase::Applied {
            return Err(Fault::new(409, "authoring_not_editable"));
        }
        let mut d = j.draft.clone();
        if let Some(outline) = review.outline {
            validate(&outline).map_err(fault)?;
            d.planned_durations = durations(
                &outline.scenes.iter().map(|s| s.weight).collect::<Vec<_>>(),
                d.request.duration_seconds,
                15.,
            )
            .map_err(fault)?;
            if d.outline.as_ref() != Some(&outline) {
                d.scenes = vec![None; outline.scenes.len()];
                d.outline = Some(outline);
                d.phase = Phase::Outline;
            }
        }
        if let Some(scenes) = review.scenes {
            let outline = d.outline.as_ref().ok_or(fault("story_outline_required"))?;
            if scenes.len() != outline.scenes.len() {
                return Err(fault("invalid_storyboard"));
            }
            for (scene, source) in scenes.iter().zip(&outline.scenes) {
                if let Some(scene) = scene {
                    reviewed(scene, source, &d.request.images).map_err(fault)?;
                }
            }
            d.scenes = scenes;
            d.phase = if d.scenes.iter().all(Option::is_some) {
                Phase::Review
            } else {
                Phase::Outline
            };
        }
        if let Some(fields) = review.fields {
            if fields.len() != d.fields.len()
                || fields.iter().zip(&d.fields).any(|(a, b)| {
                    a.target != b.target
                        || a.original != b.original
                        || a.text.as_ref().is_some_and(|t| t.len() > 65536)
                })
            {
                return Err(fault("invalid_authoring_fields"));
            }
            d.fields = fields;
        }
        if let Some(text) = review.text {
            if text.trim().is_empty()
                || text.len() > 65536
                || d.request.operation == Operation::Outline
            {
                return Err(fault("invalid_authoring_text"));
            }
            d.text = Some(text);
            d.phase = Phase::Review;
        }
        j.draft = d;
        Ok(j.draft.clone())
    }
    /// Expands from `index` on, or with `only` regenerates just that Scene,
    /// keeping the others, as Director's Regenerate Scene does.
    pub fn expand(
        self: &Arc<Self>,
        id: &str,
        index: usize,
        only: bool,
        transport: Arc<ProviderService>,
    ) -> Result<Draft, Fault> {
        let mut jobs = self.jobs.lock().unwrap();
        if jobs.values().any(|j| j.draft.phase.busy()) {
            return Err(Fault::new(409, "authoring_busy"));
        }
        let j = jobs
            .get_mut(id)
            .ok_or(Fault::new(404, "authoring_not_found"))?;
        // One row of a whole-Scene enhancement again, keeping the others.
        if !j.draft.fields.is_empty() {
            if j.draft.phase == Phase::Applied || index >= j.draft.fields.len() || !only {
                return Err(fault("invalid_story_retry_index"));
            }
            let row = &mut j.draft.fields[index];
            row.text = None;
            row.failure = None;
            j.single_row = true;
            j.draft.phase = Phase::Running;
            j.clock = std::time::Instant::now();
            j.clock_elapsed = None;
            j.cancel = CancellationToken::new();
            j.generation += 1;
            let generation = j.generation;
            let result = j.draft.clone();
            drop(jobs);
            let service = self.clone();
            let id = id.to_owned();
            tokio::spawn(async move {
                service
                    .generate(id, index, transport, false, generation)
                    .await;
            });
            return Ok(result);
        }
        let outline = j
            .draft
            .outline
            .as_ref()
            .ok_or(fault("story_outline_required"))?;
        if j.draft.phase == Phase::Applied
            || index >= outline.scenes.len()
            || (!only && j.draft.scenes[..index].iter().any(Option::is_none))
        {
            return Err(fault("invalid_story_retry_index"));
        }
        j.restore = only.then(|| (index, j.draft.scenes[index].take()));
        if !only {
            for s in &mut j.draft.scenes[index..] {
                *s = None;
            }
        }
        j.draft.failure = None;
        j.draft.failed_index = None;
        j.draft.phase = Phase::Expanding;
        j.clock = std::time::Instant::now();
        j.clock_elapsed = None;
        j.clock_completed = j.draft.scenes.iter().filter(|s| s.is_some()).count();
        j.cancel = CancellationToken::new();
        j.generation += 1;
        let generation = j.generation;
        let result = j.draft.clone();
        drop(jobs);
        let service = self.clone();
        let id = id.to_owned();
        tokio::spawn(async move {
            service
                .generate(id, index, transport, true, generation)
                .await;
        });
        Ok(result)
    }
    pub fn edit_plan(
        &self,
        id: &str,
        current: &VideoProject,
    ) -> Result<crate::video::editing::EditRequest, Fault> {
        let jobs = self.jobs.lock().unwrap();
        let job = jobs.get(id).ok_or(Fault::new(404, "authoring_not_found"))?;
        apply(current, &job.draft).map_err(|code| {
            Fault::new(
                if code == "authoring_revision_conflict" {
                    409
                } else {
                    422
                },
                code,
            )
        })
    }
    pub fn applied(&self, id: &str) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(id) {
            job.draft.phase = Phase::Applied;
        }
    }
    async fn generate(
        self: Arc<Self>,
        id: String,
        start: usize,
        transport: Arc<ProviderService>,
        expand: bool,
        generation: u64,
    ) {
        let (mut draft, base, profile, provider, cancel, grounding) = {
            let jobs = self.jobs.lock().unwrap();
            let j = &jobs[&id];
            if j.generation != generation {
                return;
            }
            (
                j.draft.clone(),
                j.base.clone(),
                j.profile.clone(),
                j.provider.clone(),
                j.cancel.clone(),
                j.grounding.clone(),
            )
        };
        if !expand && draft.request.operation == Operation::Outline {
            for (selection, image) in &grounding.images {
                if draft.inventory.iter().any(|n| n.id == selection.id) {
                    continue;
                }
                {
                    let mut jobs = self.jobs.lock().unwrap();
                    let j = jobs.get_mut(&id).unwrap();
                    if j.generation != generation {
                        return;
                    }
                    j.draft.phase = Phase::Inventory;
                }
                let result=async{
                    let parameters=profiles::resolve(&profile.profile.generation,&draft.request.provider,"image_analysis")?;
                    let system=system(&profile,"story_image_inventory")?;
                    let prompt=format!("Task:\nInspect this one image for visual-story planning.\n\n<image_id>\n{}\n</image_id>\n\n<label>\n{}\n</label>\n\n<additional_instruction>\n{}\n</additional_instruction>",selection.id,selection.label,draft.request.instruction);
                    tokio::select!{biased;_=cancel.cancelled()=>Err("authoring_cancelled"),r=transport.authoring_complete(&provider,&draft.request.provider,&system,&prompt,&parameters,std::slice::from_ref(image))=>r}
                }.await;
                let mut jobs = self.jobs.lock().unwrap();
                let j = jobs.get_mut(&id).unwrap();
                if j.generation != generation {
                    return;
                }
                match result {
                    Ok(note) => {
                        let row = InventoryNote {
                            id: selection.id.clone(),
                            note,
                        };
                        j.draft.inventory.push(row.clone());
                        draft.inventory.push(row);
                    }
                    Err(code) => {
                        j.draft.phase = Phase::Failed;
                        j.clock_elapsed = Some(j.clock.elapsed().as_secs_f64());
                        j.draft.failure = Some(code.into());
                        return;
                    }
                }
            }
            let mut jobs = self.jobs.lock().unwrap();
            let j = jobs.get_mut(&id).unwrap();
            if j.generation != generation {
                return;
            }
            j.draft.phase = Phase::Running;
        }
        let rows = !draft.fields.is_empty();
        let count = if !expand {
            if !rows {
                1
            } else if self.jobs.lock().unwrap()[&id].single_row {
                start + 1
            } else {
                draft.fields.len()
            }
        } else if self.jobs.lock().unwrap()[&id].restore.is_some() {
            start + 1
        } else {
            draft.scenes.len()
        };
        for index in start..count {
            // The text this pass enhances: a whole-Scene row, or the request's own.
            let (is_section, section_id, field) = match draft.fields.get(index).map(|f| &f.target) {
                Some(FieldTarget::Section { section_id }) => {
                    (true, Some(section_id.clone()), SceneField::default())
                }
                Some(FieldTarget::Scene { field }) => (false, None, *field),
                None => (
                    draft.request.operation == Operation::EnhanceSection,
                    draft.request.section_id.clone(),
                    draft.request.scene_field.unwrap_or_default(),
                ),
            };
            let (mut key, operation, mut user, protected) = if expand {
                let jobs = self.jobs.lock().unwrap();
                let d = &jobs[&id].draft;
                let outline = d.outline.as_ref().unwrap();
                let source = &outline.scenes[index];
                let previous = if index > 0 {
                    json!(d.scenes[index - 1])
                } else if draft.request.destination == Destination::Continue {
                    last_scene(&base)
                        .map(|s| json!({"description":s.description,"sections":s.sections}))
                        .unwrap_or(json!(null))
                } else {
                    json!(null)
                };
                (
                    "story_scene",
                    "story_scene_expansion",
                    format!(
                        "Task:\nExpand this one Scene and its Sections. Return one JSON object matching the supplied Scene shape and no markdown.\n\nVisual context rule:\nNo images are supplied. Provide detailed environment, spatial layout, appearance, lighting, and atmosphere.\n\n<previous_scene_context>\n{previous}\n</previous_scene_context>\n\n<selected_cast>\n[]\n</selected_cast>\n\n<continuity_cast>\n[]\n</continuity_cast>\n\n<next_scene_summary>\n{}\n</next_scene_summary>\n\n<additional_instruction>\n{}\n</additional_instruction>\n\n<scene_json>\n{}\n</scene_json>",
                        outline
                            .scenes
                            .get(index + 1)
                            .map(|s| s.summary.as_str())
                            .unwrap_or(""),
                        draft.request.instruction,
                        serde_json::to_string(source).unwrap()
                    ),
                    None,
                )
            } else if draft.request.operation == Operation::Outline {
                let context = if draft.request.destination == Destination::Continue {
                    last_scene(&base)
                        .into_iter()
                        .map(|s| json!({"description":s.description,"sections":s.sections}))
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                };
                (
                    "story_outline",
                    "story_outline",
                    format!(
                        "Task:\nPlan an ordered Story Outline. Return only JSON matching the supplied schema. Every Scene must fit 15 seconds and contain 1–12 sections. Requested duration: {} seconds.\n\n<story>\n{}\n</story>\n\n<latest_moment>\n{}\n</latest_moment>\n\n<earlier_continuity>\n{}\n</earlier_continuity>\n\n<selected_cast>\n[]\n</selected_cast>\n\n<additional_instruction>\n{}\n</additional_instruction>",
                        draft.request.duration_seconds,
                        draft.request.brief,
                        draft.request.brief,
                        json!(context),
                        draft.request.instruction
                    ),
                    None,
                )
            } else {
                let scene = base
                    .scene(draft.request.scene_id.as_ref().unwrap())
                    .unwrap();
                let section = scene
                    .sections
                    .iter()
                    .find(|s| is_section && Some(&s.id) == section_id.as_ref());
                let source = if is_section {
                    section.unwrap().prompt.clone()
                } else {
                    field.source(&base, scene)
                };
                let protected = super::references::Protected::new(&source);
                (
                    if section.is_some() {
                        "enhance_section"
                    } else {
                        "enhance_scene_field"
                    },
                    "text_enhancement",
                    format!(
                        "Task:\nEnhance one {}. Return only the enhanced text.\n\nVisual context rule:\nNo images are supplied. Add the environmental, spatial, lighting, and atmospheric detail needed for visual grounding.\n\nScene context:\n{}\n\nAdditional user instruction:\n{}\n\nSource text:\n{}",
                        if section.is_some() {
                            "Section (prompt)"
                        } else {
                            field.label()
                        },
                        scene.description,
                        draft.request.instruction,
                        protected.text
                    ),
                    Some(protected),
                )
            };
            if draft.request.add_dialogue {
                key = match key {
                    "story_outline" => "story_outline_dialogue",
                    "story_scene" => "story_scene_dialogue",
                    other => other,
                };
            }
            user = user.replace(
                "<selected_cast>\n[]\n</selected_cast>",
                &format!("<selected_cast>\n{}\n</selected_cast>", draft.selected_cast),
            );
            user.push_str(&format!(
                "\n\n<image_sources>\n{}\n</image_sources>",
                json!(draft.request.images)
            ));
            user.push_str(&format!("\n\n<image_inventory>\n{}\n</image_inventory>\n\n<visual_beats>\n{}\n</visual_beats>",json!(draft.inventory),json!(super::visuals::beats(&draft.request.images).unwrap().iter().map(|b|json!({"beat_key":b.key,"image_ids":b.ids,"semantic_role":b.role})).collect::<Vec<_>>())));
            let images = if expand {
                let jobs = self.jobs.lock().unwrap();
                let d = &jobs[&id].draft;
                let source = d.outline.as_ref().unwrap().scenes.get(index).unwrap();
                let ids = source
                    .image_ids
                    .iter()
                    .chain(source.sections.iter().flat_map(|s| {
                        s.image_ids
                            .iter()
                            .chain([&s.first_frame_id, &s.last_frame_id])
                    }))
                    .cloned()
                    .collect::<Vec<_>>();
                grounding.selected(&ids)
            } else if draft.request.operation == Operation::Outline {
                vec![]
            } else {
                grounding.images.iter().map(|(_, p)| p.clone()).collect()
            };
            if !images.is_empty() {
                user=user.replace("No images are supplied.","The selected source images are supplied. Ground visible details in those images; do not invent unseen facts.");
            }
            if let Some(protected) = &protected {
                let scene = base
                    .scene(draft.request.scene_id.as_ref().unwrap())
                    .unwrap();
                // The opening frame leads only the Scene fields and its first Section.
                let opening = grounding.opening
                    && (!is_section
                        || scene.sections.first().map(|s| &s.id) == section_id.as_ref());
                user = super::references::enhancement_user_prompt(
                    is_section,
                    &scene.description,
                    &draft.request.instruction,
                    &protected.text,
                    !images.is_empty(),
                    opening,
                );
            }
            let result=async {
                if user.chars().count()>100000{return Err("story_context_limit");}
                let parameters=profiles::resolve(&profile.profile.generation,&draft.request.provider,operation)?;
                let system=system_with_images(&profile,key,&draft.request.images,draft.request.add_dialogue)?;
                tokio::select!{biased;_=cancel.cancelled()=>Err("authoring_cancelled"),result=transport.authoring_complete(&provider,&draft.request.provider,&system,&user,&parameters,&images)=>result}
            }.await;
            let mut jobs = self.jobs.lock().unwrap();
            let j = jobs.get_mut(&id).unwrap();
            if j.generation != generation {
                return;
            }
            if cancel.is_cancelled() {
                j.draft.phase = Phase::Cancelled;
                j.clock_elapsed = Some(j.clock.elapsed().as_secs_f64());
                return;
            }
            let result = result.and_then(|text| {
                if expand {
                    let source = &j.draft.outline.as_ref().unwrap().scenes[index];
                    let mut scene = expansion(&text, source)?;
                    if !j.draft.request.add_music {
                        scene.non_diegetic_music.clear();
                    }
                    super::visuals::validate_cast(
                        &Storyboard {
                            scenes: vec![scene.clone()],
                        },
                        &j.draft.selected_cast,
                        j.draft.request.add_dialogue,
                    )?;
                    j.draft.scenes[index] = Some(scene);
                } else if draft.request.operation == Operation::Outline {
                    let mut board = outline_with_images(
                        &text,
                        &draft.request.images,
                        draft.request.add_dialogue,
                    )?;
                    if !draft.request.add_music {
                        for scene in &mut board.scenes {
                            scene.non_diegetic_music.clear();
                        }
                    }
                    super::visuals::validate_cast(
                        &board,
                        &j.draft.selected_cast,
                        j.draft.request.add_dialogue,
                    )?;
                    j.draft.planned_durations = durations(
                        &board.scenes.iter().map(|s| s.weight).collect::<Vec<_>>(),
                        draft.request.duration_seconds,
                        15.,
                    )?;
                    j.draft.scenes = vec![None; board.scenes.len()];
                    j.draft.outline = Some(board);
                } else {
                    if text.len() > 65536 {
                        return Err("story_text_limit");
                    }
                    let text=if let Some(protected)=&protected {
                        match protected.restore(&text){
                            Ok((restored,introduced))=>{if introduced {j.draft.warnings.push("The model introduced one or more new reference markers. Review them before applying.".into());}restored},
                            Err(code)=>{if !rows {j.draft.text=Some(protected.source.clone());}return Err(code);}
                        }
                    }else{text};
                    if rows {
                        j.draft.fields[index].text = Some(text);
                    } else {
                        j.draft.text = Some(text);
                    }
                }
                Ok(())
            });
            // A whole-Scene row fails alone; the others still run.
            if rows && let Err(code) = result {
                j.draft.fields[index].failure = Some(code.into());
                continue;
            }
            if let Err(code) = result {
                j.clock_elapsed = Some(j.clock.elapsed().as_secs_f64());
                j.draft.failure = Some(code.into());
                // A failed regeneration keeps the Scene it would have replaced.
                if let Some((index, scene)) = j.restore.take() {
                    j.draft.scenes[index] = scene;
                    j.draft.phase = settled(&j.draft);
                    return;
                }
                j.draft.phase = Phase::Failed;
                j.draft.failed_index = expand.then_some(index);
                return;
            }
        }
        let mut jobs = self.jobs.lock().unwrap();
        let j = jobs.get_mut(&id).unwrap();
        if j.generation != generation {
            return;
        }
        j.clock_elapsed = Some(j.clock.elapsed().as_secs_f64());
        j.restore = None;
        j.single_row = false;
        j.draft.phase = if !expand && draft.request.operation == Operation::Outline {
            Phase::Outline
        } else if expand {
            settled(&j.draft)
        } else {
            Phase::Review
        };
    }
}
fn last_scene(base: &VideoProject) -> Option<&Scene> {
    base.scenes.iter().max_by(|a, b| {
        let end = |s: &Scene| {
            s.start.num as f64 / s.start.den as f64 + s.duration.num as f64 / s.duration.den as f64
        };
        end(a).total_cmp(&end(b))
    })
}
