//! Project generation defaults and scene overrides, through the shared edit contract.
use super::*;
use imagine_video_contract::{
    generation::{Defaults, SeedControl, h3_defaults, h3_defaults_for, h3_model},
    planning::{H3Handoff, H3SecondStage, H3Settings, H3Stage, H3StageTwoDraft},
};
#[derive(Default)]
pub(super) struct State {
    draft: Option<Draft>,
    scenes:
        std::collections::BTreeMap<(ProjectId, SceneId, ModelSelection), (Value, SamplingDraft)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_stage_two_retains_all_fields_and_raw_input_across_apply_and_reopen() {
        let model = h3_model("reference");
        let mut draft = SamplingDraft::new(h3_defaults());
        draft.set_two_stages(true, &model);
        let stage = draft.value.stage_two.as_mut().unwrap();
        stage.sampling.steps = 7;
        stage.sampling.sampler = "alternative-sampler".into();
        stage.i2v_model_file = Some("/own/compatible stage2.model".into());
        stage.model = h3_model("i2v");
        stage.handoff = H3Handoff::ReNoiseVideoPrediction;
        stage.handoff_upscale_factor = 1.25;
        stage.trained_latent_upscale_model = Some("/own/compatible upscaler.weights".into());
        draft.second_sigmas = "0.4, unfinished".into();
        draft.set_two_stages(false, &model);
        let stored = draft.resolve().unwrap();
        assert!(stored.stage_two.is_none());
        let restored: H3Settings =
            serde_json::from_slice(&serde_json::to_vec(&stored).unwrap()).unwrap();
        let mut reopened = SamplingDraft::new(restored);
        reopened.set_two_stages(true, &model);
        let stage = reopened.value.stage_two.as_ref().unwrap();
        assert_eq!(stage.sampling.steps, 7);
        assert_eq!(stage.sampling.sampler, "alternative-sampler");
        assert_eq!(
            stage.i2v_model_file.as_deref(),
            Some("/own/compatible stage2.model")
        );
        assert_eq!(stage.model, h3_model("i2v"));
        assert_eq!(stage.handoff, H3Handoff::ReNoiseVideoPrediction);
        assert_eq!(stage.handoff_upscale_factor, 1.25);
        assert_eq!(
            stage.trained_latent_upscale_model.as_deref(),
            Some("/own/compatible upscaler.weights")
        );
        assert_eq!(reopened.second_sigmas, "0.4, unfinished");
        assert!(reopened.resolve().is_err());
    }
    #[test]
    fn asynchronous_file_choice_stays_with_its_project_and_mode_and_cancel_keeps_value() {
        let ctx = egui::Context::default();
        let mut a = Some("original-a.weights".to_owned());
        let mut b = Some("original-b.weights".to_owned());
        let draw = |scope: (&str, &str),
                    value: &mut Option<String>,
                    pick: Option<Option<PathBuf>>,
                    consume: bool| {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.push_id(scope, |ui| {
                    if let Some(pick) = pick.clone() {
                        let key =
                            ui.make_persistent_id(("h3-component-file-result", "own-component"));
                        ui.ctx().data_mut(|data| data.insert_temp(key, pick));
                    }
                    if consume {
                        component_path(ui, "Own component", "own-component", value);
                    }
                });
            });
            output.textures_delta.clear();
        };
        draw(
            ("own-project-a", "reference"),
            &mut a,
            Some(Some(PathBuf::from("/own/compatible alternative.bin"))),
            false,
        );
        draw(("own-project-b", "i2v"), &mut b, None, true);
        assert_eq!(a.as_deref(), Some("original-a.weights"));
        assert_eq!(b.as_deref(), Some("original-b.weights"));
        draw(("own-project-a", "reference"), &mut a, None, true);
        assert_eq!(a.as_deref(), Some("/own/compatible alternative.bin"));
        draw(("own-project-a", "reference"), &mut a, Some(None), true);
        assert_eq!(a.as_deref(), Some("/own/compatible alternative.bin"));
    }
    #[test]
    fn a_source_canvas_changes_size_in_the_image_shape_and_back() {
        use imagine_video_contract::generation::Canvas;
        let mut canvas = Canvas {
            ratio: "Source".into(),
            portrait: false,
            quality: "Standard".into(),
        };
        // As Animate resolves a portrait image.
        let standard = canvas_size(&canvas, Some((320, 480))).unwrap();
        assert!(standard.1 > standard.0);
        // Without the shape it fails, which used to put Standard back.
        canvas.quality = "Draft".into();
        assert!(canvas_size(&canvas, None).is_err());
        let draft = canvas_size(&canvas, Some(standard)).unwrap();
        assert!(draft.0 < standard.0 && draft.1 < standard.1 && draft.1 > draft.0);
        canvas.quality = "Standard".into();
        assert_eq!(canvas_size(&canvas, Some(standard)).unwrap(), standard);
    }
    #[test]
    fn project_mode_switch_retains_invalid_raw_text_and_independent_settings() {
        let mut d = Draft {
            session: egui::Id::new("own-draft"),
            modes: Default::default(),
            project: ProjectId("own".into()),
            model: h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
            generation: Defaults::default(),
            source: None,
            seed: "0".into(),
            sampling: SamplingDraft::new(h3_defaults()),
            error: None,
        };
        d.sampling.sigmas = "1, unfinished".into();
        let before = d.model.clone();
        d.model = h3_model("i2v");
        d.switch_mode(before).unwrap();
        assert!(d.sampling.sigmas.is_empty());
        d.sampling.value.stage_one.sampler = "alternative-sampler-name".into();
        let before = d.model.clone();
        d.model = h3_model("reference");
        d.switch_mode(before).unwrap();
        assert_eq!(d.sampling.sigmas, "1, unfinished");
        assert!(d.sampling.resolve().is_err());
        let before = d.model.clone();
        d.model = h3_model("i2v");
        d.switch_mode(before).unwrap();
        assert_eq!(
            d.sampling.value.stage_one.sampler,
            "alternative-sampler-name"
        );
    }
    #[test]
    fn sampling_drafts_survive_navigation_flush_as_one_edit_and_undo_restores_inheritance() {
        let mut p = super::super::media_picker::tests::project();
        p.default_model = h3_model("reference");
        p.generation = Some(Default::default());
        let original = p.clone();
        let sid = p.scenes[0].id.clone();
        let mut state = State::default();
        let mut draft = SamplingDraft::new(h3_defaults());
        draft.value.stage_one.sampler = "euler".into();
        draft.sigmas = "1, 0.4, 0".into();
        draft.dirty = true;
        state.scenes.insert(
            (p.id.clone(), sid.clone(), p.default_model.clone()),
            (
                p.effective_settings(&p.scenes[0]).unwrap().values.clone(),
                draft,
            ),
        );
        let commands = state.scene_edits(&p);
        assert_eq!(commands.len(), 1);
        assert!(state.input_error(&p, None).is_none());
        imagine_video_contract::editing::apply(
            &mut p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: original.project_revision,
                commands,
            },
        )
        .unwrap();
        assert_eq!(
            p.effective_settings(&p.scenes[0]).unwrap().values["stage_one"]["sampler"],
            "euler"
        );
        let undo = editing::difference(&p, &original);
        let revision = p.project_revision;
        imagine_video_contract::editing::apply(
            &mut p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands: undo,
            },
        )
        .unwrap();
        assert!(p.scenes[0].settings.is_empty());
        let (_, draft) = state.scenes.values_mut().next().unwrap();
        draft.sigmas = "1, unfinished".into();
        assert!(state.input_error(&p, None).is_some());
        assert!(
            state
                .input_error(&p, Some(&SceneId("other-own-scene".into())))
                .is_none()
        );
        assert!(state.scene_edits(&p).is_empty());
        p.id = ProjectId("other-own-project".into());
        assert!(state.input_error(&p, None).is_none());
        assert!(state.scene_edits(&p).is_empty());
    }
}
struct Draft {
    session: egui::Id,
    modes: std::collections::BTreeMap<ModelSelection, SamplingDraft>,
    project: ProjectId,
    model: ModelSelection,
    delivery: DeliverySettings,
    generation: Defaults,
    /// The image shape a "Source" canvas was made from, kept while the ratio
    /// changes so Source can be chosen again.
    source: Option<(u32, u32)>,
    seed: String,
    sampling: SamplingDraft,
    error: Option<String>,
}
/// The output size for `canvas`; a "Source" canvas keeps `source`'s shape.
fn canvas_size(
    canvas: &imagine_video_contract::generation::Canvas,
    source: Option<(u32, u32)>,
) -> Result<(u32, u32), &'static str> {
    let shape = crate::h3_protocol::Selection {
        ratio: canvas.ratio.clone(),
        portrait: canvas.portrait,
        quality: canvas.quality.clone(),
        seconds: 5.,
    }
    .resolve_with_source(source)?;
    Ok((shape.width, shape.height))
}
impl Draft {
    fn switch_mode(&mut self, previous: ModelSelection) -> Result<(), String> {
        self.modes.insert(previous.clone(), self.sampling.clone());
        if let Ok(value) = self.sampling.resolve() {
            self.generation.settings.retain(|s| s.model != previous);
            self.generation.settings.push(slot(previous, &value));
        }
        if let Some(cached) = self.modes.remove(&self.model) {
            self.sampling = cached;
            return Ok(());
        }
        let value = self
            .generation
            .settings
            .iter()
            .find(|s| s.model == self.model)
            .map(|s| serde_json::from_value(s.values.clone()))
            .transpose()
            .map_err(|e| format!("Cannot edit these H3 settings: {e}"))?;
        self.sampling =
            SamplingDraft::new(value.unwrap_or_else(|| h3_defaults_for(&self.model.variant)));
        Ok(())
    }
}
#[derive(Clone)]
struct SamplingDraft {
    dirty: bool,
    value: H3Settings,
    sigmas: String,
    second_sigmas: String,
    error: Option<String>,
}
impl State {
    pub(super) fn project_saved(&mut self) {
        self.draft = None;
    }
    pub(super) fn scene_edits(&self, project: &VideoProject) -> Vec<Edit> {
        self.scenes
            .iter()
            .filter_map(|((pid, sid, model), (_, draft))| {
                if pid != &project.id || !draft.dirty {
                    return None;
                }
                let scene = project.scene(sid).ok()?;
                let value = draft.resolve().ok()?;
                let settings = slot(model.clone(), &value);
                (!scene.settings.contains(&settings)).then(|| Edit::SetModelSettings {
                    scene_id: sid.clone(),
                    settings,
                })
            })
            .collect()
    }
    pub(super) fn input_error(
        &self,
        project: &VideoProject,
        target: Option<&SceneId>,
    ) -> Option<String> {
        self.scenes.iter().find_map(|((pid, sid, _), (_, draft))| {
            (pid == &project.id
                && target.is_none_or(|target| target == sid)
                && draft.dirty
                && project.scene(sid).is_ok())
            .then(|| draft.resolve().err())
            .flatten()
        })
    }
}
fn defaults(project: &VideoProject) -> Defaults {
    project.generation.clone().unwrap_or_else(|| Defaults {
        canvas: None,
        seed: 42,
        control_after_generate: None,
        ..Default::default()
    })
}
fn slot(model: ModelSelection, value: &H3Settings) -> SettingsSlot {
    SettingsSlot {
        model,
        schema_version: 1,
        values: serde_json::to_value(value).expect("finite H3 settings"),
    }
}
fn control(ui: &egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
fn list(values: &[f32]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
fn sigmas(text: &str) -> Result<Vec<f32>, String> {
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f32>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    "Sigmas must be finite numbers separated by commas or spaces.".into()
                })
        })
        .collect()
}
impl SamplingDraft {
    fn new(value: H3Settings) -> Self {
        Self {
            sigmas: list(&value.stage_one.sigmas),
            second_sigmas: value
                .stage_two
                .as_ref()
                .map(|s| list(&s.sampling.sigmas))
                .or_else(|| {
                    value
                        .stage_two_draft
                        .as_ref()
                        .map(|draft| draft.raw_sigmas.clone())
                })
                .unwrap_or_default(),
            value,
            dirty: false,
            error: None,
        }
    }
    fn resolve(&self) -> Result<H3Settings, String> {
        let mut v = self.value.clone();
        v.stage_one.sigmas = sigmas(&self.sigmas)?;
        if let Some(s) = &mut v.stage_two {
            s.sampling.sigmas = sigmas(&self.second_sigmas)?;
        }
        if let Some(draft) = &mut v.stage_two_draft {
            draft.raw_sigmas = self.second_sigmas.clone();
        }
        Ok(v)
    }
    fn set_two_stages(&mut self, enabled: bool, model: &ModelSelection) {
        if enabled {
            self.value.stage_two = Some(if let Some(draft) = self.value.stage_two_draft.take() {
                self.second_sigmas = draft.raw_sigmas;
                draft.settings
            } else {
                H3SecondStage {
                    handoff_upscale_factor: 1.,
                    reference_model_file: None,
                    i2v_model_file: None,
                    sampling: H3Stage {
                        steps: 4,
                        sampler: "euler".into(),
                        scheduler: "simple".into(),
                        denoise: 1.,
                        sigmas: vec![],
                    },
                    model: model.clone(),
                    handoff: H3Handoff::ContinueLatent,
                    trained_latent_upscale_model: None,
                }
            });
        } else {
            self.value.stage_two_draft =
                self.value.stage_two.take().map(|settings| H3StageTwoDraft {
                    settings,
                    raw_sigmas: self.second_sigmas.clone(),
                });
        }
    }
    fn show(&mut self, ui: &mut egui::Ui, model: &ModelSelection) -> bool {
        let mut changed = false;
        let header = egui::CollapsingHeader::new("Component files & sigma shifts")
            .id_salt("h3-component-files")
            .show(ui, |ui| {
                ui.weak("Enter an absolute path, or a path relative to the configured model root. Blank uses the deployed default. Compatibility is determined when loading.");
                changed |= component_path(ui, "Diffusion model", "component-model", &mut self.value.components.model);
                changed |= component_path(ui, "Text encoder", "component-text", &mut self.value.components.text_encoder);
                changed |= component_path(ui, "Video VAE", "component-video-vae", &mut self.value.components.video_vae);
                changed |= component_path(ui, "Audio VAE", "component-audio-vae", &mut self.value.components.audio_vae);
                ui.label("Video sigma shift");
                let r=ui.add(egui::DragValue::new(&mut self.value.shift_video).speed(0.01).range(0.01..=100.));
                control(ui, "component-video-shift", &r);changed |= r.changed();
                ui.label("Audio sigma shift");
                let r=ui.add(egui::DragValue::new(&mut self.value.shift_audio).speed(0.01).range(0.01..=100.));
                control(ui, "component-audio-shift", &r);changed |= r.changed();
            });
        control(ui, "component-files", &header.header_response);
        ui.label("Reference Quality");
        let r = egui::ComboBox::from_id_salt("reference-quality")
            .selected_text(&self.value.ref_image_size)
            .show_ui(ui, |ui| {
                for q in ["Standard", "High", "Extra High"] {
                    let r = ui.selectable_value(&mut self.value.ref_image_size, q.into(), q);
                    control(ui, &format!("reference-quality-{q}"), &r);
                    changed |= r.changed();
                }
            });
        control(ui, "reference-quality", &r.response);
        changed |= stage(ui, &mut self.value.stage_one, &mut self.sigmas, "base");
        let mut two = self.value.stage_two.is_some();
        let r = ui.checkbox(&mut two, "Two stages");
        control(ui, "sampling-two-stages", &r);
        if r.changed() {
            changed = true;
            self.set_two_stages(two, model);
        }
        if let Some(s) = &mut self.value.stage_two {
            ui.label("Base Steps to Run");
            let r =
                ui.add(egui::DragValue::new(&mut self.value.base_steps_to_run).range(1..=u32::MAX));
            control(ui, "sampling-base-steps", &r);
            changed |= r.changed();
            ui.label("Stage 2 mode");
            changed |= mode(ui, &mut s.model, "stage2-mode");
            changed |= component_path(
                ui,
                "Stage 2 Reference model · blank inherits original Reference input",
                "stage2-reference-file",
                &mut s.reference_model_file,
            );
            changed |= component_path(
                ui,
                "Stage 2 I2V model · blank inherits original I2V input",
                "stage2-i2v-file",
                &mut s.i2v_model_file,
            );
            changed |= stage(ui, &mut s.sampling, &mut self.second_sigmas, "stage2");
            ui.label("Stage handoff");
            for (value, label, key) in [
                (H3Handoff::ContinueLatent, "Continue latent", "continue"),
                (
                    H3Handoff::ReNoiseVideoPrediction,
                    "Re-noise video prediction",
                    "renoise",
                ),
            ] {
                let r = ui.selectable_value(&mut s.handoff, value, label);
                control(ui, &format!("handoff-{key}"), &r);
                changed |= r.changed();
            }
            ui.label("Handoff upscale factor");
            let r = ui.add(
                egui::DragValue::new(&mut s.handoff_upscale_factor)
                    .speed(0.05)
                    .range(1. ..=4.),
            );
            control(ui, "stage2-upscale-factor", &r);
            changed |= r.changed();
            if s.handoff_upscale_factor > 1. && s.handoff == H3Handoff::ReNoiseVideoPrediction {
                changed |= component_path(
                    ui,
                    "Trained latent upscaler · blank uses deployed default",
                    "stage2-upscale-file",
                    &mut s.trained_latent_upscale_model,
                );
                ui.weak("Upscales video latents and aligned keyframes before Stage 2. Audio keeps its original size.");
            } else if s.handoff_upscale_factor > 1. {
                ui.weak("Continue latent retains this factor without applying upscaling. Re-noise video prediction enables trained upscaling.");
            }
        }
        ui.weak(
            "H3 generates 24 fps. Project timing and final video use the project's FPS setting.",
        );
        if let Some(e) = &self.error {
            ui.colored_label(helto::ERROR, e);
        }
        changed
    }
}
pub(super) fn component_path(
    ui: &mut egui::Ui,
    label: &str,
    key: &str,
    value: &mut Option<String>,
) -> bool {
    let picked_key = ui.make_persistent_id(("h3-component-file-result", key));
    let picked = ui.ctx().data_mut(|data| {
        let picked = data.get_temp::<Option<PathBuf>>(picked_key);
        if picked.is_some() {
            data.remove::<Option<PathBuf>>(picked_key);
        }
        picked
    });
    let mut changed = false;
    if let Some(Some(path)) = picked {
        *value = Some(path.to_string_lossy().into_owned());
        changed = true;
    }
    ui.label(label);
    let mut text = value.clone().unwrap_or_default();
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 32. - ui.spacing().item_spacing.x).max(60.);
        let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(width));
        control(ui, key, &r);
        if r.changed() {
            *value = (!text.trim().is_empty()).then_some(text);
            changed = true;
        }
        let r = helto::icon_button(
            ui,
            crate::icons::Icon::Folder,
            &format!("Choose {label}"),
            false,
        );
        control(ui, &format!("{key}-browse"), &r);
        if r.clicked() {
            let ctx = ui.ctx().clone();
            let title = format!("Choose {label}");
            std::thread::spawn(move || {
                let path = rfd::FileDialog::new().set_title(title).pick_file();
                ctx.data_mut(|data| data.insert_temp(picked_key, path));
                ctx.request_repaint();
            });
        }
    });
    changed
}
fn mode(ui: &mut egui::Ui, model: &mut ModelSelection, key: &str) -> bool {
    let mut changed = false;
    let r = egui::ComboBox::from_id_salt(key)
        .selected_text(if model.variant == "i2v" {
            "Image-to-Video"
        } else if model.variant == "reference" {
            "Reference"
        } else {
            &model.variant
        })
        .show_ui(ui, |ui| {
            for (v, label) in [("reference", "Reference"), ("i2v", "Image-to-Video")] {
                let r = ui.selectable_label(model.variant == v, label);
                control(ui, &format!("{key}-{v}"), &r);
                if r.clicked() {
                    *model = h3_model(v);
                    changed = true;
                }
            }
        });
    control(ui, key, &r.response);
    changed
}
fn stage(ui: &mut egui::Ui, s: &mut H3Stage, raw: &mut String, id: &str) -> bool {
    ui.push_id(id, |ui| {
        let mut changed = false;
        ui.label("Steps");
        let r = ui.add(egui::DragValue::new(&mut s.steps).range(1..=u32::MAX));
        control(ui, &format!("{id}-steps"), &r);
        changed |= r.changed();
        for (label, key, value) in [
            ("Sampler", "sampler", &mut s.sampler),
            ("Scheduler", "scheduler", &mut s.scheduler),
        ] {
            ui.label(label);
            let r = ui.text_edit_singleline(value);
            control(ui, &format!("{id}-{key}"), &r);
            changed |= r.changed();
        }
        ui.label("Denoise");
        let r = ui.add(egui::DragValue::new(&mut s.denoise).speed(0.01));
        control(ui, &format!("{id}-denoise"), &r);
        changed |= r.changed();
        ui.label("Sigmas · optional, exact values");
        let r = ui.text_edit_singleline(raw);
        control(ui, &format!("{id}-sigmas"), &r);
        changed |= r.changed();
        changed
    })
    .inner
}
impl ImagineApp {
    pub(super) fn timeline_open_generation(&mut self) {
        let Some(d) = &self.timeline.document else {
            return;
        };
        let model = d.project.default_model.clone();
        let generation = defaults(&d.project);
        // The delivery was resolved from the source image, so it has its shape.
        let source = generation
            .canvas
            .as_ref()
            .filter(|c| c.ratio == "Source")
            .map(|_| (d.project.delivery.width, d.project.delivery.height));
        let value = generation
            .settings
            .iter()
            .find(|s| s.model == model)
            .map(|s| serde_json::from_value(s.values.clone()))
            .transpose();
        match value {
            Ok(value) => {
                self.timeline.generation_editor.draft = Some(Draft {
                    session: egui::Id::new(uuid::Uuid::new_v4()),
                    modes: Default::default(),
                    project: d.project.id.clone(),
                    model,
                    delivery: d.project.delivery.clone(),
                    seed: generation.seed.to_string(),
                    generation,
                    source,
                    sampling: SamplingDraft::new(value.unwrap_or_else(h3_defaults)),
                    error: None,
                })
            }
            Err(e) => self.timeline.error = Some(format!("Cannot edit these H3 settings: {e}")),
        }
    }
    pub(super) fn timeline_generation_window(&mut self, ctx: &egui::Context) {
        let Some(mut draft) = self.timeline.generation_editor.draft.take() else {
            return;
        };
        if self
            .timeline
            .document
            .as_ref()
            .is_none_or(|d| d.project.id != draft.project)
        {
            return;
        }
        let mut open = true;
        let mut apply = false;
        let mut cancel = false;
        egui::Window::new("Project generation")
            .id(egui::Id::new("project-generation"))
            .open(&mut open)
            .default_width(540.)
            .max_height((ctx.content_rect().height() - 70.).max(250.))
            .vscroll(true)
            .show(ctx, |ui| {
                #[cfg(feature = "ui-preview")]
                {
                    let r = ui.interact(
                        ui.clip_rect(),
                        ui.make_persistent_id("generation-scroll"),
                        egui::Sense::hover(),
                    );
                    control(ui, "generation-scroll", &r);
                }
                let hidden = !helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                helto::private_scope(ui, hidden, |ui| {
                    ui.label("Default scene mode");
                    let previous = draft.model.clone();
                    if mode(ui, &mut draft.model, "project-model") {
                        draft.error = draft.switch_mode(previous).err();
                    }
                    let mut canvas = draft.generation.canvas.clone().unwrap_or_default();
                    let mut canvas_changed = false;
                    ui.label("Quality");
                    let r = egui::ComboBox::from_id_salt("quality")
                        .selected_text(
                            draft
                                .generation
                                .canvas
                                .as_ref()
                                .map_or("Custom dimensions", |c| c.quality.as_str()),
                        )
                        .show_ui(ui, |ui| {
                            for q in crate::h3_protocol::geometry::QUALITIES {
                                let r = ui.selectable_value(&mut canvas.quality, q.into(), q);
                                control(ui, &format!("project-quality-{q}"), &r);
                                canvas_changed |= r.changed() || r.clicked();
                            }
                        });
                    control(ui, "project-quality", &r.response);
                    ui.horizontal_wrapped(|ui| {
                        let r = egui::ComboBox::from_id_salt("ratio")
                            .selected_text(&canvas.ratio)
                            .show_ui(ui, |ui| {
                                for ratio in draft
                                    .source
                                    .map(|_| "Source")
                                    .into_iter()
                                    .chain(crate::h3_protocol::geometry::RATIOS)
                                {
                                    let r =
                                        ui.selectable_value(&mut canvas.ratio, ratio.into(), ratio);
                                    control(ui, &format!("project-ratio-{ratio}"), &r);
                                    canvas_changed |= r.changed();
                                }
                            });
                        control(ui, "project-ratio", &r.response);
                        // Source already has the image's orientation.
                        let r = ui.add_enabled(
                            canvas.ratio != "Source",
                            egui::Checkbox::new(&mut canvas.portrait, "Portrait"),
                        );
                        control(ui, "project-portrait", &r);
                        canvas_changed |= r.changed();
                    });
                    if canvas_changed {
                        match canvas_size(&canvas, draft.source) {
                            Ok((width, height)) => {
                                draft.delivery.width = width;
                                draft.delivery.height = height;
                                draft.generation.canvas = Some(canvas);
                                draft.error = None;
                            }
                            Err(e) => draft.error = Some(e.into()),
                        }
                    }
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Width");
                        let r = ui.add(
                            egui::DragValue::new(&mut draft.delivery.width).range(1..=u32::MAX),
                        );
                        control(ui, "project-width", &r);
                        if r.changed() {
                            draft.generation.canvas = None;
                        }
                        ui.label("Height");
                        let r = ui.add(
                            egui::DragValue::new(&mut draft.delivery.height).range(1..=u32::MAX),
                        );
                        control(ui, "project-height", &r);
                        if r.changed() {
                            draft.generation.canvas = None;
                        }
                    });
                    ui.label("FPS");
                    let mut fps = editing::seconds(draft.delivery.fps);
                    let r = ui.add(
                        egui::DragValue::new(&mut fps)
                            .speed(1.)
                            .range(0.001..=f64::MAX),
                    );
                    control(ui, "project-fps", &r);
                    if r.changed() {
                        draft.delivery.fps = editing::time(fps);
                    }
                    ui.label("Seed · exact integer");
                    let r = ui.text_edit_singleline(&mut draft.seed);
                    control(ui, "project-seed", &r);
                    ui.label("Control after Generate");
                    let mut selected=draft.generation.control_after_generate.unwrap_or(SeedControl::Fixed);
                    let response=egui::ComboBox::from_id_salt("seed-control").selected_text(format!("{selected:?}")).show_ui(ui,|ui|{
                        for (mode,name) in [(SeedControl::Fixed,"Fixed"),(SeedControl::Increment,"Increment"),(SeedControl::Decrement,"Decrement"),(SeedControl::Randomize,"Randomize")] {
                            let r=ui.selectable_value(&mut selected,mode,name);
                            control(ui,&format!("project-seed-control-{}",name.to_lowercase()),&r);
                            if r.clicked(){draft.generation.control_after_generate=Some(mode);draft.generation.random_seed=false;}
                        }
                    }).response;
                    control(ui,"project-seed-control",&response);
                    if draft.generation.control_after_generate.is_none() && draft.generation.random_seed {
                        ui.weak("This older project uses a fresh seed before Generate. Choose a control above to replace that setting.");
                    }
                    ui.push_id((draft.session, &draft.model.variant), |ui| {
                        draft.sampling.show(ui, &draft.model);
                    });
                    let r=helto::icon_button(ui,crate::icons::Icon::Lora,"Project LoRAs…",false);control(ui,"project-loras",&r);
                    if r.clicked(){self.timeline_open_loras(None);}
                    if let Some(e) = &draft.error {
                        ui.colored_label(helto::ERROR, e);
                    }
                    ui.horizontal(|ui| {
                        let r = helto::action(ui, crate::icons::Icon::Check, "Apply", true, self.timeline.busy.then_some("Wait for the current timeline operation."));
                        control(ui, "generation-apply", &r);
                        apply = r.clicked();
                        let r = helto::icon_button(ui, crate::icons::Icon::Close, "Cancel", false);
                        control(ui, "generation-cancel", &r);
                        cancel = r.clicked();
                    });
                });
            });
        if apply {
            match (draft.seed.trim().parse::<u64>(), draft.sampling.resolve()) {
                (Ok(seed), Ok(value)) => {
                    draft.generation.seed = seed;
                    draft.generation.settings.retain(|s| s.model != draft.model);
                    draft
                        .generation
                        .settings
                        .push(slot(draft.model.clone(), &value));
                    let mut commands = self.timeline.draft_edits();
                    commands.extend([
                        Edit::SetDefaultModel {
                            model: draft.model.clone(),
                        },
                        Edit::SetDelivery {
                            delivery: draft.delivery.clone(),
                        },
                        Edit::SetGenerationDefaults {
                            generation: Some(draft.generation.clone()),
                        },
                    ]);
                    self.timeline.after_edit = Some(PostAction::GenerationSaved);
                    self.timeline_edit(ctx, commands, 0);
                    draft.error = None;
                }
                (Err(_), _) => {
                    draft.error =
                        Some("Seed must be an integer from 0 to 18446744073709551615.".into())
                }
                (_, Err(e)) => draft.error = Some(e),
            }
        }
        if open && !cancel {
            self.timeline.generation_editor.draft = Some(draft);
        }
    }
    pub(super) fn timeline_scene_generation(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        commands: &mut Vec<Edit>,
    ) {
        ui.label("Scene mode");
        let mut inherited = scene.model == ModelChoice::ProjectDefault;
        let r = ui.checkbox(&mut inherited, "Use project mode");
        control(ui, "scene-inherit-model", &r);
        let mut model = d.project.effective_model(scene).clone();
        if r.changed() {
            commands.push(Edit::SetSceneModel {
                scene_id: scene.id.clone(),
                model: if inherited {
                    ModelChoice::ProjectDefault
                } else {
                    ModelChoice::Override(model.clone())
                },
            });
        }
        if !inherited && mode(ui, &mut model, "scene-model") {
            commands.push(Edit::SetSceneModel {
                scene_id: scene.id.clone(),
                model: ModelChoice::Override(model.clone()),
            });
            if !scene.settings.iter().any(|s| s.model == model)
                && !d
                    .project
                    .generation
                    .as_ref()
                    .is_some_and(|g| g.settings.iter().any(|s| s.model == model))
            {
                commands.push(Edit::SetModelSettings {
                    scene_id: scene.id.clone(),
                    settings: slot(model.clone(), &h3_defaults()),
                });
            }
        }
        let header = egui::CollapsingHeader::new("Sampling & Stage 2")
            .id_salt(("scene-sampling", &scene.id.0))
            .show(ui, |ui| {
                let key = (d.project.id.clone(), scene.id.clone(), model.clone());
                let r = helto::icon_button(ui, crate::icons::Icon::Lora, "Scene LoRAs…", false);
                control(ui, "scene-loras", &r);
                if r.clicked() {
                    self.timeline_open_loras(Some(scene.id.clone()));
                }
                let mut inherit = !scene.settings.iter().any(|s| s.model == model);
                let r = ui.checkbox(&mut inherit, "Use project generation settings");
                control(ui, "scene-inherit-settings", &r);
                if r.changed() {
                    self.timeline.generation_editor.scenes.remove(&key);
                    if inherit {
                        let g = defaults(&d.project);
                        if d.project.generation.is_none() {
                            commands.push(Edit::SetGenerationDefaults {
                                generation: Some(g),
                            });
                        }
                        commands.push(Edit::InheritModelSettings {
                            scene_id: scene.id.clone(),
                            model: model.clone(),
                        });
                    } else {
                        let settings = d
                            .project
                            .effective_settings(scene)
                            .cloned()
                            .unwrap_or_else(|| slot(model.clone(), &h3_defaults()));
                        commands.push(Edit::SetModelSettings {
                            scene_id: scene.id.clone(),
                            settings,
                        });
                    }
                }
                let Some(settings) = d.project.effective_settings(scene) else {
                    ui.weak("Configure project generation defaults to inherit settings.");
                    return;
                };
                let cached = self
                    .timeline
                    .generation_editor
                    .scenes
                    .remove(&key)
                    .filter(|(source, draft)| source == &settings.values || draft.dirty);
                let mut draft = match cached {
                    Some((_, draft)) => draft,
                    None => match serde_json::from_value(settings.values.clone()) {
                        Ok(v) => SamplingDraft::new(v),
                        Err(e) => {
                            ui.colored_label(
                                helto::ERROR,
                                format!("Cannot edit these H3 settings: {e}"),
                            );
                            return;
                        }
                    },
                };
                if ui
                    .push_id(("h3-settings-model", &model.variant), |ui| {
                        draft.show(ui, &model)
                    })
                    .inner
                {
                    draft.dirty = true;
                    draft.error = draft.resolve().err();
                }
                if draft
                    .resolve()
                    .ok()
                    .is_some_and(|v| settings.values == slot(model.clone(), &v).values)
                {
                    draft.dirty = false;
                }
                self.timeline
                    .generation_editor
                    .scenes
                    .insert(key, (settings.values.clone(), draft));
            });
        control(ui, "scene-sampling", &header.header_response);
    }
}
