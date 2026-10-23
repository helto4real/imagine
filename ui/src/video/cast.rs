//! Explicit project cast editing; media remains project-owned, previews are independent.
use super::*;
use imagine_video_contract::creative::*;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    draft: Creative,
    base: Option<(ProjectId, Vec<VisualReference>, Option<bool>)>,
    error: Option<String>,
    pending: Option<(Vec<VisualReference>, Option<bool>)>,
    changed_at: Option<std::time::Instant>,
}
impl State {
    pub(super) fn merge_pending(&self, project: &VideoProject, creative: &mut Creative) {
        if self
            .base
            .as_ref()
            .is_some_and(|(id, _, _)| id == &project.id)
            && self.changed_at.is_some()
        {
            creative.references = self.draft.references.clone();
            creative.visual_references_enabled = self.draft.visual_references_enabled;
        }
    }
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
fn semantic_choice(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut String,
    choices: &[(&str, &str)],
) -> egui::Response {
    let selected = choices
        .iter()
        .find(|(v, _)| *v == value.to_lowercase())
        .map_or(value.as_str(), |(_, label)| *label)
        .to_owned();
    let menu = egui::ComboBox::from_id_salt(id)
        .selected_text(selected)
        .show_ui(ui, |ui| {
            for (key, label) in choices {
                let r = ui.selectable_value(value, (*key).into(), *label);
                control(ui, &format!("cast-choice-{id}-{key}"), &r);
            }
        });
    ui.push_id(("custom-value", id), |ui| {
        ui.collapsing("Custom value", |ui| {
            ui.text_edit_singleline(value);
        });
    });
    menu.response
}
impl ImagineApp {
    pub(super) fn timeline_open_cast(&mut self) {
        if let Some(d) = &self.timeline.document {
            if self
                .timeline
                .cast
                .base
                .as_ref()
                .is_some_and(|(id, _, _)| id == &d.project.id)
            {
                self.timeline.cast.open = true;
                return;
            }
            self.timeline.cast = State {
                open: true,
                draft: d.project.creative.clone(),
                base: Some((
                    d.project.id.clone(),
                    d.project.creative.references.clone(),
                    d.project.creative.visual_references_enabled,
                )),
                error: None,
                pending: None,
                changed_at: None,
            };
        }
    }
    pub(super) fn timeline_cast_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.cast.open
            && self.timeline.cast.pending.is_none()
            && self.timeline.cast.changed_at.is_none()
        {
            return;
        }
        let Some(document) = self.timeline.document.clone() else {
            return;
        };
        let mut state = std::mem::take(&mut self.timeline.cast);
        let current = (
            document.project.id.clone(),
            document.project.creative.references.clone(),
            document.project.creative.visual_references_enabled,
        );
        if !self.timeline.busy {
            if state.changed_at.is_some()
                && current.1 == state.draft.references
                && current.2 == state.draft.visual_references_enabled
            {
                state.base = Some(current.clone());
                state.changed_at = None;
            }
            if let Some((sent, enabled)) = state.pending.take() {
                if current.1 == sent && current.2 == enabled {
                    state.base = Some(current.clone());
                    state.error = None;
                } else {
                    state.error = Some(self.timeline.error.clone().unwrap_or_else(|| {
                        "Reference edit was not saved. Retry your edit.".into()
                    }));
                }
            } else if state.base.as_ref() != Some(&current) && state.changed_at.is_none() {
                state.draft = document.project.creative.clone();
                state.base = Some(current.clone());
            }
        }
        let reference_choices = state
            .draft
            .references
            .iter()
            .map(|r| (r.id.clone(), format!("{} · @{}", r.name, r.alias)))
            .collect::<Vec<_>>();
        let before = (
            state.draft.references.clone(),
            state.draft.visual_references_enabled,
        );
        let mut open = state.open;
        let mut apply = false;
        let mut reload = false;
        let mut pick_media = None;
        let mut close = false;
        let mut insert_tag = None;
        let mut library_reference = None;
        egui::Window::new("Cast & Visual References")
            .id(egui::Id::new("video-cast"))
            .open(&mut open)
            .default_width(650_f32.min((ctx.content_rect().width() - 24.).max(240.)))
            .default_height(700.)
            .max_width((ctx.content_rect().width() - 24.).max(240.))
            .max_height((ctx.content_rect().height() - 70.).max(200.))
            .vscroll(true)
            .show(ctx, |ui| {
                ui.set_max_width((ctx.content_rect().width() - 48.).max(216.));
                #[cfg(feature = "ui-preview")]
                {
                    let r = ui.interact(
                        ui.clip_rect(),
                        ui.id().with("qa-cast-scroll"),
                        egui::Sense::hover(),
                    );
                    control(ui, "cast-scroll", &r);
                }
                let hidden = !helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                helto::private_scope(ui, hidden, |ui| {
                    if self.timeline.busy {
                        ui.disable();
                    }
                    let mut enabled = state.draft.visual_references_enabled.unwrap_or(true);
                    let r = ui.checkbox(&mut enabled, "Use visual references");
                    control(ui, "cast-use-references", &r);
                    if r.changed() {
                        state.draft.visual_references_enabled = Some(enabled);
                    }
                    ui.horizontal_wrapped(|ui| {
                        let r =
                            helto::icon_button(ui, crate::icons::Icon::Plus, "Add reference", true);
                        control(ui, "cast-add", &r);
                        if r.clicked() {
                            let n = state.draft.references.len() + 1;
                            state.draft.references.push(VisualReference {
                                id: format!("reference_{}", uuid::Uuid::new_v4().simple()),
                                name: format!("Character {n}"),
                                alias: format!("character_{n}"),
                                kind: "character".into(),
                                description: String::new(),
                                tags: vec![],
                                enabled: true,
                                strength: 1.,
                                retention: String::new(),
                                relationship: None,
                                images: vec![],
                                voice: None,
                                origin: None,
                            });
                        }
                        let r = helto::icon_button(ui, crate::icons::Icon::Close, "Close", false);
                        control(ui, "cast-close", &r);
                        close = r.clicked();
                        let r = helto::icon_button(
                            ui,
                            crate::icons::Icon::Retry,
                            "Reload current project",
                            false,
                        );
                        control(ui, "cast-reload", &r);
                        reload = r.clicked();
                    });
                    let mut remove = None;
                    for (i, reference) in state.draft.references.iter_mut().enumerate() {
                        ui.push_id(reference.id.clone(), |ui| {
                            ui.group(|ui| {
                                ui.horizontal_wrapped(|ui| {
                                    let r = ui.checkbox(&mut reference.enabled, "Enabled");
                                    control(ui, &format!("cast-enabled-{i}"), &r);
                                    ui.strong(&reference.name);
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Save,
                                        "Save or update reference in library",
                                        false,
                                    );
                                    control(ui, &format!("cast-library-{i}"), &r);
                                    if r.clicked() {
                                        library_reference = Some((i, reference.clone()));
                                    }
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Trash,
                                        "Remove reference",
                                        false,
                                    );
                                    control(ui, &format!("cast-remove-{i}"), &r);
                                    if r.clicked() {
                                        remove = Some(i);
                                    }
                                });
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Name");
                                    let r = ui.text_edit_singleline(&mut reference.name);
                                    control(ui, &format!("cast-name-{i}"), &r);
                                });
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Alias @");
                                    let r = ui.text_edit_singleline(&mut reference.alias);
                                    control(ui, &format!("cast-alias-{i}"), &r);
                                });
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Kind");
                                    let r = semantic_choice(
                                        ui,
                                        "kind",
                                        &mut reference.kind,
                                        &[
                                            ("character", "Character / Creature"),
                                            ("object", "Object / Prop"),
                                            ("scene", "Scene / Environment"),
                                            ("appearance", "Appearance / Clothing / VFX"),
                                            ("style_action", "Style / Action / Pose"),
                                        ],
                                    );
                                    control(ui, &format!("cast-kind-{i}"), &r);
                                });
                                ui.horizontal_wrapped(|ui| {
                                    let tag = super::prompt_tools::reference_tag(&reference.alias);
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Copy,
                                        "Copy Reference Tag",
                                        false,
                                    );
                                    control(ui, &format!("cast-copy-tag-{i}"), &r);
                                    if r.clicked() {
                                        ui.ctx().copy_text(tag.clone());
                                    }
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Insert,
                                        "Insert Reference Tag in selected Section",
                                        false,
                                    );
                                    control(ui, &format!("cast-insert-tag-{i}"), &r);
                                    if r.clicked() {
                                        insert_tag = Some(tag);
                                    }
                                });
                                ui.label("Description");
                                let r = ui.add(
                                    egui::TextEdit::multiline(&mut reference.description)
                                        .desired_rows(2)
                                        .desired_width(f32::INFINITY),
                                );
                                control(ui, &format!("cast-description-{i}"), &r);
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Tags");
                                    let mut text = reference.tags.join(", ");
                                    if ui.text_edit_singleline(&mut text).changed() {
                                        reference.tags = text
                                            .split(',')
                                            .map(str::trim)
                                            .filter(|s| !s.is_empty())
                                            .map(str::to_owned)
                                            .collect();
                                    }
                                });
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Strength");
                                    ui.add(
                                        egui::DragValue::new(&mut reference.strength).speed(0.05),
                                    );
                                    ui.label("Retention");
                                    ui.text_edit_singleline(&mut reference.retention);
                                });
                                let mut relationship = reference.relationship.is_some();
                                let r = ui.checkbox(&mut relationship, "Reference relationship");
                                control(ui, &format!("cast-relationship-{i}"), &r);
                                if r.changed() {
                                    reference.relationship =
                                        relationship.then(|| ReferenceRelationship {
                                            marker: if reference.retention.is_empty() {
                                                "fully_preserved".into()
                                            } else {
                                                reference.retention.clone()
                                            },
                                            ..Default::default()
                                        });
                                }
                                if let Some(value) = &mut reference.relationship {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("Relationship");
                                        let previous = value.marker.clone();
                                        let r = semantic_choice(
                                            ui,
                                            "relationship",
                                            &mut value.marker,
                                            &[
                                                ("fully_preserved", "Fully preserved"),
                                                ("partially_preserved", "Partially preserved"),
                                                ("attribute_transfer", "Attribute transfer"),
                                                ("weak_reference", "Weak reference"),
                                            ],
                                        );
                                        if value.marker != previous
                                            && value.marker != "attribute_transfer"
                                        {
                                            value.target_reference_id = None;
                                        }
                                        control(ui, &format!("cast-relationship-marker-{i}"), &r);
                                    });
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("Transfer target");
                                        egui::ComboBox::from_id_salt("relationship-target")
                                            .selected_text(
                                                reference_choices
                                                    .iter()
                                                    .find(|(id, _)| {
                                                        Some(id)
                                                            == value.target_reference_id.as_ref()
                                                    })
                                                    .map_or("Choose target…", |(_, label)| {
                                                        label.as_str()
                                                    }),
                                            )
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(
                                                    &mut value.target_reference_id,
                                                    None,
                                                    "None",
                                                );
                                                for (id, label) in &reference_choices {
                                                    if id != &reference.id {
                                                        ui.selectable_value(
                                                            &mut value.target_reference_id,
                                                            Some(id.clone()),
                                                            label,
                                                        );
                                                    }
                                                }
                                            });
                                    });
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("ID");
                                        let mut target =
                                            value.target_reference_id.clone().unwrap_or_default();
                                        let r = ui.text_edit_singleline(&mut target);
                                        control(ui, &format!("cast-relationship-target-{i}"), &r);
                                        if r.changed() {
                                            value.target_reference_id =
                                                (!target.is_empty()).then_some(target);
                                        }
                                    });
                                    ui.label("Relationship note");
                                    let r = ui.text_edit_multiline(&mut value.note);
                                    control(ui, &format!("cast-relationship-note-{i}"), &r);
                                }
                                let mut default_preview = None;
                                let images = ui.collapsing("Images", |ui| {
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Import,
                                        "Add reference images…",
                                        false,
                                    );
                                    control(ui, &format!("cast-pick-image-{i}"), &r);
                                    if r.clicked() {
                                        pick_media = Some(media_picker::Target::ReferenceImage(
                                            reference.id.clone(),
                                        ));
                                    }
                                    for (ordinal, asset) in document
                                        .project
                                        .assets
                                        .values()
                                        .filter(|a| a.kind == MediaKind::Image)
                                        .enumerate()
                                    {
                                        if reference.images.iter().any(|im| im.asset == asset.id) {
                                            self.timeline_thumbnail(ui, &document, asset);
                                        }
                                        let mut included =
                                            reference.images.iter().any(|im| im.asset == asset.id);
                                        let r = ui.checkbox(
                                            &mut included,
                                            format!("Image {}", ordinal + 1),
                                        );
                                        control(ui, &format!("cast-image-{i}-{ordinal}"), &r);
                                        if r.changed() {
                                            if included {
                                                reference.images.push(ReferenceImage {
                                                    id: format!(
                                                        "image_{}",
                                                        uuid::Uuid::new_v4().simple()
                                                    ),
                                                    asset: asset.id.clone(),
                                                    enabled: true,
                                                    preview: reference.images.is_empty(),
                                                    contribution: String::new(),
                                                });
                                            } else {
                                                reference.images.retain(|im| im.asset != asset.id);
                                            }
                                        }
                                        if let Some(image) = reference
                                            .images
                                            .iter_mut()
                                            .find(|im| im.asset == asset.id)
                                        {
                                            ui.horizontal_wrapped(|ui| {
                                                let r = ui.checkbox(
                                                    &mut image.enabled,
                                                    "Use for generation",
                                                );
                                                control(
                                                    ui,
                                                    &format!("cast-image-enabled-{i}-{ordinal}"),
                                                    &r,
                                                );
                                                let r = ui.radio(image.preview, "Default preview");
                                                control(
                                                    ui,
                                                    &format!("cast-image-preview-{i}-{ordinal}"),
                                                    &r,
                                                );
                                                if r.clicked() {
                                                    default_preview = Some(image.asset.clone());
                                                }
                                            });
                                            ui.label("Image contribution");
                                            let r =
                                                ui.text_edit_singleline(&mut image.contribution);
                                            control(
                                                ui,
                                                &format!("cast-image-contribution-{i}-{ordinal}"),
                                                &r,
                                            );
                                        }
                                    }
                                });
                                control(ui, &format!("cast-images-{i}"), &images.header_response);
                                if let Some(id) = default_preview {
                                    for image in &mut reference.images {
                                        image.preview = image.asset == id;
                                    }
                                }
                                let mut voice = reference.voice.is_some();
                                let r = ui.checkbox(&mut voice, "Voice profile");
                                control(ui, &format!("cast-voice-profile-{i}"), &r);
                                if r.changed() {
                                    reference.voice = voice.then(|| Voice {
                                        language: String::new(),
                                        description: String::new(),
                                        assets: vec![],
                                        reference_settings: Default::default(),
                                    });
                                }
                                if let Some(voice) = &mut reference.voice {
                                    let r = helto::icon_button(
                                        ui,
                                        crate::icons::Icon::Import,
                                        "Add voice references…",
                                        false,
                                    );
                                    control(ui, &format!("cast-pick-voice-{i}"), &r);
                                    if r.clicked() {
                                        pick_media = Some(media_picker::Target::ReferenceVoice(
                                            reference.id.clone(),
                                        ));
                                    }
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("Language");
                                        let r = ui.text_edit_singleline(&mut voice.language);
                                        control(ui, &format!("cast-voice-language-{i}"), &r);
                                    });
                                    ui.label("Voice guidance");
                                    let r = ui.text_edit_multiline(&mut voice.description);
                                    control(ui, &format!("cast-voice-guidance-{i}"), &r);
                                    for (ordinal, asset) in document
                                        .project
                                        .assets
                                        .values()
                                        .filter(|a| a.kind == MediaKind::Audio)
                                        .enumerate()
                                    {
                                        let mut selected = voice.assets.contains(&asset.id);
                                        let r = ui.checkbox(
                                            &mut selected,
                                            format!("Voice audio {}", ordinal + 1),
                                        );
                                        control(ui, &format!("cast-voice-audio-{i}-{ordinal}"), &r);
                                        if r.changed() {
                                            voice.assets.retain(|id| id != &asset.id);
                                            if selected {
                                                voice.assets.push(asset.id.clone());
                                            }
                                        }
                                        if selected {
                                            let settings = voice
                                                .reference_settings
                                                .entry(asset.id.clone())
                                                .or_default();
                                            let r = ui.checkbox(
                                                &mut settings.enabled,
                                                "Use voice reference",
                                            );
                                            control(
                                                ui,
                                                &format!("cast-voice-enabled-{i}-{ordinal}"),
                                                &r,
                                            );
                                            ui.label("Voice description");
                                            let r = ui.text_edit_singleline(
                                                &mut settings.voice_description,
                                            );
                                            control(
                                                ui,
                                                &format!("cast-voice-description-{i}-{ordinal}"),
                                                &r,
                                            );
                                            ui.label("Voice contribution");
                                            let r =
                                                ui.text_edit_singleline(&mut settings.contribution);
                                            control(
                                                ui,
                                                &format!("cast-voice-contribution-{i}-{ordinal}"),
                                                &r,
                                            );
                                            ui.horizontal_wrapped(|ui| {
                                                ui.label("Audio relationship");
                                                ui.text_edit_singleline(&mut settings.relationship);
                                            });
                                            let mut strength = settings.strength.is_some();
                                            if ui
                                                .checkbox(
                                                    &mut strength,
                                                    "Explicit reference strength",
                                                )
                                                .changed()
                                            {
                                                settings.strength = strength.then_some(1.);
                                            }
                                            if let Some(value) = &mut settings.strength {
                                                ui.add(egui::DragValue::new(value).speed(0.05));
                                            }
                                        }
                                    }
                                }
                            });
                        });
                    }
                    if let Some(i) = remove {
                        state.draft.references.remove(i);
                    }
                    if let Some(error) = &state.error {
                        ui.colored_label(helto::ERROR, error);
                    }
                    if state.pending.is_some() {
                        ui.spinner();
                    }
                });
            });
        open &= !close;
        state.open = open;
        if before
            != (
                state.draft.references.clone(),
                state.draft.visual_references_enabled,
            )
        {
            state.changed_at = Some(std::time::Instant::now());
        }
        if state
            .changed_at
            .is_some_and(|t| t.elapsed() > Duration::from_millis(300))
            || (!open || pick_media.is_some() || library_reference.is_some())
                && state.changed_at.is_some()
        {
            apply = true;
        }
        if state.changed_at.is_some() {
            ctx.request_repaint_after(Duration::from_millis(300));
        }
        if reload {
            state.draft = document.project.creative.clone();
            state.base = Some((
                document.project.id.clone(),
                document.project.creative.references.clone(),
                document.project.creative.visual_references_enabled,
            ));
            state.error = None;
        }
        if let Some(tag) = insert_tag
            && self.timeline.section.is_some()
        {
            let text = self.timeline.draft_section.trim_end();
            self.timeline.draft_section = if text.is_empty() {
                tag
            } else {
                format!("{text} {tag}")
            };
            apply = true;
        }
        if apply && !self.timeline.busy && state.pending.is_none() {
            if state.base.as_ref()
                != Some(&(
                    document.project.id.clone(),
                    document.project.creative.references.clone(),
                    document.project.creative.visual_references_enabled,
                ))
            {
                state.error = Some(
                    "Project references changed. Reload them before applying this edit.".into(),
                );
            } else {
                let mut edits = self.timeline.draft_edits();
                let mut creative = edits
                    .iter()
                    .find_map(|edit| {
                        if let Edit::SetCreative { creative } = edit {
                            Some(creative.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| document.project.creative.clone());
                edits.retain(|edit| !matches!(edit, Edit::SetCreative { .. }));
                creative.references = state.draft.references.clone();
                creative.visual_references_enabled = state.draft.visual_references_enabled;
                state.pending = Some((
                    creative.references.clone(),
                    creative.visual_references_enabled,
                ));
                state.changed_at = None;
                edits.push(Edit::SetCreative { creative });
                self.timeline_edit(ctx, edits, 0);
            }
        }
        self.timeline.cast = state;
        if let Some((index, reference)) = library_reference {
            self.library_reference(ctx, index, &reference);
        }
        if let Some(target) = pick_media {
            self.timeline_pick_media(target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immediate_project_action_flushes_pending_cast_and_scene_text_in_one_edit() {
        let project = super::super::media_picker::tests::project();
        let mut timeline = Timeline::default();
        timeline.document = Some(Snapshot::with(project.clone(), serde_json::json!({})));
        timeline.selected.insert(SceneId("scene".into()));
        timeline.draft_scene = Some(SceneId("scene".into()));
        timeline.draft_scene_prompt.summary = "Own new scene summary".into();
        let mut creative = project.creative.clone();
        creative.references[0].name = "Own pending cast name".into();
        timeline.cast = State {
            draft: creative,
            base: Some((
                project.id.clone(),
                project.creative.references.clone(),
                project.creative.visual_references_enabled,
            )),
            changed_at: Some(std::time::Instant::now()),
            ..Default::default()
        };
        let commands = timeline.draft_edits();
        assert_eq!(
            commands
                .iter()
                .filter(|c| matches!(c, Edit::SetCreative { .. }))
                .count(),
            1
        );
        let mut result = project;
        let revision = result.project_revision;
        imagine_video_contract::editing::apply(
            &mut result,
            EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands,
            },
        )
        .unwrap();
        assert_eq!(result.creative.references[0].name, "Own pending cast name");
        assert_eq!(
            result.creative.scene_prompts[&SceneId("scene".into())].summary,
            "Own new scene summary"
        );
    }
}
