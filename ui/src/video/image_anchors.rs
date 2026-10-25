//! Section image anchors keep authored roles and reference semantics separate from scene bindings.
use super::*;
use imagine_video_contract::creative::{Creative, ImageBinding};
impl Timeline {
    pub(super) fn section_preview(&self, p: &VideoProject, id: &SectionId) -> Option<AssetId> {
        if self.selection_items(p) != [selection::Item::Section(id.clone())] {
            return None;
        }
        let image = p.creative.sections.get(id)?.images.first()?;
        (p.assets.get(&image.asset)?.kind == MediaKind::Image).then(|| image.asset.clone())
    }
}
#[derive(Default)]
pub(super) struct State {
    project: Option<ProjectId>,
    section: Option<SectionId>,
    base: Vec<ImageBinding>,
    draft: Vec<ImageBinding>,
    changed_at: Option<std::time::Instant>,
}
impl State {
    pub(super) fn merge_pending(&self, project: &VideoProject, creative: &mut Creative) {
        if self.project.as_ref() == Some(&project.id)
            && self.draft != self.base
            && let Some(section) = &self.section
            && project
                .scenes
                .iter()
                .any(|s| s.sections.iter().any(|s| &s.id == section))
        {
            creative.sections.entry(section.clone()).or_default().images = self.draft.clone();
        }
    }
}

impl ImagineApp {
    pub(super) fn timeline_image_anchors(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        section: &SectionId,
        commands: &mut Vec<Edit>,
    ) {
        let source_images = d
            .project
            .creative
            .sections
            .get(section)
            .map(|s| s.images.clone())
            .unwrap_or_default();
        let mut state = std::mem::take(&mut self.timeline.image_anchors);
        if state.project.as_ref() != Some(&d.project.id)
            || state.section.as_ref() != Some(section)
            || state.draft == source_images
            || state.draft == state.base && state.base != source_images
        {
            state = State {
                project: Some(d.project.id.clone()),
                section: Some(section.clone()),
                base: source_images.clone(),
                draft: source_images.clone(),
                changed_at: None,
            };
        }
        if state.draft.is_empty() && state.base.is_empty() {
            self.timeline.image_anchors = state;
            return;
        }
        ui.strong("Image anchors");
        let scene = d
            .project
            .scenes
            .iter()
            .find(|s| s.sections.iter().any(|x| &x.id == section))
            .cloned();
        let before_images = state.draft.clone();
        let mut images = state.draft.clone();
        let mut remove = None;
        let mut flush = false;
        for (i, image) in images.iter_mut().enumerate() {
            ui.push_id(&image.image_id, |ui| {
                ui.group(|ui| {
                    if let Some(asset) = d.project.assets.get(&image.asset) {
                        self.timeline_thumbnail(ui, d, asset);
                        let reveal = helto::shown(ui, self.prefs.privacy_mode);
                        ui.add(
                            egui::Label::new(if !reveal {
                                "Image"
                            } else if asset.name.is_empty() {
                                asset.id.0.as_str()
                            } else {
                                asset.name.as_str()
                            })
                            .truncate(),
                        );
                    }
                    let mut anchor = image.anchor.clone().unwrap_or_else(|| {
                        imagine_video_contract::creative::ImageAnchor {
                            role: match image.role {
                                InputRole::FirstFrame => "first_frame",
                                InputRole::LastFrame => "last_frame",
                                _ => "influence",
                            }
                            .into(),
                            ..Default::default()
                        }
                    });
                    let before = anchor.clone();
                    let role_label = match anchor.role.as_str() {
                        "auto" => "Auto",
                        "first_frame" => "First frame",
                        "keyframe" => "Keyframe",
                        "last_frame" => "Last frame",
                        "influence" => "Influence",
                        "custom_influence" => "Custom influence",
                        other => other,
                    };
                    let menu = egui::ComboBox::from_id_salt("image-anchor-role")
                        .selected_text(role_label)
                        .show_ui(ui, |ui| {
                            for (role, label) in [
                                ("auto", "Auto"),
                                ("first_frame", "First frame"),
                                ("keyframe", "Keyframe"),
                                ("last_frame", "Last frame"),
                                ("influence", "Influence"),
                                ("custom_influence", "Custom influence"),
                            ] {
                                // Auto resolves per shot; explicit roles follow the model's inputs.
                                let input = match role {
                                    "first_frame" => Some(InputRole::FirstFrame),
                                    "last_frame" => Some(InputRole::LastFrame),
                                    "auto" => None,
                                    _ => Some(InputRole::VisualReference),
                                };
                                let refused = anchor.role != role
                                    && input.zip(scene.as_ref()).is_some_and(|(input, scene)| {
                                        self.timeline_accepts(
                                            &d.project,
                                            scene,
                                            &input,
                                            &MediaKind::Image,
                                        ) == Some(false)
                                    });
                                let r = ui
                                    .add_enabled_ui(!refused, |ui| {
                                        ui.selectable_value(&mut anchor.role, role.into(), label)
                                    })
                                    .inner
                                    .on_disabled_hover_text(
                                        "The scene's model does not take an image in this role.",
                                    );
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(ui, format!("anchor-role-{i}-{role}"), &r);
                                let _ = r;
                            }
                        });
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("anchor-role-{i}"), &menu.response);
                    let _ = menu;
                    if anchor.role == "custom_influence" {
                        ui.label("Custom use");
                        let r = prompt_tools::textarea(
                            ui,
                            &mut anchor.custom_role,
                            2,
                            "Image custom use",
                            self.prefs.privacy_mode,
                            false,
                            prompt_tools::Source::section(
                                &d.project.creative,
                                &d.project.id,
                                section,
                            ),
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, format!("anchor-custom-{i}"), &r);
                        flush |= r.lost_focus();
                    }
                    let relationship_label = match anchor.relationship.marker.as_str() {
                        "fully_preserved" => "Fully preserved",
                        "partially_preserved" => "Partially preserved",
                        "attribute_transfer" => "Attribute transfer",
                        "weak_reference" => "Weak reference",
                        other => other,
                    };
                    egui::ComboBox::from_id_salt("image-anchor-relationship")
                        .selected_text(relationship_label)
                        .show_ui(ui, |ui| {
                            for (value, label) in [
                                ("fully_preserved", "Fully preserved"),
                                ("partially_preserved", "Partially preserved"),
                                ("attribute_transfer", "Attribute transfer"),
                                ("weak_reference", "Weak reference"),
                            ] {
                                if ui
                                    .selectable_value(
                                        &mut anchor.relationship.marker,
                                        value.into(),
                                        label,
                                    )
                                    .changed()
                                    && value != "attribute_transfer"
                                {
                                    anchor.relationship.target_reference_id = None;
                                }
                            }
                        });
                    if anchor.relationship.marker == "attribute_transfer" {
                        egui::ComboBox::from_id_salt("image-anchor-target")
                            .selected_text(
                                d.project
                                    .creative
                                    .references
                                    .iter()
                                    .find(|r| {
                                        Some(&r.id)
                                            == anchor.relationship.target_reference_id.as_ref()
                                    })
                                    .map_or("Choose target…", |r| {
                                        if helto::shown(ui, self.prefs.privacy_mode) {
                                            r.name.as_str()
                                        } else {
                                            "Hidden target"
                                        }
                                    }),
                            )
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut anchor.relationship.target_reference_id,
                                    None,
                                    "None",
                                );
                                for reference in &d.project.creative.references {
                                    ui.selectable_value(
                                        &mut anchor.relationship.target_reference_id,
                                        Some(reference.id.clone()),
                                        if helto::shown(ui, self.prefs.privacy_mode) {
                                            reference.name.as_str()
                                        } else {
                                            "Hidden target"
                                        },
                                    );
                                }
                            });
                    }
                    ui.label("Retention");
                    let r = prompt_tools::textarea(
                        ui,
                        &mut anchor.relationship.note,
                        2,
                        "Image retention",
                        self.prefs.privacy_mode,
                        false,
                        prompt_tools::Source::section(&d.project.creative, &d.project.id, section),
                    );
                    flush |= r.lost_focus();
                    if anchor != before {
                        image.anchor = Some(anchor);
                    }
                    let r =
                        helto::icon_button(ui, crate::icons::Icon::Trash, "Remove anchor", false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("anchor-remove-{i}"), &r);
                    if r.clicked() {
                        remove = Some(i);
                    }
                });
            });
        }
        if let Some(i) = remove {
            images.remove(i);
        }
        if images != before_images {
            state.draft = images;
            state.changed_at = Some(std::time::Instant::now());
        }
        flush |= state
            .changed_at
            .is_some_and(|t| t.elapsed() > Duration::from_millis(150));
        if state.changed_at.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(150));
        }
        self.timeline.image_anchors = state;
        if flush && !self.timeline.busy {
            commands.extend(self.timeline.draft_edits());
        }
    }
}
