//! Asset bindings and reference-video source options, independent of inference.
use super::*;
impl ImagineApp {
    pub(super) fn timeline_media_binding(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        asset: &Asset,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        let section = self
            .timeline
            .section
            .clone()
            .filter(|_| asset.kind == MediaKind::Video);
        let source = section
            .as_ref()
            .and_then(|id| d.project.creative.sections.get(id));
        let bindings = if section.is_some() {
            source.map(|s| s.videos.as_slice()).unwrap_or_default()
        } else {
            scene.bindings.as_slice()
        };
        let current = bindings.iter().find(|b| b.asset == asset.id);
        let mut binding = current.cloned().unwrap_or_else(|| Binding {
            id: BindingId(format!("binding_{}", uuid::Uuid::new_v4().simple())),
            asset: asset.id.clone(),
            role: match asset.kind {
                MediaKind::Image => InputRole::VisualReference,
                MediaKind::Video => InputRole::VideoReference,
                MediaKind::Audio => InputRole::AudioReference,
            },
            enabled: false,
            video: None,
        });
        let before = binding.clone();
        // Offer only inputs the scene's model takes; an enabled binding can still be turned off.
        let refused = !binding.enabled
            && self.timeline_accepts(&d.project, scene, &binding.role, &asset.kind) == Some(false);
        let response = ui
            .add_enabled(
                enabled && !refused,
                egui::Checkbox::new(&mut binding.enabled, "Use for generation"),
            )
            .on_disabled_hover_text(if refused {
                "The scene's model does not take this media as input."
            } else {
                "Wait for the current timeline operation."
            });
        #[cfg(feature = "ui-preview")]
        super::qa::control(ui, format!("media-use-{}", asset.id.0), &response);
        let _ = response;
        if current.is_some() {
            if asset.kind == MediaKind::Video {
                let mut options = binding.video.clone().unwrap_or_default();
                let mut changed = false;
                ui.add_enabled_ui(enabled, |ui| {
                    for (id, title, value, choices) in [
                        (
                            "role",
                            "Role",
                            &mut options.role,
                            vec![
                                ("video_reference", "Video reference"),
                                ("source_video_edit", "Source video edit"),
                                ("custom_video_reference", "Custom video reference"),
                            ],
                        ),
                        (
                            "soundtrack",
                            "Soundtrack",
                            &mut options.soundtrack_usage,
                            vec![
                                ("ignore", "Ignore"),
                                ("reference", "Reference"),
                                ("final_mix", "Final mix"),
                                ("both", "Reference + final mix"),
                            ],
                        ),
                    ] {
                        ui.label(title);
                        let selected = choices
                            .iter()
                            .find(|(v, _)| *v == value.as_str())
                            .or_else(|| value.is_empty().then(|| &choices[0]))
                            .map_or(value.as_str(), |(_, label)| *label);
                        let menu = egui::ComboBox::from_id_salt((&asset.id.0, id))
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                for (v, label) in &choices {
                                    let r = ui.selectable_value(value, (*v).into(), *label);
                                    #[cfg(feature = "ui-preview")]
                                    super::qa::control(
                                        ui,
                                        format!("video-{id}-{}-{v}", asset.id.0),
                                        &r,
                                    );
                                    changed |= r.changed();
                                }
                            });
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(
                            ui,
                            format!("video-{id}-{}", asset.id.0),
                            &menu.response,
                        );
                        let _ = menu;
                    }
                    if options.role == "custom_video_reference" {
                        ui.label("Custom role");
                        changed |= helto::textarea(
                            ui,
                            &mut options.custom_role,
                            2,
                            "Motion, camera or timing",
                            self.prefs.privacy_mode,
                            true,
                        )
                        .lost_focus();
                    }
                    for (_id, title, value) in [
                        ("in", "Source in", &mut options.source_in),
                        ("out", "Source out", &mut options.source_out),
                    ] {
                        let mut use_range = value.is_some();
                        let r = ui.checkbox(&mut use_range, title);
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, format!("video-{_id}-toggle-{}", asset.id.0), &r);
                        if r.changed() {
                            *value = use_range.then_some(Time { num: 0, den: 1 });
                            changed = true;
                        }
                        if let Some(time) = value {
                            let mut seconds = editing::seconds(*time);
                            let r = ui.add(
                                egui::DragValue::new(&mut seconds)
                                    .range(0.0..=f64::MAX)
                                    .speed(0.05)
                                    .suffix(" s"),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, format!("video-{_id}-{}", asset.id.0), &r);
                            if r.changed() {
                                *time = editing::time(seconds);
                                changed = true;
                            }
                        }
                    }
                });
                if changed {
                    binding.video = Some(options);
                }
            } else {
                egui::ComboBox::from_id_salt((&binding.id.0, "role"))
                    .width(130.)
                    .selected_text(role_label(&binding.role))
                    .show_ui(ui, |ui| {
                        let roles = if asset.kind == MediaKind::Image {
                            vec![
                                InputRole::FirstFrame,
                                InputRole::LastFrame,
                                InputRole::VisualReference,
                            ]
                        } else {
                            vec![InputRole::AudioReference, InputRole::VoiceReference]
                        };
                        for role in roles {
                            ui.selectable_value(&mut binding.role, role.clone(), role_label(&role));
                        }
                    });
            }
        }
        if binding != before {
            if let Some(section) = section {
                let mut creative = d.project.creative.clone();
                let target = &mut creative.sections.entry(section).or_default().videos;
                if let Some(old) = target.iter_mut().find(|b| b.id == binding.id) {
                    *old = binding;
                } else {
                    target.push(binding);
                }
                commands.push(Edit::SetCreative { creative });
            } else {
                let mut target = scene.bindings.clone();
                if let Some(old) = target.iter_mut().find(|b| b.id == binding.id) {
                    *old = binding;
                } else {
                    target.push(binding);
                }
                commands.push(Edit::SetBindings {
                    scene_id: scene.id.clone(),
                    bindings: target,
                });
            }
        }
    }
}
fn role_label(role: &InputRole) -> &'static str {
    match role {
        InputRole::FirstFrame => "First frame",
        InputRole::LastFrame => "Last frame",
        InputRole::VisualReference => "Visual reference",
        InputRole::VideoReference => "Video reference",
        InputRole::AudioReference => "Audio reference",
        InputRole::VoiceReference => "Voice reference",
    }
}
