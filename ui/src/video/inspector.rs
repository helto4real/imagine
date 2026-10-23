//! Contextual media/reference and scene/Section inspector.
use super::editing::{resize_scene, seconds, time};
use super::*;
use crate::app::workspace as chrome;
use crate::icons::Icon;
use egui::Rect;
impl ImagineApp {
    fn timeline_enhance_action(
        &mut self,
        ctx: &egui::Context,
        action: PostAction,
        commands: &mut Vec<Edit>,
    ) {
        commands.extend(self.timeline.draft_edits());
        if commands.is_empty() {
            self.timeline_action(ctx, action);
        } else {
            self.timeline.after_edit = Some(action);
        }
    }
    fn timeline_enhance_label(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        field: Option<imagine_video_contract::authoring::SceneField>,
        commands: &mut Vec<Edit>,
    ) {
        let r = ui
            .horizontal(|ui| {
                ui.label(label);
                helto::icon_button(
                    ui,
                    crate::icons::Icon::Enhance,
                    &format!("Enhance {label}"),
                    false,
                )
            })
            .inner;
        #[cfg(feature = "ui-preview")]
        super::qa::control(
            ui,
            format!(
                "enhance-field-{}",
                if field.is_some() {
                    "description"
                } else {
                    "section"
                }
            ),
            &r,
        );
        if r.clicked() {
            let action = field
                .map(PostAction::EnhanceField)
                .unwrap_or(PostAction::Authoring(
                    imagine_video_contract::authoring::Operation::EnhanceSection,
                ));
            self.timeline_enhance_action(ui.ctx(), action, commands);
        }
    }

    pub(super) fn timeline_references(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        chrome::region(ui, "timeline-references", rect.shrink(10.), |ui| {
            ui.horizontal(|ui| {
                ui.strong("References / Media");
            });
            ui.separator();
            let blocked = (!enabled).then(|| {
                self.timeline_blocked(true)
                    .unwrap_or("Wait for the current timeline operation.")
            });
            ui.horizontal(|ui| {
                let r = helto::action(ui, Icon::Person, "Cast & references…", false, blocked);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "cast-open".into(), &r);
                if r.clicked() {
                    self.timeline_open_cast();
                }
                if helto::action(ui, Icon::Import, "Import media…", false, blocked).clicked() {
                    self.timeline_choose_media(ui.ctx());
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("project-assets")
                .show(ui, |ui| {
                    let mut kind_counts = std::collections::BTreeMap::new();
                    for asset in d.project.assets.values() {
                        let count = kind_counts.entry(asset.kind.clone()).or_insert(0usize);
                        *count += 1;
                        let ordinal = *count;
                        ui.group(|ui| {
                            ui.label(format!(
                                "{} {}",
                                match asset.kind {
                                    MediaKind::Image => "Image",
                                    MediaKind::Video => "Video",
                                    MediaKind::Audio => "Audio",
                                },
                                ordinal
                            ));
                            ui.horizontal(|ui| {
                                if asset.kind == MediaKind::Audio {
                                    let response = helto::action(
                                        ui,
                                        Icon::Plus,
                                        "Add to timeline",
                                        false,
                                        blocked,
                                    );
                                    #[cfg(feature = "ui-preview")]
                                    super::qa::control(
                                        ui,
                                        format!("add-audio-asset-{ordinal}"),
                                        &response,
                                    );
                                    if response.clicked() {
                                        commands.extend(self.timeline.draft_edits());
                                        commands.push(
                                            self.timeline
                                                .add_audio(&d.project, Some(asset.id.clone())),
                                        );
                                    }
                                }
                                if let Some(scene) = self.timeline.scene().cloned() {
                                    if asset.kind == MediaKind::Video {
                                        let r = helto::action(
                                            ui,
                                            Icon::Video,
                                            "Use as scene clip",
                                            false,
                                            blocked,
                                        );
                                        #[cfg(feature = "ui-preview")]
                                        super::qa::control(
                                            ui,
                                            format!("assembly-use-video-{ordinal}"),
                                            &r,
                                        );
                                        if r.clicked() {
                                            commands.extend(self.timeline.draft_edits());
                                            let action = PostAction::UseClip {
                                                scene: scene.id.clone(),
                                                asset: asset.id.clone(),
                                            };
                                            if commands.is_empty() {
                                                self.timeline_action(ui.ctx(), action);
                                            } else {
                                                self.timeline.after_edit = Some(action);
                                            }
                                        }
                                    }
                                    if helto::icon_button(ui, Icon::Play, "Preview", false)
                                        .clicked()
                                        && let Some(id) = d.runtime_asset(&asset.id.0)
                                    {
                                        self.timeline.selected_take = None;
                                        self.timeline_load_media(ui.ctx(), id.to_owned());
                                    }
                                }
                            });
                            if let Some(scene) = self.timeline.scene().cloned() {
                                self.timeline_media_binding(
                                    ui, d, &scene, asset, enabled, commands,
                                );
                            }
                        });
                    }
                });
        });
    }
    pub(super) fn timeline_inspector(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        chrome::region(ui, "timeline-inspector", rect.shrink(10.), |ui| {
            #[cfg(feature = "ui-preview")]
            {
                let response = ui.interact(
                    ui.clip_rect(),
                    ui.id().with("qa-inspector-scroll"),
                    egui::Sense::hover(),
                );
                super::qa::control(ui, "inspector-scroll".into(), &response);
            }
            egui::ScrollArea::vertical()
                .id_salt("video-inspector-scroll")
                .show(ui, |ui| {
                    if self.timeline_audio_inspector(ui, d, enabled, commands) {
                        return;
                    }
                    let Some(scene) = self.timeline.scene().cloned() else {
                        ui.strong("Project");
                        return;
                    };
                    if self.timeline.draft_scene.as_ref() != Some(&scene.id) {
                        self.timeline.sync_draft();
                    }
                    if self.timeline.boundary {
                        ui.strong("Boundary");
                        ui.separator();
                        // Edits made while busy would be dropped, so the controls wait too.
                        ui.add_enabled_ui(enabled, |ui| {
                            self.timeline_assembly_boundary(ui, d, &scene, commands);
                            self.timeline_continuity(ui, d, &scene, commands)
                        });
                        return;
                    }
                    ui.strong(if self.timeline.section.is_some() {
                        "Section"
                    } else {
                        "Scene"
                    });
                    ui.separator();
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.label("Scene type");
                        let mut kind=d.project.scene_kind(&scene.id);
                        let before=kind;
                        let response=egui::ComboBox::from_id_salt(("scene-kind",&scene.id.0)).selected_text(kind.label()).show_ui(ui,|ui|{
                            for value in imagine_video_contract::creative::SceneKind::ALL {
                                let r=ui.selectable_value(&mut kind,value,value.label());
                                #[cfg(feature="ui-preview")]super::qa::control(ui,format!("scene-kind-{}",value.label().to_lowercase()),&r);
                                let _=r;
                            }
                        }).response;
                        #[cfg(feature="ui-preview")]super::qa::control(ui,"scene-kind".into(),&response);
                        let _=response;
                        if before!=kind {commands.push(Edit::SetSceneKind{scene_id:scene.id.clone(),kind});}
                        ui.horizontal_wrapped(|ui| {
                            for (kind, icon, label, key) in [
                                (MediaKind::Image, Icon::Image, "Attach images…", "attach-images"),
                                (MediaKind::Video, Icon::Video, "Attach videos…", "attach-videos"),
                            ] {
                                let r = helto::icon_button(ui, icon, label, false);
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(ui, key.into(), &r);
                                let _ = key;
                                if r.clicked() {
                                    self.timeline_pick_media(media_picker::Target::Generation {
                                        scene: scene.id.clone(),
                                        section: self.timeline.section.clone(),
                                        kind,
                                    });
                                }
                            }
                        });
                        ui.label("Duration");
                        let mut duration = seconds(scene.duration);
                        if ui
                            .add(
                                egui::DragValue::new(&mut duration)
                                    .range(0.25..=3600.)
                                    .speed(0.25)
                                    .suffix(" s"),
                            )
                            .changed()
                        {
                            commands.extend(resize_scene(
                                &d.project,
                                &scene,
                                duration,
                                self.timeline.mode,
                            ));
                        }
                        self.timeline_enhance_label(ui,"Description",Some(imagine_video_contract::authoring::SceneField::Description),commands);
                        self.timeline_assembly_clip(ui, d, &scene, commands);
                        let r = prompt_tools::textarea(
                            ui,
                            &mut self.timeline.draft_description,
                            3,
                            "Scene description",
                            self.prefs.privacy_mode,
                            true,
                            prompt_tools::Source::scene(&d.project.creative,&d.project.id,&scene.id),
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, "scene-description".into(), &r);
                        if r.lost_focus() {
                            commands.extend(self.timeline.draft_edits());
                        }
                        let source_header = ui.collapsing("Scene prompt fields", |ui| {
                            let p = &mut self.timeline.draft_scene_prompt;
                            let mut save = false;
                            let mut enhance_field=None;
                            for (name, label, value) in [
                                ("title", "Title", &mut p.title),
                                ("summary", "Summary", &mut p.summary),
                                ("ambient", "Ambient sound", &mut p.ambient_sound),
                                ("music", "Audience-only music", &mut p.non_diegetic_music),
                            ] {
                                if name=="title" {ui.label(label);}else{
                                    let field=match name{"summary"=>imagine_video_contract::authoring::SceneField::Summary,"ambient"=>imagine_video_contract::authoring::SceneField::AmbientSound,_=>imagine_video_contract::authoring::SceneField::NonDiegeticMusic};
                                    // This loop borrows the source draft; defer opening until it is released.
                                    let r=ui.horizontal(|ui|{ui.label(label);helto::icon_button(ui,crate::icons::Icon::Enhance,&format!("Enhance {label}"),false)}).inner;
                                    #[cfg(feature="ui-preview")]super::qa::control(ui,format!("enhance-field-{name}"),&r);
                                    if r.clicked(){enhance_field=Some(field);}
                                }
                                let r = prompt_tools::textarea(
                                    ui,
                                    value,
                                    2,
                                    label,
                                    self.prefs.privacy_mode,
                                    true,
                                    prompt_tools::Source::scene(&d.project.creative,&d.project.id,&scene.id),
                                );
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(ui, format!("scene-source-{name}"), &r);
                                let _ = name;
                                save |= r.lost_focus();
                            }
                            if save {commands.extend(self.timeline.draft_edits());}
                            if let Some(field)=enhance_field{self.timeline_enhance_action(ui.ctx(),PostAction::EnhanceField(field),commands);}

                        });
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(
                            ui,
                            "scene-source-fields".into(),
                            &source_header.header_response,
                        );
                        let _ = source_header;
                        if let Some(section) = self
                            .timeline
                            .section
                            .as_ref()
                            .and_then(|id| scene.sections.iter().find(|s| &s.id == id))
                        {
                            self.timeline_image_anchors(ui, d, &section.id, commands);
                            self.timeline_enhance_label(ui,"Section prompt",None,commands);
                            let r = prompt_tools::textarea(
                                ui,
                                &mut self.timeline.draft_section,
                                3,
                                "Section prompt",
                                self.prefs.privacy_mode,
                                true,
                                prompt_tools::Source::section(&d.project.creative,&d.project.id,&section.id),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "section-prompt".into(), &r);
                            if r.lost_focus() {
                                commands.extend(self.timeline.draft_edits());
                            }
                            ui.weak(format!(
                                "{:.3}–{:.3} s",
                                seconds(section.start),
                                seconds(section.start) + seconds(section.duration)
                            ));
                        }
                        ui.separator();
                        ui.label("Model");
                        self.timeline_scene_generation(ui,d,&scene,commands);
                        ui.weak(format!(
                            "{} · {}",
                            d.project.effective_model(&scene).model,
                            d.project.effective_model(&scene).variant
                        ));
                        let model = d.project.effective_model(&scene);
                        if model.adapter == "h3"
                            && model.variant == "reference"
                            && let Some(slot) =
                                d.project.effective_settings(&scene)
                        {
                            let mut use_audio =
                                slot.values["audio_reference_mode"].as_str() != Some("Disabled");
                            let response =
                                ui.checkbox(&mut use_audio, "Use timeline audio references");
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(
                                ui,
                                "h3-timeline-audio-references".into(),
                                &response,
                            );
                            if response.changed() {
                                let mut settings = slot.clone();
                                settings.values["audio_reference_mode"] =
                                    serde_json::json!(if use_audio {
                                        "Timeline Audio References"
                                    } else {
                                        "Disabled"
                                    });
                                commands.push(Edit::SetModelSettings {
                                    scene_id: scene.id.clone(),
                                    settings,
                                });
                            }
                        }
                        ui.horizontal(|ui| {
                        if d.project
                            .scenes
                            .first()
                            .is_some_and(|first| first.id != scene.id)
                            && helto::icon_button(ui, Icon::Settings, "Boundary settings", false)
                                .clicked()
                        {
                            commands.extend(self.timeline.draft_edits());
                            self.timeline.boundary = true;
                            self.timeline.section = None;
                        }
                        if helto::icon_button(ui, Icon::Duplicate, "Duplicate scene", false).clicked() {
                            commands.push(Edit::DuplicateScene {
                                scene_id: scene.id.clone(),
                                new_scene_id: SceneId(format!(
                                    "scene_{}",
                                    uuid::Uuid::new_v4().simple()
                                )),
                                start: time(seconds(scene.start) + seconds(scene.duration)),
                            });
                        }
                        if helto::icon_button(ui, Icon::Trash, "Remove scene", false).clicked() {
                            commands.push(Edit::RemoveScene {
                                scene_id: scene.id.clone(),
                            });
                        }
                        });
                    });
                });
        });
    }
}
