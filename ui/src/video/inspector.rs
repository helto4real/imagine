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

    /// The media bin beside the viewer: project media as draggable tiles, and
    /// the cast, each in its own tab.
    pub(super) fn timeline_references(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        let blocked = (!enabled).then(|| {
            self.timeline_blocked(true)
                .unwrap_or("Wait for the current timeline operation.")
        });
        chrome::region(ui, "timeline-references", rect.shrink(8.), |ui| {
            ui.horizontal(|ui| {
                let cast = &d.project.creative.references;
                for (tab, label, count) in [
                    (false, "Media", d.project.assets.len()),
                    (true, "Cast", cast.len()),
                ] {
                    let r = ui.selectable_label(
                        self.timeline.bin_cast == tab,
                        format!("{label} {count}"),
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("bin-tab-{}", label.to_lowercase()), &r);
                    if r.clicked() {
                        self.timeline.bin_cast = tab;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if helto::icon_button(ui, Icon::Close, "Hide media bin", false).clicked() {
                        self.timeline.shelf_collapsed = true;
                    }
                    let r = helto::action(ui, Icon::Person, "Cast & references…", false, blocked);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "cast-open".into(), &r);
                    if r.clicked() {
                        self.timeline_open_cast();
                    }
                    if helto::action(ui, Icon::Import, "Import media…", false, blocked).clicked()
                    {
                        self.timeline_choose_media(ui.ctx());
                    }
                });
            });
            ui.add_space(4.);
            egui::ScrollArea::vertical()
                .id_salt("project-assets")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.timeline.bin_cast {
                        self.timeline_bin_cast(ui, d);
                    } else {
                        self.timeline_bin_media(ui, d, enabled, blocked, commands);
                    }
                });
        });
    }
    fn timeline_bin_media(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        enabled: bool,
        blocked: Option<&str>,
        commands: &mut Vec<Edit>,
    ) {
        if d.project.assets.is_empty() {
            ui.weak("Import images, videos or audio, then drag them onto a Scene or Section.");
            return;
        }
        const TILE: f32 = 120.;
        let columns = ((ui.available_width() + 8.) / (TILE + 16.)).floor().max(1.) as usize;
        let revealed = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        let scene = self.timeline.scene().cloned();
        let mut kind_counts = std::collections::BTreeMap::new();
        egui::Grid::new("project-asset-grid")
            .spacing([8., 8.])
            .show(ui, |ui| {
                for (n, asset) in d.project.assets.values().enumerate() {
                    let count = kind_counts.entry(asset.kind.clone()).or_insert(0usize);
                    *count += 1;
                    let ordinal = *count;
                    let kind = match asset.kind {
                        MediaKind::Image => "Image",
                        MediaKind::Video => "Video",
                        MediaKind::Audio => "Audio",
                    };
                    // Media names are often file names; they follow Hide previews.
                    let name = if asset.name.is_empty() || !revealed {
                        format!("{kind} {ordinal}")
                    } else {
                        asset.name.clone()
                    };
                    egui::Frame::new()
                        .fill(helto::BG)
                        .corner_radius(5.)
                        .inner_margin(4.)
                        .show(ui, |ui| {
                            ui.set_width(TILE);
                            ui.vertical(|ui| {
                                // Drag the thumbnail or name onto a scene or Section to use it there.
                                let mut tile = |ui: &mut egui::Ui| {
                                    if asset.kind == MediaKind::Image
                                        && d.runtime_asset(&asset.id.0).is_some()
                                    {
                                        self.timeline_thumbnail(ui, d, asset);
                                    } else {
                                        let (rect, _) = ui.allocate_exact_size(
                                            egui::vec2(112., 80.),
                                            egui::Sense::hover(),
                                        );
                                        ui.painter().rect_filled(rect, 4., helto::FIELD);
                                        crate::icons::paint(
                                            ui.painter(),
                                            Rect::from_center_size(
                                                rect.center(),
                                                egui::vec2(22., 22.),
                                            ),
                                            match asset.kind {
                                                MediaKind::Image => Icon::Image,
                                                MediaKind::Video => Icon::Video,
                                                MediaKind::Audio => Icon::Audio,
                                            },
                                            helto::DIM,
                                        );
                                    }
                                    ui.add(egui::Label::new(&name).truncate());
                                };
                                if enabled {
                                    ui.dnd_drag_source(
                                        ui.id().with(("shelf-media", &asset.id.0)),
                                        super::tracks::ShelfMedia(asset.id.clone()),
                                        |ui| tile(ui),
                                    );
                                } else {
                                    tile(ui);
                                }
                                ui.horizontal(|ui| {
                                    self.timeline_bin_actions(
                                        ui, d, asset, ordinal, blocked, commands,
                                    );
                                });
                                if let Some(scene) = &scene {
                                    self.timeline_media_binding(
                                        ui, d, scene, asset, enabled, commands,
                                    );
                                }
                            });
                        });
                    if (n + 1) % columns == 0 {
                        ui.end_row();
                    }
                }
            });
    }
    /// A media tile's actions: audio onto the timeline, a video as the scene's
    /// clip, and a preview in the viewer.
    fn timeline_bin_actions(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        asset: &Asset,
        ordinal: usize,
        blocked: Option<&str>,
        commands: &mut Vec<Edit>,
    ) {
        if asset.kind == MediaKind::Audio {
            let response = helto::action(ui, Icon::Plus, "Add to timeline", false, blocked);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, format!("add-audio-asset-{ordinal}"), &response);
            if response.clicked() {
                commands.extend(self.timeline.draft_edits());
                commands.push(self.timeline.add_audio(&d.project, Some(asset.id.clone())));
            }
        }
        let Some(scene) = self.timeline.scene().cloned() else {
            return;
        };
        if asset.kind == MediaKind::Video {
            let r = helto::action(ui, Icon::Video, "Use as scene clip", false, blocked);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, format!("assembly-use-video-{ordinal}"), &r);
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
        let _ = ordinal;
        if helto::icon_button(ui, Icon::Play, "Preview", false).clicked()
            && let Some(id) = d.runtime_asset(&asset.id.0)
        {
            self.timeline.selected_take = None;
            self.timeline_load_media(ui.ctx(), id.to_owned());
        }
    }
    /// The project's cast, one row per reference.
    fn timeline_bin_cast(&mut self, ui: &mut egui::Ui, d: &Snapshot) {
        let cast = &d.project.creative.references;
        if cast.is_empty() {
            ui.weak("No cast yet. Add characters and references with Cast & references.");
            return;
        }
        let revealed = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        for (i, reference) in cast.iter().enumerate() {
            ui.horizontal(|ui| {
                if revealed {
                    ui.strong(&reference.name);
                    ui.weak(prompt_tools::reference_tag(&reference.alias));
                } else {
                    ui.strong(format!("Reference {}", i + 1));
                }
                ui.weak(&reference.kind);
                if !reference.enabled {
                    ui.weak("· off");
                }
            });
        }
    }
    /// Whether the docked inspector has a selection, or opened project or
    /// audio settings, to show.
    pub(super) fn timeline_inspector_has_content(&self) -> bool {
        self.timeline.project_open
            || self.timeline.project_audio
            || self.timeline.audio_primary.is_some()
            || self.timeline.scene().is_some()
    }
    /// The inspector docked under the tracks, as in Director: it edits the
    /// selection in columns. With nothing selected it is a one-line project strip.
    pub(super) fn timeline_inspector(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        let inner = rect.shrink2(egui::vec2(10., 6.));
        let single = self.timeline.project_audio || self.timeline.audio_primary.is_some();
        let scene = (!single).then(|| self.timeline.scene().cloned()).flatten();
        if scene.is_some() {
            self.timeline.project_open = false;
        }
        if self.timeline.inspector_collapsed || !self.timeline_inspector_has_content() {
            chrome::region(ui, "timeline-inspector", inner, |ui| {
                self.timeline_project_strip(ui, d)
            });
            return;
        }
        let Some(scene) = scene else {
            // Audio clips, project audio and project settings keep one column.
            chrome::region(ui, "timeline-inspector", inner, |ui| {
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
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if !self.timeline_audio_inspector(ui, d, enabled, commands) {
                            self.timeline_project_inspector(ui, d, enabled, commands);
                        }
                    });
            });
            return;
        };
        if self.timeline.draft_scene.as_ref() != Some(&scene.id) {
            self.timeline.sync_draft();
        }
        let header = Rect::from_min_size(inner.min, egui::vec2(inner.width(), 28.));
        let body = Rect::from_min_max(egui::pos2(inner.left(), header.bottom() + 6.), inner.max);
        chrome::region(ui, "timeline-inspector-header", header, |ui| {
            self.timeline_inspector_header(ui, d, &scene, enabled, commands)
        });
        let section = self
            .timeline
            .section
            .as_ref()
            .and_then(|id| scene.sections.iter().find(|s| &s.id == id))
            .cloned();
        let weights: &[f32] = if self.timeline.boundary {
            &[1., 1.]
        } else if section.is_some() {
            &[1.35, 1., 0.8]
        } else {
            &[0.85, 1.25, 1., 0.85]
        };
        for (index, column) in columns(body, weights).into_iter().enumerate() {
            chrome::region(ui, &format!("timeline-inspector-{index}"), column, |ui| {
                #[cfg(feature = "ui-preview")]
                if index == 0 {
                    let response = ui.interact(
                        ui.clip_rect(),
                        ui.id().with("qa-inspector-scroll"),
                        egui::Sense::hover(),
                    );
                    super::qa::control(ui, "inspector-scroll".into(), &response);
                }
                egui::ScrollArea::vertical()
                    .id_salt(("video-inspector-column", index))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Edits made while busy would be dropped, so the controls wait too.
                        ui.add_enabled_ui(enabled, |ui| {
                            if self.timeline.boundary {
                                match index {
                                    0 => self.timeline_assembly_boundary(ui, d, &scene, commands),
                                    _ => self.timeline_continuity(ui, d, &scene, commands),
                                }
                            } else if let Some(section) = &section {
                                match index {
                                    0 => self.timeline_section_prompt(ui, d, section, commands),
                                    1 => self.timeline_section_media(ui, d, &scene, section, commands),
                                    _ => self.timeline_section_timing(ui, d, &scene, section),
                                }
                            } else {
                                match index {
                                    0 => self.timeline_take_card(ui, d, &scene),
                                    1 => {
                                        self.timeline_scene_fields(ui, d, &scene, &["title"], commands);
                                        self.timeline_enhance_label(
                                            ui,
                                            "Description",
                                            Some(imagine_video_contract::authoring::SceneField::Description),
                                            commands,
                                        );
                                        let r = prompt_tools::textarea(
                                            ui,
                                            &mut self.timeline.draft_description,
                                            3,
                                            "Scene description",
                                            self.prefs.privacy_mode,
                                            true,
                                            prompt_tools::Source::scene(
                                                &d.project.creative,
                                                &d.project.id,
                                                &scene.id,
                                            ),
                                        );
                                        #[cfg(feature = "ui-preview")]
                                        super::qa::control(ui, "scene-description".into(), &r);
                                        if r.lost_focus() {
                                            commands.extend(self.timeline.draft_edits());
                                        }
                                        dialogue_warnings(ui, &self.timeline.draft_description);
                                        self.timeline_scene_fields(ui, d, &scene, &["summary"], commands);
                                    }
                                    2 => {
                                        self.timeline_scene_blockers_list(ui);
                                        self.timeline_scene_fields(
                                            ui,
                                            d,
                                            &scene,
                                            &["ambient", "music"],
                                            commands,
                                        );
                                        self.timeline_assembly_clip(ui, d, &scene, commands);
                                    }
                                    _ => self.timeline_scene_settings(ui, d, &scene, commands),
                                }
                            }
                        });
                    });
            });
        }
    }
    /// The selection's name, its Scene's place and the actions on it.
    fn timeline_inspector_header(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        let number = d
            .project
            .scenes
            .iter()
            .position(|s| s.id == scene.id)
            .map_or(0, |i| i + 1);
        let title = d
            .project
            .creative
            .scene_prompts
            .get(&scene.id)
            .map(|p| p.title.trim())
            .filter(|t| !t.is_empty())
            .map(str::to_owned);
        let revealed = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        let place = match (&title, revealed) {
            (Some(title), true) => format!("Scene {number} · {title}"),
            _ => format!("Scene {number}"),
        };
        ui.horizontal(|ui| {
            if self.timeline.boundary {
                ui.label(helto::section("Cut"));
                ui.weak(format!(
                    "Scene {} → Scene {number}",
                    number.saturating_sub(1)
                ));
            } else if let Some(index) = self
                .timeline
                .section
                .as_ref()
                .and_then(|id| scene.sections.iter().position(|s| &s.id == id))
            {
                ui.label(helto::section(&format!("Section {}", index + 1)));
                ui.weak(format!("of {place}"));
            } else {
                ui.label(helto::section(&place));
                ui.add_enabled_ui(enabled, |ui| {
                    let mut kind = d.project.scene_kind(&scene.id);
                    let before = kind;
                    let response = egui::ComboBox::from_id_salt(("scene-kind", &scene.id.0))
                        .selected_text(kind.label())
                        .show_ui(ui, |ui| {
                            for value in imagine_video_contract::creative::SceneKind::ALL {
                                let r = ui.selectable_value(&mut kind, value, value.label());
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(
                                    ui,
                                    format!("scene-kind-{}", value.label().to_lowercase()),
                                    &r,
                                );
                                let _ = r;
                            }
                        })
                        .response
                        .on_hover_text("Scene type");
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "scene-kind".into(), &response);
                    let _ = response;
                    if before != kind {
                        commands.push(Edit::SetSceneKind {
                            scene_id: scene.id.clone(),
                            kind,
                        });
                    }
                });
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let blocked = (!enabled).then_some("Wait for the current timeline operation.");
                if self.timeline.boundary || self.timeline.section.is_some() {
                    let r = helto::action(
                        ui,
                        Icon::Scene,
                        &format!("Open Scene {number}"),
                        false,
                        blocked,
                    );
                    if r.clicked() {
                        commands.extend(self.timeline.draft_edits());
                        self.timeline.boundary = false;
                        self.timeline.section = None;
                        self.timeline.section_selection.clear();
                    }
                    return;
                }
                if helto::action(ui, Icon::Trash, "Remove scene", false, blocked).clicked() {
                    commands.push(Edit::RemoveScene {
                        scene_id: scene.id.clone(),
                    });
                }
                if helto::action(ui, Icon::Duplicate, "Duplicate scene", false, blocked).clicked() {
                    commands.push(Edit::DuplicateScene {
                        scene_id: scene.id.clone(),
                        new_scene_id: SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple())),
                        start: time(seconds(scene.start) + seconds(scene.duration)),
                    });
                }
                if d.project
                    .scenes
                    .first()
                    .is_some_and(|first| first.id != scene.id)
                    && helto::action(ui, Icon::Settings, "Boundary settings", false, blocked)
                        .clicked()
                {
                    commands.extend(self.timeline.draft_edits());
                    self.timeline.boundary = true;
                    self.timeline.section = None;
                }
            });
        });
    }
    /// The server's own admission checks, shown before Generate is pressed.
    fn timeline_scene_blockers_list(&self, ui: &mut egui::Ui) {
        let blockers = self.timeline_scene_blockers();
        if blockers.is_empty() {
            return;
        }
        ui.label(helto::error_text("Generate would be refused:", true));
        for code in blockers {
            ui.label(helto::error_text(&client::error_message(code), true).small());
        }
        ui.add_space(4.);
    }
    /// The Scene's source fields named in `names`, each saved when it loses focus.
    fn timeline_scene_fields(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        names: &[&str],
        commands: &mut Vec<Edit>,
    ) {
        use imagine_video_contract::authoring::SceneField;
        let p = &mut self.timeline.draft_scene_prompt;
        let mut save = false;
        let mut enhance = None;
        for (name, label, value, rows) in [
            ("title", "Title", &mut p.title, 1),
            ("summary", "Summary", &mut p.summary, 2),
            ("ambient", "Ambient sound", &mut p.ambient_sound, 2),
            ("music", "Audience-only music", &mut p.non_diegetic_music, 2),
        ] {
            if !names.contains(&name) {
                continue;
            }
            let field = match name {
                "summary" => Some(SceneField::Summary),
                "ambient" => Some(SceneField::AmbientSound),
                "music" => Some(SceneField::NonDiegeticMusic),
                _ => None,
            };
            ui.horizontal(|ui| {
                ui.label(label);
                if let Some(field) = field {
                    // This loop borrows the source draft; open the enhancement after it.
                    let r =
                        helto::icon_button(ui, Icon::Enhance, &format!("Enhance {label}"), false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("enhance-field-{name}"), &r);
                    if r.clicked() {
                        enhance = Some(field);
                    }
                }
            });
            let r = prompt_tools::textarea(
                ui,
                value,
                rows,
                label,
                self.prefs.privacy_mode,
                // A one-line field stays one line.
                rows > 1,
                prompt_tools::Source::scene(&d.project.creative, &d.project.id, &scene.id),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, format!("scene-source-{name}"), &r);
            save |= r.lost_focus();
        }
        if save {
            commands.extend(self.timeline.draft_edits());
        }
        if let Some(field) = enhance {
            self.timeline_enhance_action(ui.ctx(), PostAction::EnhanceField(field), commands);
        }
    }
    /// The Section prompt, first and always open, with its enhancement.
    fn timeline_section_prompt(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        section: &Section,
        commands: &mut Vec<Edit>,
    ) {
        self.timeline_enhance_label(ui, "Prompt", None, commands);
        let r = prompt_tools::textarea(
            ui,
            &mut self.timeline.draft_section,
            4,
            "Section prompt",
            self.prefs.privacy_mode,
            true,
            prompt_tools::Source::section(&d.project.creative, &d.project.id, &section.id),
        );
        #[cfg(feature = "ui-preview")]
        super::qa::control(ui, "section-prompt".into(), &r);
        if r.lost_focus() {
            commands.extend(self.timeline.draft_edits());
        }
        dialogue_warnings(ui, &self.timeline.draft_section);
    }
    /// The Section's image anchors, with images and videos one click away.
    fn timeline_section_media(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        section: &Section,
        commands: &mut Vec<Edit>,
    ) {
        ui.horizontal(|ui| {
            ui.label(helto::section("Media anchors"));
            for (kind, icon, label, key) in [
                (
                    MediaKind::Image,
                    Icon::Image,
                    "Attach images…",
                    "attach-images",
                ),
                (
                    MediaKind::Video,
                    Icon::Video,
                    "Attach videos…",
                    "attach-videos",
                ),
            ] {
                let r = helto::icon_button(ui, icon, label, false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, key.into(), &r);
                let _ = key;
                if r.clicked() {
                    self.timeline_pick_media(media_picker::Target::Generation {
                        scene: scene.id.clone(),
                        section: Some(section.id.clone()),
                        kind,
                    });
                }
            }
        });
        self.timeline_image_anchors(ui, d, &section.id, commands);
    }
    /// Where the Section sits and which model renders it.
    fn timeline_section_timing(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        section: &Section,
    ) {
        let start = seconds(scene.start) + seconds(section.start);
        let length = seconds(section.duration);
        let fps = seconds(d.project.delivery.fps);
        ui.label(helto::section("Timing"));
        egui::Grid::new(("section-timing", &section.id.0))
            .num_columns(2)
            .spacing([10., 4.])
            .show(ui, |ui| {
                ui.weak("Start");
                ui.label(format!("{start:.2} s"));
                ui.end_row();
                ui.weak("Length");
                ui.label(format!("{length:.2} s"));
                ui.end_row();
                ui.weak("Frames");
                ui.label(format!("{} at {fps} fps", (length * fps).round()));
                ui.end_row();
            });
        ui.add_space(6.);
        ui.label(helto::section("Model"));
        let model = d.project.effective_model(scene);
        ui.weak(format!("{} · {}", model.model, model.variant));
        self.timeline_scene_blockers_list(ui);
    }
    /// The Scene's length, inputs and model.
    fn timeline_scene_settings(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        commands: &mut Vec<Edit>,
    ) {
        ui.horizontal(|ui| {
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
                    scene,
                    duration,
                    self.timeline.mode,
                ));
            }
            for (kind, icon, label, key) in [
                (
                    MediaKind::Image,
                    Icon::Image,
                    "Attach images…",
                    "attach-images",
                ),
                (
                    MediaKind::Video,
                    Icon::Video,
                    "Attach videos…",
                    "attach-videos",
                ),
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
        if let Some(note) = effective_timing(&d.project, scene) {
            ui.weak(note);
        }
        ui.add_space(4.);
        ui.label(helto::section("Model"));
        self.timeline_scene_generation(ui, d, scene, commands);
        let model = d.project.effective_model(scene);
        ui.weak(format!("{} · {}", model.model, model.variant));
        if model.adapter == "h3"
            && model.variant == "reference"
            && let Some(slot) = d.project.effective_settings(scene)
        {
            let mut use_audio = slot.values["audio_reference_mode"].as_str() != Some("Disabled");
            let response = ui.checkbox(&mut use_audio, "Use timeline audio references");
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "h3-timeline-audio-references".into(), &response);
            if response.changed() {
                let mut settings = slot.clone();
                settings.values["audio_reference_mode"] = serde_json::json!(if use_audio {
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
    }
    /// The project in one line, with its settings one click away.
    fn timeline_project_strip(&mut self, ui: &mut egui::Ui, d: &Snapshot) {
        let project = &d.project;
        ui.horizontal_centered(|ui| {
            ui.label(helto::section("Project"));
            ui.weak(format!(
                "{} scene{} · {:.1} s · {} × {} · {} fps · {}",
                project.scenes.len(),
                if project.scenes.len() == 1 { "" } else { "s" },
                navigation::duration(project),
                project.delivery.width,
                project.delivery.height,
                seconds(project.delivery.fps),
                project.default_model.variant,
            ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let r = helto::icon_button(ui, Icon::Settings, "Project settings", false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-settings-open".into(), &r);
                if r.clicked() {
                    self.timeline.project_open = true;
                    self.timeline.inspector_collapsed = false;
                }
            });
        });
    }
}

/// `rect` split into columns of the given relative widths, 14 points apart.
fn columns(rect: Rect, weights: &[f32]) -> Vec<Rect> {
    let gap = 14.;
    let total: f32 = weights.iter().sum();
    let width = rect.width() - gap * (weights.len() as f32 - 1.);
    let mut left = rect.left();
    weights
        .iter()
        .map(|w| {
            let column = Rect::from_min_max(
                egui::pos2(left, rect.top()),
                egui::pos2(left + width * w / total, rect.bottom()),
            );
            left = column.right() + gap;
            column
        })
        .collect()
}

/// Director's dialogue warnings for a prompt field. Only the advice is shown;
/// the quoted words stay in the field, which follows Hide previews.
fn dialogue_warnings(ui: &mut egui::Ui, text: &str) {
    let mut shown = std::collections::BTreeSet::new();
    for issue in imagine_video_contract::dialogue::issues(text) {
        if shown.insert(issue.code) {
            ui.label(
                egui::RichText::new(issue.message())
                    .small()
                    .color(helto::ACCENT),
            );
        }
    }
}

/// What H3 will actually render for the scene: the project canvas and the
/// duration after H3's frame alignment, as compiled by the server.
fn effective_timing(project: &VideoProject, scene: &Scene) -> Option<String> {
    if project.effective_model(scene).adapter != "h3" {
        return None;
    }
    let shape = crate::h3_protocol::Selection {
        seconds: seconds(scene.duration),
        ..Default::default()
    }
    .resolve()
    .ok()?;
    Some(format!(
        "Renders {} × {} · {} frames · {:.3} s",
        project.delivery.width,
        project.delivery.height,
        shape.frames,
        shape.seconds()
    ))
}

impl ImagineApp {
    /// The inspector with nothing selected: the project's canvas, length and
    /// model, and its project-wide settings.
    fn timeline_project_inspector(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        let project = &d.project;
        ui.horizontal(|ui| {
            ui.label(helto::section("Project"));
            if self.timeline.project_open {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if helto::icon_button(ui, Icon::Close, "Close project settings", false)
                        .clicked()
                    {
                        self.timeline.project_open = false;
                    }
                });
            }
        });
        ui.separator();
        let fps = seconds(project.delivery.fps);
        ui.label(format!(
            "{} × {} · {fps} fps",
            project.delivery.width, project.delivery.height
        ));
        ui.label(format!(
            "{:.2} s · {} scene{}",
            navigation::duration(project),
            project.scenes.len(),
            if project.scenes.len() == 1 { "" } else { "s" }
        ));
        let model = &project.default_model;
        ui.weak(format!("{} · {}", model.variant, model.profile));
        ui.add_space(8.);
        let blocked = (!enabled).then_some("Wait for the current timeline operation.");
        ui.horizontal_wrapped(|ui| {
            if helto::action(ui, Icon::Fit, "Timeline settings…", false, blocked).clicked() {
                self.timeline.project_timing.open(project);
            }
            if helto::action(ui, Icon::Settings, "Generation settings…", false, blocked).clicked()
            {
                self.timeline_open_generation();
            }
            if helto::action(ui, Icon::Edit, "Global prompt…", false, blocked).clicked() {
                self.timeline_open_global_prompt();
            }
            if helto::action(ui, Icon::Audio, "Project audio…", false, blocked).clicked() {
                self.timeline_open_project_audio(d, commands);
            }
        });
        if project.scenes.is_empty() {
            ui.add_space(8.);
            ui.weak("Add a scene to start.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dialogue_warnings_show_advice_once_without_the_quoted_words() {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(Default::default(), |ui| {
            dialogue_warnings(ui, "@own says Hi and \"Own secret\" and \"Own other\"");
        });
        output.textures_delta.clear();
        let texts: Vec<String> = output
            .shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some(t.galley.text().to_owned()),
                _ => None,
            })
            .collect();
        assert_eq!(texts.len(), 2, "{texts:?}");
        assert!(texts.iter().all(|t| !t.contains("Own secret")));
    }
    #[test]
    fn effective_timing_shows_the_h3_canvas_and_aligned_duration() {
        let mut project = media_picker::tests::project();
        project.default_model.adapter = "h3".into();
        let scene = project.scenes[0].clone();
        let note = effective_timing(&project, &scene).unwrap();
        let shape = crate::h3_protocol::Selection {
            seconds: seconds(scene.duration),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        assert_eq!(
            note,
            format!(
                "Renders 64 × 64 · {} frames · {:.3} s",
                shape.frames,
                shape.seconds()
            )
        );
        project.default_model.adapter = "synthetic".into();
        assert!(effective_timing(&project, &scene).is_none());
    }
}
