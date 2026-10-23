//! Timeline geometry, selection and native pointer gestures.
use super::editing::{Mode, Snap, resize_scene, seconds};
use super::*;
use crate::app::workspace as chrome;
use egui::{Rect, Sense, Stroke};
#[derive(Clone)]
pub(super) struct Drag {
    id: egui::Id,
    origin: egui::Pos2,
    before: VideoProject,
}
fn boundary_label(mode: &imagine_video_contract::assembly::BoundaryMode) -> &'static str {
    use imagine_video_contract::assembly::BoundaryMode;
    match mode {
        BoundaryMode::HardCut => "Hard cut",
        BoundaryMode::ContinuousShot => "Continuous shot",
        BoundaryMode::BlendSeam => "Blend seam",
        BoundaryMode::Transition => "Generated transition",
    }
}
fn boundary_glyph(mode: &imagine_video_contract::assembly::BoundaryMode) -> &'static str {
    use imagine_video_contract::assembly::BoundaryMode;
    match mode {
        BoundaryMode::HardCut => "|",
        BoundaryMode::ContinuousShot => "≈",
        BoundaryMode::BlendSeam => "×",
        BoundaryMode::Transition => "»",
    }
}

pub(super) fn gesture(
    timeline: &mut Timeline,
    response: &egui::Response,
    project: &VideoProject,
) -> Option<(VideoProject, f32, bool)> {
    if response.drag_started() {
        let origin = response
            .ctx
            .input(|i| i.pointer.press_origin())
            .or_else(|| response.interact_pointer_pos())?;
        timeline.drag = Some(Drag {
            id: response.id,
            origin,
            before: project.clone(),
        });
    }
    let drag = timeline.drag.as_ref()?;
    if drag.id != response.id || !(response.dragged() || response.drag_stopped()) {
        return None;
    }
    let x = response.interact_pointer_pos()?.x - drag.origin.x;
    let before = drag.before.clone();
    let done = response.drag_stopped();
    if done {
        timeline.drag = None;
    }
    Some((before, x, done))
}
pub(super) fn move_gesture(
    timeline: &mut Timeline,
    response: &egui::Response,
    project: &VideoProject,
) -> Option<(VideoProject, f32, bool)> {
    if response.drag_started()
        && response
            .ctx
            .input(|i| i.modifiers.shift || i.modifiers.ctrl || i.modifiers.command)
    {
        timeline.drag = None;
        return None;
    }
    gesture(timeline, response, project)
}
impl ImagineApp {
    pub(super) fn timeline_tracks(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        project: &VideoProject,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        chrome::region(ui, "video-timeline-tracks", rect.shrink(8.), |ui| {
            let whole = navigation::duration(project);
            let width = ui.available_width().max(1.);
            self.timeline
                .navigation
                .prepare(whole, width, &mut self.timeline.zoom);
            ui.horizontal_wrapped(|ui| {
                ui.strong("Timeline");
                ui.weak(format!(
                    "{:.0} s · {} fps",
                    whole,
                    seconds(project.delivery.fps)
                ));
                let mode = egui::ComboBox::from_id_salt("timeline-display")
                    .selected_text(self.timeline.navigation.display.label())
                    .show_ui(ui, |ui| {
                        for value in [
                            navigation::Display::Sections,
                            navigation::Display::Media,
                            navigation::Display::Prompts,
                        ] {
                            let r = ui.selectable_value(
                                &mut self.timeline.navigation.display,
                                value,
                                value.label(),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(
                                ui,
                                format!("timeline-display-{}", value.label().to_lowercase()),
                                &r,
                            );
                            let _ = r;
                        }
                    });
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-display".into(), &mode.response);
                let _ = mode;
                ui.separator();
                ui.label("Snap");
                egui::ComboBox::from_id_salt("video-snap")
                    .width(70.)
                    .selected_text(match self.timeline.snap {
                        Snap::None => "None",
                        Snap::Seconds => "Seconds",
                        Snap::Frames => "Frames",
                    })
                    .show_ui(ui, |ui| {
                        for (value, label) in [
                            (Snap::None, "None"),
                            (Snap::Seconds, "Seconds"),
                            (Snap::Frames, "Frames"),
                        ] {
                            ui.selectable_value(&mut self.timeline.snap, value, label);
                        }
                    });
                let edit_mode = egui::ComboBox::from_id_salt("video-edit-mode")
                    .width(100.)
                    .selected_text(if self.timeline.mode == Mode::Ripple {
                        "Ripple Edit"
                    } else {
                        "Trim Neighbor"
                    })
                    .show_ui(ui, |ui| {
                        let trim = ui.selectable_value(
                            &mut self.timeline.mode,
                            Mode::TrimNeighbor,
                            "Trim Neighbor",
                        );
                        let ripple = ui.selectable_value(
                            &mut self.timeline.mode,
                            Mode::Ripple,
                            "Ripple Edit",
                        );
                        #[cfg(feature = "ui-preview")]
                        {
                            super::qa::control(ui, "timeline-edit-trim".into(), &trim);
                            super::qa::control(ui, "timeline-edit-ripple".into(), &ripple);
                        }
                        let _ = (trim, ripple);
                    });
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-edit-mode".into(), &edit_mode.response);
                let _ = edit_mode;
                let zoom = ui.add_sized(
                    [150., 24.],
                    egui::Slider::new(&mut self.timeline.zoom, 0.1..=5000.0)
                        .show_value(false)
                        .text("Zoom"),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-zoom".into(), &zoom);
                if zoom.changed() {
                    let (start, _) = self.timeline.navigation.view.unwrap_or((0., whole));
                    let end = start + f64::from(width / self.timeline.zoom);
                    self.timeline
                        .navigation
                        .set(start, end, whole, width, &mut self.timeline.zoom);
                }
                let play =
                    helto::icon_button(ui, crate::icons::Icon::Play, "Play from playhead", false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-play-draft".into(), &play);
                if play.clicked() {
                    self.timeline.playback.request = Some(true);
                }
                let fit = helto::icon_button(ui, crate::icons::Icon::Fit, "Zoom to Fit", false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-fit".into(), &fit);
                if fit.clicked() {
                    self.timeline
                        .navigation
                        .set(0., whole, whole, width, &mut self.timeline.zoom);
                }
                let shelf = helto::icon_button(
                    ui,
                    crate::icons::Icon::PanelLeft,
                    "Toggle media shelf",
                    !self.timeline.shelf_collapsed,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-toggle-shelf".into(), &shelf);
                if shelf.clicked() {
                    self.timeline.shelf_collapsed = !self.timeline.shelf_collapsed;
                }
                let inspector = helto::icon_button(
                    ui,
                    crate::icons::Icon::PanelRight,
                    "Toggle inspector",
                    !self.timeline.inspector_collapsed,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-toggle-inspector".into(), &inspector);
                if inspector.clicked() {
                    self.timeline.inspector_collapsed = !self.timeline.inspector_collapsed;
                }
                let duplicate = ui
                    .add_enabled_ui(enabled, |ui| {
                        helto::icon_button(
                            ui,
                            crate::icons::Icon::Duplicate,
                            "Duplicate selection",
                            false,
                        )
                    })
                    .inner;
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-duplicate".into(), &duplicate);
                if duplicate.clicked() {
                    commands.extend(self.timeline.duplicate_selection(project));
                }
                let delete = ui
                    .add_enabled_ui(enabled, |ui| {
                        helto::icon_button(ui, crate::icons::Icon::Trash, "Delete selection", false)
                    })
                    .inner;
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-delete".into(), &delete);
                if delete.clicked() {
                    commands.extend(self.timeline.draft_edits());
                    commands.extend(self.timeline.delete_selection());
                }
                if group_commands::overflow(project) {
                    for (all, label, key) in [
                        (false, "Fit Last Section", "timeline-fit-last"),
                        (true, "Fit All Sections Evenly", "timeline-fit-all"),
                    ] {
                        let r = ui
                            .add_enabled_ui(
                                enabled && (all || group_commands::can_fit_last(project)),
                                |ui| helto::icon_button(ui, crate::icons::Icon::Fit, label, false),
                            )
                            .inner;
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, key.into(), &r);
                        let _ = key;
                        if r.clicked() {
                            match group_commands::prepared(project, self.timeline.draft_edits()) {
                                Ok(current) => {
                                    commands.extend(editing::difference(project, &current));
                                    commands.extend(group_commands::fit(&current, all));
                                }
                                Err(e) => {
                                    self.timeline.error =
                                        Some(format!("Could not fit Sections: {}", e.code))
                                }
                            }
                        }
                    }
                }
            });
            self.timeline
                .navigation
                .range_bar(ui, whole, &mut self.timeline.zoom);
            let end = project
                .scenes
                .iter()
                .map(|s| seconds(s.start) + seconds(s.duration))
                .chain(
                    project
                        .audio_tracks
                        .iter()
                        .flat_map(|t| &t.clips)
                        .map(|c| seconds(c.start) + seconds(c.duration)),
                )
                .fold(1., f64::max)
                .ceil()
                .max(whole);
            let audio_lanes = project
                .audio_tracks
                .iter()
                .map(|t| t.clips.iter().map(|c| c.lane + 1).max().unwrap_or(0))
                .sum::<u32>()
                .max(1);
            let mut area = egui::ScrollArea::both().id_salt("video-track-pan");
            if let Some(offset) = self.timeline.navigation.scroll_to.take() {
                area = area.horizontal_scroll_offset(offset);
            }
            let scroll = area.show(ui, |ui| {
                let size = egui::vec2(
                    (end as f32 * self.timeline.zoom).max(ui.available_width()),
                    90. + audio_lanes as f32 * 28.,
                );
                let (canvas, _) = ui.allocate_exact_size(size, Sense::hover());
                let ruler = ui.interact(
                    Rect::from_min_max(canvas.min, egui::pos2(canvas.right(), canvas.top() + 18.)),
                    ui.id().with("timeline-ruler"),
                    Sense::click(),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-ruler".into(), &ruler);
                if ruler.clicked()
                    && let Some(pointer) = ruler.interact_pointer_pos()
                {
                    self.timeline.playhead =
                        f64::from((pointer.x - canvas.left()) / self.timeline.zoom)
                            .clamp(0., whole);
                    self.timeline.playback.request = Some(false);
                }
                ui.painter().vline(
                    canvas.left() + self.timeline.playhead as f32 * self.timeline.zoom,
                    canvas.top()..=canvas.bottom(),
                    Stroke::new(1., helto::FOCUS),
                );
                let tick = (48. / self.timeline.zoom).ceil().max(1.) as usize;
                let visible = ui.clip_rect().intersect(canvas);
                let first =
                    ((visible.left() - canvas.left()) / self.timeline.zoom).max(0.) as usize / tick
                        * tick;
                let last = (((visible.right() - canvas.left()) / self.timeline.zoom)
                    .ceil()
                    .max(0.) as usize)
                    .min(end.ceil() as usize);
                for second in (first..=last).step_by(tick) {
                    let x = canvas.left() + second as f32 * self.timeline.zoom;
                    ui.painter().vline(
                        x,
                        canvas.top()..=canvas.bottom(),
                        Stroke::new(1., helto::FIELD),
                    );
                    ui.painter().text(
                        egui::pos2(x + 3., canvas.top()),
                        egui::Align2::LEFT_TOP,
                        format!("{second:02}s"),
                        egui::FontId::proportional(10.),
                        helto::DIM,
                    );
                }
                // Scene titles are authored text and follow Hide previews.
                let titles_revealed = helto::revealed(ui, canvas, self.prefs.privacy_mode);
                let states: Vec<_> = project
                    .scenes
                    .iter()
                    .map(|scene| self.timeline.scene_state(project, scene))
                    .collect();
                for (index, scene) in project.scenes.iter().enumerate() {
                    let state = states[index];
                    let x = canvas.left() + seconds(scene.start) as f32 * self.timeline.zoom;
                    let w = seconds(scene.duration) as f32 * self.timeline.zoom;
                    let bar = Rect::from_min_size(
                        egui::pos2(x, canvas.top() + 19.),
                        egui::vec2(w.max(12.), 28.),
                    );
                    let active = self
                        .timeline
                        .selection_items(project)
                        .contains(&selection::Item::Scene(scene.id.clone()));
                    let response = ui.interact(
                        bar,
                        ui.id().with((&scene.id.0, "scene")),
                        if enabled && scene.sections.is_empty() {
                            Sense::click_and_drag()
                        } else if enabled {
                            Sense::click()
                        } else {
                            Sense::hover()
                        },
                    );
                    ui.painter().rect_filled(
                        bar,
                        3,
                        if active {
                            helto::ACCENT_TOP
                        } else {
                            helto::FIELD
                        },
                    );
                    if let Some(color) = state.color() {
                        ui.painter().rect_filled(
                            Rect::from_min_max(egui::pos2(bar.left(), bar.bottom() - 3.), bar.max),
                            egui::CornerRadius {
                                sw: 3,
                                se: 3,
                                ..Default::default()
                            },
                            color,
                        );
                    }
                    let title = scene.description.lines().next().unwrap_or("").trim();
                    let response = response.on_hover_text(format!(
                        "Scene {} · {:.2} s · {}",
                        index + 1,
                        seconds(scene.duration),
                        state.label()
                    ));
                    ui.painter()
                        .with_clip_rect(bar.intersect(ui.clip_rect()))
                        .text(
                            bar.left_center() + egui::vec2(8., 0.),
                            egui::Align2::LEFT_CENTER,
                            if title.is_empty() || !titles_revealed {
                                format!("{:02} · {:.2} s", index + 1, seconds(scene.duration))
                            } else {
                                format!("{:02} · {title}", index + 1)
                            },
                            egui::FontId::proportional(12.),
                            if active {
                                helto::ACCENT_TEXT
                            } else {
                                helto::TEXT
                            },
                        );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("scene-{}", scene.id.0), &response);
                    if response.clicked() || response.secondary_clicked() {
                        response.request_focus();
                        self.timeline.keyboard_focus = Some(response.id);
                        self.timeline.boundary = false;
                        commands.extend(self.timeline.draft_edits());
                        self.timeline.clear_preview();
                        self.timeline.selected_take = None;
                        let modifiers = ui.input(|i| i.modifiers);
                        self.timeline.pick_item(
                            project,
                            selection::Item::Scene(scene.id.clone()),
                            scene.sections.is_empty() && (modifiers.ctrl || modifiers.command),
                            scene.sections.is_empty() && modifiers.shift,
                            response.secondary_clicked(),
                        );
                    }
                    if response.drag_started() {
                        self.timeline.boundary = false;
                        commands.extend(self.timeline.draft_edits());
                        let modifiers = ui.input(|i| i.modifiers);
                        if modifiers.shift || modifiers.ctrl || modifiers.command || !active {
                            self.timeline.pick_item(
                                project,
                                selection::Item::Scene(scene.id.clone()),
                                modifiers.ctrl || modifiers.command,
                                modifiers.shift,
                                false,
                            );
                        }
                    }
                    if let Some((before, dx, done)) =
                        move_gesture(&mut self.timeline, &response, project)
                    {
                        ui.painter().rect_stroke(
                            bar.translate(egui::vec2(dx, 0.)),
                            3,
                            Stroke::new(1.5, helto::FOCUS),
                            egui::StrokeKind::Inside,
                        );
                        if done {
                            commands.extend(self.timeline.move_selection(
                                &before,
                                seconds(scene.start),
                                f64::from(dx / self.timeline.zoom),
                            ));
                        }
                    }
                    let edge = Rect::from_min_max(egui::pos2(bar.right() - 6., bar.top()), bar.max);
                    let resize = ui
                        .interact(
                            edge,
                            ui.id().with((&scene.id.0, "end")),
                            if enabled && scene.sections.is_empty() {
                                Sense::drag()
                            } else {
                                Sense::hover()
                            },
                        )
                        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                    if resize.drag_started() {
                        commands.extend(self.timeline.draft_edits());
                        self.timeline.pick_item(
                            project,
                            selection::Item::Scene(scene.id.clone()),
                            false,
                            false,
                            false,
                        );
                        resize.request_focus();
                        self.timeline.keyboard_focus = Some(resize.id);
                    }
                    if let Some((before, dx, done)) = gesture(&mut self.timeline, &resize, project)
                        && done
                        && let Ok(source) = before.scene(&scene.id)
                    {
                        let end = editing::snapped(
                            seconds(source.start)
                                + seconds(source.duration)
                                + f64::from(dx / self.timeline.zoom),
                            self.timeline.snap,
                            project.delivery.fps,
                        );
                        commands.extend(resize_scene(
                            &before,
                            source,
                            seconds(end) - seconds(source.start),
                            self.timeline.mode,
                        ));
                    }
                    if index > 0 {
                        let marker = Rect::from_center_size(
                            egui::pos2(x, bar.center().y),
                            egui::vec2(11., 18.),
                        );
                        let mode = super::assembly::boundary_for(
                            project,
                            &project.scenes[index - 1].id,
                            &scene.id,
                        )
                        .mode;
                        let response = ui
                            .interact(
                                marker,
                                ui.id().with(("boundary", &scene.id.0)),
                                if enabled {
                                    Sense::click()
                                } else {
                                    Sense::hover()
                                },
                            )
                            .on_hover_text(format!(
                                "Boundary before Scene {} · {}",
                                index + 1,
                                boundary_label(&mode)
                            ));
                        ui.painter().rect_filled(
                            marker,
                            3,
                            if self.timeline.boundary && active {
                                helto::ACCENT_TOP
                            } else {
                                helto::SURFACE
                            },
                        );
                        ui.painter().rect_stroke(
                            marker,
                            3,
                            Stroke::new(
                                1.,
                                if response.has_focus() {
                                    helto::FOCUS
                                } else {
                                    helto::FIELD
                                },
                            ),
                            egui::StrokeKind::Inside,
                        );
                        ui.painter().text(
                            marker.center(),
                            egui::Align2::CENTER_CENTER,
                            boundary_glyph(&mode),
                            egui::FontId::proportional(10.),
                            helto::DIM,
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, format!("boundary-{index}"), &response);
                        if response.clicked() {
                            commands.extend(self.timeline.draft_edits());
                            self.timeline.clear_audio_selection();
                            self.timeline.clear_preview();
                            self.timeline.selected_take = None;
                            self.timeline.selected = BTreeSet::from([scene.id.clone()]);
                            self.timeline.section = None;
                            self.timeline.boundary = true;
                            self.timeline.sync_draft();
                        }
                    }
                    for section in &scene.sections {
                        let a = x + seconds(section.start) as f32 * self.timeline.zoom;
                        let r = Rect::from_min_size(
                            egui::pos2(a, canvas.top() + 53.),
                            egui::vec2(
                                (seconds(section.duration) as f32 * self.timeline.zoom).max(8.),
                                22.,
                            ),
                        );
                        let selected = self.timeline.section_selection.contains(&section.id)
                            || self.timeline.section.as_ref() == Some(&section.id);
                        ui.painter().rect_filled(
                            r,
                            3,
                            if selected {
                                helto::ACCENT_TOP
                            } else {
                                helto::SURFACE
                            },
                        );
                        ui.painter().rect_stroke(
                            r,
                            3,
                            Stroke::new(1., helto::FIELD),
                            egui::StrokeKind::Inside,
                        );
                        let response = ui.interact(
                            r,
                            ui.id().with(&section.id.0),
                            if enabled {
                                Sense::click_and_drag()
                            } else {
                                Sense::hover()
                            },
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, format!("section-{}", section.id.0), &response);
                        if response.clicked() || response.secondary_clicked() {
                            response.request_focus();
                            self.timeline.keyboard_focus = Some(response.id);
                            self.timeline.boundary = false;
                            commands.extend(self.timeline.draft_edits());
                            let modifiers = ui.input(|i| i.modifiers);
                            self.timeline.pick_item(
                                project,
                                selection::Item::Section(section.id.clone()),
                                modifiers.ctrl || modifiers.command,
                                modifiers.shift,
                                response.secondary_clicked(),
                            );
                        }
                        response.context_menu(|ui| {
                            if let Some(asset) = self.timeline.section_preview(project, &section.id)
                            {
                                let preview = ui.button("Preview image");
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(ui, "section-context-preview".into(), &preview);
                                if preview.clicked() {
                                    if let Some(runtime) = self
                                        .timeline
                                        .document
                                        .as_ref()
                                        .and_then(|d| d.runtime_asset(&asset.0))
                                        .map(str::to_owned)
                                    {
                                        self.timeline_load_media(ui.ctx(), runtime);
                                    } else {
                                        self.timeline.error =
                                            Some("Image source is unavailable.".into());
                                    }
                                    ui.close();
                                }
                            }
                            let duplicate =
                                ui.add_enabled(enabled, egui::Button::new("Duplicate selection"));
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "section-context-duplicate".into(), &duplicate);
                            if duplicate.clicked() {
                                commands.extend(self.timeline.duplicate_selection(project));
                                ui.close();
                            }
                            let split = ui.add_enabled(
                                enabled,
                                egui::Button::new("Split Section at midpoint"),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "section-context-split".into(), &split);
                            if split.clicked() {
                                match section_commands::split_after(
                                    project,
                                    self.timeline.draft_edits(),
                                    &section.id,
                                    self.timeline.snap,
                                ) {
                                    Ok(edits) => {
                                        if let Some((scene, id)) =
                                            edits.iter().find_map(|e| match e {
                                                Edit::AddSection { scene_id, section } => {
                                                    Some((scene_id.clone(), section.id.clone()))
                                                }
                                                _ => None,
                                            })
                                        {
                                            self.timeline.after_edit =
                                                Some(PostAction::SelectSection(scene, id));
                                        }
                                        commands.extend(edits);
                                    }
                                    Err(error) => {
                                        self.timeline.error =
                                            Some(format!("Could not split Section: {}", error.code))
                                    }
                                }
                                ui.close();
                            }
                            let delete = ui.add_enabled(
                                enabled,
                                egui::Button::new(
                                    if self.timeline.selection_items(project).len() > 1 {
                                        "Delete selection"
                                    } else {
                                        "Delete Section"
                                    },
                                ),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "section-context-delete".into(), &delete);
                            if delete.clicked() {
                                commands.extend(self.timeline.draft_edits());
                                let selected =
                                    self.timeline.section_selection.contains(&section.id);
                                if selected {
                                    commands.extend(self.timeline.delete_selection());
                                } else {
                                    commands.push(Edit::RemoveSection {
                                        scene_id: scene.id.clone(),
                                        section_id: section.id.clone(),
                                    });
                                }
                                ui.close();
                            }
                        });
                        if response.drag_started() {
                            let modifiers = ui.input(|i| i.modifiers);
                            if modifiers.shift || modifiers.ctrl || modifiers.command || !selected {
                                commands.extend(self.timeline.draft_edits());
                                self.timeline.pick_item(
                                    project,
                                    selection::Item::Section(section.id.clone()),
                                    modifiers.ctrl || modifiers.command,
                                    modifiers.shift,
                                    false,
                                );
                            }
                        }
                        if let Some((before, dx, done)) =
                            move_gesture(&mut self.timeline, &response, project)
                            && done
                            && let Ok(source) = before.scene(&scene.id)
                        {
                            match group_commands::prepared(&before, self.timeline.draft_edits()) {
                                Ok(current) => {
                                    commands.extend(editing::difference(&before, &current));
                                    commands.extend(self.timeline.move_selection(
                                        &current,
                                        seconds(source.start) + seconds(section.start),
                                        f64::from(dx / self.timeline.zoom),
                                    ));
                                }
                                Err(e) => {
                                    self.timeline.error =
                                        Some(format!("Could not move selection: {}", e.code))
                                }
                            }
                        }
                        for end in [false, true] {
                            let x = if end { r.right() - 5. } else { r.left() };
                            let handle = Rect::from_min_size(
                                egui::pos2(x, r.top()),
                                egui::vec2(5., r.height()),
                            );
                            let response = ui
                                .interact(
                                    handle,
                                    ui.id().with((&section.id.0, end)),
                                    if enabled {
                                        Sense::drag()
                                    } else {
                                        Sense::hover()
                                    },
                                )
                                .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(
                                ui,
                                format!(
                                    "section-{}-{}",
                                    section.id.0,
                                    if end { "end" } else { "start" }
                                ),
                                &response,
                            );
                            if response.drag_started() {
                                commands.extend(self.timeline.draft_edits());
                                self.timeline.pick_item(
                                    project,
                                    selection::Item::Section(section.id.clone()),
                                    false,
                                    false,
                                    false,
                                );
                                response.request_focus();
                                self.timeline.keyboard_focus = Some(response.id);
                            }
                            if let Some((before, dx, done)) =
                                gesture(&mut self.timeline, &response, project)
                                && done
                            {
                                let position = seconds(scene.start)
                                    + seconds(section.start)
                                    + if end { seconds(section.duration) } else { 0. }
                                    + f64::from(dx / self.timeline.zoom);
                                match group_commands::prepared(&before, self.timeline.draft_edits())
                                {
                                    Ok(current) => {
                                        commands.extend(editing::difference(&before, &current));
                                        commands.extend(group_commands::resize(
                                            &current,
                                            &section.id,
                                            end,
                                            position,
                                            self.timeline.snap,
                                            self.timeline.mode,
                                        ));
                                    }
                                    Err(e) => {
                                        self.timeline.error =
                                            Some(format!("Could not resize Section: {}", e.code))
                                    }
                                }
                            }
                        }
                        let label = if helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
                            navigation::section_label(
                                project,
                                section,
                                self.timeline.navigation.display,
                            )
                            .replace(['\n', '\r', '\t'], " ")
                        } else {
                            "Section".into()
                        };
                        ui.painter()
                            .with_clip_rect(r.shrink(2.).intersect(ui.clip_rect()))
                            .text(
                                egui::pos2(r.left() + 4., r.center().y),
                                egui::Align2::LEFT_CENTER,
                                label,
                                egui::FontId::proportional(10.),
                                if selected {
                                    helto::ACCENT_TEXT
                                } else {
                                    helto::DIM
                                },
                            );
                    }
                }
                self.timeline_audio_tracks(ui, canvas, project, enabled, commands);
                if project.audio_tracks.is_empty() {
                    ui.painter().text(
                        egui::pos2(canvas.left() + 4., canvas.top() + 84.),
                        egui::Align2::LEFT_TOP,
                        "Audio",
                        egui::FontId::proportional(10.),
                        helto::DIM,
                    );
                }
            });
            #[cfg(feature = "ui-preview")]
            {
                let response = ui.interact(
                    scroll.inner_rect,
                    ui.id().with("qa-track-scroll"),
                    Sense::hover(),
                );
                super::qa::control(ui, "timeline-scroll".into(), &response);
            }
            self.timeline.navigation.observe(
                scroll.state.offset.x,
                scroll.inner_rect.width(),
                self.timeline.zoom,
                whole,
            );
        });
    }
}
