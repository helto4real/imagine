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
            ui.horizontal(|ui| {
                ui.strong("Timeline");
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
                egui::ComboBox::from_id_salt("video-edit-mode")
                    .width(100.)
                    .selected_text(if self.timeline.mode == Mode::Ripple {
                        "Ripple Edit"
                    } else {
                        "Trim Neighbor"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.timeline.mode,
                            Mode::TrimNeighbor,
                            "Trim Neighbor",
                        );
                        ui.selectable_value(&mut self.timeline.mode, Mode::Ripple, "Ripple Edit");
                    });
                let zoom = ui.add(
                    egui::Slider::new(&mut self.timeline.zoom, 24.0..=160.0)
                        .show_value(false)
                        .text("Zoom"),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-zoom".into(), &zoom);
                let _ = zoom;
            });
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
                .fold(10., f64::max)
                + 2.;
            let audio_lanes = project
                .audio_tracks
                .iter()
                .map(|t| t.clips.iter().map(|c| c.lane + 1).max().unwrap_or(0))
                .sum::<u32>()
                .max(1);
            egui::ScrollArea::both()
                .id_salt("video-track-pan")
                .show(ui, |ui| {
                    let size = egui::vec2(
                        (end as f32 * self.timeline.zoom).max(ui.available_width()),
                        90. + audio_lanes as f32 * 28.,
                    );
                    let (canvas, _) = ui.allocate_exact_size(size, Sense::hover());
                    for second in 0..end.ceil() as usize {
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
                    for (index, scene) in project.scenes.iter().enumerate() {
                        let x = canvas.left() + seconds(scene.start) as f32 * self.timeline.zoom;
                        let w = seconds(scene.duration) as f32 * self.timeline.zoom;
                        let bar = Rect::from_min_size(
                            egui::pos2(x, canvas.top() + 19.),
                            egui::vec2(w.max(12.), 28.),
                        );
                        let active = self.timeline.selected.contains(&scene.id);
                        let response = ui.interact(
                            bar,
                            ui.id().with((&scene.id.0, "scene")),
                            if enabled {
                                Sense::click_and_drag()
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
                        ui.painter().text(
                            bar.left_center() + egui::vec2(8., 0.),
                            egui::Align2::LEFT_CENTER,
                            format!("{:02} · {:.2} s", index + 1, seconds(scene.duration)),
                            egui::FontId::proportional(12.),
                            if active {
                                helto::ACCENT_TEXT
                            } else {
                                helto::TEXT
                            },
                        );
                        if response.clicked() {
                            self.timeline.boundary = false;
                            commands.extend(self.timeline.draft_edits());
                            self.timeline.clear_audio_selection();
                            self.timeline.clear_preview();
                            self.timeline.selected_take = None;
                            if !ui.input(|i| i.modifiers.shift || i.modifiers.command) {
                                self.timeline.selected.clear();
                            }
                            self.timeline.selected.insert(scene.id.clone());
                            self.timeline.section = None;
                            self.timeline.sync_draft();
                        }
                        if response.drag_started() && !self.timeline.selected.contains(&scene.id) {
                            self.timeline.boundary = false;
                            commands.extend(self.timeline.draft_edits());
                            self.timeline.clear_audio_selection();
                            self.timeline.selected = BTreeSet::from([scene.id.clone()]);
                            self.timeline.section = None;
                            self.timeline.sync_draft();
                        }
                        if let Some((before, dx, done)) =
                            gesture(&mut self.timeline, &response, project)
                        {
                            ui.painter().rect_stroke(
                                bar.translate(egui::vec2(dx, 0.)),
                                3,
                                Stroke::new(1.5, helto::FOCUS),
                                egui::StrokeKind::Inside,
                            );
                            if done {
                                commands.extend(editing::move_scenes(
                                    &before,
                                    &self.timeline.selected,
                                    f64::from(dx / self.timeline.zoom),
                                    self.timeline.snap,
                                ));
                            }
                        }
                        let edge =
                            Rect::from_min_max(egui::pos2(bar.right() - 6., bar.top()), bar.max);
                        let resize = ui
                            .interact(
                                edge,
                                ui.id().with((&scene.id.0, "end")),
                                if enabled {
                                    Sense::drag()
                                } else {
                                    Sense::hover()
                                },
                            )
                            .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                        if let Some((before, dx, done)) =
                            gesture(&mut self.timeline, &resize, project)
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
                                .on_hover_text(format!("Boundary before Scene {}", index + 1));
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
                            let selected = self.timeline.section.as_ref() == Some(&section.id);
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
                            if response.clicked() {
                                self.timeline.boundary = false;
                                commands.extend(self.timeline.draft_edits());
                                self.timeline.clear_audio_selection();
                                self.timeline.selected = BTreeSet::from([scene.id.clone()]);
                                self.timeline.section = Some(section.id.clone());
                                self.timeline.sync_draft();
                            }
                            if let Some((before, dx, done)) =
                                gesture(&mut self.timeline, &response, project)
                                && done
                                && let Ok(source) = before.scene(&scene.id)
                            {
                                commands.extend(editing::section_gesture(
                                    source,
                                    &section.id,
                                    None,
                                    seconds(section.start) + f64::from(dx / self.timeline.zoom),
                                    self.timeline.snap,
                                    project.delivery.fps,
                                    self.timeline.mode,
                                ));
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
                                if let Some((before, dx, done)) =
                                    gesture(&mut self.timeline, &response, project)
                                    && done
                                    && let Ok(source) = before.scene(&scene.id)
                                {
                                    let position = seconds(section.start)
                                        + if end { seconds(section.duration) } else { 0. }
                                        + f64::from(dx / self.timeline.zoom);
                                    commands.extend(editing::section_gesture(
                                        source,
                                        &section.id,
                                        Some(end),
                                        position,
                                        self.timeline.snap,
                                        project.delivery.fps,
                                        self.timeline.mode,
                                    ));
                                }
                            }
                            ui.painter().text(
                                r.center(),
                                egui::Align2::CENTER_CENTER,
                                "Section",
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
        });
    }
}
