use super::editing::resize_clip;
use super::*;
use egui::{Rect, Sense, Stroke};
impl ImagineApp {
    pub(in crate::app::timeline) fn timeline_audio_tracks(
        &mut self,
        ui: &mut egui::Ui,
        canvas: Rect,
        project: &VideoProject,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        let mut offset = 0;
        for track in &project.audio_tracks {
            for (index, c) in track.clips.iter().enumerate() {
                let bar = Rect::from_min_size(
                    egui::pos2(
                        canvas.left() + seconds(c.start) as f32 * self.timeline.zoom,
                        canvas.top() + 87. + (offset + c.lane) as f32 * 28.,
                    ),
                    egui::vec2(
                        (seconds(c.duration) as f32 * self.timeline.zoom).max(12.),
                        24.,
                    ),
                );
                let selected = self.timeline.audio_selected.contains(&c.id);
                ui.painter().rect_filled(
                    bar,
                    3,
                    if selected {
                        helto::ACCENT_TOP
                    } else {
                        helto::FIELD
                    },
                );
                self.timeline_waveform(ui, project, c, bar);
                let response = ui.interact(
                    bar,
                    ui.id().with((&c.id.0, "audio")),
                    if !enabled {
                        Sense::hover()
                    } else if c.locked {
                        Sense::click()
                    } else {
                        Sense::click_and_drag()
                    },
                );
                #[cfg(feature = "ui-preview")]
                {
                    super::qa::control(ui, format!("audio-clip-{index}"), &response);
                    super::qa::control(ui, format!("audio-clip-id-{}", c.id.0), &response);
                }
                if response.clicked() || response.secondary_clicked() {
                    response.request_focus();
                    self.timeline.keyboard_focus = Some(response.id);
                    commands.extend(self.timeline.draft_edits());
                    let modifiers = ui.input(|i| i.modifiers);
                    self.timeline.pick_item(
                        project,
                        crate::app::timeline::selection::Item::Audio(c.id.clone()),
                        modifiers.ctrl || modifiers.command,
                        modifiers.shift,
                        response.secondary_clicked(),
                    );
                }
                response.context_menu(|ui| {
                    if let Some(asset) = &c.asset {
                        let failures = self
                            .timeline
                            .waveform
                            .values
                            .iter()
                            .filter(|((id, _, _), _)| id == asset)
                            .filter_map(|(_, v)| v.as_ref().err())
                            .cloned()
                            .collect::<Vec<_>>();
                        if let Some(error) = failures.first() {
                            ui.label(format!("Waveform unavailable: {error}"));
                            if ui.button("Retry waveform").clicked() {
                                self.timeline
                                    .waveform
                                    .values
                                    .retain(|(id, _, _), _| id != asset);
                                ui.close();
                            }
                        }
                    }
                    let r = ui.add_enabled(enabled, egui::Button::new("Duplicate selection"));
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(ui, "audio-context-duplicate".into(), &r);
                    if r.clicked() {
                        commands.extend(self.timeline.duplicate_selection(project));
                        ui.close();
                    }
                    let r = ui.add_enabled(enabled, egui::Button::new("Delete selection"));
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(ui, "audio-context-delete".into(), &r);
                    if r.clicked() {
                        commands.extend(self.timeline.draft_edits());
                        commands.extend(self.timeline.delete_selection());
                        ui.close();
                    }
                });
                if response.drag_started() {
                    let modifiers = ui.input(|i| i.modifiers);
                    if modifiers.shift || modifiers.ctrl || modifiers.command || !selected {
                        commands.extend(self.timeline.draft_edits());
                        self.timeline.pick_item(
                            project,
                            crate::app::timeline::selection::Item::Audio(c.id.clone()),
                            modifiers.ctrl || modifiers.command,
                            modifiers.shift,
                            false,
                        );
                    }
                }
                if enabled
                    && !c.locked
                    && let Some((before, dx, done)) = crate::app::timeline::tracks::move_gesture(
                        &mut self.timeline,
                        &response,
                        project,
                    )
                {
                    ui.painter().rect_stroke(
                        bar.translate(egui::vec2(dx, 0.)),
                        3,
                        Stroke::new(1., helto::FOCUS),
                        egui::StrokeKind::Inside,
                    );
                    if done {
                        commands.extend(self.timeline.move_selection(
                            &before,
                            seconds(c.start),
                            f64::from(dx / self.timeline.zoom),
                        ));
                    }
                }
                for end in [false, true] {
                    let x = if end { bar.right() - 6. } else { bar.left() };
                    let r =
                        Rect::from_min_size(egui::pos2(x, bar.top()), egui::vec2(6., bar.height()));
                    let response = ui
                        .interact(
                            r,
                            ui.id().with((&c.id.0, "audio-edge", end)),
                            if enabled && !c.locked {
                                Sense::drag()
                            } else {
                                Sense::hover()
                            },
                        )
                        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(
                        ui,
                        format!("audio-edge-{index}-{end}"),
                        &response,
                    );
                    if response.drag_started() {
                        commands.extend(self.timeline.draft_edits());
                        self.timeline.pick_item(
                            project,
                            crate::app::timeline::selection::Item::Audio(c.id.clone()),
                            false,
                            false,
                            false,
                        );
                        response.request_focus();
                        self.timeline.keyboard_focus = Some(response.id);
                    }
                    if let Some((before, dx, done)) = crate::app::timeline::tracks::gesture(
                        &mut self.timeline,
                        &response,
                        project,
                    ) && done
                        && let Some(source) = clip(&before, &c.id)
                    {
                        let at = seconds(source.start)
                            + if end { seconds(source.duration) } else { 0. }
                            + f64::from(dx / self.timeline.zoom);
                        commands.extend(resize_clip(&before, source, end, at, self.timeline.snap));
                    }
                }
                let revealed = !self.prefs.privacy_mode
                    || (ui.input(|i| i.focused)
                        && ui.rect_contains_pointer(bar.intersect(ui.clip_rect())));
                let text = if !revealed {
                    "Hidden audio".into()
                } else if c.name.is_empty() {
                    format!(
                        "Audio {}{}",
                        index + 1,
                        if c.locked { " · Locked" } else { "" }
                    )
                } else {
                    c.name.clone()
                };
                let painter = ui.painter().with_clip_rect(bar.intersect(ui.clip_rect()));
                let position = bar.left_center() + egui::vec2(7., 0.);
                // Director keeps its label above the waveform with a dark
                // text shadow. Outline the native-sized text for the same
                // legibility without obscuring the waveform's timing.
                for offset in [
                    egui::vec2(-1., 0.),
                    egui::vec2(1., 0.),
                    egui::vec2(0., -1.),
                    egui::vec2(0., 1.),
                ] {
                    painter.text(
                        position + offset,
                        egui::Align2::LEFT_CENTER,
                        &text,
                        egui::FontId::proportional(11.),
                        helto::BG,
                    );
                }
                painter.text(
                    position,
                    egui::Align2::LEFT_CENTER,
                    text,
                    egui::FontId::proportional(11.),
                    if selected {
                        helto::ACCENT_TEXT
                    } else {
                        helto::TEXT
                    },
                );
            }
            offset += track
                .clips
                .iter()
                .map(|c| c.lane)
                .max()
                .map(|n| n + 1)
                .unwrap_or(0);
        }
    }
}
