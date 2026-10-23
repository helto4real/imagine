use super::editing::{move_clips, resize_clip};
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
                super::qa::control(ui, format!("audio-clip-{index}"), &response);
                if response.clicked() {
                    commands.extend(self.timeline.draft_edits());
                    if !ui.input(|i| i.modifiers.command || i.modifiers.shift) {
                        self.timeline.audio_selected.clear();
                    }
                    self.timeline.audio_selected.insert(c.id.clone());
                    self.timeline.audio_primary = Some(c.id.clone());
                    self.timeline.project_audio = false;
                    self.timeline.section = None;
                    self.timeline.sync_draft();
                }
                if response.drag_started() && !selected {
                    self.timeline.audio_selected = BTreeSet::from([c.id.clone()]);
                    self.timeline.audio_primary = Some(c.id.clone());
                    self.timeline.project_audio = false;
                    self.timeline.section = None;
                    self.timeline.sync_draft();
                }
                if enabled
                    && !c.locked
                    && let Some((before, dx, done)) = crate::app::timeline::tracks::gesture(
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
                        commands.extend(move_clips(
                            &before,
                            &self.timeline.audio_selected,
                            f64::from(dx / self.timeline.zoom),
                            self.timeline.snap,
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
                ui.painter()
                    .with_clip_rect(bar.intersect(ui.clip_rect()))
                    .text(
                        bar.left_center() + egui::vec2(7., 0.),
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
