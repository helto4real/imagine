//! Timeline geometry, selection and native pointer gestures.
use super::editing::{Mode, Snap, resize_scene, seconds};
use super::*;
use crate::app::workspace as chrome;
use egui::{Rect, Sense, Stroke};
/// Director track rows, from the canvas top: the ruler, Scene bands, Section
/// blocks, then one 28-point lane per audio overlap level.
const BAND_TOP: f32 = 20.;
const BAND_H: f32 = 20.;
const SECTION_TOP: f32 = 44.;
const SECTION_H: f32 = 72.;
pub(super) const AUDIO_TOP: f32 = 122.;
#[derive(Clone)]
pub(super) struct Drag {
    id: egui::Id,
    origin: egui::Pos2,
    before: VideoProject,
}
/// Media dragged from the shelf; see `inspector::timeline_references`.
#[derive(Clone)]
pub(super) struct ShelfMedia(pub AssetId);
/// Where shelf media was dropped on the Timeline.
pub(super) struct Dropped {
    pub asset: AssetId,
    pub scene: SceneId,
    pub section: Option<SectionId>,
}
/// Highlights `rect` while shelf media hovers it and records a drop.
fn drop_target(
    ui: &egui::Ui,
    response: &egui::Response,
    rect: Rect,
    dropped: &mut Option<Dropped>,
    scene: &SceneId,
    section: Option<&SectionId>,
) {
    if response.dnd_hover_payload::<ShelfMedia>().is_some() {
        ui.painter().rect_stroke(
            rect,
            3,
            Stroke::new(2., helto::ACCENT),
            egui::StrokeKind::Inside,
        );
    }
    if let Some(media) = response.dnd_release_payload::<ShelfMedia>() {
        *dropped = Some(Dropped {
            asset: media.0.clone(),
            scene: scene.clone(),
            section: section.cloned(),
        });
    }
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
/// How a Section block's role chip is coloured.
#[derive(Clone, Copy, PartialEq)]
enum Tone {
    Plain,
    Inherited,
    Waiting,
}
/// The role of the media a Section block shows, or where its first frame comes from.
pub(super) struct Chip {
    text: String,
    tone: Tone,
}
impl Chip {
    pub(super) fn plain(text: &str) -> Self {
        Self {
            text: text.into(),
            tone: Tone::Plain,
        }
    }
    pub(super) fn inherited(text: String) -> Self {
        Self {
            text,
            tone: Tone::Inherited,
        }
    }
    pub(super) fn waiting(text: String) -> Self {
        Self {
            text,
            tone: Tone::Waiting,
        }
    }
    /// Whether it names bound media rather than another Scene.
    pub(super) fn is_media(&self) -> bool {
        self.tone == Tone::Plain
    }
}
/// The Scene holding `section`, and the Section itself.
fn owner<'a>(project: &'a VideoProject, section: &SectionId) -> Option<(&'a Scene, &'a Section)> {
    project
        .scenes
        .iter()
        .find_map(|s| Some((s, s.sections.iter().find(|v| &v.id == section)?)))
}
impl Timeline {
    /// While a drag lasts, the tracks show `edits` applied to the project as the
    /// drag began; once it ends they are sent.
    pub(super) fn drag_edits(
        &mut self,
        before: &VideoProject,
        edits: Vec<Edit>,
        done: bool,
        commands: &mut Vec<Edit>,
    ) {
        if done {
            self.drag_preview = None;
            commands.extend(edits);
            return;
        }
        let mut preview = before.clone();
        let applied = edits.is_empty()
            || imagine_video_contract::editing::apply(
                &mut preview,
                imagine_video_contract::editing::EditRequest {
                    schema_version: 1,
                    expected_project_revision: before.project_revision,
                    commands: edits,
                },
            )
            .is_ok();
        // An edit the project would refuse keeps the last valid picture.
        if applied {
            self.drag_preview = Some(preview);
        }
    }
}
/// Director's playhead: a gold line over every track with a marker in the ruler.
fn playhead(ui: &egui::Ui, canvas: Rect, x: f32) {
    let painter = ui.painter();
    painter.vline(
        x,
        canvas.top()..=canvas.bottom(),
        Stroke::new(2., helto::ACCENT),
    );
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(x - 6., canvas.top()),
            egui::pos2(x + 6., canvas.top()),
            egui::pos2(x, canvas.top() + 8.),
        ],
        helto::ACCENT,
        Stroke::NONE,
    ));
}
/// The Scene band's take count, coloured by the Scene's take state, at its right end.
fn take_badge(ui: &egui::Ui, bar: Rect, state: SceneState, takes: usize) -> Rect {
    let (text, fill, color) = match state {
        SceneState::Generating => ("…".to_owned(), helto::FIELD, helto::FOCUS),
        SceneState::Accepted => (format!("T{takes}"), helto::ACCENT, helto::SURFACE),
        SceneState::Outdated => (format!("T{takes}"), helto::WARN, helto::SURFACE),
        _ => (format!("T{takes}"), helto::RAISED, helto::DIM),
    };
    let galley = ui
        .painter()
        .layout_no_wrap(text, egui::FontId::proportional(10.), color);
    let size = egui::vec2(galley.size().x + 10., 14.);
    let badge = Rect::from_min_size(
        egui::pos2(bar.right() - size.x - 3., bar.center().y - size.y / 2.),
        size,
    );
    if bar.width() > size.x + 24. {
        ui.painter().rect_filled(badge, 7, fill);
        ui.painter()
            .galley(badge.center() - galley.size() / 2., galley, color);
        badge
    } else {
        Rect::from_min_size(bar.right_top(), egui::Vec2::ZERO)
    }
}
/// A Section block's label at the top, and its role chip and length at the bottom.
fn section_text(
    ui: &egui::Ui,
    r: Rect,
    label: &str,
    chip: Option<Chip>,
    duration: f64,
    selected: bool,
) {
    let painter = ui
        .painter()
        .with_clip_rect(r.shrink(2.).intersect(ui.clip_rect()));
    let galley = painter.layout(
        label.to_owned(),
        egui::FontId::proportional(12.),
        if selected {
            helto::ACCENT_TEXT
        } else {
            helto::TEXT
        },
        (r.width() - 12.).max(8.),
    );
    painter.galley(r.left_top() + egui::vec2(6., 5.), galley, helto::TEXT);
    let length = painter.layout_no_wrap(
        format!("{duration:.1} s"),
        egui::FontId::proportional(11.),
        helto::DIM,
    );
    let length_at = egui::pos2(
        r.right() - length.size().x - 6.,
        r.bottom() - length.size().y - 5.,
    );
    if r.width() > length.size().x + 16. {
        painter.galley(length_at, length, helto::DIM);
    }
    let Some(chip) = chip else {
        return;
    };
    let color = match chip.tone {
        Tone::Plain => helto::TEXT,
        Tone::Inherited => helto::ACCENT_TEXT,
        Tone::Waiting => helto::FOCUS,
    };
    let text = painter.layout_no_wrap(chip.text, egui::FontId::proportional(11.), color);
    let pill = Rect::from_min_size(
        egui::pos2(r.left() + 5., r.bottom() - 21.),
        egui::vec2(text.size().x + 12., 17.),
    );
    if pill.right() > length_at.x - 4. && r.width() < pill.width() + 10. {
        return;
    }
    painter.rect_filled(pill, 8, egui::Color32::from_black_alpha(190));
    if chip.tone != Tone::Plain {
        painter.rect_stroke(
            pill,
            8,
            Stroke::new(
                1.,
                if chip.tone == Tone::Inherited {
                    helto::ACCENT_BORDER
                } else {
                    helto::FOCUS
                },
            ),
            egui::StrokeKind::Inside,
        );
    }
    painter.galley(pill.center() - text.size() / 2., text, color);
}
impl Timeline {
    /// Takes attached to the Scene, accepted or not.
    fn scene_takes(&self, scene: &SceneId) -> usize {
        self.document.as_ref().map_or(0, |d| {
            d.view
                .takes
                .iter()
                .filter(|t| &t.scene_id == scene && review::attached(d, t))
                .count()
        })
    }
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
        // A drag in progress draws its result live; the drag sends it on release.
        if self.timeline.drag.is_none() {
            self.timeline.drag_preview = None;
        }
        let preview = self.timeline.drag_preview.clone();
        let project = preview.as_ref().unwrap_or(project);
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
                ui.separator();
                self.timeline_edit_tools(ui, project, enabled, commands);
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
                ui.separator();
                // Director's mode menus: one icon each, the current choice in its name.
                let display = self.timeline.navigation.display;
                let menu = helto::icon_button(
                    ui,
                    crate::icons::Icon::Menu,
                    &format!("Labels: {}", display.label()),
                    false,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-display".into(), &menu);
                egui::Popup::menu(&menu).show(|ui| {
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
                let mode = if self.timeline.mode == Mode::Ripple {
                    "Ripple Edit"
                } else {
                    "Trim Neighbor"
                };
                let menu = helto::icon_button(
                    ui,
                    crate::icons::Icon::Swap,
                    &format!("Edit mode: {mode}"),
                    self.timeline.mode == Mode::Ripple,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-edit-mode".into(), &menu);
                egui::Popup::menu(&menu).show(|ui| {
                    let trim = ui.selectable_value(
                        &mut self.timeline.mode,
                        Mode::TrimNeighbor,
                        "Trim Neighbor",
                    );
                    let ripple =
                        ui.selectable_value(&mut self.timeline.mode, Mode::Ripple, "Ripple Edit");
                    #[cfg(feature = "ui-preview")]
                    {
                        super::qa::control(ui, "timeline-edit-trim".into(), &trim);
                        super::qa::control(ui, "timeline-edit-ripple".into(), &ripple);
                    }
                    let _ = (trim, ripple);
                });
                let snap = match self.timeline.snap {
                    Snap::None => "None",
                    Snap::Seconds => "Seconds",
                    Snap::Frames => "Frames",
                };
                let menu = helto::icon_button(
                    ui,
                    crate::icons::Icon::Snap,
                    &format!("Snap: {snap}"),
                    self.timeline.snap != Snap::None,
                );
                egui::Popup::menu(&menu).show(|ui| {
                    for (value, label) in [
                        (Snap::None, "None"),
                        (Snap::Seconds, "Seconds"),
                        (Snap::Frames, "Frames"),
                    ] {
                        ui.selectable_value(&mut self.timeline.snap, value, label);
                    }
                });
                ui.separator();
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
                    crate::icons::Icon::PanelRight,
                    "Toggle media bin",
                    !self.timeline.shelf_collapsed,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-toggle-shelf".into(), &shelf);
                if shelf.clicked() {
                    self.timeline.shelf_collapsed = !self.timeline.shelf_collapsed;
                }
                let inspector = helto::icon_button(
                    ui,
                    crate::icons::Icon::PanelBottom,
                    "Toggle inspector",
                    !self.timeline.inspector_collapsed,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "timeline-toggle-inspector".into(), &inspector);
                if inspector.clicked() {
                    self.timeline.inspector_collapsed = !self.timeline.inspector_collapsed;
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
                    // Down to the panel's bottom, so the space under the tracks
                    // is empty track space too.
                    (AUDIO_TOP + 3. + audio_lanes as f32 * 28.).max(ui.available_height()),
                );
                // Blocks are added after it and win their own clicks; a click on
                // empty track space clears the selection, as in Director.
                let (canvas, background) = ui.allocate_exact_size(size, Sense::click());
                if background.clicked() {
                    commands.extend(self.timeline.draft_edits());
                    self.timeline.clear_preview();
                    self.timeline.clear_selection(project);
                }
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
                let head = canvas.left() + self.timeline.playhead as f32 * self.timeline.zoom;
                // Drawn last, over the blocks; see `playhead` below.
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
                        egui::pos2(x + 1., canvas.top() + BAND_TOP),
                        egui::vec2((w - 2.).max(12.), BAND_H),
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
                        10,
                        if active {
                            helto::ACCENT_TOP
                        } else {
                            helto::FIELD
                        },
                    );
                    ui.painter().rect_stroke(
                        bar,
                        10,
                        Stroke::new(1., if active { helto::ACCENT } else { helto::RAISED }),
                        egui::StrokeKind::Inside,
                    );
                    let badge = take_badge(ui, bar, state, self.timeline.scene_takes(&scene.id));
                    if enabled {
                        drop_target(ui, &response, bar, &mut self.timeline.drop, &scene.id, None);
                    }
                    let title = project
                        .creative
                        .scene_prompts
                        .get(&scene.id)
                        .map(|prompt| prompt.title.trim())
                        .filter(|title| !title.is_empty())
                        .unwrap_or_else(|| scene.description.lines().next().unwrap_or("").trim());
                    let response = response.on_hover_text(format!(
                        "Scene {} · {:.2} s · {}",
                        index + 1,
                        seconds(scene.duration),
                        state.label()
                    ));
                    let title_area =
                        Rect::from_min_max(bar.min, egui::pos2(badge.left() - 4., bar.bottom()));
                    ui.painter()
                        .with_clip_rect(title_area.intersect(ui.clip_rect()))
                        .text(
                            bar.left_center() + egui::vec2(9., 0.),
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
                    response.widget_info(|| {
                        egui::WidgetInfo::selected(
                            egui::WidgetType::Button,
                            enabled,
                            active,
                            format!("Scene {}", index + 1),
                        )
                    });
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
                        && let Ok(source) = before.scene(&scene.id)
                    {
                        let edits = self.timeline.move_selection(
                            &before,
                            seconds(source.start),
                            f64::from(dx / self.timeline.zoom),
                        );
                        self.timeline.drag_edits(&before, edits, done, commands);
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
                        && let Ok(source) = before.scene(&scene.id)
                    {
                        let end = editing::snapped(
                            seconds(source.start)
                                + seconds(source.duration)
                                + f64::from(dx / self.timeline.zoom),
                            self.timeline.snap,
                            project.delivery.fps,
                        );
                        let edits = resize_scene(
                            &before,
                            source,
                            seconds(end) - seconds(source.start),
                            self.timeline.mode,
                        );
                        self.timeline.drag_edits(&before, edits, done, commands);
                    }
                    for section in &scene.sections {
                        let a = x + seconds(section.start) as f32 * self.timeline.zoom;
                        let r = Rect::from_min_size(
                            egui::pos2(a + 1., canvas.top() + SECTION_TOP),
                            egui::vec2(
                                (seconds(section.duration) as f32 * self.timeline.zoom - 2.)
                                    .max(8.),
                                SECTION_H,
                            ),
                        );
                        let selected = self.timeline.section_selection.contains(&section.id)
                            || self.timeline.section.as_ref() == Some(&section.id);
                        let revealed = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                        let chip =
                            self.timeline_section_block(ui, project, scene, section, r, revealed);
                        let response = ui.interact(
                            r,
                            ui.id().with(&section.id.0),
                            if enabled {
                                Sense::click_and_drag()
                            } else {
                                Sense::hover()
                            },
                        );
                        if enabled {
                            drop_target(
                                ui,
                                &response,
                                r,
                                &mut self.timeline.drop,
                                &scene.id,
                                Some(&section.id),
                            );
                        }
                        let number = scene.sections.iter().position(|v| v.id == section.id);
                        response.widget_info(|| {
                            egui::WidgetInfo::selected(
                                egui::WidgetType::Button,
                                enabled,
                                selected,
                                format!(
                                    "Section {} of Scene {}",
                                    number.map_or(0, |n| n + 1),
                                    index + 1
                                ),
                            )
                        });
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
                        // Positions come from the project as the drag began; the
                        // drawn one already shows the drag.
                        if let Some((before, dx, done)) =
                            move_gesture(&mut self.timeline, &response, project)
                            && let Some((source, part)) = owner(&before, &section.id)
                        {
                            match group_commands::prepared(&before, self.timeline.draft_edits()) {
                                Ok(current) => {
                                    let mut edits = editing::difference(&before, &current);
                                    edits.extend(self.timeline.move_selection(
                                        &current,
                                        seconds(source.start) + seconds(part.start),
                                        f64::from(dx / self.timeline.zoom),
                                    ));
                                    self.timeline.drag_edits(&before, edits, done, commands);
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
                                && let Some((source, part)) = owner(&before, &section.id)
                            {
                                let position = seconds(source.start)
                                    + seconds(part.start)
                                    + if end { seconds(part.duration) } else { 0. }
                                    + f64::from(dx / self.timeline.zoom);
                                match group_commands::prepared(&before, self.timeline.draft_edits())
                                {
                                    Ok(current) => {
                                        let mut edits = editing::difference(&before, &current);
                                        edits.extend(group_commands::resize(
                                            &current,
                                            &section.id,
                                            end,
                                            position,
                                            self.timeline.snap,
                                            self.timeline.mode,
                                        ));
                                        self.timeline.drag_edits(&before, edits, done, commands);
                                    }
                                    Err(e) => {
                                        self.timeline.error =
                                            Some(format!("Could not resize Section: {}", e.code))
                                    }
                                }
                            }
                        }
                        let label = if revealed {
                            navigation::section_label(
                                project,
                                section,
                                self.timeline.navigation.display,
                            )
                            .replace(['\n', '\r', '\t'], " ")
                        } else {
                            "Section".into()
                        };
                        section_text(ui, r, &label, chip, seconds(section.duration), selected);
                    }
                    if index > 0 {
                        // On the cut itself, as in Director.
                        let marker = Rect::from_center_size(
                            egui::pos2(x, canvas.top() + SECTION_TOP - 1.),
                            egui::vec2(24., 20.),
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
                        let on = self.timeline.boundary && active;
                        if on {
                            helto::gradient(
                                ui.painter(),
                                marker,
                                5.,
                                helto::ACCENT_TOP,
                                helto::ACCENT_BOTTOM,
                            );
                        } else {
                            helto::gradient(ui.painter(), marker, 5., helto::RAISED, helto::FIELD);
                        }
                        ui.painter().rect_stroke(
                            marker,
                            3,
                            Stroke::new(
                                1.,
                                if response.has_focus() {
                                    helto::FOCUS
                                } else if on {
                                    helto::ACCENT
                                } else {
                                    helto::RAISED
                                },
                            ),
                            egui::StrokeKind::Inside,
                        );
                        ui.painter().text(
                            marker.center(),
                            egui::Align2::CENTER_CENTER,
                            boundary_glyph(&mode),
                            egui::FontId::proportional(12.),
                            if on { helto::ACCENT_TEXT } else { helto::TEXT },
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
                }
                self.timeline_audio_tracks(ui, canvas, project, enabled, commands);
                playhead(ui, canvas, head);
                if project.audio_tracks.is_empty() {
                    ui.painter().text(
                        egui::pos2(canvas.left() + 4., canvas.top() + AUDIO_TOP + 6.),
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
