//! Session-local viewport navigation and Director display labels, independent of authoring.
use super::editing::seconds;
use super::*;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Display {
    #[default]
    Sections,
    Media,
    Prompts,
}
impl Display {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Sections => "Sections",
            Self::Media => "Media",
            Self::Prompts => "Prompts",
        }
    }
}
#[derive(Default)]
pub(super) struct State {
    pub display: Display,
    pub view: Option<(f64, f64)>,
    pub scroll_to: Option<f32>,
    width: f32,
    whole: f64,
    range_drag: Option<bool>,
}
pub(super) fn duration(project: &VideoProject) -> f64 {
    if let Some(value) = project.creative.timeline_duration {
        return seconds(value).ceil().max(1.);
    }
    project
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
}
pub(super) fn clamp_range(whole: f64, start: f64, end: f64) -> (f64, f64) {
    let whole = whole.ceil().max(1.);
    let start = if start.is_finite() { start.round() } else { 0. }.clamp(0., whole - 1.);
    let end = if end.is_finite() { end.round() } else { whole }.clamp(start + 1., whole);
    (start, end)
}
impl State {
    pub(super) fn set(&mut self, start: f64, end: f64, whole: f64, width: f32, zoom: &mut f32) {
        let range = clamp_range(whole, start, end);
        *zoom = (f64::from(width.max(1.)) / (range.1 - range.0)).max(1.) as f32;
        self.view = Some(range);
        self.width = width;
        self.whole = whole;
        self.scroll_to = Some(range.0 as f32 * *zoom);
    }
    pub(super) fn prepare(&mut self, whole: f64, width: f32, zoom: &mut f32) {
        if self.view.is_none() || (self.width - width).abs() > 1. || self.whole != whole {
            let (a, b) = self.view.unwrap_or((0., whole));
            self.set(a, b, whole, width, zoom);
        }
    }
    pub(super) fn observe(&mut self, offset: f32, width: f32, zoom: f32, whole: f64) {
        self.view = Some(clamp_range(
            whole,
            f64::from(offset / zoom),
            f64::from((offset + width) / zoom),
        ));
        // Never force an offset during ordinary scrolling, mouse movement or repaint.
    }
    pub(super) fn range_bar(&mut self, ui: &mut egui::Ui, whole: f64, zoom: &mut f32) {
        let (mut start, mut end) = self.view.unwrap_or((0., whole));
        let before = (start, end);
        ui.horizontal(|ui| {
            ui.label("View");
            let a = ui.add(egui::DragValue::new(&mut start).speed(1.).suffix(" s"));
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-range-start".into(), &a);
            ui.label("–");
            let b = ui.add(egui::DragValue::new(&mut end).speed(1.).suffix(" s"));
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-range-end".into(), &b);
            let _ = (a, b);
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(40.), 18.),
                egui::Sense::hover(),
            );
            ui.painter().rect_filled(rect, 3, helto::FIELD);
            let background = ui.interact(
                rect,
                ui.id().with("timeline-range-background"),
                egui::Sense::click_and_drag(),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-range-background".into(), &background);
            if background.clicked() || background.drag_started() {
                let pointer = ui
                    .input(|i| i.pointer.press_origin())
                    .or_else(|| background.interact_pointer_pos());
                if let Some(pointer) = pointer {
                    let second = f64::from((pointer.x - rect.left()) / rect.width()) * whole;
                    self.range_drag = Some((second - end).abs() < (second - start).abs());
                }
            }
            if (background.clicked() || background.dragged())
                && let Some(pointer) = background.interact_pointer_pos()
                && let Some(right) = self.range_drag
            {
                let value = f64::from((pointer.x - rect.left()) / rect.width()) * whole;
                if right { end = value } else { start = value }
            }
            if background.drag_stopped() {
                self.range_drag = None;
            }
            let x = |t: f64| rect.left() + rect.width() * (t / whole) as f32;
            ui.painter().rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x(start), rect.top() + 3.),
                    egui::pos2(x(end), rect.bottom() - 3.),
                ),
                2,
                helto::ACCENT_TOP,
            );
            for right in [false, true] {
                let at = x(if right { end } else { start });
                let handle = egui::Rect::from_center_size(
                    egui::pos2(at, rect.center().y),
                    egui::vec2(9., 18.),
                );
                let r = ui
                    .interact(
                        handle,
                        ui.id().with(("timeline-range", right)),
                        egui::Sense::drag(),
                    )
                    .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                #[cfg(feature = "ui-preview")]
                super::qa::control(
                    ui,
                    format!(
                        "timeline-range-handle-{}",
                        if right { "end" } else { "start" }
                    ),
                    &r,
                );
                ui.painter().rect_filled(handle, 2, helto::DIM);
                if r.dragged()
                    && let Some(p) = r.interact_pointer_pos()
                {
                    let value = f64::from((p.x - rect.left()) / rect.width()) * whole;
                    if right {
                        end = value;
                    } else {
                        start = value;
                    }
                }
            }
        });
        if before != (start, end) {
            self.set(start, end, whole, self.width, zoom);
        }
    }
}
pub(super) fn section_label(project: &VideoProject, section: &Section, display: Display) -> String {
    match display {
        Display::Sections => {
            if section.prompt.is_empty() {
                "Section".into()
            } else {
                section.prompt.clone()
            }
        }
        Display::Prompts => {
            let prompt = section.prompt.trim();
            let global = project.global_prompt.trim();
            let Some(options) = project
                .creative
                .global_prompt
                .as_ref()
                .filter(|o| o.enabled && !global.is_empty())
            else {
                return if prompt.is_empty() {
                    "Section".into()
                } else {
                    prompt.into()
                };
            };
            if prompt.is_empty() {
                global.into()
            } else if options.position == "Suffix" {
                format!("{prompt}, {global}")
            } else {
                format!("{global}, {prompt}")
            }
        }
        Display::Media => {
            let Some(source) = project.creative.sections.get(&section.id) else {
                return "Section".into();
            };
            let asset = source.primary_media_asset();
            let label = asset
                .and_then(|id| project.assets.get(id))
                .map(|a| a.name.as_str())
                .filter(|n| !n.is_empty())
                .unwrap_or("Section");
            let count = source.images.len() + source.videos.len();
            if count > 1 {
                format!("{label} +{}", count - 1)
            } else {
                label.into()
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_matches_director_whole_seconds_and_keeps_native_pan_unforced() {
        let oracle: Value = serde_json::from_str(include_str!("navigation-director.json")).unwrap();
        for row in oracle["cases"].as_array().unwrap() {
            let range = clamp_range(
                row["whole"].as_f64().unwrap(),
                row["start"].as_f64().unwrap(),
                row["end"].as_f64().unwrap(),
            );
            assert_eq!(
                range,
                (
                    row["expected"]["start"].as_f64().unwrap(),
                    row["expected"]["end"].as_f64().unwrap()
                )
            );
        }
        assert_eq!(clamp_range(12., 2.5, 8.5), (3., 9.));
        assert_eq!(clamp_range(12., 11.9, 2.), (11., 12.));
        assert_eq!(clamp_range(0.25, f64::NAN, f64::INFINITY), (0., 1.));
        let mut state = State::default();
        let mut zoom = 68.;
        state.prepare(12., 600., &mut zoom);
        assert_eq!(zoom, 50.);
        state.set(3., 7., 12., 600., &mut zoom);
        assert_eq!(state.scroll_to.take(), Some(450.));
        state.observe(600., 600., zoom, 12.);
        assert_eq!(state.view, Some((4., 8.)));
        state.prepare(12., 600., &mut zoom);
        assert!(state.scroll_to.is_none());
    }
    #[test]
    fn prompt_display_preserves_prefix_suffix_and_disabled_text() {
        let mut p = super::super::media_picker::tests::project();
        p.scenes[0].sections[0].prompt = "  Own Section  ".into();
        p.global_prompt = "  Own global  ".into();
        let section = p.scenes[0].sections[0].clone();
        p.creative.global_prompt = Some(imagine_video_contract::creative::GlobalPromptOptions {
            enabled: true,
            position: "Suffix".into(),
        });
        assert_eq!(
            section_label(&p, &section, Display::Prompts),
            "Own Section, Own global"
        );
        p.creative.global_prompt.as_mut().unwrap().enabled = false;
        assert_eq!(section_label(&p, &section, Display::Prompts), "Own Section");
        assert_eq!(
            section_label(&p, &section, Display::Sections),
            "  Own Section  "
        );
    }
}
