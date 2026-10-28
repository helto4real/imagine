//! The generic paint tool over the image desk: red marks on a MOSS edit reference
//! (a marked copy is sent, the original kept) and region-map editing for MOSS
//! control. Pixels come from the server's asset and go back as a new PNG asset.
use super::*;
use crate::icons::Icon;
use crate::paint_canvas::{Canvas, PaintKind, Pixels, Tool};
use egui::{Align, Color32, Layout, Rect, RichText, Stroke, Ui};
use std::sync::Arc;

/// Mark colors: red first, as edit instructions name it.
const MARK_COLORS: [([u8; 3], &str); 5] = [
    ([230, 40, 40], "Red"),
    ([31, 139, 255], "Blue"),
    ([32, 192, 96], "Green"),
    ([255, 212, 0], "Yellow"),
    ([255, 255, 255], "White"),
];

/// A reference's painted copy.
#[derive(Clone)]
pub(super) struct Marks {
    pub id: String,
    pub layer: Arc<Pixels>,
    pub texture: egui::TextureHandle,
    /// Whether the model gets the marked copy instead of the original.
    pub send: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) enum PaintTarget {
    /// The image desk reference with this (original) asset ID.
    Reference(String),
    RegionMap,
}
pub(super) struct Saved {
    id: String,
    preview: Preview,
    size: [u32; 2],
    layer: Arc<Pixels>,
}
pub(super) struct Painter {
    pub target: PaintTarget,
    canvas: Option<Canvas>,
    tool: Tool,
    /// Brush radius in canvas pixels.
    radius: f32,
    color: [u8; 3],
    opacity: f32,
    send_marked: bool,
    base_texture: Option<egui::TextureHandle>,
    layer_texture: Option<egui::TextureHandle>,
    layer_dirty: bool,
    last: Option<[f32; 2]>,
    regions: Vec<([u8; 3], f32)>,
    saving: bool,
    error: Option<String>,
}
impl Painter {
    fn new(target: PaintTarget) -> Self {
        Self {
            target,
            canvas: None,
            tool: Tool::Brush,
            radius: 20.,
            color: MARK_COLORS[0].0,
            opacity: 1.,
            send_marked: true,
            base_texture: None,
            layer_texture: None,
            layer_dirty: true,
            last: None,
            regions: Vec::new(),
            saving: false,
            error: None,
        }
    }
    fn open(&mut self, canvas: Canvas) {
        if canvas.kind == PaintKind::Regions {
            self.regions = canvas.region_list();
            self.color = self
                .regions
                .first()
                .map_or_else(|| canvas.new_region_color(), |r| r.0);
        }
        self.canvas = Some(canvas);
        self.layer_dirty = true;
    }
    fn kind(&self) -> Option<PaintKind> {
        self.canvas.as_ref().map(|c| c.kind)
    }
    fn paint_color(&self) -> [u8; 4] {
        match self.kind() {
            Some(PaintKind::Marks) => {
                let [r, g, b] = self.color;
                [r, g, b, (self.opacity * 255.).round() as u8]
            }
            _ => {
                let [r, g, b] = self.color;
                [r, g, b, 255]
            }
        }
    }
    /// Applies the tool at canvas position `at`; `from` continues a stroke.
    fn apply(&mut self, from: Option<[f32; 2]>, at: [f32; 2]) {
        let color = self.paint_color();
        let (tool, radius) = (self.tool, self.radius);
        let Some(canvas) = self.canvas.as_mut() else {
            return;
        };
        let cell = [at[0].max(0.) as usize, at[1].max(0.) as usize];
        match tool {
            Tool::Brush | Tool::Erase => {
                let color = if tool == Tool::Erase {
                    canvas.erase_color()
                } else {
                    color
                };
                canvas.line(from.unwrap_or(at), at, radius, color);
            }
            Tool::Fill => {
                if from.is_none() {
                    canvas.fill(cell, color);
                }
            }
            Tool::Pick => {
                if let Some(color) = canvas.pick(cell) {
                    self.color = color;
                }
                return;
            }
        }
        self.layer_dirty = true;
    }
    fn edited(&mut self) {
        self.layer_dirty = true;
        if let Some(canvas) = &self.canvas
            && canvas.kind == PaintKind::Regions
        {
            self.regions = canvas.region_list();
        }
    }
}
fn swatch(ui: &mut Ui, color: [u8; 3], size: f32, selected: bool, label: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::RadioButton, true, selected, label)
    });
    let [r, g, b] = color;
    ui.painter()
        .rect_filled(rect, 4, Color32::from_rgb(r, g, b));
    ui.painter().rect_stroke(
        rect,
        4,
        Stroke::new(
            if selected { 2. } else { 1. },
            if selected {
                helto::ACCENT
            } else {
                Color32::from_black_alpha(90)
            },
        ),
        egui::StrokeKind::Outside,
    );
    response.on_hover_text(label)
}
fn pixels_texture(ctx: &egui::Context, pixels: &Pixels, name: &str) -> egui::TextureHandle {
    ctx.load_texture(
        name,
        egui::ColorImage::from_rgba_unmultiplied([pixels.width, pixels.height], &pixels.bytes()),
        egui::TextureOptions::LINEAR,
    )
}

impl ImagineApp {
    /// Paints marks on the image desk reference at `index`.
    pub(super) fn paint_reference(&mut self, ctx: &egui::Context, index: usize) {
        let Some(reference) = self.image.references.get(index) else {
            return;
        };
        let id = reference.id.clone();
        self.load_paint(ctx, PaintTarget::Reference(id.clone()), id);
    }
    /// Edits the region map, or starts an empty one at the output size.
    pub(super) fn paint_regions(&mut self, ctx: &egui::Context) {
        if let Some(map) = &self.image.color_map {
            let id = map.id.clone();
            self.load_paint(ctx, PaintTarget::RegionMap, id);
            return;
        }
        let (w, h) = match self.prefs.generation.dimensions_for(self.moss_source()) {
            Ok(size) => size,
            Err(e) => {
                self.image.error = Some(e.to_string());
                return;
            }
        };
        let mut painter = Painter::new(PaintTarget::RegionMap);
        painter.open(Canvas::regions(
            Pixels::new(w as usize, h as usize, [0, 0, 0, 255]).fitted(),
            None,
        ));
        self.painter = Some(painter);
    }
    fn load_paint(&mut self, ctx: &egui::Context, target: PaintTarget, id: String) {
        let Some(client) = self.client.clone() else {
            self.image.error = Some("Not connected to Imagine.".into());
            return;
        };
        self.painter = Some(Painter::new(target.clone()));
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.image(&id)).map(|preview| {
                Pixels::from_rgba(preview.width, preview.height, &preview.pixels).fitted()
            });
            send(&tx, &ctx, Message::PaintLoaded(target, result));
        });
    }
    pub(super) fn paint_loaded(&mut self, target: PaintTarget, result: Result<Pixels, Failure>) {
        let Some(painter) = self.painter.as_mut().filter(|p| p.target == target) else {
            return;
        };
        let pixels = match result {
            Ok(pixels) => pixels,
            Err(e) => {
                self.painter = None;
                self.image.error = Some(e.message);
                return;
            }
        };
        let canvas = match &target {
            PaintTarget::Reference(id) => {
                let reference = self.image.references.iter().find(|r| &r.id == id);
                let marks = reference.and_then(|r| r.marks.as_ref());
                painter.send_marked = marks.is_none_or(|m| m.send);
                Canvas::marks(pixels, marks.map(|m| (*m.layer).clone()))
            }
            PaintTarget::RegionMap => Canvas::regions(pixels, None),
        };
        painter.open(canvas);
    }
    /// Sends the painted copy or map to the server and closes the painter once it
    /// is stored.
    fn finish_painting(&mut self, ctx: &egui::Context) {
        let Some(painter) = self.painter.as_mut() else {
            return;
        };
        let Some(canvas) = &painter.canvas else {
            return;
        };
        // Marks cleared to nothing return the reference to its original.
        if canvas.kind == PaintKind::Marks && !canvas.has_marks() {
            if let PaintTarget::Reference(id) = &painter.target
                && let Some(reference) = self.image.references.iter_mut().find(|r| &r.id == id)
            {
                reference.marks = None;
            }
            self.painter = None;
            return;
        }
        let Some(client) = self.client.clone() else {
            painter.error = Some("Not connected to Imagine.".into());
            return;
        };
        painter.saving = true;
        painter.error = None;
        let image = canvas.composite();
        let layer = Arc::new(canvas.layer.clone());
        let target = painter.target.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| {
                let png = image.png().map_err(|e| Failure::terminal(e.to_string()))?;
                let preview =
                    client::decode(&png, true).map_err(|e| Failure::terminal(e.to_string()))?;
                let id = runtime().block_on(client.upload_png(png))?;
                Ok(Saved {
                    id,
                    preview,
                    size: [image.width as u32, image.height as u32],
                    layer,
                })
            })();
            send(&tx, &ctx, Message::PaintSaved(target, result));
        });
    }
    pub(super) fn paint_saved(
        &mut self,
        ctx: &egui::Context,
        target: PaintTarget,
        result: Result<Saved, Failure>,
    ) {
        let Some(painter) = self.painter.as_mut().filter(|p| p.target == target) else {
            return;
        };
        let saved = match result {
            Ok(saved) => saved,
            Err(e) => {
                painter.saving = false;
                painter.error = Some(e.message);
                return;
            }
        };
        let send_marked = painter.send_marked;
        let texture = texture(ctx, &saved.preview, &saved.id);
        match target {
            PaintTarget::Reference(id) => {
                if let Some(reference) = self.image.references.iter_mut().find(|r| r.id == id) {
                    reference.marks = Some(Marks {
                        id: saved.id,
                        layer: saved.layer,
                        texture,
                        send: send_marked,
                    });
                }
            }
            PaintTarget::RegionMap => {
                self.image.color_map = Some(ReferenceView {
                    number: 0,
                    name: "region-map.png".into(),
                    id: saved.id,
                    texture,
                    size: saved.size,
                    marks: None,
                });
            }
        }
        self.painter = None;
    }

    /// The painter in place of the image desk's panes.
    pub(super) fn paint_workspace(&mut self, ui: &mut Ui, rect: Rect) {
        let wide = rect.width() >= 900.;
        let (canvas, controls) = if wide {
            let width = 320.;
            (
                Rect::from_min_max(
                    rect.min,
                    egui::pos2(rect.right() - width - 8., rect.bottom()),
                ),
                Rect::from_min_max(egui::pos2(rect.right() - width, rect.top()), rect.max),
            )
        } else {
            let split = rect.top() + rect.height() * 0.62;
            (
                Rect::from_min_max(rect.min, egui::pos2(rect.right(), split)),
                Rect::from_min_max(egui::pos2(rect.left(), split + 8.), rect.max),
            )
        };
        super::workspace::panel(ui, canvas, true);
        super::workspace::panel(ui, controls, false);
        helto::canvas(ui, "paint-canvas", canvas, |ui| {
            super::workspace::region(ui, "paint-canvas", canvas.shrink(10.), |ui| {
                self.paint_toolbar(ui);
                ui.add_space(6.);
                self.paint_view(ui);
            })
        });
        helto::canvas(ui, "paint-controls", controls, |ui| {
            super::workspace::region(ui, "paint-controls", controls.shrink(12.), |ui| {
                self.paint_controls(ui)
            })
        });
    }
    fn paint_toolbar(&mut self, ui: &mut Ui) {
        let Some(painter) = self.painter.as_mut() else {
            return;
        };
        let Some(kind) = painter.kind() else {
            return;
        };
        let busy = painter.saving;
        ui.horizontal(|ui| {
            let tools: &[(Tool, &str)] = match kind {
                PaintKind::Marks => &[
                    (Tool::Brush, "Brush"),
                    (Tool::Fill, "Fill"),
                    (Tool::Erase, "Erase"),
                ],
                PaintKind::Regions => &[
                    (Tool::Brush, "Brush"),
                    (Tool::Fill, "Fill"),
                    (Tool::Erase, "Erase"),
                    (Tool::Pick, "Pick"),
                ],
            };
            for &(tool, label) in tools {
                if ui
                    .add_enabled_ui(!busy, |ui| {
                        helto::select_button(ui, label, painter.tool == tool)
                    })
                    .inner
                    .clicked()
                {
                    painter.tool = tool;
                }
            }
            ui.separator();
            let label = ui.label("Size");
            ui.add(
                egui::Slider::new(&mut painter.radius, 1.0..=200.0)
                    .logarithmic(true)
                    .suffix(" px"),
            )
            .labelled_by(label.id);
            if kind == PaintKind::Marks {
                let label = ui.label("Opacity");
                ui.add(egui::Slider::new(&mut painter.opacity, 0.2..=1.0).fixed_decimals(2))
                    .labelled_by(label.id);
            }
            ui.separator();
            let Some(canvas) = painter.canvas.as_mut() else {
                return;
            };
            let undo = ui
                .add_enabled_ui(!busy && canvas.can_undo(), |ui| {
                    helto::icon_button(ui, Icon::Undo, "Undo", false)
                })
                .inner
                .clicked();
            let redo = ui
                .add_enabled_ui(!busy && canvas.can_redo(), |ui| {
                    helto::icon_button(ui, Icon::Redo, "Redo", false)
                })
                .inner
                .clicked();
            if (undo && canvas.undo()) | (redo && canvas.redo()) {
                painter.edited();
            }
        });
    }
    fn paint_view(&mut self, ui: &mut Ui) {
        let privacy = self.prefs.privacy_mode;
        let Some(painter) = self.painter.as_mut() else {
            return;
        };
        let Some(canvas) = painter.canvas.as_mut() else {
            ui.spinner();
            return;
        };
        let ctx = ui.ctx().clone();
        if painter.base_texture.is_none()
            && let Some(base) = &canvas.base
        {
            painter.base_texture = Some(pixels_texture(&ctx, base, "paint-base"));
        }
        if painter.layer_dirty {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [canvas.layer.width, canvas.layer.height],
                &canvas.layer.bytes(),
            );
            match &mut painter.layer_texture {
                Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                None => {
                    painter.layer_texture =
                        Some(ctx.load_texture("paint-layer", image, egui::TextureOptions::LINEAR))
                }
            }
            painter.layer_dirty = false;
        }
        let [w, h] = canvas.size();
        let area = ui.available_rect_before_wrap();
        let scale = (area.width() / w as f32).min(area.height() / h as f32);
        let rect = Rect::from_center_size(area.center(), egui::vec2(w as f32, h as f32) * scale);
        let shown = helto::shown(ui, privacy);
        let response = ui.allocate_rect(
            rect,
            if shown && !painter.saving {
                egui::Sense::click_and_drag()
            } else {
                egui::Sense::hover()
            },
        );
        let full = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.));
        if !shown {
            ui.painter().rect_filled(rect, 4, helto::MEDIA);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Hidden by Hide previews",
                egui::FontId::proportional(14.),
                helto::DIM,
            );
            return;
        }
        ui.painter().rect_filled(rect, 0, Color32::BLACK);
        if let Some(base) = &painter.base_texture {
            ui.painter().image(base.id(), rect, full, Color32::WHITE);
        }
        if let Some(layer) = &painter.layer_texture {
            ui.painter().image(layer.id(), rect, full, Color32::WHITE);
        }
        let to_canvas = |pos: egui::Pos2| {
            let p = (pos - rect.min) / scale;
            [p.x, p.y]
        };
        if response.drag_started()
            && let Some(pos) = response.interact_pointer_pos()
        {
            canvas.begin();
            let at = to_canvas(pos);
            painter.apply(None, at);
            painter.last = Some(at);
        } else if response.dragged()
            && let (Some(pos), Some(last)) = (response.interact_pointer_pos(), painter.last)
        {
            let at = to_canvas(pos);
            painter.apply(Some(last), at);
            painter.last = Some(at);
        } else if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            if painter.tool != Tool::Pick {
                canvas.begin();
            }
            painter.apply(None, to_canvas(pos));
            painter.edited();
        }
        if response.drag_stopped() {
            painter.last = None;
            painter.edited();
        }
        if let Some(pos) = response.hover_pos()
            && matches!(painter.tool, Tool::Brush | Tool::Erase)
        {
            ui.painter().circle_stroke(
                pos,
                painter.radius * scale,
                Stroke::new(1.5, helto::ACCENT_TEXT),
            );
        }
    }
    fn paint_controls(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let title = match self.painter.as_ref().map(|p| &p.target) {
            Some(PaintTarget::Reference(id)) => {
                let number = self.image.references.iter().position(|r| &r.id == id);
                format!("Paint · Reference {}", number.map_or(0, |n| n + 1))
            }
            _ => "Paint · Region map".into(),
        };
        let Some(painter) = self.painter.as_mut() else {
            return;
        };
        let busy = painter.saving || painter.canvas.is_none();
        let mut done = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let blocked = busy.then_some("Wait for the image.");
                done = helto::action(ui, Icon::Check, "Done", true, blocked).clicked();
                cancel = ui
                    .add_enabled_ui(!painter.saving, |ui| {
                        helto::icon_button(ui, Icon::Close, "Discard the paint", false)
                    })
                    .inner
                    .clicked();
            });
        });
        ui.separator();
        if painter.saving {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Saving");
            });
        }
        if let Some(error) = &painter.error {
            ui.colored_label(helto::ERROR, error);
        }
        match painter.kind() {
            Some(PaintKind::Marks) => {
                ui.label(helto::section("Color"));
                ui.horizontal(|ui| {
                    for (color, name) in MARK_COLORS {
                        if swatch(ui, color, 24., painter.color == color, name).clicked() {
                            painter.color = color;
                        }
                    }
                });
                ui.add_space(8.);
                ui.horizontal(|ui| {
                    ui.label("Send with the marks");
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        helto::toggle(ui, &mut painter.send_marked, "Send with the marks");
                    });
                });
                ui.label(
                    RichText::new(if painter.send_marked {
                        "The original stays unchanged."
                    } else {
                        "The model gets the original; the marks are kept."
                    })
                    .color(helto::DIM),
                );
                ui.add_space(8.);
                let clear = ui
                    .add_enabled(!busy, egui::Button::new("Clear paint"))
                    .clicked();
                if clear && let Some(canvas) = painter.canvas.as_mut() {
                    canvas.begin();
                    canvas.clear();
                    painter.edited();
                }
            }
            Some(PaintKind::Regions) => {
                ui.label(helto::section(&format!(
                    "Regions · {}",
                    painter.regions.len()
                )));
                let mut action = None;
                egui::ScrollArea::vertical()
                    .id_salt("paint-regions")
                    .max_height((ui.available_height() - 60.).max(120.))
                    .show(ui, |ui| {
                        for (index, &(color, share)) in painter.regions.iter().enumerate() {
                            let selected = painter.color == color;
                            ui.horizontal(|ui| {
                                if swatch(
                                    ui,
                                    color,
                                    18.,
                                    selected,
                                    &format!("Region {}", index + 1),
                                )
                                .clicked()
                                {
                                    painter.color = color;
                                }
                                let label =
                                    ui.selectable_label(selected, format!("Region {}", index + 1));
                                if label.clicked() {
                                    painter.color = color;
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui
                                        .add_enabled_ui(!busy, |ui| {
                                            helto::icon_button(
                                                ui,
                                                Icon::Trash,
                                                &format!("Remove region {}", index + 1),
                                                false,
                                            )
                                        })
                                        .inner
                                        .clicked()
                                    {
                                        action = Some((color, None));
                                    }
                                    if !selected
                                        && ui
                                            .add_enabled_ui(!busy, |ui| {
                                                helto::icon_button(
                                                    ui,
                                                    Icon::Link,
                                                    &format!(
                                                        "Merge region {} into the selected one",
                                                        index + 1
                                                    ),
                                                    false,
                                                )
                                            })
                                            .inner
                                            .clicked()
                                    {
                                        action = Some((color, Some(painter.color)));
                                    }
                                    ui.label(
                                        RichText::new(format!("{:.0}%", share * 100.))
                                            .color(helto::DIM),
                                    );
                                });
                            });
                        }
                    });
                if let (Some((from, to)), Some(canvas)) = (action, painter.canvas.as_mut()) {
                    canvas.begin();
                    let to = to.map_or(canvas.erase_color(), |[r, g, b]| [r, g, b, 255]);
                    canvas.recolor(from, to);
                    painter.edited();
                }
                if ui
                    .add_enabled_ui(!busy, |ui| {
                        helto::icon_button(ui, Icon::Plus, "New region", false)
                    })
                    .inner
                    .clicked()
                    && let Some(canvas) = &painter.canvas
                {
                    painter.color = canvas.new_region_color();
                    painter.tool = Tool::Brush;
                }
            }
            None => {
                ui.spinner();
            }
        }
        if cancel {
            self.painter = None;
        } else if done {
            self.finish_painting(&ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::MossMode;

    fn frame(app: &mut ImagineApp, ctx: &egui::Context, name: &str) {
        let mut combined = egui::FullOutput::default();
        for _ in 0..3 {
            combined.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
                    )),
                    ..Default::default()
                },
                |ui| app.content(ui),
            ));
        }
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            super::super::tests::render(ctx, combined, 1280, 800)
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        } else {
            combined.textures_delta.clear();
        }
    }

    #[test]
    fn region_maps_start_empty_at_the_output_size_and_paint_regions() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.metadata =
            Some(imagine_server::contract::model(imagine_server::contract::MOSS).unwrap());
        app.prefs.generation.moss_mode = MossMode::Control;
        app.prefs.generation.max_side = 1024;
        app.paint_regions(&ctx);
        let painter = app.painter.as_mut().unwrap();
        assert_eq!(painter.target, PaintTarget::RegionMap);
        assert_eq!(painter.canvas.as_ref().unwrap().size(), [1024, 576]);
        for (color, from, to) in [
            ([230, 25, 75], [0., 0.], [1024., 300.]),
            ([60, 180, 75], [600., 350.], [1000., 560.]),
        ] {
            painter.color = color;
            painter.radius = 120.;
            painter.canvas.as_mut().unwrap().begin();
            painter.apply(None, from);
            painter.apply(Some(from), to);
            painter.edited();
        }
        assert_eq!(painter.regions.len(), 2);
        painter.tool = Tool::Pick;
        painter.apply(None, [1000., 560.]);
        assert_eq!(painter.color, [60, 180, 75]);
        frame(&mut app, &ctx, "paint-regions");
        // Discarding keeps the desk's maps as they were.
        app.painter = None;
        assert!(app.image.color_map.is_none());
    }

    #[test]
    fn marks_become_a_sent_copy_and_the_original_stays() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let original = app.image.references[0].id.clone();
        let target = PaintTarget::Reference(original.clone());
        app.painter = Some(Painter::new(target.clone()));
        let mut base = Pixels::new(480, 320, [90, 110, 140, 255]);
        for y in 120..260 {
            for x in 60..250 {
                base.rgba[y * 480 + x] = [200, 170, 120, 255];
            }
        }
        app.paint_loaded(target.clone(), Ok(base));
        let painter = app.painter.as_mut().unwrap();
        painter.tool = Tool::Fill;
        painter.canvas.as_mut().unwrap().begin();
        painter.apply(None, [100., 200.]);
        painter.edited();
        assert!(painter.canvas.as_ref().unwrap().has_marks());
        frame(&mut app, &ctx, "paint-marks");
        let layer = Arc::new(
            app.painter
                .as_ref()
                .unwrap()
                .canvas
                .as_ref()
                .unwrap()
                .layer
                .clone(),
        );
        app.paint_saved(
            &ctx,
            target.clone(),
            Ok(Saved {
                id: "marked-copy".into(),
                preview: Preview {
                    width: 1,
                    height: 1,
                    pixels: vec![230, 40, 40, 255],
                },
                size: [480, 320],
                layer,
            }),
        );
        assert!(app.painter.is_none());
        let reference = &app.image.references[0];
        assert_eq!(reference.id, original);
        assert_eq!(reference.sent_id(), "marked-copy");
        // Reopening continues the marks; sending the original keeps them for later.
        app.painter = Some(Painter::new(target.clone()));
        app.paint_loaded(target, Ok(Pixels::new(480, 320, [0, 0, 0, 255])));
        let painter = app.painter.as_ref().unwrap();
        assert!(painter.canvas.as_ref().unwrap().has_marks());
        app.image.references[0].marks.as_mut().unwrap().send = false;
        assert_eq!(app.image.references[0].sent_id(), original);
    }
}
