//! A Section's block on the Director track: its first image as a filmstrip,
//! or the previous Scene's last frame when the Scene continues from it.
use super::*;
use egui::{Rect, Stroke};
use imagine_video_contract::creative::SectionMediaId;

/// What a Section block shows behind its label.
enum Source {
    /// Media bound to the Section, or the Scene's first frame for its first Section.
    Media {
        asset: Asset,
        role: &'static str,
    },
    /// The Scene continues from `scene`; `take` is that Scene's accepted Take.
    Continued {
        scene: usize,
        take: Option<TakeId>,
        frame: bool,
    },
    None,
}

fn role_label(role: &InputRole) -> &'static str {
    match role {
        InputRole::FirstFrame => "First frame",
        InputRole::LastFrame => "Last frame",
        InputRole::VideoReference => "Video",
        _ => "Reference",
    }
}

fn source(project: &VideoProject, scene: &Scene, section: &Section) -> Source {
    let media = project.creative.sections.get(&section.id).and_then(|s| {
        let (id, asset) = s.ordered_media().into_iter().next()?;
        let role = match &id {
            SectionMediaId::Image(image) => s
                .images
                .iter()
                .find(|i| &i.image_id == image)
                .map_or("Reference", |i| role_label(&i.role)),
            SectionMediaId::Video(_) => "Video",
        };
        Some((project.assets.get(asset)?.clone(), role))
    });
    if let Some((asset, role)) = media {
        return Source::Media { asset, role };
    }
    if scene.sections.first().map(|s| &s.id) != Some(&section.id) {
        return Source::None;
    }
    if let Some(binding) = scene
        .bindings
        .iter()
        .find(|b| b.enabled && b.role == InputRole::FirstFrame)
        && let Some(asset) = project.assets.get(&binding.asset)
    {
        return Source::Media {
            asset: asset.clone(),
            role: "First frame",
        };
    }
    let Some(continuity) = &scene.continuity else {
        return Source::None;
    };
    let Some(index) = project
        .scenes
        .iter()
        .position(|s| s.id == continuity.source_scene)
    else {
        return Source::None;
    };
    Source::Continued {
        scene: index + 1,
        take: project.scenes[index]
            .accepted_take
            .as_ref()
            .map(|t| t.id.clone()),
        frame: continuity.method == ContinuityMethod::LastFrameFirstFrame,
    }
}

impl ImagineApp {
    /// Paints the block's background and returns its role chip.
    pub(super) fn timeline_section_block(
        &mut self,
        ui: &egui::Ui,
        project: &VideoProject,
        scene: &Scene,
        section: &Section,
        r: Rect,
        revealed: bool,
    ) -> Option<tracks::Chip> {
        let painter = ui.painter().with_clip_rect(r.intersect(ui.clip_rect()));
        let (texture, chip) = match source(project, scene, section) {
            Source::Media { asset, role } => {
                let path = (revealed && asset.kind == MediaKind::Image)
                    .then(|| {
                        self.timeline
                            .document
                            .as_ref()?
                            .runtime_asset(&asset.id.0)
                            .map(|id| format!("/v1/assets/{id}"))
                    })
                    .flatten();
                let texture = path.and_then(|path| {
                    self.timeline_thumbnail_texture(ui.ctx(), &project.id, &asset, Some(path))?
                        .ok()
                });
                (texture, Some(tracks::Chip::plain(role)))
            }
            Source::Continued {
                scene: number,
                take: Some(take),
                frame,
            } => {
                // H3 keeps the accepted Take's last frame; other Takes show only the chip.
                let texture = (revealed && frame)
                    .then(|| {
                        let asset = Asset {
                            id: AssetId(format!("last-frame-{}", take.0)),
                            revision: 1,
                            kind: MediaKind::Image,
                            name: String::new(),
                            library_source: None,
                        };
                        let path = format!(
                            "/v1/video/projects/{}/takes/{}/last-frame",
                            project.id.0, take.0
                        );
                        self.timeline_thumbnail_texture(ui.ctx(), &project.id, &asset, Some(path))?
                            .ok()
                    })
                    .flatten();
                let text = if frame {
                    format!("Scene {number} last frame")
                } else {
                    format!("Continues Scene {number}")
                };
                (texture, Some(tracks::Chip::inherited(text)))
            }
            Source::Continued {
                scene: number,
                take: None,
                ..
            } => {
                stripes(&painter, r);
                (
                    None,
                    Some(tracks::Chip::waiting(format!(
                        "Waits for Scene {number} take"
                    ))),
                )
            }
            Source::None => {
                helto::gradient(
                    &painter,
                    r,
                    4.,
                    egui::Color32::from_rgb(0x23, 0x46, 0x55),
                    egui::Color32::from_rgb(0x24, 0x4a, 0x39),
                );
                (None, None)
            }
        };
        if let Some(texture) = &texture {
            filmstrip(&painter, r, texture);
        } else if chip.as_ref().is_some_and(|c| c.is_media()) {
            helto::gradient(&painter, r, 4., helto::RAISED, helto::FIELD);
        }
        // Keep the label readable over pictures.
        helto::gradient(
            &painter,
            Rect::from_min_max(r.min, egui::pos2(r.right(), r.top() + r.height() * 0.6)),
            4.,
            egui::Color32::from_black_alpha(200),
            egui::Color32::TRANSPARENT,
        );
        let selected = self.timeline.section_selection.contains(&section.id)
            || self.timeline.section.as_ref() == Some(&section.id);
        painter.rect_stroke(
            r,
            4,
            if selected {
                Stroke::new(2., helto::ACCENT)
            } else {
                Stroke::new(1., egui::Color32::from_white_alpha(50))
            },
            egui::StrokeKind::Inside,
        );
        chip
    }
}

/// `texture` repeated across `r` at its own aspect ratio, as in Director.
fn filmstrip(painter: &egui::Painter, r: Rect, texture: &egui::TextureHandle) {
    let size = texture.size_vec2();
    let tile = (r.height() * size.x / size.y.max(1.)).clamp(24., 400.);
    let mut x = r.left();
    while x < r.right() {
        let frame = Rect::from_min_size(egui::pos2(x, r.top()), egui::vec2(tile, r.height()));
        painter.image(
            texture.id(),
            frame,
            Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
            egui::Color32::WHITE,
        );
        painter.vline(
            x,
            r.y_range(),
            Stroke::new(2., egui::Color32::from_black_alpha(140)),
        );
        x += tile;
    }
}

/// A placeholder for a first frame that a previous Scene has not produced yet.
fn stripes(painter: &egui::Painter, r: Rect) {
    painter.rect_filled(r, 4, egui::Color32::from_rgb(0x16, 0x30, 0x3d));
    let stroke = Stroke::new(1., egui::Color32::from_rgb(0x3e, 0x64, 0x78));
    let mut x = r.left() - r.height();
    while x < r.right() {
        painter.line_segment(
            [
                egui::pos2(x, r.bottom()),
                egui::pos2(x + r.height(), r.top()),
            ],
            stroke,
        );
        x += 12.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_video_contract::creative::{ImageBinding, SectionSource};
    fn time(num: u64) -> Time {
        Time { num, den: 1 }
    }
    fn section(id: &str, start: u64, length: u64, prompt: &str) -> Section {
        Section {
            id: SectionId(id.into()),
            start: time(start),
            duration: time(length),
            prompt: prompt.into(),
        }
    }
    /// Three Scenes: an image-led first Scene with an accepted Take, a Scene
    /// continuing from its last frame, and one waiting for the second's Take.
    fn project() -> VideoProject {
        let mut p = super::super::media_picker::tests::project();
        p.delivery.width = 160;
        p.delivery.height = 90;
        let mut first = p.scenes[0].clone();
        first.id = SceneId("own-a".into());
        first.duration = time(4);
        first.sections = vec![
            section("a1", 0, 2, "Own boats return to the harbour at dusk"),
            section("a2", 2, 2, "@keeper steps onto the wet pier"),
        ];
        first.accepted_take = Some(AcceptedTake {
            id: TakeId("own-take".into()),
            model: None,
            source_fingerprint: "own".into(),
        });
        p.creative.sections.insert(
            SectionId("a1".into()),
            SectionSource {
                images: vec![ImageBinding {
                    anchor: None,
                    image_id: "i1".into(),
                    asset: AssetId("image".into()),
                    role: InputRole::FirstFrame,
                }],
                ..Default::default()
            },
        );
        let mut second = first.clone();
        second.id = SceneId("own-b".into());
        second.start = time(4);
        second.sections = vec![section("b1", 0, 4, "Lanterns light up along the quay")];
        second.continuity = Some(Continuity {
            source_scene: first.id.clone(),
            method: ContinuityMethod::LastFrameFirstFrame,
            include_audio: false,
        });
        second.accepted_take = None;
        let mut third = second.clone();
        third.id = SceneId("own-c".into());
        third.start = time(8);
        third.sections = vec![section("c1", 0, 4, "The lighthouse beam sweeps the bay")];
        third.continuity = Some(Continuity {
            source_scene: second.id.clone(),
            method: ContinuityMethod::LastFrameFirstFrame,
            include_audio: false,
        });
        p.scenes = vec![first, second, third];
        p
    }
    /// A procedural dusk picture: sky over sea.
    fn picture(ctx: &egui::Context, name: &str, sky: [u8; 3]) -> egui::TextureHandle {
        let (w, h) = (160, 90);
        let mut image = egui::ColorImage::filled([w, h], egui::Color32::BLACK);
        for y in 0..h {
            for x in 0..w {
                let t = y as f32 / h as f32;
                let c = if t < 0.55 {
                    let k = t / 0.55;
                    [
                        sky[0] as f32 * (1. - k) + 208. * k,
                        sky[1] as f32 * (1. - k) + 132. * k,
                        sky[2] as f32 * (1. - k) + 106. * k,
                    ]
                } else {
                    [32. - 10. * t, 58. - 20. * t, 82. - 20. * t]
                };
                let shimmer = if t > 0.55 && (x + y) % 17 == 0 {
                    30.
                } else {
                    0.
                };
                image[(x, y)] = egui::Color32::from_rgb(
                    (c[0] + shimmer) as u8,
                    (c[1] + shimmer) as u8,
                    (c[2] + shimmer) as u8,
                );
            }
        }
        ctx.load_texture(name, image, egui::TextureOptions::LINEAR)
    }
    #[test]
    fn sections_show_their_media_or_the_frame_they_continue_from() {
        let p = project();
        let shown = |scene: usize, section: usize| {
            source(&p, &p.scenes[scene], &p.scenes[scene].sections[section])
        };
        assert!(matches!(
            shown(0, 0),
            Source::Media {
                role: "First frame",
                ..
            }
        ));
        assert!(matches!(shown(0, 1), Source::None));
        assert!(matches!(
            shown(1, 0),
            Source::Continued {
                scene: 1,
                take: Some(_),
                frame: true
            }
        ));
        assert!(matches!(
            shown(2, 0),
            Source::Continued {
                scene: 2,
                take: None,
                ..
            }
        ));

        let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") else {
            return;
        };
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.client = None;
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.selected.insert(p.scenes[0].id.clone());
        app.timeline.section = Some(SectionId("a2".into()));
        app.timeline.zoom = 120.;
        for (asset, sky) in [
            ("image", [42, 35, 71]),
            ("last-frame-own-take", [60, 40, 80]),
        ] {
            let texture = picture(&ctx, asset, sky);
            app.timeline
                .thumbnails
                .insert((p.id.clone(), AssetId(asset.into())), 1, texture);
        }
        app.timeline.document = Some(Snapshot::with(
            p.clone(),
            json!({"runtime_assets":{"image":"own-runtime"},
                "takes":[{"id":"own-take","scene_id":"own-a","status":"accepted","provenance":null}]}),
        ));
        let mut output = egui::FullOutput::default();
        for _ in 0..3 {
            output.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600., 1000.),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            ));
        }
        crate::app::tests::render(&ctx, output, 1600, 1000)
            .save(std::path::PathBuf::from(dir).join("timeline-director.png"))
            .unwrap();
    }
}
