//! Synthetic, headless layout evidence. No window, desktop capture or GPU.
use super::*;
use std::collections::HashMap;

#[test]
fn actual_pixels_respects_display_scale_clipping_and_privacy() {
    for pixels_per_point in [1., 2.] {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(pixels_per_point);
        let image = texture(
            &ctx,
            &Preview {
                width: 800,
                height: 600,
                pixels: [255, 0, 0, 255].repeat(800 * 600),
            },
            "zoom-test",
        );
        for (actual, hidden) in [(false, false), (true, false), (true, true)] {
            let mut pan = egui::vec2(10_000., -10_000.);
            let mut output = ctx.run_ui(
                egui::RawInput {
                    focused: false,
                    ..Default::default()
                },
                |ui| {
                    image_view(ui, &image, egui::vec2(200., 150.), hidden, actual, &mut pan);
                },
            );
            output.textures_delta.clear();
            let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
            let rendered = meshes.iter().find_map(|item| match &item.primitive {
                egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == image.id() => {
                    Some((item.clip_rect, mesh))
                }
                _ => None,
            });
            if hidden {
                assert!(rendered.is_none());
            } else {
                let (clip, mesh) = rendered.unwrap();
                let bounds = egui::Rect::from_points(
                    &mesh.vertices.iter().map(|v| v.pos).collect::<Vec<_>>(),
                );
                let expected = if actual {
                    800. / pixels_per_point
                } else {
                    200.
                };
                assert!((bounds.width() - expected).abs() < 0.01);
                assert!(clip.width() <= 200. && clip.height() <= 150.);
                if actual {
                    assert_eq!(
                        pan,
                        egui::vec2(
                            (expected - 200.) / 2.,
                            -(600. / pixels_per_point - 150.) / 2.
                        )
                    );
                }
            }
        }
    }
}

#[test]
fn privacy_off_shows_image_without_hover_or_focus_and_can_be_reenabled() {
    let ctx = egui::Context::default();
    let image = texture(
        &ctx,
        &Preview {
            width: 2,
            height: 2,
            pixels: [255, 0, 0, 255].repeat(4),
        },
        "privacy-toggle-test",
    );
    for privacy_mode in [false, true, false] {
        let mut output = ctx.run_ui(
            egui::RawInput {
                focused: false,
                events: vec![egui::Event::PointerGone],
                ..Default::default()
            },
            |ui| {
                private_image(ui, &image, egui::vec2(64., 48.), privacy_mode);
            },
        );
        output.textures_delta.clear();
        let draws_image = ctx.tessellate(output.shapes, output.pixels_per_point).iter().any(|mesh| {
            matches!(&mesh.primitive, egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == image.id())
        });
        assert_eq!(draws_image, !privacy_mode);
    }
}

#[test]
fn preferences_survive_folder_change_and_app_restart() {
    let root = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let connection = Connection {
        url: "http://127.0.0.1:8080".into(),
        access: root.path().join("unused.json"),
        client: "test".into(),
    };
    let mut app = ImagineApp::empty(
        root.path().into(),
        connection.clone(),
        Preferences::load(root.path()).unwrap(),
    );
    app.prefs.privacy_mode = false;
    app.prefs.negative_prompt = "  artifacts\nextra fingrar  ".into();
    app.prefs.generation = crate::settings::GenerationSettings {
        steps: 25,
        cfg_scale: 3.5,
        random_seed: false,
        seed: 0,
        aspect_ratio: "9:16".into(),
        max_side: 1024,
    };
    app.save_preferences();
    let folder = root.path().join("selected-images");
    app.tx.send(Message::Folder(Some(folder.clone()))).unwrap();
    app.receive(&ctx);
    assert!(app.error.is_none());
    let restarted = ImagineApp::empty(
        root.path().into(),
        connection,
        Preferences::load(root.path()).unwrap(),
    );
    assert_eq!(restarted.prefs, app.prefs);
    assert_eq!(restarted.prefs.output_folder, folder);
    assert!(!restarted.prefs.privacy_mode);
    assert_eq!(
        restarted.prefs.negative_prompt,
        "  artifacts\nextra fingrar  "
    );
}

#[test]
fn privacy_reveals_only_hovered_images_in_a_focused_window() {
    for size in [egui::vec2(64., 48.), egui::vec2(320., 180.)] {
        let ctx = egui::Context::default();
        let sample = Preview {
            width: 2,
            height: 2,
            pixels: [255, 0, 0, 255].repeat(4),
        };
        let image = texture(&ctx, &sample, "private-test");
        let other = texture(&ctx, &sample, "other-private-test");
        let inside = egui::pos2(24., 24.);
        let outside = egui::pos2(600., 400.);
        for (name, events, focused, clipped, expected) in [
            ("initial", vec![], true, false, false),
            (
                "enter",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            (
                "leave",
                vec![egui::Event::PointerMoved(outside)],
                true,
                false,
                false,
            ),
            (
                "re-enter",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            (
                "leave window",
                vec![egui::Event::PointerGone],
                true,
                false,
                false,
            ),
            (
                "return",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            ("lose focus", vec![], false, false, false),
            ("regain focus", vec![], true, false, true),
            ("clipped area", vec![], true, true, false),
        ] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800., 600.),
                    )),
                    focused,
                    events,
                    ..Default::default()
                },
                |ui| {
                    if clipped {
                        ui.set_clip_rect(egui::Rect::from_min_max(
                            egui::pos2(100., 100.),
                            egui::pos2(800., 600.),
                        ));
                    }
                    private_image(ui, &image, size, true);
                    private_image(ui, &other, size, true);
                },
            );
            output.textures_delta.clear();
            let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
            let draws_texture = |id| {
                meshes.iter().any(|mesh| {
                    matches!(&mesh.primitive, egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == id)
                })
            };
            assert_eq!(draws_texture(image.id()), expected, "{name} at {size:?}");
            assert!(!draws_texture(other.id()), "neighbor revealed: {name}");
        }
    }
}

#[test]
fn headless_layout_and_recoverable_job_state() {
    let root = tempfile::tempdir().unwrap();
    for (width, height, name) in [
        (1180, 760, "desktop"),
        (640, 900, "compact"),
        (1180, 760, "progress"),
        (640, 900, "progress-compact"),
    ] {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let mut app = ImagineApp::empty(
            root.path().into(),
            Connection {
                url: "http://127.0.0.1:8080".into(),
                access: root.path().join("unused.json"),
                client: "test".into(),
            },
            Preferences {
                output_folder: PathBuf::from("/Pictures/Imagine"),
                ..Preferences::load(root.path()).unwrap()
            },
        );
        std::fs::write(
            &app.connection.access,
            br#"{"clients":{"test":"synthetic-preview-token"}}"#,
        )
        .unwrap();
        app.client = Some(Client::connect(&app.connection).unwrap());
        app.prompt="A quiet Scandinavian harbor at sunrise, warm light on the waterfront, realistic photography.".into();
        app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
        let sample = Preview {
            width: 160,
            height: 90,
            pixels: (0..160 * 90)
                .flat_map(|i| {
                    let y = i / 160;
                    [60 + y as u8, 70 + y as u8, 92 + y as u8, 255]
                })
                .collect(),
        };
        app.references.push(ReferenceView {
            id: "synthetic-reference".into(),
            name: "reference.png".into(),
            texture: texture(&ctx, &sample, "synthetic-reference"),
        });
        app.result = Some((
            texture(&ctx, &sample, "synthetic-result"),
            PathBuf::from("/Pictures/Imagine/imagine-example.png"),
            Some(42),
        ));
        app.status = "Saved".into();
        if name.starts_with("progress") {
            app.working = true;
            app.status = "Sampling".into();
            app.sampling = Some(client::SamplingProgress {
                completed: 12,
                total: 40,
                remaining_seconds: 25.,
                updated_at_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            });
        }
        let mut combined = egui::FullOutput::default();
        for _ in 0..3 {
            let frame = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            combined.append(frame);
        }
        assert!(!combined.shapes.is_empty());
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            render(&ctx, combined, width, height)
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        } else {
            combined.textures_delta.clear();
        }
        // A retrieval/save failure keeps the original job and idempotency key.
        app.pending = Some(Submission {
            request: serde_json::json!({}),
            key: "original-key".into(),
            folder: root.path().into(),
            job: Some("job_original".into()),
        });
        app.working = true;
        app.tx
            .send(Message::Finished(Err(Failure {
                message: "disk full".into(),
                retryable: true,
            })))
            .unwrap();
        app.receive(&ctx);
        assert!(!app.working);
        assert!(app.sampling.is_none());
        assert_eq!(
            app.pending.as_ref().unwrap().job.as_deref(),
            Some("job_original")
        );
        app.tx
            .send(Message::Finished(Err(Failure {
                message: "cancelled".into(),
                retryable: false,
            })))
            .unwrap();
        app.receive(&ctx);
        assert!(app.pending.is_none());
    }
}

// Minimal CPU rasterization of egui's own triangle output for inspection.
// This is not a substitute for native compositor/font-renderer acceptance.
fn render(
    ctx: &egui::Context,
    mut output: egui::FullOutput,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut textures: HashMap<egui::TextureId, egui::ColorImage> = HashMap::new();
    for (id, deltas) in std::mem::take(&mut output.textures_delta.set) {
        for delta in deltas {
            let egui::ImageData::Color(data) = delta.image;
            if let Some([x, y]) = delta.pos {
                let target = textures.get_mut(&id).unwrap();
                for row in 0..data.size[1] {
                    for col in 0..data.size[0] {
                        target.pixels[(y + row) * target.size[0] + x + col] =
                            data.pixels[row * data.size[0] + col];
                    }
                }
            } else {
                textures.insert(id, (*data).clone());
            }
        }
    }
    let mut image =
        image::RgbaImage::from_pixel(width, height, image::Rgba(helto::SURFACE.to_array()));
    for primitive in ctx.tessellate(output.shapes, output.pixels_per_point) {
        let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
            continue;
        };
        let tex = &textures[&mesh.texture_id];
        for tri in mesh.indices.chunks_exact(3) {
            let v = [
                mesh.vertices[tri[0] as usize],
                mesh.vertices[tri[1] as usize],
                mesh.vertices[tri[2] as usize],
            ];
            let cross = |a: egui::Vec2, b: egui::Vec2| a.x * b.y - a.y * b.x;
            let area = cross(v[1].pos - v[0].pos, v[2].pos - v[0].pos);
            if area.abs() < 0.0001 {
                continue;
            }
            let bounds = egui::Rect::from_points(&v.map(|v| v.pos))
                .intersect(primitive.clip_rect)
                .intersect(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                ));
            for y in
                bounds.min.y.ceil().max(0.) as u32..bounds.max.y.ceil().min(height as f32) as u32
            {
                for x in
                    bounds.min.x.ceil().max(0.) as u32..bounds.max.x.ceil().min(width as f32) as u32
                {
                    let p = egui::pos2(x as f32 + 0.5, y as f32 + 0.5);
                    let w = [
                        cross(v[1].pos - p, v[2].pos - p) / area,
                        cross(v[2].pos - p, v[0].pos - p) / area,
                        cross(v[0].pos - p, v[1].pos - p) / area,
                    ];
                    if w.iter().any(|v| *v < -0.0001) {
                        continue;
                    }
                    let uv = v[0].uv.to_vec2() * w[0]
                        + v[1].uv.to_vec2() * w[1]
                        + v[2].uv.to_vec2() * w[2];
                    let tx = (uv.x * tex.size[0] as f32 - 0.5).clamp(0., (tex.size[0] - 1) as f32);
                    let ty = (uv.y * tex.size[1] as f32 - 0.5).clamp(0., (tex.size[1] - 1) as f32);
                    let ix = tx.floor() as usize;
                    let iy = ty.floor() as usize;
                    let fx = tx - ix as f32;
                    let fy = ty - iy as f32;
                    let samples = [
                        (ix, iy, (1. - fx) * (1. - fy)),
                        ((ix + 1).min(tex.size[0] - 1), iy, fx * (1. - fy)),
                        (ix, (iy + 1).min(tex.size[1] - 1), (1. - fx) * fy),
                        (
                            (ix + 1).min(tex.size[0] - 1),
                            (iy + 1).min(tex.size[1] - 1),
                            fx * fy,
                        ),
                    ];
                    let mut color = [0.; 4];
                    for (channel, value) in color.iter_mut().enumerate() {
                        let pixel: f32 = samples
                            .iter()
                            .map(|(x, y, k)| {
                                tex.pixels[y * tex.size[0] + x].to_array()[channel] as f32 * k
                            })
                            .sum();
                        let vertex: f32 = (0..3)
                            .map(|i| v[i].color.to_array()[channel] as f32 * w[i])
                            .sum();
                        *value = pixel * vertex / 255.;
                    }
                    let dst = image.get_pixel_mut(x, y);
                    let alpha = color[3] / 255.;
                    for channel in 0..3 {
                        dst[channel] = (color[channel] + dst[channel] as f32 * (1. - alpha))
                            .clamp(0., 255.) as u8;
                    }
                    dst[3] = 255;
                }
            }
        }
    }
    image
}
