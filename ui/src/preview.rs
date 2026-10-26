//! Explicit synthetic fixture, excluded from the normal application build.
use super::*;

impl ImagineApp {
    /// Uses only a caller-owned temporary directory and procedural pixels.
    /// Port 1 is deliberately not an Imagine endpoint; this never connects at startup.
    pub fn synthetic_preview(ctx: &egui::Context, project: PathBuf) -> Self {
        helto::install(ctx);
        let connection = Connection {
            url: "http://127.0.0.1:1".into(),
        };
        let mut app = Self::empty(
            project.clone(),
            connection,
            Preferences {
                prompt_catalog_folder: None,
                preferred_prompt_profile: None,
                model_group: crate::model_catalog::QWEN_GROUP.into(),
                model_groups: Default::default(),
                output_folder: project.join("Pictures/Imagine"),
                privacy_mode: false,
                generation: Default::default(),
                sheet_width: 1536,
                upscale_target: Default::default(),
                layout: Default::default(),
                media_zoom: Default::default(),
            },
        );
        app.client = Some(Client::connect(&app.connection).unwrap());
        app.image.prompt = "A quiet mountain lake at dawn".into();
        let sample = synthetic_landscape();
        for index in 1..=2 {
            app.image.references.push(ReferenceView {
                number: 0,
                id: format!("synthetic-reference-{index}"),
                name: format!("reference-{index}.png"),
                texture: texture(ctx, &sample, &format!("synthetic-reference-{index}")),
            });
        }
        app.image.result = Some(Results {
            assets: vec![ResultView {
                asset_id: "asset_synthetic".into(),
                role: OutputRole::Image,
                texture: texture(ctx, &sample, "synthetic-result"),
                file: project.join("Pictures/Imagine/imagine-example.png"),
            }],
            seed: Some(284731),
        });
        app.image.status = "Saved".into();
        app.image.last_elapsed = Some(Duration::from_secs_f64(4.2));
        app
    }
}

fn synthetic_landscape() -> Preview {
    let (width, height) = (1600, 896);
    let mut pixels = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let reflected = if ny > 0.58 { 1.16 - ny } else { ny };
            let ridge = 0.16 + 0.2 * (nx * 14.).sin().abs() + 0.045 * (nx * 45.).sin();
            let mut rgb = if reflected > ridge {
                [
                    45. + reflected * 30.,
                    60. + reflected * 30.,
                    82. + reflected * 30.,
                ]
            } else {
                [
                    115. + reflected * 210.,
                    135. + reflected * 125.,
                    170. + reflected * 30.,
                ]
            };
            if ny > 0.58 {
                let wave = ((nx * 80. + ny * 130.).sin() * 0.03) + 0.77;
                for c in &mut rgb {
                    *c *= wave;
                }
            }
            pixels.extend(rgb.map(|c| c as u8));
            pixels.push(255);
        }
    }
    Preview {
        width,
        height,
        pixels,
    }
}
