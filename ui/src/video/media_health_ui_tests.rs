//! Actual egui frames and CPU-rasterizable meshes. No server, native display or GPU.
use super::*;
use std::path::PathBuf;
fn evidence_root() -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("IMAGINE_WORKER_A_EVIDENCE").expect("explicit crypto UI evidence root"),
    )
    .canonicalize()
    .unwrap();
    assert!(super::task_fixtures::is_owned(&root));
    root
}
pub(in crate::app::timeline) fn capture(
    ctx: &egui::Context,
    name: &str,
    width: f32,
    height: f32,
    mut paint: impl FnMut(&mut egui::Ui),
) -> Value {
    ctx.enable_accesskit();
    ctx.all_styles_mut(|style| style.animation_time = 0.0);
    let root = evidence_root();
    let mut textures = vec![];
    let mut final_output = None;
    for frame in 0..3 {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, height),
                )),
                events: vec![egui::Event::PointerGone],
                focused: true,
                ..Default::default()
            },
            |ui| paint(ui),
        );
        for (index, (id, deltas)) in output.textures_delta.set.iter().enumerate() {
            for (part, delta) in deltas.iter().enumerate() {
                let egui::ImageData::Color(image) = &delta.image;
                let file = format!("{name}-texture-{frame}-{index}-{part}.rgba");
                std::fs::write(
                    root.join(&file),
                    image
                        .pixels
                        .iter()
                        .flat_map(|p| p.to_array())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                textures.push(
                    json!({"id":format!("{id:?}"),"file":file,"size":image.size,"pos":delta.pos}),
                );
            }
        }
        output.textures_delta.clear();
        final_output = Some(output);
    }
    let output = final_output.unwrap();
    let nodes=output.platform_output.accesskit_update.unwrap().nodes.into_iter().map(|(_,n)|json!({"label":n.label(),"value":n.value(),"disabled":n.is_disabled(),"bounds":n.bounds().map(|b|[b.x0,b.y0,b.x1,b.y1])})).collect::<Vec<_>>();
    let primitives=ctx.tessellate(output.shapes,output.pixels_per_point).into_iter().filter_map(|p|if let egui::epaint::Primitive::Mesh(mesh)=p.primitive {Some(json!({"clip":[p.clip_rect.min.x,p.clip_rect.min.y,p.clip_rect.max.x,p.clip_rect.max.y],"texture":format!("{:?}",mesh.texture_id),"vertices":mesh.vertices.iter().map(|v|json!([v.pos.x,v.pos.y,v.uv.x,v.uv.y,v.color.to_array()])).collect::<Vec<_>>(),"indices":mesh.indices}))}else{None}).collect::<Vec<_>>();
    let evidence = json!({"width":width,"height":height,"pixels_per_point":output.pixels_per_point,"textures":textures,"primitives":primitives,"nodes":nodes,"method":"Actual egui CPU frame/tessellation; software mesh rasterization, no native-window acceptance"});
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec(&evidence).unwrap(),
    )
    .unwrap();
    evidence
}
fn labels(evidence: &Value) -> Vec<&str> {
    evidence["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|n| [n["label"].as_str(), n["value"].as_str()])
        .flatten()
        .collect()
}
fn assert_controls(evidence: &Value, names: &[&str], width: f32) {
    for name in names {
        let node = evidence["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["label"] == *name && n["bounds"].is_array())
            .unwrap_or_else(|| panic!("Missing control {name}"));
        let b = node["bounds"].as_array().unwrap();
        assert!(
            b[0].as_f64().unwrap() >= 0. && b[2].as_f64().unwrap() <= f64::from(width),
            "{name} exceeds {width}: {b:?}"
        );
    }
}
#[test]
fn media_health_ui_actual_frames_normal_small_privacy_and_preview() {
    for width in [1100., 600.] {
        for privacy in [false, true] {
            for preview in [false, true] {
                let ctx = egui::Context::default();
                let root = evidence_root();
                let mut app = ImagineApp::synthetic_preview(&ctx, root);
                app.client = None;
                app.prefs.privacy_mode = privacy;
                let mut project = super::super::media_picker::tests::project();
                project
                    .assets
                    .get_mut(&AssetId("image".into()))
                    .unwrap()
                    .name = "OWN_PRIVATE_MEDIA_NAME".into();
                app.timeline.document = Some(Snapshot::with(
                    project.clone(),
                    json!({"runtime_assets":{}}),
                ));
                app.timeline.media_health = State {
                    open: true,
                    project: Some(project.id.clone()),
                    revision: project.project_revision,
                    report: Some(
                        json!({"project_revision":project.project_revision,"entries":[{"asset_id":"image","status":"missing","impacts":[{"kind":"scene","owner":"OWN_PRIVATE_BINDING","binding":"binding"}]}]}),
                    ),
                    ..Default::default()
                };
                if preview {
                    app.timeline.media_health.preview = Some(
                        json!({"request":{"asset_id":"image","replacement_asset_id":"synthetic-replacement"},"replacement_label":"OWN_PRIVATE_REPLACEMENT","impacts":[{"kind":"scene","owner":"OWN_PRIVATE_BINDING","binding":"binding"}]}),
                    );
                }
                let name = format!("media-health-{width}-{privacy}-{preview}");
                let evidence = capture(&ctx, &name, width, 760., |ui| {
                    app.timeline_media_health_window(ui.ctx())
                });
                let labels = labels(&evidence);
                if privacy {
                    assert!(!labels.iter().any(|s| s.contains("OWN_PRIVATE")));
                } else if preview {
                    assert_controls(&evidence, &["Back", "Relink asset", "Check media"], width);
                    assert!(labels.iter().any(|s| s.contains("OWN_PRIVATE_BINDING")));
                } else {
                    assert_controls(&evidence, &["Relink…", "Check media"], width);
                    assert!(labels.contains(&"OWN_PRIVATE_MEDIA_NAME"));
                }
            }
        }
    }
}
#[test]
fn media_health_ui_changed_revision_releases_pending_and_rejects_late_preview() {
    let project = super::super::media_picker::tests::project();
    let mut s = State {
        open: true,
        project: Some(project.id.clone()),
        revision: project.project_revision,
        ..Default::default()
    };
    let nonce = s.begin();
    let mut changed = project.clone();
    changed.project_revision += 1;
    s.receive(&changed, nonce, Reply::Preview, Ok(json!({"impacts":[]})));
    assert!(!s.pending);
    assert!(s.preview.is_none());
    assert!(s.error.is_some());
    s.close();
    let nonce = s.begin();
    s.open = true;
    let newer = s.begin();
    s.receive(&project, nonce, Reply::Preview, Ok(json!({"impacts":[]})));
    assert!(s.pending);
    assert!(s.preview.is_none());
    assert!(newer != nonce);
}
