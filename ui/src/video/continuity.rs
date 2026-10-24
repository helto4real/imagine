//! Explicit source/method intent. Unsupported methods remain visible without substitution.
use super::*;
impl ImagineApp {
    pub(super) fn timeline_continuity(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        commands: &mut Vec<Edit>,
    ) {
        ui.separator();
        ui.label("Continuity");
        let mut continuity = scene.continuity.clone();
        let mut source = continuity.as_ref().map(|value| value.source_scene.clone());
        let previous = d
            .project
            .scenes
            .iter()
            .take_while(|candidate| candidate.id != scene.id)
            .collect::<Vec<_>>();
        let label = source
            .as_ref()
            .and_then(|id| previous.iter().position(|candidate| &candidate.id == id))
            .map(|index| format!("Scene {}", index + 1))
            .unwrap_or_else(|| "No source".into());
        let response = egui::ComboBox::from_id_salt(("continuity-source", &scene.id.0))
            .selected_text(label)
            .show_ui(ui, |ui| {
                let response = ui.selectable_value(&mut source, None, "No source");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "continuity-no-source".into(), &response);
                #[cfg(not(feature = "ui-preview"))]
                let _ = response;
                for (index, candidate) in previous.iter().enumerate() {
                    let response = ui.selectable_value(
                        &mut source,
                        Some(candidate.id.clone()),
                        format!("Scene {}", index + 1),
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("continuity-source-{index}"), &response);
                    #[cfg(not(feature = "ui-preview"))]
                    let _ = response;
                }
            })
            .response;
        #[cfg(feature = "ui-preview")]
        super::qa::control(ui, "continuity-source".into(), &response);
        #[cfg(not(feature = "ui-preview"))]
        let _ = response;
        if source != continuity.as_ref().map(|value| value.source_scene.clone()) {
            continuity = source.map(|source_scene| Continuity {
                include_audio: continuity.as_ref().is_some_and(|value| value.include_audio),
                source_scene,
                method: continuity
                    .as_ref()
                    .map_or(ContinuityMethod::LastFrameFirstFrame, |value| {
                        value.method.clone()
                    }),
            });
        }
        if let Some(value) = &mut continuity {
            let method_response = egui::ComboBox::from_id_salt(("continuity-method", &scene.id.0))
                .selected_text(method_label(&value.method))
                .show_ui(ui, |ui| {
                    let response = ui.selectable_value(
                        &mut value.method,
                        ContinuityMethod::LastFrameFirstFrame,
                        "Last Frame FirstFrame",
                    );
                    if response.clicked() {
                        value.include_audio = false;
                    }
                    for (method, name) in [
                        (ContinuityMethod::MotionContext, "continuity-motion"),
                        (ContinuityMethod::MotionContextExp, "continuity-exp"),
                    ] {
                        let response = ui.selectable_value(
                            &mut value.method,
                            method.clone(),
                            method_label(&method),
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, name.into(), &response);
                        #[cfg(not(feature = "ui-preview"))]
                        let _ = (response, name);
                    }
                    for (method, name) in [
                        (ContinuityMethod::Combo, "continuity-combo"),
                        (ContinuityMethod::Influence, "continuity-influence"),
                    ] {
                        let response = ui.selectable_value(
                            &mut value.method,
                            method.clone(),
                            method_label(&method),
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, name.into(), &response);
                        #[cfg(not(feature = "ui-preview"))]
                        let _ = (response, name);
                    }
                })
                .response;
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "continuity-method".into(), &method_response);
            #[cfg(not(feature = "ui-preview"))]
            let _ = method_response;
            let motion = matches!(
                value.method,
                ContinuityMethod::MotionContext | ContinuityMethod::MotionContextExp
            );
            if motion {
                let response=ui.checkbox(&mut value.include_audio,"Continue source audio").on_hover_text("Use the source Take's paired audio latent tail. Off generates new audio without that guide.");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "continuity-audio".into(), &response);
                #[cfg(not(feature = "ui-preview"))]
                let _ = response;
            }
            if matches!(
                value.method,
                ContinuityMethod::Combo | ContinuityMethod::Influence
            ) {
                ui.weak("Combo/Influence use the incoming non-Hard-Cut boundary. Choose opening audio in that boundary's settings.");
            }
            if matches!(
                value.method,
                ContinuityMethod::Combo | ContinuityMethod::Influence
            ) {
                let active = imagine_video_contract::boundary::resolve(&d.project, &scene.id)
                    .ok()
                    .flatten()
                    .is_some_and(|b| b.policy != imagine_video_contract::boundary::Policy::None);
                let response = ui.weak(if !active {
                    "Hard Cut does not apply Combo/Influence conditioning."
                } else if boundary_video_ready(&d.project, &d.view.resources, &value.source_scene) {
                    "Source video ready; opening conditioning uses the incoming boundary settings."
                } else {
                    "Choose a source video or accept a source Take."
                });
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "continuity-source-status".into(), &response);
                #[cfg(not(feature = "ui-preview"))]
                let _ = response;
            } else if let Ok(source) = d.project.scene(&value.source_scene) {
                let accepted = source.accepted_take.as_ref().is_some_and(|selected| {
                    d.take(&selected.id)
                        .is_some_and(|take| take.status == snapshot::TakeStatus::Accepted)
                });
                let endpoint = source.accepted_take.as_ref().is_some_and(|take| {
                    d.view
                        .resources
                        .iter()
                        .filter(|resource| {
                            resource.kind() == if motion { "h3_motion" } else { "h3_endpoint" }
                                && resource.take_id() == Some(take.id.0.as_str())
                                && (if motion {
                                    &resource.kind["descriptor"]["schema_version"]
                                } else {
                                    &resource.kind["schema_version"]
                                }) == 1
                        })
                        .count()
                        == 1
                });
                ui.weak(if !accepted {
                    "Accept a source Take before generating."
                } else if endpoint && motion {
                    "Source latent tail ready"
                } else if endpoint {
                    "Source endpoint ready"
                } else {
                    "Source needs a new Take with continuity data."
                });
            }
            if (motion || value.method == ContinuityMethod::LastFrameFirstFrame)
                && d.project.effective_model(scene).variant
                    != if motion { "reference" } else { "i2v" }
            {
                let response = helto::icon_button(
                    ui,
                    crate::icons::Icon::Video,
                    if motion {
                        "Use H3 Reference for target"
                    } else {
                        "Use H3 I2V for target"
                    },
                    false,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "continuity-i2v".into(), &response);
                if response.clicked() {
                    commands.extend(self.timeline.draft_edits());
                    commands.extend(model_edits(
                        &d.project,
                        scene,
                        if motion { "reference" } else { "i2v" },
                    ));
                }
            }
            if scene
                .bindings
                .iter()
                .any(|binding| binding.enabled && binding.role == InputRole::FirstFrame)
            {
                ui.weak("Clear the explicit First Frame binding to use this endpoint.");
            }
        }
        if continuity != scene.continuity {
            commands.extend(self.timeline.draft_edits());
            commands.push(Edit::SetContinuity {
                scene_id: scene.id.clone(),
                continuity,
            });
        }
    }
}
/// Edits that put `scene` on H3 `variant`: Reference for carried motion,
/// I2V for a last frame. The project's settings for that model are inherited
/// when it has them; otherwise the Scene gets that model's defaults.
pub(super) fn model_edits(project: &VideoProject, scene: &Scene, variant: &str) -> Vec<Edit> {
    let model = imagine_video_contract::generation::h3_model(variant);
    let mut edits = vec![];
    if project.settings_for(scene, &model).is_none() {
        edits.push(Edit::SetModelSettings {
            scene_id: scene.id.clone(),
            settings: SettingsSlot {
                model: model.clone(),
                schema_version: 1,
                values: serde_json::to_value(imagine_video_contract::generation::h3_defaults_for(
                    variant,
                ))
                .expect("finite defaults"),
            },
        });
    }
    edits.push(Edit::SetSceneModel {
        scene_id: scene.id.clone(),
        model: ModelChoice::Override(model),
    });
    edits
}
impl ImagineApp {
    /// Asks whether Scenes whose continuity carries motion should move to
    /// H3 Reference, which motion needs; it waits for the edit that set it.
    pub(super) fn timeline_model_switch(&mut self, ctx: &egui::Context) {
        if self.timeline.model_switch.is_empty() || self.timeline.busy {
            return;
        }
        let Some(project) = self.timeline.document.as_ref().map(|d| d.project.clone()) else {
            self.timeline.model_switch.clear();
            return;
        };
        let scenes = need_reference(&project, &self.timeline.model_switch);
        if scenes.is_empty() {
            self.timeline.model_switch.clear();
            return;
        }
        let numbers = scenes
            .iter()
            .filter_map(|id| project.scenes.iter().position(|s| &s.id == id))
            .map(|i| (i + 1).to_string())
            .collect::<Vec<_>>();
        let (mut switch, mut keep) = (false, false);
        let modal = egui::Modal::new(egui::Id::new("model-switch")).show(ctx, |ui| {
            ui.set_max_width(420.);
            ui.label(helto::section("Switch to H3 Reference?"));
            ui.label(format!(
                "A Continuous shot carries the previous Scene's motion, which needs H3 Reference. {} {} on another model.",
                if numbers.len() == 1 { "Scene" } else { "Scenes" },
                numbers.join(", ")
            ));
            ui.add_space(6.);
            ui.horizontal(|ui| {
                let r = helto::action(
                    ui,
                    crate::icons::Icon::Check,
                    "Switch to H3 Reference",
                    true,
                    None,
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "model-switch-yes".into(), &r);
                switch = r.clicked();
                let r = helto::icon_button(ui, crate::icons::Icon::Close, "Keep the current model", false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "model-switch-no".into(), &r);
                keep = r.clicked();
            });
        });
        // Escape or a click outside keeps the model, as Keep does.
        keep |= modal.should_close();
        if switch {
            let mut commands = self.timeline.draft_edits();
            for scene in scenes.iter().filter_map(|id| project.scene(id).ok()) {
                commands.extend(model_edits(&project, scene, "reference"));
            }
            self.timeline_edit(ctx, commands, 0);
        }
        if switch || keep {
            self.timeline.model_switch.clear();
        }
    }
}
/// Of `scenes`, those whose continuity carries motion while they are not on
/// H3 Reference, which motion needs.
pub(super) fn need_reference(project: &VideoProject, scenes: &[SceneId]) -> Vec<SceneId> {
    scenes
        .iter()
        .filter_map(|id| project.scene(id).ok())
        .filter(|s| {
            s.continuity.as_ref().is_some_and(|c| {
                matches!(
                    c.method,
                    ContinuityMethod::MotionContext | ContinuityMethod::MotionContextExp
                )
            }) && project.effective_model(s).variant != "reference"
        })
        .map(|s| s.id.clone())
        .collect()
}
fn method_label(method: &ContinuityMethod) -> &'static str {
    match method {
        ContinuityMethod::LastFrameFirstFrame => "Last Frame FirstFrame",
        ContinuityMethod::Combo => "Combo",
        ContinuityMethod::Influence => "Influence",
        ContinuityMethod::MotionContext => "Motion Context",
        ContinuityMethod::MotionContextExp => "Motion Context Exp",
    }
}

// Boundary conditioning consumes video bytes, not the latent/endpoint artifacts
// used by Motion Context and Last Frame FirstFrame.
fn boundary_video_ready(
    project: &VideoProject,
    resources: &[snapshot::ResourceView],
    source: &SceneId,
) -> bool {
    let Ok(scene) = project.scene(source) else {
        return false;
    };
    let mut resources = resources.iter();
    if let Some(take) = &scene.accepted_take {
        return resources
            .any(|r| r.kind() == "take_video" && r.take_id() == Some(take.id.0.as_str()));
    }
    project.scene_clip(source).is_some_and(|clip| {
        project.assets.get(&clip.asset).is_some_and(|asset| {
            asset.kind == MediaKind::Video
                && resources.any(|r| {
                    r.kind() == "reference"
                        && r.asset_id() == Some(asset.id.0.as_str())
                        && r.kind["revision"].as_u64() == Some(asset.revision)
                })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn motion_project(generation: bool) -> VideoProject {
        let mut p = VideoProject::create(CreateProject {
            schema_version: 1,
            generation: generation.then(imagine_video_contract::generation::Defaults::default),
            default_model: imagine_video_contract::generation::h3_model("i2v"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        for (i, id) in ["own-first", "own-second"].into_iter().enumerate() {
            p.scenes.push(Scene {
                id: SceneId(id.into()),
                start: Time {
                    num: 2 * i as u64,
                    den: 1,
                },
                duration: Time { num: 2, den: 1 },
                description: String::new(),
                model: ModelChoice::ProjectDefault,
                settings: vec![],
                sections: vec![],
                bindings: vec![],
                continuity: (i == 1).then(|| Continuity {
                    source_scene: SceneId("own-first".into()),
                    method: ContinuityMethod::MotionContextExp,
                    include_audio: false,
                }),
                accepted_take: None,
            });
        }
        p
    }
    #[test]
    fn carried_motion_off_reference_asks_and_the_switch_inherits_project_settings() {
        let p = motion_project(true);
        let ids = [SceneId("own-first".into()), SceneId("own-second".into())];
        assert_eq!(need_reference(&p, &ids), [ids[1].clone()]);
        let scene = p.scene(&ids[1]).unwrap();
        // The project has Reference settings, so the Scene only changes model.
        let edits = model_edits(&p, scene, "reference");
        assert!(
            matches!(edits.as_slice(), [Edit::SetSceneModel { model: ModelChoice::Override(m), .. }] if m.variant == "reference")
        );
        // Without them the Scene gets Reference's defaults, with its locked file.
        let bare = motion_project(false);
        let edits = model_edits(&bare, bare.scene(&ids[1]).unwrap(), "reference");
        let [
            Edit::SetModelSettings { settings, .. },
            Edit::SetSceneModel { .. },
        ] = edits.as_slice()
        else {
            panic!("{edits:?}");
        };
        assert_eq!(
            settings.values["components"]["model"],
            "diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        );
        let mut switched = p.clone();
        switched.scenes[1].model =
            ModelChoice::Override(imagine_video_contract::generation::h3_model("reference"));
        assert!(need_reference(&switched, &ids).is_empty());
    }
    #[test]
    fn the_switch_popup_asks_once_the_continuity_is_set_and_either_answer_closes_it() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        // Nothing listens there: an edit is either still out or has failed.
        app.client = Some(
            crate::client::Client::connect(&crate::settings::Connection {
                url: "http://127.0.0.1:1".into(),
            })
            .unwrap(),
        );
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.document = Some(Snapshot::with(motion_project(true), json!({})));
        let frame = |app: &mut ImagineApp, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 900.),
                    )),
                    events,
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            output.textures_delta.clear();
            output.platform_output.accesskit_update.unwrap().nodes
        };
        let click = |app: &mut ImagineApp, label: &str| {
            let mut nodes = vec![];
            for _ in 0..2 {
                nodes = frame(app, vec![]);
            }
            let b = nodes
                .iter()
                .find(|(_, n)| n.label() == Some(label))
                .and_then(|(_, n)| n.bounds())
                .unwrap_or_else(|| panic!("{label}"));
            let pos = egui::pos2(((b.x0 + b.x1) / 2.) as f32, ((b.y0 + b.y1) / 2.) as f32);
            for pressed in [true, false] {
                frame(
                    app,
                    vec![
                        egui::Event::PointerMoved(pos),
                        egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
        };
        // A Scene without carried motion asks nothing.
        app.timeline.model_switch = vec![SceneId("own-first".into())];
        frame(&mut app, vec![]);
        assert!(app.timeline.model_switch.is_empty());
        app.timeline.model_switch = vec![SceneId("own-second".into())];
        let nodes = frame(&mut app, vec![]);
        assert!(nodes.iter().any(|(_, n)| {
            n.value()
                .or(n.label())
                .is_some_and(|t| t.contains("Scene 2 on another model"))
        }));
        click(&mut app, "Keep the current model");
        assert!(app.timeline.model_switch.is_empty() && !app.timeline.busy);
        app.timeline.model_switch = vec![SceneId("own-second".into())];
        click(&mut app, "Switch to H3 Reference");
        assert!(app.timeline.model_switch.is_empty());
        assert!(
            app.timeline.busy || app.timeline.error.is_some(),
            "the switch edit was sent"
        );
    }
    fn resources(value: serde_json::Value) -> Vec<snapshot::ResourceView> {
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn boundary_readiness_uses_current_video_bytes_not_endpoint_or_latent_artifacts() {
        let mut p = VideoProject::create(CreateProject {
            schema_version: 1,
            generation: None,
            default_model: imagine_video_contract::generation::h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        let id = SceneId("own-source".into());
        p.scenes.push(Scene {
            id: id.clone(),
            start: Time { num: 0, den: 1 },
            duration: Time { num: 1, den: 1 },
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: Some(AcceptedTake {
                id: TakeId("own-take".into()),
                model: None,
                source_fingerprint: String::new(),
            }),
        });
        assert!(!boundary_video_ready(
            &p,
            &resources(json!([{"kind":{"kind":"h3_endpoint","take_id":"own-take"}}])),
            &id
        ));
        assert!(boundary_video_ready(
            &p,
            &resources(json!([{"kind":{"kind":"take_video","take_id":"own-take"}}])),
            &id
        ));
        p.scenes[0].accepted_take = None;
        let asset = AssetId("own-video".into());
        p.assets.insert(
            asset.clone(),
            Asset {
                id: asset.clone(),
                kind: MediaKind::Video,
                revision: 2,
                name: String::new(),
                library_source: None,
            },
        );
        p.assembly.clips.insert(
            id.clone(),
            imagine_video_contract::assembly::ClipInstance {
                asset,
                source_in: Time { num: 0, den: 1 },
                source_out: None,
                speed: 1.,
                include_audio: true,
            },
        );
        assert!(!boundary_video_ready(
            &p,
            &resources(json!([{"kind":{"kind":"reference","asset_id":"own-video","revision":1}}])),
            &id
        ));
        assert!(boundary_video_ready(
            &p,
            &resources(json!([{"kind":{"kind":"reference","asset_id":"own-video","revision":2}}])),
            &id
        ));
    }
}
