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
                    let model = ModelSelection {
                        adapter: "h3".into(),
                        model: "minimax-h3".into(),
                        variant: if motion { "reference" } else { "i2v" }.into(),
                        profile: if motion {
                            "h3-av-noref-v2"
                        } else {
                            "h3-fl2va-i2va-v1"
                        }
                        .into(),
                        profile_revision: "2".into(),
                    };
                    commands.extend(self.timeline.draft_edits());
                    if !scene.settings.iter().any(|slot| slot.model == model) {
                        commands.push(Edit::SetModelSettings { scene_id: scene.id.clone(), settings: SettingsSlot { model: model.clone(), schema_version: 1, values: json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}) } });
                    }
                    commands.push(Edit::SetSceneModel {
                        scene_id: scene.id.clone(),
                        model: ModelChoice::Override(model),
                    });
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
