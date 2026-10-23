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
                    ui.add_enabled_ui(false, |ui| {
                        for method in [ContinuityMethod::Combo, ContinuityMethod::Influence] {
                            ui.selectable_value(
                                &mut value.method,
                                method.clone(),
                                method_label(&method),
                            )
                            .on_hover_text("This method is not connected to the worker yet.");
                        }
                    });
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
            if let Ok(source) = d.project.scene(&value.source_scene) {
                let accepted = source.accepted_take.as_ref().is_some_and(|selected| {
                    d.raw["takes"].as_array().into_iter().flatten().any(|take| {
                        take["id"].as_str() == Some(selected.id.0.as_str())
                            && take["status"] == "accepted"
                    })
                });
                let endpoint = source.accepted_take.as_ref().is_some_and(|take| {
                    d.raw["resources"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|resource| {
                            resource["kind"]["kind"]
                                == if motion { "h3_motion" } else { "h3_endpoint" }
                                && resource["kind"]["take_id"].as_str() == Some(take.id.0.as_str())
                                && (if motion {
                                    &resource["kind"]["descriptor"]["schema_version"]
                                } else {
                                    &resource["kind"]["schema_version"]
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
                let response = ui.button(if motion {
                    "Use H3 Reference for target"
                } else {
                    "Use H3 I2V for target"
                });
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
