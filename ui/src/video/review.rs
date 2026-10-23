//! Manual Take review; generation results stay outside authoring Undo/Redo.
use super::*;
use imagine_video_contract::review::{ReviewAction, ReviewRequest};

impl ImagineApp {
    pub(super) fn timeline_review(
        &mut self,
        ctx: &egui::Context,
        scene: SceneId,
        take: TakeId,
        action: ReviewAction,
    ) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        let request = ReviewRequest {
            schema_version: 1,
            expected_project_revision: document.project.project_revision,
            scene_id: scene,
            take_id: take,
            action,
        };
        self.timeline_request(
            ctx,
            true,
            format!("/v1/video/projects/{}/review", document.project.id.0),
            Some(json!(request)),
            false,
            None,
        );
    }

    pub(super) fn timeline_take_strip(&mut self, ui: &mut egui::Ui, document: &Snapshot) {
        let selected = self.timeline.scene().map(|scene| scene.id.clone());
        for slot in &mut self.timeline.comparison_selection {
            if slot.as_ref().is_some_and(|id| {
                !document.raw["takes"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|t| {
                        t["id"].as_str() == Some(id)
                            && t["scene_id"].as_str() == selected.as_ref().map(|s| s.0.as_str())
                    })
            }) {
                *slot = None;
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.weak("Takes");
            for (index, take) in document.raw["takes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|take| {
                    take["scene_id"].as_str() == selected.as_ref().map(|id| id.0.as_str())
                })
                .enumerate()
            {
                let Some(id) = take["id"].as_str() else {
                    continue;
                };
                let state = &document.raw["take_states"][id];
                let accepted = state["accepted"].as_bool().unwrap_or(false);
                let outdated = state["outdated"].as_bool().unwrap_or(false);
                let label = format!(
                    "{}{}{}",
                    index + 1,
                    if accepted { " · Accepted" } else { "" },
                    if outdated { " · Outdated" } else { "" }
                );
                let response = helto::select_button(
                    ui,
                    &label,
                    self.timeline.selected_take.as_deref() == Some(id),
                )
                .on_hover_text(format!(
                    "{}{}",
                    take["status"].as_str().unwrap_or("candidate"),
                    if outdated { " · Inputs changed" } else { "" }
                ));
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, format!("take-{index}"), &response);
                if response.clicked() {
                    self.timeline.clear_audio_selection();
                    self.timeline.selected_take = Some(id.to_owned());
                    if let Some(asset) = document.raw["runtime_takes"][id].as_str() {
                        self.timeline_load_media(ui.ctx(), asset.to_owned());
                    }
                }
            }
        });
        let Some(scene) = selected else { return };
        let Some(id) = self.timeline.selected_take.clone().filter(|id| {
            document.raw["takes"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|take| {
                    take["id"].as_str() == Some(id.as_str())
                        && take["scene_id"].as_str() == Some(scene.0.as_str())
                })
        }) else {
            return;
        };
        let outdated = document.raw["take_states"][&id]["outdated"]
            .as_bool()
            .unwrap_or(false);
        let mut action = None;
        let mut compare = false;
        ui.add_enabled_ui(!self.timeline.busy && !self.timeline.admitting, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (name, label, value) in [
                    (
                        "take-accept",
                        if outdated {
                            "Accept older Take"
                        } else {
                            "Accept"
                        },
                        ReviewAction::Accept,
                    ),
                    ("take-reject", "Reject", ReviewAction::Reject),
                    ("take-candidate", "Candidate", ReviewAction::Candidate),
                ] {
                    let response = ui.small_button(label);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, name.into(), &response);
                    #[cfg(not(feature = "ui-preview"))]
                    let _ = name;
                    if response.clicked() {
                        action = Some(value);
                    }
                }
                if outdated {
                    ui.weak("Inputs changed");
                }
                for (side, label, name) in [(0, "Set A", "take-set-a"), (1, "Set B", "take-set-b")]
                {
                    let response = ui.selectable_label(
                        self.timeline.comparison_selection[side].as_deref() == Some(&id),
                        label,
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, name.into(), &response);
                    let _ = name;
                    if response.clicked() {
                        self.timeline.comparison_selection[side] = Some(id.clone());
                    }
                }
                let slots = &self.timeline.comparison_selection;
                let response = ui.add_enabled(
                    slots[0].is_some() && slots[1].is_some() && slots[0] != slots[1],
                    egui::Button::new("Compare A/B"),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "take-compare".into(), &response);
                compare = response.clicked();
                let menu = ui.menu_button("Differences", |ui| {
                    let reveal = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                    if !reveal {
                        ui.label("Hidden metadata");
                        return;
                    }
                    let pair = self.timeline.comparison_selection.each_ref().map(|id| {
                        id.as_ref().and_then(|id| {
                            document.raw["takes"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .find(|t| t["id"].as_str() == Some(id))
                        })
                    });
                    egui::Grid::new("take-metadata-differences").show(ui, |ui| {
                        ui.strong("Setting");
                        ui.strong("A");
                        ui.strong("B");
                        ui.end_row();
                        for (label, path) in [
                            ("Status", "/status"),
                            ("Seed", "/provenance/seed"),
                            ("Model", "/provenance/model/model"),
                            ("Variant", "/provenance/model/variant"),
                            ("Profile", "/provenance/model/profile"),
                            ("Profile revision", "/provenance/model/profile_revision"),
                            (
                                "First stage steps",
                                "/provenance/settings/values/stage_one/steps",
                            ),
                        ] {
                            let values = pair.map(|take| {
                                take.and_then(|t| t.pointer(path))
                                    .map(|v| {
                                        v.as_str()
                                            .map(str::to_owned)
                                            .unwrap_or_else(|| v.to_string())
                                    })
                                    .unwrap_or_else(|| "—".into())
                            });
                            ui.label(label);
                            let different = values[0] != values[1];
                            for value in values {
                                ui.colored_label(
                                    if different { helto::ACCENT } else { helto::DIM },
                                    value,
                                );
                            }
                            ui.end_row();
                        }
                    });
                });
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "take-differences".into(), &menu.response);
                let _ = menu;
            });
        });
        if compare {
            self.timeline_compare(ui.ctx(), document);
        }
        if let Some(action) = action {
            let next = PostAction::Review {
                scene,
                take: TakeId(id),
                action,
            };
            let commands = self.timeline.draft_edits();
            if commands.is_empty() {
                self.timeline_action(ui.ctx(), next);
            } else {
                self.timeline.after_edit = Some(next);
                self.timeline_edit(ui.ctx(), commands, 0);
            }
        }
    }
}
