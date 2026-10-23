//! Manual Take review; generation results stay outside authoring Undo/Redo.
use super::*;
use imagine_video_contract::review::{ReviewAction, ReviewRequest};

pub(super) fn attached(document: &Snapshot, take: &snapshot::TakeView) -> bool {
    document
        .capture(&take.id)
        .is_none_or(|capture| !capture.detached && !capture.removed)
}

/// Settings that differ between two Takes, highlighted; recipe fields are
/// shown as recorded, and an imported Take has none.
pub(super) fn take_differences(ui: &mut egui::Ui, pair: [Option<&snapshot::TakeView>; 2]) {
    egui::Grid::new("take-metadata-differences").show(ui, |ui| {
        ui.strong("Setting");
        ui.strong("A");
        ui.strong("B");
        ui.end_row();
        for (label, path) in [
            ("Status", ""),
            ("Seed", "/seed"),
            ("Model", "/model/model"),
            ("Variant", "/model/variant"),
            ("Profile", "/model/profile"),
            ("Profile revision", "/model/profile_revision"),
            ("First stage steps", "/settings/values/stage_one/steps"),
        ] {
            let values = pair.map(|take| match take {
                Some(t) if path.is_empty() => t.status.label().into(),
                Some(t) => t
                    .provenance
                    .as_ref()
                    .and_then(|p| p.pointer(path))
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                    })
                    .unwrap_or_else(|| "—".into()),
                None => "—".into(),
            });
            ui.label(label);
            let different = values[0] != values[1];
            for value in values {
                ui.colored_label(if different { helto::ACCENT } else { helto::DIM }, value);
            }
            ui.end_row();
        }
    });
}

pub(super) fn comparison_entries(
    document: &Snapshot,
    scene: &SceneId,
) -> Vec<player::ComparisonEntry> {
    document
        .view
        .takes
        .iter()
        .filter(|take| &take.scene_id == scene)
        .filter(|take| attached(document, take))
        .enumerate()
        .map(|(index, take)| player::ComparisonEntry {
            id: take.id.0.clone(),
            label: format!("Take {:03} · {}", index + 1, take.status.label()),
            accepted: document.take_state(&take.id).accepted,
            rejected: take.status == snapshot::TakeStatus::Rejected,
        })
        .collect()
}

impl Timeline {
    fn sync_take_pair(&mut self, scene: Option<SceneId>, ids: &[String], accepted: Option<&str>) {
        if self.comparison_scene != scene {
            self.comparison_scene = scene;
            self.comparison_explicit = false;
            self.comparison_selection = [None, None];
        }
        let previous = if self.comparison_explicit || self.comparison.is_some() {
            self.comparison_selection.clone()
        } else {
            [None, None]
        };
        self.comparison_selection = take_selection::normalize(ids, accepted, &previous);
    }
}

impl ImagineApp {
    pub(super) fn timeline_apply_review(
        &mut self,
        ctx: &egui::Context,
        scene: SceneId,
        take: TakeId,
        action: ReviewAction,
    ) {
        let next = PostAction::Review {
            scene,
            take,
            action,
        };
        let commands = self.timeline.draft_edits();
        if commands.is_empty() {
            self.timeline_action(ctx, next);
        } else {
            self.timeline.after_edit = Some(next);
            self.timeline_edit(ctx, commands, 0);
        }
    }
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
        let entries = selected
            .as_ref()
            .map(|scene| comparison_entries(document, scene))
            .unwrap_or_default();
        let ids = entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>();
        let accepted = selected
            .as_ref()
            .and_then(|id| document.project.scene(id).ok())
            .and_then(|scene| scene.accepted_take.as_ref())
            .map(|take| take.id.0.as_str());
        self.timeline
            .sync_take_pair(selected.clone(), &ids, accepted);
        let mut compare = false;
        ui.horizontal_wrapped(|ui| {
            ui.weak("Takes");
            let captures =
                helto::icon_button(ui, crate::icons::Icon::Folder, "Open Captures", false);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "take-captures".into(), &captures);
            if captures.clicked() {
                self.timeline.clear_preview();
                self.timeline.captures.open = true;
            }
            for (index, take) in document
                .view
                .takes
                .iter()
                .filter(|take| Some(&take.scene_id) == selected.as_ref())
                .filter(|take| attached(document, take))
                .enumerate()
            {
                let id = take.id.0.as_str();
                let state = document.take_state(&take.id);
                let (accepted, outdated) = (state.accepted, state.outdated);
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
                    take.status.label(),
                    if outdated { " · Inputs changed" } else { "" }
                ));
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, format!("take-{index}"), &response);
                if response.clicked() {
                    self.timeline.clear_audio_selection();
                    self.timeline.selected_take = Some(id.to_owned());
                    if let Some(asset) = document.runtime_take(id) {
                        self.timeline_load_media(ui.ctx(), asset.to_owned());
                    }
                }
            }
            let slots = &self.timeline.comparison_selection;
            let button = ui
                .allocate_ui(egui::vec2(32., 32.), |ui| {
                    ui.add_enabled_ui(
                        !self.timeline.busy
                            && !self.timeline.admitting
                            && slots.iter().all(Option::is_some),
                        |ui| {
                            helto::icon_button(
                                ui,
                                crate::icons::Icon::Compare,
                                "Compare A/B",
                                false,
                            )
                        },
                    )
                    .inner
                })
                .inner;
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "take-compare".into(), &button);
            compare = button.clicked();
        });
        let Some(scene) = selected else { return };
        if compare {
            self.timeline_compare(ui.ctx(), document);
        }
        let Some(id) =
            self.timeline.selected_take.clone().filter(|id| {
                document.view.takes.iter().any(|take| {
                    take.id.0 == *id && take.scene_id == scene && attached(document, take)
                })
            })
        else {
            return;
        };
        let outdated = document.take_state(&TakeId(id.clone())).outdated;
        let mut action = None;
        ui.add_enabled_ui(!self.timeline.busy && !self.timeline.admitting, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (name, icon, label, value) in [
                    (
                        "take-accept",
                        crate::icons::Icon::Check,
                        if outdated {
                            "Accept older Take"
                        } else {
                            "Accept"
                        },
                        ReviewAction::Accept,
                    ),
                    (
                        "take-reject",
                        crate::icons::Icon::Close,
                        "Reject Take",
                        ReviewAction::Reject,
                    ),
                    (
                        "take-candidate",
                        crate::icons::Icon::Retry,
                        "Return to Candidate",
                        ReviewAction::Candidate,
                    ),
                ] {
                    let response = helto::icon_button(ui, icon, label, false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, name.into(), &response);
                    #[cfg(not(feature = "ui-preview"))]
                    let _ = name;
                    if response.clicked() {
                        action = Some(value);
                    }
                }
                // Removal confirms in Captures, which also explains retained receipts.
                let remove =
                    helto::icon_button(ui, crate::icons::Icon::Trash, "Remove Take", false);
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "take-remove".into(), &remove);
                if remove.clicked() {
                    self.timeline.clear_preview();
                    self.timeline.captures.confirm = Some(TakeId(id.clone()));
                    self.timeline.captures.open = true;
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
                        self.timeline.comparison_explicit = true;
                        self.timeline.comparison_selection = take_selection::update(
                            &ids,
                            accepted,
                            &self.timeline.comparison_selection,
                            side,
                            &id,
                        );
                    }
                }
                let menu = helto::icon_button(ui, crate::icons::Icon::Info, "Differences", false);
                egui::Popup::menu(&menu).show(|ui| {
                    let reveal = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                    if !reveal {
                        ui.label("Hidden metadata");
                        return;
                    }
                    let pair = self.timeline.comparison_selection.each_ref().map(|id| {
                        id.as_ref()
                            .and_then(|id| document.take(&TakeId(id.clone())))
                    });
                    take_differences(ui, pair);
                });
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "take-differences".into(), &menu);
                let _ = menu;
            });
        });
        if let Some(action) = action {
            self.timeline_apply_review(ui.ctx(), scene, TakeId(id), action);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_pair_follows_new_takes_until_an_explicit_choice_and_resets_for_another_scene() {
        let mut state = Timeline::default();
        let scene = Some(SceneId("own_scene".into()));
        let mut ids = vec!["first".to_owned()];
        state.sync_take_pair(scene.clone(), &ids, Some("first"));
        ids.push("second".into());
        state.sync_take_pair(scene.clone(), &ids, Some("first"));
        ids.push("third".into());
        state.sync_take_pair(scene.clone(), &ids, Some("first"));
        assert_eq!(
            state.comparison_selection,
            [Some("first".into()), Some("third".into())]
        );
        state.comparison_explicit = true;
        state.comparison_selection = [Some("second".into()), Some("first".into())];
        ids.push("fourth".into());
        state.sync_take_pair(scene, &ids, Some("third"));
        assert_eq!(
            state.comparison_selection,
            [Some("second".into()), Some("first".into())]
        );
        state.sync_take_pair(
            Some(SceneId("another_own_scene".into())),
            &ids,
            Some("fourth"),
        );
        assert_eq!(
            state.comparison_selection,
            [Some("third".into()), Some("fourth".into())]
        );
        assert!(!state.comparison_explicit);
    }
}

#[cfg(test)]
mod difference_tests {
    use super::*;
    #[test]
    fn differences_show_recorded_recipes_and_mark_unknown_imported_takes() {
        let takes: Vec<snapshot::TakeView> = serde_json::from_value(json!([
            {"id":"a","scene_id":"s","status":"accepted","provenance":{"seed":7,"model":{"model":"own-h3"}}},
            {"id":"b","scene_id":"s","status":"candidate","provenance":null}
        ]))
        .unwrap();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(Default::default(), |ui| {
            take_differences(ui, [takes.first(), takes.get(1)]);
        });
        output.textures_delta.clear();
        let mut texts = vec![];
        for shape in &output.shapes {
            if let egui::Shape::Text(text) = &shape.shape {
                texts.push(text.galley.text().to_owned());
            }
        }
        for expected in ["accepted", "candidate", "7", "own-h3", "—"] {
            assert!(texts.iter().any(|t| t == expected), "{expected}: {texts:?}");
        }
    }
}
