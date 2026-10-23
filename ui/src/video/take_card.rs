//! The Scene inspector's current Take, as in Director: its picture, status,
//! recent history and one-click review.
use super::*;
use crate::icons::Icon;
use egui::Rect;
use imagine_video_contract::review::ReviewAction;

impl ImagineApp {
    pub(super) fn timeline_take_card(&mut self, ui: &mut egui::Ui, d: &Snapshot, scene: &Scene) {
        let takes: Vec<_> = d
            .view
            .takes
            .iter()
            .filter(|t| t.scene_id == scene.id && review::attached(d, t))
            .collect();
        ui.horizontal(|ui| {
            ui.label(helto::section("Current take"));
            ui.weak(match takes.len() {
                1 => "1 take".to_owned(),
                n => format!("{n} takes"),
            });
        });
        // The accepted Take, else the newest one.
        let current = scene
            .accepted_take
            .as_ref()
            .and_then(|a| takes.iter().find(|t| t.id == a.id))
            .or(takes.last())
            .copied();
        let Some(take) = current else {
            ui.weak("No takes yet. Generate this Scene to make one.");
            return;
        };
        let number = takes
            .iter()
            .position(|t| t.id == take.id)
            .map_or(0, |i| i + 1);
        let accepted_id = scene.accepted_take.as_ref().map(|a| &a.id);
        let accepted = accepted_id == Some(&take.id);
        let outdated = d.take_state(&take.id).outdated;
        let revealed = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        let busy = self.timeline.busy || self.timeline.admitting;
        let wait = busy.then_some("Wait for the current timeline operation.");
        let mut review = None;
        // Status and history sit beside the picture; the actions follow under both.
        ui.horizontal_top(|ui| {
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(150., 84.), egui::Sense::click());
            ui.painter().rect_filled(rect, 5, helto::MEDIA);
            let endpoint = d
                .view
                .resources
                .iter()
                .any(|r| r.kind() == "h3_endpoint" && r.take_id() == Some(take.id.0.as_str()));
            let texture = (revealed && endpoint)
                .then(|| {
                    let asset = Asset {
                        id: AssetId(format!("last-frame-{}", take.id.0)),
                        revision: 1,
                        kind: MediaKind::Image,
                        name: String::new(),
                        library_source: None,
                    };
                    let path = format!(
                        "/v1/video/projects/{}/takes/{}/last-frame",
                        d.project.id.0, take.id.0
                    );
                    self.timeline_thumbnail_texture(ui.ctx(), &d.project.id, &asset, Some(path))?
                        .ok()
                })
                .flatten();
            if let Some(texture) = texture {
                let size = texture.size_vec2();
                let scale = (rect.width() / size.x).min(rect.height() / size.y);
                ui.painter().image(
                    texture.id(),
                    Rect::from_center_size(rect.center(), size * scale),
                    Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                    egui::Color32::WHITE,
                );
            } else {
                crate::icons::paint(
                    ui.painter(),
                    Rect::from_center_size(rect.center(), egui::vec2(24., 24.)),
                    Icon::Video,
                    helto::DIM,
                );
            }
            let response = response
                .on_hover_text("Play in the viewer")
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "take-card-open".into(), &response);
            if response.clicked() {
                self.timeline_show_take(ui.ctx(), d, &take.id);
            }
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Take {number:03}"));
                    let (text, color) = if accepted && outdated {
                        ("Accepted · outdated", helto::WARN)
                    } else if accepted {
                        ("Accepted", helto::OK)
                    } else {
                        match take.status {
                            snapshot::TakeStatus::Rejected => ("Rejected", helto::DIM),
                            _ => ("Candidate", helto::FOCUS),
                        }
                    };
                    ui.colored_label(color, text);
                });
                // The last ten Takes, oldest first: accepted, shown, outdated or rejected.
                let (dots, _) = ui.allocate_exact_size(egui::vec2(130., 12.), egui::Sense::hover());
                for (i, t) in takes.iter().rev().take(10).rev().enumerate() {
                    let color = if accepted_id == Some(&t.id) {
                        helto::ACCENT
                    } else if t.id == take.id {
                        helto::TEXT
                    } else if d.take_state(&t.id).outdated {
                        helto::WARN
                    } else if t.status == snapshot::TakeStatus::Rejected {
                        helto::FIELD
                    } else {
                        helto::RAISED
                    };
                    ui.painter().circle_filled(
                        egui::pos2(dots.left() + 5. + i as f32 * 13., dots.center().y),
                        4.5,
                        color,
                    );
                }
            });
        });
        ui.horizontal_wrapped(|ui| {
            let r = helto::action(
                ui,
                Icon::Check,
                if outdated {
                    "Accept older Take"
                } else {
                    "Accept"
                },
                accepted,
                wait.or(accepted.then_some("This Take is accepted.")),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "take-card-accept".into(), &r);
            if r.clicked() {
                review = Some(ReviewAction::Accept);
            }
            let r = helto::action(ui, Icon::Close, "Reject Take", false, wait);
            if r.clicked() {
                review = Some(ReviewAction::Reject);
            }
            let previous = takes
                .iter()
                .rev()
                .find(|t| t.id != take.id)
                .map(|t| t.id.0.clone());
            let r = helto::action(
                ui,
                Icon::Compare,
                "Compare A/B",
                false,
                wait.or(previous.is_none().then_some("A/B needs two takes.")),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "take-card-compare".into(), &r);
            if r.clicked()
                && let Some(previous) = previous
            {
                self.timeline.comparison_explicit = true;
                self.timeline.comparison_selection = [Some(previous), Some(take.id.0.clone())];
                self.timeline_compare(ui.ctx(), d);
            }
            if helto::action(ui, Icon::Folder, "Open Captures", false, None).clicked() {
                self.timeline.clear_preview();
                self.timeline.captures.open = true;
            }
            // Removal confirms in Captures, which also explains retained receipts.
            if helto::action(ui, Icon::Trash, "Remove Take", false, wait).clicked() {
                self.timeline.clear_preview();
                self.timeline.captures.confirm = Some(take.id.clone());
                self.timeline.captures.open = true;
            }
        });
        if let Some(action) = review {
            self.timeline_apply_review(ui.ctx(), scene.id.clone(), take.id.clone(), action);
        }
    }
    /// Shows `take` in the viewer, as choosing it in the Take strip does.
    fn timeline_show_take(&mut self, ctx: &egui::Context, d: &Snapshot, take: &TakeId) {
        self.timeline.clear_audio_selection();
        self.timeline.selected_take = Some(take.0.clone());
        if let Some(asset) = d.runtime_take(&take.0) {
            self.timeline_load_media(ctx, asset.to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_scene_inspector_shows_the_accepted_take_with_its_history_and_review() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.client = None;
        let mut project = super::super::media_picker::tests::project();
        project.scenes[0].accepted_take = Some(AcceptedTake {
            id: TakeId("own-first".into()),
            model: None,
            source_fingerprint: "own".into(),
        });
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.selected.insert(project.scenes[0].id.clone());
        let take = |id: &str, status: &str| json!({"id":id,"scene_id":"scene","status":status,"provenance":null});
        app.timeline.document = Some(Snapshot::with(
            project,
            json!({"takes":[take("own-first","accepted"),take("own-second","candidate")]}),
        ));
        let mut nodes = vec![];
        for _ in 0..3 {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600., 1000.),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            output.textures_delta.clear();
            nodes = output.platform_output.accesskit_update.unwrap().nodes;
        }
        let node = |label: &str| {
            nodes
                .iter()
                .map(|(_, n)| n)
                .find(|n| {
                    n.label()
                        .into_iter()
                        .chain(n.value())
                        .any(|t| t.contains(label))
                })
                .unwrap_or_else(|| panic!("{label}"))
        };
        for label in [
            "Current take",
            "2 takes",
            "Take 001",
            "Accepted",
            "Reject Take",
        ] {
            node(label);
        }
        // The accepted Take cannot be accepted again; the pair can be compared.
        assert!(node("Accept").is_disabled());
        assert!(!node("Compare A/B").is_disabled());
    }
}
