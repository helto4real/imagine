//! Project Captures retain the frozen recipe, independently of Scene attachment.
use super::*;
use imagine_video_contract::captures::{Action, Request};

#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    confirm: Option<TakeId>,
}

impl ImagineApp {
    pub(super) fn timeline_import_take(
        &mut self,
        ctx: &egui::Context,
        scene: SceneId,
        asset: AssetId,
    ) {
        let Some(d) = &self.timeline.document else {
            return;
        };
        self.timeline_request(
            ctx,
            true,
            format!("/v1/video/projects/{}/captures/import", d.project.id.0),
            Some(json!(imagine_video_contract::captures::Import {
                schema_version: 1,
                expected_project_revision: d.project.project_revision,
                scene_id: scene,
                asset_id: asset,
                accept: false,
            })),
            false,
            None,
        );
    }
    pub(super) fn timeline_capture(
        &mut self,
        ctx: &egui::Context,
        scene: SceneId,
        take: TakeId,
        action: Action,
    ) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        self.timeline_request(
            ctx,
            true,
            format!("/v1/video/projects/{}/captures", document.project.id.0),
            Some(json!(Request {
                schema_version: 1,
                expected_project_revision: document.project.project_revision,
                scene_id: scene,
                take_id: take,
                action,
            })),
            false,
            None,
        );
    }

    pub(super) fn timeline_captures_dialog(&mut self, ctx: &egui::Context, d: &Snapshot) {
        let scene = self.timeline.scene().map(|s| s.id.clone());
        let mut action = None;
        let mut preview = None;
        let mut review_action = None;
        let mut pick = false;
        let bounds = ctx.content_rect();
        let response = egui::Modal::new(egui::Id::new("project-captures")).show(ctx, |ui| {
            let width = (bounds.width() - 48.).clamp(260., 1000.);
            ui.set_width(width);
            ui.set_max_width(width);
            ui.set_max_height((bounds.height() - 80.).max(240.));
            ui.horizontal(|ui| {
                ui.heading("Captures");
                let r=ui.add_enabled_ui(scene.is_some()&&!self.timeline.busy,|ui|helto::icon_button(ui,crate::icons::Icon::Insert,"Attach video as Take",false)).inner;
                control(ui,"capture-import",&r);pick=r.clicked();
                let close = helto::icon_button(ui, crate::icons::Icon::Close, "Close Captures", false);
                control(ui, "captures-close", &close);
                if close.clicked() { self.timeline.captures.open = false; }
            });
            if let Some(error) = &self.timeline.error { ui.colored_label(helto::ERROR, error); }
            if d.raw["capture_result"]["retained_for_provenance"] == true { ui.label("Capture removed from the list; its source receipt is retained for dependent Takes."); }
            if let Some(code) = d.raw["capture_result"]["cleanup_warning"].as_str() { ui.colored_label(helto::ERROR, format!("Capture removed. Some owned project files could not be cleaned: {code}. External originals are unchanged.")); }
            if scene.is_none() { ui.weak("Choose a Scene to attach a Capture. Existing Captures can still be previewed or removed."); }
            if let Some(id) = self.timeline.captures.confirm.clone() {
                ui.label("Remove this Capture and its owned project files? External originals are preserved. Receipts used by other Takes are retained as provenance.");
                ui.horizontal(|ui| {
                    let remove = helto::action(ui, crate::icons::Icon::Trash, "Remove Capture", false, self.timeline.busy.then_some("Wait for the current timeline operation."));
                    control(ui, "capture-confirm-remove", &remove);
                    if remove.clicked() {
                        action = Some((id, Action::Delete));
                        self.timeline.captures.confirm = None;
                    }
                    let cancel = helto::icon_button(ui, crate::icons::Icon::Close, "Cancel", false); control(ui, "capture-confirm-cancel", &cancel);
                    if cancel.clicked() { self.timeline.captures.confirm = None; }
                });
                return;
            }
            let reveal = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
            let scroll = egui::ScrollArea::vertical().id_salt("captures-scroll").auto_shrink([false, false]).max_height((bounds.height() - 160.).max(180.)).show(ui, |ui| {
                let takes: Vec<_> = d.view.takes.iter().collect();
                for section in [Section::Catalog, Section::Attached, Section::Retained] {
                    if section == Section::Retained && !takes.iter().any(|take| section.includes(d, take, scene.as_ref())) { continue; }
                    ui.strong(section.title());
                    let rows: Vec<_> = takes.iter().copied().filter(|take| {
                        section.includes(d, take, scene.as_ref())
                    }).collect();
                    if rows.is_empty() { ui.weak(if section == Section::Catalog { "No captured Takes." } else { "No attached Takes." }); }
                    for (index, take) in rows.into_iter().enumerate() {
                        let id = take.id.0.as_str();
                        let local = takes.iter().copied().find(|entry| scene.as_ref() == Some(&entry.scene_id) && review::attached(d, entry) && (
                            entry.id == take.id || d.capture(&entry.id).and_then(|c| c.source_capture.as_ref()) == Some(&take.id)
                        ));
                        let attached = local.is_some();
                        let status = local.unwrap_or(take);
                        ui.push_id((section.name(), id), |ui| {
                            ui.group(|ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(format!("{} {:03}", if section == Section::Attached { "Take" } else { "Capture" }, index + 1));
                                    ui.weak(if attached { status.status.label() } else { "unattached" });
                                    match (&take.provenance, reveal) {
                                        (None, true) => { ui.small("Imported video · generation recipe unknown"); }
                                        (Some(recipe), true) => { ui.small(format!("Seed {} · {}", recipe["seed"], recipe["model"]["model"].as_str().unwrap_or(""))).on_hover_text(id); }
                                        _ => {}
                                    }
                                });
                                if section == Section::Retained {
                                    ui.weak(if has_dependents(d, id) { "Removed from Captures; kept for linked Takes." } else { "No linked Takes remain. Remove this receipt to release its owned files." });
                                }
                                ui.horizontal_wrapped(|ui| {
                                    for (name, icon, tip, enabled, next) in [
                                        ("preview", crate::icons::Icon::Video, "Preview Capture", true, None),
                                        ("attach", crate::icons::Icon::Plus, "Attach as Candidate Take", scene.is_some() && !attached, Some(Action::Attach)),
                                        ("accept", crate::icons::Icon::Check, "Attach and Accept", scene.is_some() && (status.status != snapshot::TakeStatus::Accepted || !attached), Some(Action::AttachAndAccept)),
                                        ("detach", crate::icons::Icon::Unlink, "Detach Take; keep Capture", attached, Some(Action::Detach)),
                                        ("remove", crate::icons::Icon::Trash, "Remove Capture", true, Some(Action::Delete)),
                                    ] {
                                        let button = ui.add_enabled_ui(enabled && !self.timeline.busy && !self.timeline.admitting, |ui| helto::icon_button(ui, icon, tip, false)).inner;
                                        control(ui, &format!("capture-{}-{index}-{name}", section.name()), &button);
                                        if button.clicked() {
                                            if name == "preview" { preview = d.runtime_take(id).map(str::to_owned); }
                                            else if name == "remove" { self.timeline.captures.confirm = Some(TakeId(id.into())); }
                                            else if let Some(next) = next {
                                                let target = if name == "detach" { status.id.0.as_str() } else { id };
                                                action = Some((TakeId(target.into()), next));
                                            }
                                        }
                                    }
                                    for (name, icon, tip, value, unchanged) in [
                                        ("reject", crate::icons::Icon::Close, "Mark Take Rejected", imagine_video_contract::review::ReviewAction::Reject, snapshot::TakeStatus::Rejected),
                                        ("candidate", crate::icons::Icon::Retry, "Restore Candidate Take", imagine_video_contract::review::ReviewAction::Candidate, snapshot::TakeStatus::Candidate),
                                    ] {
                                        let button = ui.add_enabled_ui(attached && status.status != unchanged && !self.timeline.busy && !self.timeline.admitting, |ui| helto::icon_button(ui, icon, tip, false)).inner;
                                        control(ui, &format!("capture-{}-{index}-{name}", section.name()), &button);
                                        if button.clicked() { review_action = Some((status.id.clone(), value)); }
                                    }
                                });
                                if reveal && let Some(source) = d.capture(&take.id).and_then(|c| c.source_capture.as_ref()) { ui.small(format!("Source capture: {}", source.0)); }
                            });
                        });
                    }
                    ui.add_space(8.);
                }
            });
            let response = ui.interact(scroll.inner_rect, ui.id().with("captures-scroll-qa"), egui::Sense::hover());
            control(ui, "captures-scroll", &response);
        });
        if response.should_close() {
            self.timeline.captures.open = false;
        }
        if pick && let Some(scene) = scene.clone() {
            self.timeline.captures.open = false;
            self.timeline_pick_media(super::media_picker::Target::Take(scene));
        }
        if let Some(asset) = preview {
            self.timeline.captures.open = false;
            self.timeline_load_media(ctx, asset);
        }
        if let Some((take, action)) = action
            && let Some(scene) = scene.clone().or_else(|| {
                (action == Action::Delete)
                    .then(|| d.take(&take).map(|t| t.scene_id.clone()))
                    .flatten()
            })
        {
            let next = PostAction::Capture {
                scene,
                take,
                action,
            };
            let edits = self.timeline.draft_edits();
            if edits.is_empty() {
                self.timeline_action(ctx, next);
            } else {
                self.timeline.after_edit = Some(next);
                self.timeline_edit(ctx, edits, 0);
            }
        } else if let Some((take, action)) = review_action
            && let Some(scene) = scene
        {
            self.timeline_apply_review(ctx, scene, take, action);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    Catalog,
    Attached,
    Retained,
}
impl Section {
    fn name(self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::Attached => "attached",
            Self::Retained => "receipt",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Self::Catalog => "Project Captures",
            Self::Attached => "Attached Takes",
            Self::Retained => "Retained source receipts",
        }
    }
    fn includes(
        self,
        document: &Snapshot,
        take: &snapshot::TakeView,
        scene: Option<&SceneId>,
    ) -> bool {
        let state = document.capture(&take.id).cloned().unwrap_or_default();
        match self {
            Self::Catalog => !state.removed && state.source_capture.is_none(),
            Self::Attached => scene == Some(&take.scene_id) && review::attached(document, take),
            Self::Retained => state.removed,
        }
    }
}
fn has_dependents(document: &Snapshot, id: &str) -> bool {
    document
        .view
        .takes
        .iter()
        .any(|take| take.source_take() == Some(id))
        || document
            .view
            .captures
            .values()
            .any(|capture| capture.source_capture.as_ref().is_some_and(|s| s.0 == id))
}

fn control(ui: &egui::Ui, name: &str, response: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), response);
    #[cfg(not(feature = "ui-preview"))]
    let _ = (ui, name, response);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_receipt_stays_reachable_after_last_attachment_or_scene_is_removed() {
        let project = media_picker::tests::project();
        let scene = project.scenes[0].id.clone();
        let mut document = Snapshot::with(
            project,
            json!({
                "takes":[{"id":"source","scene_id":scene,"status":"candidate","provenance":{"seed":42}},
                    {"id":"copy","scene_id":"other","status":"candidate","provenance":{"seed":42}}],
                "captures":{"source":{"removed":true,"detached":true},"copy":{"source_capture":"source"}}
            }),
        );
        let take = document.view.takes[0].clone();
        assert!(!Section::Catalog.includes(&document, &take, Some(&scene)));
        assert!(Section::Retained.includes(&document, &take, Some(&scene)));
        assert!(has_dependents(&document, "source"));
        document.view.takes.pop();
        document.view.captures.remove(&TakeId("copy".into()));
        document.project.scenes.clear();
        assert!(!has_dependents(&document, "source"));
        assert!(Section::Retained.includes(&document, &take, None));
        assert!(!Section::Attached.includes(&document, &take, None));
        assert_eq!(
            document.view.takes.len(),
            1,
            "Presenting a releasable receipt must not silently delete it"
        );
    }
}
