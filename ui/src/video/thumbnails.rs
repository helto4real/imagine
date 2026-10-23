//! Session-only image thumbnails shared by reference cards and contextual selectors.
use super::*;
type Key = (ProjectId, AssetId);
#[derive(Default)]
pub(super) struct State {
    values: std::collections::BTreeMap<Key, Result<egui::TextureHandle, String>>,
    loading: std::collections::BTreeMap<Key, u64>,
    versions: std::collections::BTreeMap<Key, u64>,
}
impl State {
    fn expect(&mut self, id: &Key, revision: u64) {
        if self.versions.insert(id.clone(), revision) != Some(revision) {
            self.values.remove(id);
        }
    }
    fn accepts(&mut self, id: &Key, revision: u64) -> bool {
        if self.loading.get(id) != Some(&revision) {
            return false;
        }
        self.loading.remove(id);
        self.versions.get(id) == Some(&revision)
    }
}
impl ImagineApp {
    pub(super) fn timeline_thumbnail(&mut self, ui: &mut egui::Ui, d: &Snapshot, asset: &Asset) {
        if asset.kind != MediaKind::Image {
            return;
        }
        let path = d
            .runtime_asset(&asset.id.0)
            .map(|id| format!("/v1/assets/{id}"));
        self.timeline_thumbnail_path(ui, &d.project.id, asset, path);
    }
    pub(super) fn timeline_thumbnail_path(
        &mut self,
        ui: &mut egui::Ui,
        project: &ProjectId,
        asset: &Asset,
        path: Option<String>,
    ) {
        let state = &mut self.timeline.thumbnails;
        let key = (project.clone(), asset.id.clone());
        let (rect, _) = ui.allocate_exact_size(egui::vec2(112., 80.), egui::Sense::hover());
        if !helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
            ui.painter().rect_filled(rect, 4., helto::FIELD);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Hidden image",
                egui::FontId::proportional(11.),
                helto::DIM,
            );
            return;
        }
        if !ui.is_rect_visible(rect) {
            return;
        }
        state.expect(&key, asset.revision);
        if let Some(value) = state.values.get(&key) {
            match value {
                Ok(texture) => {
                    let size = texture.size_vec2();
                    let scale = (rect.width() / size.x).min(rect.height() / size.y);
                    let dst = egui::Rect::from_center_size(rect.center(), size * scale);
                    ui.painter().image(
                        texture.id(),
                        dst,
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                        egui::Color32::WHITE,
                    );
                }
                Err(message) => {
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Preview unavailable",
                        egui::FontId::proportional(11.),
                        helto::DIM,
                    );
                    ui.interact(rect, ui.id().with(&asset.id.0), egui::Sense::hover())
                        .on_hover_text(message);
                }
            }
            return;
        }
        if state.loading.contains_key(&key) || state.loading.len() >= 4 {
            return;
        }
        let (Some(client), Some(session), Some(path)) =
            (self.client.clone(), self.timeline.session.clone(), path)
        else {
            return;
        };
        state.loading.insert(key, asset.revision);
        let revision = asset.revision;
        let epoch = self.timeline.epoch;
        let id = asset.id.clone();
        let project = project.clone();
        let tx = self.tx.clone();
        let ctx = ui.ctx().clone();
        std::thread::spawn(move || {
            let result = runtime()
                .block_on(client.timeline_bytes_path(&session, &path))
                .map_err(|e| e.message)
                .and_then(|bytes| client::decode(&bytes, true).map_err(|e| e.to_string()));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Thumbnail {
                    epoch,
                    project,
                    asset: id,
                    revision,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_thumbnail_reply(
        &mut self,
        ctx: &egui::Context,
        project: ProjectId,
        asset: AssetId,
        revision: u64,
        result: Result<client::Preview, String>,
    ) {
        let state = &mut self.timeline.thumbnails;
        let key = (project.clone(), asset.clone());
        if !state.accepts(&key, revision) {
            return;
        }
        state.values.insert(
            key,
            result.map(|p| {
                ctx.load_texture(
                    format!("project-thumbnail-{}-{}", project.0, asset.0),
                    egui::ColorImage::from_rgba_unmultiplied([p.width, p.height], &p.pixels),
                    egui::TextureOptions::LINEAR,
                )
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relink_thumbnail_ignores_old_asset_revision_and_duplicate_reply() {
        let id = (
            ProjectId("own-project".into()),
            AssetId("stable-asset".into()),
        );
        let mut state = State::default();
        state.expect(&id, 1);
        state.loading.insert(id.clone(), 1);
        state.expect(&id, 2);
        assert!(!state.accepts(&id, 1));
        state.loading.insert(id.clone(), 2);
        assert!(state.accepts(&id, 2));
        assert!(!state.accepts(&id, 2));
        state.loading.insert(id.clone(), 2);
        state = State::default();
        assert!(!state.accepts(&id, 2));
    }
    #[test]
    fn simultaneous_project_and_library_thumbnails_do_not_evict_each_other() {
        let project = (ProjectId("own-project".into()), AssetId("image".into()));
        let library = (ProjectId("library-own".into()), AssetId("image".into()));
        let mut state = State::default();
        for key in [&project, &library] {
            state.expect(key, 1);
            state.loading.insert(key.clone(), 1);
        }
        assert!(state.accepts(&project, 1));
        state
            .values
            .insert(project.clone(), Err("own marker".into()));
        assert!(state.accepts(&library, 1));
        state
            .values
            .insert(library.clone(), Err("other own marker".into()));
        state.expect(&project, 2);
        assert!(!state.values.contains_key(&project));
        assert!(state.values.contains_key(&library));
    }
}
