//! Session-only editing state. Jobs and uploads use the shared GenerationDesk lifecycle.
use super::*;

pub(super) struct EditSession {
    pub desk: GenerationDesk,
    pub versions: Vec<EditVersion>,
    pub selected: usize,
    pub next_reference_number: usize,
    pub open: bool,
}
pub(super) struct EditVersion {
    pub result: Results,
    /// Index of the version used as source, including edits branching from Original.
    pub source: Option<usize>,
}
impl Default for EditSession {
    fn default() -> Self {
        Self {
            desk: GenerationDesk::default(),
            versions: Vec::new(),
            selected: 0,
            next_reference_number: 1,
            open: false,
        }
    }
}
impl ImagineApp {
    pub(super) fn can_edit_image(&self) -> bool {
        self.workspace == Workspace::Image
            && self.queue.loading.is_none()
            && !self.busy()
            && !self.connecting
            && self.image.pending.is_none()
            && self
                .image
                .selected_result()
                .is_some_and(|r| !r.asset_id.is_empty())
    }
    pub(super) fn start_editing(&mut self) {
        if !self.can_edit_image() {
            return;
        }
        if self.edit.versions.is_empty() {
            let original = self.image.result.clone().unwrap();
            self.edit.desk.result = Some(original.clone());
            self.edit.versions.push(EditVersion {
                result: original,
                source: None,
            });
        }
        self.edit.open = true;
        self.workspace = Workspace::ImageEdit;
    }
    pub(super) fn finish_editing(&mut self) {
        if self.workspace != Workspace::ImageEdit || self.busy() || self.edit.desk.pending.is_some()
        {
            return;
        }
        self.image.result = self.edit.desk.result.clone();
        self.image.status = "Saved".into();
        self.image.last_elapsed = None;
        self.image.actual_pixels = false;
        self.image.image_pan = egui::Vec2::ZERO;
        self.edit.open = false;
        self.workspace = Workspace::Image;
    }
    pub(super) fn select_edit_version(&mut self, index: usize) {
        if self.busy() || self.edit.desk.pending.is_some() {
            return;
        }
        if let Some(version) = self.edit.versions.get(index) {
            self.edit.selected = index;
            self.edit.desk.result = Some(version.result.clone());
            self.edit.desk.actual_pixels = false;
            self.edit.desk.image_pan = egui::Vec2::ZERO;
        }
    }
    pub(super) fn edit_comparison_source(&self) -> Option<&ResultView> {
        if self.workspace != Workspace::ImageEdit {
            return None;
        }
        let source = self.edit.versions.get(self.edit.selected)?.source?;
        self.edit
            .versions
            .get(source)?
            .result
            .assets
            .iter()
            .find(|asset| asset.role == OutputRole::Image)
    }
    pub(super) fn edit_request(&self) -> anyhow::Result<Value> {
        let source = self
            .edit
            .desk
            .selected_result()
            .ok_or_else(|| anyhow::anyhow!("Select an image to edit."))?;
        let [width, height] = source.texture.size();
        let references = self
            .edit
            .desk
            .references
            .iter()
            .map(|r| (r.number, r.id.clone()))
            .collect::<Vec<_>>();
        client::make_edit_request(
            &self.edit.desk.prompt,
            &self.prefs.negative_prompt,
            client::EditSource {
                asset_id: &source.asset_id,
                width: width as u32,
                height: height as u32,
            },
            &references,
            self.metadata
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Reconnect to load model settings."))?,
            &self.prefs.generation,
        )
    }
}
