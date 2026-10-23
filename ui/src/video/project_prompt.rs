//! Project-scoped global prompt. Turning it off retains its authored text.
use super::*;
use imagine_video_contract::creative::GlobalPromptOptions;
impl ImagineApp {
    pub(super) fn timeline_project_prompt_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.project_prompt_open || self.timeline.document.is_none() {
            return;
        }
        let creative = self
            .timeline
            .document
            .as_ref()
            .unwrap()
            .project
            .creative
            .clone();
        let mut open = true;
        let mut close = false;
        let mut commit = false;
        egui::Window::new("Project prompt")
            .id(egui::Id::new("project-prompt"))
            .open(&mut open)
            .default_width(540.)
            .min_size(egui::vec2(360., 220.))
            .show(ctx, |ui| {
                ui.add_enabled_ui(!self.timeline.busy, |ui| {
                    let options = self.timeline.global_prompt_options.get_or_insert_with(|| {
                        GlobalPromptOptions {
                            enabled: false,
                            position: "Prefix".into(),
                        }
                    });
                    let r = ui.checkbox(&mut options.enabled, "Use global prompt");
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "global-prompt-enabled".into(), &r);
                    commit |= r.changed();
                    ui.horizontal(|ui| {
                        for (id, label) in [("Prefix", "Prefix"), ("Suffix", "Suffix")] {
                            let r = ui.selectable_value(&mut options.position, id.into(), label);
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, format!("global-prompt-{id}"), &r);
                            commit |= r.changed();
                        }
                    });
                    let r = prompt_tools::textarea(
                        ui,
                        &mut self.timeline.draft_global,
                        5,
                        "Project global prompt",
                        self.prefs.privacy_mode,
                        true,
                        prompt_tools::Source::project(
                            &creative,
                            &self.timeline.document.as_ref().unwrap().project.id,
                        ),
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "global-prompt-text".into(), &r);
                    commit |= r.lost_focus();
                    let r = helto::icon_button(ui, crate::icons::Icon::Close, "Close", false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "global-prompt-close".into(), &r);
                    close = r.clicked();
                });
            });
        self.timeline.project_prompt_open = open && !close;
        if commit || !self.timeline.project_prompt_open {
            let edits = self.timeline.draft_edits();
            self.timeline_edit(ctx, edits, 0);
        }
    }
}
