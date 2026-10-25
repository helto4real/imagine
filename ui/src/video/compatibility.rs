//! Optional older-format export, never a restriction on normal saving/generation.
use super::*;
impl ImagineApp {
    pub(super) fn timeline_compatibility_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.compatibility_open {
            return;
        }
        let mut open = true;
        let mut export = false;
        let mut close = false;
        egui::Window::new("Older-compatible project copy").id(egui::Id::new("compatible-project-copy")).open(&mut open).default_width(520.).show(ctx,|ui|{
let _canvas = helto::window_canvas(ui);
            ui.scope(|ui| {
                if let Some(report)=&self.timeline.compatibility_report{
                    ui.strong("Copy exported. The original project is still active.");
                    ui.label(format!("Scene settings resolved: {}. Newer Enhance drafts kept only in the original: {}. Inactive sampling presets kept only in the original: {}. Mixed media ordering kept only in the original: {}.",report["materialized_settings"],report["retained_newer_authoring_drafts_in_original"],report["retained_sampling_drafts_in_original"],report["omitted_section_media_orders"]));
                }else{
                    ui.label("Create a separate copy readable by older Imagine versions. The original project keeps all data and remains active.");
                    ui.label("Project defaults become concrete scene settings. New media labels, mixed image/video ordering, library links, image-role annotations, inactive sampling presets and newer Enhance drafts remain in the original.");
                    ui.label("Existing media and Takes keep their contents and provenance. Scene-type labels and inactive clip bindings remain in the original. Older engines cannot reproduce features they do not support, and use their own seed controls. New boundary settings and applied conditioning receipts require the current format.");
                    let r=helto::icon_button(ui,crate::icons::Icon::Export,if self.timeline.private_storage {"Export private older-compatible copy"} else {"Choose export folder…"},true);#[cfg(feature="ui-preview")]super::qa::control(ui,"compatible-choose-folder".into(),&r);export=r.clicked();
                }
                let r=helto::icon_button(ui,crate::icons::Icon::Close,if self.timeline.compatibility_report.is_some(){"Close"}else{"Cancel"},false);#[cfg(feature="ui-preview")]super::qa::control(ui,"compatible-close".into(),&r);close=r.clicked();
            });
        });
        self.timeline.compatibility_open = open && !close && !export;
        if export {
            self.timeline_choose_folder(ctx, FolderMode::ExportCompatible);
        }
    }
}
