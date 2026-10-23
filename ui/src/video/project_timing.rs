//! Explicit duration edit: closing/cancel preserves the project and the raw draft.
use super::*;
#[derive(Default)]
pub(super) struct State {
    open: bool,
    project: Option<ProjectId>,
    text: String,
    error: Option<String>,
}
impl State {
    pub(super) fn open(&mut self, p: &VideoProject) {
        self.open = true;
        self.project = Some(p.id.clone());
        self.text = group_commands::duration(p).to_string();
        self.error = None;
    }
}
impl ImagineApp {
    pub(super) fn timeline_project_timing_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.project_timing.open {
            return;
        }
        if self.timeline.document.as_ref().map(|d| &d.project.id)
            != self.timeline.project_timing.project.as_ref()
        {
            self.timeline.project_timing.open = false;
            return;
        }
        let mut state = std::mem::take(&mut self.timeline.project_timing);
        let mut open = state.open;
        let mut request = None;
        egui::Window::new("Timeline settings").id(egui::Id::new("project-timing")).open(&mut open).collapsible(false).resizable(false).default_width(360.).show(ctx,|ui|{
            ui.add_enabled_ui(!self.timeline.busy,|ui|{
                ui.label("Duration (seconds)");
                let r=ui.add(egui::TextEdit::singleline(&mut state.text).desired_width(ui.available_width()));
                #[cfg(feature="ui-preview")]
                super::qa::control(ui,"project-timeline-duration".into(),&r);
                let _=r;
                ui.label("Sections keep their timing. Fit Last or Fit All can bring overflow into this duration.");
                if let Some(error)=&state.error{ui.colored_label(helto::ERROR,error);}
                ui.horizontal(|ui|{
                    let r=ui.button("Apply");
                    #[cfg(feature="ui-preview")]
                    super::qa::control(ui,"project-timeline-apply".into(),&r);
                    if r.clicked(){match state.text.trim().parse::<f64>().ok().filter(|v|v.is_finite()&&*v>0.).map(editing::time).filter(|v|v.positive()) {Some(value)=>request=Some(Some(value)),None=>state.error=Some("Enter a positive duration in seconds.".into())}}
                    let r=ui.button("Cancel");
                    #[cfg(feature="ui-preview")]
                    super::qa::control(ui,"project-timeline-cancel".into(),&r);
                    if r.clicked(){state.open=false;}
                    if helto::icon_button(ui,crate::icons::Icon::Fit,"Follow content length",false).clicked(){request=Some(None);}
                });
            });
        });
        state.open &= open;
        if let Some(value) = request {
            state.open = false;
            let mut creative = self
                .timeline
                .document
                .as_ref()
                .unwrap()
                .project
                .creative
                .clone();
            creative.timeline_duration = value;
            let mut edits = self.timeline.draft_edits();
            edits.push(Edit::SetCreative { creative });
            self.timeline_edit(ctx, edits, 0);
        }
        self.timeline.project_timing = state;
    }
}
