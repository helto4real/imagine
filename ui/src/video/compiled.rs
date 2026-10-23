//! Source compilation review, independent of model loading and queue admission.
use super::*;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    pub value: Option<Value>,
    pub error: Option<String>,
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl ImagineApp {
    pub(super) fn timeline_compiled(&mut self, ctx: &egui::Context) {
        if self.timeline.busy {
            return;
        }
        let (Some(client), Some(session), Some(doc), Some(scene)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.as_ref(),
            self.timeline.scene(),
        ) else {
            return;
        };
        let path = format!(
            "/v1/video/projects/{}/scenes/{}/compiled",
            doc.project.id.0, scene.id.0
        );
        let epoch = self.timeline.epoch;
        self.timeline.busy = true;
        self.timeline.compiled = State {
            open: true,
            ..Default::default()
        };
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(&session, false, &path, None));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Compiled {
                    session,
                    epoch,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_compiled_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.compiled.open {
            return;
        }
        let mut state = std::mem::take(&mut self.timeline.compiled);
        let mut open = true;
        let mut close = false;
        let mut refresh = false;
        egui::Window::new("Plan Review").id(egui::Id::new("video-compiled")).open(&mut open).default_width(810.).default_height(590.).max_height((ctx.content_rect().height()-70.).max(200.)).vscroll(true).show(ctx,|ui|{
            if !helto::reveal_in_viewport(ui,self.prefs.privacy_mode){ui.label("Hidden plan · hover to reveal");return;}
            ui.horizontal(|ui|{ui.weak("Prepared source · generation has not started");let r=ui.add_enabled(!self.timeline.busy,egui::Button::new("Refresh plan"));control(ui,"compiled-refresh",&r);refresh=r.clicked();let r=ui.button("Close review");control(ui,"compiled-close",&r);close=r.clicked();});
            if let Some(error)=&state.error{ui.colored_label(helto::ERROR,error);}
            if let Some(plan)=&state.value{
                let stale=self.timeline.document.as_ref().is_none_or(|d|json!(d.project.id)!=plan["project_id"]||json!(d.project.project_revision)!=plan["project_revision"]);
                if stale{ui.colored_label(helto::ACCENT,"Project changed since this plan. Refresh to see current inputs.");}
                let mode=match plan["metadata"]["format"].as_str(){Some("full_reference_six_sections")=>"H3 Reference",Some("base_three_fields")=>"H3 I2V",_=>"H3"};
                ui.label(format!("{mode} · revision {}",plan["project_revision"]));
                if let Some(frames)=plan["effective_shape"]["frames"].as_u64(){ui.label(format!("{} × {} · {} frames / 24 fps · {:.3} s",plan["effective_shape"]["width"],plan["effective_shape"]["height"],frames,frames as f64/24.));}
                for issue in plan["runtime_issues"].as_array().into_iter().flatten(){let code=issue.as_str().unwrap_or("");ui.colored_label(helto::ERROR,match code{"h3_voice_reference_unavailable"=>"This worker does not yet accept voice audio references.","h3_reference_media_unavailable"=>"This worker does not yet accept character reference pictures.","h3_section_reference_unavailable"=>"This worker does not yet accept all selected Section image roles.",_=>code});}
                ui.label("Compiled prompt");let mut prompt=plan["prompt"].as_str().unwrap_or("");let r=ui.add(egui::TextEdit::multiline(&mut prompt).code_editor().desired_rows(15).desired_width(f32::INFINITY));control(ui,"compiled-prompt",&r);
                let header=ui.collapsing("Input bindings",|ui|{for key in ["subjects","images","audios","videos"]{for input in plan["reference_manifest"][key].as_array().into_iter().flatten(){ui.label(format!("{} · {} · {}",input["prompt_label"].as_str().unwrap_or(""),input["role"].as_str().or_else(||input["reference_kind"].as_str()).unwrap_or(key),input["asset_id"].as_str().or_else(||input["source_alias"].as_str()).unwrap_or("")));}}});control(ui,"compiled-bindings",&header.header_response);
                if plan["metadata"]["format"]=="base_three_fields"&&let Some(refs)=plan["text_references"].as_array()&&!refs.is_empty(){ui.weak("I2V cast descriptions and voice guidance are text conditioning; attached cast media are not identity/timbre inputs in this mode.");}
            }else if self.timeline.busy{ui.spinner();}
        });
        state.open = open && !close;
        self.timeline.compiled = state;
        if refresh {
            self.timeline_compiled(ctx);
        }
    }
}
