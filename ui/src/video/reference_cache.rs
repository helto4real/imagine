//! Session cache controls. No inferred or thumbnail-derived usage statistics.
use super::*;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    nonce: u64,
    pending: bool,
    status: Option<Value>,
    limit_mib: u64,
    enabled: bool,
    error: Option<String>,
}
fn bytes_label(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1048576 {
        format!("{:.1} KiB", bytes as f64 / 1024.)
    } else if bytes < 1073741824 {
        format!("{:.1} MiB", bytes as f64 / 1048576.)
    } else {
        format!("{:.1} GiB", bytes as f64 / 1073741824.)
    }
}
impl State {
    #[cfg(feature = "ui-preview")]
    pub(super) fn pending(&self) -> bool {
        self.pending
    }
    pub fn begin(&mut self) -> u64 {
        self.nonce = self.nonce.wrapping_add(1);
        self.pending = true;
        self.error = None;
        self.nonce
    }
    pub fn receive(&mut self, nonce: u64, result: Result<Value, String>) {
        if !self.open || nonce != self.nonce {
            return;
        }
        self.pending = false;
        match result {
            Ok(value) => {
                self.limit_mib = value["limit_bytes"].as_u64().unwrap_or(0) / 1048576;
                self.enabled = value["mode"] != "off";
                self.status = Some(value);
                self.error = None;
            }
            Err(error) => {
                self.status = None;
                self.error = Some(error);
            }
        }
    }
    pub fn close(&mut self) {
        self.open = false;
        self.nonce = self.nonce.wrapping_add(1);
        self.pending = false;
    }
    pub fn show(&mut self, ctx: &egui::Context) -> Option<(bool, String, Option<Value>)> {
        if !self.open {
            return None;
        }
        #[cfg(feature = "ui-preview")]
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.close();
            return None;
        }
        let mut open = true;
        let mut action = None;
        egui::Window::new("Reference session cache").open(&mut open).default_width(390.).show(ctx,|ui|{
            let available=self.status.as_ref().is_some_and(|v|v["available"]==true);
            if let Some(error)=&self.error{ui.colored_label(helto::ERROR,error);}
            if let Some(value)=&self.status{if available{ui.label(format!("{} of {} · {} cached encodings · {} in progress",bytes_label(value["used_bytes"].as_u64().unwrap_or(0)),bytes_label(value["limit_bytes"].as_u64().unwrap_or(0)),value["entries"],value["busy"]));}else{ui.label("Reference encoding cache unavailable.");}}
            ui.add_enabled_ui(available && !self.pending,|ui|{
                let r=ui.checkbox(&mut self.enabled,"Session cache");
                #[cfg(feature="ui-preview")] super::qa::control(ui,"cache-enabled".into(),&r);
                let _=r;
                ui.horizontal(|ui|{ui.label("Memory limit (MiB)");let r=ui.add(egui::DragValue::new(&mut self.limit_mib).range(0..=u64::MAX/1048576));
                #[cfg(feature="ui-preview")] super::qa::control(ui,"cache-budget".into(),&r);
                let _=r;});
                let r=ui.button("Apply budget");
                #[cfg(feature="ui-preview")] super::qa::control(ui,"cache-apply".into(),&r);
                if r.clicked(){action=Some((true,"/v1/video/reference-cache".into(),Some(json!({"schema_version":1,"limit_bytes":self.limit_mib.saturating_mul(1048576),"enabled":self.enabled}))));}
                let r=helto::icon_button(ui,crate::icons::Icon::Trash,"Clear reference cache",false);
                #[cfg(feature="ui-preview")] super::qa::control(ui,"cache-clear".into(),&r);
                if r.clicked(){action=Some((true,"/v1/video/reference-cache/clear".into(),Some(json!({}))));}
            });
            ui.add_enabled_ui(!self.pending,|ui|{
                let r=helto::icon_button(ui,crate::icons::Icon::Retry,"Refresh reference cache",false);
                #[cfg(feature="ui-preview")] super::qa::control(ui,"cache-refresh".into(),&r);
                if r.clicked(){action=Some((false,"/v1/video/reference-cache".into(),None));}
            });
        });
        if !open {
            self.close();
        }
        action
    }
}
impl ImagineApp {
    pub(super) fn timeline_reference_cache_open(&mut self, ctx: &egui::Context) {
        self.timeline.reference_cache.open = true;
        self.timeline_reference_cache_call(ctx, false, "/v1/video/reference-cache".into(), None);
    }
    fn timeline_reference_cache_call(
        &mut self,
        ctx: &egui::Context,
        post: bool,
        path: String,
        body: Option<Value>,
    ) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        let nonce = self.timeline.reference_cache.begin();
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_request(&session, post, &path, body.as_ref()));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::ReferenceCache {
                    epoch,
                    nonce,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_reference_cache_window(&mut self, ctx: &egui::Context) {
        if let Some((post, path, body)) = self.timeline.reference_cache.show(ctx) {
            self.timeline_reference_cache_call(ctx, post, path, body);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_cache_ui_actual_frames_unavailable_and_connected() {
        for width in [1100., 600.] {
            for available in [false, true] {
                let ctx = egui::Context::default();
                helto::install(&ctx);
                let mut state = State {
                    open: true,
                    ..Default::default()
                };
                let nonce = state.begin();
                state.receive(nonce,Ok(json!({"available":available,"mode":"session","limit_bytes":536870912,"used_bytes":134217728,"entries":2,"busy":0})));
                let evidence = super::super::media_health::ui_evidence::capture(
                    &ctx,
                    &format!("reference-cache-{width}-{available}"),
                    width,
                    760.,
                    |ui| {
                        state.show(ui.ctx());
                    },
                );
                let nodes = evidence["nodes"].as_array().unwrap();
                for label in ["Apply budget", "Clear reference cache"] {
                    let node = nodes.iter().find(|n| n["label"] == label).unwrap();
                    assert_eq!(node["disabled"], !available);
                    let bounds = node["bounds"].as_array().unwrap();
                    assert!(
                        bounds[0].as_f64().unwrap() >= 0.
                            && bounds[2].as_f64().unwrap() <= f64::from(width)
                    );
                }
                assert!(
                    nodes
                        .iter()
                        .any(|n| n["label"] == "Reference encoding cache unavailable."
                            || n["value"] == "Reference encoding cache unavailable.")
                        != available
                );
            }
        }
    }
    #[test]
    fn reference_cache_ui_late_reply_after_close_is_ignored() {
        let mut s = State {
            open: true,
            ..Default::default()
        };
        let nonce = s.begin();
        s.close();
        s.receive(nonce, Ok(json!({"available":true})));
        assert!(s.status.is_none());
    }
}
