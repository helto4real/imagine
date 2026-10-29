//! The server's execution choices in Settings: read for each server session,
//! saved on change; a change applies to jobs started after it.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Attention {
    #[default]
    Auto,
    KitchenInt8,
    PytorchSdpa,
}
impl Attention {
    const ALL: [Self; 3] = [Self::Auto, Self::KitchenInt8, Self::PytorchSdpa];
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::KitchenInt8 => "Comfy Kitchen INT8",
            Self::PytorchSdpa => "PyTorch SDPA",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Settings {
    pub attention: Attention,
    pub dynamic_vram: bool,
}
#[derive(Default)]
pub(super) struct State {
    /// The server session these choices were read from.
    session: Option<String>,
    settings: Option<Settings>,
    pending: bool,
    error: Option<String>,
}
impl ImagineApp {
    pub(super) fn execution_settings(&mut self, ui: &mut egui::Ui) {
        ui.label(helto::section("Execution"));
        let session = self.timeline.server_session();
        if session.is_none() {
            ui.weak("Connect to Imagine to choose how jobs run.");
            return;
        }
        if self.execution.session != session && !self.execution.pending {
            self.execution.session = session.clone();
            self.execution.settings = None;
            self.execution_request(ui.ctx(), None);
        }
        let Some(mut settings) = self.execution.settings else {
            ui.weak("Reading the server's choices…");
            return;
        };
        ui.horizontal(|ui| {
            let label = ui.label("Attention");
            ui.add_enabled_ui(!self.execution.pending, |ui| {
                egui::ComboBox::from_id_salt("execution-attention")
                    .selected_text(settings.attention.label())
                    .show_ui(ui, |ui| {
                        for choice in Attention::ALL {
                            ui.selectable_value(&mut settings.attention, choice, choice.label());
                        }
                    })
                    .response
                    .labelled_by(label.id)
            });
        });
        ui.weak(match settings.attention {
            Attention::Auto => {
                "Comfy Kitchen INT8 where the GPU supports it, otherwise PyTorch SDPA."
            }
            Attention::KitchenInt8 => "Jobs fail by name on a GPU without Comfy Kitchen INT8.",
            Attention::PytorchSdpa => {
                "A checkpoint's own attention preference still applies, as in ComfyUI."
            }
        });
        ui.add_enabled_ui(!self.execution.pending, |ui| {
            ui.checkbox(&mut settings.dynamic_vram, "Dynamic VRAM");
        });
        ui.weak(if settings.dynamic_vram {
            "Diffusion models bring each layer into VRAM when it runs and release it under memory pressure, as ComfyUI."
        } else {
            "Diffusion models are placed in VRAM once, before sampling."
        });
        ui.weak("Applies to jobs started after the change.");
        if let Some(error) = &self.execution.error {
            ui.colored_label(helto::ERROR, error);
        }
        if Some(settings) != self.execution.settings {
            self.execution_request(ui.ctx(), Some(settings));
        }
    }
    /// Reads the choices, or saves `settings`, off the UI thread.
    fn execution_request(&mut self, ctx: &egui::Context, settings: Option<Settings>) {
        let (Some(client), Some(session)) = (self.client.clone(), self.execution.session.clone())
        else {
            return;
        };
        self.execution.pending = true;
        self.execution.error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let body = settings.map(|s| serde_json::to_value(s).unwrap());
            let result = runtime()
                .block_on(client.timeline_request(
                    &session,
                    body.is_some(),
                    "/v1/runtime/execution",
                    body.as_ref(),
                ))
                .map_err(|failure| failure.message)
                .and_then(|value| {
                    serde_json::from_value(value)
                        .map_err(|_| "Imagine returned unreadable execution settings.".into())
                });
            send(&tx, &ctx, Message::Execution { session, result });
        });
    }
    pub(super) fn execution_reply(&mut self, session: String, result: Result<Settings, String>) {
        if self.execution.session.as_deref() != Some(&session) {
            return;
        }
        self.execution.pending = false;
        match result {
            Ok(settings) => self.execution.settings = Some(settings),
            Err(error) => self.execution.error = Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn choices_use_the_server_names() {
        assert_eq!(
            serde_json::to_value(Settings {
                attention: Attention::PytorchSdpa,
                dynamic_vram: false,
            })
            .unwrap(),
            serde_json::json!({"attention":"pytorch_sdpa","dynamic_vram":false})
        );
        let read: Settings = serde_json::from_value(
            serde_json::json!({"attention":"kitchen_int8","dynamic_vram":true}),
        )
        .unwrap();
        assert_eq!(read.attention, Attention::KitchenInt8);
        assert!(read.dynamic_vram);
    }
}
