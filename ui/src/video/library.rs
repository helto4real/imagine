//! Explicit local library selection; copies never become live project dependencies.
use super::*;
#[derive(Clone, Copy)]
pub(in crate::app) enum Reply {
    Catalog,
    Copied { replace: bool },
}
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    path: String,
    catalog: Option<Value>,
    title: String,
    search: String,
    reference: usize,
    selected: Option<String>,
    pub busy: bool,
    error: Option<String>,
    notice: Option<String>,
    owner_session: Option<String>,
}
impl State {
    pub(super) fn session(&mut self, id: &str) {
        if self.owner_session.as_deref().is_some_and(|old| old != id) {
            *self = Self {
                open: self.open,
                ..Default::default()
            };
        }
        self.owner_session = Some(id.into());
    }
    #[cfg(feature = "ui-preview")]
    pub(super) fn entries(&self) -> usize {
        self.catalog
            .as_ref()
            .and_then(|c| c["entries"].as_array())
            .map_or(0, Vec::len)
    }
    pub(super) fn path_from_dialog(&mut self, path: PathBuf) {
        self.path = path.display().to_string();
    }
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl ImagineApp {
    fn library_call(
        &mut self,
        ctx: &egui::Context,
        kind: Reply,
        post: bool,
        path: String,
        body: Option<Value>,
    ) {
        if self.timeline.busy || self.timeline.library.busy {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.library.busy = true;
        self.timeline.busy = true;
        self.timeline.library.error = None;
        self.timeline.library.notice = None;
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_request(&session, post, &path, body.as_ref()));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Library {
                    session,
                    epoch,
                    kind,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_library_reply(
        &mut self,
        ctx: &egui::Context,
        session: String,
        epoch: u64,
        kind: Reply,
        result: Result<Value, client::TimelineFailure>,
    ) {
        if self.timeline.session.as_deref() != Some(&session) || self.timeline.epoch != epoch {
            return;
        }
        self.timeline.library.busy = false;
        self.timeline.busy = false;
        match kind {
            Reply::Catalog => match result {
                Ok(value) => {
                    let entry = value["entry_id"].as_str().map(str::to_owned);
                    self.timeline.library.catalog = Some(if value["library"].is_object() {
                        value["library"].clone()
                    } else {
                        value
                    });
                    if let Some(entry) = entry {
                        self.timeline.library.selected = Some(entry);
                    }
                    self.timeline.library.notice =
                        Some("Library ready · project copies remain independent".into());
                }
                Err(error) => {
                    self.timeline.library.error = Some(if error.retryable {
                        format!(
                            "{} · operation unconfirmed; refresh the library before repeating a save.",
                            error.message
                        )
                    } else {
                        error.message
                    })
                }
            },
            Reply::Copied { replace } => {
                if result.is_ok() && replace {
                    self.timeline.folder = None;
                }
                self.timeline_message(
                    ctx,
                    Event::Document {
                        epoch,
                        replace,
                        before: None,
                        history: 0,
                        result,
                    },
                );
                self.timeline.library.error = self.timeline.error.clone();
                if self.timeline.library.error.is_none() {
                    self.timeline.library.notice = Some("Independent project copy created".into());
                }
            }
        }
    }
    fn library_select(&mut self, ctx: &egui::Context, mode: &str) {
        if self.timeline.busy {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.busy = true;
        self.timeline.library.busy = true;
        self.timeline.library.error = None;
        let path = self.timeline.library.path.clone();
        let mode = mode.to_owned();
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result=runtime().block_on(async{
                let location=client.timeline_request(&session,true,"/v1/video/locations",Some(&json!({"schema_version":1,"path":path}))).await?;
                client.timeline_request(&session,true,"/v1/video/libraries",Some(&json!({"schema_version":1,"location_id":location["location_id"],"mode":mode}))).await
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Library {
                    session,
                    epoch,
                    kind: Reply::Catalog,
                    result,
                }),
            );
        });
    }
    fn library_save(&mut self, ctx: &egui::Context, character: bool, replace: bool) {
        let (Some(catalog), Some(document)) =
            (&self.timeline.library.catalog, &self.timeline.document)
        else {
            return;
        };
        let mut body = json!({"schema_version":1,"expected_library_revision":catalog["revision"],"title":self.timeline.library.title});
        if replace {
            let Some(entry) = catalog["entries"].as_array().and_then(|list| {
                list.iter()
                    .find(|e| e["id"].as_str() == self.timeline.library.selected.as_deref())
            }) else {
                self.timeline.library.error = Some("Select the library entry to replace.".into());
                return;
            };
            body["replace"] =
                json!({"entry_id":entry["id"],"expected_entry_revision":entry["revision"]});
        }
        if character {
            let Some(reference) = document
                .project
                .creative
                .references
                .get(self.timeline.library.reference)
            else {
                self.timeline.library.error = Some("Select a project reference to save.".into());
                return;
            };
            let media = |asset: &AssetId| {
                document.raw["runtime_assets"][&asset.0]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        "Project media is unavailable; refresh the project and try again."
                            .to_owned()
                    })
            };
            let images = reference
                .images
                .iter()
                .map(|i| media(&i.asset))
                .collect::<Result<Vec<_>, _>>();
            let voice = reference
                .voice
                .as_ref()
                .map(|v| v.assets.iter().map(media).collect::<Result<Vec<_>, _>>())
                .transpose();
            let (images, voice) = match images.and_then(|i| voice.map(|v| (i, v))) {
                Ok(pair) => pair,
                Err(error) => {
                    self.timeline.library.error = Some(error);
                    return;
                }
            };
            body["character"] = json!({"name":reference.name,"alias":reference.alias,"description":reference.description,"tags":reference.tags,"visual_asset_ids":images,"voice":reference.voice.as_ref().map(|v|json!({"language":v.language,"description":v.description,"reference_asset_ids":voice.as_deref().unwrap_or_default()}))});
        } else {
            body["project_id"] = json!(document.project.id);
            body["expected_project_revision"] = json!(document.project.project_revision);
        }
        let route = format!(
            "/v1/video/libraries/{}/{}",
            catalog["id"].as_str().unwrap_or(""),
            if character { "characters" } else { "projects" }
        );
        self.library_call(ctx, Reply::Catalog, true, route, Some(body));
    }
    pub(super) fn timeline_library_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.library.open {
            return;
        }
        let mut state = std::mem::take(&mut self.timeline.library);
        let mut open = true;
        let mut action = None;
        let mut close = false;
        let document = self.timeline.document.clone();
        egui::Window::new("Project & Cast Library")
            .id(egui::Id::new("video-library"))
            .open(&mut open)
            .default_width(650.)
            .max_height((ctx.content_rect().height() - 70.).max(200.))
            .vscroll(true)
            .show(ctx, |ui| {
                if !helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
                    ui.label("Hidden library · hover to reveal");
                    return;
                }
                ui.weak("Choose a local library. Save explicitly; copies stay independent.");
                ui.horizontal(|ui| {
                    ui.label("Folder");
                    let r = ui.add(egui::TextEdit::singleline(&mut state.path).desired_width(380.));
                    control(ui, "library-path", &r);
                    let r = ui.add_enabled(!self.timeline.busy, egui::Button::new("Choose…"));
                    control(ui, "library-choose", &r);
                    if r.clicked() {
                        action = Some("choose");
                    }
                });
                ui.add_enabled_ui(!self.timeline.busy, |ui| {
                    ui.horizontal(|ui| {
                        for (name, label) in [
                            ("create", "Create library"),
                            ("open", "Open library"),
                            ("refresh", "Refresh"),
                        ] {
                            let r = ui.button(label);
                            control(ui, &format!("library-{name}"), &r);
                            if r.clicked() {
                                action = Some(name);
                            }
                        }
                    });
                });
                if let Some(catalog) = &state.catalog {
                    ui.separator();
                    ui.label(format!("Library revision {}", catalog["revision"]));
                    ui.horizontal(|ui| {
                        ui.label("Find");
                        ui.text_edit_singleline(&mut state.search);
                    });
                    if let Some(entries) = catalog["entries"].as_array() {
                        for (i, entry) in entries.iter().enumerate() {
                            let title = entry["title"].as_str().unwrap_or("");
                            if !title.to_lowercase().contains(&state.search.to_lowercase()) {
                                continue;
                            }
                            let id = entry["id"].as_str().unwrap_or("");
                            let kind = entry["content"]["kind"].as_str().unwrap_or("");
                            let r = ui.selectable_label(
                                state.selected.as_deref() == Some(id),
                                format!("{title} · {kind} · r{}", entry["revision"]),
                            );
                            control(ui, &format!("library-entry-{i}"), &r);
                            if r.clicked() {
                                state.selected = Some(id.into());
                                state.title = title.into();
                            }
                        }
                    }
                    ui.add_enabled_ui(!self.timeline.busy, |ui| {
                        let r = ui.button("Copy selected into project");
                        control(ui, "library-copy", &r);
                        if r.clicked() {
                            action = Some("copy");
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Entry title");
                        let r = ui.text_edit_singleline(&mut state.title);
                        control(ui, "library-title", &r);
                    });
                    if let Some(document) = &document {
                        egui::ComboBox::from_id_salt("library-reference")
                            .selected_text(
                                document
                                    .project
                                    .creative
                                    .references
                                    .get(state.reference)
                                    .map(|r| r.name.as_str())
                                    .unwrap_or("Select project reference"),
                            )
                            .show_ui(ui, |ui| {
                                for (i, r) in
                                    document.project.creative.references.iter().enumerate()
                                {
                                    ui.selectable_value(&mut state.reference, i, &r.name);
                                }
                            });
                        ui.add_enabled_ui(!self.timeline.busy, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for (name, label) in [
                                    ("save-project", "Save project as new"),
                                    ("replace-project", "Replace selected with project"),
                                    ("save-cast", "Save reference as new"),
                                    ("replace-cast", "Replace selected with reference"),
                                ] {
                                    let r = ui.button(label);
                                    control(ui, &format!("library-{name}"), &r);
                                    if r.clicked() {
                                        action = Some(name);
                                    }
                                }
                            });
                        });
                    }
                }
                if state.busy {
                    ui.spinner();
                }
                if let Some(error) = &state.error {
                    ui.colored_label(helto::ERROR, error);
                }
                if let Some(notice) = &state.notice {
                    ui.label(notice);
                }
                let r = ui.button("Close library");
                control(ui, "library-close", &r);
                close = r.clicked();
            });
        state.open = open && !close;
        self.timeline.library = state;
        match action {
            Some("choose") => {
                self.timeline.busy = true;
                self.timeline.library.busy = true;
                let tx = self.tx.clone();
                let ctx = ctx.clone();
                let epoch = self.timeline.epoch;
                std::thread::spawn(move || {
                    let path = rfd::FileDialog::new()
                        .set_title("Choose local library folder")
                        .pick_folder();
                    send(
                        &tx,
                        &ctx,
                        Message::Timeline(Event::LibraryFolder { epoch, path }),
                    );
                });
            }
            Some("create") => self.library_select(ctx, "create"),
            Some("open") => self.library_select(ctx, "open"),
            Some("refresh") => {
                if let Some(c) = &self.timeline.library.catalog {
                    self.library_call(
                        ctx,
                        Reply::Catalog,
                        false,
                        format!("/v1/video/libraries/{}", c["id"].as_str().unwrap_or("")),
                        None,
                    );
                }
            }
            Some("copy") => {
                if let Some(c) = &self.timeline.library.catalog
                    && let Some(entry) = c["entries"].as_array().and_then(|e| {
                        e.iter()
                            .find(|e| e["id"].as_str() == self.timeline.library.selected.as_deref())
                    })
                {
                    let body =
                        json!({"schema_version":1,"expected_entry_revision":entry["revision"]});
                    if entry["content"]["kind"] == "project" {
                        self.library_call(
                            ctx,
                            Reply::Copied { replace: true },
                            true,
                            format!(
                                "/v1/video/libraries/{}/entries/{}/project-copy",
                                c["id"].as_str().unwrap_or(""),
                                entry["id"].as_str().unwrap_or("")
                            ),
                            Some(body),
                        );
                    } else if let Some(document) = &self.timeline.document {
                        self.library_call(ctx,Reply::Copied{replace:false},true,format!("/v1/video/projects/{}/cast/from-library",document.project.id.0),Some(json!({"schema_version":1,"expected_project_revision":document.project.project_revision,"library_id":c["id"],"entry_id":entry["id"],"expected_entry_revision":entry["revision"]})));
                    } else {
                        self.timeline.library.error =
                            Some("Open a project for the character copy.".into());
                    }
                } else {
                    self.timeline.library.error = Some("Select a library entry to copy.".into());
                }
            }
            Some("save-project") => self.library_save(ctx, false, false),
            Some("replace-project") => self.library_save(ctx, false, true),
            Some("save-cast") => self.library_save(ctx, true, false),
            Some("replace-cast") => self.library_save(ctx, true, true),
            _ => {}
        }
    }
}
