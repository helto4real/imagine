//! Explicit local library selection; copies never become live project dependencies.
use super::*;
use std::collections::BTreeMap;
#[derive(Clone, Copy)]
pub(in crate::app) enum Reply {
    Catalog,
    Opened,
    Copied { replace: bool },
}

fn catalog_error(kind: Reply, code: &str, editing_character: bool, message: String) -> String {
    if matches!(kind, Reply::Opened) {
        match code {
            "storage_io_failed" | "storage_error" | "resource_missing" => {
                return "Could not open the library files. Check that the folder and its media are available, then retry Open library.".into();
            }
            "invalid_storage_root" | "storage_root_changed" => {
                return "The selected library folder is unavailable. Choose the library folder again.".into();
            }
            _ => {}
        }
    }
    if editing_character && matches!(code, "storage_io_failed" | "storage_error") {
        "Could not save the library character. Check the library folder and retry Save.".into()
    } else {
        message
    }
}
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    path: String,
    pub(super) catalog: Option<Value>,
    title: String,
    search: String,
    reference: usize,
    tab: String,
    recent: bool,
    description: String,
    asset: Option<AssetId>,
    saving_reference: Option<(ProjectId, String)>,
    pub(super) editor: Option<library_character::Editor>,
    selected: Option<String>,
    pub busy: bool,
    pub(super) error: Option<String>,
    notice: Option<String>,
    owner_session: Option<String>,
}
impl State {
    pub(super) fn selected_id(&self) -> Option<String> {
        self.catalog
            .as_ref()?
            .get("id")?
            .as_str()
            .map(str::to_owned)
    }
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

fn project_asset_name(project: &VideoProject, asset: &Asset) -> String {
    if !asset.name.is_empty() {
        return asset.name.clone();
    }
    let ordinal = project
        .assets
        .values()
        .filter(|a| a.kind == asset.kind)
        .position(|a| a.id == asset.id)
        .map_or(1, |i| i + 1);
    format!("{:?} {ordinal}", asset.kind)
}
fn thumbnail_resource(entry: &Value) -> Option<(String, String)> {
    let doc = &entry["content"]["document"];
    let resource = match entry["content"]["kind"].as_str()? {
        "asset" if doc["asset"]["kind"] == "image" => &doc["resource"],
        "character" => {
            let id = doc["visual_ids"].as_array()?.first()?;
            doc["resources"]
                .as_array()?
                .iter()
                .find(|r| &r["id"] == id)?
        }
        "project" => doc["resources"].as_array()?.iter().find(|r| {
            r["kind"]["kind"] == "reference"
                && doc["project"]["assets"][r["kind"]["asset_id"].as_str().unwrap_or("")]["kind"]
                    == "image"
        })?,
        _ => return None,
    };
    Some((
        resource["id"].as_str()?.into(),
        entry["title"].as_str().unwrap_or("").into(),
    ))
}

impl ImagineApp {
    pub(super) fn library_call(
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
            Reply::Catalog | Reply::Opened => match result {
                Ok(value) => {
                    if value["entry_id"].is_string()
                        && self
                            .timeline
                            .library
                            .editor
                            .as_ref()
                            .is_some_and(|editor| editor.saving)
                    {
                        self.timeline.library.editor = None;
                    }
                    let entry = value["entry_id"].as_str().map(str::to_owned);
                    self.timeline.library.catalog = Some(if value["library"].is_object() {
                        value["library"].clone()
                    } else {
                        value
                    });
                    if let Some(entry) = entry {
                        self.timeline.library.selected = Some(entry.clone());
                        if let Some(saved) = self
                            .timeline
                            .library
                            .catalog
                            .as_ref()
                            .and_then(|catalog| catalog["entries"].as_array())
                            .and_then(|rows| rows.iter().find(|saved| saved["id"] == entry))
                        {
                            self.timeline.library.title =
                                saved["title"].as_str().unwrap_or("").into();
                            self.timeline.library.description =
                                saved["content"]["document"]["description"]
                                    .as_str()
                                    .unwrap_or("")
                                    .into();
                        }
                        if let Some((project_id, reference_id)) =
                            self.timeline.library.saving_reference.take()
                            && let Some(document) = &self.timeline.document
                            && document.project.id == project_id
                            && let Some(catalog) = &self.timeline.library.catalog
                            && let Some(saved) = catalog["entries"]
                                .as_array()
                                .and_then(|rows| rows.iter().find(|row| row["id"] == entry))
                        {
                            let mut creative = document.project.creative.clone();
                            self.timeline
                                .cast
                                .merge_pending(&document.project, &mut creative);
                            if let Some(reference) = creative
                                .references
                                .iter_mut()
                                .find(|r| r.id == reference_id)
                            {
                                reference.origin = Some(imagine_video_contract::creative::Origin {
                                    library_id: catalog["id"].as_str().unwrap_or("").into(),
                                    entry_id: entry,
                                    revision: saved["revision"].as_u64().unwrap_or(0),
                                });
                                self.timeline_edit(ctx, vec![Edit::SetCreative { creative }], 0);
                            }
                        }
                    }
                    self.timeline.library.notice = Some(
                        if self.timeline.private_storage {
                            "Library ready · original media stay linked"
                        } else {
                            "Library ready · project copies remain independent"
                        }
                        .into(),
                    );
                }
                Err(error) => {
                    let message = catalog_error(
                        kind,
                        &error.code,
                        self.timeline.library.editor.is_some(),
                        error.message,
                    );
                    if let Some(editor) = &mut self.timeline.library.editor {
                        editor.saving = false;
                    }
                    self.timeline.library.saving_reference = None;
                    self.timeline.library.error = Some(
                        if error.retryable
                            && !matches!(error.code.as_str(), "storage_io_failed" | "storage_error")
                        {
                            format!(
                                "{} · operation unconfirmed; refresh the library before repeating a save.",
                                message
                            )
                        } else {
                            message
                        },
                    )
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
                    self.timeline.library.notice = Some(
                        if self.timeline.private_storage {
                            "Private project copy created; original media stay linked"
                        } else {
                            "Independent project copy created"
                        }
                        .into(),
                    );
                }
            }
        }
    }
    pub(super) fn library_select(&mut self, ctx: &egui::Context, mode: &str) {
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
        let private_storage = self.timeline.private_storage;
        let mode = mode.to_owned();
        let kind = if mode == "open" {
            Reply::Opened
        } else {
            Reply::Catalog
        };
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                let mut body = json!({"schema_version":1,"mode":mode});
                if !private_storage {
                    let location = client
                        .timeline_request(
                            &session,
                            true,
                            "/v1/video/locations",
                            Some(&json!({"schema_version":1,"path":path})),
                        )
                        .await?;
                    body["location_id"] = location["location_id"].clone();
                }
                client
                    .timeline_request(&session, true, "/v1/video/libraries", Some(&body))
                    .await
            });
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
    pub(super) fn library_reference(
        &mut self,
        ctx: &egui::Context,
        index: usize,
        reference: &imagine_video_contract::creative::VisualReference,
    ) {
        let state = &mut self.timeline.library;
        state.reference = index;
        state.title = reference.name.clone();
        state.description = reference.description.clone();
        state.tab = "character".into();
        state.selected = reference
            .origin
            .as_ref()
            .filter(|o| {
                Some(o.library_id.as_str()) == state.catalog.as_ref().and_then(|c| c["id"].as_str())
            })
            .filter(|o| {
                state
                    .catalog
                    .as_ref()
                    .and_then(|c| c["entries"].as_array())
                    .is_some_and(|rows| rows.iter().any(|row| row["id"] == o.entry_id))
            })
            .map(|o| o.entry_id.clone());
        state.open = true;
        if state.catalog.is_some() {
            let replace = state.selected.is_some();
            if self.timeline.busy {
                self.timeline.after_edit = Some(PostAction::LibrarySave {
                    character: true,
                    replace,
                });
            } else {
                self.library_save(ctx, true, replace);
            }
        }
    }
    pub(super) fn library_save(&mut self, ctx: &egui::Context, character: bool, replace: bool) {
        let pending = self.timeline.draft_edits();
        if !pending.is_empty() {
            self.timeline.after_edit = Some(PostAction::LibrarySave { character, replace });
            self.timeline_edit(ctx, pending, 0);
            return;
        }
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
        let asset = !character && self.timeline.library.tab == "asset";
        if asset {
            let Some(id) = &self.timeline.library.asset else {
                self.timeline.library.error = Some("Select project media to save.".into());
                return;
            };
            body["asset_id"] = json!(id);
            body["project_id"] = json!(document.project.id);
            body["expected_project_revision"] = json!(document.project.project_revision);
        } else if character {
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
            self.timeline.library.saving_reference =
                Some((document.project.id.clone(), reference.id.clone()));
            let mut portable = reference.clone();
            for (image, media) in portable.images.iter_mut().zip(&images) {
                image.asset = AssetId(media.clone());
            }
            if let Some(profile) = &mut portable.voice {
                let mut settings = BTreeMap::new();
                for (asset, media) in profile
                    .assets
                    .iter_mut()
                    .zip(voice.as_deref().unwrap_or_default())
                {
                    let mapped = AssetId(media.clone());
                    if let Some(value) = profile.reference_settings.get(asset) {
                        settings.insert(mapped.clone(), value.clone());
                    }
                    *asset = mapped;
                }
                profile.reference_settings = settings;
            }
            body["character"] = json!({"name":reference.name,"alias":reference.alias,"description":reference.description,"tags":reference.tags,"visual_asset_ids":images,"voice":reference.voice.as_ref().map(|v|json!({"language":v.language,"description":v.description,"reference_asset_ids":voice.as_deref().unwrap_or_default()})),"reference":portable});
        } else {
            body["project_id"] = json!(document.project.id);
            body["expected_project_revision"] = json!(document.project.project_revision);
        }
        let route = format!(
            "/v1/video/libraries/{}/{}",
            catalog["id"].as_str().unwrap_or(""),
            if character {
                "characters"
            } else if asset {
                "assets"
            } else {
                "projects"
            }
        );
        self.library_call(ctx, Reply::Catalog, true, route, Some(body));
    }
    fn library_edit(&mut self, ctx: &egui::Context, remove: bool) {
        let state = &self.timeline.library;
        let Some(catalog) = &state.catalog else {
            return;
        };
        let Some(entry) = catalog["entries"].as_array().and_then(|rows| {
            rows.iter()
                .find(|e| e["id"].as_str() == state.selected.as_deref())
        }) else {
            return;
        };
        self.library_call(ctx, Reply::Catalog, true,
            format!("/v1/video/libraries/{}/entries/{}/edit", catalog["id"].as_str().unwrap_or(""), entry["id"].as_str().unwrap_or("")),
            Some(json!({"schema_version":1,"expected_library_revision":catalog["revision"],"expected_entry_revision":entry["revision"],"title":state.title,"description":state.description,"remove":remove})));
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
        egui::Window::new("Library")
            .id(egui::Id::new("video-library"))
            .open(&mut open)
            .default_width(650.)
            .default_height(680.)
            .max_height((ctx.content_rect().height() - 70.).max(200.))
            .vscroll(true)
            .show(ctx, |ui| {
                #[cfg(feature="ui-preview")] { let r=ui.interact(ui.clip_rect(),ui.id().with("qa-library-scroll"),egui::Sense::hover());control(ui,"library-scroll",&r); }
                if !helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
                    ui.label("Hidden library · hover to reveal");
                    return;
                }
                if self.timeline.private_storage {
                    ui.weak("Private library on the encrypted volume. Media stay linked to their originals.");
                } else {
                ui.weak("Choose a local library. Save explicitly; copies stay independent.");
                ui.horizontal(|ui| {
                    ui.label("Folder");
                    let width = (ui.available_width() - 40.).max(80.);
                    let r = ui.add(egui::TextEdit::singleline(&mut state.path).desired_width(width));
                    control(ui, "library-path", &r);
                    let r = ui.add_enabled_ui(!self.timeline.busy, |ui| {
                        helto::icon_button(ui, crate::icons::Icon::Folder, "Choose library folder", false)
                    }).inner;
                    control(ui, "library-choose", &r);
                    if r.clicked() {
                        action = Some("choose");
                    }
                });
                }
                ui.add_enabled_ui(!self.timeline.busy, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (name, label) in [
                            ("create", "Create library"),
                            ("open", "Open library"),
                        ] {
                            let r = ui.button(label);
                            control(ui, &format!("library-{name}"), &r);
                            if r.clicked() {
                                action = Some(name);
                            }
                        }
                        let r = helto::icon_button(ui, crate::icons::Icon::Retry, "Refresh library", false);
                        control(ui, "library-refresh", &r);
                        if r.clicked() { action = Some("refresh"); }
                    });
                });
                if let Some(catalog) = &state.catalog {
                    ui.separator();
                    ui.label(format!("Library revision {}", catalog["revision"]));
                    ui.horizontal_wrapped(|ui| {
                        for (kind, label) in [("project", "Projects"), ("character", "Characters"), ("asset", "Assets")] {
                            let selected = state.tab == kind || state.tab.is_empty() && kind == "project";
                            let r = ui.selectable_label(selected, label);
                            control(ui, &format!("library-tab-{kind}"), &r);
                            if r.clicked() { state.tab = kind.into(); state.selected = None; }
                        }
                    });
                    if state.tab == "character" {
                        ui.horizontal(|ui| {
                            let r = helto::icon_button(ui, crate::icons::Icon::Plus, "Create library character", false); control(ui,"library-character-new",&r); if r.clicked() { action = Some("character-new"); }
                            let r = helto::icon_button(ui, crate::icons::Icon::Edit, "Edit selected character", false); control(ui,"library-character-edit",&r); if r.clicked() { action = Some("character-edit"); }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Find");
                        let r = ui.add(egui::TextEdit::singleline(&mut state.search).desired_width(220.));
                        control(ui, "library-search", &r);
                        let r = ui.selectable_value(&mut state.recent, false, "Name"); control(ui, "library-sort-name", &r);
                        let r = ui.selectable_value(&mut state.recent, true, "Recent"); control(ui, "library-sort-recent", &r);
                    });
                    if let Some(entries) = catalog["entries"].as_array() {
                        let mut rows = entries.iter().enumerate().filter(|(_, e)| {
                            let kind = if state.tab.is_empty() { "project" } else { &state.tab };
                            e["content"]["kind"] == kind && e["title"].as_str().unwrap_or("").to_lowercase().contains(&state.search.to_lowercase())
                        }).collect::<Vec<_>>();
                        if state.recent { rows.reverse(); } else { rows.sort_by_key(|(_, e)| e["title"].as_str().unwrap_or("").to_lowercase()); }
                        egui::ScrollArea::vertical().id_salt("library-entries").max_height(210.).show(ui, |ui| {
                            for (i, entry) in rows {
                                ui.horizontal(|ui| {
                                    if let Some((resource, name)) = thumbnail_resource(entry) {
                                        let lib = catalog["id"].as_str().unwrap_or(""); let id = entry["id"].as_str().unwrap_or(""); let revision = entry["revision"].as_u64().unwrap_or(0);
                                        let asset = Asset { id: AssetId(format!("library-{lib}-{id}-{revision}-{resource}")), revision, kind: MediaKind::Image, name, library_source: None };
                                        let project = ProjectId(format!("library-{lib}"));
                                        self.timeline_thumbnail_path(ui, &project, &asset, Some(format!("/v1/video/libraries/{lib}/entries/{id}/revisions/{revision}/media/{resource}")));
                                    }
                                    let id = entry["id"].as_str().unwrap_or(""); let title = entry["title"].as_str().unwrap_or("");
                                    let r = ui.selectable_label(state.selected.as_deref() == Some(id), format!("{title} · r{}", entry["revision"]));
                                    control(ui, &format!("library-entry-{i}"), &r);
                                    if r.clicked() { state.selected = Some(id.into()); state.title = title.into(); state.description = entry["content"]["document"]["description"].as_str().unwrap_or("").into(); }
                                });
                            }
                        });
                    }
                    ui.add_enabled_ui(!self.timeline.busy, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let r = ui.button("Copy selected into project"); control(ui, "library-copy", &r); if r.clicked() { action = Some("copy"); }
                            for (key, icon, label) in [("edit", crate::icons::Icon::Save, "Save entry metadata"), ("delete", crate::icons::Icon::Trash, "Delete library entry; keep project and original media")] {
                                let r = helto::icon_button(ui, icon, label, false); control(ui, &format!("library-{key}"), &r); if r.clicked() { action = Some(key); }
                            }
                        });
                    });
                    ui.separator();
                    ui.horizontal(|ui| { ui.label("Entry title"); let r = ui.add(egui::TextEdit::singleline(&mut state.title).desired_width(ui.available_width())); control(ui, "library-title", &r); });
                    if state.tab == "character" {
                        ui.label("Description"); let r = ui.add(egui::TextEdit::multiline(&mut state.description).desired_width(ui.available_width()).desired_rows(2)); control(ui, "library-description", &r);
                    }
                    if let Some(document) = &document {
                        if state.tab == "character" {
                            let r = egui::ComboBox::from_id_salt("library-reference").selected_text(document.project.creative.references.get(state.reference).map(|r| r.name.as_str()).unwrap_or("Select project reference")).show_ui(ui, |ui| {
                                for (i, reference) in document.project.creative.references.iter().enumerate() { let r = ui.selectable_value(&mut state.reference, i, &reference.name); control(ui, &format!("library-reference-{i}"), &r); }
                            }).response; control(ui, "library-reference", &r);
                        } else if state.tab == "asset" {
                            let r = egui::ComboBox::from_id_salt("library-project-asset").selected_text(state.asset.as_ref().and_then(|id| document.project.assets.get(id)).map(|a| project_asset_name(&document.project, a)).unwrap_or_else(|| "Select project media".into())).show_ui(ui, |ui| {
                                for (i, asset) in document.project.assets.values().enumerate() { let r = ui.selectable_value(&mut state.asset, Some(asset.id.clone()), project_asset_name(&document.project, asset)); control(ui, &format!("library-project-asset-{i}"), &r); }
                            }).response; control(ui, "library-project-asset", &r);
                        }
                        ui.add_enabled_ui(!self.timeline.busy, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                let (save, replace) = match state.tab.as_str() { "character" => ("save-cast", "replace-cast"), "asset" => ("save-asset", "replace-asset"), _ => ("save-project", "replace-project") };
                                for (name, label) in [(save, "Save as new library entry"), (replace, "Update selected from project")] { let r = ui.button(label); control(ui, &format!("library-{name}"), &r); if r.clicked() { action = Some(name); } }
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
            Some("character-new") => {
                self.timeline.library.editor = Some(library_character::Editor::new(None))
            }
            Some("character-edit") => {
                if let Some(entry) = self
                    .timeline
                    .library
                    .catalog
                    .as_ref()
                    .and_then(|catalog| catalog["entries"].as_array())
                    .and_then(|rows| {
                        rows.iter().find(|entry| {
                            entry["id"].as_str() == self.timeline.library.selected.as_deref()
                                && entry["content"]["kind"] == "character"
                        })
                    })
                {
                    self.timeline.library.editor =
                        Some(library_character::Editor::new(Some(entry)));
                }
            }
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
            Some("edit") => self.library_edit(ctx, false),
            Some("delete") => self.library_edit(ctx, true),
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
                    } else if entry["content"]["kind"] == "asset" {
                        let resource = &entry["content"]["document"]["resource"];
                        let media = json!({"entry_id":entry["id"],"entry_revision":entry["revision"],"resource_id":resource["id"],"name":entry["title"],"kind":entry["content"]["document"]["asset"]["kind"]});
                        let source = media_picker::ImportSource::Library {
                            library: c["id"].as_str().unwrap_or("").into(),
                            entry: media,
                        };
                        self.timeline_import_sources(
                            ctx,
                            vec![source],
                            media_picker::Target::Project,
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
            Some("save-asset") => self.library_save(ctx, false, false),
            Some("replace-asset") => self.library_save(ctx, false, true),
            Some("save-project") => self.library_save(ctx, false, false),
            Some("replace-project") => self.library_save(ctx, false, true),
            Some("save-cast") => self.library_save(ctx, true, false),
            Some("replace-cast") => self.library_save(ctx, true, true),
            _ => {}
        }
        self.timeline_character_window(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::{Reply, catalog_error};

    #[test]
    fn library_open_errors_describe_opening_without_changing_save_errors() {
        for code in ["storage_io_failed", "storage_error", "resource_missing"] {
            let message = catalog_error(Reply::Opened, code, false, "project Save".into());
            assert!(message.contains("open the library files"));
            assert!(message.contains("folder and its media"));
            assert!(!message.contains("project Save"));
        }
        assert!(
            catalog_error(Reply::Opened, "invalid_storage_root", false, String::new())
                .contains("library folder is unavailable")
        );
        assert_eq!(
            catalog_error(
                Reply::Catalog,
                "storage_io_failed",
                true,
                "project Save".into()
            ),
            "Could not save the library character. Check the library folder and retry Save."
        );
        assert_eq!(
            catalog_error(
                Reply::Catalog,
                "storage_io_failed",
                false,
                "original Save".into()
            ),
            "original Save"
        );
        assert_eq!(
            catalog_error(
                Reply::Opened,
                "session_changed",
                false,
                "session changed".into()
            ),
            "session changed"
        );
    }
}
