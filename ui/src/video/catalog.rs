//! Native catalog editor and Phrase Composer. Source files are explicitly selected documents.
use super::*;
use imagine_video_contract::prompt_catalog::{Catalog, Choice, Expansion};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
mod files;
mod form;
#[derive(Clone)]
pub(in crate::app) struct Loaded {
    pub folder: PathBuf,
    pub sources: BTreeMap<String, String>,
    pub selected: Option<String>,
    pub warning: Option<String>,
    pub written: Option<String>,
    pub private_copy: bool,
}
#[derive(Clone)]
pub(super) struct Target {
    pub widget: egui::Id,
    pub text: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Clone)]
pub(super) struct Insertion {
    pub target: Target,
    pub expansion: Expansion,
}
pub(super) fn active(ctx: &egui::Context, target: Target) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("video-active-prompt"), target));
}
pub(super) fn resolved(ctx: &egui::Context) -> Option<Arc<Catalog>> {
    ctx.data(|d| d.get_temp(egui::Id::new("video-prompt-catalog")))
}
pub(super) fn pending(ctx: &egui::Context, id: egui::Id) -> Option<Insertion> {
    ctx.data_mut(|d| {
        let key = egui::Id::new("video-prompt-insert");
        let value = d
            .get_temp::<Insertion>(key)
            .filter(|p| p.target.widget == id);
        if value.is_some() {
            d.remove::<Insertion>(key);
        }
        value
    })
}
pub(super) fn insertion_error(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.insert_temp(
            egui::Id::new("video-prompt-insert-error"),
            "The target text changed. Focus the intended field and insert again.".to_owned(),
        )
    });
}
/// The shipped catalog folder, read once: its files and merged catalog, or
/// why Director would disable completion.
type Builtins = Result<(BTreeMap<String, String>, Arc<Catalog>), String>;
fn builtins() -> &'static Builtins {
    static LOADED: OnceLock<Builtins> = OnceLock::new();
    LOADED.get_or_init(|| {
        imagine_video_contract::prompt_catalog::load_folder(&crate::video_catalog_folder())
            .map(|(sources, catalog)| (sources, Arc::new(catalog)))
            .map_err(|issue| format!("Prompt IntelliSense catalog is unavailable: {issue}"))
    })
}
/// The shipped sources and catalog, or none with the reason.
fn shipped() -> (BTreeMap<String, String>, Arc<Catalog>, Option<String>) {
    match builtins() {
        Ok((sources, catalog)) => (sources.clone(), catalog.clone(), None),
        Err(error) => (
            BTreeMap::new(),
            Arc::new(Catalog::default()),
            Some(error.clone()),
        ),
    }
}
pub(super) struct State {
    pub open: bool,
    pub busy: bool,
    initialized: bool,
    confirm_delete: bool,
    nonce: u64,
    pub sources: BTreeMap<String, String>,
    pub resolved: Arc<Catalog>,
    pub folder: Option<PathBuf>,
    selected: String,
    draft: String,
    drafts: BTreeMap<(Option<PathBuf>, String), String>,
    form: Option<Value>,
    form_path: Vec<(String, usize)>,
    yaml: bool,
    editor: bool,
    filter: String,
    validation: Option<String>,
    error: Option<String>,
    stale: bool,
    choice: Option<Choice>,
    values: BTreeMap<String, String>,
    insertion_target: Option<Target>,
    private_copy: bool,
    save_as_name: Option<String>,
}
impl Default for State {
    fn default() -> Self {
        let (sources, resolved, error) = shipped();
        Self {
            open: false,
            busy: false,
            initialized: false,
            confirm_delete: false,
            nonce: 0,
            selected: sources.keys().next().cloned().unwrap_or_default(),
            draft: sources.values().next().cloned().unwrap_or_default(),
            drafts: BTreeMap::new(),
            sources,
            resolved,
            folder: None,
            form: None,
            form_path: vec![],
            yaml: true,
            editor: false,
            filter: String::new(),
            validation: None,
            error,
            stale: false,
            choice: None,
            values: BTreeMap::new(),
            insertion_target: None,
            private_copy: false,
            save_as_name: None,
        }
    }
}
impl State {
    #[cfg(feature = "ui-preview")]
    pub(super) fn qa_check(&self, stale: bool, error_contains: &str) -> Result<(), String> {
        if self.stale != stale
            || !self
                .error
                .as_deref()
                .unwrap_or_default()
                .contains(error_contains)
        {
            return Err(format!(
                "Unexpected own catalog state: stale={}, error={:?}",
                self.stale, self.error
            ));
        }
        Ok(())
    }
    fn receive(&mut self, nonce: u64, result: Result<Loaded, String>) {
        if self.nonce != nonce {
            return;
        }
        self.busy = false;
        let loaded = match result {
            Ok(loaded) => loaded,
            Err(error) => {
                self.error = Some(error);
                return;
            }
        };
        self.retain_draft();
        if loaded.private_copy && self.folder.as_ref() != Some(&loaded.folder) {
            let previous = self.folder.clone();
            let retained: Vec<_> = self
                .drafts
                .iter()
                .filter(|((folder, _), _)| folder == &previous)
                .map(|((_, name), draft)| (name.clone(), draft.clone()))
                .collect();
            self.drafts.retain(|(folder, _), _| folder != &previous);
            for (name, draft) in retained {
                self.drafts
                    .insert((Some(loaded.folder.clone()), name), draft);
            }
        }
        self.private_copy = loaded.private_copy
            || (self.private_copy && self.folder.as_ref() == Some(&loaded.folder));
        if let Some(name) = &loaded.written {
            let key = (Some(loaded.folder.clone()), name.clone());
            if !loaded.sources.contains_key(name)
                || self.drafts.get(&key) == loaded.sources.get(name)
            {
                self.drafts.remove(&key);
            }
        }
        let previous = self.selected.clone();
        self.folder = Some(loaded.folder);
        self.sources = loaded.sources;
        match imagine_video_contract::prompt_catalog::load(&self.sources) {
            Ok(catalog) => {
                self.resolved = Arc::new(catalog);
                self.stale = false;
                self.error = loaded.warning;
            }
            Err(error) => {
                self.error = Some(format!("Using last good catalog: {error}"));
                self.stale = true;
            }
        }
        if let Some(name) = loaded
            .selected
            .or_else(|| self.sources.contains_key(&previous).then_some(previous))
            .or_else(|| self.sources.keys().next().cloned())
        {
            self.load_selection(name);
        }
    }
    fn duplicate_file(&mut self) -> Result<(), String> {
        let mut value = imagine_video_contract::prompt_catalog::source_value(&self.draft)?;
        form::canonical_form(&mut value);
        value["mode"] = json!("merge");
        let path = std::path::Path::new(&self.selected);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("catalog");
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("yaml");
        let base = format!("{stem}-copy");
        let mut n = 1;
        let name = loop {
            let name = if n == 1 {
                format!("{base}.{extension}")
            } else {
                format!("{base}-{n}.{extension}")
            };
            if !self.sources.contains_key(&name)
                && !self
                    .drafts
                    .contains_key(&(self.folder.clone(), name.clone()))
            {
                break name;
            }
            n += 1;
        };
        let text = imagine_video_contract::prompt_catalog::source_text(&value)?;
        self.retain_draft();
        self.load_selection(name);
        self.draft = text;
        Ok(())
    }
    fn retain_draft(&mut self) {
        let key = (self.folder.clone(), self.selected.clone());
        if self.sources.get(&self.selected) != Some(&self.draft) {
            self.drafts.insert(key, self.draft.clone());
        } else {
            self.drafts.remove(&key);
        }
    }
    fn load_selection(&mut self, name: String) {
        self.draft = self
            .drafts
            .get(&(self.folder.clone(), name.clone()))
            .or_else(|| self.sources.get(&name))
            .cloned()
            .unwrap_or_default();
        self.selected = name;
        self.form = None;
        self.form_path.clear();
        self.validation = None;
        self.confirm_delete = false;
    }
    fn select(&mut self, name: String) {
        self.retain_draft();
        self.load_selection(name);
    }
    fn install(&self, ctx: &egui::Context) {
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("video-prompt-catalog"), self.resolved.clone())
        });
    }
    fn choose(&mut self, choice: Choice, ctx: &egui::Context) {
        self.values.clear();
        self.choice = Some(choice);
        self.insertion_target = ctx.data(|d| d.get_temp(egui::Id::new("video-active-prompt")));
    }
    fn validate(&self) -> Result<Catalog, String> {
        let mut sources = self.sources.clone();
        sources.insert(self.selected.clone(), self.draft.clone());
        imagine_video_contract::prompt_catalog::load(&sources).map_err(|e| e.to_string())
    }
}
fn control(ui: &egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl ImagineApp {
    pub(super) fn timeline_catalog_install(&mut self, ctx: &egui::Context) {
        if !self.timeline.catalog.initialized {
            self.timeline.catalog.initialized = true;
            if let Some(folder) = self.prefs.prompt_catalog_folder.clone() {
                self.timeline_catalog_folder(ctx, Some(folder));
            }
        }
        self.timeline.catalog.install(ctx);
    }
    pub(super) fn timeline_catalog_reply(&mut self, nonce: u64, result: Result<Loaded, String>) {
        let state = &mut self.timeline.catalog;
        state.receive(nonce, result);
        if self.prefs.prompt_catalog_folder != state.folder {
            self.prefs.prompt_catalog_folder = state.folder.clone();
            self.save_preferences();
        }
    }
    pub(super) fn timeline_catalog_folder(&mut self, ctx: &egui::Context, folder: Option<PathBuf>) {
        if self.timeline.catalog.busy {
            return;
        }
        self.timeline.catalog.busy = true;
        self.timeline.catalog.nonce += 1;
        let nonce = self.timeline.catalog.nonce;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let path = folder.or_else(|| {
                rfd::FileDialog::new()
                    .set_title("Choose prompt catalog folder")
                    .pick_folder()
            });
            let result = path
                .ok_or_else(|| {
                    "Catalog selection cancelled; the current catalog is retained.".into()
                })
                .and_then(|p| files::load(&p, None));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::CatalogIo { nonce, result }),
            );
        });
    }
    pub(super) fn timeline_catalog_write(
        &mut self,
        ctx: &egui::Context,
        save_as: bool,
        delete: bool,
        explicit_path: Option<PathBuf>,
    ) {
        if self.timeline.catalog.busy {
            return;
        }
        let state = &mut self.timeline.catalog;
        state.busy = true;
        state.nonce += 1;
        let nonce = state.nonce;
        // Saving from the Form tab merges linked @identity placeholders, as in Director.
        if !state.yaml
            && let Some(form) = state.form.take()
        {
            state.draft = imagine_video_contract::prompt_catalog::editor::form_save(&form);
        }
        let folder = state.folder.clone();
        let name = state.selected.clone();
        let content = state.draft.clone();
        let expected = state.sources.get(&name).cloned();
        let sources = state.sources.clone();
        let private = match &self.preferences_store {
            crate::settings::PreferencesStore::Fixture(_) => None,
            store => Some(store.private_root()),
        };
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            if let Some(private) = private {
                let result = private.and_then(|root| {
                    let selected = explicit_path.as_ref().map_or_else(
                        || Ok(name.clone()),
                        |path| {
                            path.to_str().map(str::to_owned).ok_or_else(|| {
                                "Choose a YAML filename within the private catalog.".to_owned()
                            })
                        },
                    )?;
                    files::save_private(
                        &root,
                        folder.as_deref(),
                        &sources,
                        &selected,
                        &content,
                        if !save_as { expected.as_deref() } else { None },
                        delete,
                    )
                });
                send(
                    &tx,
                    &ctx,
                    Message::Timeline(Event::CatalogIo { nonce, result }),
                );
                return;
            }
            let path = explicit_path
                .or_else(|| {
                    if !save_as {
                        folder.as_ref().map(|p| p.join(&name))
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    rfd::FileDialog::new()
                        .set_title("Save prompt catalog")
                        .set_file_name(&name)
                        .add_filter("YAML catalog", &["yaml", "yml"])
                        .save_file()
                });
            let result = path
                .ok_or_else(|| "Save cancelled; the draft is retained.".into())
                .and_then(|path| {
                    files::save(
                        &path,
                        &content,
                        if !save_as && folder.is_some() {
                            expected.as_deref()
                        } else {
                            None
                        },
                        delete,
                    )
                });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::CatalogIo { nonce, result }),
            );
        });
    }
    pub(super) fn timeline_catalog_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.catalog.open {
            return;
        }
        let state = &mut self.timeline.catalog;
        let mut open = state.open;
        let mut load = false;
        let mut reload = false;
        let mut save = None;
        let mut filename = None;
        let mut close = false;
        let private = matches!(
            self.preferences_store,
            crate::settings::PreferencesStore::Home { .. }
        );
        egui::Window::new("Prompt tools")
            .id(egui::Id::new("video-prompt-tools"))
            .open(&mut open)
            .default_width(850.)
            .default_height(620.)
            .min_width(400.)
            .max_height((ctx.content_rect().height() - 70.).max(250.))
            .vscroll(true)
            .show(ctx, |ui| {
let _canvas = helto::window_canvas(ui);
                #[cfg(feature="ui-preview")]
                {let r=ui.interact(ui.clip_rect(),ui.make_persistent_id("catalog-scroll"),egui::Sense::hover());control(ui,"catalog-scroll",&r);}
                let hidden = !helto::shown(ui, self.prefs.privacy_mode);
                helto::private_scope(ui, hidden, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (value, label) in [
                            (false, "Browse & Phrase Composer"),
                            (true, "Catalog Editor"),
                        ] {
                            let r = ui.selectable_value(&mut state.editor, value, label);
                            control(
                                ui,
                                if value {
                                    "catalog-editor"
                                } else {
                                    "catalog-browse"
                                },
                                &r,
                            );
                        }
                        if helto::icon_button(
                            ui,
                            crate::icons::Icon::Folder,
                            "Choose Catalog Folder",
                            false,
                        )
                        .clicked()
                        {
                            load = true;
                        }
                        let r = helto::icon_button(
                            ui,
                            crate::icons::Icon::Retry,
                            "Reload Catalog",
                            false,
                        );
                        control(ui, "catalog-reload", &r);
                        reload = r.clicked();
                        let r = helto::icon_button(ui, crate::icons::Icon::Close, "Close Prompt Tools", false);
                        control(ui, "catalog-close", &r);
                        close = r.clicked();
                    });
                    ui.weak(
                        state.folder.as_ref().map_or(
                            "Built-in source templates (choose a folder to use your catalog)"
                                .to_owned(),
                            |p| p.display().to_string(),
                        ),
                    );
                    if private {
                        ui.weak(if state.private_copy { "Private copy; original catalog unchanged" }
                            else { "Save/Delete creates a private copy; original catalog unchanged" });
                    }
                    if state.busy {
                        ui.spinner();
                    }
                    if let Some(e) = ctx.data_mut(|d| {
                        d.remove_temp::<String>(egui::Id::new("video-prompt-insert-error"))
                    }) {
                        state.error = Some(e);
                    }
                    if let Some(e) = &state.error {
                        ui.colored_label(helto::ERROR, e);
                    }
                    if state.stale {
                        ui.weak(
                            "Completion uses the last valid catalog. Fix the source and reload.",
                        );
                    }
                    if state.editor {
                        ui.horizontal_wrapped(|ui| {
                            egui::ComboBox::from_id_salt("catalog-source")
                                .selected_text(&state.selected)
                                .show_ui(ui, |ui| {
                                    let mut names = state.sources.keys().cloned().collect::<std::collections::BTreeSet<_>>();
                                    names.extend(state.drafts.keys().filter(|(folder,_)|folder==&state.folder).map(|(_,name)|name.clone()));
                                    names.insert(state.selected.clone());
                                    for n in names {
                                        if ui.selectable_label(n == state.selected, &n).clicked() {
                                            state.select(n);
                                        }
                                    }
                                });
                            let r = helto::icon_button(
                                ui,
                                crate::icons::Icon::Plus,
                                "New Catalog File",
                                false,
                            );
                            control(ui, "catalog-new", &r);
                            if r.clicked() {
                                state.retain_draft();
                                let mut index = 1;
                                let name = loop {
                                    let name = format!("new-catalog-{index}.yaml");
                                    if !state.sources.contains_key(&name) && !state.drafts.contains_key(&(state.folder.clone(), name.clone())) { break name; }
                                    index += 1;
                                };
                                state.selected = name;
                                state.draft = form::starter();
                                state.form = None;
                                state.form_path.clear();
                                state.validation = None;
                            }
                            let r=helto::icon_button(ui,crate::icons::Icon::Duplicate,"Duplicate Catalog File",false);control(ui,"catalog-duplicate-file",&r);
                            if r.clicked()&&let Err(e)=state.duplicate_file(){state.error=Some(e);}
                            let r = helto::icon_button(ui, crate::icons::Icon::Check, "Validate Catalog", false);
                            control(ui, "catalog-validate", &r);
                            if r.clicked() {
                                state.validation = Some(match state.validate() {
                                    Ok(_) => "Valid catalog".into(),
                                    Err(e) => e,
                                });
                            }
                            let r = helto::action(ui, crate::icons::Icon::Save, "Save", true, state.busy.then_some("Saving."));
                            control(ui, "catalog-save", &r);
                            if r.clicked() {
                                save = Some((false, false));
                            }
                            let r = helto::action(ui, crate::icons::Icon::Duplicate, "Save as…", false, state.busy.then_some("Saving."));
                            control(ui, "catalog-save-as", &r);
                            if r.clicked() {
                                if private { state.save_as_name = Some(state.selected.clone()); }
                                else { save = Some((true, false)); }
                            }
                            let r=ui.add_enabled_ui(state.folder.is_some() && state.sources.contains_key(&state.selected) && !state.busy, |ui|helto::icon_button(ui,crate::icons::Icon::Trash,"Delete Catalog File",false)).inner;
                            control(ui,"catalog-delete",&r);
                            if r.clicked(){state.confirm_delete=true;}
                        });
                        if let Some(name) = &mut state.save_as_name {
                            let mut done = false;
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Private catalog filename");
                                let r = ui.text_edit_singleline(name); control(ui, "catalog-private-filename", &r);
                                let r = helto::icon_button(ui, crate::icons::Icon::Save, "Save", true); control(ui, "catalog-private-save-as", &r);
                                if r.clicked() { filename = Some(PathBuf::from(name.as_str())); save = Some((true, false)); done = true; }
                                if helto::icon_button(ui, crate::icons::Icon::Close, "Cancel", false).clicked() { done = true; }
                            });
                            if done { state.save_as_name = None; }
                        }
                        if state.confirm_delete {
                            let answer = helto::confirm(ui, &format!("Delete {}?", state.selected), (crate::icons::Icon::Trash, "Delete"), "Cancel", None);
                            control(ui, "catalog-delete-confirm", &answer.yes);
                            control(ui, "catalog-delete-cancel", &answer.no);
                            if answer.yes.clicked() { save = Some((false, true)); state.confirm_delete = false; }
                            if answer.no.clicked() { state.confirm_delete = false; }
                        }
                        ui.horizontal(|ui| {
                            for (yaml, label) in [(true, "YAML"), (false, "Form")] {
                                let r = ui.selectable_value(&mut state.yaml, yaml, label);
                                control(ui, if yaml { "catalog-yaml" } else { "catalog-form" }, &r);
                            }
                        });
                        if state.yaml {
                            let r = helto::textarea(
                                ui,
                                &mut state.draft,
                                16,
                                "Catalog YAML",
                                self.prefs.privacy_mode,
                                true,
                            );
                            control(ui, "catalog-yaml-text", &r);
                            if r.changed() {
                                state.form = None;
                                state.validation = None;
                            }
                        } else {
                            form::show(ui, state, self.prefs.privacy_mode);
                        }
                        if let Some(v) = &state.validation {
                            ui.label(v);
                        }
                    } else {
                        ui.horizontal(|ui| {
                            ui.label("Search");
                            let r = ui.text_edit_singleline(&mut state.filter);
                            control(ui, "catalog-search", &r);
                        });
                        let roots = state.resolved.clone();
                        egui::ScrollArea::vertical()
                            .id_salt("catalog-tree")
                            .max_height(210.)
                            .show(ui, |ui| {
                                for root in &roots.namespaces {
                                    browse(ui, root, &mut vec![root.key.clone()], state, ctx);
                                }
                            });
                        if let Some(choice) = &state.choice {
                            ui.separator();
                            ui.strong(&choice.label);
                            if !choice.description.is_empty() {
                                ui.label(&choice.description);
                            }
                            let slots = choice.expand(&BTreeMap::new()).slots;
                            for slot in slots {
                                ui.label(&slot.label);
                                let value =
                                    state.values.entry(slot.id.clone()).or_insert_with(|| {
                                        let e = choice.expand(&BTreeMap::new());
                                        let o = &e
                                            .slots
                                            .iter()
                                            .find(|p| p.id == slot.id)
                                            .unwrap()
                                            .occurrences[0];
                                        e.text.chars().skip(o.start).take(o.end - o.start).collect()
                                    });
                                let r = ui.text_edit_singleline(value).on_hover_text(&slot.hint);
                                control(ui, &format!("composer-slot-{}", slot.id), &r);
                            }
                            let expansion = choice.expand(&state.values);
                            ui.label("Preview");
                            let mut preview = expansion.text.clone();
                            helto::textarea(
                                ui,
                                &mut preview,
                                4,
                                "Phrase preview",
                                self.prefs.privacy_mode,
                                false,
                            );
                            ui.horizontal(|ui| {
                                let r = helto::icon_button(
                                    ui,
                                    crate::icons::Icon::Copy,
                                    "Copy Phrase",
                                    false,
                                );
                                control(ui, "composer-copy", &r);
                                if r.clicked() {
                                    ctx.copy_text(expansion.text.clone());
                                }
                                let r = helto::action(
                                    ui,
                                    crate::icons::Icon::Insert,
                                    "Insert in prompt",
                                    true,
                                    state
                                        .insertion_target
                                        .is_none()
                                        .then_some("Place the cursor in a prompt field first."),
                                );
                                control(ui, "composer-insert", &r);
                                if r.clicked() {
                                    let target = state.insertion_target.clone().unwrap();
                                    let live=ctx.data(|d|d.get_temp::<(u64,String)>(target.widget.with("prompt-live-target")));
                                    if live.is_some_and(|(frame,text)|ctx.cumulative_frame_nr().saturating_sub(frame)<=1&&text==target.text){
                                        ctx.data_mut(|d|d.insert_temp(egui::Id::new("video-prompt-insert"),Insertion{target,expansion:expansion.clone()}));
                                        close=true;
                                    }else{state.error=Some("The target changed. Focus the intended field and choose the phrase again.".into());}
                                }
                            });
                        }
                    }
                });
            });
        state.open = open && !close;
        let folder = state.folder.clone();
        if load {
            self.timeline_catalog_folder(ctx, None);
        } else if reload {
            if folder.is_some() {
                self.timeline_catalog_folder(ctx, folder);
            } else {
                let state = &mut self.timeline.catalog;
                state.retain_draft();
                let (sources, resolved, error) = shipped();
                state.sources = sources;
                state.resolved = resolved;
                state.stale = false;
                state.error = error;
                state.load_selection(state.selected.clone());
            }
        }
        if let Some((save_as, delete)) = save {
            self.timeline_catalog_write(ctx, save_as, delete, filename);
        }
    }
}
fn browse(
    ui: &mut egui::Ui,
    node: &imagine_video_contract::prompt_catalog::Container,
    path: &mut Vec<String>,
    state: &mut State,
    ctx: &egui::Context,
) {
    let filter = state.filter.to_lowercase();
    let matches = |c: &Choice| {
        let f = filter.clone();
        f.is_empty()
            || c.label.to_lowercase().contains(&f)
            || c.key.contains(&f)
            || c.aliases.iter().any(|s| s.contains(&f))
    };
    let filtered = !state.filter.is_empty();
    fn contains(node: &imagine_video_contract::prompt_catalog::Container, f: &str) -> bool {
        node.choices.iter().any(|c| {
            c.label.to_lowercase().contains(f)
                || c.key.contains(f)
                || c.aliases.iter().any(|s| s.contains(f))
        }) || node.groups.iter().any(|g| contains(g, f))
    }
    if filtered && !contains(node, &filter) {
        return;
    }
    egui::CollapsingHeader::new(&node.label)
        .id_salt(path.clone())
        .open(filtered.then_some(true))
        .show(ui, |ui| {
            for c in &node.choices {
                if matches(c) {
                    let r = ui.selectable_label(state.choice.as_ref() == Some(c), &c.label);
                    control(
                        ui,
                        &format!(
                            "catalog-choice-{}",
                            path.iter()
                                .chain(std::iter::once(&c.key))
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(".")
                        ),
                        &r,
                    );
                    if r.clicked() {
                        state.choose(c.clone(), ctx);
                    }
                }
            }
            for g in &node.groups {
                path.push(g.key.clone());
                browse(ui, g, path, state, ctx);
                path.pop();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_copy_retargets_unsaved_other_parts_and_failed_save_retains_editor() {
        let mut state = State {
            folder: Some(PathBuf::from("/own/external")),
            ..State::default()
        };
        let first = state.selected.clone();
        state.draft.push_str("\n# OWN OTHER UNSAVED PART\n");
        let first_draft = state.draft.clone();
        let second = state
            .sources
            .keys()
            .find(|name| *name != &first)
            .unwrap()
            .clone();
        state.select(second.clone());
        state.draft.push_str("\n# OWN SAVED PART\n");
        let second_draft = state.draft.clone();
        state.receive(0, Err("private_storage_write_failed".into()));
        assert_eq!(state.folder, Some(PathBuf::from("/own/external")));
        assert_eq!(state.draft, second_draft);
        let mut sources = state.sources.clone();
        sources.insert(second.clone(), second_draft.clone());
        let folder = PathBuf::from("/own/private/library/prompt-catalog/catalog-own");
        state.receive(
            0,
            Ok(Loaded {
                folder: folder.clone(),
                sources,
                selected: Some(second.clone()),
                written: Some(second),
                warning: None,
                private_copy: true,
            }),
        );
        assert_eq!(state.folder, Some(folder));
        assert!(state.private_copy);
        assert_eq!(state.draft, second_draft);
        state.select(first);
        assert_eq!(state.draft, first_draft);
    }
    #[test]
    fn invalid_reload_retains_completion_and_draft_then_valid_reload_recovers() {
        let mut state = State::default();
        let folder = PathBuf::from("/own/synthetic/catalog");
        let sources = state.sources.clone();
        let good = state.resolved.clone();
        let selected = state.selected.clone();
        state.draft.push_str("\n# own unsaved draft\n");
        let draft = state.draft.clone();
        state.nonce = 2;
        state.busy = true;
        state.receive(1, Err("outdated reply".into()));
        assert!(state.busy);
        assert!(state.error.is_none());
        state.receive(2, Err("Catalog selection cancelled".into()));
        assert!(!state.busy);
        assert!(Arc::ptr_eq(&good, &state.resolved));
        assert_eq!(state.draft, draft);
        state.receive(
            2,
            Ok(Loaded {
                folder: folder.clone(),
                sources: [(selected.clone(), "not valid".into())].into(),
                selected: Some(selected.clone()),
                warning: None,
                written: None,
                private_copy: false,
            }),
        );
        assert!(state.stale);
        assert!(Arc::ptr_eq(&good, &state.resolved));
        assert!(
            state
                .error
                .as_deref()
                .unwrap()
                .contains("Using last good catalog")
        );
        state.receive(
            2,
            Ok(Loaded {
                folder,
                sources,
                selected: Some(selected),
                warning: None,
                written: None,
                private_copy: false,
            }),
        );
        assert!(!state.stale);
        assert!(state.error.is_none());
        assert!(!state.resolved.namespaces.is_empty());
    }
    #[test]
    fn builtins_and_switching_files_preserve_unsaved_drafts() {
        let mut state = State::default();
        assert!(!state.resolved.namespaces.is_empty());
        let first = state.selected.clone();
        let second = state.sources.keys().find(|n| *n != &first).unwrap().clone();
        state.draft.push_str("\n# own unsaved draft\n");
        let draft = state.draft.clone();
        state.select(second);
        state.select(first);
        assert_eq!(state.draft, draft);
        assert!(state.validate().is_ok());
    }
    #[test]
    fn duplicate_file_is_unsaved_merge_and_retains_original_draft() {
        let mut state = State::default();
        let original = state.selected.clone();
        let text = state.draft.clone();
        state.duplicate_file().unwrap();
        let first = state.selected.clone();
        assert_ne!(first, original);
        assert_eq!(state.sources[&original], text);
        assert!(!state.sources.contains_key(&first));
        assert!(state.validate().is_ok());
        state.select(original);
        assert_eq!(state.draft, text);
        state.duplicate_file().unwrap();
        assert_ne!(state.selected, first);
    }
}
