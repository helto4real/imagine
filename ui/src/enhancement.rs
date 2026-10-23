use super::*;
use crate::{
    enhancement_types::{
        ModelsRequest, Profile, ProfileUpdate, Request, TokenParameter, reserved_parameter,
    },
    icons::Icon,
};
use std::collections::HashMap;
use tokio::sync::oneshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Field {
    Prompt,
    Character,
    Clothing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Target {
    pub workspace: Workspace,
    pub field: Field,
}
#[derive(Clone, PartialEq)]
struct ContextStamp {
    family: String,
    references: Vec<String>,
    source: Option<String>,
}
struct Run {
    id: String,
    target: Target,
    original: String,
    context: ContextStamp,
    widget: egui::Id,
    cancel: Option<oneshot::Sender<()>>,
}
struct Proposal {
    run: Run,
    text: String,
}
#[derive(Default)]
pub(super) struct EnhancementUi {
    pub profiles: Vec<Profile>,
    draft: Option<Profile>,
    loading: bool,
    saving: bool,
    loaded: bool,
    settings_error: Option<String>,
    models: Vec<String>,
    models_for: Option<ModelsRequest>,
    models_pending: Option<ModelsRequest>,
    models_error: Option<String>,
    extra_profile: Option<String>,
    extra_parameters: String,
    extra_error: Option<String>,
    error: Option<(Target, String)>,
    fields: HashMap<Target, egui::Id>,
    active: Option<Run>,
    proposal: Option<Proposal>,
    /// A profile change held until the user confirms discarding or deleting.
    confirm: Option<ProfileChange>,
}
#[derive(Clone, PartialEq)]
enum ProfileChange {
    New,
    Duplicate,
    Select(Option<String>),
    Delete,
}
pub(super) enum Event {
    Models {
        lookup: ModelsRequest,
        result: Result<Vec<String>, Failure>,
    },
    Profiles {
        result: Result<Vec<Profile>, Failure>,
        select: Option<String>,
        deleted: Option<String>,
    },
    Finished {
        id: String,
        result: Result<String, Failure>,
    },
}
fn send_event(tx: &Sender<Message>, ctx: &egui::Context, event: Event) {
    send(tx, ctx, Message::Enhancement(event));
}

impl ImagineApp {
    fn enhancement_models_request(&self) -> Option<ModelsRequest> {
        let draft = self.enhancement.draft.as_ref()?;
        Some(ModelsRequest {
            profile_id: draft.id.clone(),
            base_url: draft.base_url.clone(),
        })
    }
    fn load_enhancement_models(&mut self, ctx: &egui::Context) {
        let (Some(client), Some(lookup)) = (self.client.clone(), self.enhancement_models_request())
        else {
            return;
        };
        self.enhancement.models_pending = Some(lookup.clone());
        self.enhancement.models_error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.enhancement_models(&lookup));
            send_event(&tx, &ctx, Event::Models { lookup, result });
        });
    }
    fn enhancement_profile_dirty(&self) -> bool {
        self.enhancement.draft.as_ref().is_some_and(|draft| {
            self.enhancement.profiles.iter().find(|p| p.id == draft.id) != Some(draft)
                || self.enhancement.extra_error.is_some()
        })
    }
    pub(super) fn enhancement_save_status(&self) -> Option<&'static str> {
        if self.enhancement.saving {
            Some("Saving profile")
        } else if self.enhancement.settings_error.is_some() {
            Some("Profile not saved")
        } else if self.enhancement_profile_dirty() {
            Some("Unsaved profile")
        } else {
            None
        }
    }
    fn enhancement_text(&self, target: Target) -> &String {
        let desk = self.desk_for(target.workspace);
        match target.field {
            Field::Prompt => &desk.prompt,
            Field::Character => &desk.character_description,
            Field::Clothing => &desk.clothing_description,
        }
    }
    fn enhancement_text_mut(&mut self, target: Target) -> &mut String {
        let desk = self.desk_for_mut(target.workspace);
        match target.field {
            Field::Prompt => &mut desk.prompt,
            Field::Character => &mut desk.character_description,
            Field::Clothing => &mut desk.clothing_description,
        }
    }
    fn enhancement_context(&self, target: Target) -> ContextStamp {
        let desk = self.desk_for(target.workspace);
        ContextStamp {
            family: self.prefs.model_group.clone(),
            references: desk.references.iter().map(|r| r.id.clone()).collect(),
            source: (target.workspace == Workspace::ImageEdit)
                .then(|| desk.selected_result().map(|r| r.asset_id.clone()))
                .flatten(),
        }
    }
    pub(super) fn load_enhancement_profiles(&mut self, ctx: &egui::Context) {
        if self.enhancement.loading || self.enhancement.saving {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        self.enhancement.loading = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.enhancement_profiles());
            send_event(
                &tx,
                &ctx,
                Event::Profiles {
                    result,
                    select: None,
                    deleted: None,
                },
            );
        });
    }
    pub(super) fn enhancement_message(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::Models { lookup, result } => {
                if self.enhancement.models_pending.as_ref() != Some(&lookup) {
                    return;
                }
                self.enhancement.models_pending = None;
                if self.enhancement_models_request().as_ref() != Some(&lookup) {
                    return;
                }
                self.enhancement.models_for = Some(lookup);
                match result {
                    Ok(models) => {
                        self.enhancement.models = models;
                        self.enhancement.models_error =
                            self.enhancement.models.is_empty().then(|| {
                                "The server returned no models. Enter a model ID manually.".into()
                            });
                    }
                    Err(error) => {
                        self.enhancement.models.clear();
                        self.enhancement.models_error = Some(error.message);
                    }
                }
            }
            Event::Profiles {
                result,
                select,
                deleted,
            } => {
                self.enhancement.loading = false;
                self.enhancement.saving = false;
                match result {
                    Ok(profiles) => {
                        let selected = select
                            .or_else(|| self.enhancement.draft.as_ref().map(|p| p.id.clone()));
                        self.enhancement.draft = selected
                            .and_then(|id| profiles.iter().find(|p| p.id == id).cloned())
                            .or_else(|| profiles.first().cloned());
                        self.enhancement.profiles = profiles;
                        self.enhancement.loaded = true;
                        self.enhancement.settings_error = None;
                        self.enhancement.extra_profile = None;
                        if let Some(id) = deleted {
                            for group in self.prefs.model_groups.values_mut() {
                                if group.enhancement_profile_id.as_ref() == Some(&id) {
                                    group.enhancement_profile_id = None;
                                }
                            }
                            self.save_preferences();
                        }
                    }
                    Err(error) => self.enhancement.settings_error = Some(error.message),
                }
            }
            Event::Finished { id, result } => {
                if self
                    .enhancement
                    .active
                    .as_ref()
                    .is_none_or(|run| run.id != id)
                {
                    return;
                }
                let run = self.enhancement.active.take().unwrap();
                match result {
                    Ok(_) if self.enhancement_context(run.target) != run.context => {
                        self.enhancement.error = Some((
                            run.target,
                            "The model family or image references changed. Enhance again.".into(),
                        ));
                    }
                    Ok(text) if *self.enhancement_text(run.target) == run.original => {
                        self.apply_enhancement(ctx, &run, text)
                    }
                    Ok(text) => self.enhancement.proposal = Some(Proposal { run, text }),
                    Err(error) => self.enhancement.error = Some((run.target, error.message)),
                }
            }
        }
    }
    fn apply_enhancement(&mut self, ctx: &egui::Context, run: &Run, text: String) {
        let original = self.enhancement_text(run.target).clone();
        if let Some(mut state) = egui::TextEdit::load_state(ctx, run.widget) {
            let before = state.cursor.char_range().unwrap_or_default();
            let after =
                egui::text::CCursorRange::one(egui::text::CCursor::new(text.chars().count()));
            let mut undoer = state.undoer();
            undoer.add_undo(&(before, original));
            undoer.feed_state(ctx.input(|input| input.time), &(after, text.clone()));
            undoer.add_undo(&(after, text.clone()));
            state.set_undoer(undoer);
            state.cursor.set_char_range(Some(after));
            state.store(ctx, run.widget);
        }
        *self.enhancement_text_mut(run.target) = text;
        if self.workspace == run.target.workspace {
            ctx.memory_mut(|m| m.request_focus(run.widget));
        }
    }
    fn start_enhancement(&mut self, ctx: &egui::Context, target: Target) {
        if self.enhancement.active.is_some()
            || self.enhancement.proposal.is_some()
            || self.enhancement_text(target).trim().is_empty()
        {
            return;
        }
        let Some(profile_id) = self
            .prefs
            .model_groups
            .get(&self.prefs.model_group)
            .and_then(|g| g.enhancement_profile_id.clone())
        else {
            return;
        };
        if !self.enhancement.profiles.iter().any(|p| p.id == profile_id) {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        let Some(&widget) = self.enhancement.fields.get(&target) else {
            return;
        };
        let id = uuid::Uuid::new_v4().to_string();
        let original = self.enhancement_text(target).clone();
        let request = Request {
            id: id.clone(),
            profile_id,
            prompt: original.clone(),
        };
        let (cancel, cancelled) = oneshot::channel();
        self.enhancement.error = None;
        self.enhancement.active = Some(Run {
            id: id.clone(),
            target,
            original,
            context: self.enhancement_context(target),
            widget,
            cancel: Some(cancel),
        });
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            runtime().block_on(async {
                let result = match client.start_enhancement(&request).await {
                    Ok(()) => tokio::select! {
                        biased;
                        _ = cancelled => { client.remove_enhancement(&id).await; return; }
                        result = client.wait_enhancement(&id) => result,
                    },
                    Err(error) => Err(error),
                };
                client.remove_enhancement(&id).await;
                send_event(&tx, &ctx, Event::Finished { id, result });
            });
        });
    }
    pub(super) fn enhancement_label(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        field: Field,
    ) -> egui::Response {
        let target = Target {
            workspace: self.workspace,
            field,
        };
        ui.push_id(target, |ui| {
            ui.horizontal(|ui| {
                let label = ui.label(label);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self
                        .enhancement
                        .active
                        .as_ref()
                        .is_some_and(|r| r.target == target)
                    {
                        if helto::icon_button(ui, Icon::Stop, "Cancel enhancement", false).clicked()
                            && let Some(mut run) = self.enhancement.active.take()
                            && let Some(cancel) = run.cancel.take()
                        {
                            let _ = cancel.send(());
                        }
                        ui.spinner();
                    } else {
                        let profile = self
                            .prefs
                            .model_groups
                            .get(&self.prefs.model_group)
                            .and_then(|g| g.enhancement_profile_id.as_ref())
                            .and_then(|id| self.enhancement.profiles.iter().find(|p| &p.id == id));
                        let tooltip = profile
                            .map(|p| format!("Enhance prompt · {}", p.name))
                            .unwrap_or_else(|| "Choose an enhancement profile in Settings.".into());
                        let enabled = profile.is_some()
                            && self.client.is_some()
                            && self.enhancement.active.is_none()
                            && self.enhancement.proposal.is_none()
                            && !self.enhancement_text(target).trim().is_empty();
                        if ui
                            .add_enabled_ui(enabled, |ui| {
                                helto::icon_button(ui, Icon::Enhance, "Enhance prompt", false)
                            })
                            .inner
                            .on_hover_text(&tooltip)
                            .on_disabled_hover_text(&tooltip)
                            .clicked()
                        {
                            self.start_enhancement(ui.ctx(), target);
                        }
                    }
                });
                label
            })
            .inner
        })
        .inner
    }
    pub(super) fn enhancement_field(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        field: Field,
    ) {
        let target = Target {
            workspace: self.workspace,
            field,
        };
        self.enhancement.fields.insert(target, response.id);
        if let Some((error_target, error)) = &self.enhancement.error
            && *error_target == target
        {
            ui.colored_label(helto::ERROR, error);
        }
        if self
            .enhancement
            .proposal
            .as_ref()
            .is_some_and(|p| p.run.target == target)
        {
            self.enhancement_proposal(ui, target);
        }
    }
    /// A result that arrived after its field was edited, shown under that field
    /// until it is used, retried or discarded.
    fn enhancement_proposal(&mut self, ui: &mut egui::Ui, target: Target) {
        let mut text = self.enhancement.proposal.as_ref().unwrap().text.clone();
        let (mut use_text, mut retry, mut discard) = (false, false, false);
        // Applying after a context change would mix the old references in; keep the text instead.
        let stale = self
            .enhancement
            .proposal
            .as_ref()
            .is_some_and(|p| self.enhancement_context(p.run.target) != p.run.context);
        let retry_blocked = if self.client.is_none() {
            Some("Not connected to Imagine.")
        } else if self.enhancement_text(target).trim().is_empty() {
            Some("Enter text to enhance.")
        } else {
            None
        };
        ui.add_space(4.);
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(helto::section("Enhanced prompt"));
                ui.weak("The text changed while enhancement was running.");
            });
            helto::textarea(
                ui,
                &mut text,
                5,
                "enhancement-proposal",
                self.prefs.privacy_mode,
                true,
            );
            ui.horizontal(|ui| {
                use_text = helto::action(
                    ui,
                    Icon::Check,
                    "Use",
                    true,
                    stale.then_some(
                        "The model family or image references changed. Retry or Discard.",
                    ),
                )
                .clicked();
                retry = helto::action(ui, Icon::Retry, "Retry", false, retry_blocked).clicked();
                discard = helto::icon_button(ui, Icon::Close, "Discard", false).clicked();
            });
        });
        if use_text || retry || discard {
            let proposal = self.enhancement.proposal.take().unwrap();
            if use_text && self.enhancement_context(proposal.run.target) == proposal.run.context {
                self.apply_enhancement(ui.ctx(), &proposal.run, text);
            }
            if retry {
                self.start_enhancement(ui.ctx(), target);
            }
        } else if let Some(proposal) = &mut self.enhancement.proposal {
            proposal.text = text;
        }
    }
    pub(super) fn enhancement_settings(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Prompt enhancement")
            .id_salt("enhancement-settings")
            .show(ui, |ui| {
                if !self.enhancement.loaded {
                    let blocked = if self.client.is_none() {
                        Some("Not connected to Imagine.")
                    } else {
                        self.enhancement.loading.then_some("Loading profiles.")
                    };
                    if helto::action(ui, Icon::Download, "Load profiles", false, blocked).clicked()
                    {
                        self.load_enhancement_profiles(ui.ctx());
                    }
                } else {
                    let mut families = std::collections::BTreeSet::from([
                        crate::model_catalog::QWEN_GROUP.to_owned(),
                        crate::model_catalog::MOSS_GROUP.to_owned(),
                    ]);
                    if let Some(catalog) = &self.models.catalog {
                        families.extend(
                            catalog
                                .models
                                .iter()
                                .filter_map(|m| m["group"].as_str().map(str::to_owned)),
                        );
                    }
                    families.extend(self.prefs.model_groups.keys().cloned());
                    let profile_width = (ui.available_width() - 140.).max(100.);
                    egui::Grid::new("enhancement-family-profiles")
                        .num_columns(2)
                        .show(ui, |ui| {
                            for family in families {
                                ui.label(super::models::model_group_name(&family));
                                let selected = &mut self
                                    .prefs
                                    .model_groups
                                    .entry(family.clone())
                                    .or_default()
                                    .enhancement_profile_id;
                                let before = selected.clone();
                                let name = selected
                                    .as_ref()
                                    .map(|id| {
                                        self.enhancement
                                            .profiles
                                            .iter()
                                            .find(|p| &p.id == id)
                                            .map(|p| p.name.as_str())
                                            .unwrap_or("Missing profile")
                                    })
                                    .unwrap_or("Off");
                                egui::ComboBox::from_id_salt((&family, "enhancement-profile"))
                                    .selected_text(name)
                                    .width(profile_width)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(selected, None, "Off");
                                        for profile in &self.enhancement.profiles {
                                            ui.selectable_value(
                                                selected,
                                                Some(profile.id.clone()),
                                                &profile.name,
                                            );
                                        }
                                    });
                                if *selected != before {
                                    self.save_preferences();
                                }
                                ui.end_row();
                            }
                        });
                    ui.add_space(12.);
                    ui.add_enabled_ui(
                        !self.enhancement.saving && !self.enhancement.loading,
                        |ui| self.enhancement_profile_editor(ui),
                    );
                }
                if self.enhancement.loading || self.enhancement.saving {
                    ui.spinner();
                }
                if let Some(error) = &self.enhancement.settings_error {
                    ui.colored_label(helto::ERROR, error);
                }
            });
    }
    pub(super) fn enhancement_provider_settings(&mut self, ui: &mut egui::Ui) {
        if !self.enhancement.loaded {
            self.load_enhancement_profiles(ui.ctx());
        }
        ui.add_enabled_ui(
            !self.enhancement.saving && !self.enhancement.loading,
            |ui| self.enhancement_profile_editor(ui),
        );
        if self.enhancement.loading || self.enhancement.saving {
            ui.spinner();
        }
        if let Some(error) = &self.enhancement.settings_error {
            ui.colored_label(helto::ERROR, error);
        }
    }
    fn apply_profile_change(&mut self, ctx: &egui::Context, change: ProfileChange) {
        let draft = self.enhancement.draft.clone();
        match change {
            ProfileChange::New => {
                self.enhancement.draft = Some(Profile {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: "New profile".into(),
                    ..Default::default()
                });
            }
            ProfileChange::Duplicate => {
                if let Some(draft) = draft {
                    self.enhancement.draft = Some(Profile {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: format!("{} copy", draft.name),
                        ..draft
                    });
                }
            }
            ProfileChange::Select(selected) => {
                self.enhancement.draft = selected.and_then(|id| {
                    self.enhancement
                        .profiles
                        .iter()
                        .find(|p| p.id == id)
                        .cloned()
                });
            }
            ProfileChange::Delete => {
                if draft.is_some_and(|d| self.enhancement.profiles.iter().any(|p| p.id == d.id)) {
                    self.save_enhancement_profile(ctx, true);
                } else {
                    self.enhancement.draft = self.enhancement.profiles.first().cloned();
                }
            }
        }
    }
    fn enhancement_profile_editor(&mut self, ui: &mut egui::Ui) {
        // Leave room for the text area's border inside the settings clip rect.
        ui.set_max_width((ui.available_width() - 8.).max(120.));
        let mut selected = self.enhancement.draft.as_ref().map(|p| p.id.clone());
        let previous = selected.clone();
        let mut change = None;
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("enhancement-editor-profile")
                .selected_text(
                    self.enhancement
                        .draft
                        .as_ref()
                        .map(|p| p.name.as_str())
                        .unwrap_or("Profile"),
                )
                .width((ui.available_width() - 110.).max(80.))
                .show_ui(ui, |ui| {
                    for profile in &self.enhancement.profiles {
                        ui.selectable_value(&mut selected, Some(profile.id.clone()), &profile.name);
                    }
                });
            if helto::icon_button(ui, Icon::Plus, "New enhancement profile", false).clicked() {
                change = Some(ProfileChange::New);
            }
            if self.enhancement.draft.is_some() {
                if helto::icon_button(ui, Icon::Duplicate, "Duplicate enhancement profile", false)
                    .clicked()
                {
                    change = Some(ProfileChange::Duplicate);
                }
                if helto::icon_button(ui, Icon::Trash, "Delete enhancement profile", false)
                    .clicked()
                {
                    change = Some(ProfileChange::Delete);
                }
            }
        });
        if selected != previous {
            change = Some(ProfileChange::Select(selected));
        }
        // Deleting always asks. New or another profile asks only when it would drop
        // edits; a duplicate carries them over.
        if let Some(change) = change {
            if change == ProfileChange::Delete
                || change != ProfileChange::Duplicate && self.enhancement_profile_dirty()
            {
                self.enhancement.confirm = Some(change);
            } else {
                self.apply_profile_change(ui.ctx(), change);
            }
        }
        if let Some(change) = self.enhancement.confirm.clone() {
            let delete = change == ProfileChange::Delete;
            let answer = helto::confirm(
                ui,
                if delete {
                    "Delete this profile?"
                } else {
                    "Discard unsaved profile changes?"
                },
                (
                    Icon::Trash,
                    if delete {
                        "Delete profile"
                    } else {
                        "Discard changes"
                    },
                ),
                "Keep profile",
                None,
            );
            if answer.yes.clicked() {
                self.enhancement.confirm = None;
                self.apply_profile_change(ui.ctx(), change);
            }
            if answer.no.clicked() {
                self.enhancement.confirm = None;
            }
        }
        let Some(draft) = &mut self.enhancement.draft else {
            return;
        };
        if self.enhancement.extra_profile.as_ref() != Some(&draft.id) {
            self.enhancement.extra_profile = Some(draft.id.clone());
            let extras: serde_json::Map<_, _> = draft
                .parameters
                .iter()
                .filter(|(key, _)| !SAMPLING_KEYS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            self.enhancement.extra_parameters = serde_json::to_string_pretty(&extras).unwrap();
            self.enhancement.extra_error = None;
        }
        for (label, value) in [
            ("Name", &mut draft.name),
            ("Base URL", &mut draft.base_url),
            ("Model ID", &mut draft.model),
        ] {
            let label = ui.label(label);
            ui.add(egui::TextEdit::singleline(value).desired_width(f32::INFINITY))
                .labelled_by(label.id);
        }
        ui.horizontal(|ui| {
            ui.label("Story API");
            egui::ComboBox::from_id_salt("enhancement-story-transport")
                .selected_text(draft.story_transport.name())
                .show_ui(ui, |ui| {
                    for transport in [
                        crate::enhancement_types::StoryTransport::Koboldcpp,
                        crate::enhancement_types::StoryTransport::Ollama,
                    ] {
                        ui.selectable_value(
                            &mut draft.story_transport,
                            transport,
                            transport.name(),
                        );
                    }
                });
        });
        let lookup = Some(ModelsRequest {
            profile_id: draft.id.clone(),
            base_url: draft.base_url.clone(),
        });
        let have_models = self.enhancement.models_for == lookup;
        let loading_models = self.enhancement.models_pending == lookup;
        let mut load_models = false;
        ui.horizontal(|ui| {
            if have_models && !self.enhancement.models.is_empty() {
                egui::ComboBox::from_id_salt("enhancement-server-models")
                    .selected_text("Choose model")
                    .width((ui.available_width() - 110.).max(80.))
                    .show_ui(ui, |ui| {
                        for model in &self.enhancement.models {
                            ui.selectable_value(&mut draft.model, model.clone(), model);
                        }
                    });
            }
            let blocked = if self.client.is_none() {
                Some("Not connected to Imagine.")
            } else {
                loading_models.then_some("Loading models.")
            };
            load_models = helto::action(
                ui,
                if have_models {
                    Icon::Retry
                } else {
                    Icon::Download
                },
                if have_models {
                    "Refresh models"
                } else {
                    "Load models"
                },
                false,
                blocked,
            )
            .clicked();
            if loading_models {
                ui.spinner();
            }
        });
        if have_models && let Some(error) = &self.enhancement.models_error {
            ui.colored_label(helto::ERROR, error);
        }
        ui.horizontal(|ui| {
            let label = ui.label("Max tokens");
            ui.add(egui::DragValue::new(&mut draft.max_tokens).range(1..=131072))
                .labelled_by(label.id);
        });
        for (key, label, default, range, integer) in [
            ("top_p", "Top P", 1., 0.0..=1.0, false),
            ("top_k", "Top K", 40., 0.0..=i32::MAX as f64, true),
            ("min_p", "Min P", 0.05, 0.0..=1.0, false),
            ("typical_p", "Typical P", 1., 0.0..=1.0, false),
            (
                "repetition_penalty",
                "Repetition penalty",
                1.,
                0.0..=100.0,
                false,
            ),
            (
                "frequency_penalty",
                "Frequency penalty",
                0.,
                -2.0..=2.0,
                false,
            ),
            (
                "presence_penalty",
                "Presence penalty",
                0.,
                -2.0..=2.0,
                false,
            ),
        ] {
            optional_parameter(
                ui,
                &mut draft.parameters,
                key,
                label,
                default,
                range,
                integer,
            );
        }
        ui.horizontal(|ui| {
            let mut enabled = draft.parameters.contains_key("seed");
            if ui.checkbox(&mut enabled, "Seed").changed() {
                if enabled {
                    draft.parameters.insert("seed".into(), serde_json::json!(0));
                } else {
                    draft.parameters.remove("seed");
                }
            }
            if enabled {
                let mut seed = draft.parameters["seed"].as_i64().unwrap_or(0);
                if ui.add(egui::DragValue::new(&mut seed)).changed() {
                    draft
                        .parameters
                        .insert("seed".into(), serde_json::json!(seed));
                }
            }
        });
        let mut temperature = draft.temperature.is_some();
        ui.horizontal(|ui| {
            if ui.checkbox(&mut temperature, "Temperature").changed() {
                draft.temperature = temperature.then_some(0.7);
            }
            if let Some(value) = &mut draft.temperature {
                ui.add(egui::DragValue::new(value).range(0.0..=2.0).speed(0.05));
            }
        });
        egui::CollapsingHeader::new("Advanced")
            .id_salt("enhancement-advanced")
            .show(ui, |ui| {
                ui.label("Token parameter");
                egui::ComboBox::from_id_salt("enhancement-token-parameter")
                    .selected_text(draft.token_parameter.name())
                    .show_ui(ui, |ui| {
                        for parameter in [
                            TokenParameter::MaxTokens,
                            TokenParameter::MaxCompletionTokens,
                        ] {
                            ui.selectable_value(
                                &mut draft.token_parameter,
                                parameter,
                                parameter.name(),
                            );
                        }
                    });
                let label = ui.label("Extra parameters (JSON)");
                let changed = helto::textarea(
                    ui,
                    &mut self.enhancement.extra_parameters,
                    3,
                    "enhancement-extra-parameters",
                    self.prefs.privacy_mode,
                    true,
                )
                .labelled_by(label.id)
                .changed();
                if changed {
                    match parse_extra_parameters(&self.enhancement.extra_parameters) {
                        Ok(extras) => {
                            draft
                                .parameters
                                .retain(|key, _| SAMPLING_KEYS.contains(&key.as_str()));
                            draft.parameters.extend(extras);
                            self.enhancement.extra_error = None;
                        }
                        Err(error) => self.enhancement.extra_error = Some(error.into()),
                    }
                }
                if let Some(error) = &self.enhancement.extra_error {
                    ui.colored_label(helto::ERROR, error);
                }
            });
        let label = ui.label("System prompt");
        helto::textarea(
            ui,
            &mut draft.system_prompt,
            6,
            "enhancement-system-prompt",
            self.prefs.privacy_mode,
            true,
        )
        .labelled_by(label.id);
        let valid = !draft.system_prompt.trim().is_empty()
            && self.enhancement.extra_error.is_none()
            && !draft.name.trim().is_empty()
            && !draft.model.trim().is_empty();
        let dirty = self.enhancement_profile_dirty();
        let blocked = if self.client.is_none() {
            Some("Not connected to Imagine.")
        } else if self.enhancement.extra_error.is_some() {
            Some("Fix the extra parameters in Advanced.")
        } else if !valid {
            Some("Enter a name, model ID and system prompt.")
        } else if !dirty {
            Some("No unsaved changes.")
        } else {
            None
        };
        if helto::action(ui, Icon::Save, "Save profile", true, blocked).clicked() {
            self.save_enhancement_profile(ui.ctx(), false);
        }
        if load_models {
            self.load_enhancement_models(ui.ctx());
        }
    }
    fn save_enhancement_profile(&mut self, ctx: &egui::Context, delete: bool) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let Some(profile) = self.enhancement.draft.clone() else {
            return;
        };
        self.enhancement.saving = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let id = profile.id.clone();
            let result = runtime().block_on(async {
                if delete {
                    client.delete_enhancement_profile(&id).await
                } else {
                    client
                        .save_enhancement_profile(ProfileUpdate { profile })
                        .await
                }
            });
            send_event(
                &tx,
                &ctx,
                Event::Profiles {
                    result,
                    select: (!delete).then_some(id.clone()),
                    deleted: delete.then_some(id),
                },
            );
        });
    }
}

const SAMPLING_KEYS: &[&str] = &[
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "seed",
];
fn parse_extra_parameters(
    text: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, &'static str> {
    let extras: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text)
        .map_err(|_| "Enter a JSON object with parameter names and values.")?;
    if extras
        .keys()
        .any(|key| reserved_parameter(key) || SAMPLING_KEYS.contains(&key.as_str()))
    {
        return Err(
            "Use the dedicated controls for model, tokens and sampling. Messages, stream and n are reserved.",
        );
    }
    Ok(extras)
}
fn optional_parameter(
    ui: &mut egui::Ui,
    parameters: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
    default: f64,
    range: std::ops::RangeInclusive<f64>,
    integer: bool,
) {
    ui.horizontal(|ui| {
        let mut enabled = parameters.contains_key(key);
        if ui.checkbox(&mut enabled, label).changed() {
            if enabled {
                parameters.insert(
                    key.into(),
                    if integer {
                        serde_json::json!(default as u64)
                    } else {
                        serde_json::json!(default)
                    },
                );
            } else {
                parameters.remove(key);
            }
        }
        if enabled {
            let mut value = parameters[key].as_f64().unwrap_or(default);
            let mut drag = egui::DragValue::new(&mut value)
                .range(range)
                .speed(if integer { 1. } else { 0.01 });
            if integer {
                drag = drag.max_decimals(0);
            }
            if ui.add(drag).changed() {
                parameters.insert(
                    key.into(),
                    if integer {
                        serde_json::json!(value as u64)
                    } else {
                        serde_json::json!(value)
                    },
                );
            }
        }
    });
}

#[cfg(test)]
#[path = "enhancement_ui_tests.rs"]
mod tests;
