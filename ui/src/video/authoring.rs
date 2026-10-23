//! Reviewable source proposals. Closing the window retains its session draft.
use super::*;
use imagine_video_contract::authoring::*;
#[derive(Clone, Copy)]
pub(in crate::app) enum Reply {
    Catalog,
    Providers,
    Draft,
    Applied,
}
pub(super) struct State {
    open: bool,
    pub(super) busy: bool,
    operation: Operation,
    brief: String,
    instruction: String,
    scene_field: Option<SceneField>,
    target_scene: Option<SceneId>,
    target_section: Option<SectionId>,
    duration: f64,
    destination: Destination,
    profile: String,
    provider: String,
    transport_override: bool,
    provider_id: String,
    profiles: Value,
    providers: Value,
    pub(super) draft: Option<Draft>,
    history: Vec<Draft>,
    pub(super) failure: Option<String>,
    selected: usize,
    poll_at: Option<std::time::Instant>,
    owner_session: Option<String>,
    cast: BTreeSet<String>,
    images: std::collections::BTreeMap<AssetId, ImageUse>,
    add_dialogue: bool,
    pairs: std::collections::BTreeMap<AssetId, AssetId>,
    provider_settings: bool,
    show_profiles: bool,
    progress: Value,
    project: Option<ProjectId>,
}
fn provider_transport(state: &State) -> String {
    if state.transport_override {
        return state.provider.clone();
    }
    state
        .providers
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .find(|profile| profile["id"] == state.provider_id)
        })
        .and_then(|profile| profile["story_transport"].as_str())
        .unwrap_or("koboldcpp")
        .into()
}
impl Default for State {
    fn default() -> Self {
        Self {
            open: false,
            busy: false,
            operation: Operation::Outline,
            brief: String::new(),
            instruction: String::new(),
            scene_field: None,
            target_scene: None,
            target_section: None,
            duration: 10.,
            destination: Destination::Append,
            profile: "normal".into(),
            provider: "koboldcpp".into(),
            transport_override: false,
            provider_id: "proxy".into(),
            profiles: Value::Null,
            providers: Value::Null,
            draft: None,
            history: vec![],
            failure: None,
            selected: 0,
            poll_at: None,
            owner_session: None,
            cast: Default::default(),
            images: Default::default(),
            add_dialogue: false,
            pairs: Default::default(),
            provider_settings: false,
            show_profiles: false,
            progress: Value::Null,
            project: None,
        }
    }
}
impl State {
    #[cfg(feature = "ui-preview")]
    pub(super) fn source_image_count(&self) -> usize {
        self.images.len()
    }
    #[cfg(feature = "ui-preview")]
    pub(super) fn progress_ready(&self, completed: u64, eta: bool) -> bool {
        self.progress["completed"]
            .as_u64()
            .is_some_and(|value| value >= completed)
            && (!eta
                || self.progress["eta_seconds"]
                    .as_f64()
                    .is_some_and(|value| value > 0.))
    }
    pub(super) fn add_image(&mut self, id: AssetId) {
        self.images.entry(id).or_insert(ImageUse::SettingReference);
    }
    pub(super) fn session(&mut self, id: &str) {
        if self.owner_session.as_deref().is_some_and(|old| old != id) {
            self.busy = false;
            if let Some(draft) = &mut self.draft
                && draft.phase != Phase::Applied
            {
                draft.phase = Phase::Cancelled;
                self.failure=Some("The server session ended. This local proposal is retained; it was not resumed.".into());
            }
        }
        self.owner_session = Some(id.into());
    }
}
fn control(ui: &mut egui::Ui, name: &str, response: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), response);
    let _ = (ui, name, response);
}
fn image_label(id: &str, images: &[StoryImage]) -> String {
    if id.is_empty() {
        return "None".into();
    }
    images
        .iter()
        .find(|i| i.id == id)
        .map(|i| i.label.clone())
        .unwrap_or_else(|| "Unavailable source image".into())
}
fn source_image_label(project: &VideoProject, id: &AssetId) -> String {
    project
        .assets
        .get(id)
        .filter(|asset| !asset.name.is_empty())
        .map(|asset| asset.name.clone())
        .unwrap_or_else(|| {
            format!(
                "Source image {}",
                project
                    .assets
                    .values()
                    .filter(|asset| asset.kind == MediaKind::Image)
                    .position(|asset| &asset.id == id)
                    .map_or(0, |i| i + 1)
            )
        })
}
impl ImagineApp {
    pub(super) fn timeline_open_authoring(&mut self, ctx: &egui::Context, operation: Operation) {
        self.timeline_open_authoring_target(ctx, operation, None);
    }
    pub(super) fn timeline_open_field_enhance(&mut self, ctx: &egui::Context, field: SceneField) {
        self.timeline_open_authoring_target(ctx, Operation::EnhanceScene, Some(field));
    }
    fn timeline_open_authoring_target(
        &mut self,
        ctx: &egui::Context,
        operation: Operation,
        scene_field: Option<SceneField>,
    ) {
        if self
            .timeline
            .authoring
            .draft
            .as_ref()
            .is_some_and(|d| d.phase.busy())
        {
            self.timeline.authoring.open = true;
            return;
        }

        let project = self
            .timeline
            .document
            .as_ref()
            .map(|d| d.project.id.clone());
        if self.timeline.authoring.project != project {
            self.timeline.authoring.cast.clear();
            self.timeline.authoring.images.clear();
            self.timeline.authoring.pairs.clear();
            self.timeline.authoring.add_dialogue = false;
            self.timeline.authoring.project = project.clone();
        }
        let target_scene = (operation != Operation::Outline)
            .then(|| {
                self.timeline
                    .document
                    .as_ref()
                    .and_then(|d| self.timeline.primary_scene(&d.project))
            })
            .flatten();
        let target_section = (operation == Operation::EnhanceSection)
            .then(|| self.timeline.section.clone())
            .flatten();
        if self.timeline.authoring.draft.as_ref().is_some_and(|d| {
            !d.phase.busy()
                && (Some(&d.project_id) != project.as_ref()
                    || d.request.operation != operation
                    || d.request.scene_id != target_scene
                    || d.request.section_id != target_section
                    || d.request.scene_field != scene_field)
        }) && let Some(draft) = self.timeline.authoring.draft.take()
        {
            self.timeline.authoring.history.push(draft);
        }
        self.timeline.authoring.scene_field = scene_field;
        self.timeline.authoring.target_scene = target_scene;
        self.timeline.authoring.target_section = target_section;
        self.timeline.authoring.open = true;
        self.timeline.authoring.operation = operation;
        if self.timeline.authoring.profiles.is_null() {
            self.authoring_call(
                ctx,
                Reply::Catalog,
                false,
                "/v1/video/authoring/profiles".into(),
                None,
            );
        }
    }
    fn authoring_call(
        &mut self,
        ctx: &egui::Context,
        kind: Reply,
        post: bool,
        path: String,
        body: Option<Value>,
    ) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        if self.timeline.authoring.busy {
            return;
        }
        self.timeline.authoring.busy = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_request(&session, post, &path, body.as_ref()));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Authoring {
                    session,
                    kind,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_authoring_reply(
        &mut self,
        ctx: &egui::Context,
        session: String,
        kind: Reply,
        result: Result<Value, client::TimelineFailure>,
    ) {
        if self.timeline.session.as_deref() != Some(&session) {
            return;
        }
        self.timeline.authoring.busy = false;
        let value = match result {
            Ok(v) => v,
            Err(e) => {
                self.timeline.authoring.failure = Some(
                    if e.code == "authoring_revision_conflict" {
                        "The project changed. This draft is retained; Apply is blocked. Start a new proposal against the current project.".into()
                    } else if matches!(e.code.as_str(), "storage_io_failed" | "storage_error") {
                        "Could not save the proposal. Your edits are retained; retry the action."
                            .into()
                    } else {
                        e.message
                    },
                );
                return;
            }
        };
        self.timeline.authoring.failure = None;
        match kind {
            Reply::Catalog => {
                if self.timeline.authoring.profiles.is_null() {
                    self.timeline.authoring.profile = self
                        .prefs
                        .preferred_prompt_profile
                        .clone()
                        .unwrap_or_else(|| value["default_id"].as_str().unwrap_or("normal").into());
                }
                self.timeline.authoring.profiles = value;
                self.authoring_call(
                    ctx,
                    Reply::Providers,
                    false,
                    "/v1/enhancement/profiles".into(),
                    None,
                );
            }
            Reply::Providers => self.timeline.authoring.providers = value,
            Reply::Draft => {
                self.timeline.authoring.progress = value["progress"].clone();
                if let Ok(draft) = serde_json::from_value::<Draft>(value) {
                    if draft.phase == Phase::Applied
                        && self
                            .timeline
                            .document
                            .as_ref()
                            .is_some_and(|d| d.project.id == draft.project_id)
                    {
                        self.timeline_refresh(ctx);
                    }
                    self.timeline.authoring.draft = Some(draft);
                    self.timeline.authoring.poll_at = Some(std::time::Instant::now());
                } else {
                    self.timeline.authoring.failure =
                        Some("Could not read the proposal. Reconnect to this draft.".into());
                }
            }
            Reply::Applied => {
                if let Some(d) = &mut self.timeline.authoring.draft {
                    d.phase = Phase::Applied;
                }
                self.timeline_message(
                    ctx,
                    Event::Document {
                        epoch: self.timeline.epoch,
                        replace: false,
                        before: self
                            .timeline
                            .document
                            .as_ref()
                            .map(|s| Box::new(s.project.clone())),
                        history: 0,
                        result: Ok(value),
                    },
                );
            }
        }
    }
    pub(super) fn timeline_authoring_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.authoring.open {
            return;
        }
        if let Some(d) = &self.timeline.authoring.draft
            && d.phase.busy()
            && !self.timeline.authoring.busy
            && self
                .timeline
                .authoring
                .poll_at
                .is_none_or(|t| t.elapsed() > Duration::from_millis(200))
        {
            self.authoring_call(
                ctx,
                Reply::Draft,
                false,
                format!("/v1/video/authoring/{}", d.id),
                None,
            );
        }
        ctx.request_repaint_after(Duration::from_millis(200));
        let mut state = std::mem::take(&mut self.timeline.authoring);
        let mut open = state.open;
        let mut call = None;
        let mut new_proposal = false;
        let mut close_review = false;
        let mut pick_images = false;
        let document = self.timeline.document.clone();
        if let Some(d) = &document {
            for value in d.raw["authoring"].as_array().into_iter().flatten() {
                if let Ok(draft) = serde_json::from_value::<Draft>(value.clone())
                    && !state.history.iter().any(|old| old.id == draft.id)
                    && state
                        .draft
                        .as_ref()
                        .is_none_or(|current| current.id != draft.id)
                {
                    state.history.push(draft);
                }
            }
        }
        egui::Window::new("Story & Enhance").id(egui::Id::new("video-authoring")).open(&mut open).default_width(760.).default_height(580.).max_height((ctx.content_rect().height()-70.).max(200.)).vscroll(true).show(ctx,|ui|{
            let reveal=helto::reveal_in_viewport(ui,self.prefs.privacy_mode);
            helto::private_scope(ui,!reveal,|ui|{
                #[cfg(feature="ui-preview")]{let r=ui.interact(ui.clip_rect(),ui.id().with("qa-story-scroll"),egui::Sense::hover());super::qa::control(ui,"story-scroll".into(),&r);}

            ui.horizontal_wrapped(|ui|{ui.label("Review and edit generated source before applying. Manual authoring remains available.");let r=helto::icon_button(ui,crate::icons::Icon::Settings,"Provider settings",false);control(ui,"story-provider-settings",&r);if r.clicked(){state.provider_settings=true;}let r=helto::icon_button(ui,crate::icons::Icon::Edit,"View system prompt files and generation parameters",false);control(ui,"story-profile-settings",&r);if r.clicked(){state.show_profiles = !state.show_profiles;}});
            if state.show_profiles && let Some(profile)=state.profiles["profiles"].as_array().and_then(|rows|rows.iter().find(|p|p["id"]==state.profile)) {
                    ui.group(|ui| {
                        ui.strong(format!("{} · prompt files",profile["label"].as_str().unwrap_or(&state.profile)));
                        for file in profile["prompt_files"].as_array().into_iter().flatten() {
                            ui.horizontal_wrapped(|ui| { ui.monospace(file["file"].as_str().unwrap_or("")); ui.weak(if file["inherited"]==true {format!("Inherited from {}",file["source_profile_id"].as_str().unwrap_or(""))}else{"Own".into()}); });
                        }
                        ui.collapsing("Generation parameters",|ui| {let mut text=serde_json::to_string_pretty(&profile["generation"]).unwrap_or_default();ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY).desired_rows(10).interactive(false));});
                        ui.weak("System prompts use file-level inheritance. Reload prompts after editing the selected source profile files.");
                    });
            }
            let (operation,field)=state.draft.as_ref().map_or((state.operation,state.scene_field),|d|(d.request.operation,d.request.scene_field));
            if operation==Operation::EnhanceScene {ui.strong(format!("Enhance {}",field.unwrap_or_default().label()));}else if operation==Operation::EnhanceSection {ui.strong("Enhance Section prompt");}
            ui.horizontal(|ui|{ui.weak("Project cast · image inventory · editable source and bindings");let r=ui.button("Close review");control(ui,"story-close",&r);close_review=r.clicked();});
            if let Some(error)=&state.failure{ui.colored_label(helto::ERROR,error);if let Some(d)=&state.draft {let r=ui.button("Reconnect proposal");control(ui,"story-reconnect",&r);if r.clicked(){call=Some((Reply::Draft,false,format!("/v1/video/authoring/{}",d.id),None));}}}
            if !state.history.is_empty()&&!state.busy {
                let menu=egui::ComboBox::from_id_salt("recent-story-proposals").selected_text("Recent proposals").show_ui(ui,|ui|{
                    for (i,draft) in state.history.iter().enumerate() {let r=ui.button(format!("{:?} · revision {} · {} scenes",draft.request.operation,draft.request.expected_project_revision,draft.scenes.len()));control(ui,&format!("story-history-{i}"),&r);if r.clicked(){state.draft=Some(draft.clone());}}
                });control(ui,"story-history-menu",&menu.response);
            }
            if let Some(draft)=&mut state.draft {
                let busy=state.busy||draft.phase.busy();
                ui.horizontal_wrapped(|ui|{
                    ui.strong(format!("{:?}",draft.phase));
                    ui.weak(format!("{} · {}",draft.prompt_profile_id,draft.provider_snapshot["model"].as_str().unwrap_or("Provider")));
                    if !draft.profile_warning.is_empty(){ui.label(&draft.profile_warning);}
                    for warning in &draft.warnings{ui.colored_label(helto::ACCENT,warning);}
                    if busy {ui.spinner();let r=ui.button("Cancel");control(ui,"story-cancel",&r);if r.clicked(){call=Some((Reply::Draft,true,format!("/v1/video/authoring/{}/cancel",draft.id),Some(json!({}))));}}
                    let r=ui.add_enabled(!busy,egui::Button::new("New proposal"));control(ui,"story-new",&r);if r.clicked(){state.history.push(draft.clone());new_proposal=true;}
                });
                if state.progress["draft_id"]==draft.id {
                    let completed=state.progress["completed"].as_u64().unwrap_or(0);let total=state.progress["total"].as_u64().unwrap_or(1).max(1);
                    ui.add(egui::ProgressBar::new(completed as f32/total as f32).text(format!("{completed} / {total}")));
                    let elapsed=state.progress["elapsed_seconds"].as_f64().unwrap_or(0.);ui.weak(if let Some(eta)=state.progress["eta_seconds"].as_f64(){format!("{elapsed:.1} s elapsed · about {eta:.1} s remaining")}else{format!("{elapsed:.1} s elapsed")});
                }
                if let Some(failure)=&draft.failure{ui.colored_label(helto::ERROR,format!("{failure} · previous completed scenes are retained"));}
                if !busy&&draft.outline.is_none()&&matches!(draft.phase,Phase::Failed|Phase::Cancelled){let r=ui.button("Retry proposal");control(ui,"story-retry",&r);if r.clicked(){call=Some((Reply::Draft,true,format!("/v1/video/authoring/{}/retry",draft.id),Some(json!({}))));}}
                if !draft.inventory.is_empty(){let header=ui.collapsing(format!("Image inventory · {}",draft.inventory.len()),|ui|{for note in &draft.inventory{ui.strong(image_label(&note.id,&draft.request.images));ui.label(&note.note);}});control(ui,"story-inventory-open",&header.header_response);}
                if let Some(cast)=draft.selected_cast.as_array(){ui.weak(cast.iter().filter_map(|c|c["reference_tag"].as_str()).collect::<Vec<_>>().join(" · "));}
                let stale=document.as_ref().is_none_or(|s|s.project.id!=draft.project_id||s.project.project_revision!=draft.request.expected_project_revision);
                if stale&&draft.phase!=Phase::Applied {ui.colored_label(helto::ERROR,"Project changed. Apply is blocked; the proposal remains editable.");}
                let mut expand=None;let mut apply=false;
                let editable=draft.phase!=Phase::Applied;
                ui.add_enabled_ui(!busy,|ui|{
                    ui.horizontal_wrapped(|ui|{
                        if draft.outline.is_some(){let r=ui.add_enabled(editable,egui::Button::new(if draft.failed_index.is_some(){"Retry from failed scene"}else{"Expand scenes"}));control(ui,"story-expand",&r);if r.clicked(){expand=Some(draft.failed_index.unwrap_or(0));}}
                        let r=ui.add_enabled(!stale&&draft.phase==Phase::Review,egui::Button::new("Apply reviewed proposal"));control(ui,"story-apply",&r);apply=r.clicked();
                    });
                    ui.vertical(|ui|{
                        if let Some(text)=&mut draft.text {
                            if let Some(d)=&document && let Some(s)=d.project.scenes.iter().find(|s|Some(&s.id)==draft.request.scene_id.as_ref()) {
                                ui.collapsing("Original",|ui|{ui.label(if draft.request.operation==Operation::EnhanceSection{s.sections.iter().find(|s|Some(&s.id)==draft.request.section_id.as_ref()).map(|s|s.prompt.clone()).unwrap_or_default()}else{draft.request.scene_field.unwrap_or_default().source(&d.project,s)});});
                            }
                            ui.label("Proposal");let r=ui.add(egui::TextEdit::multiline(text).desired_width(f32::INFINITY).desired_rows(8).interactive(editable));control(ui,"story-proposal-text",&r);
                        }
                        if let Some(outline)=&mut draft.outline {
                            ui.horizontal_wrapped(|ui|{for i in 0..outline.scenes.len(){let r=ui.selectable_label(state.selected==i,format!("Scene {}",i+1));control(ui,&format!("story-scene-{i}"),&r);if r.clicked(){state.selected=i;}}});
                            let index=state.selected.min(outline.scenes.len()-1);
                            let ready=draft.scenes.iter().filter(|s|s.is_some()).count();ui.label(format!("{ready} / {} scenes expanded",outline.scenes.len()));
                            if let Some(duration)=draft.planned_durations.get(index){ui.weak(format!("Planned duration: {duration:.3} s"));}
                            let expanded=draft.scenes[index].is_some();
                            let source=draft.scenes[index].as_mut().unwrap_or(&mut outline.scenes[index]);
                            if !source.image_ids.is_empty(){ui.weak(format!("Setting images: {}",source.image_ids.iter().map(|id|image_label(id,&draft.request.images)).collect::<Vec<_>>().join(", ")));}
                            ui.add_enabled_ui(!expanded,|ui|{ui.horizontal(|ui|{ui.label("Relative weight");ui.add(egui::DragValue::new(&mut source.weight).range(0.001..=100000.));egui::ComboBox::from_id_salt("story-boundary").selected_text(&source.boundary_from_previous).show_ui(ui,|ui|{for (id,label) in [("hard_cut","Hard cut"),("continuous_shot","Continuous shot")]{ui.selectable_value(&mut source.boundary_from_previous,id.into(),label);}});});});
                            ui.add_enabled_ui(editable,|ui|{
                            ui.label("Title");let r=ui.add(egui::TextEdit::singleline(&mut source.title).desired_width(f32::INFINITY));control(ui,"story-scene-title",&r);
                            ui.label("Summary");let r=ui.add(egui::TextEdit::multiline(&mut source.summary).desired_rows(2).desired_width(f32::INFINITY));control(ui,"story-scene-summary",&r);
                            ui.label("Description");let r=ui.add(egui::TextEdit::multiline(&mut source.description).desired_rows(3).desired_width(f32::INFINITY));control(ui,"story-scene-description",&r);
                            ui.label("Ambient sound");let r=ui.add(egui::TextEdit::singleline(&mut source.ambient_sound).desired_width(f32::INFINITY));control(ui,"story-scene-ambient",&r);
                            ui.label("Music");let r=ui.add(egui::TextEdit::singleline(&mut source.non_diegetic_music).desired_width(f32::INFINITY));control(ui,"story-scene-music",&r);
                            for (i,section) in source.sections.iter_mut().enumerate(){ui.horizontal(|ui| { ui.label(format!("Section {}",i+1)); if !expanded { ui.label("Relative weight"); let r = ui.add(egui::DragValue::new(&mut section.weight).speed(0.1)); control(ui, &format!("story-section-weight-{i}"), &r); } });let r=ui.add(egui::TextEdit::multiline(&mut section.prompt).desired_rows(3).desired_width(f32::INFINITY));control(ui,&format!("story-section-{i}"),&r);if !section.image_ids.is_empty()||!section.first_frame_id.is_empty()||!section.last_frame_id.is_empty(){ui.weak(format!("Images: {} · first: {} · last: {}",section.image_ids.iter().map(|id|image_label(id,&draft.request.images)).collect::<Vec<_>>().join(", "),image_label(&section.first_frame_id,&draft.request.images),image_label(&section.last_frame_id,&draft.request.images)));}
                            if let Some(intent)=&mut section.dialogue_intent{ui.weak(format!("Dialogue: {} · at most {} turns",intent.participants.join(" / "),intent.max_turns));ui.text_edit_singleline(&mut intent.purpose);}}
                            });
                        }
                    });
                });
                if let Some(index)=expand {call=Some((Reply::Draft,true,format!("/v1/video/authoring/{}/review",draft.id),Some(json!({"schema_version":1,"outline":draft.outline,"scenes":draft.scenes,"text":draft.text,"then_expand":index}))));}
                if apply {call=Some((Reply::Applied,true,format!("/v1/video/authoring/{}/review",draft.id),Some(json!({"schema_version":1,"outline":draft.outline,"scenes":(draft.request.operation==Operation::Outline).then_some(&draft.scenes),"text":draft.text,"then_apply":draft.request.expected_project_revision}))));}
            } else {
                ui.horizontal_wrapped(|ui|{
                    let menu=egui::ComboBox::from_id_salt("authoring-profile").selected_text(&state.profile).show_ui(ui,|ui|{for p in state.profiles["profiles"].as_array().into_iter().flatten(){if let Some(id)=p["id"].as_str(){let r=ui.selectable_value(&mut state.profile,id.into(),p["label"].as_str().unwrap_or(id));control(ui,&format!("story-profile-{id}"),&r);}}});control(ui,"story-profile-menu",&menu.response);
                    let r=helto::icon_button(ui,crate::icons::Icon::Save,"Use this prompt profile by default",self.prefs.preferred_prompt_profile.as_deref()==Some(state.profile.as_str()));control(ui,"story-profile-default",&r);if r.clicked(){self.prefs.preferred_prompt_profile=Some(state.profile.clone());self.save_preferences();}
                    let name=state.providers.as_array().and_then(|rows|rows.iter().find(|p|p["id"]==state.provider_id)).and_then(|p|p["name"].as_str()).unwrap_or(&state.provider_id).to_owned();
                    let menu=egui::ComboBox::from_id_salt("authoring-provider").selected_text(name).show_ui(ui,|ui|{for p in state.providers.as_array().into_iter().flatten(){if let Some(id)=p["id"].as_str(){let r=ui.selectable_value(&mut state.provider_id,id.into(),p["name"].as_str().unwrap_or(id));control(ui,&format!("story-provider-{id}"),&r);}}});control(ui,"story-provider-menu",&menu.response);
                    ui.weak(format!("API: {}",provider_transport(&state)));
                    let r=helto::icon_button(ui,crate::icons::Icon::Retry,"Reload prompts",false);control(ui,"story-reload",&r);if r.clicked(){call=Some((Reply::Catalog,false,"/v1/video/authoring/profiles".into(),None));}
                });
                ui.collapsing("Advanced transport override",|ui| {ui.checkbox(&mut state.transport_override,"Override the connection profile's Story API");if state.transport_override {egui::ComboBox::from_id_salt("authoring-transport").selected_text(&state.provider).show_ui(ui,|ui|{for id in ["koboldcpp","ollama"]{ui.selectable_value(&mut state.provider,id.into(),id);}});}});
                if state.operation==Operation::Outline {
                    if let Some(d)=&document {
                        let header=ui.collapsing("Cast & source images",|ui|{
                            let r=ui.button("Add source images…");control(ui,"story-pick-images",&r);pick_images=r.clicked();
                            for (i,r)in d.project.creative.references.iter().enumerate(){let mut selected=state.cast.contains(&r.id);let response=ui.checkbox(&mut selected,format!("{} · @{}{}",r.name,r.alias,if r.enabled{""}else{" · disabled in project"}));control(ui,&format!("story-cast-{i}"),&response);if response.changed(){state.cast.remove(&r.id);if selected{state.cast.insert(r.id.clone());}}}
                            let r=ui.checkbox(&mut state.add_dialogue,"Plan dialogue");control(ui,"story-dialogue",&r);
                            for (i,asset)in d.project.assets.values().filter(|a|a.kind==MediaKind::Image).enumerate(){let mut selected=state.images.contains_key(&asset.id);let response=ui.checkbox(&mut selected,format!("Source image {}",i+1));control(ui,&format!("story-image-{i}"),&response);if response.changed(){state.images.remove(&asset.id);if selected{state.images.insert(asset.id.clone(),ImageUse::SettingReference);}}
                            if let Some(usage)=state.images.get_mut(&asset.id){let menu=egui::ComboBox::from_id_salt(("story-image-use",i)).selected_text(format!("{usage:?}")).show_ui(ui,|ui|{for(value,label)in [(ImageUse::SettingReference,"Setting reference"),(ImageUse::RequiredMoment,"Required moment"),(ImageUse::StartFrame,"Directed start frame"),(ImageUse::EndFrame,"Directed end frame")]{let r=ui.selectable_value(usage,value,label);control(ui,&format!("story-image-{i}-use-{value:?}"),&r);}});control(ui,&format!("story-image-{i}-usage"),&menu.response);}}
                            for (start,usage) in &state.images { if *usage!=ImageUse::StartFrame {continue;} let selected=state.pairs.entry(start.clone()).or_insert_with(||AssetId(String::new()));let menu=egui::ComboBox::from_id_salt(("story-pair",&start.0)).selected_text(if selected.0.is_empty(){"Choose directed end frame".to_owned()}else{source_image_label(&d.project,selected)}).show_ui(ui,|ui|{for end in state.images.keys() {let r=ui.selectable_value(selected,end.clone(),source_image_label(&d.project,end));control(ui,&format!("story-pair-{}-{}",start.0,end.0),&r);}});control(ui,&format!("story-pair-{}",start.0),&menu.response);}

                        });control(ui,"story-sources-open",&header.header_response);
                    }
                    ui.label("Story brief");let r=ui.add(egui::TextEdit::multiline(&mut state.brief).desired_rows(5).desired_width(f32::INFINITY));control(ui,"story-brief",&r);
                    ui.horizontal(|ui|{ui.label("Duration");ui.add(egui::DragValue::new(&mut state.duration).range(0.25..=3600.).suffix(" s"));for (d,label) in [(Destination::Append,"Append"),(Destination::Replace,"Replace scenes"),(Destination::Continue,"Continue")]{let name=serde_json::to_value(d).unwrap().as_str().unwrap().to_owned();let r=ui.selectable_value(&mut state.destination,d,label);control(ui,&format!("story-destination-{name}"),&r);}});
                }
                ui.label("Additional instruction");let r=ui.add(egui::TextEdit::multiline(&mut state.instruction).desired_rows(3).desired_width(f32::INFINITY));control(ui,"story-instruction",&r);
                let r=ui.add_enabled(!state.busy&&document.is_some(),egui::Button::new(if state.operation==Operation::Outline{"Generate outline"}else{"Generate enhancement"}));control(ui,"story-generate",&r);
                if r.clicked()&&let Some(d)=&document{call=Some((Reply::Draft,true,format!("/v1/video/projects/{}/authoring",d.project.id.0),Some(json!(Request{schema_version:1,expected_project_revision:d.project.project_revision,provider_profile_id:state.provider_id.clone(),provider:provider_transport(&state),prompt_profile_id:state.profile.clone(),operation:state.operation,scene_field:state.scene_field,scene_id:state.target_scene.clone(),section_id:state.target_section.clone(),brief:state.brief.clone(),instruction:state.instruction.clone(),duration_seconds:state.duration,destination:state.destination,cast:if state.operation==Operation::Outline{state.cast.iter().cloned().collect()}else{vec![]},images:if state.operation==Operation::Outline{state.images.iter().enumerate().map(|(i,(asset,usage))|StoryImage{pair_with:state.pairs.get(asset).filter(|id|!id.0.is_empty()).map(|id|id.0.clone()),id:asset.0.clone(),asset:asset.clone(),label:format!("Source image {}",i+1),usage:*usage}).collect()}else{vec![]},add_dialogue:state.operation==Operation::Outline&&state.add_dialogue}))));}
            }
            });
        });
        if new_proposal {
            state.draft = None;
        }
        state.open = open && !close_review;
        if state.provider_settings {
            let mut settings_open = true;
            egui::Window::new("Story provider settings")
                .id(egui::Id::new("story-provider-settings-window"))
                .open(&mut settings_open)
                .default_width(600.)
                .max_height((ctx.content_rect().height() - 70.).max(200.))
                .vscroll(true)
                .show(ctx, |ui| {
                    let r = ui.button("Close settings");
                    control(ui, "story-provider-close", &r);
                    if r.clicked() {
                        state.provider_settings = false;
                    }
                    self.enhancement_provider_settings(ui);
                });
            if !settings_open || !state.provider_settings {
                state.provider_settings = false;
                call = Some((
                    Reply::Providers,
                    false,
                    "/v1/enhancement/profiles".into(),
                    None,
                ));
            }
        }
        self.timeline.authoring = state;
        if pick_images {
            self.timeline_pick_media(media_picker::Target::StoryImage);
        }
        if let Some((kind, post, path, mut body)) = call {
            if path.ends_with("/review") {
                let expand = body
                    .as_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove("then_expand");
                let apply = body
                    .as_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove("then_apply");
                self.authoring_review_then(ctx, kind, path, body.unwrap(), expand, apply);
            } else {
                self.authoring_call(ctx, kind, post, path, body);
            }
        }
    }
    fn authoring_review_then(
        &mut self,
        ctx: &egui::Context,
        kind: Reply,
        path: String,
        body: Value,
        expand: Option<Value>,
        apply: Option<Value>,
    ) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        if self.timeline.authoring.busy {
            return;
        }
        self.timeline.authoring.busy = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                client
                    .timeline_request(&session, true, &path, Some(&body))
                    .await?;
                let base = path.trim_end_matches("/review");
                let (path, body) = if let Some(index) = expand {
                    (
                        format!("{base}/expand"),
                        json!({"schema_version":1,"start_index":index}),
                    )
                } else {
                    (
                        format!("{base}/apply"),
                        json!({"schema_version":1,"expected_project_revision":apply}),
                    )
                };
                client
                    .timeline_request(&session, true, &path, Some(&body))
                    .await
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Authoring {
                    session,
                    kind,
                    result,
                }),
            );
        });
    }
}

#[cfg(test)]
mod provider_tests {
    use super::*;
    #[test]
    fn selected_connection_owns_transport_unless_explicitly_overridden() {
        let mut state = State {
            providers: json!([{"id":"proxy","story_transport":"ollama"},{"id":"legacy"}]),
            ..Default::default()
        };
        assert_eq!(provider_transport(&state), "ollama");
        state.transport_override = true;
        assert_eq!(provider_transport(&state), "koboldcpp");
        state.transport_override = false;
        state.provider_id = "legacy".into();
        assert_eq!(provider_transport(&state), "koboldcpp");
    }
}
