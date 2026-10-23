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
    duration: f64,
    destination: Destination,
    profile: String,
    provider: String,
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
    project: Option<ProjectId>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            open: false,
            busy: false,
            operation: Operation::Outline,
            brief: String::new(),
            instruction: String::new(),
            duration: 10.,
            destination: Destination::Append,
            profile: "normal".into(),
            provider: "koboldcpp".into(),
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
            project: None,
        }
    }
}
impl State {
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
impl ImagineApp {
    pub(super) fn timeline_open_authoring(&mut self, ctx: &egui::Context, operation: Operation) {
        let project = self
            .timeline
            .document
            .as_ref()
            .map(|d| d.project.id.clone());
        if self.timeline.authoring.project != project {
            self.timeline.authoring.cast.clear();
            self.timeline.authoring.images.clear();
            self.timeline.authoring.add_dialogue = false;
            self.timeline.authoring.project = project;
        }
        if self
            .timeline
            .authoring
            .draft
            .as_ref()
            .is_some_and(|d| d.request.operation != operation && !d.phase.busy())
            && let Some(draft) = self.timeline.authoring.draft.take()
        {
            self.timeline.authoring.history.push(draft);
        }
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
            if !reveal {ui.label("Hidden proposal · hover to reveal");return;}
            ui.label("Review and edit generated source before applying. Manual authoring remains available.");
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
                    egui::ScrollArea::vertical().max_height(360.).id_salt("story-review").show(ui,|ui|{
                        if let Some(text)=&mut draft.text {
                            if let Some(s)=document.as_ref().and_then(|d|d.project.scenes.iter().find(|s|Some(&s.id)==draft.request.scene_id.as_ref())) {
                                ui.collapsing("Original",|ui|{ui.label(if draft.request.operation==Operation::EnhanceSection{s.sections.iter().find(|s|Some(&s.id)==draft.request.section_id.as_ref()).map(|s|s.prompt.as_str()).unwrap_or("")}else{&s.description});});
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
                            ui.label("Title");ui.text_edit_singleline(&mut source.title);
                            ui.label("Summary");ui.text_edit_multiline(&mut source.summary);
                            ui.label("Description");let r=ui.add(egui::TextEdit::multiline(&mut source.description).desired_rows(3).desired_width(f32::INFINITY));control(ui,"story-scene-description",&r);
                            ui.horizontal(|ui|{ui.label("Ambient sound");ui.text_edit_singleline(&mut source.ambient_sound);});
                            ui.horizontal(|ui|{ui.label("Music");ui.text_edit_singleline(&mut source.non_diegetic_music);});
                            for (i,section) in source.sections.iter_mut().enumerate(){ui.label(format!("Section {}",i+1));let r=ui.add(egui::TextEdit::multiline(&mut section.prompt).desired_rows(3).desired_width(f32::INFINITY));control(ui,&format!("story-section-{i}"),&r);if !section.image_ids.is_empty()||!section.first_frame_id.is_empty()||!section.last_frame_id.is_empty(){ui.weak(format!("Images: {} · first: {} · last: {}",section.image_ids.iter().map(|id|image_label(id,&draft.request.images)).collect::<Vec<_>>().join(", "),image_label(&section.first_frame_id,&draft.request.images),image_label(&section.last_frame_id,&draft.request.images)));}
                            if let Some(intent)=&mut section.dialogue_intent{ui.weak(format!("Dialogue: {} · at most {} turns",intent.participants.join(" / "),intent.max_turns));ui.text_edit_singleline(&mut intent.purpose);}}
                            });
                        }
                    });
                });
                if let Some(index)=expand {call=Some((Reply::Draft,true,format!("/v1/video/authoring/{}/review",draft.id),Some(json!({"schema_version":1,"outline":draft.outline,"scenes":draft.scenes,"text":draft.text,"then_expand":index}))));}
                if apply {call=Some((Reply::Applied,true,format!("/v1/video/authoring/{}/review",draft.id),Some(json!({"schema_version":1,"outline":draft.outline,"scenes":(draft.request.operation==Operation::Outline).then_some(&draft.scenes),"text":draft.text,"then_apply":draft.request.expected_project_revision}))));}
            } else {
                ui.horizontal_wrapped(|ui|{
                    egui::ComboBox::from_id_salt("authoring-profile").selected_text(&state.profile).show_ui(ui,|ui|{for p in state.profiles["profiles"].as_array().into_iter().flatten(){if let Some(id)=p["id"].as_str(){ui.selectable_value(&mut state.profile,id.into(),p["label"].as_str().unwrap_or(id));}}});
                    egui::ComboBox::from_id_salt("authoring-provider").selected_text(&state.provider_id).show_ui(ui,|ui|{for p in state.providers.as_array().into_iter().flatten(){if let Some(id)=p["id"].as_str(){ui.selectable_value(&mut state.provider_id,id.into(),p["name"].as_str().unwrap_or(id));}}});
                    egui::ComboBox::from_id_salt("authoring-transport").selected_text(&state.provider).show_ui(ui,|ui|{for id in ["koboldcpp","ollama"]{ui.selectable_value(&mut state.provider,id.into(),id);}});
                    let r=ui.button("Reload prompts");control(ui,"story-reload",&r);if r.clicked(){call=Some((Reply::Catalog,false,"/v1/video/authoring/profiles".into(),None));}
                });
                if state.operation==Operation::Outline {
                    if let Some(d)=&document {
                        let header=ui.collapsing("Cast & source images",|ui|{
                            for (i,r)in d.project.creative.references.iter().enumerate(){let mut selected=state.cast.contains(&r.id);let response=ui.checkbox(&mut selected,format!("{} · @{}{}",r.name,r.alias,if r.enabled{""}else{" · disabled in project"}));control(ui,&format!("story-cast-{i}"),&response);if response.changed(){state.cast.remove(&r.id);if selected{state.cast.insert(r.id.clone());}}}
                            let r=ui.checkbox(&mut state.add_dialogue,"Plan dialogue");control(ui,"story-dialogue",&r);
                            for (i,asset)in d.project.assets.values().filter(|a|a.kind==MediaKind::Image).enumerate(){let mut selected=state.images.contains_key(&asset.id);let response=ui.checkbox(&mut selected,format!("Source image {}",i+1));control(ui,&format!("story-image-{i}"),&response);if response.changed(){state.images.remove(&asset.id);if selected{state.images.insert(asset.id.clone(),ImageUse::SettingReference);}}
                            if let Some(usage)=state.images.get_mut(&asset.id){egui::ComboBox::from_id_salt(("story-image-use",i)).selected_text(format!("{usage:?}")).show_ui(ui,|ui|{for(value,label)in [(ImageUse::SettingReference,"Setting reference"),(ImageUse::RequiredMoment,"Required moment"),(ImageUse::StartFrame,"Directed start frame"),(ImageUse::EndFrame,"Directed end frame")]{ui.selectable_value(usage,value,label);}});}}
                        });control(ui,"story-sources-open",&header.header_response);
                    }
                    ui.label("Story brief");let r=ui.add(egui::TextEdit::multiline(&mut state.brief).desired_rows(5).desired_width(f32::INFINITY));control(ui,"story-brief",&r);
                    ui.horizontal(|ui|{ui.label("Duration");ui.add(egui::DragValue::new(&mut state.duration).range(0.25..=3600.).suffix(" s"));for (d,label) in [(Destination::Append,"Append"),(Destination::Replace,"Replace scenes"),(Destination::Continue,"Continue")]{ui.selectable_value(&mut state.destination,d,label);}});
                }
                ui.label("Additional instruction");let r=ui.add(egui::TextEdit::multiline(&mut state.instruction).desired_rows(3).desired_width(f32::INFINITY));control(ui,"story-instruction",&r);
                let r=ui.add_enabled(!state.busy&&document.is_some(),egui::Button::new(if state.operation==Operation::Outline{"Generate outline"}else{"Generate enhancement"}));control(ui,"story-generate",&r);
                if r.clicked()&&let Some(d)=&document{call=Some((Reply::Draft,true,format!("/v1/video/projects/{}/authoring",d.project.id.0),Some(json!(Request{schema_version:1,expected_project_revision:d.project.project_revision,provider_profile_id:state.provider_id.clone(),provider:state.provider.clone(),prompt_profile_id:state.profile.clone(),operation:state.operation,scene_id:self.timeline.selected.iter().next().cloned(),section_id:self.timeline.section.clone(),brief:state.brief.clone(),instruction:state.instruction.clone(),duration_seconds:state.duration,destination:state.destination,cast:if state.operation==Operation::Outline{state.cast.iter().cloned().collect()}else{vec![]},images:if state.operation==Operation::Outline{state.images.iter().enumerate().map(|(i,(asset,usage))|StoryImage{id:asset.0.clone(),asset:asset.clone(),label:format!("Source image {}",i+1),usage:*usage}).collect()}else{vec![]},add_dialogue:state.operation==Operation::Outline&&state.add_dialogue}))));}
            }
        });
        if new_proposal {
            state.draft = None;
        }
        state.open = open && !close_review;
        self.timeline.authoring = state;
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
