//! One contextual media selection path. Targets are captured before asynchronous import.
use super::*;
use imagine_video_contract::creative::{ImageBinding, ReferenceImage, Voice};

#[derive(Clone, Debug, PartialEq)]
pub(in crate::app) enum Target {
    Project,
    ReferenceImage(String),
    ReferenceVoice(String),
    Generation {
        scene: SceneId,
        section: Option<SectionId>,
        kind: MediaKind,
    },
    TimelineAudio,
    AudioClip(AudioClipId),
    StoryImage,
    SceneClip(SceneId),
    Take(SceneId),
    LibraryCharacter {
        editor: String,
        voice: bool,
    },
}
impl Target {
    fn kind(&self) -> Option<MediaKind> {
        match self {
            Self::LibraryCharacter { voice, .. } => Some(if *voice {
                MediaKind::Audio
            } else {
                MediaKind::Image
            }),
            Self::Project => None,
            Self::ReferenceImage(_) | Self::StoryImage => Some(MediaKind::Image),
            Self::ReferenceVoice(_) | Self::TimelineAudio | Self::AudioClip(_) => {
                Some(MediaKind::Audio)
            }
            Self::Generation { kind, .. } => Some(kind.clone()),
            Self::SceneClip(_) | Self::Take(_) => Some(MediaKind::Video),
        }
    }
    fn title(&self) -> &'static str {
        match self {
            Self::LibraryCharacter { voice, .. } => {
                if *voice {
                    "Library character voice audio"
                } else {
                    "Library character images"
                }
            }
            Self::Project => "Project media",
            Self::ReferenceImage(_) => "Reference images",
            Self::ReferenceVoice(_) => "Voice references",
            Self::Generation { .. } => "Generation media",
            Self::TimelineAudio | Self::AudioClip(_) => "Timeline audio",
            Self::StoryImage => "Story source images",
            Self::SceneClip(_) => "Imported scene clip",
            Self::Take(_) => "Video as Take",
        }
    }
}
#[derive(Clone, Copy, Default, PartialEq)]
enum SourceTab {
    #[default]
    Project,
    Library,
    Folders,
}
#[derive(Clone)]
pub(super) enum ImportSource {
    File(PathBuf),
    Library { library: String, entry: Value },
}
#[derive(Default)]
pub(super) struct State {
    open: bool,
    project: Option<ProjectId>,
    pub(super) target: Option<Target>,
    search: String,
    selected: BTreeSet<AssetId>,
    tab: SourceTab,
    library_id: Option<String>,
    library_assets: Vec<Value>,
    library_selected: BTreeSet<usize>,
    library_loading: bool,
    library_error: Option<String>,
    library_nonce: u64,
    import_error: Option<String>,
}
impl State {
    pub(super) fn finish_import(&mut self, target: &Target, error: Option<String>) {
        if self.target.as_ref() == Some(target) {
            self.import_error = error;
        }
    }
    pub(super) fn close(&mut self) {
        self.open = false;
    }
}
fn control(ui: &mut egui::Ui, name: &str, response: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), response);
    let _ = (ui, name, response);
}
fn matches(asset: &Asset, kind: Option<&MediaKind>, query: &str) -> bool {
    kind.is_none_or(|k| k == &asset.kind)
        && format!("{:?} {} {}", asset.kind, asset.name, asset.id.0)
            .to_lowercase()
            .contains(&query.to_lowercase())
}

/// Apply a selection once, preserving input order and not duplicating existing bindings.
fn bind(project: &VideoProject, target: &Target, assets: &[AssetId]) -> Result<Vec<Edit>, String> {
    for id in assets {
        let asset = project
            .assets
            .get(id)
            .ok_or("The selected media is no longer in this project.")?;
        if target
            .kind()
            .as_ref()
            .is_some_and(|kind| kind != &asset.kind)
        {
            return Err(format!(
                "Choose {:?} media for this field.",
                target.kind().unwrap()
            ));
        }
    }
    let mut creative = project.creative.clone();
    match target {
        Target::SceneClip(id) | Target::Take(id) => {
            project
                .scene(id)
                .map_err(|_| "The target scene was removed.")?;
            if assets.len() != 1 {
                return Err("Choose one video for this scene clip.".into());
            }
            Ok(vec![])
        }
        Target::ReferenceImage(id) | Target::ReferenceVoice(id) => {
            let reference = creative
                .references
                .iter_mut()
                .find(|r| &r.id == id)
                .ok_or("The target reference was removed. Choose a reference again.")?;
            for asset in assets {
                if matches!(target, Target::ReferenceImage(_)) {
                    if !reference.images.iter().any(|im| &im.asset == asset) {
                        reference.images.push(ReferenceImage {
                            id: format!("image_{}", uuid::Uuid::new_v4().simple()),
                            asset: asset.clone(),
                            enabled: true,
                            preview: reference.images.is_empty(),
                            contribution: String::new(),
                        });
                    }
                } else {
                    let voice = reference.voice.get_or_insert_with(|| Voice {
                        language: "English".into(),
                        description: String::new(),
                        assets: vec![],
                        reference_settings: Default::default(),
                    });
                    if !voice.assets.contains(asset) {
                        voice.assets.push(asset.clone());
                    }
                    voice.reference_settings.entry(asset.clone()).or_default();
                }
            }
            Ok(if creative == project.creative {
                vec![]
            } else {
                vec![Edit::SetCreative { creative }]
            })
        }
        Target::Generation {
            scene,
            section,
            kind,
        } => {
            let scene = project
                .scenes
                .iter()
                .find(|s| &s.id == scene)
                .ok_or("The target scene was removed. Select a scene again.")?;
            if let Some(section) = section {
                if !scene.sections.iter().any(|s| &s.id == section) {
                    return Err("The target Section was removed. Select a Section again.".into());
                }
                let source = creative.sections.entry(section.clone()).or_default();
                for asset in assets {
                    if *kind == MediaKind::Image {
                        if !source.images.iter().any(|im| &im.asset == asset) {
                            source.push_image(ImageBinding {
                                anchor: Some(Default::default()),
                                image_id: asset.0.clone(),
                                asset: asset.clone(),
                                role: InputRole::VisualReference,
                            });
                        }
                    } else if !source.videos.iter().any(|b| &b.asset == asset) {
                        source.push_video(binding(asset.clone(), InputRole::VideoReference));
                    }
                }
                Ok(if creative == project.creative {
                    vec![]
                } else {
                    vec![Edit::SetCreative { creative }]
                })
            } else {
                let mut bindings = scene.bindings.clone();
                for asset in assets {
                    if !bindings.iter().any(|b| &b.asset == asset) {
                        bindings.push(binding(
                            asset.clone(),
                            match kind {
                                MediaKind::Image => InputRole::VisualReference,
                                MediaKind::Video => InputRole::VideoReference,
                                MediaKind::Audio => InputRole::AudioReference,
                            },
                        ));
                    }
                }
                Ok(vec![Edit::SetBindings {
                    scene_id: scene.id.clone(),
                    bindings,
                }])
            }
        }
        Target::AudioClip(id) => {
            let mut clip = project
                .audio_tracks
                .iter()
                .flat_map(|t| &t.clips)
                .find(|c| &c.id == id)
                .cloned()
                .ok_or("The target audio clip was removed.")?;
            if let Some(asset) = assets.first() {
                clip.asset = Some(asset.clone());
                Ok(vec![Edit::UpdateAudioClip {
                    clip: Box::new(clip),
                }])
            } else {
                Ok(vec![])
            }
        }
        _ => Ok(vec![]),
    }
}
fn binding(asset: AssetId, role: InputRole) -> Binding {
    Binding {
        id: BindingId(format!("binding_{}", uuid::Uuid::new_v4().simple())),
        asset,
        role,
        enabled: true,
        video: None,
    }
}
impl ImagineApp {
    pub(super) fn timeline_pick_media(&mut self, target: Target) {
        if self.timeline.document.is_none() && !matches!(target, Target::LibraryCharacter { .. }) {
            return;
        }
        self.timeline.media_picker = State {
            open: true,
            library_nonce: self.timeline.media_picker.library_nonce.wrapping_add(1),
            project: self
                .timeline
                .document
                .as_ref()
                .map(|document| document.project.id.clone()),
            target: Some(target),
            ..Default::default()
        };
    }
    pub(super) fn timeline_media_picker_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.media_picker.open {
            return;
        }
        let document = self.timeline.document.clone();
        let mut state = std::mem::take(&mut self.timeline.media_picker);
        if state.project.as_ref() != document.as_ref().map(|document| &document.project.id) {
            return;
        }
        let Some(target) = state.target.clone() else {
            return;
        };
        let mut open = true;
        let mut use_selection = false;
        let mut import = false;
        let mut cancel = false;
        let mut choose_library = false;
        let mut refresh_library = false;
        egui::Window::new(format!("Choose {}", target.title()))
            .id(egui::Id::new("context-media-picker"))
            .open(&mut open)
            .default_width(560.)
            .max_height((ctx.content_rect().height() - 80.).max(180.))
            .show(ctx, |ui| {
                if !helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
                    ui.label("Hidden media · hover to reveal");
                    return;
                }
                ui.horizontal(|ui| {
                    for (tab,label,key) in [(SourceTab::Project,"Project assets","project"),(SourceTab::Library,"Library","library"),(SourceTab::Folders,"Folders","folders")] {
                        let r=ui.selectable_value(&mut state.tab,tab,label);control(ui,&format!("media-picker-tab-{key}"),&r);
                    }
                });
                ui.horizontal(|ui|{
                    let r = helto::action(ui, crate::icons::Icon::Import, "Import from files…", false, self.timeline.busy.then_some("Wait for the current timeline operation."));
                    control(ui, "media-picker-import", &r);
                    import = r.clicked();
                });
                let r =
                    ui.add(egui::TextEdit::singleline(&mut state.search).hint_text("Search media"));
                control(ui, "media-picker-search", &r);
                if state.tab==SourceTab::Library {
                    let selected_library=self.timeline.library.selected_id();
                    if selected_library!=state.library_id {state.library_id=selected_library;state.library_assets.clear();state.library_selected.clear();state.library_loading=false;refresh_library=state.library_id.is_some();}
                    ui.horizontal(|ui|{
                        let r=helto::icon_button(ui,crate::icons::Icon::Library,"Choose library…",false);control(ui,"media-picker-library-choose",&r);choose_library=r.clicked();
                        let r=helto::action(ui,crate::icons::Icon::Retry,"Refresh",false,if state.library_id.is_none(){Some("Choose a library first.")}else{state.library_loading.then_some("Loading.")});control(ui,"media-picker-library-refresh",&r);refresh_library|=r.clicked();
                    });
                    if state.library_id.is_none(){ui.weak("Choose an Imagine library.");}
                    if state.library_loading{ui.spinner();}
                    if let Some(error)=&state.library_error{ui.colored_label(helto::ERROR,error);}
                    egui::ScrollArea::vertical().id_salt("picker-library").max_height(350.).show(ui,|ui|{
                        for (i,entry) in state.library_assets.iter().enumerate(){
                            let Ok(kind)=serde_json::from_value::<MediaKind>(entry["kind"].clone()) else{continue;};
                            let name=entry["name"].as_str().unwrap_or("");
                            if target.kind().as_ref().is_some_and(|k|k!=&kind)||!name.to_lowercase().contains(&state.search.to_lowercase()){continue;}
                            if kind==MediaKind::Image {let library=state.library_id.as_deref().unwrap_or("");let resource=entry["resource_id"].as_str().unwrap_or("");let eid=entry["entry_id"].as_str().unwrap_or("");let rev=entry["entry_revision"].as_u64().unwrap_or(0);let asset=Asset{library_source:None,name:name.into(),id:AssetId(format!("library-{library}-{eid}-{rev}-{resource}")),revision:rev,kind:kind.clone()};let project=ProjectId(format!("library-{library}"));self.timeline_thumbnail_path(ui,&project,&asset,Some(format!("/v1/video/libraries/{library}/entries/{eid}/revisions/{rev}/media/{resource}")));}
                            let mut selected=state.library_selected.contains(&i);let r=ui.checkbox(&mut selected,name);control(ui,&format!("media-picker-library-asset-{i}"),&r);
                            if r.changed(){state.library_selected.remove(&i);if selected{if matches!(target,Target::AudioClip(_) | Target::Take(_)){state.library_selected.clear();}state.library_selected.insert(i);}}
                        }
                    });
                } else if state.tab==SourceTab::Folders {ui.weak("Import selected files into this field.");}
                if state.tab==SourceTab::Project && let Some(document)=&document {
                egui::ScrollArea::vertical()
                    .max_height(350.)
                    .show(ui, |ui| {
                        for (i, asset) in document
                            .project
                            .assets
                            .values()
                            .filter(|a| matches(a, target.kind().as_ref(), &state.search))
                            .enumerate()
                        {
                            self.timeline_thumbnail(ui, document, asset);
                            let mut selected = state.selected.contains(&asset.id);
                            let r = ui.checkbox(
                                &mut selected,
                                if asset.name.is_empty() {
                                    format!("{:?} · {}", asset.kind, asset.id.0)
                                } else {
                                    asset.name.clone()
                                },
                            );
                            control(ui, &format!("media-picker-asset-{i}"), &r);
                            control(ui, &format!("media-picker-id-{}", asset.id.0), &r);
                            if r.changed() {
                                state.selected.remove(&asset.id);
                                if selected {
                                    if matches!(target, Target::AudioClip(_) | Target::Take(_)) {
                                        state.selected.clear();
                                    }
                                    state.selected.insert(asset.id.clone());
                                }
                            }
                        }
                    });
                }
                if state.tab==SourceTab::Project && document.is_none() {ui.weak("No project is open. Choose media from Library or Folders.");}
                if let Some(error) = &state.import_error {
                    let response = ui.colored_label(helto::ERROR, error);
                    control(ui, "media-picker-error", &response);
                }
                ui.horizontal(|ui| {
                    let has_selection=match state.tab{SourceTab::Project=>!state.selected.is_empty(),SourceTab::Library=>!state.library_selected.is_empty(),SourceTab::Folders=>false};
                    let r = helto::action(
                        ui,
                        crate::icons::Icon::Check,
                        "Use selected",
                        true,
                        self.timeline.busy.then_some("Wait for the current timeline operation.").or((!has_selection).then_some("Select media first.")),
                    );
                    control(ui, "media-picker-use", &r);
                    use_selection = r.clicked();
                    let r = helto::icon_button(ui, crate::icons::Icon::Close, "Cancel", false);
                    control(ui, "media-picker-cancel", &r);
                    if r.clicked() {
                        cancel = true;
                    }
                });
            });
        state.open = open && !use_selection && !cancel;
        self.timeline.media_picker = state;
        if refresh_library {
            self.timeline_picker_library(ctx);
        }
        if choose_library {
            self.timeline.library.open = true;
        }
        if use_selection && self.timeline.media_picker.tab == SourceTab::Library {
            let state = &self.timeline.media_picker;
            let library = state.library_id.clone().unwrap_or_default();
            let sources = state
                .library_selected
                .iter()
                .filter_map(|i| state.library_assets.get(*i))
                .map(|entry| ImportSource::Library {
                    library: library.clone(),
                    entry: entry.clone(),
                })
                .collect();
            self.timeline_import_sources(ctx, sources, target);
        } else if use_selection {
            let selected = self
                .timeline
                .media_picker
                .selected
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            self.timeline_use_media(ctx, target, selected);
        } else if import {
            self.timeline_choose_context_media(ctx, target);
        }
    }
    pub(super) fn timeline_choose_context_media(&mut self, ctx: &egui::Context, target: Target) {
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        #[cfg(feature = "ui-preview")]
        let qa_directory = super::qa::picker_directory(self);
        std::thread::spawn(move || {
            let dialog = rfd::FileDialog::new().set_title(format!("Import {}", target.title()));
            #[cfg(feature = "ui-preview")]
            let dialog = super::qa::configure_picker(dialog, qa_directory);
            let dialog = match target.kind() {
                Some(MediaKind::Image) => {
                    dialog.add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                }
                Some(MediaKind::Audio) => {
                    dialog.add_filter("Audio", &["wav", "flac", "mp3", "ogg"])
                }
                Some(MediaKind::Video) => {
                    dialog.add_filter("Video", &["mp4", "mkv", "webm", "mov"])
                }
                None => dialog.add_filter(
                    "Image, video or audio",
                    &[
                        "png", "jpg", "jpeg", "webp", "mp4", "mkv", "webm", "mov", "wav", "flac",
                        "mp3", "ogg",
                    ],
                ),
            };
            let paths = if matches!(target, Target::Take(_)) {
                dialog.pick_file().into_iter().collect()
            } else {
                dialog.pick_files().unwrap_or_default()
            };
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Files {
                    epoch,
                    paths,
                    target,
                }),
            );
        });
    }
    pub(super) fn timeline_picker_library(&mut self, ctx: &egui::Context) {
        let (Some(client), Some(session), Some(id)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.media_picker.library_id.clone(),
        ) else {
            return;
        };
        self.timeline.media_picker.library_loading = true;
        self.timeline.media_picker.library_error = None;
        self.timeline.media_picker.library_nonce =
            self.timeline.media_picker.library_nonce.wrapping_add(1);
        let nonce = self.timeline.media_picker.library_nonce;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let epoch = self.timeline.epoch;
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                &format!("/v1/video/libraries/{id}/media"),
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::PickerLibrary {
                    epoch,
                    nonce,
                    id,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_picker_library_reply(
        &mut self,
        nonce: u64,
        id: String,
        result: Result<Value, client::TimelineFailure>,
    ) {
        let state = &mut self.timeline.media_picker;
        if state.library_nonce != nonce || state.library_id.as_ref() != Some(&id) {
            return;
        }
        state.library_loading = false;
        match result {
            Ok(value) => {
                state.library_assets = value["assets"].as_array().cloned().unwrap_or_default();
                state.library_selected.clear();
            }
            Err(e) => state.library_error = Some(e.message),
        }
    }
    pub(super) fn timeline_use_media(
        &mut self,
        ctx: &egui::Context,
        target: Target,
        assets: Vec<AssetId>,
    ) {
        if let Target::LibraryCharacter { editor, voice } = &target {
            let values = self
                .timeline
                .document
                .as_ref()
                .map(|document| {
                    assets
                        .iter()
                        .filter_map(|id| {
                            document.raw["runtime_assets"][&id.0]
                                .as_str()
                                .map(|id| AssetId(id.into()))
                        })
                        .collect()
                })
                .unwrap_or_default();
            self.library_character_media(editor, *voice, values);
            self.timeline.media_picker.open = false;
            return;
        }
        let Some(project) = self.timeline.document.as_ref().map(|d| d.project.clone()) else {
            return;
        };
        match bind(&project, &target, &assets) {
            Ok(commands) => {
                match target {
                    Target::SceneClip(scene) => {
                        self.timeline_clip_source(ctx, scene, assets[0].clone())
                    }
                    Target::Take(scene) => {
                        self.timeline.captures.open = true;
                        let next = PostAction::ImportTake {
                            scene,
                            asset: assets[0].clone(),
                        };
                        let edits = self.timeline.draft_edits();
                        if edits.is_empty() {
                            self.timeline_action(ctx, next);
                        } else {
                            self.timeline.after_edit = Some(next);
                            self.timeline_edit(ctx, edits, 0);
                        }
                    }
                    Target::StoryImage => {
                        for id in assets {
                            self.timeline.authoring.add_image(id);
                        }
                    }
                    Target::TimelineAudio => {
                        let commands = assets
                            .into_iter()
                            .map(|id| self.timeline.add_audio(&project, Some(id)))
                            .collect();
                        self.timeline_edit(ctx, commands, 0);
                    }
                    _ => self.timeline_edit(ctx, commands, 0),
                }
                self.timeline.media_picker.open = false;
            }
            Err(error) => {
                self.timeline
                    .media_picker
                    .finish_import(&target, Some(error.clone()));
                self.timeline.error = Some(error);
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    #[test]
    fn older_same_library_reply_cannot_replace_a_newer_picker_result() {
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(
            &ctx,
            PathBuf::from("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"),
        );
        app.client = None;
        let p = project();
        app.timeline.document = Some(Snapshot {
            project: p.clone(),
            raw: json!({"project":p}),
        });
        app.timeline.media_picker = State {
            open: true,
            library_id: Some("own-library".into()),
            library_nonce: 1,
            ..Default::default()
        };
        // A second contextual chooser starts while the first query is pending.
        app.timeline_pick_media(Target::ReferenceImage("cast".into()));
        app.timeline.media_picker.library_id = Some("own-library".into());
        let current = app.timeline.media_picker.library_nonce;
        app.timeline_picker_library_reply(
            current,
            "own-library".into(),
            Ok(json!({"assets":[{"entry_id":"new-own-result"}]})),
        );
        app.timeline.media_picker.library_selected.insert(0);
        app.timeline_picker_library_reply(
            1,
            "own-library".into(),
            Ok(json!({"assets":[{"entry_id":"old-own-result"}]})),
        );
        assert_eq!(
            app.timeline.media_picker.library_assets[0]["entry_id"],
            "new-own-result"
        );
        assert!(app.timeline.media_picker.library_selected.contains(&0));
    }
    pub(in crate::app::timeline) fn project() -> VideoProject {
        let mut p = VideoProject::create(CreateProject {
            generation: None,
            schema_version: 1,
            default_model: ModelSelection {
                adapter: "synthetic".into(),
                model: "own".into(),
                variant: "test".into(),
                profile: "test".into(),
                profile_revision: "1".into(),
            },
            delivery: DeliverySettings {
                width: 64,
                height: 64,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        for (id, kind) in [
            ("image", MediaKind::Image),
            ("voice", MediaKind::Audio),
            ("video", MediaKind::Video),
        ] {
            let id = AssetId(id.into());
            p.assets.insert(
                id.clone(),
                Asset {
                    library_source: None,
                    name: String::new(),
                    id,
                    revision: 1,
                    kind,
                },
            );
        }
        p.creative
            .references
            .push(imagine_video_contract::creative::VisualReference {
                id: "cast".into(),
                name: "Own keeper".into(),
                alias: "keeper".into(),
                kind: "Character".into(),
                description: String::new(),
                tags: vec![],
                enabled: true,
                strength: 1.,
                retention: String::new(),
                relationship: None,
                images: vec![],
                voice: None,
                origin: None,
            });
        p.scenes.push(Scene {
            id: SceneId("scene".into()),
            start: Time { num: 0, den: 1 },
            duration: Time { num: 2, den: 1 },
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![Section {
                id: SectionId("section".into()),
                start: Time { num: 0, den: 1 },
                duration: Time { num: 2, den: 1 },
                prompt: String::new(),
            }],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        p
    }
    fn apply(p: &mut VideoProject, commands: Vec<Edit>) {
        imagine_video_contract::editing::apply(
            p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: p.project_revision,
                commands,
            },
        )
        .unwrap();
    }
    #[test]
    fn reference_selection_is_idempotent_and_undoable_without_touching_other_creative_fields() {
        let mut p = project();
        let original = p.clone();
        let target = Target::ReferenceImage("cast".into());
        let commands = bind(
            &p,
            &target,
            &[AssetId("image".into()), AssetId("image".into())],
        )
        .unwrap();
        apply(&mut p, commands);
        assert_eq!(p.creative.references[0].images.len(), 1);
        assert!(p.creative.references[0].images[0].preview);
        assert!(p.creative.references[0].images[0].enabled);
        assert!(p.scenes[0].bindings.is_empty());
        let mut history = editing::History::default();
        history.record(original.clone(), p.clone());
        let undo = history.undo_commands(&p);
        apply(&mut p, undo);
        assert_eq!(p.creative, original.creative);
    }
    #[test]
    fn wrong_type_or_removed_reference_fails_before_any_binding_mutation() {
        let p = project();
        assert!(
            bind(
                &p,
                &Target::ReferenceImage("cast".into()),
                &[AssetId("voice".into())]
            )
            .is_err()
        );
        assert!(
            bind(
                &p,
                &Target::ReferenceImage("removed".into()),
                &[AssetId("image".into())]
            )
            .is_err()
        );
        assert!(p.creative.references[0].images.is_empty());
    }
    #[test]
    fn section_selection_stays_in_captured_section_and_does_not_become_a_scene_binding() {
        let mut p = project();
        let target = Target::Generation {
            scene: SceneId("scene".into()),
            section: Some(SectionId("section".into())),
            kind: MediaKind::Image,
        };
        let commands = bind(&p, &target, &[AssetId("image".into())]).unwrap();
        apply(&mut p, commands);
        assert_eq!(
            p.creative.sections[&SectionId("section".into())].images[0].asset,
            AssetId("image".into())
        );
        assert!(p.scenes[0].bindings.is_empty());
    }
    fn append_section_media(p: &mut VideoProject, kind: MediaKind) {
        let asset = AssetId(
            if kind == MediaKind::Image {
                "image"
            } else {
                "video"
            }
            .into(),
        );
        let commands = bind(
            p,
            &Target::Generation {
                scene: SceneId("scene".into()),
                section: Some(SectionId("section".into())),
                kind,
            },
            &[asset],
        )
        .unwrap();
        apply(p, commands);
    }
    #[test]
    fn mixed_media_label_uses_the_first_appended_binding_like_director() {
        for (kinds, expected) in [
            ([MediaKind::Video, MediaKind::Image], "Own video +1"),
            ([MediaKind::Image, MediaKind::Video], "Own image +1"),
        ] {
            let mut p = project();
            p.assets.get_mut(&AssetId("video".into())).unwrap().name = "Own video".into();
            p.assets.get_mut(&AssetId("image".into())).unwrap().name = "Own image".into();
            for kind in kinds {
                append_section_media(&mut p, kind);
            }
            assert_eq!(
                super::super::navigation::section_label(
                    &p,
                    &p.scenes[0].sections[0],
                    super::super::navigation::Display::Media
                ),
                expected
            );
            let restored: VideoProject =
                serde_json::from_slice(&serde_json::to_vec(&p).unwrap()).unwrap();
            assert_eq!(restored, p);
        }
    }
    #[test]
    fn cross_type_display_order_does_not_invalidate_existing_generation() {
        let mut p = project();
        append_section_media(&mut p, MediaKind::Video);
        append_section_media(&mut p, MediaKind::Image);
        let scene = p.scenes[0].id.clone();
        let section = p.scenes[0].sections[0].id.clone();
        let before = imagine_video_contract::planning::input_revision(&p, &scene).unwrap();
        p.creative
            .sections
            .get_mut(&section)
            .unwrap()
            .media_order
            .reverse();
        assert_eq!(
            imagine_video_contract::planning::input_revision(&p, &scene).unwrap(),
            before
        );
    }
    #[test]
    fn split_duplicate_and_undo_preserve_mixed_binding_order() {
        let mut p = project();
        p.creative.timeline_duration = Some(super::super::editing::time(12.));
        append_section_media(&mut p, MediaKind::Video);
        append_section_media(&mut p, MediaKind::Image);
        let section = SectionId("section".into());
        let expected = p.creative.sections[&section].media_order.clone();
        let commands =
            super::super::section_commands::split(&p, &section, super::super::editing::Snap::None);
        apply(&mut p, commands);
        assert_eq!(p.creative.sections.len(), 2);
        assert!(
            p.creative
                .sections
                .values()
                .all(|source| source.media_order == expected)
        );
        let before = p.clone();
        let duplicated = super::super::group_commands::duplicate(
            &p,
            std::slice::from_ref(&section),
            &BTreeSet::new(),
        );
        assert_eq!(duplicated.sections.len(), 1);
        let copied = duplicated.sections[0].clone();
        apply(&mut p, duplicated.edits);
        assert_eq!(p.creative.sections[&copied].media_order, expected);
        let undo = super::super::editing::difference(&p, &before);
        apply(&mut p, undo);
        assert_eq!(p.creative, before.creative);
        assert_eq!(p.scenes, before.scenes);
    }
    #[test]
    fn voice_selection_creates_original_language_default_and_preserves_multiple_samples() {
        let mut p = project();
        let commands = bind(
            &p,
            &Target::ReferenceVoice("cast".into()),
            &[AssetId("voice".into())],
        )
        .unwrap();
        apply(&mut p, commands);
        let voice = p.creative.references[0].voice.as_ref().unwrap();
        assert_eq!(voice.language, "English");
        assert_eq!(voice.assets.len(), 1);
        assert!(voice.reference_settings[&AssetId("voice".into())].enabled);
    }
    #[test]
    fn display_name_changes_do_not_mark_a_take_outdated() {
        let mut p = project();
        p.scenes[0]
            .bindings
            .push(binding(AssetId("image".into()), InputRole::VisualReference));
        let scene = SceneId("scene".into());
        let before = imagine_video_contract::planning::input_revision(&p, &scene).unwrap();
        p.assets.get_mut(&AssetId("image".into())).unwrap().name = "Own renamed fixture.png".into();
        assert_eq!(
            before,
            imagine_video_contract::planning::input_revision(&p, &scene).unwrap()
        );
    }
    #[test]
    fn contextual_filter_combines_kind_and_search_without_an_extension_allowlist() {
        let asset = Asset {
            library_source: None,
            name: String::new(),
            id: AssetId("own-sky".into()),
            revision: 1,
            kind: MediaKind::Image,
        };
        assert!(matches(&asset, Some(&MediaKind::Image), "SKY"));
        assert!(!matches(&asset, Some(&MediaKind::Audio), "sky"));
        assert!(!matches(&asset, None, "forest"));
    }
}
