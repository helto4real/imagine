//! Standalone library character drafts. Media keys are selected resources or uploads.
use super::*;
use imagine_video_contract::creative::{
    ReferenceImage, VisualReference, Voice, VoiceReferenceSettings,
};

pub(super) struct Editor {
    pub nonce: String,
    pub saving: bool,
    entry: Option<(String, u64)>,
    reference: VisualReference,
    tags: String,
}
impl Editor {
    pub fn new(entry: Option<&Value>) -> Self {
        let doc = entry.map(|entry| &entry["content"]["document"]);
        let mut reference = doc
            .and_then(|doc| {
                serde_json::from_value::<VisualReference>(doc["reference"].clone()).ok()
            })
            .unwrap_or_else(|| VisualReference {
                id: uuid::Uuid::new_v4().to_string(),
                name: "New character".into(),
                alias: "character".into(),
                kind: "character".into(),
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
        if let Some(doc) = doc {
            reference.name = doc["name"].as_str().unwrap_or("").into();
            reference.alias = doc["alias"].as_str().unwrap_or("").into();
            reference.description = doc["description"].as_str().unwrap_or("").into();
            reference.tags = serde_json::from_value(doc["tags"].clone()).unwrap_or_default();
            if doc["reference"].is_null() {
                for (i, key) in doc["visual_ids"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    reference.images.push(ReferenceImage {
                        id: uuid::Uuid::new_v4().to_string(),
                        asset: AssetId(key.as_str().unwrap_or("").into()),
                        enabled: true,
                        preview: i == 0,
                        contribution: String::new(),
                    });
                }
                if doc["voice"].is_object() {
                    reference.voice = Some(Voice {
                        language: doc["voice"]["language"].as_str().unwrap_or("").into(),
                        description: doc["voice"]["description"].as_str().unwrap_or("").into(),
                        assets: serde_json::from_value(doc["voice"]["reference_ids"].clone())
                            .unwrap_or_default(),
                        reference_settings: Default::default(),
                    });
                }
            }
        }
        let tags = reference.tags.join(", ");
        Self {
            nonce: uuid::Uuid::new_v4().to_string(),
            saving: false,
            entry: entry.map(|e| {
                (
                    e["id"].as_str().unwrap_or("").into(),
                    e["revision"].as_u64().unwrap_or(0),
                )
            }),
            reference,
            tags,
        }
    }
    pub fn add(&mut self, voice: bool, assets: Vec<AssetId>) {
        for asset in assets {
            if voice {
                let profile = self.reference.voice.get_or_insert_with(|| Voice {
                    language: "English".into(),
                    description: String::new(),
                    assets: vec![],
                    reference_settings: Default::default(),
                });
                if !profile.assets.contains(&asset) {
                    profile.assets.push(asset);
                }
            } else if !self
                .reference
                .images
                .iter()
                .any(|image| image.asset == asset)
            {
                self.reference.images.push(ReferenceImage {
                    id: uuid::Uuid::new_v4().to_string(),
                    asset,
                    enabled: true,
                    preview: self.reference.images.is_empty(),
                    contribution: String::new(),
                });
            }
        }
    }
}
fn control(ui: &mut egui::Ui, name: &str, response: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), response);
    let _ = (ui, name, response);
}
/// A text field; names and descriptive text are private, kinds, IDs and
/// languages are not.
fn field(
    ui: &mut egui::Ui,
    hidden: bool,
    label: &str,
    key: &str,
    value: &mut String,
    multiline: bool,
) {
    let private = !matches!(
        label,
        "Reference kind" | "Marker" | "Target reference ID" | "Language"
    );
    let label = ui.label(label);
    let edit = if multiline {
        egui::TextEdit::multiline(value).desired_rows(2)
    } else {
        egui::TextEdit::singleline(value)
    };
    let r = helto::private_scope(ui, hidden && private, |ui| {
        ui.add(edit.desired_width(f32::INFINITY))
    })
    .labelled_by(label.id);
    control(ui, key, &r);
}
impl ImagineApp {
    fn library_character_preview(&mut self, ui: &mut egui::Ui, editor: &Editor, asset: &AssetId) {
        let Some(catalog) = &self.timeline.library.catalog else {
            return;
        };
        let library = catalog["id"].as_str().unwrap_or("");
        let path=editor.entry.as_ref().map_or_else(||format!("/v1/video/assets/{}",asset.0), |(entry,revision)| {
            let existing=catalog["entries"].as_array().and_then(|rows|rows.iter().find(|row|row["id"]==*entry)).and_then(|row|row["content"]["document"]["resources"].as_array()).is_some_and(|rows|rows.iter().any(|row|row["id"]==asset.0));
            if existing {format!("/v1/video/libraries/{library}/entries/{entry}/revisions/{revision}/media/{}",asset.0)}else{format!("/v1/video/assets/{}",asset.0)}
        });
        let project = ProjectId(format!("library-{library}"));
        let image = Asset {
            id: AssetId(format!("editor-{}-{}", editor.nonce, asset.0)),
            kind: MediaKind::Image,
            revision: 1,
            name: String::new(),
            library_source: None,
        };
        self.timeline_thumbnail_path(ui, &project, &image, Some(path));
    }
    pub(super) fn timeline_character_window(&mut self, ctx: &egui::Context) {
        let Some(mut editor) = self.timeline.library.editor.take() else {
            return;
        };
        let mut open = true;
        let mut save = false;
        let mut cancel = false;
        let mut pick = None;
        egui::Window::new("Library character")
            .id(egui::Id::new("library-character-editor"))
            .open(&mut open)
            .default_width(680.)
            .default_height(650.)
            .max_height((ctx.content_rect().height() - 70.).max(200.))
            .vscroll(true)
            .show(ctx, |ui| {
                let _canvas = helto::window_canvas(ui);
                #[cfg(feature = "ui-preview")]
                {
                    let r =
                        ui.interact(ui.clip_rect(), ui.id().with("scroll"), egui::Sense::hover());
                    control(ui, "library-character-scroll", &r);
                }
                let hidden = !helto::shown(ui, self.prefs.privacy_mode);
                ui.scope(|ui| {
                    if self.timeline.busy {
                        ui.disable();
                    }
                    ui.horizontal(|ui| {
                        let r = helto::icon_button(
                            ui,
                            crate::icons::Icon::Save,
                            "Save character",
                            true,
                        );
                        control(ui, "library-character-save", &r);
                        save = r.clicked();
                        let r = helto::icon_button(ui, crate::icons::Icon::Close, "Cancel", false);
                        control(ui, "library-character-cancel", &r);
                        cancel = r.clicked();
                    });
                    if let Some(error) = &self.timeline.library.error {
                        ui.colored_label(helto::ERROR, error);
                    }
                    field(
                        ui,
                        hidden,
                        "Name",
                        "library-character-name",
                        &mut editor.reference.name,
                        false,
                    );
                    field(
                        ui,
                        hidden,
                        "Alias",
                        "library-character-alias",
                        &mut editor.reference.alias,
                        false,
                    );
                    field(
                        ui,
                        hidden,
                        "Description",
                        "library-character-description",
                        &mut editor.reference.description,
                        true,
                    );
                    field(
                        ui,
                        hidden,
                        "Tags (comma separated)",
                        "library-character-tags",
                        &mut editor.tags,
                        false,
                    );
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut editor.reference.enabled, "Enabled");
                        ui.label("Strength");
                        let r = ui
                            .add(egui::DragValue::new(&mut editor.reference.strength).speed(0.05));
                        control(ui, "library-character-strength", &r);
                    });
                    field(
                        ui,
                        hidden,
                        "Reference kind",
                        "library-character-kind",
                        &mut editor.reference.kind,
                        false,
                    );
                    field(
                        ui,
                        hidden,
                        "Retention",
                        "library-character-retention",
                        &mut editor.reference.retention,
                        false,
                    );
                    ui.collapsing("Relationship", |ui| {
                        let mut enabled = editor.reference.relationship.is_some();
                        let checkbox = ui.checkbox(&mut enabled, "Use relationship");
                        control(ui, "library-character-use-relationship", &checkbox);
                        if checkbox.changed() {
                            editor.reference.relationship = enabled.then(Default::default);
                        }
                        let Some(r) = &mut editor.reference.relationship else {
                            return;
                        };
                        field(
                            ui,
                            hidden,
                            "Marker",
                            "library-character-relationship",
                            &mut r.marker,
                            false,
                        );
                        field(
                            ui,
                            hidden,
                            "Note",
                            "library-character-relationship-note",
                            &mut r.note,
                            true,
                        );
                        let mut target = r.target_reference_id.clone().unwrap_or_default();
                        field(
                            ui,
                            hidden,
                            "Target reference ID",
                            "library-character-target",
                            &mut target,
                            false,
                        );
                        r.target_reference_id = (!target.is_empty()).then_some(target);
                    });
                    ui.separator();
                    ui.strong("Images");
                    let r =
                        helto::icon_button(ui, crate::icons::Icon::Import, "Add images…", false);
                    control(ui, "library-character-images", &r);
                    if r.clicked() {
                        pick = Some(false);
                    }
                    let mut remove = None;
                    let mut preview = None;
                    for i in 0..editor.reference.images.len() {
                        let asset = editor.reference.images[i].asset.clone();
                        self.library_character_preview(ui, &editor, &asset);
                        let image = &mut editor.reference.images[i];
                        ui.horizontal(|ui| {
                            ui.label(format!("Image {}", i + 1));
                            let r = ui.checkbox(&mut image.enabled, "Enabled");
                            control(ui, &format!("library-character-image-{i}-enabled"), &r);
                            let r = ui.selectable_label(image.preview, "Default preview");
                            control(ui, &format!("library-character-image-{i}-preview"), &r);
                            if r.clicked() {
                                preview = Some(i);
                            }
                            let r = helto::icon_button(
                                ui,
                                crate::icons::Icon::Trash,
                                "Remove image",
                                false,
                            );
                            control(ui, &format!("library-character-image-{i}-remove"), &r);
                            if r.clicked() {
                                remove = Some(i);
                            }
                        });
                        field(
                            ui,
                            hidden,
                            "Contribution",
                            &format!("library-character-image-{i}-contribution"),
                            &mut image.contribution,
                            false,
                        );
                    }
                    if let Some(i) = preview {
                        for (n, image) in editor.reference.images.iter_mut().enumerate() {
                            image.preview = n == i;
                        }
                    }
                    if let Some(i) = remove {
                        let was_preview = editor.reference.images[i].preview;
                        editor.reference.images.remove(i);
                        if was_preview && let Some(first) = editor.reference.images.first_mut() {
                            first.preview = true;
                        }
                    }
                    ui.separator();
                    let mut voice = editor.reference.voice.is_some();
                    let r = ui.checkbox(&mut voice, "Voice profile");
                    control(ui, "library-character-voice", &r);
                    if r.changed() {
                        editor.reference.voice = voice.then(|| Voice {
                            language: "English".into(),
                            description: String::new(),
                            assets: vec![],
                            reference_settings: Default::default(),
                        });
                    }
                    if let Some(voice) = &mut editor.reference.voice {
                        field(
                            ui,
                            hidden,
                            "Language",
                            "library-character-language",
                            &mut voice.language,
                            false,
                        );
                        field(
                            ui,
                            hidden,
                            "Voice guidance",
                            "library-character-voice-guidance",
                            &mut voice.description,
                            true,
                        );
                        let r = helto::icon_button(
                            ui,
                            crate::icons::Icon::Import,
                            "Add voice audio…",
                            false,
                        );
                        control(ui, "library-character-audio", &r);
                        if r.clicked() {
                            pick = Some(true);
                        }
                        let mut remove = None;
                        for (i, asset) in voice.assets.iter().enumerate() {
                            let settings = voice
                                .reference_settings
                                .entry(asset.clone())
                                .or_insert_with(VoiceReferenceSettings::default);
                            ui.horizontal(|ui| {
                                ui.label(format!("Voice {}", i + 1));
                                let r = ui.checkbox(&mut settings.enabled, "Enabled");
                                control(ui, &format!("library-character-audio-{i}-enabled"), &r);
                                let r = helto::icon_button(
                                    ui,
                                    crate::icons::Icon::Trash,
                                    "Remove voice audio",
                                    false,
                                );
                                control(ui, &format!("library-character-audio-{i}-remove"), &r);
                                if r.clicked() {
                                    remove = Some(i);
                                }
                            });
                            field(
                                ui,
                                hidden,
                                "Voice description",
                                &format!("library-character-audio-{i}-description"),
                                &mut settings.voice_description,
                                true,
                            );
                            field(
                                ui,
                                hidden,
                                "Contribution",
                                &format!("library-character-audio-{i}-contribution"),
                                &mut settings.contribution,
                                false,
                            );
                            field(
                                ui,
                                hidden,
                                "Relationship",
                                &format!("library-character-audio-{i}-relationship"),
                                &mut settings.relationship,
                                false,
                            );
                            let mut strength = settings.strength.is_some();
                            let r = ui.checkbox(&mut strength, "Override voice strength");
                            control(ui, &format!("library-character-audio-{i}-override"), &r);
                            if r.changed() {
                                settings.strength = strength.then_some(1.);
                            }
                            if let Some(strength) = &mut settings.strength {
                                let r = ui.add(egui::DragValue::new(strength).speed(0.05));
                                control(ui, &format!("library-character-audio-{i}-strength"), &r);
                            }
                        }
                        if let Some(i) = remove {
                            let asset = voice.assets.remove(i);
                            voice.reference_settings.remove(&asset);
                        }
                    }
                });
            });
        if open && !cancel {
            editor.reference.tags = editor
                .tags
                .split(',')
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(str::to_owned)
                .collect();
            let nonce = editor.nonce.clone();
            let body=save.then(||json!({"schema_version":1,"expected_library_revision":self.timeline.library.catalog.as_ref().map(|c|c["revision"].clone()),"title":editor.reference.name,"replace":editor.entry.as_ref().map(|(id,revision)|json!({"entry_id":id,"expected_entry_revision":revision})),"character":{"name":editor.reference.name,"alias":editor.reference.alias,"description":editor.reference.description,"tags":editor.reference.tags,"visual_asset_ids":editor.reference.images.iter().map(|image|image.asset.0.clone()).collect::<Vec<_>>(),"voice":editor.reference.voice.as_ref().map(|v|json!({"language":v.language,"description":v.description,"reference_asset_ids":v.assets.iter().map(|asset|asset.0.clone()).collect::<Vec<_>>()})),"reference":editor.reference}}));
            editor.saving |= save;
            self.timeline.library.editor = Some(editor);
            if let Some(body) = body
                && let Some(catalog) = &self.timeline.library.catalog
            {
                let route = format!(
                    "/v1/video/libraries/{}/characters",
                    catalog["id"].as_str().unwrap_or("")
                );
                self.library_call(ctx, library::Reply::Catalog, true, route, Some(body));
            }
            if let Some(voice) = pick {
                let target = media_picker::Target::LibraryCharacter {
                    editor: nonce,
                    voice,
                };
                self.timeline_pick_media(target);
            }
        }
    }
    pub(super) fn library_character_media(
        &mut self,
        nonce: &str,
        voice: bool,
        values: Vec<AssetId>,
    ) {
        if let Some(editor) = &mut self.timeline.library.editor
            && editor.nonce == nonce
        {
            editor.add(voice, values);
        }
    }
    pub(super) fn library_character_import(
        &mut self,
        ctx: &egui::Context,
        sources: Vec<media_picker::ImportSource>,
        editor: String,
        voice: bool,
    ) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        self.timeline.media_picker.close();
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result=runtime().block_on(async {
                let mut assets=vec![];
                for source in sources {
                    let value=match source {
                        media_picker::ImportSource::File(path)=>client.upload_media(&session,&path).await?,
                        media_picker::ImportSource::Library{library,entry}=>client.timeline_request(&session,true,&format!("/v1/video/libraries/{library}/media-copy"),Some(&json!({"schema_version":1,"entry_id":entry["entry_id"],"expected_entry_revision":entry["entry_revision"],"resource_id":entry["resource_id"]}))).await?,
                    };
                    let mime=value["mime_type"].as_str().unwrap_or("");
                    if mime.starts_with(if voice {"audio/"}else{"image/"}) {
                        if let Some(id)=value["id"].as_str(){assets.push(AssetId(id.into()));}
                    } else {return Err(client::TimelineFailure { code:"character_media_type_mismatch".into(), message:"The selected media does not match this image or voice field.".into(), retryable:false, snapshot:None });}
                }
                Ok(assets)
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::LibraryImported {
                    epoch,
                    editor,
                    voice,
                    result,
                }),
            );
        });
    }
}
