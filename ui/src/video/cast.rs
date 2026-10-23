//! Explicit project cast editing; media remains project-owned, previews are independent.
use super::*;
use imagine_video_contract::creative::*;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    draft: Creative,
    base: Option<(ProjectId, Vec<VisualReference>, Option<bool>)>,
    error: Option<String>,
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl ImagineApp {
    pub(super) fn timeline_open_cast(&mut self) {
        if let Some(d) = &self.timeline.document {
            self.timeline.cast = State {
                open: true,
                draft: d.project.creative.clone(),
                base: Some((
                    d.project.id.clone(),
                    d.project.creative.references.clone(),
                    d.project.creative.visual_references_enabled,
                )),
                error: None,
            };
        }
    }
    pub(super) fn timeline_cast_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.cast.open {
            return;
        }
        let Some(document) = self.timeline.document.clone() else {
            return;
        };
        let mut state = std::mem::take(&mut self.timeline.cast);
        let mut open = true;
        let mut apply = false;
        let mut reload = false;
        egui::Window::new("Cast & Visual References").id(egui::Id::new("video-cast")).open(&mut open).default_width(650.).max_height((ctx.content_rect().height()-70.).max(200.)).vscroll(true).show(ctx,|ui|{
            if !helto::reveal_in_viewport(ui,self.prefs.privacy_mode){ui.label("Hidden references · hover to reveal");return;}
            ui.weak("Project copies · text-only references are valid · preview selection does not enable generation");
            let mut enabled=state.draft.visual_references_enabled.unwrap_or(true);
            let r=ui.checkbox(&mut enabled,"Use visual references");control(ui,"cast-use-references",&r);
            if r.changed(){state.draft.visual_references_enabled=Some(enabled);}
            ui.horizontal(|ui|{
                let r=ui.button("Add reference");control(ui,"cast-add",&r);
                if r.clicked(){let n=state.draft.references.len()+1;state.draft.references.push(VisualReference{id:format!("reference_{}",uuid::Uuid::new_v4().simple()),name:format!("Character {n}"),alias:format!("character_{n}"),kind:"Character".into(),description:String::new(),tags:vec![],enabled:true,strength:1.,retention:String::new(),relationship:None,images:vec![],voice:None,origin:None});}
                let r=ui.button("Reload current project");control(ui,"cast-reload",&r);reload=r.clicked();
            });
            let mut remove=None;
            for (i,reference) in state.draft.references.iter_mut().enumerate(){
                ui.push_id(&reference.id,|ui|{ui.group(|ui|{
                    ui.horizontal(|ui|{let r=ui.checkbox(&mut reference.enabled,"Enabled");control(ui,&format!("cast-enabled-{i}"),&r);ui.strong(&reference.name);if ui.small_button("Remove").clicked(){remove=Some(i);}});
                    ui.horizontal(|ui|{ui.label("Name");let r=ui.text_edit_singleline(&mut reference.name);control(ui,&format!("cast-name-{i}"),&r);});
                    ui.horizontal(|ui|{ui.label("Alias @");let r=ui.text_edit_singleline(&mut reference.alias);control(ui,&format!("cast-alias-{i}"),&r);ui.label("Kind");ui.text_edit_singleline(&mut reference.kind);});
                    ui.label("Description");let r=ui.add(egui::TextEdit::multiline(&mut reference.description).desired_rows(2).desired_width(f32::INFINITY));control(ui,&format!("cast-description-{i}"),&r);
                    ui.horizontal(|ui|{ui.label("Tags");let mut text=reference.tags.join(", ");if ui.text_edit_singleline(&mut text).changed(){reference.tags=text.split(',').map(str::trim).filter(|s|!s.is_empty()).map(str::to_owned).collect();}});
                    ui.horizontal(|ui|{ui.label("Strength");ui.add(egui::DragValue::new(&mut reference.strength).speed(0.05));ui.label("Retention");ui.text_edit_singleline(&mut reference.retention);});
                    let mut relationship=reference.relationship.is_some();
                    let r=ui.checkbox(&mut relationship,"Reference relationship");control(ui,&format!("cast-relationship-{i}"),&r);
                    if r.changed(){reference.relationship=relationship.then(||ReferenceRelationship{marker:if reference.retention.is_empty(){"fully_preserved".into()}else{reference.retention.clone()},..Default::default()});}
                    if let Some(value)=&mut reference.relationship {
                        ui.horizontal(|ui|{ui.label("Relationship");let r=ui.text_edit_singleline(&mut value.marker);control(ui,&format!("cast-relationship-marker-{i}"),&r);});
                        ui.horizontal(|ui|{ui.label("Target reference ID");let mut target=value.target_reference_id.clone().unwrap_or_default();let r=ui.text_edit_singleline(&mut target);control(ui,&format!("cast-relationship-target-{i}"),&r);if r.changed(){value.target_reference_id=(!target.is_empty()).then_some(target);}});
                        ui.label("Relationship note");let r=ui.text_edit_multiline(&mut value.note);control(ui,&format!("cast-relationship-note-{i}"),&r);
                    }
                    let images=ui.collapsing("Images",|ui|{for(ordinal,asset)in document.project.assets.values().filter(|a|a.kind==MediaKind::Image).enumerate(){
                        let mut included=reference.images.iter().any(|im|im.asset==asset.id);let r=ui.checkbox(&mut included,format!("Image {}",ordinal+1));control(ui,&format!("cast-image-{i}-{ordinal}"),&r);
                        if r.changed(){if included {reference.images.push(ReferenceImage{id:format!("image_{}",uuid::Uuid::new_v4().simple()),asset:asset.id.clone(),enabled:true,preview:false,contribution:String::new()});}else{reference.images.retain(|im|im.asset!=asset.id);}}
                        if let Some(image)=reference.images.iter_mut().find(|im|im.asset==asset.id){ui.horizontal(|ui|{ui.checkbox(&mut image.enabled,"Use for generation");ui.checkbox(&mut image.preview,"Preview");});ui.label("Image contribution");let r=ui.text_edit_singleline(&mut image.contribution);control(ui,&format!("cast-image-contribution-{i}-{ordinal}"),&r);}
                    }});control(ui,&format!("cast-images-{i}"),&images.header_response);
                    let mut voice=reference.voice.is_some();if ui.checkbox(&mut voice,"Voice profile").changed(){reference.voice=voice.then(||Voice{language:String::new(),description:String::new(),assets:vec![],reference_settings:Default::default()});}
                    if let Some(voice)=&mut reference.voice {ui.horizontal(|ui|{ui.label("Language");ui.text_edit_singleline(&mut voice.language);});ui.label("Voice guidance");ui.text_edit_multiline(&mut voice.description);for (ordinal,asset)in document.project.assets.values().filter(|a|a.kind==MediaKind::Audio).enumerate(){let mut selected=voice.assets.contains(&asset.id);let r=ui.checkbox(&mut selected,format!("Voice audio {}",ordinal+1));control(ui,&format!("cast-voice-audio-{i}-{ordinal}"),&r);if r.changed(){voice.assets.retain(|id|id!=&asset.id);if selected {voice.assets.push(asset.id.clone());}}
                        if selected {let settings=voice.reference_settings.entry(asset.id.clone()).or_default();
                            let r=ui.checkbox(&mut settings.enabled,"Use voice reference");control(ui,&format!("cast-voice-enabled-{i}-{ordinal}"),&r);
                            ui.label("Voice description");let r=ui.text_edit_singleline(&mut settings.voice_description);control(ui,&format!("cast-voice-description-{i}-{ordinal}"),&r);
                            ui.label("Voice contribution");let r=ui.text_edit_singleline(&mut settings.contribution);control(ui,&format!("cast-voice-contribution-{i}-{ordinal}"),&r);
                            ui.horizontal(|ui|{ui.label("Audio relationship");ui.text_edit_singleline(&mut settings.relationship);});
                            let mut strength=settings.strength.is_some();if ui.checkbox(&mut strength,"Explicit reference strength").changed(){settings.strength=strength.then_some(1.);}
                            if let Some(value)=&mut settings.strength {ui.add(egui::DragValue::new(value).speed(0.05));}
                        }
                    }}
                });});
            }
            if let Some(i)=remove{state.draft.references.remove(i);}
            if let Some(error)=&state.error{ui.colored_label(helto::ERROR,error);}
            let r=ui.add_enabled(!self.timeline.busy,egui::Button::new("Apply project references"));control(ui,"cast-apply",&r);apply=r.clicked();
        });
        state.open = open;
        if reload {
            state.draft = document.project.creative.clone();
            state.base = Some((
                document.project.id.clone(),
                document.project.creative.references.clone(),
                document.project.creative.visual_references_enabled,
            ));
            state.error = None;
        }
        if apply {
            if state.base.as_ref()
                != Some(&(
                    document.project.id.clone(),
                    document.project.creative.references.clone(),
                    document.project.creative.visual_references_enabled,
                ))
            {
                state.error = Some(
                    "Project references changed. Reload them before applying this edit.".into(),
                );
            } else {
                let mut creative = document.project.creative.clone();
                creative.references = state.draft.references.clone();
                creative.visual_references_enabled = state.draft.visual_references_enabled;
                state.open = false;
                self.timeline_edit(ctx, vec![Edit::SetCreative { creative }], 0);
            }
        }
        self.timeline.cast = state;
    }
}
