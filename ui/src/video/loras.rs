//! H3's shared main-target stack and scene merge editor.
use super::*;
use imagine_video_contract::loras::{Merge, Override, Row, Stack};
use std::path::Path;
#[path = "lora_catalog.rs"]
mod catalog;
#[derive(Default)]
pub(super) struct State {
    draft: Option<Draft>,
    catalog: catalog::State,
}
struct Draft {
    rows: Vec<egui::Id>,
    project: ProjectId,
    scene: Option<SceneId>,
    global: Stack,
    local: Override,
    pending: bool,
    error: Option<String>,
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    #[cfg(test)]
    ui.ctx()
        .data_mut(|data| data.insert_temp(egui::Id::new(("lora-test-control", name)), r.clone()));
    let _ = (ui, name, r);
}
fn commands(
    project: &VideoProject,
    mut pending: Vec<Edit>,
    scene: Option<&SceneId>,
    global: &Stack,
    local: &Override,
) -> Vec<Edit> {
    let mut creative = pending
        .iter()
        .find_map(|edit| {
            if let Edit::SetCreative { creative } = edit {
                Some(creative.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| project.creative.clone());
    pending.retain(|edit| !matches!(edit, Edit::SetCreative { .. }));
    if let Some(id) = scene {
        creative.scene_loras.insert(id.clone(), local.clone());
    } else {
        creative.h3_loras = Some(global.clone());
    }
    pending.push(Edit::SetCreative { creative });
    pending
}
/// A file inside the LoRA folder as the catalog's relative name, which the
/// worker resolves under its own `loras` folder; any other file stays absolute.
fn relative_name(root: Option<&str>, path: &Path) -> String {
    let inside = |path: &Path, root: &Path| {
        path.strip_prefix(root)
            .ok()
            .filter(|r| !r.as_os_str().is_empty())
            .map(Path::to_path_buf)
    };
    let relative = root.map(Path::new).and_then(|root| {
        inside(path, root)
            .or_else(|| inside(&path.canonicalize().ok()?, &root.canonicalize().ok()?))
    });
    match relative {
        Some(relative) => relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
        None => path.to_string_lossy().into_owned(),
    }
}
impl ImagineApp {
    pub(super) fn timeline_open_loras(&mut self, scene: Option<SceneId>) {
        if let Some(d) = &self.timeline.document {
            let count = if let Some(id) = &scene {
                d.project
                    .creative
                    .scene_loras
                    .get(id)
                    .map_or(0, |s| s.main.loras.len())
            } else {
                d.project
                    .creative
                    .h3_loras
                    .as_ref()
                    .map_or(0, |s| s.loras.len())
            };
            self.timeline.loras.draft = Some(Draft {
                rows: (0..count)
                    .map(|_| egui::Id::new(uuid::Uuid::new_v4()))
                    .collect(),
                project: d.project.id.clone(),
                global: d.project.creative.h3_loras.clone().unwrap_or_default(),
                local: scene
                    .as_ref()
                    .and_then(|id| d.project.creative.scene_loras.get(id))
                    .cloned()
                    .unwrap_or_default(),
                scene,
                pending: false,
                error: None,
            });
        }
    }
    pub(super) fn timeline_loras_window(&mut self, ctx: &egui::Context) {
        let Some(mut draft) = self.timeline.loras.draft.take() else {
            return;
        };
        let Some(document) = self.timeline.document.clone().filter(|d| {
            d.project.id == draft.project
                && draft
                    .scene
                    .as_ref()
                    .is_none_or(|id| d.project.scene(id).is_ok())
        }) else {
            self.timeline.loras.catalog.bind(None);
            return;
        };
        if draft.pending && !self.timeline.busy {
            let saved = if let Some(id) = &draft.scene {
                document.project.creative.scene_loras.get(id) == Some(&draft.local)
            } else {
                document.project.creative.h3_loras.as_ref() == Some(&draft.global)
            };
            if saved {
                self.timeline.loras.catalog.close_info();
                self.timeline.loras.catalog.close_filter();
                return;
            }
            draft.pending = false;
            draft.error = Some(
                self.timeline
                    .error
                    .clone()
                    .unwrap_or_else(|| "LoRA settings were not saved. Retry Apply.".into()),
            );
        }
        self.timeline.loras.catalog.bind(
            self.timeline
                .session
                .clone()
                .map(|session| (session, draft.project.0.clone())),
        );
        self.timeline
            .loras
            .catalog
            .ensure(self.client.as_ref(), ctx);
        let root = self
            .timeline
            .loras
            .catalog
            .catalog
            .as_ref()
            .and_then(|c| c.root.clone());
        let mut inspect = None;
        let mut open = true;
        let mut apply = false;
        let mut cancel = false;
        let response = egui::Modal::new(egui::Id::new("h3-loras")).show(ctx,|ui| {
            ui.set_width((ctx.content_rect().width()-48.).clamp(240.,540.));
            ui.set_height((ctx.content_rect().height()-120.).max(260.));
            ui.horizontal(|ui| {
                ui.heading(if draft.scene.is_some(){"Scene LoRAs"}else{"Project LoRAs"});
                let r=helto::icon_button(ui,crate::icons::Icon::Close,"Close LoRAs",false);control(ui,"lora-close",&r);if r.clicked(){open=false;}
            });
            egui::ScrollArea::vertical().id_salt("h3-loras-scroll").auto_shrink([false,false]).max_height((ui.available_height()-44.).max(120.)).show(ui,|ui| {
                let r=ui.interact(ui.clip_rect(),ui.id().with("lora-scroll"),egui::Sense::hover());control(ui,"lora-scroll",&r);
            let hidden=!helto::reveal_in_viewport(ui,self.prefs.privacy_mode);
            helto::private_scope(ui,hidden,|ui| {
                ui.label("MiniMax H3 · main model target");
                if draft.scene.is_some() {
                    let r=ui.checkbox(&mut draft.local.enabled,"Override project LoRAs");control(ui,"lora-override",&r);
                    let r=egui::ComboBox::from_id_salt("lora-merge").selected_text(draft.local.merge_mode.label()).show_ui(ui,|ui| {
                        for mode in Merge::ALL { let r=ui.selectable_value(&mut draft.local.merge_mode,mode,mode.label());control(ui,&format!("lora-merge-{}",mode.label()),&r); }
                    });control(ui,"lora-merge",&r.response);
                    ui.weak(format!("Effective stack: {} active rows",imagine_video_contract::loras::resolve(&draft.global,Some(&draft.local)).active().len()));
                }
                let stack=if draft.scene.is_some(){&mut draft.local.main}else{&mut draft.global};
                ui.horizontal_wrapped(|ui| {
                    ui.label("Filter");
                    let r=ui.add(egui::TextEdit::singleline(&mut stack.ui.r#match).hint_text("Regex").desired_width((ui.available_width()-55.).max(70.)));
                    control(ui,"lora-filter",&r);
                    let r=ui.add_enabled_ui(!self.timeline.loras.catalog.loading,|ui|helto::icon_button(ui,crate::icons::Icon::Retry,"Refresh LoRA catalog",false)).inner;
                    control(ui,"lora-refresh",&r);
                    if r.clicked(){self.timeline.loras.catalog.refresh(self.client.as_ref(),ctx);}
                });
                if self.timeline.loras.catalog.loading {ui.weak("Loading LoRA catalog…");}
                if self.timeline.session.is_none() || self.client.is_none() {ui.weak("Connect to the server to load the LoRA catalog.");}
                if let Some(error)=&self.timeline.loras.catalog.error {ui.colored_label(helto::ERROR,error);}
                let available=self.timeline.loras.catalog.catalog.clone();
                let choices=available.as_ref().map(|c|self.timeline.loras.catalog.filtered(c,&stack.ui.r#match,ctx));
                if let Some(choices)=&choices {
                    if choices.invalid {ui.colored_label(helto::ERROR,"Invalid regex. Showing all files.");}
                    if choices.pending {ui.weak("Filtering LoRAs… Showing all files until the result is ready.");}
                    if let Some(error)=&choices.error {ui.colored_label(helto::ERROR,error);}
                    ui.weak(format!("{} files",choices.entries.len()));
                    if choices.entries.is_empty() {ui.weak(if available.as_ref().is_some_and(|c|c.files.is_empty()){"LoRA folder is empty. Type a file path below or refresh."}else{"No files match this filter. Type a file path below or change the filter."});}
                }
                if available.as_ref().is_some_and(|value|!value.warnings.is_empty()) {
                    ui.colored_label(helto::ERROR,"Some files or folders were skipped. Refresh to retry.");
                }
                let mut remove=None;let mut swap=None;
                for (index,row) in stack.loras.iter_mut().enumerate() {
                    ui.push_id(draft.rows[index],|ui| {
                        ui.separator();
                        ui.horizontal_wrapped(|ui| {
                            let r=ui.checkbox(&mut row.enabled,format!("LoRA {}",index+1));control(ui,&format!("lora-{index}-enabled"),&r);
                            ui.label("Model strength");let r=ui.add(egui::DragValue::new(&mut row.strength_model).speed(0.05));control(ui,&format!("lora-{index}-strength"),&r);
                            if r.changed(){ row.strength_clip=row.strength_model; }
                            for (icon,label,key) in [(crate::icons::Icon::Up,"Move LoRA up","up"),(crate::icons::Icon::Down,"Move LoRA down","down"),(crate::icons::Icon::Trash,"Remove LoRA","remove")] {
                                let r=helto::icon_button(ui,icon,label,false);control(ui,&format!("lora-{index}-{key}"),&r);
                                if r.clicked(){match key {"up" if index>0=>swap=Some((index,index-1)),"down"=>swap=Some((index,index+1)),"remove"=>remove=Some(index),_=>{}}}
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            let r=catalog::picker(ui,index,&mut row.name,choices.as_ref(),self.prefs.privacy_mode);
                            control(ui,&format!("lora-{index}-catalog"),&r);
                            let r=ui.add_enabled_ui(!row.name.trim().is_empty(),|ui|helto::icon_button(ui,crate::icons::Icon::Eye,"LoRA information",false)).inner;control(ui,&format!("lora-{index}-info"),&r);
                            if r.clicked(){inspect=Some(row.name.clone());}
                        });
                        let mut name=Some(row.name.clone());
                        if super::generation_ui::component_file(ui,"LoRA file · absolute or relative to models/loras",&format!("lora-{index}-file"),&mut name,root.as_deref().map(Path::new),&|path|relative_name(root.as_deref(),path)){row.name=name.unwrap_or_default();}
                    });
                }
                if let Some(index)=remove {stack.loras.remove(index);draft.rows.remove(index);}
                if let Some((a,b))=swap.filter(|(_,b)|*b<stack.loras.len()) {stack.loras.swap(a,b);draft.rows.swap(a,b);}
                let r=helto::icon_button(ui,crate::icons::Icon::Plus,"Add LoRA",false);control(ui,"lora-add",&r);if r.clicked(){stack.loras.push(Row::default());draft.rows.push(egui::Id::new(uuid::Uuid::new_v4()));}
                if let Some(error)=&draft.error {ui.colored_label(helto::ERROR,error);}
            });
            });
            ui.horizontal(|ui| {
                let r=helto::action(ui,crate::icons::Icon::Check,"Apply",true,self.timeline.busy.then_some("Wait for the current timeline operation."));control(ui,"lora-apply",&r);apply=r.clicked();
                let r=helto::icon_button(ui,crate::icons::Icon::Close,"Cancel",false);control(ui,"lora-cancel",&r);cancel=r.clicked();
            });
        });
        if response.should_close() {
            open = false;
        }
        if let Some(file) = inspect {
            self.timeline
                .loras
                .catalog
                .inspect(file, self.client.as_ref(), ctx);
        }
        self.timeline
            .loras
            .catalog
            .info_window(ctx, self.prefs.privacy_mode, self.client.as_ref());
        if apply {
            // Typed or pasted paths inside the LoRA folder become relative too.
            for row in draft
                .global
                .loras
                .iter_mut()
                .chain(&mut draft.local.main.loras)
            {
                if Path::new(&row.name).is_absolute() {
                    row.name = relative_name(root.as_deref(), Path::new(&row.name));
                }
            }
            let edits = commands(
                &document.project,
                self.timeline.draft_edits(),
                draft.scene.as_ref(),
                &draft.global,
                &draft.local,
            );
            self.timeline_edit(ctx, edits, 0);
            draft.pending = true;
            draft.error = None;
        }
        if open && !cancel {
            self.timeline.loras.draft = Some(draft);
        } else {
            self.timeline.loras.catalog.close_info();
            self.timeline.loras.catalog.close_filter();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picked_files_inside_the_lora_folder_become_relative_names() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("loras");
        std::fs::create_dir_all(root.join("h3/style")).unwrap();
        let file = root.join("h3/style/own.safetensors");
        std::fs::write(&file, b"").unwrap();
        let text = root.to_string_lossy().into_owned();
        assert_eq!(
            relative_name(Some(&text), &file),
            "h3/style/own.safetensors"
        );
        // Through a symlinked path to the same folder.
        let link = dir.path().join("linked");
        std::os::unix::fs::symlink(&root, &link).unwrap();
        assert_eq!(
            relative_name(Some(&text), &link.join("h3/style/own.safetensors")),
            "h3/style/own.safetensors"
        );
        // Outside the folder, or without a known folder, the path stays absolute.
        let outside = dir.path().join("own-outside.safetensors");
        std::fs::write(&outside, b"").unwrap();
        for root in [Some(text.as_str()), None] {
            assert_eq!(relative_name(root, &outside), outside.to_string_lossy());
        }
        assert_eq!(relative_name(Some(&text), &root), text);
    }
    #[test]
    fn lora_apply_keeps_pending_cast_and_section_edits_in_one_creative_command() {
        let project = super::super::media_picker::tests::project();
        let mut creative = project.creative.clone();
        creative.global_prompt = Some(imagine_video_contract::creative::GlobalPromptOptions {
            enabled: true,
            position: "Suffix".into(),
        });
        let global = Stack {
            loras: vec![Row {
                name: "/own/free alternative.weights".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let edits = commands(
            &project,
            vec![Edit::SetCreative {
                creative: creative.clone(),
            }],
            None,
            &global,
            &Override::default(),
        );
        assert_eq!(edits.len(), 1);
        let Edit::SetCreative { creative: actual } = &edits[0] else {
            panic!("expected one creative edit")
        };
        creative.h3_loras = Some(global);
        assert_eq!(*actual, creative);
    }
    #[test]
    fn catalog_failure_and_filter_never_change_manual_paths_or_disable_apply() {
        let project = super::super::media_picker::tests::project();
        let ctx = egui::Context::default();
        helto::install(&ctx);
        ctx.enable_accesskit();
        let mut app = ImagineApp::empty(
            std::path::PathBuf::from("/own/synthetic-fixture"),
            crate::settings::Connection {
                url: "http://127.0.0.1:1/imagine".into(),
            },
            serde_json::from_value(
                json!({"output_folder":"/own/synthetic-fixture","privacy_mode":false}),
            )
            .unwrap(),
        );
        app.timeline.document = Some(super::super::Snapshot::with(project, json!({})));
        app.timeline_open_loras(None);
        app.timeline.loras.catalog.filter_unavailable_for_test();
        let name = "/manual/free alternative.weights";
        let draft = app.timeline.loras.draft.as_mut().unwrap();
        draft.global.loras.push(Row {
            name: name.into(),
            ..Default::default()
        });
        draft.rows.push(egui::Id::new("synthetic-row"));
        draft.global.ui.r#match = "[".into();
        for width in [600., 960.] {
            for frame in 0..3 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 600.),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        app.timeline_loras_window(ui.ctx());
                    },
                );
                output.textures_delta.clear();
                let apply = ctx
                    .data_mut(|data| {
                        data.get_temp::<egui::Response>(egui::Id::new((
                            "lora-test-control",
                            "lora-apply",
                        )))
                    })
                    .unwrap();
                if frame == 2 {
                    assert!(
                        apply.enabled(),
                        "Apply must remain available with a manual path"
                    );
                }
            }
            assert_eq!(
                app.timeline.loras.draft.as_ref().unwrap().global.loras[0].name,
                name
            );
            assert!(!app.timeline.busy);
        }
    }
}
