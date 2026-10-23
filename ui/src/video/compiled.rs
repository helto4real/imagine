//! Source compilation review, independent of model loading and queue admission.
use super::*;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    pub value: Option<Value>,
    pub error: Option<String>,
    review: Option<Review>,
}

struct Review {
    project: VideoProject,
    scene: SceneId,
    input_revision: String,
    seed: u64,
    advance: Option<seed::Advance>,
}
impl Review {
    fn capture(project: &VideoProject, scene: &SceneId, random: u64) -> Result<Self, String> {
        let input_revision = imagine_video_contract::planning::input_revision(project, scene)
            .map_err(|_| "Could not prepare the selected scene.".to_owned())?;
        let (seed, advance) = seed::prepare(project, random);
        Ok(Self {
            project: project.clone(),
            scene: scene.clone(),
            input_revision,
            seed,
            advance,
        })
    }
    fn matches(&self, project: &VideoProject) -> bool {
        let seed_options = |p: &VideoProject| {
            p.generation
                .as_ref()
                .map(|g| (g.seed, g.random_seed, g.control_after_generate))
        };
        project.id == self.project.id
            && seed_options(project) == seed_options(&self.project)
            && project.scene_kind(&self.scene) == self.project.scene_kind(&self.scene)
            && imagine_video_contract::planning::input_revision(project, &self.scene)
                .ok()
                .as_ref()
                == Some(&self.input_revision)
    }
    fn request(&self, project: &VideoProject) -> Option<Value> {
        self.matches(project).then(|| {
            json!({
                "kind":"video_scene", "schema_version":1, "project_id":project.id,
                "expected_project_revision":project.project_revision,
                "scene_id":self.scene, "seed":self.seed,
            })
        })
    }
}
impl State {
    pub(super) fn stale(&self, project: Option<&VideoProject>, pending_edits: bool) -> bool {
        pending_edits
            || self
                .review
                .as_ref()
                .is_none_or(|r| project.is_none_or(|p| !r.matches(p)))
    }
    #[cfg(feature = "ui-preview")]
    pub(super) fn reviewed_seed(&self) -> Option<u64> {
        self.review.as_ref().map(|r| r.seed)
    }
    pub(super) fn receive(&mut self, value: Value) {
        if self.review.as_ref().is_some_and(|r| {
            value["project_id"] == json!(r.project.id)
                && value["scene_id"] == json!(r.scene)
                && value["project_revision"] == json!(r.project.project_revision)
                && value["input_revision"] == r.input_revision
        }) {
            self.value = Some(value);
            self.error = None;
        } else {
            self.value = None;
            self.error = Some(
                "Inputs changed while preparing this plan. Refresh the project and review again."
                    .into(),
            );
        }
    }
}
fn control(ui: &mut egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl ImagineApp {
    pub(super) fn timeline_compiled(&mut self, ctx: &egui::Context) {
        if self.timeline.busy {
            return;
        }
        let (Some(client), Some(session), Some(doc), Some(scene)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.as_ref(),
            self.timeline.scene(),
        ) else {
            return;
        };
        let review = match Review::capture(
            &doc.project,
            &scene.id,
            uuid::Uuid::new_v4().as_u128() as u64,
        ) {
            Ok(review) => review,
            Err(error) => {
                self.timeline.error = Some(error);
                return;
            }
        };
        let path = format!(
            "/v1/video/projects/{}/scenes/{}/compiled",
            doc.project.id.0, scene.id.0
        );
        let epoch = self.timeline.epoch;
        self.timeline.busy = true;
        self.timeline.compiled = State {
            open: true,
            review: Some(review),
            ..Default::default()
        };
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(&session, false, &path, None));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Compiled {
                    session,
                    epoch,
                    result,
                }),
            );
        });
    }
    /// Both Refresh and Approve first commit visible edits, just like direct Generate.
    fn timeline_compiled_action(&mut self, ctx: &egui::Context, action: PostAction) {
        let commands = self.timeline.draft_edits();
        if commands.is_empty() && !self.timeline.busy {
            self.timeline_action(ctx, action);
        } else if !self.timeline.busy {
            self.timeline.after_edit = Some(action);
            self.timeline_edit(ctx, commands, 0);
        }
    }
    pub(super) fn timeline_approve_compiled(&mut self, ctx: &egui::Context) {
        if self.timeline.busy || self.timeline.admitting {
            return;
        }
        if self.timeline.pending.is_some() {
            self.timeline.compiled.error = Some(
                "Retrieve the unconfirmed submission in the queue before starting this plan."
                    .into(),
            );
            return;
        }
        let Some(project) = self.timeline.document.as_ref().map(|d| &d.project) else {
            return;
        };
        let request = self
            .timeline
            .compiled
            .review
            .as_ref()
            .and_then(|r| r.request(project));
        let Some(request) = request.filter(|_| self.timeline.compiled.value.is_some()) else {
            self.timeline.compiled.error = Some(
                "Reviewed inputs changed. Refresh and review the new plan before generating."
                    .into(),
            );
            return;
        };
        self.timeline.pending_seed_advance = self
            .timeline
            .compiled
            .review
            .as_ref()
            .and_then(|r| r.advance.clone());
        self.timeline.pending = Some((request, uuid::Uuid::new_v4().to_string()));
        self.timeline.compiled.open = false;
        self.timeline_submit_pending(ctx);
    }
    pub(super) fn timeline_compiled_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.compiled.open {
            return;
        }
        let mut state = std::mem::take(&mut self.timeline.compiled);
        let mut open = true;
        let mut close = false;
        let mut refresh = false;
        let mut approve = false;
        let stale = state.stale(
            self.timeline.document.as_ref().map(|d| &d.project),
            !self.timeline.draft_edits().is_empty(),
        );
        egui::Window::new("Plan Review").id(egui::Id::new("video-compiled")).open(&mut open).default_width(810.).default_height(590.).max_height((ctx.content_rect().height()-70.).max(200.)).vscroll(true).show(ctx,|ui|{
            if !helto::reveal_in_viewport(ui,self.prefs.privacy_mode){ui.label("Hidden plan · hover to reveal");return;}
            ui.weak("Prepared source · generation has not started");
            ui.horizontal_wrapped(|ui|{let r=ui.add_enabled_ui(!self.timeline.busy,|ui|helto::icon_button(ui,crate::icons::Icon::Retry,"Refresh plan",false)).inner;control(ui,"compiled-refresh",&r);refresh=r.clicked();let r=helto::icon_button(ui,crate::icons::Icon::Close,"Close review",false);control(ui,"compiled-close",&r);close=r.clicked();let r=helto::action(ui,crate::icons::Icon::Generate,"Approve and Generate",true,if stale{Some("The plan changed. Refresh the review.")}else if state.value.is_none(){Some("Loading the plan.")}else{(self.timeline.busy||self.timeline.admitting||self.timeline.pending.is_some()).then_some("Wait for the current timeline operation.")});control(ui,"compiled-approve",&r);approve=r.clicked();});
            if let Some(error)=&state.error{ui.colored_label(helto::ERROR,error);}
            if let Some(plan)=&state.value{
                if stale{ui.colored_label(helto::ACCENT,"Reviewed inputs changed. Refresh and review the new plan.");}
                let mode=match plan["metadata"]["format"].as_str(){Some("full_reference_six_sections")=>"H3 Reference",Some("base_three_fields")=>"H3 I2V",_=>"H3"};
                ui.label(format!("{mode} · revision {}",plan["project_revision"]));
                if let Some(review)=&state.review {
                    ui.label(format!("Seed: {}",review.seed));
                    if let Ok(scene)=review.project.scene(&review.scene) {
                        let model=review.project.effective_model(scene);
                        ui.label(format!("Model: {} / {} / {}",model.adapter,model.variant,model.profile));
                        ui.collapsing("Generation settings",|ui|{
                            if let Some(settings)=review.project.effective_settings(scene) {
                                let mut text=serde_json::to_string_pretty(&settings.values).unwrap_or_default();
                                ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_rows(8).desired_width(f32::INFINITY).interactive(false));
                            }
                        });
                    }
                }
                if let Some(frames)=plan["effective_shape"]["frames"].as_u64(){ui.label(format!("Base sampling: {} × {} · {} frames / 24 fps · {:.3} s",plan["effective_shape"]["width"],plan["effective_shape"]["height"],frames,frames as f64/24.));}
                for issue in plan["runtime_issues"].as_array().into_iter().flatten(){let code=issue.as_str().unwrap_or("");ui.colored_label(helto::ERROR,match code{"h3_voice_reference_unavailable"=>"This worker does not yet accept voice audio references.","h3_reference_media_unavailable"=>"This worker does not yet accept character reference pictures.","h3_section_reference_unavailable"=>"This worker does not yet accept all selected Section image roles.",_=>code});}
                ui.label("Compiled prompt");let mut prompt=plan["prompt"].as_str().unwrap_or("");let r=ui.add(egui::TextEdit::multiline(&mut prompt).code_editor().desired_rows(15).desired_width(f32::INFINITY));control(ui,"compiled-prompt",&r);
                let header=ui.collapsing("Input bindings",|ui|{for key in ["subjects","images","audios","videos"]{for input in plan["reference_manifest"][key].as_array().into_iter().flatten(){ui.label(format!("{} · {} · {}",input["prompt_label"].as_str().unwrap_or(""),input["role"].as_str().or_else(||input["reference_kind"].as_str()).unwrap_or(key),input["asset_id"].as_str().or_else(||input["source_alias"].as_str()).unwrap_or("")));}}});control(ui,"compiled-bindings",&header.header_response);
                if plan["metadata"]["format"]=="base_three_fields"&&let Some(refs)=plan["text_references"].as_array()&&!refs.is_empty(){ui.weak("I2V cast descriptions and voice guidance are text conditioning; attached cast media are not identity/timbre inputs in this mode.");}
                ui.weak("This optional action uses the displayed seed and reviewed inputs. Ordinary Generate remains available without review.");
            }else if self.timeline.busy{ui.spinner();}
        });
        state.open = open && !close;
        self.timeline.compiled = state;
        if refresh {
            self.timeline_compiled_action(ctx, PostAction::Compiled);
        } else if approve {
            self.timeline_compiled_action(ctx, PostAction::ApproveCompiled);
        }
    }
}

#[cfg(test)]
mod review_tests {
    use super::super::editing::time;
    use super::*;
    use imagine_video_contract::generation::{Defaults, SeedControl};
    fn project() -> VideoProject {
        let mut p = VideoProject::create(CreateProject {
            schema_version: 1,
            generation: Some(Defaults {
                seed: u64::MAX,
                control_after_generate: Some(SeedControl::Increment),
                ..Default::default()
            }),
            default_model: imagine_video_contract::generation::h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: time(24.),
            },
        })
        .unwrap();
        p.scenes.push(Scene {
            id: SceneId("own-scene".into()),
            start: time(0.),
            duration: time(1.),
            description: "Own fixture".into(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        p
    }
    #[test]
    fn reviewed_seed_is_exact_and_irrelevant_revision_changes_do_not_invalidate_inputs() {
        let mut p = project();
        let scene = p.scenes[0].id.clone();
        let review = Review::capture(&p, &scene, 19).unwrap();
        assert_eq!(review.seed, u64::MAX);
        assert_eq!(review.request(&p).unwrap()["seed"], json!(u64::MAX));
        p.project_revision += 1;
        assert!(review.matches(&p));
        assert_eq!(
            review.request(&p).unwrap()["expected_project_revision"],
            p.project_revision
        );
        let expected = review.advance.clone().unwrap().apply(&p).unwrap();
        assert_eq!(expected.seed, SeedControl::Increment.next(u64::MAX, 19));
        p.scenes[0].description = "Changed input".into();
        assert!(review.request(&p).is_none());
    }
    #[test]
    fn review_rejects_seed_model_geometry_and_selection_identity_changes() {
        let p = project();
        let scene = p.scenes[0].id.clone();
        let review = Review::capture(&p, &scene, 31).unwrap();
        let mut changed = p.clone();
        changed.generation.as_mut().unwrap().seed = 7;
        assert!(!review.matches(&changed));
        changed = p.clone();
        changed.default_model.variant = "i2v".into();
        assert!(!review.matches(&changed));
        changed = p.clone();
        changed.delivery.width = 448;
        assert!(!review.matches(&changed));
        changed = p.clone();
        changed.scenes.clear();
        assert!(!review.matches(&changed));
        changed = p.clone();
        changed.id = ProjectId("another-own-project".into());
        assert!(!review.matches(&changed));
        changed = p.clone();
        changed
            .creative
            .scene_kinds
            .insert(scene, imagine_video_contract::creative::SceneKind::Imported);
        assert!(!review.matches(&changed));
    }
    #[test]
    fn compile_response_must_belong_to_the_captured_inputs_and_legacy_random_seed_is_chosen_once() {
        let mut p = project();
        p.generation.as_mut().unwrap().control_after_generate = None;
        p.generation.as_mut().unwrap().random_seed = true;
        let scene = p.scenes[0].id.clone();
        let review = Review::capture(&p, &scene, 123).unwrap();
        assert_eq!(review.seed, 123);
        assert!(review.advance.is_none());
        let value = json!({"project_id":p.id,"scene_id":scene,"project_revision":p.project_revision,"input_revision":review.input_revision});
        let mut state = State {
            review: Some(review),
            ..Default::default()
        };
        state.receive(value.clone());
        assert!(state.value.is_some());
        let mut stale = value;
        stale["project_revision"] = json!(p.project_revision + 1);
        state.receive(stale);
        assert!(state.value.is_none());
        assert!(state.error.is_some());
    }
}
