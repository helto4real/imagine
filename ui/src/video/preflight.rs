//! Server model capabilities and project-only admission blockers, so the
//! editor shows what Generate would refuse before it is pressed.
use super::*;
use imagine_video_contract::planning::Capabilities;

type Key = (ProjectId, u64, SceneId);

#[derive(Default)]
pub(super) struct State {
    capabilities: Option<Vec<(ModelSelection, Capabilities)>>,
    capabilities_requested: bool,
    /// The scene and project revision `blockers` describe.
    key: Option<Key>,
    pending: Option<Key>,
    blockers: Vec<String>,
}

impl State {
    pub(super) fn session_changed(&mut self) {
        *self = Self::default();
    }
}

impl ImagineApp {
    /// Requests capabilities once per session and blockers for the selected
    /// scene whenever its project revision changes.
    pub(super) fn timeline_preflight_poll(&mut self, ctx: &egui::Context) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        let epoch = self.timeline.epoch;
        if !self.timeline.preflight.capabilities_requested {
            self.timeline.preflight.capabilities_requested = true;
            let (client, session, tx, ctx) = (
                client.clone(),
                session.clone(),
                self.tx.clone(),
                ctx.clone(),
            );
            std::thread::spawn(move || {
                let result = runtime().block_on(client.timeline_request(
                    &session,
                    false,
                    "/v1/video/capabilities",
                    None,
                ));
                send(
                    &tx,
                    &ctx,
                    Message::Timeline(Event::Capabilities { epoch, result }),
                );
            });
        }
        let Some(document) = &self.timeline.document else {
            return;
        };
        let Some(scene) = self.timeline.primary_scene(&document.project) else {
            return;
        };
        let key = (
            document.project.id.clone(),
            document.project.project_revision,
            scene,
        );
        if self.timeline.busy
            || self.timeline.preflight.key.as_ref() == Some(&key)
            || self.timeline.preflight.pending.as_ref() == Some(&key)
        {
            return;
        }
        self.timeline.preflight.pending = Some(key.clone());
        let path = format!(
            "/v1/video/projects/{}/scenes/{}/preflight",
            key.0.0, key.2.0
        );
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(&session, false, &path, None));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Preflight { epoch, key, result }),
            );
        });
    }
    pub(super) fn timeline_capabilities_reply(
        &mut self,
        result: Result<Value, client::TimelineFailure>,
    ) {
        // Without capabilities every role stays offered; admission still decides.
        let models = result.ok().and_then(|value| {
            value["models"]
                .as_array()?
                .iter()
                .map(|entry| {
                    Some((
                        serde_json::from_value(entry["model"].clone()).ok()?,
                        serde_json::from_value(entry["capabilities"].clone()).ok()?,
                    ))
                })
                .collect::<Option<Vec<_>>>()
        });
        self.timeline.preflight.capabilities = models;
    }
    pub(super) fn timeline_preflight_reply(
        &mut self,
        key: Key,
        result: Result<Value, client::TimelineFailure>,
    ) {
        let state = &mut self.timeline.preflight;
        if state.pending.as_ref() != Some(&key) {
            return;
        }
        state.pending = None;
        // A failed check shows nothing rather than a guess; Generate reports it.
        state.blockers = result
            .ok()
            .filter(|value| value["project_revision"] == key.1)
            .and_then(|value| serde_json::from_value(value["blockers"].clone()).ok())
            .unwrap_or_default();
        state.key = Some(key);
    }
    /// Blockers for the selected scene at the open project revision.
    pub(super) fn timeline_scene_blockers(&self) -> &[String] {
        let state = &self.timeline.preflight;
        let current = self.timeline.document.as_ref().and_then(|d| {
            let scene = self.timeline.primary_scene(&d.project)?;
            Some((d.project.id.clone(), d.project.project_revision, scene))
        });
        if current.is_some() && state.key == current {
            &state.blockers
        } else {
            &[]
        }
    }
    /// Whether the scene's model accepts `kind` media in `role`; None while unknown.
    pub(super) fn timeline_accepts(
        &self,
        project: &VideoProject,
        scene: &Scene,
        role: &InputRole,
        kind: &MediaKind,
    ) -> Option<bool> {
        let model = project.effective_model(scene);
        let (_, capabilities) = self
            .timeline
            .preflight
            .capabilities
            .as_ref()?
            .iter()
            .find(|(m, _)| m.adapter == model.adapter && m.variant == model.variant)?;
        Some(
            capabilities
                .input_roles
                .get(role)
                .is_some_and(|kinds| kinds.contains(kind)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn app(ctx: &egui::Context, root: &std::path::Path) -> (ImagineApp, VideoProject) {
        let mut app = ImagineApp::synthetic_preview(ctx, root.into());
        let mut project = super::super::media_picker::tests::project();
        project.default_model = ModelSelection {
            adapter: "h3".into(),
            model: "minimax-h3".into(),
            variant: "i2v".into(),
            profile: "i2v".into(),
            profile_revision: "2".into(),
        };
        app.timeline.document = Some(Snapshot::with(project.clone(), json!({})));
        app.timeline.selected.insert(project.scenes[0].id.clone());
        (app, project)
    }
    #[test]
    fn capabilities_decide_offered_roles_and_unknown_models_stay_open() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let (mut app, project) = app(&ctx, root.path());
        let scene = &project.scenes[0];
        assert_eq!(
            app.timeline_accepts(&project, scene, &InputRole::LastFrame, &MediaKind::Image),
            None
        );
        let caps = |roles| {
            json!({"revision":"r","runtime_available":true,"experimental":true,
            "validation_status":"unvalidated","input_roles":roles,"continuity":[],"incompatible_roles":[]})
        };
        app.timeline_capabilities_reply(Ok(json!({"models":[
            {"model":{"adapter":"h3","model":"minimax-h3","variant":"i2v","profile":"i2v","profile_revision":"2"},
             "capabilities":caps(json!({"first_frame":["image"]}))}]})));
        let accepts = |role, kind| app.timeline_accepts(&project, scene, &role, &kind);
        assert_eq!(accepts(InputRole::FirstFrame, MediaKind::Image), Some(true));
        assert_eq!(
            accepts(InputRole::FirstFrame, MediaKind::Video),
            Some(false)
        );
        assert_eq!(accepts(InputRole::LastFrame, MediaKind::Image), Some(false));
    }
    #[test]
    fn blockers_apply_only_to_the_revision_they_were_checked_for() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let (mut app, project) = app(&ctx, root.path());
        let key = (
            project.id.clone(),
            project.project_revision,
            project.scenes[0].id.clone(),
        );
        // An unrequested reply is ignored.
        app.timeline_preflight_reply(
            key.clone(),
            Ok(json!({"project_revision":key.1,"blockers":["x"]})),
        );
        assert!(app.timeline_scene_blockers().is_empty());
        app.timeline.preflight.pending = Some(key.clone());
        app.timeline_preflight_reply(
            key.clone(),
            Ok(json!({"project_revision":key.1,"blockers":["scene_runtime_unsupported"]})),
        );
        assert_eq!(app.timeline_scene_blockers(), ["scene_runtime_unsupported"]);
        app.timeline
            .document
            .as_mut()
            .unwrap()
            .project
            .project_revision += 1;
        assert!(app.timeline_scene_blockers().is_empty());
    }
}
