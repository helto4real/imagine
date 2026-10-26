//! Animate starts the ordinary Timeline workflow; it never submits a generation.
use super::*;

/// Seconds of a new Animate Scene and the one Section that spans it.
const SECONDS: u64 = 10;

impl ImagineApp {
    pub(in crate::app) fn can_animate_image(&self) -> bool {
        self.animate_image_blocked().is_none()
    }
    pub(in crate::app) fn animate_image_blocked(&self) -> Option<&'static str> {
        self.edit_image_blocked()
            .or((self.timeline.busy || self.timeline.admitting)
                .then_some("Wait for the current video operation."))
    }

    pub(in crate::app) fn start_animating(&mut self, ctx: &egui::Context) {
        if !self.can_animate_image() || self.timeline_hold_replace(super::Replace::Animate) {
            return;
        }
        self.animate_now(ctx);
    }
    pub(in crate::app) fn animate_now(&mut self, ctx: &egui::Context) {
        // Also runs after confirmation, when the Video workspace already shows the dialog.
        if self.timeline.busy || self.timeline.admitting {
            return;
        }
        let (Some(client), Some(image)) = (self.client.clone(), self.image.selected_result())
        else {
            return;
        };
        let file = image.file.clone();
        let [width, height] = image.texture.size();
        let epoch = self.timeline.epoch;
        self.timeline.busy = true;
        self.timeline.error = None;
        self.workspace = Workspace::Video;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(create(&client, &file, (width as u32, height as u32)));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Animated { epoch, result }),
            );
        });
    }
}

fn request(source: (u32, u32)) -> Result<CreateProject, &'static str> {
    let mut generation = imagine_video_contract::generation::Defaults::default();
    let canvas = generation.canvas.as_mut().expect("default canvas");
    canvas.ratio = "Source".into();
    let shape = crate::h3_protocol::Selection {
        ratio: canvas.ratio.clone(),
        portrait: false,
        quality: canvas.quality.clone(),
        seconds: SECONDS as f64,
    }
    .resolve_with_source(Some(source))?;
    Ok(CreateProject {
        schema_version: 1,
        default_model: imagine_video_contract::generation::h3_model("i2v"),
        generation: Some(generation),
        delivery: DeliverySettings {
            width: shape.width,
            height: shape.height,
            fps: Time { num: 24, den: 1 },
        },
    })
}

async fn create(
    client: &Client,
    file: &std::path::Path,
    source: (u32, u32),
) -> Result<(client::TimelineSession, Value), client::TimelineFailure> {
    let request = request(source).map_err(|message| client::TimelineFailure {
        code: "invalid_image_dimensions".into(),
        message: message.into(),
        snapshot: None,
        retryable: false,
    })?;
    let connection = client.timeline_session().await?;
    let session = &connection.id;
    // Link the saved image through the same original-file path as Timeline import.
    let media = client.upload_media(session, file).await?;
    let mut raw = client
        .timeline_request(session, true, "/v1/video/projects", Some(&json!(request)))
        .await?;
    let project = raw["project"]["id"].as_str().unwrap_or_default().to_owned();
    raw = client.timeline_request(session, true, &format!("/v1/video/projects/{project}/assets"), Some(&json!({
        "schema_version": 1, "expected_project_revision": raw["project"]["project_revision"],
        "asset_id": media["id"], "name": file.file_name().and_then(|n| n.to_str()).unwrap_or("First Frame"),
    }))).await?;
    let asset = raw["runtime_assets"]
        .as_object()
        .and_then(|assets| assets.iter().find(|(_, id)| **id == media["id"]))
        .map(|(id, _)| AssetId(id.clone()))
        .ok_or_else(|| client::TimelineFailure {
            code: "invalid_response".into(),
            message: "The new project's First Frame could not be attached.".into(),
            snapshot: None,
            retryable: false,
        })?;
    let scene = Scene {
        id: SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple())),
        start: Time { num: 0, den: 1 },
        duration: Time {
            num: SECONDS,
            den: 1,
        },
        description: String::new(),
        model: ModelChoice::ProjectDefault,
        settings: vec![],
        sections: vec![Section {
            id: SectionId(format!("section_{}", uuid::Uuid::new_v4().simple())),
            start: Time { num: 0, den: 1 },
            duration: Time {
                num: SECONDS,
                den: 1,
            },
            prompt: String::new(),
        }],
        bindings: vec![Binding {
            id: BindingId(format!("binding_{}", uuid::Uuid::new_v4().simple())),
            asset,
            role: InputRole::FirstFrame,
            enabled: true,
            video: None,
        }],
        continuity: None,
        accepted_take: None,
    };
    raw = client
        .timeline_request(
            session,
            true,
            &format!("/v1/video/projects/{project}/edits"),
            Some(&json!(EditRequest {
                schema_version: 1,
                expected_project_revision: raw["project"]["project_revision"].as_u64().unwrap_or(0),
                commands: vec![Edit::AddScene {
                    scene: Box::new(scene)
                }],
            })),
        )
        .await?;
    Ok((connection, raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_server::{api, engine, scheduler, storage::Store};
    use std::sync::Arc;

    struct NoInference;
    impl engine::Backend for NoInference {
        fn generate<'a>(
            &'a mut self,
            _: &'a imagine_server::generation::ImagePass,
        ) -> engine::EngineFuture<'a, Vec<u8>> {
            Box::pin(async { panic!("Animate must not generate") })
        }
        fn stop(&mut self) -> engine::EngineFuture<'_, ()> {
            Box::pin(async { Ok(()) })
        }
    }

    #[test]
    fn animate_requires_a_selected_image_and_an_idle_timeline() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.workspace = Workspace::Image;
        assert_eq!(app.animate_image_blocked(), None);
        app.timeline.busy = true;
        assert_eq!(
            app.animate_image_blocked(),
            Some("Wait for the current video operation.")
        );
        app.timeline.busy = false;
        app.image.result = None;
        assert_eq!(
            app.animate_image_blocked(),
            Some("Generate or choose an image first.")
        );
    }

    #[tokio::test]
    async fn animate_creates_distinct_i2v_projects_links_first_frame_and_reopens() {
        let own = tempfile::tempdir().unwrap();
        let image = own.path().join("Own First Frame Å.png");
        image::RgbImage::from_pixel(320, 480, image::Rgb([70, 100, 160]))
            .save(&image)
            .unwrap();
        let mut store = Store::open(&own.path().join("server")).await.unwrap();
        store.videos.destination = Some(crate::test_destination::destination(
            &own.path().join("output"),
        ));
        let (control, worker) = scheduler::start(store.clone(), Box::new(NoInference))
            .await
            .unwrap();
        let (_, state) = tokio::sync::watch::channel(engine::EngineState::default());
        let app = api::App {
            video: control.video.clone(),
            enhancement: imagine_server::enhancement::Service::open(
                own.path().join("enhancement.json"),
            )
            .unwrap(),
            library: Default::default(),
            profiles: Arc::new(
                imagine_server::profiles::Catalog::load(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../config/image/profiles"
                )))
                .unwrap(),
            ),
            store: store.clone(),
            control: control.clone(),
            engine: state,
            uploads: Arc::new(tokio::sync::Semaphore::new(2)),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = Client::connect(&crate::settings::Connection {
            url: format!("http://{}", listener.local_addr().unwrap()),
        })
        .unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, api::router(app)).await.unwrap() });
        let (connection, first) = create(&client, &image, (320, 480)).await.unwrap();
        let project: VideoProject = serde_json::from_value(first["project"].clone()).unwrap();
        project.validate().unwrap();
        assert_eq!(
            project.default_model,
            imagine_video_contract::generation::h3_model("i2v")
        );
        assert!(project.delivery.height > project.delivery.width);
        let canvas = project.generation.as_ref().unwrap().canvas.clone().unwrap();
        assert_eq!(
            (canvas.ratio.as_str(), canvas.quality.as_str()),
            ("Source", "Quick Draft")
        );
        assert!(project.delivery.width * project.delivery.height <= 110_000);
        assert_eq!(project.scenes.len(), 1);
        let scene = &project.scenes[0];
        assert_eq!(scene.duration, Time { num: 10, den: 1 });
        assert_eq!(project.effective_model(scene).variant, "i2v");
        let [section] = scene.sections.as_slice() else {
            panic!("one Section across the Scene");
        };
        assert_eq!(section.start, Time { num: 0, den: 1 });
        assert_eq!(section.duration, scene.duration);
        assert_eq!(scene.bindings.len(), 1);
        assert_eq!(scene.bindings[0].role, InputRole::FirstFrame);
        assert!(scene.bindings[0].enabled);
        assert!(scene.accepted_take.is_none());
        assert_eq!(first["resources"][0]["source"], json!(image));
        // Generate names a selected model file the H3 model root does not have.
        let models = own.path().join("models");
        control.video.lock().await.h3_model_root = Some(models.clone());
        let preflight = format!(
            "/v1/video/projects/{}/scenes/{}/preflight",
            project.id.0, scene.id.0
        );
        let checked = client
            .timeline_request(&connection.id, false, &preflight, None)
            .await
            .unwrap();
        let file = "diffusion_models/minimax_h3/minimaxH3.int8.safetensors";
        assert!(
            checked["blockers"]
                .as_array()
                .unwrap()
                .contains(&json!("h3_model_file_missing"))
        );
        assert_eq!(checked["missing_model_files"], json!([file]));
        std::fs::create_dir_all(models.join(file).parent().unwrap()).unwrap();
        std::fs::write(models.join(file), b"own").unwrap();
        let checked = client
            .timeline_request(&connection.id, false, &preflight, None)
            .await
            .unwrap();
        assert_eq!(checked["missing_model_files"], json!([]));
        assert!(
            !checked["blockers"]
                .as_array()
                .unwrap()
                .contains(&json!("h3_model_file_missing"))
        );
        let (_, second) = create(&client, &image, (320, 480)).await.unwrap();
        assert_ne!(first["project"]["id"], second["project"]["id"]);
        let folder = own.path().join("saved");
        std::fs::create_dir(&folder).unwrap();
        let location = client
            .timeline_request(
                &connection.id,
                true,
                "/v1/video/locations",
                Some(&json!({"schema_version":1,"path":folder})),
            )
            .await
            .unwrap();
        // The project keeps the Quality chosen for it.
        let mut generation = project.generation.clone().unwrap();
        generation.canvas.as_mut().unwrap().quality = "Standard".into();
        let edited = client
            .timeline_request(
                &connection.id,
                true,
                &format!("/v1/video/projects/{}/edits", project.id.0),
                Some(&json!({"schema_version":1,
                "expected_project_revision":project.project_revision,
                "commands":[imagine_video_contract::editing::Edit::SetGenerationDefaults {
                    generation: Some(generation),
                }]})),
            )
            .await
            .unwrap();
        let project: VideoProject = serde_json::from_value(edited["project"].clone()).unwrap();
        let saved = client.timeline_request(&connection.id, true, &format!("/v1/video/projects/{}/save", project.id.0), Some(&json!({"schema_version":1,"expected_project_revision":project.project_revision,"location_id":location["location_id"]}))).await.unwrap();
        let reopened = client
            .timeline_request(
                &connection.id,
                true,
                "/v1/video/projects/open",
                Some(&json!({"schema_version":1,"location_id":location["location_id"]})),
            )
            .await
            .unwrap();
        assert_eq!(saved["project"], reopened["project"]);
        assert_eq!(
            reopened["project"]["generation"]["canvas"]["quality"],
            "Standard"
        );
        assert_eq!(saved["resources"], reopened["resources"]);
        assert!(
            !folder.join("media").exists()
                || std::fs::read_dir(folder.join("media"))
                    .unwrap()
                    .next()
                    .is_none()
        );

        let ctx = egui::Context::default();
        let mut ui = ImagineApp::synthetic_preview(&ctx, own.path().join("ui"));
        ui.timeline_message(
            &ctx,
            Event::Animated {
                epoch: 0,
                result: Ok((connection, first)),
            },
        );
        assert_eq!(
            ui.timeline.document.as_ref().unwrap().project.id,
            project.id
        );
        assert!(ui.timeline.selected.contains(&scene.id));
        assert!(ui.timeline.pending.is_none() && !ui.timeline.admitting);
        let epoch = ui.timeline.epoch;
        ui.timeline.busy = true;
        ui.timeline_message(
            &ctx,
            Event::Animated {
                epoch,
                result: Err(client::TimelineFailure {
                    code: "source_missing".into(),
                    message: "Own missing source".into(),
                    snapshot: None,
                    retryable: false,
                }),
            },
        );
        assert_eq!(
            ui.timeline.document.as_ref().unwrap().project.id,
            project.id
        );
        assert!(!ui.timeline.busy);
        assert_eq!(ui.timeline.error.as_deref(), Some("Own missing source"));
        server.abort();
        let _ = server.await;
        control.call(scheduler::Action::Shutdown).await.unwrap();
        worker.await.unwrap().unwrap();
        store.close().await.unwrap();
    }
}
