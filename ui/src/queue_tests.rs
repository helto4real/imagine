//! Synthetic job events and egui frames only; no installed service or user data.
use super::*;
use queue::{Completion, Event};

fn stage(desk: &mut Desk, prompts: &[&str]) -> Vec<String> {
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    let prompts = prompts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let prepared = desk.app.prepare_prompts(&prompts).unwrap();
    desk.app.queue.stage(prepared);
    desk.app.queue.jobs.iter().map(|j| j.id.clone()).collect()
}

fn viewer_text(desk: &mut Desk) -> Vec<String> {
    let mut output = desk.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            ..Default::default()
        },
        |ui| desk.app.content(ui),
    );
    output.textures_delta.clear();
    output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) => Some(text.galley.text().to_owned()),
            _ => None,
        })
        .collect()
}

fn sampling(completed: u32, total: u32) -> client::SamplingProgress {
    client::SamplingProgress {
        completed,
        total,
        remaining_seconds: 10.,
        updated_at_ms: 0,
    }
}

#[test]
fn main_progress_tracks_the_running_queue_job_without_replacing_the_selected_image() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["First", "Second", "Third"]);
    desk.app.queue.follow = false;
    let selected = desk.app.image.selected_result().unwrap().asset_id.clone();
    let draft = desk.app.image.prompt.clone();
    let done = completed(desk._root.path(), "finished-first", [1, 2, 3, 255]);
    desk.app.queue_message(
        &desk.ctx,
        Event::Finished(ids[0].clone(), Ok(Completion::new(done))),
    );
    desk.app.queue_message(
        &desk.ctx,
        Event::Accepted(vec![
            (ids[1].clone(), "job_second".into()),
            (ids[2].clone(), "job_third".into()),
        ]),
    );
    for (step, percent) in [(12, "60%"), (16, "80%")] {
        desk.app.queue_message(
            &desk.ctx,
            Event::Progress(
                ids[1].clone(),
                Progress::Status {
                    label: "Sampling".into(),
                    sampling: Some(sampling(step, 20)),
                },
            ),
        );
        let text = viewer_text(&mut desk);
        assert!(text.iter().any(|t| t == percent), "{text:?}");
        assert!(
            text.iter()
                .any(|t| t.starts_with(&format!("#2 · Sampling {step} / 20"))),
            "{text:?}"
        );
        assert!(!text.iter().any(|t| t == "100%"));
        assert_eq!(desk.app.image.selected_result().unwrap().asset_id, selected);
        assert_eq!(desk.app.image.prompt, draft);
        assert!(!desk.app.queue.follow);
    }
    desk.frame(vec![]);
    assert!(!desk.control("Cancel generation").is_disabled());
    desk.app.queue.jobs[1].cancelling = true;
    desk.app.queue.jobs[1].status = "Cancelling".into();
    let text = viewer_text(&mut desk);
    assert!(text.iter().any(|t| t.starts_with("#2 · Cancelling")));
    assert!(!text.iter().any(|t| t.contains("ETA")));
    desk.frame(vec![]);
    assert!(desk.control("Cancel generation").is_disabled());
}

#[test]
fn main_progress_prefers_inference_over_local_saving_and_preserves_other_workspace_progress() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["Saving first", "Running second"]);
    desk.app.queue.jobs[0].status = "Saving".into();
    desk.app.queue.jobs[1].status = "Queued".into();
    let text = viewer_text(&mut desk);
    assert!(text.iter().any(|t| t == "#1 · Saving"));
    assert!(!text.iter().any(|t| t == "100%"));
    desk.app.queue_message(
        &desk.ctx,
        Event::Progress(
            ids[1].clone(),
            Progress::Status {
                label: "Preparing inputs".into(),
                sampling: None,
            },
        ),
    );
    let text = viewer_text(&mut desk);
    assert!(text.iter().any(|t| t == "#2 · Preparing inputs"));
    assert!(!text.iter().any(|t| t == "100%"));
    // A queued upscale must not cover the progress of an image already running.
    desk.app.image.working = true;
    desk.app.image.status = "Queued".into();
    desk.app.queue_message(
        &desk.ctx,
        Event::Progress(
            ids[1].clone(),
            Progress::Status {
                label: "Sampling".into(),
                sampling: Some(sampling(2, 20)),
            },
        ),
    );
    assert!(viewer_text(&mut desk).iter().any(|t| t == "10%"));
    desk.app.workspace = Workspace::CharacterSheet;
    desk.app.sheet.working = true;
    desk.app.sheet.status = "Portrait · Sampling".into();
    desk.app.sheet.sampling = Some(sampling(5, 20));
    let text = viewer_text(&mut desk);
    assert!(text.iter().any(|t| t == "25%"));
    assert!(!text.iter().any(|t| t == "10%"));
    desk.app.workspace = Workspace::Image;
    desk.app.image.working = false;
    desk.app.image.status = "Saved".into();
    desk.app.queue.jobs[0].working = false;
    desk.app.queue_message(
        &desk.ctx,
        Event::Progress(
            ids[1].clone(),
            Progress::Status {
                label: "Queued".into(),
                sampling: None,
            },
        ),
    );
    let text = viewer_text(&mut desk);
    assert!(text.iter().any(|t| t == "#2 · Queued"));
    assert!(!text.iter().any(|t| t == "100%"));
    desk.app.queue.jobs[1].working = false;
    assert!(viewer_text(&mut desk).iter().any(|t| t == "100%"));
}

fn completed(root: &std::path::Path, name: &str, color: [u8; 4]) -> Completed {
    let file = root.join(format!("{name}.png"));
    image::RgbaImage::from_pixel(24, 16, image::Rgba(color))
        .save(&file)
        .unwrap();
    Completed {
        assets: vec![client::CompletedAsset {
            asset_id: name.into(),
            role: OutputRole::Image,
            file,
            preview: Preview {
                width: 24,
                height: 16,
                pixels: color.repeat(24 * 16),
            },
        }],
        seed: Some(123),
    }
}

#[test]
fn series_freezes_requests_references_settings_and_destination_without_locking_the_composer() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["  First\nparagraph  ", "Second"]);
    let first = desk.app.queue.jobs[0].submission.clone();
    desk.app.image.prompt = "Next draft".into();
    desk.app.image.references.clear();
    desk.app.prefs.generation.steps = 12;
    desk.app.prefs.output_folder = desk._root.path().join("different-output");
    assert_eq!(
        desk.app.queue.jobs[0].snapshot.prompt,
        "  First\nparagraph  "
    );
    assert_eq!(desk.app.queue.jobs[0].snapshot.references.len(), 2);
    assert_eq!(desk.app.queue.jobs[0].submission.request, first.request);
    assert_eq!(desk.app.queue.jobs[0].submission.folder, first.folder);
    assert_ne!(ids[0], ids[1]);
    assert_eq!(desk.app.queue.jobs[0].batch, desk.app.queue.jobs[1].batch);
    desk.frame(vec![]);
    assert!(!desk.control("Add to queue").is_disabled());
    assert!(!desk.control("Multiple prompts").is_disabled());
    assert!(
        desk.app
            .prepare_prompts(&["Valid".into(), " ".into()])
            .is_err()
    );
    assert_eq!(desk.app.queue.jobs.len(), 2);
}

#[test]
fn completed_jobs_do_not_replace_a_selected_result_or_modify_the_next_prompt() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["First", "Second", "Third"]);
    desk.app.image.prompt = "Unsubmitted draft".into();
    for (index, id) in ids.iter().enumerate() {
        if index == 1 {
            desk.app.queue.follow = false;
        }
        let done = completed(
            desk._root.path(),
            &format!("image-{index}"),
            [index as u8, 30, 50, 255],
        );
        desk.app.queue_message(
            &desk.ctx,
            Event::Finished(id.clone(), Ok(Completion::new(done))),
        );
    }
    assert_eq!(desk.app.queue.gallery.len(), 3);
    assert_eq!(desk.app.queue.selected.as_ref(), Some(&ids[0]));
    assert_eq!(
        desk.app.image.selected_result().unwrap().asset_id,
        "image-0"
    );
    assert_eq!(desk.app.image.prompt, "Unsubmitted draft");
    let folder = desk.app.prefs.output_folder.clone();
    desk.app.reuse_queue_settings(&ids[1]);
    assert_eq!(desk.app.image.prompt, "Second");
    assert_eq!(desk.app.prefs.output_folder, folder);
    assert_eq!(
        desk.app.image.selected_result().unwrap().asset_id,
        "image-0"
    );
    desk.app.queue.loading = Some(ids[2].clone());
    let stale = completed(desk._root.path(), "stale", [0, 0, 0, 255]);
    desk.app
        .queue_message(&desk.ctx, Event::Loaded(ids[1].clone(), Ok(stale)));
    assert_eq!(
        desk.app.image.selected_result().unwrap().asset_id,
        "image-0"
    );
    let newest = completed(desk._root.path(), "selected", [1, 2, 3, 255]);
    desk.app
        .queue_message(&desk.ctx, Event::Loaded(ids[2].clone(), Ok(newest)));
    assert_eq!(
        desk.app.image.selected_result().unwrap().asset_id,
        "selected"
    );
    assert_eq!(desk.app.upscale_source().unwrap().asset_id, "selected");
    desk.app.start_editing();
    assert_eq!(
        desk.app.edit.desk.selected_result().unwrap().asset_id,
        "selected"
    );
}

#[test]
fn errors_cancellation_and_late_cancel_acknowledgements_are_scoped_to_the_origin_job() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["First", "Second"]);
    desk.app.queue_message(
        &desk.ctx,
        Event::Accepted(vec![
            (ids[0].clone(), "server-a".into()),
            (ids[1].clone(), "server-b".into()),
        ]),
    );
    desk.app.queue_message(
        &desk.ctx,
        Event::Finished(
            ids[0].clone(),
            Err(Failure {
                message: "Synthetic download failure".into(),
                retryable: true,
            }),
        ),
    );
    assert!(!desk.app.queue.jobs[0].working);
    assert_eq!(
        desk.app.queue.jobs[0].submission.job.as_deref(),
        Some("server-a")
    );
    assert!(desk.app.queue.jobs[1].working);
    desk.app.queue.jobs[1].cancelling = true;
    // An acknowledgement alone is not terminal completion.
    desk.app.queue_message(
        &desk.ctx,
        Event::Cancelled(ids[1].clone(), "server-b".into(), Ok(())),
    );
    assert!(desk.app.queue.jobs[1].working);
    desk.app.queue_message(
        &desk.ctx,
        Event::Finished(
            ids[1].clone(),
            Err(Failure {
                message: "Generation cancelled.".into(),
                retryable: false,
            }),
        ),
    );
    desk.app.queue_message(
        &desk.ctx,
        Event::Cancelled(
            ids[1].clone(),
            "server-b".into(),
            Err(Failure {
                message: "stale".into(),
                retryable: true,
            }),
        ),
    );
    assert_eq!(desk.app.queue.jobs[1].status, "Cancelled");
    assert!(desk.app.queue.jobs[1].error.is_none());
    assert_eq!(
        desk.app.queue.jobs[0].error.as_deref(),
        Some("Synthetic download failure")
    );
}

#[test]
fn dock_reveals_all_thumbnails_on_panel_hover_including_empty_space_and_hides_on_leave_or_blur() {
    let mut desk = Desk::new();
    let ids = stage(&mut desk, &["First", "Second"]);
    for (index, id) in ids.iter().enumerate() {
        let done = completed(
            desk._root.path(),
            &format!("privacy-{index}"),
            [120, index as u8, 0, 255],
        );
        desk.app.queue_message(
            &desk.ctx,
            Event::Finished(id.clone(), Ok(Completion::new(done))),
        );
    }
    let textures = desk
        .app
        .queue
        .gallery
        .iter()
        .map(|r| r.result.assets[0].texture.id())
        .collect::<Vec<_>>();
    let rect = egui::Rect::from_min_max(egui::pos2(20., 100.), egui::pos2(980., 320.));
    for (privacy, focused, pos, visible) in [
        (true, true, egui::pos2(10., 10.), false),
        (true, true, egui::pos2(40., 110.), true), // panel header, outside all thumbnails
        (true, true, egui::pos2(975., 305.), true), // blank area to the right
        (true, true, egui::pos2(500., 200.), true), // one thumbnail reveals both
        (true, false, egui::pos2(500., 200.), false),
        (true, true, egui::pos2(10., 10.), false),
        (false, false, egui::pos2(10., 10.), true),
    ] {
        desk.app.prefs.privacy_mode = privacy;
        let mut output = desk.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000., 400.),
                )),
                focused,
                events: vec![egui::Event::PointerMoved(pos)],
                ..Default::default()
            },
            |ui| desk.app.queue_panel(ui, rect),
        );
        output.textures_delta.clear();
        let meshes = desk.ctx.tessellate(output.shapes, output.pixels_per_point);
        for texture in &textures {
            let drawn=meshes.iter().any(|m|matches!(&m.primitive,egui::epaint::Primitive::Mesh(m) if m.texture_id==*texture));
            assert_eq!(
                drawn, visible,
                "privacy={privacy}, focused={focused}, pos={pos:?}"
            );
        }
    }
}

#[test]
fn headless_queue_layout() {
    for (width, height, name, privacy) in [
        (1800, 950, "queue-wide", false),
        (1280, 800, "queue-desktop", false),
        (640, 1000, "queue-compact", false),
        (1280, 800, "queue-hidden", true),
    ] {
        // Capture the first texture deltas too: the CPU renderer needs the
        // complete atlas rather than updates from an already-rendered Desk.
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let mut desk = Desk {
            ctx,
            app,
            nodes: Vec::new(),
            _root: root,
        };
        let ids = stage(
            &mut desk,
            &[
                "First example",
                "Second example",
                "Third example",
                "Waiting example",
            ],
        );
        for (index, id) in ids.iter().take(3).enumerate() {
            let done = completed(
                desk._root.path(),
                &format!("layout-{index}"),
                [60, 80, index as u8 * 80, 255],
            );
            desk.app.queue_message(
                &desk.ctx,
                Event::Finished(id.clone(), Ok(Completion::new(done))),
            );
        }
        desk.app.queue.jobs[3].status = "Sampling".into();
        desk.app.queue.jobs[3].submission.job = Some("synthetic-running".into());
        desk.app.queue.jobs[3].sampling = Some(client::SamplingProgress {
            completed: 12,
            total: 20,
            remaining_seconds: 8.,
            updated_at_ms: 0,
        });
        desk.app.prefs.privacy_mode = privacy;
        let mut output = egui::FullOutput::default();
        for _ in 0..3 {
            output.append(desk.ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| desk.app.content(ui),
            ));
        }
        let nodes = &output
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes;
        for label in ["Collapse queue", "View result 1", "Follow latest"] {
            let bounds = nodes
                .iter()
                .map(|(_, n)| n)
                .find(|n| n.label() == Some(label))
                .unwrap_or_else(|| panic!("Missing {label} at {width}"))
                .bounds()
                .unwrap();
            assert!(
                bounds.x0 >= 0.
                    && bounds.x1 <= width as f64
                    && bounds.y0 >= 0.
                    && bounds.y1 <= height as f64,
                "{label}: {bounds:?}"
            );
        }
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            render(&desk.ctx, output, width, height)
                .save(PathBuf::from(dir).join(format!("{name}.png")))
                .unwrap();
        } else {
            output.textures_delta.clear();
        }
    }
}

#[test]
fn multi_prompt_editor_keeps_paragraphs_separate_and_can_add_remove_and_close() {
    let mut desk = Desk::new();
    desk.click("Multiple prompts");
    assert_eq!(desk.app.queue.drafts.as_ref().unwrap().len(), 2);
    desk.app.queue.drafts.as_mut().unwrap()[1] = "One prompt\n\nwith paragraphs".into();
    desk.frame(vec![]);
    assert!(!desk.control("Add all to queue").is_disabled());
    desk.click("Add prompt");
    assert_eq!(desk.app.queue.drafts.as_ref().unwrap().len(), 3);
    assert!(desk.control("Add all to queue").is_disabled());
    desk.click("Remove prompt 3");
    assert_eq!(desk.app.queue.drafts.as_ref().unwrap().len(), 2);
    assert_eq!(
        desk.app.queue.drafts.as_ref().unwrap()[1],
        "One prompt\n\nwith paragraphs"
    );
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert!(desk.app.queue.drafts.is_none());
}

#[tokio::test]
async fn ui_retries_uncertain_batch_admission_and_failed_retrieval_without_duplicate_jobs() {
    use axum::{
        Json, Router,
        extract::{Path, State},
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::{get, post},
    };
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    #[derive(Clone, Default)]
    struct Fake {
        admissions: Arc<Mutex<Vec<(String, Value)>>>,
        polls: Arc<AtomicUsize>,
        failed_poll: Arc<AtomicBool>,
        bytes: Arc<Vec<u8>>,
    }
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_pixel(24, 16, image::Rgba([20, 40, 60, 255]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let state = Fake {
        bytes: Arc::new(bytes.into_inner()),
        ..Default::default()
    };
    let router=Router::new()
        .route("/v1/jobs/batch",post(|State(state):State<Fake>,headers:HeaderMap,Json(body):Json<Value>|async move {
            let mut admissions=state.admissions.lock().unwrap();
            admissions.push((headers["Idempotency-Key"].to_str().unwrap().into(),body));
            if admissions.len()==1 {
                // Simulate a response lost after the server already admitted these IDs.
                (StatusCode::INTERNAL_SERVER_ERROR,Json(serde_json::json!({"error":{"code":"internal_error"}})))
            } else { (StatusCode::ACCEPTED,Json(serde_json::json!({"jobs":[{"id":"job_a"},{"id":"job_b"}]}))) }
        }))
        .route("/v1/jobs/{id}",get(|State(state):State<Fake>,Path(id):Path<String>|async move {
            state.polls.fetch_add(1,Ordering::SeqCst);
            if id=="job_a" && !state.failed_poll.swap(true,Ordering::SeqCst) {
                return (StatusCode::SERVICE_UNAVAILABLE,Json(serde_json::json!({"error":{"code":"temporarily_unavailable"}})));
            }
            (StatusCode::OK,Json(serde_json::json!({"status":"succeeded","output":{"assets":[{"id":format!("asset_{id}"),"role":"image"}]},"effective":{"seed":99}})))
        }))
        .route("/v1/assets/{id}",get(|State(state):State<Fake>|async move { ([("content-type","image/png")],state.bytes.as_ref().clone()).into_response() }))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut desk = Desk::new();
    desk.app.image.references.clear();
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.app.client = Some(Client::connect(&Connection { url }).unwrap());
    assert!(desk.app.enqueue_prompts(
        &desk.ctx,
        vec!["Exact first prompt".into(), "Second\n\nparagraph".into()]
    ));
    async fn drain(desk: &mut Desk) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                desk.app.receive(&desk.ctx);
                if desk.app.queue.jobs.iter().all(|j| !j.working) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
    drain(&mut desk).await;
    assert!(
        desk.app
            .queue
            .jobs
            .iter()
            .all(|j| j.error.is_some() && j.submission.job.is_none())
    );
    let key = desk.app.queue.jobs[0].submission.key.clone();
    let id = desk.app.queue.jobs[0].id.clone();
    desk.app.image.prompt = "New unsent draft".into();
    desk.app.prefs.generation.steps = 12;
    desk.app.retry_queue_job(&desk.ctx, &id);
    drain(&mut desk).await;
    assert_eq!(desk.app.queue.jobs[0].submission.key, key);
    assert_eq!(
        desk.app.queue.jobs[0].submission.job.as_deref(),
        Some("job_a")
    );
    assert_eq!(desk.app.queue.gallery.len(), 1);
    assert_eq!(desk.app.queue.jobs[1].status, "Saved");
    desk.app.retry_queue_job(&desk.ctx, &id);
    drain(&mut desk).await;
    assert_eq!(desk.app.queue.gallery.len(), 2);
    assert_eq!(desk.app.image.prompt, "New unsent draft");
    let first_result = desk.app.queue.gallery[0].id.clone();
    desk.frame(vec![]);
    desk.click("View result 1");
    tokio::time::timeout(Duration::from_secs(5), async {
        while desk.app.queue.loading.is_some() {
            desk.app.receive(&desk.ctx);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(desk.app.queue.selected.as_ref(), Some(&first_result));
    assert!(!desk.app.queue.follow);
    assert_eq!(
        desk.app.image.selected_result().unwrap().texture.size(),
        [24, 16]
    );
    assert_eq!(desk.app.image.prompt, "New unsent draft");
    let admissions = state.admissions.lock().unwrap();
    assert_eq!(admissions.len(), 2);
    assert_eq!(admissions[0], admissions[1]);
    assert_eq!(admissions[0].1["jobs"][0]["prompt"], "Exact first prompt");
    assert_eq!(admissions[0].1["jobs"][1]["prompt"], "Second\n\nparagraph");
    server.abort();
}
