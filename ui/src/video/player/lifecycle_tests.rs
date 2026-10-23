//! Async response ownership on source templates only; no jobs, disk or network.
use super::*;

fn app() -> (egui::Context, ImagineApp) {
    let ctx = egui::Context::default();
    let mut app = ImagineApp::synthetic_preview(
        &ctx,
        PathBuf::from("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"),
    );
    app.client = None;
    let p = super::super::media_picker::tests::project();
    app.timeline.document = Some(Snapshot {
        project: p.clone(),
        raw: json!({"project":p}),
    });
    app.timeline.epoch = 9;
    (ctx, app)
}
fn preview(marker: u8) -> Preview {
    Preview {
        width: 2,
        height: 2,
        frames: vec![vec![marker; 12]; 3],
        pcm: vec![],
        duration: 3. / 24.,
    }
}

#[test]
fn late_preview_and_comparison_cannot_reopen_closed_or_replace_successor() {
    let (ctx, mut app) = app();
    let cancelled = Arc::new(AtomicBool::new(false));
    app.timeline.loading = Some((10, cancelled.clone()));
    app.timeline.clear_preview();
    assert!(cancelled.load(Ordering::Acquire));
    let successor = Arc::new(AtomicBool::new(false));
    app.timeline.loading = Some((11, successor.clone()));
    app.timeline.error = Some("Own current operation".into());
    app.timeline_message(
        &ctx,
        Event::Preview {
            epoch: 9,
            nonce: 10,
            asset: "old-own-source".into(),
            result: Ok(preview(20)),
        },
    );
    app.timeline_message(
        &ctx,
        Event::Comparison {
            epoch: 9,
            nonce: 10,
            result: Ok(Box::new(
                Comparison::new(preview(20), preview(80), ["a".into(), "b".into()]).unwrap(),
            )),
        },
    );
    assert!(app.timeline.player.is_none() && app.timeline.comparison.is_none());
    assert_eq!(app.timeline.loading.as_ref().unwrap().0, 11);
    assert!(!successor.load(Ordering::Acquire));
    assert_eq!(app.timeline.error.as_deref(), Some("Own current operation"));
    app.timeline_message(
        &ctx,
        Event::Preview {
            epoch: 9,
            nonce: 11,
            asset: "new-own-source".into(),
            result: Err("Own selected source cannot be decoded".into()),
        },
    );
    assert!(app.timeline.loading.is_none());
    assert_eq!(
        app.timeline.error.as_deref(),
        Some("Own selected source cannot be decoded")
    );
}

#[test]
fn replacing_project_cancels_preview_and_discards_old_files_import_and_preview_replies() {
    let (ctx, mut app) = app();
    let cancel = Arc::new(AtomicBool::new(false));
    app.timeline.loading = Some((12, cancel.clone()));
    let old = app.timeline.document.as_ref().unwrap().project.clone();
    let new = super::super::media_picker::tests::project();
    let new_id = new.id.clone();
    app.timeline_message(
        &ctx,
        Event::Document {
            epoch: 9,
            replace: true,
            before: None,
            history: 0,
            result: Ok(json!({"project":new})),
        },
    );
    assert_eq!(app.timeline.epoch, 10);
    assert!(cancel.load(Ordering::Acquire));
    // An old native chooser cancellation must not clear a new operation's busy.
    app.timeline.busy = true;
    app.timeline_message(
        &ctx,
        Event::Files {
            epoch: 9,
            paths: vec![],
            target: super::super::media_picker::Target::StoryImage,
        },
    );
    app.timeline_message(
        &ctx,
        Event::Imported {
            epoch: 9,
            target: super::super::media_picker::Target::ReferenceImage("cast".into()),
            assets: vec![AssetId("image".into())],
            result: Ok(json!({"project":old})),
        },
    );
    app.timeline_message(
        &ctx,
        Event::Preview {
            epoch: 9,
            nonce: 12,
            asset: "old-own-source".into(),
            result: Ok(preview(40)),
        },
    );
    assert!(app.timeline.busy && app.timeline.player.is_none());
    assert_eq!(app.timeline.document.as_ref().unwrap().project.id, new_id);
}

#[test]
fn late_import_keeps_captured_removed_target_instead_of_binding_to_current_picker() {
    use super::super::media_picker::Target;
    for (target, asset) in [
        (Target::ReferenceImage("removed".into()), "image"),
        (
            Target::Generation {
                scene: SceneId("scene".into()),
                section: Some(SectionId("removed".into())),
                kind: MediaKind::Image,
            },
            "image",
        ),
        (Target::AudioClip(AudioClipId("removed".into())), "voice"),
    ] {
        let (ctx, mut app) = app();
        let p = app.timeline.document.as_ref().unwrap().project.clone();
        app.timeline_pick_media(Target::ReferenceImage("cast".into()));
        app.timeline.busy = true;
        app.timeline_message(
            &ctx,
            Event::Imported {
                epoch: 9,
                target,
                assets: vec![AssetId(asset.into())],
                result: Ok(json!({"project":p})),
            },
        );
        assert!(!app.timeline.busy);
        assert!(app.timeline.error.as_ref().unwrap().contains("removed"));
        assert!(
            app.timeline
                .document
                .as_ref()
                .unwrap()
                .project
                .creative
                .references[0]
                .images
                .is_empty()
        );
    }
}

#[test]
fn failed_partial_import_preserves_available_resources_without_applying_target_binding() {
    let (ctx, mut app) = app();
    let mut partial = app.timeline.document.as_ref().unwrap().project.clone();
    partial.project_revision += 1;
    app.timeline.busy = true;
    app.timeline_message(
        &ctx,
        Event::Imported {
            epoch: 9,
            target: super::super::media_picker::Target::ReferenceImage("cast".into()),
            assets: vec![AssetId("image".into())],
            result: Err(client::TimelineFailure {
                code: "invalid_media".into(),
                message: "Own second fixture is not decodable".into(),
                snapshot: Some(json!({"project":partial})),
                retryable: false,
            }),
        },
    );
    assert!(!app.timeline.busy);
    assert!(
        app.timeline
            .error
            .as_ref()
            .unwrap()
            .contains("not decodable")
    );
    let project = &app.timeline.document.as_ref().unwrap().project;
    assert_eq!(project.project_revision, 1);
    assert!(project.assets.contains_key(&AssetId("image".into())));
    assert!(project.creative.references[0].images.is_empty());
}
