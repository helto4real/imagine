//! Losing server handles must not discard the displayed project or uncommitted text.
use super::*;
impl Timeline {
    pub(super) fn session_ended(&mut self) {
        for run in &mut self.runs {
            if !run.status.terminal() {
                run.status = imagine_video_contract::runs::Status::Interrupted;
                run.failure = Some("run_session_ended".into());
            }
        }
        for job in &mut self.jobs {
            if !job.terminal() {
                job.observation["status"] = json!("interrupted");
                job.monitoring = false;
                job.acting = false;
                job.error = Some("Server session ended. No job was resumed.".into());
            }
        }
        self.clear_preview();
        self.thumbnails = Default::default();
        self.needs_reopen = self.document.is_some();
        self.session = None;
        self.session_checked = false;
        self.epoch = self.epoch.wrapping_add(1);
        self.pending_run = None;
        self.pending = None;
        self.pending_seed_advance = None;
        self.after_edit = None;
        self.busy = false;
        self.admitting = false;
        self.run_acting = false;
        self.runs_loading = false;
        self.assembly.project_changed();
    }
}
impl ImagineApp {
    pub(super) fn timeline_reopen(&mut self, ctx: &egui::Context) {
        if self.timeline.busy {
            return;
        }
        if self.timeline.session.is_none() {
            self.timeline.session_checked = false;
            self.timeline_session(ctx);
        } else if self.timeline.private_storage
            && self
                .timeline
                .document
                .as_ref()
                .is_some_and(|d| d.raw["storage_layout"] == 1)
        {
            let id = self.timeline.document.as_ref().unwrap().project.id.clone();
            self.timeline_request(
                ctx,
                true,
                "/v1/video/projects/open".into(),
                Some(json!({"schema_version":1,"project_id":id})),
                false,
                None,
            );
        } else if let Some(folder) = self.timeline.folder.clone() {
            self.timeline_folder(ctx, folder, FolderMode::Open);
        } else {
            self.timeline_choose_folder(ctx, FolderMode::Open);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn failure(code: &str) -> client::TimelineFailure {
        client::TimelineFailure {
            code: code.into(),
            message: "Own server session ended".into(),
            snapshot: None,
            retryable: false,
        }
    }
    #[test]
    fn session_change_from_runs_or_document_retains_project_draft_and_rebinds_without_new_project()
    {
        for from_runs in [true, false] {
            let ctx = egui::Context::default();
            let mut app = ImagineApp::synthetic_preview(
                &ctx,
                PathBuf::from(
                    "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01",
                ),
            );
            app.client = None;
            let p = super::super::media_picker::tests::project();
            let id = p.id.clone();
            let scene = p.scenes[0].id.clone();
            let section = p.scenes[0].sections[0].id.clone();
            app.timeline.document = Some(Snapshot {
                project: p.clone(),
                raw: json!({"project":p,"persistence":{"state":"saved"}}),
            });
            app.timeline.session = Some("old-own-session".into());
            app.timeline.folder = Some(PathBuf::from(
                "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/own-unit-folder",
            ));
            app.timeline.selected.insert(scene.clone());
            app.timeline.section = Some(section);
            app.timeline.sync_draft();
            app.timeline.draft_section = "Own uncommitted text".into();
            app.timeline.draft_global = "Own uncommitted global".into();
            let epoch = app.timeline.epoch;
            let folder = app.timeline.folder.clone();
            if from_runs {
                app.timeline_run_reply(
                    &ctx,
                    "old-own-session".into(),
                    runs::Reply::List,
                    Err(failure("session_changed")),
                );
            } else {
                app.timeline_message(
                    &ctx,
                    Event::Document {
                        epoch,
                        replace: false,
                        before: None,
                        history: 0,
                        result: Err(failure("session_changed")),
                    },
                );
            }
            assert!(app.timeline.needs_reopen && app.timeline.session.is_none());
            assert_eq!(app.timeline.document.as_ref().unwrap().project.id, id);
            assert_eq!(app.timeline.folder, folder);
            assert_eq!(app.timeline.draft_section, "Own uncommitted text");
            assert!(app.timeline.selected.contains(&scene));
            app.timeline_message(&ctx, Event::Session(Ok("new-own-session".into())));
            assert!(app.timeline.needs_reopen);
            assert_eq!(app.timeline.document.as_ref().unwrap().project.id, id);
            let epoch = app.timeline.epoch;
            app.timeline_message(
                &ctx,
                Event::Document {
                    epoch,
                    replace: false,
                    before: None,
                    history: 0,
                    result: Err(failure("storage_unavailable")),
                },
            );
            assert!(app.timeline.needs_reopen);
            assert_eq!(app.timeline.draft_section, "Own uncommitted text");
            app.timeline_message(
                &ctx,
                Event::Document {
                    epoch,
                    replace: false,
                    before: None,
                    history: 0,
                    result: Ok(json!({"project":p,"persistence":{"state":"saved"}})),
                },
            );
            assert!(!app.timeline.needs_reopen);
            assert_eq!(app.timeline.document.as_ref().unwrap().project.id, id);
            assert_eq!(app.timeline.draft_section, "Own uncommitted text");
            assert_eq!(app.timeline.draft_global, "Own uncommitted global");
        }
    }
}
