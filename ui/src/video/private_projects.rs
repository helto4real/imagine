//! Saved private projects, including agent workspace projects, opened through
//! the same location flow as a picked folder; and the agent workspaces with
//! their keys, which the user hands to an agent to continue its work.
use super::*;

#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    loading: bool,
    projects: Option<Result<Vec<Value>, String>>,
    workspaces: Option<Result<Vec<Value>, String>>,
    /// The workspace whose key replacement awaits confirmation.
    replacing: Option<String>,
    /// The last key action's result, shown under the workspaces.
    notice: Option<Result<String, String>>,
}
pub(super) enum Choice {
    Open(PathBuf),
    Browse,
    Refresh,
    CopyKey(String),
    ReplaceKey(String),
}
fn age(saved_ms: Option<u64>) -> String {
    let Some(saved) = saved_ms else {
        return String::new();
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let minutes = now.saturating_sub(saved) / 60_000;
    match minutes {
        0 => "just now".into(),
        1..=59 => format!("{minutes} min ago"),
        60..=1439 => format!("{} h ago", minutes / 60),
        _ => format!("{} d ago", minutes / 1440),
    }
}
fn length(seconds: f64) -> String {
    let seconds = seconds.max(0.).round() as u64;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}
impl State {
    pub(super) fn reply(&mut self, result: Result<Value, client::TimelineFailure>) {
        self.loading = false;
        self.projects = Some(
            result
                .map(|value| value["projects"].as_array().cloned().unwrap_or_default())
                .map_err(|failure| failure.message),
        );
    }
    pub(super) fn workspaces_reply(&mut self, result: Result<Value, client::TimelineFailure>) {
        self.workspaces = Some(
            result
                .map(|value| value["workspaces"].as_array().cloned().unwrap_or_default())
                .map_err(|failure| failure.message),
        );
    }
    /// A fetched key: copied to the clipboard, never shown.
    pub(super) fn key_reply(
        &mut self,
        ctx: &egui::Context,
        replaced: bool,
        result: Result<Value, client::TimelineFailure>,
    ) {
        self.notice = Some(match result {
            Ok(value) => match value["key"].as_str() {
                Some(key) => {
                    ctx.copy_text(key.to_owned());
                    Ok(if replaced {
                        "New key copied. The old key no longer works.".into()
                    } else {
                        "Agent key copied.".into()
                    })
                }
                None => Err("The video service returned no key.".into()),
            },
            Err(failure) => Err(failure.message),
        });
    }
    pub(super) fn window(&mut self, ctx: &egui::Context, privacy: bool) -> Option<Choice> {
        if !self.open {
            return None;
        }
        let mut open = true;
        let mut choice = None;
        egui::Window::new("Open private project")
            .id(egui::Id::new("private-projects"))
            .open(&mut open)
            .default_width(560.)
            .default_height(420.)
            .show(ctx, |ui| {
                let _canvas = helto::window_canvas(ui);
                let hidden = !helto::shown(ui, privacy);
                egui::ScrollArea::vertical()
                    .id_salt("private-projects-scroll")
                    .auto_shrink([false, true])
                    .max_height(360.)
                    .show(ui, |ui| match &self.projects {
                        None => {
                            ui.label("Loading projects…");
                        }
                        Some(Err(error)) => {
                            ui.colored_label(helto::ERROR, error);
                        }
                        Some(Ok(projects)) if projects.is_empty() => {
                            ui.label("No saved private projects.");
                        }
                        Some(Ok(projects)) => {
                            for (i, project) in projects.iter().enumerate() {
                                ui.push_id(i, |ui| {
                                    ui.horizontal(|ui| {
                                        let failed = project["error"].as_str();
                                        let r = ui.add_enabled(
                                            failed.is_none(),
                                            egui::Button::new("Open"),
                                        );
                                        #[cfg(feature = "ui-preview")]
                                        super::qa::control(
                                            ui,
                                            format!("private-project-open-{i}"),
                                            &r,
                                        );
                                        if r.clicked()
                                            && let Some(path) = project["path"].as_str()
                                        {
                                            choice = Some(Choice::Open(PathBuf::from(path)));
                                        }
                                        // Folder names show; scene titles are private.
                                        ui.scope(|ui| {
                                            ui.vertical(|ui| {
                                                let title = project["first_scene_title"]
                                                    .as_str()
                                                    .filter(|_| !hidden)
                                                    .unwrap_or_default();
                                                let folder =
                                                    project["folder"].as_str().unwrap_or_default();
                                                ui.strong(if title.is_empty() {
                                                    folder
                                                } else {
                                                    title
                                                });
                                                let mut details = vec![];
                                                if !title.is_empty() {
                                                    details.push(folder.to_owned());
                                                }
                                                if let Some(code) = failed {
                                                    details.push(client::error_message(code));
                                                } else {
                                                    let scenes = project["scene_count"]
                                                        .as_u64()
                                                        .unwrap_or(0);
                                                    details.push(format!(
                                                        "{scenes} scene{}",
                                                        if scenes == 1 { "" } else { "s" }
                                                    ));
                                                    details.push(length(
                                                        project["duration_seconds"]
                                                            .as_f64()
                                                            .unwrap_or(0.),
                                                    ));
                                                }
                                                let saved = age(project["saved_at_ms"].as_u64());
                                                if !saved.is_empty() {
                                                    details.push(saved);
                                                }
                                                if project["source"] == "agent" {
                                                    let workspace = project["workspace"]["title"]
                                                        .as_str()
                                                        .filter(|t| !t.is_empty());
                                                    details.push(match workspace {
                                                        Some(title) => format!("Agent · {title}"),
                                                        None => "Agent".into(),
                                                    });
                                                }
                                                if project["open"] == true {
                                                    details.push("Open in this session".into());
                                                }
                                                ui.weak(details.join(" · "));
                                            });
                                        });
                                    });
                                });
                                ui.separator();
                            }
                        }
                    });
                self.workspaces_section(ui, &mut choice);
                ui.horizontal(|ui| {
                    let r = helto::icon_button(ui, crate::icons::Icon::Folder, "Browse…", false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "private-projects-browse".into(), &r);
                    if r.clicked() {
                        choice = Some(Choice::Browse);
                    }
                    let r = helto::icon_button(ui, crate::icons::Icon::Retry, "Refresh", false);
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "private-projects-refresh".into(), &r);
                    if r.clicked() && !self.loading {
                        choice = Some(Choice::Refresh);
                    }
                });
            });
        self.open = open && !matches!(choice, Some(Choice::Open(_) | Choice::Browse));
        choice
    }
    fn workspaces_section(&mut self, ui: &mut egui::Ui, choice: &mut Option<Choice>) {
        let Some(Ok(workspaces)) = &self.workspaces else {
            return;
        };
        if workspaces.is_empty() {
            return;
        }
        ui.add_space(6.);
        ui.label(helto::section("Agent workspaces"));
        egui::ScrollArea::vertical()
            .id_salt("agent-workspaces-scroll")
            .auto_shrink([false, true])
            .max_height(180.)
            .show(ui, |ui| {
                for (i, workspace) in workspaces.iter().enumerate() {
                    let id = workspace["workspace_id"].as_str().unwrap_or_default();
                    ui.push_id(i, |ui| {
                        ui.horizontal(|ui| {
                            let r = helto::icon_button(ui, crate::icons::Icon::Copy, "Copy agent key", false);
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, format!("workspace-copy-key-{i}"), &r);
                            if r.clicked() {
                                *choice = Some(Choice::CopyKey(id.to_owned()));
                            }
                            let r = helto::icon_button(ui, crate::icons::Icon::Retry, "Replace key", false);
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, format!("workspace-replace-key-{i}"), &r);
                            if r.clicked() {
                                self.replacing = Some(id.to_owned());
                            }
                            ui.vertical(|ui| {
                                let title = workspace["title"].as_str().unwrap_or_default();
                                let folder = workspace["folder"].as_str().unwrap_or_default();
                                ui.strong(if title.is_empty() { folder } else { title });
                                ui.weak(folder);
                            });
                        });
                        if self.replacing.as_deref() == Some(id) {
                            let asked = helto::confirm(
                                ui,
                                "Replace this workspace's key? Agents using the old key lose access.",
                                (crate::icons::Icon::Retry, "Replace key"),
                                "Keep key",
                                None,
                            );
                            if asked.yes.clicked() {
                                *choice = Some(Choice::ReplaceKey(id.to_owned()));
                                self.replacing = None;
                            }
                            if asked.no.clicked() {
                                self.replacing = None;
                            }
                        }
                    });
                    ui.separator();
                }
            });
        match &self.notice {
            Some(Ok(text)) => {
                ui.colored_label(helto::OK, text);
            }
            Some(Err(text)) => {
                ui.colored_label(helto::ERROR, text);
            }
            None => {}
        }
    }
}
impl ImagineApp {
    /// The project list in private storage; otherwise the folder picker.
    pub(super) fn timeline_open_project(&mut self, ctx: &egui::Context) {
        if !self.timeline.private_storage {
            self.timeline_choose_folder(ctx, FolderMode::Open);
            return;
        }
        self.timeline.private_projects.open = true;
        self.timeline_private_projects_fetch(ctx);
    }
    fn timeline_private_projects_fetch(&mut self, ctx: &egui::Context) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        let state = &mut self.timeline.private_projects;
        state.loading = true;
        state.projects = None;
        state.workspaces = None;
        let (epoch, tx, ctx) = (self.timeline.epoch, self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                "/v1/video/projects/private",
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::PrivateProjects { epoch, result }),
            );
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                "/v1/workspaces",
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Workspaces { epoch, result }),
            );
        });
    }
    /// Fetches (or replaces) a workspace key and copies it.
    fn timeline_workspace_key(&mut self, ctx: &egui::Context, id: String, replace: bool) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.private_projects.notice = None;
        let (epoch, tx, ctx) = (self.timeline.epoch, self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                replace,
                &format!("/v1/workspaces/{id}/key"),
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::WorkspaceKey {
                    epoch,
                    replaced: replace,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_private_projects_window(&mut self, ctx: &egui::Context) {
        match self
            .timeline
            .private_projects
            .window(ctx, self.prefs.privacy_mode)
        {
            Some(Choice::Open(path)) => self.timeline_folder(ctx, path, FolderMode::Open),
            Some(Choice::Browse) => self.timeline_choose_folder(ctx, FolderMode::Open),
            Some(Choice::Refresh) => self.timeline_private_projects_fetch(ctx),
            Some(Choice::CopyKey(id)) => self.timeline_workspace_key(ctx, id, false),
            Some(Choice::ReplaceKey(id)) => self.timeline_workspace_key(ctx, id, true),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn painted(shape: &egui::Shape, needle: &str) -> bool {
        match shape {
            egui::Shape::Text(t) => t.opacity_factor > 0. && t.galley.text().contains(needle),
            egui::Shape::Vec(shapes) => shapes.iter().any(|s| painted(s, needle)),
            _ => false,
        }
    }
    #[test]
    fn rows_show_folder_scenes_length_and_age_and_hide_titles_in_privacy_mode() {
        assert_eq!(length(65.4), "1:05");
        assert_eq!(age(None), "");
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let mut state = State {
            open: true,
            loading: false,
            workspaces: Some(Ok(vec![
                json!({"workspace_id":"ws_own","title":"OWN-workspace","folder":"2026-10-25/1432-own"}),
            ])),
            replacing: None,
            notice: None,
            projects: Some(Ok(vec![
                json!({"folder":"OWN-folder","first_scene_title":"SYNTHETIC_PRIVATE title",
                    "scene_count":3,"duration_seconds":20.04,"path":"/synthetic/own","source":"agent","open":false}),
                json!({"folder":"OWN-broken","error":"invalid_manifest","path":"/synthetic/broken"}),
            ])),
        };
        for privacy in [false, true, false] {
            let (mut title, mut folders, mut workspace) = (false, false, false);
            for _ in 0..3 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(800., 600.),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        assert!(state.window(ui.ctx(), privacy).is_none());
                    },
                );
                output.textures_delta.clear();
                title = output
                    .shapes
                    .iter()
                    .any(|s| painted(&s.shape, "SYNTHETIC_PRIVATE"));
                folders = ["OWN-folder", "OWN-broken"]
                    .iter()
                    .all(|f| output.shapes.iter().any(|s| painted(&s.shape, f)));
                workspace = output
                    .shapes
                    .iter()
                    .any(|s| painted(&s.shape, "OWN-workspace"));
            }
            // Scene titles are private; folder and workspace names are not.
            assert_eq!(title, !privacy);
            assert!(folders, "privacy={privacy}");
            assert!(workspace, "privacy={privacy}");
        }
        assert!(state.open);
    }
    #[test]
    fn a_fetched_key_is_copied_never_shown() {
        let ctx = egui::Context::default();
        let mut state = State::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            state.key_reply(
                ui.ctx(),
                true,
                Ok(json!({"key":"imk-ownownownownownownownownow"})),
            );
        });
        output.textures_delta.clear();
        assert!(output.platform_output.commands.iter().any(|c| matches!(
            c,
            egui::OutputCommand::CopyText(text) if text == "imk-ownownownownownownownownow"
        )));
        assert_eq!(
            state.notice,
            Some(Ok("New key copied. The old key no longer works.".into()))
        );
    }
}
