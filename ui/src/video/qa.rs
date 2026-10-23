//! Opt-in native renderer/input QA. Only the fixture window is captured.
use super::*;
use serde::Deserialize;
use std::{collections::VecDeque, io::Read, time::Instant};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    destination: crate::video_destination::Config,
    actions: Vec<Action>,
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    WaitAssemblyPhase {
        phase: String,
    },
    WaitIdle,
    WaitJobs,
    WaitFrames {
        frames: u32,
    },
    WaitTime {
        milliseconds: u64,
    },
    WaitComparison {
        ready: bool,
    },
    ExpectComparison {
        playing: bool,
        audio: String,
        audio_child: bool,
        #[serde(default)]
        loops_min: u64,
        #[serde(default)]
        position_min: Option<f64>,
        #[serde(default)]
        position_max: Option<f64>,
    },
    Open {
        folder: PathBuf,
    },
    Save,
    Click {
        x: f32,
        y: f32,
    },
    ClickPixels {
        x: f32,
        y: f32,
    },
    ScrollPixels {
        x: f32,
        y: f32,
        delta_y: f32,
    },
    ClickControl {
        name: String,
    },
    DragControl {
        name: String,
        delta_x: f32,
    },
    WaitStatus {
        status: String,
        count: usize,
    },
    WaitRunStatus {
        status: String,
        count: usize,
    },
    WaitAuthoringPhase {
        phase: String,
    },
    Key {
        key: String,
        #[serde(default)]
        ctrl: bool,
        #[serde(default)]
        shift: bool,
    },
    Text {
        text: String,
    },
    Privacy {
        enabled: bool,
    },
    Focus {
        focused: bool,
    },
    Resize {
        width: f32,
        height: f32,
    },
    Screenshot {
        file: String,
    },
    Expect(Box<Expectations>),
    Quit,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expectations {
    #[serde(default)]
    compiled_contains: Option<String>,
    #[serde(default)]
    compiled_issues: Option<usize>,
    #[serde(default)]
    scene_prompts: Option<usize>,
    #[serde(default)]
    authoring_warnings: Option<usize>,
    #[serde(default)]
    authoring_contains: Option<String>,
    #[serde(default)]
    library_entries: Option<usize>,
    #[serde(default)]
    cast_references: Option<usize>,
    #[serde(default)]
    inventory_notes: Option<usize>,
    #[serde(default)]
    dialogue_sections: Option<usize>,
    #[serde(default)]
    image_sections: Option<usize>,
    #[serde(default)]
    imported_clips: Option<usize>,
    #[serde(default)]
    imported_speed: Option<f64>,
    #[serde(default)]
    imported_source_out: Option<f64>,
    #[serde(default)]
    assembly_error: Option<bool>,
    #[serde(default)]
    deliveries: Option<usize>,
    #[serde(default)]
    assembly_frames: Option<u64>,
    #[serde(default)]
    runs: Option<usize>,
    #[serde(default)]
    run_completed: Option<usize>,
    #[serde(default)]
    run_status: Option<Box<str>>,
    #[serde(default)]
    scenes: Option<usize>,
    #[serde(default)]
    takes: Option<usize>,
    #[serde(default)]
    accepted_takes: Option<usize>,
    #[serde(default)]
    outdated_takes: Option<usize>,
    #[serde(default)]
    endpoints: Option<usize>,
    #[serde(default)]
    motion_artifacts: Option<usize>,
    #[serde(default)]
    continuity_method: Option<Box<str>>,
    #[serde(default)]
    continuity_audio: Option<bool>,
    #[serde(default)]
    continuity: Option<bool>,
    #[serde(default)]
    continuity_source: Option<Box<str>>,
    #[serde(default)]
    scene_variant: Option<String>,
    #[serde(default)]
    boundary: Option<bool>,
    #[serde(default)]
    saved: Option<bool>,
    #[serde(default)]
    preview: Option<bool>,
    #[serde(default)]
    playing: Option<bool>,
    #[serde(default)]
    audio_running: Option<bool>,
    #[serde(default)]
    audio_clips: Option<usize>,
    #[serde(default)]
    audio_name: Option<Box<str>>,
    #[serde(default)]
    audio_generation: Option<bool>,
    #[serde(default)]
    audio_final_mix: Option<bool>,
    #[serde(default)]
    project_audio_native: Option<bool>,
    #[serde(default)]
    audio_locked: Option<bool>,
    #[serde(default)]
    audio_start: Option<f64>,
    #[serde(default)]
    audio_duration: Option<f64>,
    #[serde(default)]
    error: Option<bool>,
}
pub(in crate::app) struct Runner {
    plan: Plan,
    index: usize,
    started: Instant,
    delay: u32,
    input: VecDeque<Vec<egui::Event>>,
    focused: bool,
    pending: Option<String>,
    screenshots: Vec<String>,
    done: bool,
    layout: Value,
    wait_until: Option<Instant>,
    comparisons: Vec<Value>,
}
impl Runner {
    fn guard(&self) -> anyhow::Result<()> {
        let (_, fd) = crate::video_destination::inspect(
            &self.plan.destination,
            &crate::video_destination::System,
        )
        .map_err(anyhow::Error::msg)?;
        crate::video_destination::writable(&fd, 0).map_err(anyhow::Error::msg)
    }
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let mut bytes = vec![];
        std::fs::File::open(path)?
            .take(256 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(bytes.len() <= 256 * 1024, "QA plan too large");
        let plan: Plan = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            path.is_absolute() && path.starts_with(&plan.destination.root),
            "QA plan must be inside its own destination"
        );
        let runner = Self {
            plan,
            index: 0,
            started: Instant::now(),
            delay: 4,
            input: VecDeque::new(),
            focused: true,
            pending: None,
            screenshots: vec![],
            done: false,
            layout: Value::Null,
            wait_until: None,
            comparisons: vec![],
        };
        runner.guard()?;
        Ok(runner)
    }
    pub(in crate::app) fn input(&mut self, raw: &mut egui::RawInput) {
        // Preserve the renderer's reply; filtering it would leave a screenshot
        // request waiting forever while all project/API actions already passed.
        raw.events
            .retain(|event| matches!(event, egui::Event::Screenshot { .. }));
        raw.focused = self.focused;
        if let Some(events) = self.input.pop_front() {
            raw.events.extend(events);
        }
    }
    fn finish(&mut self, ok: bool, error: Option<String>) -> anyhow::Result<()> {
        self.guard()?;
        let report = json!({"passed":ok,"error":error,"actions_completed":self.index,"screenshots":self.screenshots,"layout":self.layout,"comparisons":self.comparisons,"seconds":self.started.elapsed().as_secs_f64(),"evidence":"native Glow-rendered own window with fixture input; own server API; no trained-model claim"});
        std::fs::write(
            self.plan.destination.root.join("native-qa-result.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        self.done = true;
        Ok(())
    }
    fn tick(&mut self, app: &mut ImagineApp, ctx: &egui::Context) -> anyhow::Result<()> {
        if self.done {
            return Ok(());
        }
        self.layout = json!({"pixels_per_point":ctx.pixels_per_point(),"logical_size":[ctx.content_rect().width(),ctx.content_rect().height()]});
        anyhow::ensure!(
            self.started.elapsed() < Duration::from_secs(120),
            "Native QA timed out at action {}",
            self.index
        );
        let screenshots = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|event| {
                    if let egui::Event::Screenshot {
                        user_data, image, ..
                    } = event
                    {
                        user_data
                            .data
                            .as_ref()
                            .and_then(|data| data.downcast_ref::<String>())
                            .map(|name| (name.clone(), image.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });
        for (name, image) in screenshots {
            if self.pending.as_deref() != Some(&name) {
                continue;
            }
            self.guard()?;
            let bytes = image
                .pixels
                .iter()
                .flat_map(|c| c.to_array())
                .collect::<Vec<_>>();
            image::save_buffer(
                self.plan.destination.root.join(&name),
                &bytes,
                image.width() as u32,
                image.height() as u32,
                image::ColorType::Rgba8,
            )?;
            self.screenshots.push(name);
            self.pending = None;
            self.index += 1;
            self.delay = 2;
        }
        ctx.request_repaint_after(Duration::from_millis(15));
        // Delay begins after injected input drains, so a release is rendered
        // before WaitIdle may observe the controller. Long drags span >6 frames.
        if !self.input.is_empty() {
            return Ok(());
        }
        if self.delay > 0 {
            self.delay -= 1;
            return Ok(());
        }
        if self.pending.is_some() {
            return Ok(());
        }
        let Some(action) = self.plan.actions.get(self.index) else {
            self.finish(true, None)?;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return Ok(());
        };
        match action {
            Action::WaitAssemblyPhase { phase } => {
                if app.timeline.assembly.progress["phase"] != json!(phase) {
                    return Ok(());
                }
            }
            Action::WaitIdle => {
                if app.timeline.busy
                    || app.timeline.assembly.busy
                    || app.timeline.admitting
                    || app.timeline.loading.is_some()
                    || app.connecting
                    || app.timeline.session.is_none()
                {
                    return Ok(());
                }
            }
            Action::WaitJobs => {
                if app.timeline.admitting || app.timeline.jobs.iter().any(|j| !j.terminal()) {
                    return Ok(());
                }
            }
            Action::WaitRunStatus { status, count } => {
                if app
                    .timeline
                    .runs
                    .iter()
                    .filter(|r| serde_json::to_value(r.status).unwrap() == json!(status))
                    .count()
                    != *count
                {
                    return Ok(());
                }
            }
            Action::WaitAuthoringPhase { phase } => {
                anyhow::ensure!(
                    app.timeline.authoring.failure.is_none(),
                    "Authoring request failed: {}",
                    app.timeline.authoring.failure.as_deref().unwrap_or("")
                );
                if app.timeline.authoring.busy
                    || app
                        .timeline
                        .authoring
                        .draft
                        .as_ref()
                        .is_none_or(|d| serde_json::to_value(d.phase).unwrap() != json!(phase))
                {
                    return Ok(());
                }
            }
            Action::WaitFrames { frames } => self.delay = *frames,
            Action::WaitTime { milliseconds } => {
                anyhow::ensure!(*milliseconds <= 10000, "QA wait is bounded to 10 seconds");
                let until = self
                    .wait_until
                    .get_or_insert_with(|| Instant::now() + Duration::from_millis(*milliseconds));
                if Instant::now() < *until {
                    return Ok(());
                }
                self.wait_until = None;
            }
            Action::WaitComparison { ready } => {
                if app.timeline.loading.is_some() || app.timeline.comparison.is_some() != *ready {
                    return Ok(());
                }
            }
            Action::ExpectComparison {
                playing,
                audio,
                audio_child,
                loops_min,
                position_min,
                position_max,
            } => {
                let comparison = app
                    .timeline
                    .comparison
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("No active A/B comparison"))?;
                let mut state = comparison.snapshot();
                state["project_revision"] = json!(
                    app.timeline
                        .document
                        .as_ref()
                        .map(|d| d.project.project_revision)
                );
                anyhow::ensure!(
                    state["playing"] == *playing
                        && state["audio"] == json!(audio)
                        && state["audio_child"] == *audio_child,
                    "Unexpected A/B playback/audio state: {state}"
                );
                anyhow::ensure!(
                    state["frames"][0] == state["frames"][1],
                    "A/B frames diverged: {state}"
                );
                anyhow::ensure!(
                    state["loops"].as_u64().unwrap() >= *loops_min,
                    "A/B loop did not wrap: {state}"
                );
                let position = state["position"].as_f64().unwrap();
                anyhow::ensure!(
                    position_min.is_none_or(|p| position >= p)
                        && position_max.is_none_or(|p| position <= p),
                    "Unexpected A/B position: {state}"
                );
                self.comparisons.push(state);
            }
            Action::ScrollPixels { x, y, delta_y } => {
                let scale = ctx.pixels_per_point();
                self.input.push_back(vec![
                    egui::Event::PointerMoved(egui::pos2(*x / scale, *y / scale)),
                    egui::Event::MouseWheel {
                        phase: egui::TouchPhase::Move,
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(0., *delta_y / scale),
                        modifiers: Default::default(),
                    },
                ]);
                self.delay = 8;
            }
            Action::WaitStatus { status, count } => {
                if app
                    .timeline
                    .jobs
                    .iter()
                    .filter(|j| j.observation["status"].as_str() == Some(status))
                    .count()
                    != *count
                {
                    return Ok(());
                }
            }
            Action::ClickControl { name } => {
                let Some((frame, rect, enabled)) = ctx.data(|d| {
                    d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                        "timeline-qa-control",
                        name,
                    )))
                }) else {
                    return Ok(());
                };
                if !enabled || ctx.cumulative_frame_nr().saturating_sub(frame) > 1 {
                    return Ok(());
                }
                let pos = rect.center();
                self.input.push_back(vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                ]);
                self.input.push_back(vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }]);
                self.delay = 4;
            }
            Action::DragControl { name, delta_x } => {
                let Some((frame, rect, enabled)) = ctx.data(|d| {
                    d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                        "timeline-qa-control",
                        name,
                    )))
                }) else {
                    return Ok(());
                };
                if !enabled || ctx.cumulative_frame_nr().saturating_sub(frame) > 1 {
                    return Ok(());
                }
                let start = rect.center();
                let end = start + egui::vec2(*delta_x, 0.);
                self.input.push_back(vec![
                    egui::Event::PointerMoved(start),
                    egui::Event::PointerButton {
                        pos: start,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                ]);
                for i in 1..=6 {
                    self.input.push_back(vec![egui::Event::PointerMoved(
                        start + (end - start) * (i as f32 / 6.),
                    )]);
                }
                self.input.push_back(vec![egui::Event::PointerButton {
                    pos: end,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }]);
                self.delay = 6;
            }
            Action::Open { folder } => {
                if app.timeline.busy || app.timeline.session.is_none() {
                    return Ok(());
                }
                app.timeline_folder(ctx, folder.clone(), FolderMode::Open);
            }
            Action::Save => app.timeline_save(ctx),
            Action::Privacy { enabled } => app.prefs.privacy_mode = *enabled,
            Action::Focus { focused } => {
                self.focused = *focused;
                self.delay = 3;
            }
            Action::Resize { width, height } => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                    *width, *height,
                )));
                self.delay = 8;
            }
            Action::Click { x, y } | Action::ClickPixels { x, y } => {
                let scale = if matches!(action, Action::ClickPixels { .. }) {
                    ctx.pixels_per_point()
                } else {
                    1.
                };
                let pos = egui::pos2(*x / scale, *y / scale);
                self.input.push_back(vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                ]);
                self.input.push_back(vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }]);
                self.delay = 4;
            }
            Action::Text { text } => {
                self.input.push_back(vec![egui::Event::Text(text.clone())]);
                self.delay = 4;
            }
            Action::Key { key, ctrl, shift } => {
                let key = match key.as_str() {
                    "A" => egui::Key::A,
                    "S" => egui::Key::S,
                    "Z" => egui::Key::Z,
                    "Enter" => egui::Key::Enter,
                    "Tab" => egui::Key::Tab,
                    "Escape" => egui::Key::Escape,
                    "Delete" => egui::Key::Delete,
                    "Space" => egui::Key::Space,
                    _ => anyhow::bail!("Unsupported QA key"),
                };
                let modifiers = egui::Modifiers {
                    ctrl: *ctrl,
                    command: *ctrl,
                    shift: *shift,
                    ..Default::default()
                };
                self.input.push_back(vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers,
                }]);
                self.input.push_back(vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers,
                }]);
                self.delay = 4;
            }
            Action::Screenshot { file } => {
                anyhow::ensure!(
                    file.ends_with(".png")
                        && !file.contains('/')
                        && !file.contains('\\')
                        && file != "..",
                    "Invalid QA screenshot name"
                );
                self.pending = Some(file.clone());
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                    file.clone(),
                )));
                return Ok(());
            }
            Action::Expect(expected) => {
                let Expectations {
                    compiled_contains,
                    compiled_issues,
                    scene_prompts,
                    authoring_warnings,
                    authoring_contains,
                    library_entries,
                    cast_references,
                    inventory_notes,
                    dialogue_sections,
                    image_sections,
                    imported_clips,
                    imported_speed,
                    imported_source_out,
                    assembly_error,
                    deliveries,
                    assembly_frames,
                    runs,
                    run_completed,
                    run_status,
                    scenes,
                    takes,
                    accepted_takes,
                    outdated_takes,
                    endpoints,
                    motion_artifacts,
                    continuity_method,
                    continuity_audio,
                    continuity,
                    continuity_source,
                    scene_variant,
                    boundary,
                    saved,
                    preview,
                    playing,
                    audio_running,
                    audio_clips,
                    audio_name,
                    audio_generation,
                    audio_final_mix,
                    project_audio_native,
                    audio_locked,
                    audio_start,
                    audio_duration,
                    error,
                } = expected.as_ref();
                if let Some(text) = authoring_contains {
                    anyhow::ensure!(
                        app.timeline
                            .authoring
                            .draft
                            .as_ref()
                            .and_then(|d| d.text.as_deref())
                            .is_some_and(|s| s.contains(text)),
                        "Authored source does not contain expected text"
                    );
                }
                if let Some(text) = compiled_contains {
                    anyhow::ensure!(
                        app.timeline
                            .compiled
                            .value
                            .as_ref()
                            .is_some_and(|p| p["prompt"]
                                .as_str()
                                .is_some_and(|p| p.contains(text))),
                        "Compiled source does not contain expected text: {text}"
                    );
                }
                if let Some(count) = compiled_issues {
                    anyhow::ensure!(
                        app.timeline.compiled.error.is_none()
                            && app.timeline.compiled.value.as_ref().is_some_and(|p| {
                                p["runtime_issues"]
                                    .as_array()
                                    .is_some_and(|a| a.len() == *count)
                            }),
                        "Unexpected compiled runtime issue count"
                    );
                }
                if let Some(count) = scene_prompts {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .creative
                            .scene_prompts
                            .len()
                            == *count),
                        "Unexpected structured scene prompt count"
                    );
                }
                if let Some(count) = authoring_warnings {
                    anyhow::ensure!(
                        app.timeline
                            .authoring
                            .draft
                            .as_ref()
                            .is_some_and(|d| d.warnings.len() == *count),
                        "Unexpected authoring warning count"
                    );
                }
                if let Some(count) = library_entries {
                    anyhow::ensure!(
                        app.timeline.library.entries() == *count,
                        "library entry count"
                    );
                }
                if let Some(count) = cast_references {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .creative
                            .references
                            .len()
                            == *count),
                        "Unexpected cast count"
                    );
                }
                if let Some(count) = inventory_notes {
                    anyhow::ensure!(
                        app.timeline
                            .authoring
                            .draft
                            .as_ref()
                            .is_some_and(|d| d.inventory.len() == *count),
                        "Unexpected inventory count"
                    );
                }
                if let Some(count) = dialogue_sections {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .creative
                            .sections
                            .values()
                            .filter(|s| s.dialogue_intent.is_some())
                            .count()
                            == *count),
                        "Unexpected dialogue binding count"
                    );
                }
                if let Some(count) = image_sections {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .creative
                            .sections
                            .values()
                            .filter(|s| !s.images.is_empty())
                            .count()
                            == *count),
                        "Unexpected section image binding count"
                    );
                }
                if let Some(value) = assembly_error {
                    anyhow::ensure!(
                        app.timeline.assembly.error.is_some() == *value,
                        "Unexpected assembly error"
                    );
                }
                if let Some(value) = imported_clips {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .assembly
                            .clips
                            .len()
                            == *value),
                        "Unexpected imported clip count"
                    );
                }
                if imported_speed.is_some() || imported_source_out.is_some() {
                    let clip = app
                        .timeline
                        .scene()
                        .and_then(|s| {
                            app.timeline
                                .document
                                .as_ref()
                                .and_then(|d| d.project.assembly.clips.get(&s.id))
                        })
                        .ok_or_else(|| anyhow::anyhow!("Imported clip missing"))?;
                    if let Some(value) = imported_speed {
                        anyhow::ensure!(
                            (clip.speed - value).abs() < 1e-6,
                            "Unexpected imported speed"
                        );
                    }
                    if let Some(value) = imported_source_out {
                        anyhow::ensure!(
                            clip.source_out
                                .is_some_and(|v| (super::editing::seconds(v) - value).abs() < 1e-6),
                            "Unexpected imported trim"
                        );
                    }
                }
                if let Some(count) = deliveries {
                    anyhow::ensure!(
                        app.timeline
                            .document
                            .as_ref()
                            .is_some_and(|d| d.raw["deliveries"]
                                .as_array()
                                .map(Vec::len)
                                .unwrap_or(0)
                                == *count),
                        "Unexpected delivery count"
                    );
                }
                if let Some(count) = assembly_frames {
                    anyhow::ensure!(
                        app.timeline
                            .assembly
                            .plan
                            .as_ref()
                            .is_some_and(|p| p.frames == *count),
                        "Unexpected assembly frame count"
                    );
                }
                if let Some(count) = runs {
                    anyhow::ensure!(app.timeline.runs.len() == *count, "Unexpected QA run count");
                }
                if let Some(count) = run_completed {
                    anyhow::ensure!(
                        app.timeline
                            .runs
                            .last()
                            .is_some_and(|r| r.completed() == *count),
                        "Unexpected QA run completion count"
                    );
                }
                if let Some(status) = run_status {
                    anyhow::ensure!(
                        app.timeline
                            .runs
                            .last()
                            .is_some_and(
                                |r| serde_json::to_value(r.status).unwrap() == json!(status)
                            ),
                        "Unexpected QA run status"
                    );
                }
                if let Some(playing) = playing {
                    anyhow::ensure!(
                        app.timeline.player.as_ref().is_some_and(|p| p.is_playing()) == *playing,
                        "Unexpected QA playback state"
                    );
                }
                if let Some(running) = audio_running {
                    anyhow::ensure!(
                        app.timeline
                            .player
                            .as_mut()
                            .is_some_and(|p| p.audio_running())
                            == *running,
                        "Unexpected QA audio process state"
                    );
                }
                if let Some(value) = error {
                    anyhow::ensure!(
                        app.timeline.error.is_some() == *value,
                        "Unexpected QA error state"
                    );
                }
                if let Some(preview) = preview {
                    anyhow::ensure!(
                        app.timeline.player.is_some() == *preview,
                        "Unexpected QA preview state"
                    );
                }
                let document = app
                    .timeline
                    .document
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No active QA project"))?;
                let clips = document
                    .project
                    .audio_tracks
                    .iter()
                    .flat_map(|t| &t.clips)
                    .collect::<Vec<_>>();
                if let Some(expected) = endpoints {
                    anyhow::ensure!(
                        document.raw["resources"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter(|resource| resource["kind"]["kind"] == "h3_endpoint")
                            .count()
                            == *expected,
                        "Unexpected QA endpoint count"
                    );
                }
                if let Some(expected) = motion_artifacts {
                    anyhow::ensure!(
                        document.raw["resources"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|r| r["kind"]["kind"] == "h3_motion")
                            .count()
                            == *expected,
                        "Unexpected motion artifact count"
                    );
                }
                if let Some(expected) = continuity_method {
                    let scene = document
                        .project
                        .scene(
                            app.timeline
                                .selected
                                .first()
                                .ok_or_else(|| anyhow::anyhow!("No Scene"))?,
                        )
                        .map_err(|_| anyhow::anyhow!("No Scene"))?;
                    anyhow::ensure!(
                        scene
                            .continuity
                            .as_ref()
                            .is_some_and(|c| serde_json::to_value(&c.method).unwrap().as_str()
                                == Some(expected.as_ref())),
                        "Unexpected continuity method"
                    );
                }
                if let Some(expected) = continuity_audio {
                    let scene = document
                        .project
                        .scene(
                            app.timeline
                                .selected
                                .first()
                                .ok_or_else(|| anyhow::anyhow!("No Scene"))?,
                        )
                        .map_err(|_| anyhow::anyhow!("No Scene"))?;
                    anyhow::ensure!(
                        scene
                            .continuity
                            .as_ref()
                            .is_some_and(|c| c.include_audio == *expected),
                        "Unexpected audio continuity intent"
                    );
                }
                if let Some(expected) = boundary {
                    anyhow::ensure!(
                        app.timeline.boundary == *expected,
                        "Unexpected boundary inspector state"
                    );
                }
                if let Some(scene) = app.timeline.scene() {
                    if let Some(expected) = continuity {
                        anyhow::ensure!(
                            scene.continuity.is_some() == *expected,
                            "Unexpected continuity selection"
                        );
                    }
                    if let Some(expected) = continuity_source {
                        anyhow::ensure!(
                            scene.continuity.as_ref().is_some_and(|value| value
                                .source_scene
                                .0
                                .as_str()
                                == expected.as_ref()),
                            "Unexpected continuity source"
                        );
                    }
                    if let Some(expected) = scene_variant {
                        anyhow::ensure!(
                            document.project.effective_model(scene).variant == *expected,
                            "Unexpected scene model variant"
                        );
                    }
                } else {
                    anyhow::ensure!(
                        continuity.is_none()
                            && continuity_source.is_none()
                            && scene_variant.is_none(),
                        "No QA scene selected"
                    );
                }
                for (expected, field) in
                    [(accepted_takes, "accepted"), (outdated_takes, "outdated")]
                {
                    if let Some(expected) = expected {
                        let actual = document.raw["take_states"]
                            .as_object()
                            .into_iter()
                            .flat_map(|states| states.values())
                            .filter(|state| state[field].as_bool() == Some(true))
                            .count();
                        anyhow::ensure!(actual == *expected, "Unexpected QA Take {field} count");
                    }
                }
                if let Some(value) = audio_locked {
                    anyhow::ensure!(
                        clips.first().is_some_and(|c| c.locked == *value),
                        "Unexpected audio lock"
                    );
                }
                if let Some(value) = audio_start {
                    anyhow::ensure!(
                        clips
                            .first()
                            .is_some_and(
                                |c| (super::editing::seconds(c.start) - *value).abs() < 1e-6
                            ),
                        "Unexpected audio start"
                    );
                }
                if let Some(value) = audio_duration {
                    anyhow::ensure!(
                        clips
                            .first()
                            .is_some_and(
                                |c| (super::editing::seconds(c.duration) - *value).abs() < 1e-6
                            ),
                        "Unexpected audio duration"
                    );
                }
                if let Some(value) = project_audio_native {
                    anyhow::ensure!(
                        document.project.audio.use_native_audio == *value,
                        "Unexpected native audio mix setting"
                    );
                }
                if let Some(n) = audio_clips {
                    anyhow::ensure!(clips.len() == *n, "Unexpected QA audio-clip count");
                }
                if let Some(name) = audio_name {
                    anyhow::ensure!(
                        clips
                            .first()
                            .is_some_and(|c| c.name.as_str() == name.as_ref()),
                        "Unexpected QA audio name"
                    );
                }
                if let Some(value) = audio_generation {
                    anyhow::ensure!(
                        clips
                            .first()
                            .is_some_and(|c| c.usage.generation_reference == *value),
                        "Unexpected generation routing"
                    );
                }
                if let Some(value) = audio_final_mix {
                    anyhow::ensure!(
                        clips.first().is_some_and(|c| c.usage.final_mix == *value),
                        "Unexpected mix routing"
                    );
                }
                if let Some(n) = scenes {
                    anyhow::ensure!(
                        document.project.scenes.len() == *n,
                        "Expected {n} scenes, got {}",
                        document.project.scenes.len()
                    );
                }
                if let Some(n) = takes {
                    anyhow::ensure!(
                        document.raw["takes"].as_array().map(Vec::len) == Some(*n),
                        "Unexpected QA Take count"
                    );
                }
                if let Some(saved) = saved {
                    anyhow::ensure!(
                        (document.raw["persistence"]["state"] == "saved") == *saved,
                        "Unexpected QA save state"
                    );
                }
            }
            Action::Quit => {
                self.index += 1;
                self.finish(true, None)?;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return Ok(());
            }
        }
        self.index += 1;
        Ok(())
    }
}
pub(in crate::app) fn control(ui: &egui::Ui, name: String, response: &egui::Response) {
    if ui.is_rect_visible(response.rect) {
        let value = (
            ui.ctx().cumulative_frame_nr(),
            response.rect.intersect(ui.clip_rect()),
            response.enabled(),
        );
        ui.ctx()
            .data_mut(|d| d.insert_temp(egui::Id::new(("timeline-qa-control", name)), value));
    }
}
impl ImagineApp {
    pub fn install_timeline_qa(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        self.qa = Some(Runner::load(path)?);
        self.workspace = Workspace::Video;
        self.queue.collapsed = true;
        Ok(())
    }
    pub(in crate::app) fn qa_tick(&mut self, ctx: &egui::Context) {
        let Some(mut qa) = self.qa.take() else {
            return;
        };
        if let Err(error) = qa.tick(self, ctx) {
            let _ = qa.finish(false, Some(error.to_string()));
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.qa = Some(qa);
    }
}
