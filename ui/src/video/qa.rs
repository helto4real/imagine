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
    PrepareAnimateImage {
        path: PathBuf,
    },
    WaitSamplingPreview {
        minimum_sequence: u64,
    },
    ExpectSamplingPreview {
        label: String,
        has_clip: bool,
        has_texture: bool,
    },
    SignalFixture {
        name: String,
    },
    WaitProjectSession {
        attached: bool,
        project: String,
        draft: String,
    },
    WaitJobFailure {
        code: String,
        message_contains: String,
    },
    WritePromptCatalogFixture {
        path: PathBuf,
        text: String,
    },
    ExpectCatalog {
        stale: bool,
        error_contains: String,
    },
    ExportCompatible {
        folder: PathBuf,
    },
    ExpectProject {
        pointer: String,
        value: Value,
    },
    ExpectErrorContains {
        text: String,
    },
    ExpectLibraryError {
        text: String,
    },
    ExpectAuthoringError {
        text: String,
    },
    ScrollToControl {
        scroll: String,
        name: String,
    },
    ScrollControl {
        name: String,
        #[serde(default)]
        delta_x: f32,
        delta_y: f32,
    },
    ObserveScroll {
        name: String,
        label: String,
        #[serde(default)]
        compare_to: Option<String>,
        #[serde(default)]
        minimum_y: f32,
        #[serde(default)]
        visible_control: Option<String>,
    },
    ExpectCombo {
        name: String,
        open: bool,
    },
    OpenLibrary {
        folder: PathBuf,
    },
    OpenPromptCatalog {
        folder: PathBuf,
    },
    SavePromptCatalog {
        path: PathBuf,
    },
    ImportPicker {
        paths: Vec<PathBuf>,
    },
    WaitAssemblyPhase {
        phase: String,
    },
    WaitFinalFailure {
        code: String,
    },
    ExpectNavigation {
        start: f64,
        end: f64,
        display: String,
    },
    ExpectSplitSelection {
        previous: String,
    },
    ExpectSelectionPositions {
        positions: Vec<(f64, f64)>,
        audio: usize,
    },
    ExpectSelectionItems {
        items: Vec<String>,
    },
    ExpectSelectedScene {
        start: f64,
        duration: f64,
        sections: usize,
    },
    ExpectSectionLabel {
        section: SectionId,
        text: String,
    },
    WaitWaveform {
        clip: String,
        visible: bool,
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
        #[serde(default)]
        ids: Option<[String; 2]>,
    },
    WaitTakeStatus {
        take: String,
        status: String,
        accepted: bool,
    },
    WaitCapture {
        take: String,
        exists: bool,
        attached: bool,
        #[serde(default)]
        removed: bool,
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
        #[serde(default)]
        expected_enabled: Option<bool>,
        #[serde(default)]
        ctrl: bool,
        #[serde(default)]
        shift: bool,
        #[serde(default)]
        secondary: bool,
    },
    HoverControl {
        name: String,
    },
    DragControl {
        name: String,
        delta_x: f32,
        #[serde(default)]
        ctrl: bool,
        #[serde(default)]
        shift: bool,
    },
    WaitStatus {
        status: String,
        count: usize,
    },
    WaitImageJobs {
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
    WaitAuthoringProgress {
        completed: u64,
        eta: bool,
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
    ZoomUi {
        factor: f32,
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
    section_prompt: Option<String>,
    #[serde(default)]
    sections: Option<usize>,
    #[serde(default)]
    selected_sections: Option<Vec<String>>,
    #[serde(default)]
    compiled_contains: Option<String>,
    #[serde(default)]
    compiled_excludes: Option<String>,
    #[serde(default)]
    compiled_issues: Option<usize>,
    #[serde(default)]
    compiled_seed: Option<u64>,
    #[serde(default)]
    compiled_stale: Option<bool>,
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
    cast_images: Option<usize>,
    #[serde(default)]
    story_images: Option<usize>,
    #[serde(default)]
    inventory_notes: Option<usize>,
    #[serde(default)]
    dialogue_sections: Option<usize>,
    #[serde(default)]
    image_sections: Option<usize>,
    #[serde(default)]
    video_sections: Option<usize>,
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
    privacy_captures: std::collections::BTreeMap<String, Value>,
    control_wait: Option<(String, Instant)>,
    scroll_motion: Option<(String, egui::Rect, Instant)>,
    job_ids: Vec<String>,
    final_diagnostics: Value,
    final_failures: Vec<Value>,
    navigation_checks: Vec<Value>,
    scroll_checks: std::collections::BTreeMap<String, Value>,
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
            privacy_captures: Default::default(),
            control_wait: None,
            scroll_motion: None,
            job_ids: vec![],
            final_diagnostics: Value::Null,
            final_failures: vec![],
            navigation_checks: vec![],
            scroll_checks: Default::default(),
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
        raw.events
            .push(egui::Event::ModifiersChanged(Default::default()));
        if let Some(events) = self.input.pop_front() {
            for event in &events {
                if let egui::Event::PointerButton { modifiers, .. }
                | egui::Event::Key { modifiers, .. } = event
                {
                    raw.events.push(egui::Event::ModifiersChanged(*modifiers));
                }
            }
            raw.events.extend(events);
        }
    }
    fn finish(&mut self, ok: bool, error: Option<String>) -> anyhow::Result<()> {
        self.guard()?;
        if self
            .plan
            .actions
            .iter()
            .any(|action| matches!(action, Action::ObserveScroll { .. }))
        {
            // The image scroll harness consumes only this geometry report.
            // It must not parse the general runner's job/session diagnostics.
            let geometry = json!({"passed":ok,"actions_completed":self.index,
                "scroll_checks":self.scroll_checks,"layout":self.layout});
            std::fs::write(
                self.plan.destination.root.join("native-scroll-result.json"),
                serde_json::to_vec_pretty(&geometry)?,
            )?;
        }
        let report = json!({"passed":ok,"error":error,"job_ids":self.job_ids,"final_diagnostics":self.final_diagnostics,"final_failures":self.final_failures,"navigation_checks":self.navigation_checks,"scroll_checks":self.scroll_checks,"privacy_captures":self.privacy_captures,"actions_completed":self.index,"screenshots":self.screenshots,"layout":self.layout,"comparisons":self.comparisons,"seconds":self.started.elapsed().as_secs_f64(),"evidence":"native Glow-rendered own window with fixture input; own server API; no trained-model claim"});
        std::fs::write(
            self.plan.destination.root.join("native-qa-result.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        self.done = true;
        Ok(())
    }
    fn tick(&mut self, app: &mut ImagineApp, ctx: &egui::Context) -> anyhow::Result<()> {
        self.job_ids = app.timeline.jobs.iter().map(|job| job.id.clone()).collect();
        self.final_diagnostics = app.timeline_final_diagnostics();
        if self.done {
            return Ok(());
        }
        self.layout = json!({"pixels_per_point":ctx.pixels_per_point(),"logical_size":[ctx.content_rect().width(),ctx.content_rect().height()],"pixels_per_point":ctx.pixels_per_point()});
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
            let privacy = ctx
                .data(|d| {
                    d.get_temp::<std::collections::BTreeMap<String, (u64, bool, bool, bool)>>(
                        egui::Id::new("qa-text-privacy"),
                    )
                })
                .unwrap_or_default();
            self.privacy_captures.insert(name.clone(),json!({"fields":privacy,"frame":ctx.cumulative_frame_nr(),"input_focused":ctx.input(|i|i.focused),"logical_size":[ctx.content_rect().width(),ctx.content_rect().height()],"pixels_per_point":ctx.pixels_per_point()}));
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
            Action::PrepareAnimateImage { path } => {
                let path = path.canonicalize()?;
                anyhow::ensure!(
                    path.starts_with(
                        "/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"
                    ),
                    "Own Animate fixture required"
                );
                let image = image::open(&path)?.to_rgba8();
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width() as usize, image.height() as usize],
                    image.as_raw(),
                );
                app.image.result = Some(Results {
                    assets: vec![ResultView {
                        asset_id: "asset_own_animate_fixture".into(),
                        role: OutputRole::Image,
                        texture: ctx.load_texture("own-animate", color, Default::default()),
                        file: path,
                    }],
                    seed: None,
                });
                app.workspace = Workspace::Image;
            }
            Action::WaitSamplingPreview { minimum_sequence } => {
                if app.sampling_preview.qa_state()["sequence"]
                    .as_u64()
                    .unwrap_or(0)
                    < *minimum_sequence
                {
                    return Ok(());
                }
            }
            Action::ExpectSamplingPreview {
                label,
                has_clip,
                has_texture,
            } => {
                let state = app.sampling_preview.qa_state();
                anyhow::ensure!(
                    state["has_clip"] == *has_clip && state["has_texture"] == *has_texture,
                    "Unexpected sampling preview state: {state}"
                );
                self.privacy_captures
                    .insert(format!("sampling-preview-{label}"), state);
            }
            Action::SignalFixture { name } => {
                anyhow::ensure!(
                    !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric()),
                    "Invalid own fixture signal"
                );
                std::fs::write(
                    self.plan.destination.root.join(format!("{name}.signal")),
                    b"ready",
                )?;
            }
            Action::WaitProjectSession {
                attached,
                project,
                draft,
            } => {
                if app.timeline.session.is_none() || app.timeline.needs_reopen == *attached {
                    return Ok(());
                }
                anyhow::ensure!(
                    app.timeline
                        .document
                        .as_ref()
                        .is_some_and(|d| d.project.id.0 == *project),
                    "Displayed project was lost or replaced on session change"
                );
                anyhow::ensure!(
                    app.timeline.draft_section == *draft,
                    "Uncommitted text was replaced on session change"
                );
                self.navigation_checks.push(json!({"project_session_attached":attached,"project_preserved":true,"draft_preserved":true}));
            }
            Action::WaitJobFailure {
                code,
                message_contains,
            } => {
                let Some(job) = app.timeline.jobs.iter().find(|j| {
                    j.status() == super::jobs::JobStatus::Failed
                        && j.observation["error"]["code"] == json!(code)
                }) else {
                    return Ok(());
                };
                anyhow::ensure!(
                    job.failure_message()
                        .is_some_and(|message| message.contains(message_contains)),
                    "Specific worker failure message was not projected to the UI"
                );
                self.navigation_checks
                    .push(json!({"worker_failure_code":code,"specific_message":true}));
            }
            Action::ExpectLibraryError { text } => {
                anyhow::ensure!(
                    app.timeline
                        .library
                        .error
                        .as_ref()
                        .is_some_and(|error| error.contains(text)),
                    "Expected library error {text}: {:?}",
                    app.timeline.library.error
                );
            }
            Action::ExpectAuthoringError { text } => {
                anyhow::ensure!(
                    app.timeline
                        .authoring
                        .failure
                        .as_ref()
                        .is_some_and(|error| error.contains(text)),
                    "Expected proposal save error {text}: {:?}",
                    app.timeline.authoring.failure
                );
            }
            Action::ExpectErrorContains { text } => {
                anyhow::ensure!(
                    app.timeline
                        .error
                        .as_ref()
                        .is_some_and(|error| error.contains(text)),
                    "Own UI error did not contain {text}"
                );
            }
            Action::WritePromptCatalogFixture { path, text } => {
                let path = path.canonicalize()?;
                anyhow::ensure!(
                    path.starts_with(&self.plan.destination.root)
                        && path.extension().is_some_and(|x| x == "yaml"),
                    "Only existing own YAML catalog fixtures may be changed"
                );
                std::fs::write(path, text)?;
            }
            Action::ExpectCatalog {
                stale,
                error_contains,
            } => {
                app.timeline
                    .catalog
                    .qa_check(*stale, error_contains)
                    .map_err(anyhow::Error::msg)?;
            }
            Action::ExportCompatible { folder } => {
                anyhow::ensure!(
                    folder.starts_with(&self.plan.destination.root),
                    "Only own compatibility fixtures may be exported"
                );
                app.timeline.compatibility_open = false;
                app.timeline_folder(ctx, folder.clone(), FolderMode::ExportCompatible);
            }
            Action::ExpectProject { pointer, value } => {
                let document = app
                    .timeline
                    .document
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No own fixture project open"))?;
                let project = serde_json::to_value(&document.project)?;
                anyhow::ensure!(
                    project.pointer(pointer) == Some(value),
                    "Unexpected own fixture project field {pointer}: {:?}",
                    project.pointer(pointer)
                );
            }
            Action::WaitAssemblyPhase { phase } => {
                if app.timeline.assembly.progress["phase"] != json!(phase) {
                    return Ok(());
                }
            }
            Action::ExpectSelectionItems { items } => {
                let p = &app
                    .timeline
                    .document
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No own fixture"))?
                    .project;
                let actual = app
                    .timeline
                    .selection_items(p)
                    .iter()
                    .map(|i| match i {
                        super::selection::Item::Scene(id) => format!("scene:{}", id.0),
                        super::selection::Item::Section(id) => format!("section:{}", id.0),
                        super::selection::Item::Audio(id) => format!("audio:{}", id.0),
                    })
                    .collect::<Vec<_>>();
                anyhow::ensure!(
                    &actual == items,
                    "Expected ordered selection {items:?}, got {actual:?}"
                );
            }
            Action::ExpectSectionLabel { section, text } => {
                let p = &app
                    .timeline
                    .document
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No own fixture"))?
                    .project;
                let source = p
                    .scenes
                    .iter()
                    .flat_map(|s| &s.sections)
                    .find(|s| &s.id == section)
                    .ok_or_else(|| anyhow::anyhow!("No own Section"))?;
                anyhow::ensure!(
                    super::navigation::section_label(p, source, app.timeline.navigation.display)
                        == *text,
                    "Unexpected own Section display label"
                );
            }
            Action::ExpectSelectedScene {
                start,
                duration,
                sections,
            } => {
                let scene = app
                    .timeline
                    .scene()
                    .ok_or_else(|| anyhow::anyhow!("No selected own Scene"))?;
                anyhow::ensure!(
                    (super::editing::seconds(scene.start) - start).abs() < 1e-6
                        && (super::editing::seconds(scene.duration) - duration).abs() < 1e-6
                        && scene.sections.len() == *sections,
                    "Unexpected selected own Scene geometry"
                );
            }
            Action::ExpectSelectionPositions { positions, audio } => {
                let p = &app
                    .timeline
                    .document
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No own project"))?
                    .project;
                let mut actual = p
                    .scenes
                    .iter()
                    .flat_map(|s| {
                        s.sections
                            .iter()
                            .filter(|v| app.timeline.section_selection.contains(&v.id))
                            .map(move |v| {
                                let a = super::editing::seconds(s.start)
                                    + super::editing::seconds(v.start);
                                (a, a + super::editing::seconds(v.duration))
                            })
                    })
                    .collect::<Vec<_>>();
                actual.sort_by(|a, b| a.0.total_cmp(&b.0));
                anyhow::ensure!(
                    actual.len() == positions.len()
                        && actual
                            .iter()
                            .zip(positions)
                            .all(|(a, b)| (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5)
                        && app.timeline.audio_selected.len() == *audio,
                    "Unexpected grouped selection: {:?} / {}",
                    actual,
                    app.timeline.audio_selected.len()
                );
                self.navigation_checks
                    .push(json!({"selected_positions":actual,"selected_audio":audio}));
            }
            Action::WaitWaveform { clip, visible } => {
                let value =
                    ctx.data(|d| d.get_temp::<Value>(egui::Id::new(("waveform-painted", clip))));
                let Some(value) = value else { return Ok(()) };
                if value["visible"] != json!(visible) {
                    if *visible {
                        anyhow::ensure!(
                            !app.timeline.waveform.values.values().any(|v| v.is_err()),
                            "Own waveform request failed"
                        );
                    }
                    return Ok(());
                }
                if *visible {
                    anyhow::ensure!(
                        value["count"].as_u64().is_some_and(|v| v >= 16)
                            && value["maximum"].as_f64().is_some_and(|v| v > 0.),
                        "Own waveform was not drawn"
                    );
                }
                self.navigation_checks.push(json!({"waveform":value}));
            }
            Action::ExpectNavigation {
                start,
                end,
                display,
            } => {
                let actual = app.timeline.navigation.view;
                anyhow::ensure!(
                    actual
                        .is_some_and(|(a, b)| (a - *start).abs() < 1e-5 && (b - *end).abs() < 1e-5)
                        && app.timeline.navigation.display.label() == display,
                    "Unexpected Timeline viewport {:?} / {}",
                    actual,
                    app.timeline.navigation.display.label()
                );
                self.navigation_checks
                    .push(json!({"range":actual,"display":display}));
            }
            Action::ExpectSplitSelection { previous } => {
                let selected = app.timeline.section.as_ref();
                anyhow::ensure!(
                    selected.is_some_and(|s| s.0 != *previous
                        && app.timeline.section_selection == vec![s.clone()]
                        && app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .scenes
                            .iter()
                            .flat_map(|s| &s.sections)
                            .any(|section| &section.id == s))),
                    "Split did not select the new right Section"
                );
                self.navigation_checks
                    .push(json!({"split_right_selected":true}));
            }
            Action::WaitFinalFailure { code } => {
                let details = app.timeline_final_diagnostics();
                if details["failure"]["code"] != json!(code) {
                    return Ok(());
                }
                self.final_failures.push(details);
            }
            Action::WaitIdle => {
                if app.timeline.busy
                    || app.timeline.media_health.pending()
                    || app.timeline.reference_cache.pending()
                    || app.timeline.catalog.busy
                    || app.timeline.authoring.busy
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
            Action::WaitAuthoringProgress { completed, eta } => {
                if !app.timeline.authoring.progress_ready(*completed, *eta) {
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
                ids,
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
                    ids.as_ref().is_none_or(|ids| state["ids"] == json!(ids)),
                    "Unexpected A/B pair: {state}"
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
            Action::WaitCapture {
                take,
                exists,
                attached,
                removed,
            } => {
                anyhow::ensure!(
                    app.timeline.error.is_none(),
                    "Own Capture mutation failed: {:?}",
                    app.timeline.error
                );
                let Some(document) = &app.timeline.document else {
                    return Ok(());
                };
                let entry = document.view.takes.iter().find(|t| t.id.0 == *take);
                if entry.is_some() != *exists
                    || entry.is_some_and(|t| super::review::attached(document, t)) != *attached
                    || document
                        .view
                        .captures
                        .iter()
                        .any(|(id, c)| id.0 == *take && c.removed)
                        != *removed
                {
                    return Ok(());
                }
            }
            Action::WaitTakeStatus {
                take,
                status,
                accepted,
            } => {
                anyhow::ensure!(
                    app.timeline.error.is_none(),
                    "Own Take review failed: {:?}",
                    app.timeline.error
                );
                let Some(document) = &app.timeline.document else {
                    return Ok(());
                };
                let current = document.view.takes.iter().find(|entry| entry.id.0 == *take);
                if current.is_none_or(|entry| entry.status.label() != status.as_str())
                    || document.take_state(&TakeId(take.clone())).accepted != *accepted
                {
                    return Ok(());
                }
            }
            Action::ObserveScroll {
                name,
                label,
                compare_to,
                minimum_y,
                visible_control,
            } => {
                anyhow::ensure!(!app.queue.collapsed, "Scroll probe requires expanded Queue");
                let Some(sample) = ctx.data(|d| {
                    d.get_temp::<ScrollObservation>(egui::Id::new(("timeline-qa-scroll", name)))
                }) else {
                    return Ok(());
                };
                if ctx.cumulative_frame_nr().saturating_sub(sample.frame) > 1 {
                    return Ok(());
                }
                let baseline = compare_to
                    .as_ref()
                    .map(|label| {
                        self.scroll_checks
                            .get(label)
                            .and_then(|value| value["offset"][1].as_f64())
                            .map(|value| value as f32)
                            .ok_or_else(|| anyhow::anyhow!("Missing scroll baseline {label}"))
                    })
                    .transpose()?;
                let control = visible_control
                    .as_ref()
                    .map(|name| {
                        let full = ctx.data(|d| {
                            d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                                "timeline-qa-layout",
                                name,
                            )))
                        });
                        let clip = ctx.data(|d| {
                            d.get_temp::<egui::Rect>(egui::Id::new(("timeline-qa-clip", name)))
                        });
                        match (full, clip) {
                            (Some((frame, rect, enabled)), Some(clip))
                                if ctx.cumulative_frame_nr().saturating_sub(frame) <= 1 =>
                            {
                                Ok((rect, clip, enabled))
                            }
                            _ => anyhow::bail!("Missing current scroll control {name}"),
                        }
                    })
                    .transpose()?;
                check_scroll(&sample, *minimum_y, baseline, control)?;
                self.scroll_checks.insert(
                    label.clone(),
                    json!({"area":name,"frame":sample.frame,
                        "offset":[sample.offset.x,sample.offset.y],
                        "viewport":[sample.viewport.left(),sample.viewport.top(),sample.viewport.right(),sample.viewport.bottom()],
                        "content_size":[sample.content.x,sample.content.y],
                        "compared_to":compare_to,"visible_control":visible_control,
                        "queue_expanded":true}),
                );
            }
            Action::ExpectCombo { name, open } => {
                let Some(id) = ctx.data(|d| {
                    d.get_temp::<egui::Id>(egui::Id::new(("timeline-qa-control-id", name)))
                }) else {
                    return Ok(());
                };
                anyhow::ensure!(
                    egui::ComboBox::is_open(ctx, id) == *open,
                    "Unexpected ComboBox state for {name} (expected open={open})"
                );
            }
            Action::ScrollControl {
                name,
                delta_x,
                delta_y,
            } => {
                let Some((frame, rect, _)) = ctx.data(|d| {
                    d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                        "timeline-qa-control",
                        name,
                    )))
                }) else {
                    return Ok(());
                };
                if ctx.cumulative_frame_nr().saturating_sub(frame) > 1 {
                    return Ok(());
                }
                self.input.push_back(vec![
                    egui::Event::PointerMoved(rect.center()),
                    egui::Event::MouseWheel {
                        phase: egui::TouchPhase::Move,
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(*delta_x, *delta_y),
                        modifiers: Default::default(),
                    },
                ]);
                self.delay = 8;
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
            Action::WaitImageJobs { status, count } => {
                if app
                    .queue
                    .jobs
                    .iter()
                    .filter(|job| &job.status == status)
                    .count()
                    != *count
                {
                    return Ok(());
                }
            }
            Action::WaitStatus { status, count } => {
                if app
                    .timeline
                    .jobs
                    .iter()
                    .filter(|j| j.status().as_str() == status.as_str())
                    .count()
                    != *count
                {
                    return Ok(());
                }
            }
            Action::ScrollToControl { scroll, name } => {
                if self
                    .control_wait
                    .as_ref()
                    .is_none_or(|(old, _)| old != name)
                {
                    self.control_wait = Some((name.clone(), Instant::now()));
                }

                let lookup = |name: &String| {
                    ctx.data(|d| {
                        d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                            "timeline-qa-layout",
                            name,
                        )))
                    })
                };
                anyhow::ensure!(
                    self.control_wait.as_ref().unwrap().1.elapsed() < Duration::from_secs(8),
                    "QA scroll target {name}: {:?}; viewport {scroll}: {:?}",
                    lookup(name),
                    lookup(scroll)
                );
                let (Some((frame, rect, _)), Some((scroll_frame, viewport, _))) =
                    (lookup(name), lookup(scroll))
                else {
                    return Ok(());
                };
                if ctx.cumulative_frame_nr().saturating_sub(frame) > 1
                    || ctx.cumulative_frame_nr().saturating_sub(scroll_frame) > 1
                {
                    return Ok(());
                }
                // Wheel motion is smoothed over wall time. A target can cross
                // the viewport while still moving and then leave it again.
                // Observe a settled rectangle before another wheel or click.
                match &mut self.scroll_motion {
                    Some((target, previous, since)) if target == name => {
                        if (previous.min - rect.min).length() > 0.25
                            || (previous.max - rect.max).length() > 0.25
                        {
                            *previous = rect;
                            *since = Instant::now();
                            return Ok(());
                        }
                        if since.elapsed() < Duration::from_millis(120) {
                            return Ok(());
                        }
                    }
                    _ => {
                        self.scroll_motion = Some((name.clone(), rect, Instant::now()));
                        return Ok(());
                    }
                }
                // A first/last row cannot scroll past the content edge. Allow
                // subpixel layout rounding while still rejecting clipped controls.
                if rect.top() < viewport.top() - 0.5 || rect.bottom() > viewport.bottom() + 0.5 {
                    self.input.push_back(vec![
                        egui::Event::PointerMoved(viewport.center()),
                        egui::Event::MouseWheel {
                            phase: egui::TouchPhase::Move,
                            unit: egui::MouseWheelUnit::Point,
                            delta: egui::vec2(
                                0.,
                                (viewport.center().y - rect.center().y).clamp(-300., 300.),
                            ),
                            modifiers: Default::default(),
                        },
                    ]);
                    self.scroll_motion = None;
                    self.delay = 8;
                    return Ok(());
                }
                self.control_wait = None;
                self.scroll_motion = None;
            }
            Action::HoverControl { name } => {
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
                self.input
                    .push_back(vec![egui::Event::PointerMoved(rect.center())]);
                self.delay = 4;
            }
            Action::ClickControl {
                name,
                expected_enabled,
                ctrl,
                shift,
                secondary,
            } => {
                if self
                    .control_wait
                    .as_ref()
                    .is_none_or(|(old, _)| old != name)
                {
                    self.control_wait = Some((name.clone(), Instant::now()));
                }
                anyhow::ensure!(
                    self.control_wait.as_ref().unwrap().1.elapsed() < Duration::from_secs(8),
                    "Own UI control {name} did not become clickable"
                );
                let Some((frame, rect, enabled)) = ctx.data(|d| {
                    d.get_temp::<(u64, egui::Rect, bool)>(egui::Id::new((
                        "timeline-qa-control",
                        name,
                    )))
                }) else {
                    return Ok(());
                };
                if (!enabled && *expected_enabled != Some(false))
                    || ctx.cumulative_frame_nr().saturating_sub(frame) > 1
                {
                    return Ok(());
                }
                if let Some(expected) = expected_enabled {
                    anyhow::ensure!(enabled == *expected, "Unexpected enabled state for {name}");
                }
                let modifiers = egui::Modifiers {
                    ctrl: *ctrl,
                    command: *ctrl,
                    shift: *shift,
                    ..Default::default()
                };
                let button = if *secondary {
                    egui::PointerButton::Secondary
                } else {
                    egui::PointerButton::Primary
                };
                self.control_wait = None;
                let pos = rect.center();
                self.input.push_back(vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button,
                        pressed: true,
                        modifiers,
                    },
                ]);
                self.input.push_back(vec![egui::Event::PointerButton {
                    pos,
                    button,
                    pressed: false,
                    modifiers,
                }]);
                self.delay = 4;
            }
            Action::DragControl {
                name,
                delta_x,
                ctrl,
                shift,
            } => {
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
                let modifiers = egui::Modifiers {
                    ctrl: *ctrl,
                    command: *ctrl,
                    shift: *shift,
                    ..Default::default()
                };
                self.input.push_back(vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::PointerMoved(start),
                    egui::Event::PointerButton {
                        pos: start,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers,
                    },
                ]);
                for i in 1..=6 {
                    self.input.push_back(vec![
                        egui::Event::ModifiersChanged(modifiers),
                        egui::Event::PointerMoved(start + (end - start) * (i as f32 / 6.)),
                    ]);
                }
                self.input.push_back(vec![egui::Event::PointerButton {
                    pos: end,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers,
                }]);
                self.delay = 6;
            }
            Action::Open { folder } => {
                if app.timeline.busy || app.timeline.session.is_none() {
                    return Ok(());
                }
                app.timeline_folder(ctx, folder.clone(), FolderMode::Open);
            }
            Action::ImportPicker { paths } => {
                anyhow::ensure!(
                    paths
                        .iter()
                        .all(|p| p.starts_with(&self.plan.destination.root)),
                    "QA imports must use own fixtures under the test root"
                );
                let target = app.timeline.media_picker.target.clone().ok_or_else(|| {
                    anyhow::anyhow!("Open the contextual picker before importing")
                })?;
                app.timeline_import_into(ctx, paths.clone(), target);
            }
            Action::OpenLibrary { folder } => {
                anyhow::ensure!(
                    folder.starts_with(&self.plan.destination.root),
                    "Only own QA library fixtures may be opened"
                );
                app.timeline.library.path_from_dialog(folder.clone());
                app.library_select(ctx, "open");
            }
            Action::OpenPromptCatalog { folder } => {
                anyhow::ensure!(
                    folder.starts_with(&self.plan.destination.root),
                    "Only own catalog fixtures may be opened"
                );
                app.timeline_catalog_folder(ctx, Some(folder.clone()));
            }
            Action::SavePromptCatalog { path } => {
                anyhow::ensure!(
                    path.starts_with(&self.plan.destination.root),
                    "Only own catalog fixtures may be saved"
                );
                app.timeline_catalog_write(ctx, true, false, Some(path.clone()));
            }
            Action::Save => app.timeline_save(ctx),
            Action::Privacy { enabled } => app.prefs.privacy_mode = *enabled,
            Action::Focus { focused } => {
                self.focused = *focused;
                self.delay = 3;
            }
            Action::ZoomUi { factor } => {
                ctx.set_zoom_factor(*factor);
                self.delay = 8;
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
                    "ArrowDown" => egui::Key::ArrowDown,
                    "ArrowUp" => egui::Key::ArrowUp,
                    "ArrowLeft" => egui::Key::ArrowLeft,
                    "ArrowRight" => egui::Key::ArrowRight,
                    "Home" => egui::Key::Home,
                    "End" => egui::Key::End,
                    "A" => egui::Key::A,
                    "S" => egui::Key::S,
                    "Z" => egui::Key::Z,
                    "Y" => egui::Key::Y,
                    "Enter" => egui::Key::Enter,
                    "Tab" => egui::Key::Tab,
                    "Escape" => egui::Key::Escape,
                    "Delete" => egui::Key::Delete,
                    "Backspace" => egui::Key::Backspace,
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
                    section_prompt,
                    sections,
                    selected_sections,
                    compiled_contains,
                    compiled_excludes,
                    compiled_issues,
                    compiled_seed,
                    compiled_stale,
                    scene_prompts,
                    authoring_warnings,
                    authoring_contains,
                    library_entries,
                    cast_references,
                    cast_images,
                    story_images,
                    inventory_notes,
                    dialogue_sections,
                    image_sections,
                    video_sections,
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
                if let Some(count) = sections {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().map(|d| d
                            .project
                            .scenes
                            .iter()
                            .map(|s| s.sections.len())
                            .sum::<usize>())
                            == Some(*count),
                        "Unexpected total Section count: expected {count}, got {:?}; selection {:?}, text focused {}",
                        app.timeline.document.as_ref().map(|d| d
                            .project
                            .scenes
                            .iter()
                            .map(|s| s.sections.len())
                            .sum::<usize>()),
                        app.timeline.section_selection,
                        ctx.text_edit_focused()
                    );
                }
                if let Some(ids) = selected_sections {
                    anyhow::ensure!(
                        app.timeline
                            .section_selection
                            .iter()
                            .map(|s| &s.0)
                            .collect::<Vec<_>>()
                            == ids.iter().collect::<Vec<_>>(),
                        "Unexpected ordered Section selection"
                    );
                }
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
                if let Some(text) = section_prompt {
                    anyhow::ensure!(
                        app.timeline.draft_section == *text,
                        "Unexpected Section prompt text: {:?}",
                        app.timeline.draft_section
                    );
                }
                if let Some(text) = compiled_excludes {
                    anyhow::ensure!(
                        !app.timeline
                            .compiled
                            .value
                            .as_ref()
                            .and_then(|v| v["prompt"].as_str())
                            .is_some_and(|p| p.contains(text)),
                        "Disabled global prompt leaked into compiled source"
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
                if let Some(seed) = compiled_seed {
                    anyhow::ensure!(
                        app.timeline.compiled.reviewed_seed() == Some(*seed),
                        "Reviewed seed differs"
                    );
                }
                if let Some(stale) = compiled_stale {
                    anyhow::ensure!(
                        app.timeline.compiled.stale(
                            app.timeline.document.as_ref().map(|d| &d.project),
                            !app.timeline.draft_edits().is_empty()
                        ) == *stale,
                        "Unexpected reviewed plan freshness"
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
                if let Some(count) = cast_images {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().map(|d| d
                            .project
                            .creative
                            .references
                            .iter()
                            .map(|r| r.images.len())
                            .sum::<usize>())
                            == Some(*count),
                        "Reference image count mismatch"
                    );
                }
                if let Some(count) = story_images {
                    anyhow::ensure!(
                        app.timeline.authoring.source_image_count() == *count,
                        "Unexpected own Story source image count"
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
                if let Some(count) = video_sections {
                    anyhow::ensure!(
                        app.timeline.document.as_ref().is_some_and(|d| d
                            .project
                            .creative
                            .sections
                            .values()
                            .filter(|s| !s.videos.is_empty())
                            .count()
                            == *count),
                        "Unexpected section video binding count"
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
                            .is_some_and(|d| d.view.deliveries.len() == *count),
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
                        document
                            .view
                            .resources
                            .iter()
                            .filter(|resource| resource.kind() == "h3_endpoint")
                            .count()
                            == *expected,
                        "Unexpected QA endpoint count"
                    );
                }
                if let Some(expected) = motion_artifacts {
                    anyhow::ensure!(
                        document
                            .view
                            .resources
                            .iter()
                            .filter(|r| r.kind() == "h3_motion")
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
                        let actual = document
                            .view
                            .take_states
                            .values()
                            .filter(|state| {
                                if field == "accepted" {
                                    state.accepted
                                } else {
                                    state.outdated
                                }
                            })
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
                    anyhow::ensure!(document.view.takes.len() == *n, "Unexpected QA Take count");
                }
                if let Some(saved) = saved {
                    anyhow::ensure!(
                        (document.save_state() == Some(&super::snapshot::SaveState::Saved))
                            == *saved,
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
#[derive(Clone, Copy)]
struct ScrollObservation {
    frame: u64,
    offset: egui::Vec2,
    viewport: egui::Rect,
    content: egui::Vec2,
}

fn check_scroll(
    sample: &ScrollObservation,
    minimum_y: f32,
    baseline_y: Option<f32>,
    control: Option<(egui::Rect, egui::Rect, bool)>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        sample.offset.y.is_finite() && sample.offset.y >= minimum_y,
        "Scroll offset {} is below {minimum_y}",
        sample.offset.y
    );
    if let Some(baseline) = baseline_y {
        anyhow::ensure!(
            (sample.offset.y - baseline).abs() <= 0.75,
            "Scroll changed from {baseline} to {} after pointer movement",
            sample.offset.y
        );
    }
    if let Some((rect, clip, enabled)) = control {
        // Full widget bounds, not merely their clipped/interactable fragment.
        anyhow::ensure!(
            enabled
                && rect.is_positive()
                && sample.viewport.expand(0.5).contains_rect(rect)
                && clip.expand(0.5).contains_rect(rect),
            "Bottom control is disabled or clipped: {rect:?}, viewport {:?}, clip {clip:?}",
            sample.viewport
        );
    }
    Ok(())
}

/// Geometry only: no prompt, job, media, pixels or product-state mutation.
/// Call with the actual ScrollAreaOutput after show() has completed.
pub(in crate::app) fn scroll_area(
    ui: &egui::Ui,
    name: &str,
    _id: egui::Id,
    offset: egui::Vec2,
    viewport: egui::Rect,
    content: egui::Vec2,
) {
    let frame = ui.ctx().cumulative_frame_nr();
    ui.ctx().data_mut(|data| {
        data.insert_temp(
            egui::Id::new(("timeline-qa-scroll", name)),
            ScrollObservation {
                frame,
                offset,
                viewport,
                content,
            },
        );
        for prefix in ["timeline-qa-layout", "timeline-qa-control"] {
            data.insert_temp(egui::Id::new((prefix, name)), (frame, viewport, true));
        }
    });
}

pub(in crate::app) fn control(ui: &egui::Ui, name: String, response: &egui::Response) {
    let frame = ui.ctx().cumulative_frame_nr();
    ui.ctx().data_mut(|d| {
        d.insert_temp(egui::Id::new(("timeline-qa-clip", &name)), ui.clip_rect());
        d.insert_temp(
            egui::Id::new(("timeline-qa-control-id", &name)),
            response.id,
        );
        d.insert_temp(
            egui::Id::new(("timeline-qa-layout", &name)),
            (frame, response.rect, response.enabled()),
        )
    });
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

// Real OS picker QA stays within the explicitly declared own test directory.
// This changes only ui-preview runs with an explicit QA directory, never normal UI.
pub(in crate::app) fn picker_directory(app: &ImagineApp) -> Option<PathBuf> {
    app.qa
        .as_ref()
        .map(|runner| runner.plan.destination.root.clone())
        .or_else(|| app.qa_manual_picker_root.clone())
}
pub(in crate::app) fn configure_picker(
    dialog: rfd::FileDialog,
    directory: Option<PathBuf>,
) -> rfd::FileDialog {
    match directory {
        Some(directory) => dialog
            .set_directory(directory)
            .set_title(format!("Imagine own picker QA {}", std::process::id())),
        None => dialog,
    }
}
impl ImagineApp {
    /// Manual OS-dialog acceptance retains normal input; no automated Runner.
    /// The test-only initial folder is confined to our declared fixture tree.
    pub fn install_manual_picker_root(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let path = path.canonicalize()?;
        anyhow::ensure!(
            path.is_dir()
                && path.starts_with(
                    "/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"
                ),
            "Manual picker QA requires an own fixture directory"
        );
        self.qa_manual_picker_root = Some(path);
        Ok(())
    }
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

#[cfg(test)]
mod scroll_tests {
    use super::*;
    fn sample(y: f32) -> ScrollObservation {
        ScrollObservation {
            frame: 1,
            offset: egui::vec2(0., y),
            viewport: egui::Rect::from_min_max(egui::pos2(8., 72.), egui::pos2(592., 540.)),
            content: egui::vec2(584., 920.),
        }
    }
    #[test]
    fn pointer_motion_must_preserve_nonzero_offset() {
        assert!(check_scroll(&sample(400.), 300., Some(400.), None).is_ok());
        assert!(check_scroll(&sample(400.5), 300., Some(400.), None).is_ok());
        assert!(check_scroll(&sample(0.), 300., Some(400.), None).is_err());
        assert!(check_scroll(&sample(398.), 300., Some(400.), None).is_err());
    }
    #[test]
    fn clipped_fragment_is_not_an_accessible_bottom_control() {
        let sample = sample(400.);
        let full = egui::Rect::from_min_max(egui::pos2(300., 500.), egui::pos2(560., 532.));
        assert!(check_scroll(&sample, 300., None, Some((full, sample.viewport, true))).is_ok());
        assert!(check_scroll(&sample, 300., None, Some((full, sample.viewport, false))).is_err());
        let clip = egui::Rect::from_min_max(egui::pos2(8., 72.), egui::pos2(592., 520.));
        assert!(check_scroll(&sample, 300., None, Some((full, clip, true))).is_err());
        let below = full.translate(egui::vec2(0., 50.));
        assert!(check_scroll(&sample, 300., None, Some((below, sample.viewport, true))).is_err());
    }
}
