//! Video presets from the shared presets file. The server turns a preset into
//! edits with the same code as MCP; the UI submits them as one undoable batch.
use super::*;
use imagine_video_contract::creative::{PresetState, preset_states};

#[derive(Clone, Debug, PartialEq)]
pub(in crate::app) enum Target {
    Scenes(Vec<SceneId>),
    Project,
}
#[derive(Clone, Debug, Default, serde::Deserialize)]
struct Entry {
    key: String,
    variant: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    default: bool,
    #[serde(default)]
    problem: Option<String>,
    #[serde(default)]
    settings: Value,
    #[serde(default)]
    loras: Vec<Lora>,
    #[serde(default)]
    lora_merge: String,
}
#[derive(Clone, Debug, serde::Deserialize)]
struct Lora {
    name: String,
    #[serde(default = "one")]
    strength_model: f64,
}
fn one() -> f64 {
    1.
}
#[derive(Clone, Debug, Default, serde::Deserialize)]
struct Catalog {
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    problem: Option<String>,
    presets: Vec<Entry>,
}
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    loading: bool,
    catalog: Option<Result<Catalog, String>>,
    /// A preset chosen in the inspector or generation window, applied once
    /// the timeline is idle.
    chosen: Option<(String, Target)>,
}
fn control(ui: &egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), r);
    #[cfg(test)]
    ui.ctx()
        .data_mut(|data| data.insert_temp(egui::Id::new(("preset-test-control", name)), r.clone()));
    let _ = (ui, name, r);
}
fn variant(name: &str) -> &str {
    match name {
        "reference" => "Reference",
        "i2v" => "I2V",
        other => other,
    }
}
/// Named settings as `path value`, except the model file shown on its own.
fn settings(prefix: &str, value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if path != "components.model" {
                    settings(&path, value, out);
                }
            }
        }
        Value::Null => {}
        Value::String(text) => out.push(format!("{prefix} {text}")),
        other => out.push(format!("{prefix} {other}")),
    }
}
impl State {
    pub(super) fn reply(&mut self, result: Result<Value, client::TimelineFailure>) {
        self.loading = false;
        self.catalog = Some(match result {
            Ok(value) => serde_json::from_value(value)
                .map_err(|_| "The video service returned an invalid preset list.".to_owned()),
            Err(failure) => Err(failure.message),
        });
    }
    fn entry(&self, key: &str) -> Option<&Entry> {
        self.catalog
            .as_ref()?
            .as_ref()
            .ok()?
            .presets
            .iter()
            .find(|p| p.key == key)
    }
    /// The preset list; returns the key the user chose.
    fn picker(&self, ui: &mut egui::Ui, name: &str, selected: &str) -> Option<String> {
        let mut chosen = None;
        let r = egui::ComboBox::from_id_salt(name)
            .selected_text(selected)
            .width(ui.available_width().min(260.))
            .show_ui(ui, |ui| match &self.catalog {
                None => {
                    ui.label("Loading presets…");
                }
                Some(Err(error)) => {
                    ui.colored_label(helto::ERROR, error);
                }
                Some(Ok(catalog)) => {
                    for (i, entry) in catalog.presets.iter().enumerate() {
                        let mut r = ui.add_enabled(
                            entry.problem.is_none(),
                            egui::Button::selectable(entry.key == selected, &entry.key),
                        );
                        if let Some(code) = &entry.problem {
                            r = r.on_disabled_hover_text(client::error_message(code));
                        } else if !entry.description.is_empty() {
                            r = r.on_hover_text(&entry.description);
                        }
                        control(ui, &format!("{name}-{i}"), &r);
                        if r.clicked() {
                            chosen = Some(entry.key.clone());
                        }
                    }
                }
            });
        control(ui, name, &r.response);
        chosen
    }
    fn window(&mut self, ctx: &egui::Context, privacy: bool) -> bool {
        if !self.open {
            return false;
        }
        let mut open = true;
        let mut reload = false;
        egui::Window::new("Video presets")
            .id(egui::Id::new("video-presets"))
            .open(&mut open)
            .default_width(560.)
            .default_height(440.)
            .show(ctx, |ui| {
                let hidden = !helto::reveal_in_viewport(ui, privacy);
                egui::ScrollArea::vertical()
                    .id_salt("video-presets-scroll")
                    .auto_shrink([false, true])
                    .max_height(380.)
                    .show(ui, |ui| match &self.catalog {
                        None => {
                            ui.label("Loading presets…");
                        }
                        Some(Err(error)) => {
                            ui.colored_label(helto::ERROR, error);
                        }
                        Some(Ok(catalog)) => {
                            helto::private_scope(ui, hidden, |ui| {
                                for entry in &catalog.presets {
                                    row(ui, entry);
                                    ui.separator();
                                }
                            });
                        }
                    });
                if let Some(Ok(catalog)) = &self.catalog {
                    if let Some(code) = &catalog.problem {
                        ui.colored_label(helto::ERROR, client::error_message(code));
                    }
                    if let Some(file) = &catalog.file {
                        helto::private_scope(ui, hidden, |ui| ui.weak(file));
                    }
                }
                let r = helto::icon_button(ui, crate::icons::Icon::Retry, "Reload", false);
                control(ui, "video-presets-reload", &r);
                reload = r.clicked() && !self.loading;
            });
        self.open = open;
        reload
    }
}
fn row(ui: &mut egui::Ui, entry: &Entry) {
    ui.horizontal_wrapped(|ui| {
        ui.strong(&entry.key);
        let mut kind = variant(&entry.variant).to_owned();
        if entry.default {
            kind.push_str(" · Built-in");
        }
        ui.weak(kind);
    });
    if !entry.description.is_empty() {
        ui.weak(&entry.description);
    }
    if let Some(model) = entry.settings["components"]["model"].as_str() {
        ui.label(format!("Model · {model}"));
    }
    let mut values = vec![];
    settings("", &entry.settings, &mut values);
    if !values.is_empty() {
        ui.label(values.join(" · "));
    }
    if !entry.loras.is_empty() {
        let names = entry
            .loras
            .iter()
            .map(|l| format!("{} {}", l.name, l.strength_model))
            .collect::<Vec<_>>()
            .join(", ");
        ui.label(format!("LoRAs · {names}"));
    }
    if entry.lora_merge != "Add To Global" && !entry.lora_merge.is_empty() {
        ui.label(format!("Scene LoRAs · {}", entry.lora_merge));
    }
    if let Some(code) = &entry.problem {
        ui.colored_label(helto::ERROR, client::error_message(code));
    }
}
/// The scenes a scene preset applies to: the selected scenes in timeline
/// order, always including the one in the inspector.
fn targets(p: &VideoProject, items: Vec<selection::Item>, primary: &SceneId) -> Vec<SceneId> {
    let mut ids = items
        .into_iter()
        .filter_map(|item| match item {
            selection::Item::Scene(id) => Some(id),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !ids.contains(primary) {
        ids.push(primary.clone());
    }
    ids.retain(|id| p.scenes.iter().any(|s| &s.id == id));
    ids.sort_by_key(|id| p.scenes.iter().position(|s| &s.id == id));
    ids.dedup();
    ids
}
/// What the scene dropdown shows: the shared preset or "Mixed", or "Custom";
/// and whether any of the scenes was changed since its preset was applied.
fn shown(p: &VideoProject, scenes: &[SceneId]) -> (String, Option<String>, bool) {
    let states = preset_states(p);
    let keys = scenes
        .iter()
        .map(|id| p.creative.scene_presets.get(id).map(|a| a.key.clone()))
        .collect::<BTreeSet<_>>();
    let modified = scenes
        .iter()
        .any(|id| states.get(id) == Some(&PresetState::Modified));
    match keys.into_iter().collect::<Vec<_>>().as_slice() {
        [Some(key)] => (key.clone(), Some(key.clone()), modified),
        [None] | [] => ("Custom".into(), None, false),
        _ => ("Mixed".into(), None, modified),
    }
}
impl ImagineApp {
    fn timeline_presets_fetch(&mut self, ctx: &egui::Context) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.presets.loading = true;
        let (epoch, tx, ctx) = (self.timeline.epoch, self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                "/v1/video/presets",
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Presets { epoch, result }),
            );
        });
    }
    fn timeline_presets_ensure(&mut self, ctx: &egui::Context) {
        let state = &self.timeline.presets;
        if state.catalog.is_none() && !state.loading {
            self.timeline_presets_fetch(ctx);
        }
    }
    pub(super) fn timeline_open_presets(&mut self, ctx: &egui::Context) {
        self.timeline.presets.open = true;
        if !self.timeline.presets.loading {
            self.timeline_presets_fetch(ctx);
        }
    }
    /// The scene inspector's Preset row, applying to every selected scene.
    pub(super) fn timeline_scene_preset(&mut self, ui: &mut egui::Ui, d: &Snapshot, scene: &Scene) {
        self.timeline_presets_ensure(ui.ctx());
        let scenes = targets(
            &d.project,
            self.timeline.selection_items(&d.project),
            &scene.id,
        );
        let (text, key, modified) = shown(&d.project, &scenes);
        ui.label(if scenes.len() > 1 {
            format!("Preset · {} scenes", scenes.len())
        } else {
            "Preset".into()
        });
        let hidden = !helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            helto::private_scope(ui, hidden, |ui| {
                chosen = self.timeline.presets.picker(ui, "scene-preset", &text);
            });
            if modified {
                ui.colored_label(helto::WARN, "Modified");
                let known = key
                    .as_deref()
                    .and_then(|k| self.timeline.presets.entry(k))
                    .is_some_and(|e| e.problem.is_none());
                let r = ui
                    .add_enabled(known, egui::Button::new("Reapply"))
                    .on_disabled_hover_text("This preset is unavailable in the presets file.");
                control(ui, "scene-preset-reapply", &r);
                if r.clicked() {
                    chosen = key.clone();
                }
            }
        });
        if let Some(key) = chosen {
            self.timeline.presets.chosen = Some((key, Target::Scenes(scenes)));
        }
    }
    /// The Project generation window's Default preset row.
    pub(super) fn timeline_default_preset(&mut self, ui: &mut egui::Ui) -> Option<String> {
        self.timeline_presets_ensure(ui.ctx());
        ui.label("Default preset");
        self.timeline
            .presets
            .picker(ui, "project-preset", "Choose a preset")
    }
    pub(super) fn timeline_choose_project_preset(&mut self, key: String) {
        self.timeline.presets.chosen = Some((key, Target::Project));
    }
    /// Commits pending drafts, then plans the chosen preset once idle.
    pub(super) fn timeline_presets_pending(&mut self, ctx: &egui::Context) {
        if self.timeline.busy {
            return;
        }
        if let Some((key, target)) = self.timeline.presets.chosen.take() {
            self.timeline_after_drafts(ctx, PostAction::Preset { key, target });
        }
    }
    pub(super) fn timeline_preset_plan(
        &mut self,
        ctx: &egui::Context,
        key: String,
        target: Target,
    ) {
        if self.timeline.busy {
            self.timeline.presets.chosen = Some((key, target));
            return;
        }
        let Some(d) = self.timeline.document.as_ref() else {
            return;
        };
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            self.timeline.error =
                Some("Not connected to the video service. Reconnect and try again.".into());
            return;
        };
        let mut body = json!({"schema_version":1,
            "expected_project_revision":d.project.project_revision,"preset":key});
        match &target {
            Target::Scenes(ids) => body["scenes"] = json!(ids),
            Target::Project => body["project"] = json!(true),
        }
        let path = format!("/v1/video/projects/{}/presets/plan", d.project.id.0);
        let project = target == Target::Project;
        self.timeline.busy = true;
        self.timeline.error = None;
        let (epoch, tx, ctx) = (self.timeline.epoch, self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_request(&session, true, &path, Some(&body)));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::PresetPlan {
                    epoch,
                    project,
                    result,
                }),
            );
        });
    }
    /// Submits the planned batch through the normal edit path, so it is one
    /// undo step and revision-checked like any edit.
    pub(super) fn timeline_preset_planned(
        &mut self,
        ctx: &egui::Context,
        project: bool,
        result: Result<Value, client::TimelineFailure>,
    ) {
        self.timeline.busy = false;
        let request = match result {
            Ok(value) => serde_json::from_value::<EditRequest>(value)
                .map_err(|_| "The video service returned an invalid preset plan.".to_owned()),
            Err(failure) => Err(failure.message),
        };
        match request {
            Ok(request)
                if self.timeline.document.as_ref().is_some_and(|d| {
                    d.project.project_revision == request.expected_project_revision
                }) =>
            {
                if project {
                    self.timeline.after_edit = Some(PostAction::ProjectPresetApplied);
                }
                self.timeline_edit(ctx, request.commands, 0);
            }
            Ok(_) => {
                self.timeline.error = Some("The project changed. Choose the preset again.".into());
            }
            Err(error) => self.timeline.error = Some(error),
        }
    }
    pub(super) fn timeline_presets_window(&mut self, ctx: &egui::Context) {
        if self.timeline.presets.window(ctx, self.prefs.privacy_mode) {
            self.timeline_presets_fetch(ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_video_contract::creative::{AppliedPreset, scene_preset_fingerprint};
    fn catalog() -> Catalog {
        serde_json::from_value(json!({"schema_version":1,"file":"/synthetic/SYNTHETIC_PRIVATE-presets.yaml","problem":null,"presets":[
            {"key":"Default ref2v","variant":"reference","description":"Standard","default":true,"available":true,"problem":null,"settings":null,"loras":[],"lora_merge":"Add To Global"},
            {"key":"SYNTHETIC_PRIVATE tuned","variant":"reference","description":"","default":false,"available":true,"problem":null,
                "settings":{"components":{"model":"diffusion_models/own.safetensors"},"stage_one":{"steps":24}},"loras":[{"name":"h3/own.safetensors","strength_model":0.8}],"lora_merge":"Replace Global"},
            {"key":"Own broken","variant":"i2v","description":"","default":false,"available":false,"problem":"preset_file_missing","settings":null,"loras":[],"lora_merge":"Add To Global"}
        ]})).unwrap()
    }
    fn run(
        ctx: &egui::Context,
        body: impl FnMut(&mut egui::Ui),
    ) -> Vec<egui::epaint::ClippedShape> {
        run_with(ctx, vec![], body)
    }
    fn run_with(
        ctx: &egui::Context,
        events: Vec<egui::Event>,
        mut body: impl FnMut(&mut egui::Ui),
    ) -> Vec<egui::epaint::ClippedShape> {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800., 600.),
                )),
                events,
                ..Default::default()
            },
            |ui| body(ui),
        );
        output.textures_delta.clear();
        output.shapes
    }
    fn painted(shapes: &[egui::epaint::ClippedShape], needle: &str) -> bool {
        fn find(shape: &egui::Shape, needle: &str) -> bool {
            match shape {
                egui::Shape::Text(t) => t.opacity_factor > 0. && t.galley.text().contains(needle),
                egui::Shape::Vec(shapes) => shapes.iter().any(|s| find(s, needle)),
                _ => false,
            }
        }
        shapes.iter().any(|s| find(&s.shape, needle))
    }
    #[test]
    fn settings_list_named_values_without_the_model_file() {
        let mut out = vec![];
        settings("", &catalog().presets[1].settings, &mut out);
        assert_eq!(out, ["stage_one.steps 24"]);
    }
    #[test]
    fn scene_targets_follow_selected_scenes_in_timeline_order_and_include_the_inspected_scene() {
        let mut p = super::super::media_picker::tests::project();
        p.scenes = ["own_a", "own_b", "own_c"]
            .iter()
            .enumerate()
            .map(|(i, id)| Scene {
                id: SceneId((*id).into()),
                start: Time {
                    num: 5 * i as u64,
                    den: 1,
                },
                duration: Time { num: 5, den: 1 },
                description: String::new(),
                model: ModelChoice::ProjectDefault,
                settings: vec![],
                sections: vec![],
                bindings: vec![],
                continuity: None,
                accepted_take: None,
            })
            .collect();
        let id = |s: &str| SceneId(s.into());
        let items = vec![
            selection::Item::Scene(id("own_c")),
            selection::Item::Section(SectionId("own_s".into())),
            selection::Item::Scene(id("own_a")),
        ];
        assert_eq!(
            targets(&p, items.clone(), &id("own_a")),
            [id("own_a"), id("own_c")]
        );
        assert_eq!(
            targets(&p, items, &id("own_b")),
            [id("own_a"), id("own_b"), id("own_c")]
        );
        assert_eq!(shown(&p, &[id("own_a")]), ("Custom".into(), None, false));
        // A scene set to a preset shows it; a changed scene shows Modified.
        let scene = p.scenes[0].clone();
        let fingerprint = scene_preset_fingerprint(&p, &scene);
        p.creative.scene_presets.insert(
            id("own_a"),
            AppliedPreset {
                key: "Own tuned".into(),
                fingerprint,
            },
        );
        assert_eq!(
            shown(&p, &[id("own_a")]),
            ("Own tuned".into(), Some("Own tuned".into()), false)
        );
        assert_eq!(shown(&p, &[id("own_a"), id("own_b")]).0, "Mixed");
        p.scenes[0].settings.push(SettingsSlot {
            model: p.default_model.clone(),
            schema_version: 1,
            values: json!({"own":true}),
        });
        assert!(shown(&p, &[id("own_a")]).2);
    }
    #[test]
    fn picker_lists_presets_disables_unavailable_ones_and_returns_the_choice() {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let state = State {
            catalog: Some(Ok(catalog())),
            ..Default::default()
        };
        let get = |name: &str| {
            ctx.data(|d| d.get_temp::<egui::Response>(egui::Id::new(("preset-test-control", name))))
        };
        let click = |pos: egui::Pos2| {
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ]
        };
        let frame = |events| {
            let mut chosen = None;
            run_with(&ctx, events, |ui| {
                if let Some(key) = state.picker(ui, "own-picker", "Custom") {
                    chosen = Some(key);
                }
            });
            chosen
        };
        frame(vec![]);
        frame(click(get("own-picker").unwrap().rect.center()));
        for _ in 0..3 {
            frame(vec![]);
        }
        assert!(get("own-picker-1").unwrap().enabled());
        assert!(!get("own-picker-2").unwrap().enabled());
        // The unavailable preset cannot be chosen; an available one can.
        assert_eq!(
            frame(click(get("own-picker-2").unwrap().rect.center())),
            None
        );
        for _ in 0..3 {
            frame(vec![]);
        }
        if !egui::Popup::is_id_open(
            &ctx,
            egui::Popup::default_response_id(&get("own-picker").unwrap()),
        ) {
            frame(click(get("own-picker").unwrap().rect.center()));
            for _ in 0..3 {
                frame(vec![]);
            }
        }
        assert_eq!(
            frame(click(get("own-picker-1").unwrap().rect.center())).as_deref(),
            Some("SYNTHETIC_PRIVATE tuned")
        );
    }
    #[test]
    fn presets_window_shows_values_and_problems_and_hides_names_in_privacy_mode() {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let mut state = State {
            open: true,
            catalog: Some(Ok(catalog())),
            ..Default::default()
        };
        let mut shapes = vec![];
        for _ in 0..3 {
            shapes = run(&ctx, |ui| assert!(!state.window(ui.ctx(), false)));
        }
        for text in [
            "SYNTHETIC_PRIVATE tuned",
            "Model · diffusion_models/own.safetensors",
            "stage_one.steps 24",
            "LoRAs · h3/own.safetensors 0.8",
            "Scene LoRAs · Replace Global",
            "Reference · Built-in",
            "SYNTHETIC_PRIVATE-presets.yaml",
        ] {
            assert!(painted(&shapes, text), "{text}");
        }
        assert!(painted(
            &shapes,
            &client::error_message("preset_file_missing")
        ));
        for _ in 0..3 {
            shapes = run(&ctx, |ui| assert!(!state.window(ui.ctx(), true)));
        }
        assert!(!painted(&shapes, "SYNTHETIC_PRIVATE"));
        assert!(state.open);
    }
    #[test]
    fn a_planned_preset_goes_through_the_edit_path_and_a_stale_plan_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.client = None;
        let p = super::super::media_picker::tests::project();
        app.timeline.document = Some(Snapshot::with(p.clone(), json!({})));
        let mut creative = p.creative.clone();
        creative.scene_presets.insert(
            p.scenes[0].id.clone(),
            AppliedPreset {
                key: "Own tuned".into(),
                fingerprint: "own".into(),
            },
        );
        let plan = |revision| {
            serde_json::to_value(EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands: vec![Edit::SetCreative {
                    creative: creative.clone(),
                }],
            })
            .unwrap()
        };
        app.timeline.busy = true;
        app.timeline_preset_planned(&ctx, false, Ok(plan(p.project_revision + 1)));
        assert!(!app.timeline.busy);
        assert_eq!(
            app.timeline.error.as_deref(),
            Some("The project changed. Choose the preset again.")
        );
        assert!(app.timeline.after_edit.is_none());
        app.timeline_preset_planned(&ctx, true, Ok(plan(p.project_revision)));
        // Submitted as an edit; this test has no video service to take it.
        assert!(
            app.timeline
                .error
                .as_deref()
                .is_some_and(|e| e.starts_with("Not connected"))
        );
        assert!(matches!(
            app.timeline.after_edit,
            Some(PostAction::ProjectPresetApplied)
        ));
        // A choice made while the timeline is busy waits for it.
        app.timeline.after_edit = None;
        app.timeline.busy = true;
        app.timeline.presets.chosen = Some(("Own tuned".into(), Target::Project));
        app.timeline_presets_pending(&ctx);
        assert!(app.timeline.presets.chosen.is_some());
    }
}
