//! Session-scoped LoRA discovery. Catalog/metadata never validate an editable stack.
use crate::{client::Client, helto};
use eframe::egui;
use std::sync::mpsc::{self, Receiver, Sender};
#[path = "lora_filter.rs"]
mod filter;
#[path = "../../../shared/lora_catalog.rs"]
mod wire;
pub use wire::{Catalog, Entry, Info};

type Scope = (String, String);
enum Payload {
    Catalog(Result<Catalog, String>),
    Info(String, Result<Info, String>),
}
struct Reply {
    scope: Scope,
    nonce: u64,
    payload: Payload,
}
#[derive(Default)]
struct InfoView {
    file: String,
    value: Option<Info>,
    error: Option<String>,
    loading: bool,
}
pub(super) struct State {
    scope: Option<Scope>,
    tx: Sender<Reply>,
    rx: Receiver<Reply>,
    catalog_nonce: u64,
    info_nonce: u64,
    requested: bool,
    pub catalog: Option<Catalog>,
    pub loading: bool,
    pub error: Option<String>,
    info: Option<InfoView>,
    catalog_generation: u64,
    filter_nonce: u64,
    filter_key: Option<(u64, String)>,
    filter_engine: Option<filter::Engine>,
    filter_matches: Option<filter::Matches>,
    filter_pending: bool,
    filter_error: Option<filter::Error>,
    #[cfg(test)]
    filter_program: Option<std::ffi::OsString>,
    #[cfg(test)]
    filter_requests: usize,
}
impl Default for State {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            scope: None,
            tx,
            rx,
            catalog_nonce: 0,
            info_nonce: 0,
            requested: false,
            catalog: None,
            loading: false,
            error: None,
            info: None,
            catalog_generation: 0,
            filter_nonce: 0,
            filter_key: None,
            filter_engine: None,
            filter_matches: None,
            filter_pending: false,
            filter_error: None,
            #[cfg(test)]
            filter_program: None,
            #[cfg(test)]
            filter_requests: 0,
        }
    }
}
fn next(n: &mut u64) -> u64 {
    // Nonces only identify in-memory requests. Scope changes also invalidate replies.
    *n = n.wrapping_add(1);
    *n
}
fn message(code: &str) -> String {
    match code {
        "lora_catalog_unavailable" => {
            "LoRA folder is unavailable. Check the model library folder, then refresh."
        }
        "lora_info_missing" => "File not found. Check the path, then retry.",
        "lora_info_unreadable" => "File cannot be read. Check access, then retry.",
        "lora_info_format" => {
            "File metadata could not be read. The manual file selection is still available."
        }
        "invalid_response" => "Invalid catalog response. Retry after checking the server version.",
        "network_timeout" | "network_connect" | "network_body" | "network_request"
        | "network_error" => "Server connection interrupted. Reconnect, then retry.",
        _ => "LoRA request failed. Check the server connection, then retry.",
    }
    .into()
}
impl State {
    #[cfg(test)]
    pub(super) fn filter_unavailable_for_test(&mut self) {
        self.filter_program = Some("/own/nonexistent-node-engine".into());
        self.catalog = Some(Catalog {
            files: vec![Entry {
                file: "own/catalog.weights".into(),
                size: 1,
            }],
            warnings: vec![],
            root: None,
        });
    }
    pub fn bind(&mut self, scope: Option<Scope>) {
        if self.scope != scope {
            self.close_filter();
            self.scope = scope;
            next(&mut self.catalog_nonce);
            next(&mut self.info_nonce);
            self.requested = false;
            self.catalog = None;
            self.loading = false;
            self.error = None;
            self.info = None;
        }
        while let Ok(reply) = self.rx.try_recv() {
            self.accept(reply);
        }
    }
    fn accept(&mut self, reply: Reply) {
        if self.scope.as_ref() != Some(&reply.scope) {
            return;
        }
        match reply.payload {
            Payload::Catalog(result) if reply.nonce == self.catalog_nonce => {
                self.loading = false;
                match result {
                    Ok(value) => {
                        self.close_filter();
                        next(&mut self.catalog_generation);
                        self.catalog = Some(value);
                        self.error = None;
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Payload::Info(file, result) if reply.nonce == self.info_nonce => {
                if let Some(view) = self.info.as_mut().filter(|view| view.file == file) {
                    view.loading = false;
                    match result {
                        Ok(value) => {
                            view.value = Some(value);
                            view.error = None;
                        }
                        Err(error) => view.error = Some(error),
                    }
                }
            }
            _ => {}
        }
    }
    pub fn ensure(&mut self, client: Option<&Client>, ctx: &egui::Context) {
        if !self.requested {
            self.refresh(client, ctx);
        }
        if self.loading || self.filter_pending || self.info.as_ref().is_some_and(|i| i.loading) {
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
        }
    }
    pub fn refresh(&mut self, client: Option<&Client>, ctx: &egui::Context) {
        let (Some(client), Some(scope)) = (client.cloned(), self.scope.clone()) else {
            return;
        };
        if self.filter_error.is_some() {
            self.close_filter();
        }
        self.requested = true;
        self.loading = true;
        self.error = None;
        let nonce = next(&mut self.catalog_nonce);
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = crate::async_runtime::runtime()
                .block_on(client.timeline_request(&scope.0, false, "/v1/video/loras", None))
                .map_err(|e| message(&e.code))
                .and_then(|v| serde_json::from_value(v).map_err(|_| message("invalid_response")));
            let _ = tx.send(Reply {
                scope,
                nonce,
                payload: Payload::Catalog(result),
            });
            ctx.request_repaint();
        });
    }
    pub fn inspect(&mut self, file: String, client: Option<&Client>, ctx: &egui::Context) {
        self.info = Some(InfoView {
            file: file.clone(),
            ..Default::default()
        });
        let nonce = next(&mut self.info_nonce);
        let (Some(client), Some(scope)) = (client.cloned(), self.scope.clone()) else {
            self.info.as_mut().unwrap().error =
                Some("Connect to the server to read file information.".into());
            return;
        };
        self.info.as_mut().unwrap().loading = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let body = serde_json::json!({"schema_version":1,"file":file});
            let result = crate::async_runtime::runtime()
                .block_on(client.timeline_request(
                    &scope.0,
                    true,
                    "/v1/video/loras/info",
                    Some(&body),
                ))
                .map_err(|e| message(&e.code))
                .and_then(|v| serde_json::from_value(v).map_err(|_| message("invalid_response")));
            let _ = tx.send(Reply {
                scope,
                nonce,
                payload: Payload::Info(file, result),
            });
            ctx.request_repaint();
        });
    }
    pub fn close_info(&mut self) {
        self.info = None;
        next(&mut self.info_nonce);
    }
    /// Stop/reap only this editor's optional helper. Catalog/metadata remain cached.
    pub fn close_filter(&mut self) {
        self.filter_engine = None;
        next(&mut self.filter_nonce);
        self.filter_key = None;
        self.filter_matches = None;
        self.filter_pending = false;
        self.filter_error = None;
    }
    fn accept_filter(&mut self, reply: filter::Reply) {
        if reply.nonce != self.filter_nonce || !self.filter_pending {
            return;
        }
        self.filter_pending = false;
        match reply.result {
            Ok(matches) => {
                self.filter_matches = Some(matches);
                self.filter_error = None;
            }
            Err(error) => {
                self.filter_matches = None;
                self.filter_error = Some(error);
                self.filter_engine = None;
            }
        }
    }
    pub fn filtered<'a>(
        &mut self,
        catalog: &'a Catalog,
        pattern: &str,
        ctx: &egui::Context,
    ) -> Filter<'a> {
        while let Some(reply) = self
            .filter_engine
            .as_ref()
            .and_then(filter::Engine::try_reply)
        {
            self.accept_filter(reply);
        }
        let key = (self.catalog_generation, pattern.to_owned());
        if self.filter_key.as_ref() != Some(&key) {
            // A stuck obsolete expression cannot prevent the next user choice.
            // Completed helpers are reused; only in-flight supersession restarts.
            if self.filter_pending {
                self.filter_engine = None;
            }
            self.filter_key = Some(key);
            self.filter_matches = None;
            self.filter_error = None;
            self.filter_pending = false;
            let nonce = next(&mut self.filter_nonce);
            if !pattern.is_empty() {
                #[cfg(test)]
                let program = self
                    .filter_program
                    .as_deref()
                    .unwrap_or(std::ffi::OsStr::new("node"));
                #[cfg(not(test))]
                let program = std::ffi::OsStr::new("node");
                let result = (|| {
                    if self.filter_engine.is_none() {
                        self.filter_engine = Some(filter::Engine::start(program)?);
                    }
                    self.filter_engine.as_ref().unwrap().submit(
                        nonce,
                        catalog
                            .files
                            .iter()
                            .map(|entry| entry.file.clone())
                            .collect(),
                        pattern.to_owned(),
                        ctx,
                    )
                })();
                match result {
                    Ok(()) => {
                        self.filter_pending = true;
                        #[cfg(test)]
                        {
                            self.filter_requests += 1;
                        }
                    }
                    Err(error) => {
                        self.filter_engine = None;
                        self.filter_error = Some(error);
                    }
                }
            }
        }
        if self.filter_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
        }
        Filter {
            entries: self.filter_matches.as_ref().map_or_else(
                || all_entries(catalog),
                |matches| {
                    matches
                        .indices
                        .iter()
                        .filter_map(|index| catalog.files.get(*index))
                        .collect()
                },
            ),
            invalid: self
                .filter_matches
                .as_ref()
                .is_some_and(|matches| matches.invalid),
            pending: self.filter_pending,
            error: self.filter_error.map(|error| error.message().to_owned()),
        }
    }
    pub fn info_window(&mut self, ctx: &egui::Context, client: Option<&Client>) {
        let Some(view) = self.info.as_ref() else {
            return;
        };
        let file = view.file.clone();
        let mut open = true;
        let mut retry = false;
        let response = egui::Modal::new(egui::Id::new("h3-lora-info")).show(ctx, |ui| {
            let _canvas = helto::window_canvas(ui);
            ui.set_width((ctx.content_rect().width() - 48.).clamp(180., 460.));
            ui.set_height((ctx.content_rect().height() - 180.).max(240.));
            ui.horizontal(|ui| {
                ui.heading("LoRA info");
                let r = helto::icon_button(
                    ui,
                    crate::icons::Icon::Close,
                    "Close LoRA information",
                    false,
                );
                super::control(ui, "lora-info-close", &r);
                if r.clicked() {
                    open = false;
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("h3-lora-info-scroll")
                .auto_shrink([false, false])
                .max_height(ui.available_height().max(150.))
                .show(ui, |ui| {
                    ui.scope(|ui| {
                        field(ui, "File", &view.file);
                        if view.loading {
                            ui.label("Reading file information…");
                        }
                        if let Some(error) = &view.error {
                            ui.colored_label(helto::ERROR, error);
                        }
                        if let Some(value) = &view.value {
                            for (label, text) in [
                                ("Name", &value.name),
                                ("Base model", &value.base_model),
                                ("Type", &value.model_type),
                                ("Format", &value.format),
                                ("SHA-256", &value.sha256),
                            ] {
                                field(ui, label, text);
                            }
                            field(ui, "Triggers", &value.trained_words.join(", "));
                        }
                        let r = ui
                            .add_enabled_ui(!view.loading, |ui| {
                                helto::icon_button(
                                    ui,
                                    crate::icons::Icon::Retry,
                                    "Retry LoRA information",
                                    false,
                                )
                            })
                            .inner;
                        super::control(ui, "lora-info-retry", &r);
                        retry = r.clicked();
                    });
                });
        });
        if response.should_close() {
            open = false;
        }
        if !open {
            self.close_info();
        } else if retry {
            self.inspect(file, client, ctx);
        }
    }
}
fn field(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.weak(label);
    ui.add(egui::Label::new(if value.is_empty() { "—" } else { value }).wrap());
}
pub(super) fn short_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}
/// Director's case-sensitive regex filter; invalid patterns keep the catalog usable.
pub(super) struct Filter<'a> {
    pub entries: Vec<&'a Entry>,
    pub invalid: bool,
    pub pending: bool,
    pub error: Option<String>,
}
fn all_entries(catalog: &Catalog) -> Vec<&Entry> {
    catalog
        .files
        .iter()
        .filter(|entry| !entry.file.is_empty())
        .collect()
}
/// LoRA names are not private; they show with Hide previews on.
pub(super) fn picker(
    ui: &mut egui::Ui,
    index: usize,
    name: &mut String,
    choices: Option<&Filter<'_>>,
) -> egui::Response {
    egui::ComboBox::from_id_salt("catalog-choice")
        .selected_text("Choose file…")
        .width((ui.available_width() - 64.).max(100.))
        .show_ui(ui, |ui| {
            ui.scope(|ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                if let Some(choices) = choices {
                    for entry in &choices.entries {
                        let r = ui
                            .selectable_label(*name == entry.file, short_name(&entry.file))
                            .on_hover_ui(|ui| {
                                ui.add(egui::Label::new(&entry.file).wrap());
                            });
                        super::control(ui, &format!("lora-{index}-choice-{}", entry.file), &r);
                        if r.clicked() {
                            *name = entry.file.clone();
                        }
                    }
                }
            });
        })
        .response
}

#[cfg(test)]
mod tests {
    use super::*;
    fn catalog() -> Catalog {
        Catalog {
            files: [
                "people/Face.safetensors",
                "style/face.weights",
                "style/ink.weights",
            ]
            .into_iter()
            .map(|f| Entry {
                file: f.into(),
                size: 1,
            })
            .collect(),
            warnings: vec![],
            root: None,
        }
    }
    fn scope(name: &str) -> Scope {
        (name.into(), "project".into())
    }
    fn completed<'a>(
        state: &mut State,
        catalog: &'a Catalog,
        pattern: &str,
        ctx: &egui::Context,
    ) -> Filter<'a> {
        let start = std::time::Instant::now();
        loop {
            let result = state.filtered(catalog, pattern, ctx);
            if !result.pending {
                return result;
            }
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "synthetic filter fixture did not complete"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    #[test]
    fn filter_matches_director_case_regex_and_lookahead() {
        let c = catalog();
        let mut state = State::default();
        let ctx = egui::Context::default();
        assert_eq!(completed(&mut state, &c, "face", &ctx).entries.len(), 1);
        let invalid = completed(&mut state, &c, "(?i)face", &ctx);
        assert!(invalid.invalid);
        assert_eq!(invalid.entries.len(), 3);
        assert_eq!(
            completed(&mut state, &c, "style/(?=ink)", &ctx).entries[0].file,
            "style/ink.weights"
        );
        assert_eq!(completed(&mut state, &c, "missing", &ctx).entries.len(), 0);
        assert_eq!(state.filtered(&c, "", &ctx).entries.len(), 3);
    }
    #[test]
    fn invalid_filter_preserves_all_choices() {
        let c = catalog();
        let f = completed(&mut State::default(), &c, "[", &egui::Context::default());
        assert!(f.invalid);
        assert_eq!(f.entries.len(), 3);
    }
    #[test]
    fn filter_cache_reuses_helper_over_repaints_and_failed_catalog_refresh() {
        let c = catalog();
        let ctx = egui::Context::default();
        let mut state = State::default();
        state.bind(Some(scope("a")));
        completed(&mut state, &c, "face", &ctx);
        let child = state.filter_engine.as_ref().unwrap().child_id();
        for _ in 0..20 {
            let result = state.filtered(&c, "face", &ctx);
            assert!(!result.pending);
            assert_eq!(result.entries.len(), 1);
        }
        assert_eq!(state.filter_requests, 1);
        assert_eq!(state.filter_engine.as_ref().unwrap().child_id(), child);
        state.accept(Reply {
            scope: scope("a"),
            nonce: state.catalog_nonce,
            payload: Payload::Catalog(Err("retry".into())),
        });
        completed(&mut state, &c, "ink", &ctx);
        assert_eq!(state.filter_requests, 2);
        assert_eq!(state.filter_engine.as_ref().unwrap().child_id(), child);
        state.close_filter();
        assert!(state.filter_engine.is_none());
    }
    #[test]
    fn filter_stale_replies_and_catalog_scope_changes_cannot_replace_current_choices() {
        let c = catalog();
        let ctx = egui::Context::default();
        let mut state = State::default();
        state.bind(Some(scope("a")));
        state.filtered(&c, "face", &ctx);
        let old = state.filter_nonce;
        completed(&mut state, &c, "ink", &ctx);
        state.accept_filter(filter::Reply {
            nonce: old,
            result: Ok(filter::Matches {
                indices: vec![0],
                invalid: false,
            }),
        });
        assert_eq!(
            state.filtered(&c, "ink", &ctx).entries[0].file,
            "style/ink.weights"
        );
        let old = state.filter_nonce;
        state.bind(Some(scope("b")));
        state.accept_filter(filter::Reply {
            nonce: old,
            result: Ok(filter::Matches {
                indices: vec![0],
                invalid: false,
            }),
        });
        assert!(state.filter_matches.is_none());
        assert!(state.filter_engine.is_none());
        completed(&mut state, &c, "face", &ctx);
        let old = state.filter_nonce;
        state.accept(Reply {
            scope: scope("b"),
            nonce: state.catalog_nonce,
            payload: Payload::Catalog(Ok(c.clone())),
        });
        assert!(state.filter_key.is_none());
        state.accept_filter(filter::Reply {
            nonce: old,
            result: Ok(filter::Matches {
                indices: vec![0],
                invalid: false,
            }),
        });
        assert!(state.filter_matches.is_none());
    }
    #[test]
    fn missing_engine_is_cached_all_choices_and_empty_filter_needs_no_engine() {
        let c = catalog();
        let ctx = egui::Context::default();
        let mut state = State {
            filter_program: Some("/own/nonexistent-node-engine".into()),
            ..Default::default()
        };
        assert_eq!(state.filtered(&c, "", &ctx).entries.len(), 3);
        assert!(state.filter_engine.is_none());
        for _ in 0..3 {
            let result = state.filtered(&c, "face", &ctx);
            assert_eq!(result.entries.len(), 3);
            assert!(result.error.unwrap().contains("optional Node.js"));
            assert!(!result.pending);
        }
        assert_eq!(state.filter_requests, 0);
    }
    #[test]
    fn refresh_error_keeps_last_good_catalog() {
        let mut state = State::default();
        state.bind(Some(scope("a")));
        state.catalog = Some(catalog());
        state.accept(Reply {
            scope: scope("a"),
            nonce: state.catalog_nonce,
            payload: Payload::Catalog(Err("retry".into())),
        });
        assert_eq!(state.catalog.unwrap().files.len(), 3);
        assert_eq!(state.error.as_deref(), Some("retry"));
    }
    #[test]
    fn scope_and_request_nonce_reject_late_results() {
        let mut state = State::default();
        state.bind(Some(scope("a")));
        let old = state.catalog_nonce;
        state.bind(Some(scope("b")));
        state.accept(Reply {
            scope: scope("a"),
            nonce: old,
            payload: Payload::Catalog(Ok(catalog())),
        });
        assert!(state.catalog.is_none());
        state.accept(Reply {
            scope: scope("b"),
            nonce: old,
            payload: Payload::Catalog(Ok(catalog())),
        });
        assert!(state.catalog.is_none());
    }
    #[test]
    fn closed_info_does_not_reopen_for_late_reply() {
        let mut state = State::default();
        state.bind(Some(scope("a")));
        state.info = Some(InfoView {
            file: "manual.any".into(),
            ..Default::default()
        });
        let old = state.info_nonce;
        state.close_info();
        state.accept(Reply {
            scope: scope("a"),
            nonce: old,
            payload: Payload::Info("manual.any".into(), Err("retry".into())),
        });
        assert!(state.info.is_none());
    }
    #[test]
    fn short_names_do_not_change_full_paths() {
        assert_eq!(short_name("a/b.weights"), "b.weights");
        assert_eq!(short_name(r"a\b.weights"), "b.weights");
    }
    #[test]
    fn unknown_error_text_is_never_displayed() {
        assert!(!message("private/path/secret").contains("secret"));
    }
    #[test]
    fn info_window_shows_paths_and_metadata_without_hover() {
        // LoRA names and metadata are not private: they show with Hide
        // previews on, in an unfocused window, without the pointer.
        fn painted(shape: &egui::Shape) -> bool {
            match shape {
                egui::Shape::Text(t) => {
                    t.opacity_factor > 0. && t.galley.text().contains("SYNTHETIC_NAME")
                }
                egui::Shape::Vec(shapes) => shapes.iter().any(painted),
                _ => false,
            }
        }
        let ctx = egui::Context::default();
        helto::install(&ctx);
        helto::set_privacy_mode(&ctx, true);
        let mut state = State {
            info: Some(InfoView {
                file: "SYNTHETIC_NAME/own.weights".into(),
                value: Some(Info {
                    name: "SYNTHETIC_NAME metadata".into(),
                    trained_words: vec!["SYNTHETIC_NAME trigger".into()],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        for width in [600., 960.] {
            for frame in 0..3 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 600.),
                        )),
                        focused: false,
                        ..Default::default()
                    },
                    |ui| state.info_window(ui.ctx(), None),
                );
                output.textures_delta.clear();
                if frame == 2 {
                    assert!(output.shapes.iter().any(|s| painted(&s.shape)));
                }
            }
        }
    }
    #[test]
    fn metadata_error_leaves_explicit_manual_file_unchanged() {
        let mut state = State::default();
        state.bind(Some(scope("a")));
        let file = "/manual/compatible.custom".to_owned();
        state.info = Some(InfoView {
            file: file.clone(),
            loading: true,
            ..Default::default()
        });
        state.accept(Reply {
            scope: scope("a"),
            nonce: state.info_nonce,
            payload: Payload::Info(file.clone(), Err(message("lora_info_format"))),
        });
        let info = state.info.unwrap();
        assert_eq!(info.file, file);
        assert!(!info.loading);
        assert!(info.error.is_some());
    }
    #[tokio::test]
    async fn isolated_http_instance_checks_catalog_info_protocol_and_retry_errors() {
        use axum::{
            Json, Router,
            http::{HeaderMap, StatusCode},
            routing::{get, post},
        };
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_route = calls.clone();
        let app=Router::new().route("/v1/video/loras",get(move |headers:HeaderMap| {
            let calls=calls_route.clone();
            async move {
                assert_eq!(headers["x-imagine-session"],"synthetic-session");
                let n=calls.fetch_add(1,Ordering::SeqCst);
                if n==0 {(StatusCode::OK,Json(serde_json::json!({"files":[{"file":"own/custom.weights","size":8}],"warnings":[]})))}
                else if n==1 {(StatusCode::SERVICE_UNAVAILABLE,Json(serde_json::json!({"error":{"code":"lora_catalog_unavailable","message":"DO_NOT_SHOW_PRIVATE_PATH"}})))}
                else {(StatusCode::OK,Json(serde_json::json!({"broken":true})))}
            }
        })).route("/v1/video/loras/info",post(|headers:HeaderMap,Json(body):Json<serde_json::Value>|async move {
            assert_eq!(headers["x-imagine-session"],"synthetic-session");
            assert_eq!(body,serde_json::json!({"schema_version":1,"file":"/manual/alternative.custom"}));
            (StatusCode::UNPROCESSABLE_ENTITY,Json(serde_json::json!({"error":{"code":"lora_info_format","message":"DO_NOT_SHOW_PRIVATE_PATH"}})))
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = Client::connect(&crate::settings::Connection { url }).unwrap();
        let ctx = egui::Context::default();
        let scope = ("synthetic-session".into(), "synthetic-project".into());
        let mut state = State::default();
        state.bind(Some(scope.clone()));
        for n in 0..3 {
            state.refresh(Some(&client), &ctx);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while state.loading && std::time::Instant::now() < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                state.bind(Some(scope.clone()));
            }
            assert!(!state.loading, "catalog request timed out");
            assert_eq!(
                state.catalog.as_ref().unwrap().files[0].file,
                "own/custom.weights"
            );
            if n > 0 {
                assert!(state.error.is_some());
                assert!(!state.error.as_ref().unwrap().contains("PRIVATE_PATH"));
            }
        }
        state.inspect("/manual/alternative.custom".into(), Some(&client), &ctx);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while state.info.as_ref().unwrap().loading && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            state.bind(Some(scope.clone()));
        }
        let view = state.info.unwrap();
        assert!(!view.loading);
        assert_eq!(view.file, "/manual/alternative.custom");
        assert!(view.error.unwrap().contains("manual file selection"));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        task.abort();
    }
    #[test]
    fn open_catalog_popup_shows_lora_names_with_hide_previews_on() {
        fn painted(shape: &egui::Shape) -> bool {
            match shape {
                egui::Shape::Text(t) => {
                    t.opacity_factor > 0. && t.galley.text().contains("OWN_CHOICE")
                }
                egui::Shape::Vec(s) => s.iter().any(painted),
                _ => false,
            }
        }
        let ctx = egui::Context::default();
        helto::install(&ctx);
        helto::set_privacy_mode(&ctx, true);
        let c = Catalog {
            files: vec![Entry {
                file: "own/OWN_CHOICE.weights".into(),
                size: 1,
            }],
            warnings: vec![],
            root: None,
        };
        let f = State::default().filtered(&c, "", &ctx);
        let mut name = String::new();
        let mut draw = |events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(600., 600.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    picker(ui, 0, &mut name, Some(&f));
                },
            );
            output.textures_delta.clear();
            output
        };
        draw(vec![]);
        draw(vec![]);
        let pos = egui::pos2(50., 18.);
        draw(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        draw(vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        draw(vec![]);
        // Moving off the list and the window keeps the names readable.
        draw(vec![egui::Event::PointerMoved(egui::pos2(590., 590.))]);
        let output = draw(vec![]);
        assert!(output.shapes.iter().any(|s| painted(&s.shape)));
    }
}
