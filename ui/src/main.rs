#[cfg(feature = "ui-preview")]
use anyhow::Context;
use anyhow::Result;
use imagine_ui::{
    app::ImagineApp,
    settings::{Connection, DEFAULT_API_URL, PreferencesStore},
};
use std::path::PathBuf;
const USAGE: &str = "imagine-ui [--project PATH] [--config PATH] [--api http://127.0.0.1:8189/imagine] [--title TITLE]";
/// A command-line mistake. Only these are printed; other startup errors can
/// name private paths and stay generic.
#[derive(Debug)]
struct Usage(String);
impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Usage {}
fn usage(message: impl Into<String>) -> anyhow::Error {
    Usage(message.into()).into()
}
fn main() {
    if let Err(error) = run() {
        match error.downcast_ref::<Usage>() {
            Some(usage) => eprintln!("imagine-ui: {usage}\n{USAGE}"),
            None => eprintln!("Imagine UI could not start or close successfully."),
        }
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    // Private text/previews remain in process memory, never an automatic core file.
    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) } != 0 {
        anyhow::bail!("private_process_initialization_failed");
    }
    let mut project = std::env::current_dir()?;
    let mut config_dir: Option<PathBuf> = None;
    let mut url = DEFAULT_API_URL.to_owned();
    let mut title = "Imagine".to_owned();
    #[cfg(feature = "ui-preview")]
    let mut qa_script: Option<PathBuf> = None;
    #[cfg(feature = "ui-preview")]
    let mut qa_window_size: Option<[f32; 2]> = None;
    #[cfg(feature = "ui-preview")]
    let mut qa_manual_picker_root: Option<PathBuf> = None;
    #[cfg(feature = "ui-preview")]
    let mut qa_private_home: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            #[cfg(feature = "ui-preview")]
            "--qa-private-home" => {
                qa_private_home = Some(PathBuf::from(
                    args.next()
                        .context("--qa-private-home needs an own case directory")?,
                ));
            }
            #[cfg(feature = "ui-preview")]
            "--qa-window-size" => {
                let width = args
                    .next()
                    .context("--qa-window-size needs width and height")?
                    .parse()?;
                let height = args
                    .next()
                    .context("--qa-window-size needs height")?
                    .parse()?;
                qa_window_size = Some([width, height]);
            }
            #[cfg(feature = "ui-preview")]
            "--qa-manual-picker-root" => {
                qa_manual_picker_root = Some(PathBuf::from(
                    args.next()
                        .context("--qa-manual-picker-root needs a directory")?,
                ));
            }
            #[cfg(feature = "ui-preview")]
            "--qa-script" => {
                qa_script = Some(PathBuf::from(
                    args.next().context("--qa-script needs a path")?,
                ))
            }
            "--config" => {
                config_dir = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage("--config needs a path"))?,
                ))
            }
            "--project" => {
                project = PathBuf::from(args.next().ok_or_else(|| usage("--project needs a path"))?)
            }
            "--api" => url = args.next().ok_or_else(|| usage("--api needs a URL"))?,
            "--title" => {
                title = args
                    .next()
                    .ok_or_else(|| usage("--title needs a window title"))?
            }
            "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(usage(format!("Unknown option: {arg}"))),
        }
    }
    // Checked before any settings are opened.
    let connection = Connection { url };
    connection
        .validate()
        .map_err(|error| usage(error.to_string()))?;
    let project = project.canonicalize()?;
    // The Director prompt catalog is read once; a restart applies edits. As in
    // Director, a missing or invalid catalog disables `#` completion only.
    imagine_ui::config_files::init(config_dir.unwrap_or_else(|| project.join("config")));
    // Only explicit QA inputs use the old source-owned fixture API. The normal
    // entry never reads project/.imagine/ui-settings.json or migrates user data.
    #[cfg(feature = "ui-preview")]
    let fixture_settings = qa_script.is_some() || qa_manual_picker_root.is_some();
    #[cfg(not(feature = "ui-preview"))]
    let fixture_settings = false;
    #[cfg(feature = "ui-preview")]
    let injected_store = qa_private_home
        .as_deref()
        .map(private_qa_store)
        .transpose()?;
    #[cfg(not(feature = "ui-preview"))]
    let injected_store: Option<PreferencesStore> = None;
    let settings_store = if let Some(store) = injected_store {
        store
    } else if fixture_settings {
        PreferencesStore::Fixture(project.clone())
    } else {
        PreferencesStore::system()?
    };
    let (prefs, settings_warning) = settings_store.load()?;
    let mut options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(title.clone())
            .with_inner_size([1280., 800.])
            .with_min_inner_size([600., 540.]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    // NVIDIA EGL/Wayland can block the UI thread in swap_buffers on a hidden workspace.
    options.glow_options.vsync = false;
    #[cfg(feature = "ui-preview")]
    if let Some(size) = qa_window_size {
        // Ordinary application size hints for our isolated visual QA window.
        // No compositor configuration or other window is queried or modified.
        options.viewport = options
            .viewport
            .with_inner_size(size)
            .with_min_inner_size(size)
            .with_max_inner_size(size)
            .with_resizable(false);
    }
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            let app = ImagineApp::new(project, connection, prefs, &cc.egui_ctx)
                .with_preferences_store(settings_store, settings_warning);
            #[cfg(feature = "ui-preview")]
            let mut app = app;
            #[cfg(feature = "ui-preview")]
            if let Some(path) = qa_manual_picker_root {
                app.install_manual_picker_root(&path)?;
            }
            #[cfg(feature = "ui-preview")]
            if let Some(path) = qa_script {
                app.install_timeline_qa(&path)?;
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Could not open the window: {e}"))
}

/// Explicit source-owned Home-branch QA. No production mount policy or HOME
/// environment is changed, and this entry point does not exist in normal builds.
#[cfg(feature = "ui-preview")]
fn private_qa_store(case: &std::path::Path) -> Result<PreferencesStore> {
    use imagine_ui::{private_storage::Root, video_destination as d};
    use std::sync::Arc;
    let case = case.canonicalize()?;
    anyhow::ensure!(
        case.starts_with("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"),
        "Only own private QA case directories are allowed"
    );
    struct Environment(String);
    impl d::Environment for Environment {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-ui-private-home".into())
        }
    }
    let config = d::Config {
        root: case.join("private"),
        mount: case.clone(),
        filesystem: "fixturefs".into(),
        source: "own-ui-private-home".into(),
        fsid: d::fsid(&d::directory(&case).map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?,
    };
    let runtime = case.join("runtime/imagine-ui");
    let private = Root::with_environment(
        config,
        Arc::new(Environment(format!(
            "1 0 0:1 / {} rw - fixturefs own-ui-private-home rw\n",
            case.display()
        ))),
    )
    .and_then(|root| root.with_runtime_directory(runtime.clone()));
    Ok(PreferencesStore::for_home(
        &case.join("home"),
        runtime,
        private,
    ))
}
