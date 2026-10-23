use anyhow::{Context, Result};
use imagine_ui::{
    app::ImagineApp,
    settings::{Connection, DEFAULT_API_URL, Preferences},
};
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut project = std::env::current_dir()?;
    let mut url = DEFAULT_API_URL.to_owned();
    let mut title = "Imagine".to_owned();
    #[cfg(feature = "ui-preview")]
    let mut qa_script: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            #[cfg(feature = "ui-preview")]
            "--qa-script" => {
                qa_script = Some(PathBuf::from(
                    args.next().context("--qa-script needs a path")?,
                ))
            }
            "--project" => project = PathBuf::from(args.next().context("--project needs a path")?),
            "--api" => url = args.next().context("--api needs a URL")?,
            "--title" => title = args.next().context("--title needs a window title")?,
            "--help" => {
                println!(
                    "imagine-ui [--project PATH] [--api http://127.0.0.1:8189/imagine] [--title TITLE]"
                );
                return Ok(());
            }
            _ => anyhow::bail!("Unknown option: {arg}"),
        }
    }
    let project = project.canonicalize()?;
    let prefs = Preferences::load(&project)?;
    let connection = Connection { url };
    connection.validate()?;
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
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            let app = ImagineApp::new(project, connection, prefs, &cc.egui_ctx);
            #[cfg(feature = "ui-preview")]
            let mut app = app;
            #[cfg(feature = "ui-preview")]
            if let Some(path) = qa_script {
                app.install_timeline_qa(&path)?;
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Could not open the window: {e}"))
}
