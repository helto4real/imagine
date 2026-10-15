use anyhow::{Context, Result};
use imagine_ui::{
    app::ImagineApp,
    settings::{Connection, DEFAULT_API_URL, Preferences},
};
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut project = std::env::current_dir()?;
    let mut url = DEFAULT_API_URL.to_owned();
    let mut access = None;
    let mut client = "local".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" => project = PathBuf::from(args.next().context("--project needs a path")?),
            "--api" => url = args.next().context("--api needs a URL")?,
            "--access" => {
                access = Some(PathBuf::from(args.next().context("--access needs a path")?))
            }
            "--client" => client = args.next().context("--client needs a name")?,
            "--help" => {
                println!(
                    "imagine-ui [--project PATH] [--api http://127.0.0.1:8189/imagine] [--access PATH] [--client local]"
                );
                return Ok(());
            }
            _ => anyhow::bail!("Unknown option: {arg}"),
        }
    }
    let project = project.canonicalize()?;
    let prefs = Preferences::load(&project)?;
    let connection = Connection {
        url,
        access: access.unwrap_or_else(|| project.join(".imagine/access.json")),
        client,
    };
    connection.validate()?;
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Imagine")
            .with_inner_size([1180., 760.])
            .with_min_inner_size([600., 540.]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Imagine",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ImagineApp::new(
                project,
                connection,
                prefs,
                &cc.egui_ctx,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Could not open the window: {e}"))
}
