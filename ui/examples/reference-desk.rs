//! Explicit opt-in native acceptance, with no normal preferences, assets or API.
use imagine_ui::app::ImagineApp;

fn main() -> eframe::Result {
    let temporary = tempfile::tempdir().expect("isolated preview directory");
    let project = temporary.path().to_owned();
    let mut options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Imagine — synthetic UI preview")
            .with_inner_size([1280., 800.])
            .with_min_inner_size([600., 540.]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    // Match the application's non-waiting EGL presentation on hidden workspaces.
    options.glow_options.vsync = false;
    eframe::run_native(
        "Imagine — synthetic UI preview",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ImagineApp::synthetic_preview(
                &cc.egui_ctx,
                project,
            )))
        }),
    )
}
