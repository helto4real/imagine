//! Project defaults shared by the UI and the server; no I/O.
use imagine_video_contract::contract::{CreateProject, DeliverySettings, Time};
use imagine_video_contract::generation::Canvas;

/// A new project: the default canvas, Quick Draft 16:9, for five seconds.
pub fn new_project() -> CreateProject {
    with_canvas(Canvas::default()).expect("default video geometry")
}
/// A new project on `canvas`; delivery size follows the canvas as the UI resolves it.
pub fn with_canvas(canvas: Canvas) -> Result<CreateProject, &'static str> {
    let shape = crate::h3_protocol::Selection {
        ratio: canvas.ratio.clone(),
        portrait: canvas.portrait,
        quality: canvas.quality.clone(),
        seconds: 5.,
    }
    .resolve()?;
    Ok(CreateProject {
        generation: Some(imagine_video_contract::generation::Defaults {
            canvas: Some(canvas),
            ..Default::default()
        }),
        schema_version: 1,
        default_model: imagine_video_contract::generation::h3_model("reference"),
        delivery: DeliverySettings {
            width: shape.width,
            height: shape.height,
            fps: Time { num: 24, den: 1 },
        },
    })
}
