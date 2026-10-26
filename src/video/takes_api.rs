//! Take selection with continuation restore, and bounded frame previews for agents.
use super::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::{fs::File, sync::atomic::AtomicBool};

pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/projects/{id}/takes/{take}/select", post(select))
        .route("/v1/video/projects/{id}/takes/{take}/frames", get(frames))
}

async fn select(
    State(app): State<App>,
    Path((id, take)): Path<(String, String)>,
    body: Result<Json<super::super::review::SelectRequest>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    let id = ProjectId(id);
    let take = TakeId(take);
    let mut service = app.video.lock().await;
    let session = service.projects.session_mut(&id)?;
    let selected = super::super::review::select(session.document(), &take, request)?;
    session.publish(selected.document)?;
    let restored = selected
        .restored
        .iter()
        .map(|(scene, take)| json!({"scene_id":scene,"take_id":take}))
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "schema_version": 1,
        "selected": take,
        "prompt_restored": selected.prompt_restored,
        "restored": restored,
        "needs_render": selected.needs_render,
        "final_video_outdated": selected.final_video_outdated,
        "snapshot": service.snapshot(&id)?,
    })))
}

/// Longest side of a preview frame.
const FRAME_SIDE: u32 = 768;

/// One PNG frame: the first, the middle, or the last of the video.
fn frame(file: &File, position: &str) -> Result<Vec<u8>, StorageError> {
    let scale = format!(
        "scale=w='min({FRAME_SIDE},iw)':h='min({FRAME_SIDE},ih)':force_original_aspect_ratio=decrease"
    );
    let middle;
    let (input, filter): (Vec<&str>, String) = match position {
        "first" => (vec![], scale),
        "middle" => {
            let duration = media::detect_streamed(file)
                .ok()
                .and_then(|info| info.duration)
                .map_or(0.0, |d| d.num as f64 / d.den as f64);
            middle = format!("{:.3}", duration / 2.0);
            (vec!["-ss", &middle], scale)
        }
        // Decode only the last half second and take its final frame.
        _ => (vec!["-sseof", "-0.5"], format!("reverse,{scale}")),
    };
    let cancel = AtomicBool::new(false);
    super::super::assembly::media::decode_with_input(
        file,
        &input,
        &[
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            &filter,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
        ],
        16 * 1024 * 1024,
        &cancel,
    )
    .ok()
    .filter(|png| png.starts_with(b"\x89PNG"))
    .ok_or_else(|| StorageError::new("take_frame_unavailable"))
}

fn take_video(session: &ProjectSession, take: &TakeId) -> Result<File, StorageError> {
    let resource = session
        .document()
        .resources
        .iter()
        .find(|r| matches!(&r.kind, ResourceKind::TakeVideo { take_id } if take_id == take))
        .ok_or_else(|| StorageError::new("take_resource_missing"))?;
    session.resource_file(&resource.id)
}

/// The first, middle and last frame of a Take; for a continuation also the last frame
/// of the source Take it continued from, to judge the join.
async fn frames(
    State(app): State<App>,
    Path((id, take)): Path<(String, String)>,
) -> Result<Json<Value>, Error> {
    let take = TakeId(take);
    let (scene, video, source) = {
        let service = app.video.lock().await;
        let session = service.projects.session(&ProjectId(id))?;
        let stored = session
            .document()
            .takes
            .iter()
            .find(|t| t.id == take)
            .ok_or_else(|| StorageError::new("take_not_found"))?;
        let source = stored
            .provenance
            .as_ref()
            .and_then(|p| p.source_take.clone())
            .and_then(|source| take_video(session, &source).ok().map(|f| (source, f)));
        (stored.scene_id.clone(), take_video(session, &take)?, source)
    };
    let frames = tokio::task::spawn_blocking(move || {
        let png = |position: &str, take: &TakeId, file: &File| {
            frame(file, position).map(|bytes| {
                json!({"position":position,"take_id":take,"mime":"image/png","base64":STANDARD.encode(bytes)})
            })
        };
        let mut frames = ["first", "middle", "last"]
            .into_iter()
            .map(|position| png(position, &take, &video))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some((source, file)) = source {
            let mut last = png("last", &source, &file)?;
            last["position"] = json!("source_last");
            frames.insert(0, last);
        }
        Ok::<_, StorageError>(json!({"schema_version":1,"take_id":take,"scene_id":scene,"frames":frames}))
    })
    .await
    .map_err(internal)??;
    Ok(Json(frames))
}
