//! Route adapter for descriptor-only ingress; inherited session/no-store guards stay intact.
use super::super::streaming_import;
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub(super) async fn upload(app: App, request: Request) -> Result<(StatusCode, Json<Value>), Error> {
    let mime = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    if !media::MIME_TYPES.contains(&mime.as_str()) {
        return Err(Error::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
        ));
    }
    let expected = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .map(|v| {
            v.to_str()
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| Error::new(StatusCode::BAD_REQUEST, "invalid_content_length"))
        })
        .transpose()?;
    let _permit = app
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::new(StatusCode::TOO_MANY_REQUESTS, "upload_busy"))?;
    let target = staging(&app)?.temporary()?;
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(flag.clone());
    let stream = request
        .into_body()
        .into_data_stream()
        .map(|chunk| chunk.map_err(std::io::Error::other));
    let mut reader = tokio_util::io::StreamReader::new(stream);
    let receipt = streaming_import::receive(
        &mut reader,
        target,
        expected,
        &flag,
        std::time::Duration::from_secs(30),
    )
    .await
    .map_err(|error| {
        let status = if error == streaming_import::Error::Space {
            StatusCode::INSUFFICIENT_STORAGE
        } else {
            StatusCode::UNPROCESSABLE_ENTITY
        };
        Error::new(status, error.code())
    })?;
    let digest = receipt.sha256;
    let bytes = receipt.bytes;
    let file = receipt.file;
    let (file, info) = tokio::task::spawn_blocking(move || {
        media::inspect_streamed(&file, &mime).map(|info| (file, info))
    })
    .await
    .map_err(internal)?
    .map_err(media_error)?;
    let result = app
        .store
        .register_media("local", file, &info)
        .await
        .map_err(internal)?;
    if result["sha256"].as_str() != Some(digest.as_str()) || result["bytes"].as_u64() != Some(bytes)
    {
        return Err(media_error("asset_content_changed"));
    }
    Ok((StatusCode::CREATED, Json(result)))
}
