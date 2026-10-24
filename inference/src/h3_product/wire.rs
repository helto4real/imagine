//! Bounded private framing and one-request control flow; no CUDA/tensor dependency.
use super::protocol::{self as p, Deployment, ErrorCode, Event, Request, Stream};
use std::io::{Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

pub type Emit<'a> = dyn FnMut(&Event, &[u8]) -> Result<(), ErrorCode> + 'a;
pub trait Generator {
    fn generate(
        &mut self,
        request: &Request,
        emit: &mut Emit<'_>,
        input: &mut dyn Read,
    ) -> Result<(), ErrorCode>;
}
/// Diagnostics for the one request this process serves: the current stage and
/// the first classified failure. Enums and integers only.
static FAILURE: std::sync::Mutex<p::Failure> = std::sync::Mutex::new(p::Failure {
    stage: None,
    kind: None,
    exception: None,
    accelerator_code: None,
    sizes: Vec::new(),
});
pub fn stage(stage: p::Stage) {
    if let Ok(mut failure) = FAILURE.lock()
        && failure.kind.is_none()
    {
        failure.stage = Some(stage);
    }
}
/// Keeps the first failure; later cleanup errors do not replace the cause.
pub fn record(
    kind: p::FailureKind,
    exception: Option<&str>,
    accelerator_code: Option<i32>,
    sizes: Vec<i64>,
) {
    if let Ok(mut failure) = FAILURE.lock()
        && failure.kind.is_none()
    {
        failure.kind = Some(kind);
        failure.exception = exception.map(str::to_owned);
        failure.accelerator_code = accelerator_code;
        failure.sizes = sizes;
    }
}
fn failure() -> Option<p::Failure> {
    FAILURE
        .lock()
        .ok()
        .map(|failure| failure.clone())
        .filter(|failure| !failure.is_empty() && failure.valid())
}
pub fn write(out: &mut impl Write, event: &Event, payload: &[u8]) -> Result<(), ErrorCode> {
    let header = serde_json::to_vec(event).map_err(|_| ErrorCode::GenerationFailed)?;
    if header.len() > p::MAX_HEADER || payload.len() > event.payload_limit() {
        return Err(ErrorCode::GenerationFailed);
    }
    out.write_all(&(header.len() as u32).to_be_bytes())
        .and_then(|_| out.write_all(&header))
        .and_then(|_| out.write_all(&(payload.len() as u32).to_be_bytes()))
        .and_then(|_| out.write_all(payload))
        .and_then(|_| out.flush())
        .map_err(|_| ErrorCode::GenerationFailed)
}
fn length(input: &mut impl Read) -> Result<usize, ErrorCode> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| ErrorCode::InvalidRequest)?;
    Ok(u32::from_be_bytes(bytes) as usize)
}
pub fn request(input: &mut impl Read) -> Result<Request, ErrorCode> {
    let n = length(input)?;
    if n == 0 || n > p::MAX_HEADER {
        return Err(ErrorCode::InvalidRequest);
    }
    let mut header = vec![0; n];
    input
        .read_exact(&mut header)
        .map_err(|_| ErrorCode::InvalidRequest)?;
    let size = length(input)?;
    let mut request: Request =
        serde_json::from_slice(&header).map_err(|_| ErrorCode::InvalidRequest)?;
    if !request.valid() {
        return Err(ErrorCode::InvalidRequest);
    }
    let image_size = request.image.as_ref().map_or(0, |v| v.bytes());
    let context_size = request
        .continuation
        .as_ref()
        .map_or(0, |v| v.descriptor.bytes());
    if size != image_size + context_size + request.references.bytes()
        || size > p::MAX_INPUT_IMAGE + p::MAX_PAYLOAD + p::references::MAX_BYTES
    {
        return Err(ErrorCode::InvalidRequest);
    }
    request.image_rgb = vec![0; image_size];
    input
        .read_exact(&mut request.image_rgb)
        .map_err(|_| ErrorCode::InvalidRequest)?;
    if let Some(image) = &request.image {
        use sha2::{Digest, Sha256};
        if format!("{:x}", Sha256::digest(&request.image_rgb)) != image.sha256 {
            return Err(ErrorCode::InvalidRequest);
        }
    }
    request.continuation_bytes = vec![0; context_size];
    input
        .read_exact(&mut request.continuation_bytes)
        .map_err(|_| ErrorCode::InvalidRequest)?;
    if let Some(context) = &request.continuation {
        use sha2::{Digest, Sha256};
        if !context.descriptor.valid_bytes(&request.continuation_bytes)
            || format!("{:x}", Sha256::digest(&request.continuation_bytes)) != context.sha256
        {
            return Err(ErrorCode::InvalidRequest);
        }
    }
    for image in &mut request.references.images {
        image.rgb = vec![0; image.bytes()];
        input
            .read_exact(&mut image.rgb)
            .map_err(|_| ErrorCode::InvalidRequest)?;
    }
    for video in &mut request.references.videos {
        video.rgb = vec![0; video.rgb_bytes()];
        input
            .read_exact(&mut video.rgb)
            .map_err(|_| ErrorCode::InvalidRequest)?;
        if let Some(audio) = &mut video.audio {
            audio.pcm = vec![0; audio.bytes()];
            input
                .read_exact(&mut audio.pcm)
                .map_err(|_| ErrorCode::InvalidRequest)?;
        }
    }
    for audio in &mut request.references.audios {
        audio.pcm = vec![0; audio.bytes()];
        input
            .read_exact(&mut audio.pcm)
            .map_err(|_| ErrorCode::InvalidRequest)?;
    }
    if !request.references.valid_payload() {
        return Err(ErrorCode::InvalidRequest);
    }
    Ok(request)
}
/// Factory invocation is the admission boundary. Startup/request rejection can
/// never construct the native runtime or read model files through this path.
pub fn serve<G: Generator>(
    input: &mut impl Read,
    output: &mut impl Write,
    deployment: Deployment,
    factory: impl FnOnce(&Request) -> Result<G, ErrorCode>,
) -> Result<(), ErrorCode> {
    if !deployment.valid() {
        return Err(ErrorCode::DeploymentMismatch);
    }
    write(output, &Event::Ready { deployment }, &[])?;
    let request = match request(input) {
        Ok(request) => request,
        Err(code) => {
            stage(p::Stage::Admission);
            let failure = failure();
            let _ = write(
                output,
                &Event::Error {
                    id: None,
                    code,
                    failure,
                },
                &[],
            );
            return Err(code);
        }
    };
    let id = request.id.clone();
    let mut stream = Stream::with_steps(id.clone(), request.shape, request.total_steps())
        .ok_or(ErrorCode::InvalidRequest)?;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut generator = factory(&request)?;
        let mut emit = |event: &Event, payload: &[u8]| {
            if matches!(
                event,
                Event::Ready { .. } | Event::Complete { .. } | Event::Error { .. }
            ) {
                return Err(ErrorCode::GenerationFailed);
            }
            stream
                .accept(event, payload)
                .map_err(|_| ErrorCode::GenerationFailed)?;
            write(output, event, payload)
        };
        generator.generate(&request, &mut emit, input)
    }))
    .unwrap_or_else(|_| {
        record(p::FailureKind::Panic, None, None, vec![]);
        Err(ErrorCode::GenerationFailed)
    });
    match result {
        Ok(()) => {
            let complete = Event::Complete {
                id: id.clone(),
                frames: request.shape.frames,
                audio_samples: request.shape.audio_samples(),
            };
            if (!request.capture_continuation || stream.captured_continuation())
                && stream.accept(&complete, &[]).is_ok()
            {
                return write(output, &complete, &[]);
            }
            let code = ErrorCode::GenerationFailed;
            let failure = failure();
            let _ = write(
                output,
                &Event::Error {
                    id: Some(id),
                    code,
                    failure,
                },
                &[],
            );
            Err(code)
        }
        Err(code) => {
            let failure = failure();
            let _ = write(
                output,
                &Event::Error {
                    id: Some(id),
                    code,
                    failure,
                },
                &[],
            );
            Err(code)
        }
    }
}

/// Used only by a dedicated worker process, never by the application or tests
/// in-process: no third-party panic text reaches its stderr/diagnostic channel.
pub fn quiet_entry(run: impl FnOnce() -> Result<(), ErrorCode>) -> bool {
    std::panic::set_hook(Box::new(|_| {}));
    catch_unwind(AssertUnwindSafe(run)).is_ok_and(|result| result.is_ok())
}
