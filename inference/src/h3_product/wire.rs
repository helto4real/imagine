//! Bounded private framing and one-request control flow; no CUDA/tensor dependency.
use super::protocol::{self as p, Deployment, ErrorCode, Event, Request, Stream};
use std::io::{Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

pub type Emit<'a> = dyn FnMut(&Event, &[u8]) -> Result<(), ErrorCode> + 'a;
pub trait Generator {
    fn generate(&mut self, request: &Request, emit: &mut Emit<'_>) -> Result<(), ErrorCode>;
}
pub fn write(out: &mut impl Write, event: &Event, payload: &[u8]) -> Result<(), ErrorCode> {
    let header = serde_json::to_vec(event).map_err(|_| ErrorCode::GenerationFailed)?;
    if header.len() > p::MAX_HEADER || payload.len() > p::MAX_PAYLOAD {
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
    // Request payload is empty: reject its length before allocating any body.
    if length(input)? != 0 {
        return Err(ErrorCode::InvalidRequest);
    }
    let request: Request =
        serde_json::from_slice(&header).map_err(|_| ErrorCode::InvalidRequest)?;
    if !request.valid() {
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
            let _ = write(output, &Event::Error { id: None, code }, &[]);
            return Err(code);
        }
    };
    let id = request.id.clone();
    let mut stream = Stream::new(id.clone()).ok_or(ErrorCode::InvalidRequest)?;
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
        generator.generate(&request, &mut emit)
    }))
    .unwrap_or(Err(ErrorCode::GenerationFailed));
    match result {
        Ok(()) => {
            let complete = Event::Complete {
                id: id.clone(),
                frames: p::FRAMES,
                audio_samples: p::AUDIO_SAMPLES,
            };
            if stream.accept(&complete, &[]).is_ok() {
                return write(output, &complete, &[]);
            }
            let code = ErrorCode::GenerationFailed;
            let _ = write(output, &Event::Error { id: Some(id), code }, &[]);
            Err(code)
        }
        Err(code) => {
            let _ = write(output, &Event::Error { id: Some(id), code }, &[]);
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
