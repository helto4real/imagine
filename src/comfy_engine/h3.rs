//! H3 uses the same supervised child slot and scheduler-owned Ticket as images.
use super::*;
use crate::{h3_protocol as wire, short_av::Av};
impl PipeBackend {
    pub(super) async fn run_video(&mut self, request: &wire::Request) -> Result<Av> {
        if !request.valid() {
            bail!(EngineError("invalid_h3_request"));
        }
        let runtime = self
            .rust_runtime
            .as_ref()
            .and_then(|r| r.h3.as_ref())
            .ok_or(EngineError("h3_runtime_unavailable"))?
            .clone();
        runtime
            .verify()
            .map_err(|_| EngineError("h3_deployment_mismatch"))?;
        // Never retain an image worker or create a second simultaneous child.
        self.stop().await?;
        let mut command = Command::new(&runtime.binary);
        let cache = self.config.data.join("h3-cache");
        command
            .envs(&runtime.environment)
            .env("IMAGINE_WORKER_CACHE", &cache)
            .env("CUDA_CACHE_PATH", cache.join("cuda"))
            .env("IMAGINE_H3_MODEL_ROOT", &runtime.model_root)
            .env("IMAGINE_H3_DEPENDENCY_MAP", &runtime.dependency_map)
            .env(
                "IMAGINE_H3_DEPLOYMENT",
                serde_json::to_string(&runtime.identity)?,
            );
        self.launch(command)
            .await
            .map_err(|_| EngineError("h3_startup_failed"))?;
        self.h3_active = true;
        self.phase("preparing_inputs");
        let (ready, body) =
            tokio::time::timeout(Duration::from_secs(30), read(self.output.as_mut().unwrap()))
                .await
                .map_err(|_| EngineError("h3_startup_timeout"))??;
        if !body.is_empty()
            || !matches!(ready,wire::Event::Ready {deployment} if deployment==runtime.identity)
        {
            bail!(EngineError("h3_deployment_mismatch"));
        }
        self.state.send_modify(|s| {
            s.requested_steps = wire::STEPS;
            s.progress = None;
        });
        let header = serde_json::to_vec(request)?;
        let input = self.input.as_mut().unwrap();
        input.write_u32(header.len() as u32).await?;
        input.write_all(&header).await?;
        input.write_u32(0).await?;
        input.flush().await?;
        let mut stream =
            wire::Stream::new(request.id.clone()).ok_or(EngineError("invalid_h3_request"))?;
        let mut result = Av {
            frames: Vec::with_capacity(wire::FRAMES as usize),
            pcm: Vec::new(),
        };
        loop {
            let (event, payload) = read(self.output.as_mut().unwrap()).await?;
            stream.accept(&event, &payload).map_err(EngineError)?;
            match event {
                wire::Event::Phase { phase, .. } => self.phase(match phase {
                    wire::Phase::LoadingText => "loading_text",
                    wire::Phase::EncodingText => "encoding_text",
                    wire::Phase::LoadingDit => "loading_dit",
                    wire::Phase::Sampling => "sampling",
                    wire::Phase::LoadingVideoVae => "loading_video_vae",
                    wire::Phase::DecodingVideo => "decoding_video",
                    wire::Phase::LoadingAudioVae => "loading_audio_vae",
                    wire::Phase::DecodingAudio => "decoding_audio",
                }),
                wire::Event::Progress {
                    completed,
                    total,
                    elapsed_seconds,
                    ..
                } => self.state.send_modify(|s| {
                    s.progress = Some(SamplingProgress {
                        completed,
                        total,
                        elapsed_seconds,
                        remaining_seconds: elapsed_seconds / f64::from(completed)
                            * f64::from(total - completed),
                        updated_at_ms: crate::storage::now() as u64,
                    });
                }),
                wire::Event::VideoFrame { .. } => result.frames.push(payload),
                wire::Event::AudioChunk { .. } => result.pcm.extend(payload),
                wire::Event::Complete { .. } => {
                    // Release all trained state and confirm exit while Ticket is still held.
                    self.stop().await?;
                    return Ok(result);
                }
                wire::Event::Error { code, .. } => bail!(EngineError(match code {
                    wire::ErrorCode::InvalidRequest => "invalid_h3_request",
                    wire::ErrorCode::DeploymentMismatch => "h3_deployment_mismatch",
                    wire::ErrorCode::ModelIdentityMismatch => "h3_model_identity_mismatch",
                    wire::ErrorCode::ComponentLoadFailed => "h3_component_load_failed",
                    wire::ErrorCode::OutOfMemory => "h3_out_of_memory",
                    wire::ErrorCode::GenerationFailed => "h3_generation_failed",
                })),
                wire::Event::Ready { .. } => unreachable!("stream rejects second Ready"),
            }
        }
    }
}
async fn read(reader: &mut (impl AsyncRead + Unpin)) -> Result<(wire::Event, Vec<u8>)> {
    async fn part(reader: &mut (impl AsyncRead + Unpin), limit: usize) -> Result<Vec<u8>> {
        let n = reader
            .read_u32()
            .await
            .map_err(|_| EngineError("h3_protocol_error"))? as usize;
        if n > limit {
            bail!(EngineError("h3_protocol_error"));
        }
        let mut b = vec![0; n];
        reader
            .read_exact(&mut b)
            .await
            .map_err(|_| EngineError("h3_protocol_error"))?;
        Ok(b)
    }
    let header = part(reader, wire::MAX_HEADER).await?;
    let event = serde_json::from_slice(&header).map_err(|_| EngineError("h3_protocol_error"))?;
    Ok((event, part(reader, wire::MAX_PAYLOAD).await?))
}
