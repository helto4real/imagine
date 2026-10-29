//! H3 uses the same supervised child slot and scheduler-owned Ticket as images.
use super::*;
use crate::{h3_protocol as wire, short_av::Av};
impl PipeBackend {
    pub(super) async fn run_video(&mut self, request: &wire::Request) -> Result<Av> {
        let preview = self.state.borrow().sampling_preview.clone();
        let public_job = self.state.borrow().job_id.clone();
        let mut effective = request.clone();
        effective.preview = Some(preview.settings());
        let request = &effective;
        if !request.valid()
            || !request.references.valid_payload()
            || !request
                .image
                .as_ref()
                .map_or(request.image_rgb.is_empty(), |v| {
                    v.bytes() == request.image_rgb.len()
                        && crate::storage::hash(&request.image_rgb) == v.sha256
                })
            || !request
                .continuation
                .as_ref()
                .map_or(request.continuation_bytes.is_empty(), |v| {
                    v.descriptor.valid_bytes(&request.continuation_bytes)
                        && crate::storage::hash(&request.continuation_bytes) == v.sha256
                })
        {
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
        self.launch(command, "h3")
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
            s.requested_steps = request.total_steps();
            s.progress = None;
        });
        let header = serde_json::to_vec(request)?;
        let input = self.input.as_mut().unwrap();
        input.write_u32(header.len() as u32).await?;
        input.write_all(&header).await?;
        input
            .write_u32(
                (request.image_rgb.len()
                    + request.continuation_bytes.len()
                    + request.references.bytes()) as u32,
            )
            .await?;
        input.write_all(&request.image_rgb).await?;
        input.write_all(&request.continuation_bytes).await?;
        for image in &request.references.images {
            input.write_all(&image.rgb).await?;
        }
        for video in &request.references.videos {
            input.write_all(&video.rgb).await?;
            if let Some(audio) = &video.audio {
                input.write_all(&audio.pcm).await?;
            }
        }
        for audio in &request.references.audios {
            input.write_all(&audio.pcm).await?;
        }
        input.flush().await?;
        let mut stream =
            wire::Stream::with_steps(request.id.clone(), request.shape, request.total_steps())
                .ok_or(EngineError("invalid_h3_request"))?;
        let mut result = Av {
            continuation: None,
            shape: request.shape,
            frames: Vec::with_capacity(request.shape.frames as usize),
            pcm: Vec::new(),
        };
        let mut reference_cache =
            crate::video::reference_cache_owner::Exchange::new(self.reference_cache.clone());
        loop {
            let (event, payload) = read(self.output.as_mut().unwrap()).await?;
            stream.accept(&event, &payload).map_err(EngineError)?;
            match event {
                wire::Event::BoundaryApplied { receipt, .. } => {
                    anyhow::ensure!(
                        request.references.boundary_receipt().as_ref() == Some(&receipt),
                        EngineError("h3_protocol_error")
                    );
                }
                wire::Event::SamplingPreview { metadata, .. } => {
                    if let Some(id) = &public_job {
                        preview.publish(id, metadata, payload);
                    }
                }
                wire::Event::PreviewWarning { code, .. } => {
                    if let Some(id) = &public_job {
                        preview.warn(id, &code);
                    }
                }
                wire::Event::ReferenceCache { command, .. } => {
                    let (reply, bytes) = reference_cache
                        .handle(command, &payload)
                        .map_err(EngineError)?;
                    let header = serde_json::to_vec(&reply)?;
                    let input = self
                        .input
                        .as_mut()
                        .ok_or(EngineError("h3_protocol_error"))?;
                    input.write_u32(header.len() as u32).await?;
                    input.write_all(&header).await?;
                    input.write_u32(bytes.len() as u32).await?;
                    input.write_all(&bytes).await?;
                    input.flush().await?;
                }
                wire::Event::Phase { phase, .. } => self.phase(match phase {
                    wire::Phase::LoadingText => "loading_text",
                    wire::Phase::EncodingText => "encoding_text",
                    wire::Phase::LoadingDit => "loading_dit",
                    wire::Phase::LoadingUpscaler => "loading_upscaler",
                    wire::Phase::UpscalingLatents => "upscaling_latents",
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
                wire::Event::Continuation { descriptor, .. } => {
                    anyhow::ensure!(
                        request.capture_continuation,
                        EngineError("h3_protocol_error")
                    );
                    result.continuation = Some(wire::continuation::Artifact {
                        descriptor,
                        bytes: payload,
                    });
                }
                wire::Event::Complete { .. } => {
                    anyhow::ensure!(
                        request.references.boundary_receipt().is_none()
                            || stream.boundary_applied(),
                        EngineError("h3_boundary_not_applied")
                    );
                    anyhow::ensure!(
                        !request.capture_continuation || result.continuation.is_some(),
                        EngineError("h3_protocol_error")
                    );
                    // Release all trained state and confirm exit while Ticket is still held.
                    self.stop().await?;
                    return Ok(result);
                }
                wire::Event::Error { code, failure, .. } => {
                    let code = EngineError(match code {
                        wire::ErrorCode::InvalidRequest => "invalid_h3_request",
                        wire::ErrorCode::DeploymentMismatch => "h3_deployment_mismatch",
                        wire::ErrorCode::ModelIdentityMismatch => "h3_model_identity_mismatch",
                        wire::ErrorCode::ComponentLoadFailed => "h3_component_load_failed",
                        wire::ErrorCode::OutOfMemory => "h3_out_of_memory",
                        wire::ErrorCode::GenerationFailed => "h3_generation_failed",
                        wire::ErrorCode::LatentUpscaleFailed => "h3_latent_upscale_failed",
                        wire::ErrorCode::LoraLoadFailed => "h3_lora_load_failed",
                        wire::ErrorCode::AttentionUnavailable => "h3_attention_unavailable",
                        wire::ErrorCode::UpscaleContextUnavailable => {
                            "h3_upscale_context_unavailable"
                        }
                    });
                    match failure.filter(wire::Failure::valid) {
                        Some(failure) => {
                            return Err(
                                anyhow::Error::new(crate::engine::H3Failure(failure)).context(code)
                            );
                        }
                        None => bail!(code),
                    }
                }
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
    let event: wire::Event =
        serde_json::from_slice(&header).map_err(|_| EngineError("h3_protocol_error"))?;
    let limit = event.payload_limit();
    Ok((event, part(reader, limit).await?))
}
