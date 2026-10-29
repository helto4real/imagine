//! Required Take resources are prepared with the video; no container re-decode or silent fallback.
use super::{contract::*, persistence::*};
use crate::{h3_protocol as h3, short_av::Av};
use std::{fs::File, io::Cursor};
pub enum Produced {
    BoundaryApplied(serde_json::Value),
    /// The operator's execution choices the Take was made with.
    Execution(crate::execution_settings::Settings),
    H3Endpoint {
        shape: h3::Shape,
        png: Vec<u8>,
    },
    H3Motion(h3::continuation::Artifact),
}
impl Produced {
    pub fn endpoint(av: &Av) -> anyhow::Result<Self> {
        anyhow::ensure!(av.valid(), "invalid_endpoint_output");
        let image = image::RgbImage::from_raw(
            av.shape.width,
            av.shape.height,
            av.frames
                .last()
                .ok_or_else(|| anyhow::anyhow!("endpoint_missing"))?
                .clone(),
        )
        .ok_or_else(|| anyhow::anyhow!("invalid_endpoint_output"))?;
        let mut bytes = Cursor::new(vec![]);
        image.write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(Self::H3Endpoint {
            shape: av.shape,
            png: bytes.into_inner(),
        })
    }
    pub fn prepare(
        &self,
        take: &StoredTake,
        staging: &SessionStaging,
    ) -> Result<(Resource, File), StorageError> {
        let provenance = take.generated()?;
        match self {
            Self::BoundaryApplied(_) => Err(StorageError::new("boundary_receipt_is_not_media")),
            Self::Execution(_) => Err(StorageError::new("execution_record_is_not_media")),
            Self::H3Motion(artifact) => {
                if !artifact.valid() {
                    return Err(StorageError::new("invalid_continuation"));
                }
                let mut file = super::execution::media_file(staging, &artifact.bytes)?;
                let resource = Resource::inspect(
                    format!("motion_{}", uuid::Uuid::new_v4().simple()),
                    ResourceKind::H3Motion {
                        take_id: take.id.clone(),
                        model: provenance.model.clone(),
                        descriptor: artifact.descriptor.clone(),
                    },
                    &mut file,
                )?;
                Ok((resource, file))
            }
            Self::H3Endpoint { shape, png } => {
                validate_png(png, *shape)?;
                let mut file = super::execution::media_file(staging, png)?;
                let resource = Resource::inspect(
                    format!("endpoint_{}", uuid::Uuid::new_v4().simple()),
                    ResourceKind::H3Endpoint {
                        take_id: take.id.clone(),
                        schema_version: 1,
                        model: provenance.model.clone(),
                        shape: *shape,
                    },
                    &mut file,
                )?;
                Ok((resource, file))
            }
        }
    }
}
fn validate_png(bytes: &[u8], shape: h3::Shape) -> Result<(), StorageError> {
    if !shape.valid() || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(StorageError::new("invalid_endpoint"));
    }
    let (image, _) =
        super::image::decode(bytes).map_err(|_| StorageError::new("invalid_endpoint"))?;
    if image.width != shape.width || image.height != shape.height {
        return Err(StorageError::new("endpoint_geometry_mismatch"));
    }
    Ok(())
}
pub struct FrozenEndpoint {
    pub source: EndpointSource,
    pub resource: Resource,
    pub file: File,
    pub motion: Option<h3::continuation::Input>,
}
pub fn freeze_endpoint(
    session: &ProjectSession,
    target: &Scene,
    staging: &SessionStaging,
) -> Result<Option<FrozenEndpoint>, StorageError> {
    let Some(continuity) = &target.continuity else {
        return Ok(None);
    };
    if matches!(
        continuity.method,
        ContinuityMethod::Combo | ContinuityMethod::Influence
    ) {
        // These consume the frozen source video and optional explicit end image,
        // through reference_inputs rather than a precomputed untrimmed endpoint.
        return Ok(None);
    }
    if !matches!(
        continuity.method,
        ContinuityMethod::LastFrameFirstFrame
            | ContinuityMethod::MotionContext
            | ContinuityMethod::MotionContextExp
    ) {
        return Err(StorageError::new("continuity_method_unbound"));
    }
    if continuity.method == ContinuityMethod::LastFrameFirstFrame && continuity.include_audio {
        return Err(StorageError::new("continuity_endpoint_incompatible"));
    }
    let scene = session.document().project.scene(&continuity.source_scene)?;
    let selected = scene
        .accepted_take
        .as_ref()
        .ok_or_else(|| StorageError::new("continuity_source_not_accepted"))?;
    let take = session
        .document()
        .takes
        .iter()
        .find(|take| {
            take.id == selected.id
                && take.scene_id == scene.id
                && take.status == TakeStatus::Accepted
        })
        .ok_or_else(|| StorageError::new("continuity_source_not_accepted"))?;
    let resources = session
        .document()
        .resources
        .iter()
        .filter(|resource| match &resource.kind {
            ResourceKind::H3Endpoint { take_id, .. } => {
                continuity.method == ContinuityMethod::LastFrameFirstFrame && take_id == &take.id
            }
            ResourceKind::H3Motion { take_id, .. } => {
                continuity.method != ContinuityMethod::LastFrameFirstFrame && take_id == &take.id
            }
            _ => false,
        })
        .collect::<Vec<_>>();
    if resources.len() != 1 {
        return Err(StorageError::new(
            "continuity_endpoint_missing_or_ambiguous",
        ));
    }
    let resource = resources[0];
    if let ResourceKind::H3Motion {
        model, descriptor, ..
    } = &resource.kind
    {
        if model != &take.generated()?.model {
            return Err(StorageError::new("continuity_endpoint_incompatible"));
        }
        let mut file = session.resource_file(&resource.id)?;
        let file = staging.copy(&mut file)?;
        let mut bytes = vec![];
        super::bundle::stream(&file, &mut bytes)?;
        if !descriptor.valid_bytes(&bytes) {
            return Err(StorageError::new("invalid_continuation"));
        }
        let method = if continuity.method == ContinuityMethod::MotionContext {
            h3::continuation::Method::MotionContext
        } else {
            h3::continuation::Method::MotionContextExp
        };
        return Ok(Some(FrozenEndpoint {
            source: EndpointSource {
                scene_id: scene.id.clone(),
                take_id: take.id.clone(),
                resource_id: resource.id.clone(),
                sha256: resource.sha256.clone(),
                schema_version: descriptor.schema_version,
                shape: descriptor.shape,
                method: continuity.method.clone(),
                include_audio: continuity.include_audio,
            },
            resource: resource.clone(),
            file,
            motion: Some(h3::continuation::Input {
                method,
                include_audio: continuity.include_audio,
                descriptor: descriptor.clone(),
                sha256: resource.sha256.clone(),
            }),
        }));
    }
    let ResourceKind::H3Endpoint {
        schema_version,
        model,
        shape,
        ..
    } = &resource.kind
    else {
        unreachable!()
    };
    if *schema_version != 1 || model != &take.generated()?.model {
        return Err(StorageError::new("continuity_endpoint_incompatible"));
    }
    let mut file = session.resource_file(&resource.id)?;
    let copy = staging.copy(&mut file)?;
    if copy
        .metadata()
        .map_err(|_| StorageError::new("storage_io_failed"))?
        .len()
        > h3::MAX_INPUT_IMAGE as u64
    {
        return Err(StorageError::new("invalid_endpoint"));
    }
    let mut bytes = vec![];
    super::bundle::stream(&copy, &mut bytes)?;
    validate_png(&bytes, *shape)?;
    Ok(Some(FrozenEndpoint {
        source: EndpointSource {
            scene_id: scene.id.clone(),
            take_id: take.id.clone(),
            resource_id: resource.id.clone(),
            sha256: resource.sha256.clone(),
            schema_version: *schema_version,
            shape: *shape,
            method: continuity.method.clone(),
            include_audio: continuity.include_audio,
        },
        resource: resource.clone(),
        file: copy,
        motion: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_is_exactly_the_last_rgb_frame_before_lossy_container_encoding() {
        let shape = h3::Shape {
            width: 32,
            height: 32,
            frames: 22,
            fps: 24,
        };
        let mut av = Av {
            continuation: None,
            shape,
            frames: vec![vec![70; shape.frame_bytes()]; 22],
            pcm: vec![0; shape.audio_samples() as usize * 8],
        };
        av.frames[21].fill(137);
        let Produced::H3Endpoint {
            shape: recorded,
            png,
        } = Produced::endpoint(&av).unwrap()
        else {
            panic!("expected endpoint")
        };
        assert_eq!(shape, recorded);
        validate_png(&png, shape).unwrap();
        let (_, pixels) = super::super::image::decode(&png).unwrap();
        assert_eq!(pixels, vec![137; shape.frame_bytes()]);
        assert!(validate_png(&png, h3::Shape { width: 64, ..shape }).is_err());
        assert!(validate_png(b"not a PNG", shape).is_err());
        av.pcm[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Produced::endpoint(&av).is_err());
    }
}
