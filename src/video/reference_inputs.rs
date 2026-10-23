//! Frozen project resources become ordered RGB/PCM only inside the queued worker path.
use super::{contract::*, persistence::*};
use crate::h3_protocol::references as wire;
use serde_json::Value;
use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
pub struct Media {
    entry: Value,
    kind: MediaKind,
    file: File,
}
pub fn freeze(
    session: &ProjectSession,
    manifest: &Value,
    staging: &SessionStaging,
    resources: &mut Vec<(Resource, File)>,
    versions: &mut Vec<AssetVersion>,
) -> Result<Vec<Media>, StorageError> {
    let mut files = vec![];
    for (key, kind) in [
        ("images", MediaKind::Image),
        ("videos", MediaKind::Video),
        ("audios", MediaKind::Audio),
    ] {
        for entry in manifest[key].as_array().into_iter().flatten() {
            if key == "audios" && !entry["paired_video_key"].is_null() {
                continue;
            }
            let mut entry = entry.clone();
            if key == "videos" {
                entry["include_paired_audio"] = serde_json::json!(
                    manifest["audios"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|a| a["paired_video_key"] == entry["native_key"])
                );
            }
            let id = AssetId(
                entry["asset_id"]
                    .as_str()
                    .ok_or_else(|| StorageError::new("reference_asset_missing"))?
                    .into(),
            );
            let asset = session
                .document()
                .project
                .assets
                .get(&id)
                .filter(|a| a.kind == kind)
                .ok_or_else(|| StorageError::new("reference_asset_missing"))?;
            let resource=session.document().resources.iter().find(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id==&id && *revision==asset.revision)).ok_or_else(||StorageError::new("reference_resource_missing"))?;
            let file = if let Some((_, file)) = resources.iter().find(|(r, _)| r.id == resource.id)
            {
                file.try_clone()
                    .map_err(|_| StorageError::new("storage_io_failed"))?
            } else {
                let mut source = session.resource_file(&resource.id)?;
                let mut copy = staging.copy(&mut source)?;
                if Resource::inspect(resource.id.clone(), resource.kind.clone(), &mut copy)?
                    != *resource
                {
                    return Err(StorageError::new("resource_integrity_failed"));
                }
                let file = copy
                    .try_clone()
                    .map_err(|_| StorageError::new("storage_io_failed"))?;
                resources.push((resource.clone(), copy));
                versions.push(AssetVersion {
                    asset_id: id,
                    revision: asset.revision,
                });
                file
            };
            files.push(Media {
                entry,
                kind: kind.clone(),
                file,
            });
        }
    }
    Ok(files)
}
struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub async fn prepare(mut input: wire::Input, files: Vec<Media>) -> anyhow::Result<wire::Input> {
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(flag.clone());
    tokio::task::spawn_blocking(move || {
        for media in files {
            match media.kind {
                MediaKind::Image => {
                    let mut bytes = vec![];
                    super::bundle::stream(&media.file, &mut bytes)?;
                    let (width, height, rgb) =
                        super::authoring::vision::decode_rgb(&bytes).map_err(anyhow::Error::msg)?;
                    input.images.push(wire::Image { width, height, rgb });
                }
                MediaKind::Audio => {
                    input.audios.push(super::reference_video::audio(
                        &media.file,
                        &media.entry,
                        &flag,
                    )?);
                }
                MediaKind::Video => {
                    let mut video =
                        super::reference_video::decode(&media.file, &media.entry, &flag)?;
                    if media.entry["include_paired_audio"] == true {
                        video.audio = Some(super::reference_video::audio(
                            &media.file,
                            &media.entry,
                            &flag,
                        )?);
                    }
                    input.videos.push(video);
                }
            }
        }
        anyhow::ensure!(input.valid_payload(), "invalid_h3_reference_media");
        Ok(input)
    })
    .await?
}
