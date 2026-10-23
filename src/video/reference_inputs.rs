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
    file: FrozenFile,
    endpoint: Option<FrozenFile>,
}
/// The selected descriptor and its freeze-time identity travel together through
/// the queue. Original files remain writable by their owner, so decoding must
/// reject changes since admission as well as changes during decoding.
pub struct FrozenFile {
    file: File,
    witness: SourceWitness,
}
impl FrozenFile {
    pub(super) fn owned(file: File) -> Result<Self, StorageError> {
        let witness = source_witness(&file)?;
        Ok(Self { file, witness })
    }
    pub(super) fn resource(resource: &Resource, mut file: File) -> Result<Self, StorageError> {
        let frozen = Self::owned(
            file.try_clone()
                .map_err(|_| StorageError::new("storage_io_failed"))?,
        )?;
        let inspected = Resource::inspect(resource.id.clone(), resource.kind.clone(), &mut file)?;
        if inspected.sha256 != resource.sha256
            || inspected.bytes != resource.bytes
            || frozen.witness != source_witness(&file)?
        {
            return Err(StorageError::new(if resource.source.is_some() {
                "source_changed"
            } else {
                "resource_integrity_failed"
            }));
        }
        Ok(frozen)
    }
    pub(super) fn file(&self) -> &File {
        &self.file
    }
    pub(super) fn decode<T>(
        &self,
        decode: impl FnOnce(&File) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        anyhow::ensure!(
            self.witness == source_witness(&self.file)?,
            "source_changed"
        );
        let result = decode(&self.file);
        // Check even after a decoder error, so a mutated original is classified
        // as such instead of an unrelated codec/format failure.
        anyhow::ensure!(
            self.witness == source_witness(&self.file)?,
            "source_changed"
        );
        result
    }
}
fn frozen_file(
    session: &ProjectSession,
    entry: &Value,
    kind: &MediaKind,
    staging: &SessionStaging,
    resources: &mut Vec<(Resource, File)>,
    versions: &mut Vec<AssetVersion>,
) -> Result<FrozenFile, StorageError> {
    let (resource, version) = source(session, entry, kind)?;
    if let Some((_, file)) = resources.iter().find(|(r, _)| r.id == resource.id) {
        return FrozenFile::resource(
            resource,
            file.try_clone()
                .map_err(|_| StorageError::new("storage_io_failed"))?,
        );
    }
    let mut original = session.resource_file(&resource.id)?;
    let file = if resource.source.is_some() {
        original
    } else {
        staging.copy(&mut original)?
    };
    let frozen = FrozenFile::resource(resource, file)?;
    let retained = frozen
        .file
        .try_clone()
        .map_err(|_| StorageError::new("storage_io_failed"))?;
    resources.push((resource.clone(), retained));
    if let Some(version) = version {
        versions.push(version);
    }
    Ok(frozen)
}
fn source<'a>(
    session: &'a ProjectSession,
    entry: &Value,
    kind: &MediaKind,
) -> Result<(&'a Resource, Option<AssetVersion>), StorageError> {
    if let Some(take) = entry["take_id"].as_str() {
        let take = session
            .document()
            .takes
            .iter()
            .find(|t| t.id.0 == take && entry["source_scene_id"] == t.scene_id.0)
            .ok_or_else(|| StorageError::new("reference_take_missing"))?;
        let resource = session
            .document()
            .resources
            .iter()
            .find(|r| matches!(&r.kind, ResourceKind::TakeVideo { take_id } if take_id == &take.id))
            .ok_or_else(|| StorageError::new("reference_resource_missing"))?;
        return Ok((resource, None));
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
        .filter(|a| &a.kind == kind)
        .ok_or_else(|| StorageError::new("reference_asset_missing"))?;
    let resource = session.document().resources.iter().find(|r| {
        matches!(&r.kind, ResourceKind::Reference { asset_id, revision } if asset_id == &id && *revision == asset.revision)
    }).ok_or_else(|| StorageError::new("reference_resource_missing"))?;
    Ok((
        resource,
        Some(AssetVersion {
            asset_id: id,
            revision: asset.revision,
        }),
    ))
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
            // The boundary's fallback end image is derived from the same frozen
            // video after exact temporal selection, not a fabricated project Asset.
            if key == "images"
                && entry["role"] == "boundary_first_frame"
                && entry["from_boundary_video"] == true
            {
                continue;
            }
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
            let file = frozen_file(session, &entry, &kind, staging, resources, versions)?;
            let endpoint = entry["endpoint_asset_id"]
                .as_str()
                .map(|asset| {
                    frozen_file(
                        session,
                        &serde_json::json!({"asset_id":asset}),
                        &MediaKind::Image,
                        staging,
                        resources,
                        versions,
                    )
                })
                .transpose()?;
            files.push(Media {
                entry,
                kind: kind.clone(),
                file,
                endpoint,
            });
        }
    }
    Ok(files)
}
// Ordinary images and explicit boundary endpoints share the same framing bound
// before decoding; the limit is transport/memory validity, not a model allowlist.
fn decode_image(file: &File, budget: usize) -> anyhow::Result<wire::Image> {
    let info = super::media::detect(file).map_err(anyhow::Error::msg)?;
    let expected = u64::from(info.width)
        .saturating_mul(u64::from(info.height))
        .saturating_mul(3);
    anyhow::ensure!(
        info.kind == MediaKind::Image
            && expected <= crate::h3_protocol::MAX_INPUT_IMAGE as u64
            && expected <= budget as u64,
        "reference_decode_memory_limit"
    );
    let mut bytes = vec![];
    super::bundle::stream(file, &mut bytes)?;
    let (width, height, rgb) =
        super::authoring::vision::decode_rgb(&bytes).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        rgb.len() <= budget && rgb.len() <= crate::h3_protocol::MAX_INPUT_IMAGE,
        "reference_decode_memory_limit"
    );
    Ok(wire::Image { width, height, rgb })
}
struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub struct Prepared {
    pub input: wire::Input,
    pub frame0: Option<wire::Image>,
}
pub async fn prepare(
    mut input: wire::Input,
    files: Vec<Media>,
    target_frames: u32,
) -> anyhow::Result<Prepared> {
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(flag.clone());
    tokio::task::spawn_blocking(move || {
        let mut frame0 = None;
        for media in files {
            media.file.decode(|file| {
                match media.kind {
                    MediaKind::Image => {
                        input.images.push(decode_image(
                            file,
                            wire::MAX_BYTES.saturating_sub(input.bytes()),
                        )?);
                    }
                    MediaKind::Audio => {
                        input.audios.push(super::reference_video::audio_with_budget(
                            file,
                            &media.entry,
                            &flag,
                            wire::MAX_BYTES.saturating_sub(input.bytes()) as u64,
                        )?);
                    }
                    MediaKind::Video => {
                        if media.entry["role"] == "boundary_continuation" {
                            let boundary = super::reference_video::boundary(
                                file,
                                &media.entry,
                                target_frames,
                                &flag,
                                wire::MAX_BYTES.saturating_sub(input.bytes()) as u64,
                            )?;
                            input.manifest["boundary_selection"] = serde_json::json!({
                                "effective_tail_frames":boundary.effective_tail_frames,
                                "source_in":boundary.source_in,"source_out":boundary.source_out,
                                "include_paired_audio":boundary.video.audio.is_some(),
                            });
                            let end = if let Some(endpoint) = &media.endpoint {
                                let budget = if input.manifest["frame0_guide"]["enabled"] == true {
                                    crate::h3_protocol::MAX_INPUT_IMAGE
                                } else {
                                    wire::MAX_BYTES
                                        .saturating_sub(input.bytes())
                                        .saturating_sub(boundary.video.bytes())
                                };
                                endpoint.decode(|file| decode_image(file, budget))?
                            } else {
                                boundary.last_frame
                            };
                            if input.manifest["frame0_guide"]["enabled"] == true {
                                frame0 = Some(end);
                            } else {
                                input.images.insert(0, end);
                            }
                            if media.entry["boundary_source_only"] != true {
                                input.videos.push(boundary.video);
                            }
                            return Ok(());
                        }
                        let mut video = super::reference_video::decode_with_budget(
                            file,
                            &media.entry,
                            &flag,
                            wire::MAX_BYTES.saturating_sub(input.bytes()) as u64,
                        )?;
                        if media.entry["include_paired_audio"] == true {
                            video.audio = Some(super::reference_video::audio_with_budget(
                                file,
                                &media.entry,
                                &flag,
                                wire::MAX_BYTES
                                    .saturating_sub(input.bytes())
                                    .saturating_sub(video.rgb.len())
                                    as u64,
                            )?);
                        }
                        input.videos.push(video);
                    }
                }
                Ok(())
            })?;
        }
        anyhow::ensure!(input.valid_payload(), "invalid_h3_reference_media");
        Ok(Prepared { input, frame0 })
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_reference_freeze_keeps_original_descriptor_without_staging_copy() {
        use std::os::unix::fs::MetadataExt;
        let root =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let path = root.path().join("own-reference.bin");
        std::fs::write(&path, b"OWN PROCEDURAL REFERENCE").unwrap();
        let mut file = File::open(&path).unwrap();
        let asset = AssetId("own-reference".into());
        let resource = Resource::external(
            "own-resource".into(),
            ResourceKind::Reference {
                asset_id: asset.clone(),
                revision: 1,
            },
            &path,
            &mut file,
        )
        .unwrap();
        let project = VideoProject::create(
            serde_json::from_str(include_str!(
                "../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut session = ProjectSession::new(project).unwrap();
        let mut document = session.document().clone();
        document.project.assets.insert(
            asset.clone(),
            Asset {
                id: asset,
                revision: 1,
                kind: MediaKind::Image,
                name: String::new(),
                library_source: None,
            },
        );
        document.resources.push(resource);
        let staging = SessionStaging::select(root.path()).unwrap();
        session
            .update(0, document, std::collections::BTreeMap::new(), &staging)
            .unwrap();
        let mut resources = vec![];
        let mut versions = vec![];
        let frozen = frozen_file(
            &session,
            &serde_json::json!({"asset_id":"own-reference"}),
            &MediaKind::Image,
            &staging,
            &mut resources,
            &mut versions,
        )
        .unwrap();
        let original = std::fs::metadata(&path).unwrap();
        let actual = frozen.file.metadata().unwrap();
        assert_eq!(
            (actual.dev(), actual.ino()),
            (original.dev(), original.ino())
        );
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].0.source.as_deref(), Some(path.as_path()));
        assert_eq!(versions.len(), 1);
    }
    #[test]
    fn image_preflight_accepts_exact_budget_and_rejects_insufficient_budget() {
        let root = tempfile::tempdir().unwrap();
        assert!(crate::task_fixtures::is_owned(root.path()));
        let path = root.path().join("own-endpoint.png");
        let mut file = File::create(&path).unwrap();
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3, 2, image::Rgb([2, 3, 4])))
            .write_to(&mut file, image::ImageFormat::Png)
            .unwrap();
        let file = File::open(&path).unwrap();
        assert_eq!(
            decode_image(&file, 17).err().unwrap().to_string(),
            "reference_decode_memory_limit"
        );
        let image = decode_image(&file, 18).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.rgb, [2, 3, 4].repeat(6));
    }
    fn own_linked_picture() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        ProjectSession,
        SessionStaging,
    ) {
        let root =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let path = root.path().join("own-picture.png");
        image::RgbImage::from_pixel(3, 2, image::Rgb([2, 3, 4]))
            .save(&path)
            .unwrap();
        let asset = AssetId("own-picture".into());
        let resource = Resource::external(
            "own-picture-resource".into(),
            ResourceKind::Reference {
                asset_id: asset.clone(),
                revision: 1,
            },
            &path,
            &mut File::open(&path).unwrap(),
        )
        .unwrap();
        let project = VideoProject::create(
            serde_json::from_str(include_str!(
                "../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut session = ProjectSession::new(project).unwrap();
        let mut document = session.document().clone();
        document.project.assets.insert(
            asset.clone(),
            Asset {
                id: asset,
                revision: 1,
                kind: MediaKind::Image,
                name: String::new(),
                library_source: None,
            },
        );
        document.resources.push(resource);
        let staging = SessionStaging::select(root.path()).unwrap();
        session
            .update(0, document, std::collections::BTreeMap::new(), &staging)
            .unwrap();
        (root, path, session, staging)
    }
    #[tokio::test]
    async fn linked_reference_prepare_preserves_rgb_and_rejects_change_since_freeze() {
        use std::os::unix::fs::FileExt;
        let (_root, path, session, staging) = own_linked_picture();
        let manifest = serde_json::json!({"images":[{"asset_id":"own-picture"}]});
        let input = || wire::Input {
            manifest: manifest.clone(),
            ..Default::default()
        };
        let freeze =
            || super::freeze(&session, &manifest, &staging, &mut vec![], &mut vec![]).unwrap();
        let prepared = prepare(input(), freeze(), 5).await.unwrap();
        assert_eq!(prepared.input.images[0].rgb, [2, 3, 4].repeat(6));
        let frozen = freeze();
        let original_size = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(b"x", 0)
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), original_size);
        assert_eq!(
            prepare(input(), frozen, 5).await.err().unwrap().to_string(),
            "source_changed"
        );
    }
    #[test]
    fn frozen_source_rejects_mutation_during_decode_even_when_decoder_fails() {
        use std::os::unix::fs::FileExt;
        let (_root, path, session, staging) = own_linked_picture();
        let frozen = frozen_file(
            &session,
            &serde_json::json!({"asset_id":"own-picture"}),
            &MediaKind::Image,
            &staging,
            &mut vec![],
            &mut vec![],
        )
        .unwrap();
        let result: anyhow::Result<()> = frozen.decode(|_| {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)?
                .write_all_at(b"x", 0)?;
            anyhow::bail!("codec_error_after_mutation")
        });
        assert_eq!(result.err().unwrap().to_string(), "source_changed");
    }
}
