//! Explicit health checks and revision-guarded relink of owned media resources.
use super::{bundle::Bundle, contract::*, media, persistence::*};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Impact {
    pub kind: String,
    pub owner: String,
    pub binding: String,
}
/// Structured identities only. Labels and media paths are never needed to describe impact.
pub fn impacts(document: &ProjectDocument, id: &AssetId) -> Vec<Impact> {
    let mut out = vec![];
    let mut push = |kind: &str, owner: &str, binding: &str| {
        out.push(Impact {
            kind: kind.into(),
            owner: owner.into(),
            binding: binding.into(),
        })
    };
    for scene in &document.project.scenes {
        for binding in &scene.bindings {
            if &binding.asset == id {
                push("scene", &scene.id.0, &binding.id.0);
            }
        }
    }
    for (section, source) in &document.project.creative.sections {
        for image in &source.images {
            if &image.asset == id {
                push("section_image", &section.0, &image.image_id);
            }
        }
        for video in &source.videos {
            if &video.asset == id {
                push("section_video", &section.0, &video.id.0);
            }
        }
    }
    for reference in &document.project.creative.references {
        for image in &reference.images {
            if &image.asset == id {
                push("reference_image", &reference.id, &image.id);
            }
        }
        for (index, asset) in reference.voice.iter().flat_map(|v| &v.assets).enumerate() {
            if asset == id {
                push("reference_voice", &reference.id, &index.to_string());
            }
        }
    }
    for track in &document.project.audio_tracks {
        for clip in &track.clips {
            if clip.asset.as_ref() == Some(id) {
                push("audio", &track.id.0, &clip.id.0);
            }
        }
    }
    for (scene, clip) in &document.project.assembly.clips {
        if &clip.asset == id {
            push("scene_clip", &scene.0, &scene.0);
        }
    }
    for take in &document.takes {
        for version in take.provenance.iter().flat_map(|p| &p.references) {
            if &version.asset_id == id {
                push("retained_take", &take.id.0, &version.revision.to_string());
            }
        }
    }
    for (take, capture) in &document.captures {
        if let Some(source) = &capture.imported_from
            && &source.asset_id == id
        {
            push("retained_take", &take.0, &source.revision.to_string());
        }
    }
    if out.is_empty() {
        out.push(Impact {
            kind: "asset".into(),
            owner: id.0.clone(),
            binding: String::new(),
        });
    }
    out
}
#[derive(Serialize)]
pub struct Entry {
    pub asset_id: AssetId,
    pub asset_revision: u64,
    pub status: String,
    pub code: String,
    pub impacts: Vec<Impact>,
}
pub struct Check {
    pub revision: u64,
    entries: Vec<(Entry, MediaKind, Result<File, StorageError>)>,
}
impl Check {
    pub fn capture(session: &ProjectSession, expected: u64) -> Result<Self, StorageError> {
        let document = session.document();
        if document.project.project_revision != expected {
            return Err(StorageError::new("revision_conflict"));
        }
        let entries = document.project.assets.values().map(|asset| {
            let file = document.resources.iter().find(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id==&asset.id && *revision==asset.revision))
                .ok_or_else(||StorageError::new("reference_resource_missing")).and_then(|r| session.resource_file(&r.id));
            (Entry {asset_id:asset.id.clone(),asset_revision:asset.revision,status:String::new(),code:String::new(),impacts:impacts(document,&asset.id)}, asset.kind.clone(),file)
        }).collect();
        Ok(Self {
            revision: expected,
            entries,
        })
    }
    pub fn run(self, cancel: &AtomicBool) -> Result<Report, StorageError> {
        let mut entries = vec![];
        for (mut entry, kind, file) in self.entries {
            cancelled(cancel)?;
            let result = file.and_then(|file| validate(&file, &kind, cancel).map(|_| ()));
            cancelled(cancel)?;
            match result {
                Ok(()) => {
                    entry.status = "ok".into();
                }
                Err(error) => {
                    entry.status = if matches!(
                        error.code.as_str(),
                        "resource_missing"
                            | "reference_resource_missing"
                            | "storage_blob_missing"
                            | "source_missing"
                    ) {
                        "missing"
                    } else {
                        "decode_error"
                    }
                    .into();
                    entry.code = error.code;
                }
            }
            entries.push(entry);
        }
        Ok(Report {
            schema_version: 1,
            project_revision: self.revision,
            entries,
        })
    }
}
#[derive(Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub project_revision: u64,
    pub entries: Vec<Entry>,
}
pub fn cancelled(cancel: &AtomicBool) -> Result<(), StorageError> {
    if cancel.load(Ordering::Acquire) {
        Err(StorageError::new("media_process_cancelled"))
    } else {
        Ok(())
    }
}
/// Inspect format and decode a bounded sample. Never decode the whole video/audio for health.
pub fn validate(
    file: &File,
    kind: &MediaKind,
    cancel: &AtomicBool,
) -> Result<media::Info, StorageError> {
    cancelled(cancel)?;
    let info = media::detect_streamed(file).map_err(StorageError::new)?;
    if &info.kind != kind {
        return Err(StorageError::new("media_type_mismatch"));
    }
    let args: &[&str] = if *kind == MediaKind::Audio {
        &[
            "-map", "0:a:0", "-vn", "-t", "0.05", "-ar", "44100", "-ac", "2", "-f", "f32le",
        ]
    } else {
        &[
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            "1",
            "-vf",
            "scale=64:64",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
        ]
    };
    // 50 ms at 44.1 kHz, stereo float32. This exceeds the one-frame RGB budget.
    let sample_bytes = if *kind == MediaKind::Audio {
        2205 * 2 * 4
    } else {
        64 * 64 * 3
    };
    let bytes = super::assembly::media::decode(file, args, sample_bytes, cancel)
        .map_err(StorageError::new)?;
    if bytes.is_empty() {
        return Err(StorageError::new("media_decode_failed"));
    }
    cancelled(cancel)?;
    Ok(info)
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Relink {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub asset_id: AssetId,
    pub expected_asset_revision: u64,
    pub replacement_asset_id: String,
}
pub fn prepare_relink(
    session: &ProjectSession,
    request: &Relink,
    file: &mut File,
    kind: &MediaKind,
) -> Result<(ProjectDocument, Resource, Vec<Impact>), StorageError> {
    prepare_relink_source(session, request, file, kind, None)
}
pub fn prepare_relink_source(
    session: &ProjectSession,
    request: &Relink,
    file: &mut File,
    kind: &MediaKind,
    source: Option<&std::path::Path>,
) -> Result<(ProjectDocument, Resource, Vec<Impact>), StorageError> {
    if request.schema_version != 1 {
        return Err(StorageError::new("unsupported_schema"));
    }
    let old = session.document();
    if old.project.project_revision != request.expected_project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    let asset = old
        .project
        .assets
        .get(&request.asset_id)
        .ok_or_else(|| StorageError::new("asset_not_found"))?;
    if asset.revision != request.expected_asset_revision {
        return Err(StorageError::new("asset_revision_conflict"));
    }
    if &asset.kind != kind {
        return Err(StorageError::new("media_type_mismatch"));
    }
    let revision = asset
        .revision
        .checked_add(1)
        .filter(|v| *v <= MAX_INTEGER)
        .ok_or_else(|| StorageError::new("revision_exhausted"))?;
    let resource_id = format!("resource_{}", uuid::Uuid::new_v4().simple());
    let resource_kind = ResourceKind::Reference {
        asset_id: asset.id.clone(),
        revision,
    };
    let resource = if let Some(path) = source {
        Resource::external(resource_id, resource_kind, path, file)?
    } else {
        Resource::inspect(resource_id, resource_kind, file)?
    };
    let mut document = old.clone();
    let asset = document.project.assets.get_mut(&request.asset_id).unwrap();
    asset.revision = revision;
    asset.library_source = None;
    // Historical resources/provenance keep the previous revision. Every semantic binding stays.
    document.resources.push(resource.clone());
    document.validate()?;
    Ok((document, resource, impacts(old, &request.asset_id)))
}
pub fn apply_relink(
    session: &mut ProjectSession,
    request: &Relink,
    file: File,
    kind: &MediaKind,
    staging: &SessionStaging,
) -> Result<Vec<Impact>, StorageError> {
    apply_relink_source(session, request, file, kind, staging, None)
}
pub fn apply_relink_source(
    session: &mut ProjectSession,
    request: &Relink,
    mut file: File,
    kind: &MediaKind,
    staging: &SessionStaging,
    source: Option<std::path::PathBuf>,
) -> Result<Vec<Impact>, StorageError> {
    let (document, resource, impacts) =
        prepare_relink_source(session, request, &mut file, kind, source.as_deref())?;
    let prepared = session.prepare_update(
        request.expected_project_revision,
        document,
        BTreeMap::from([(resource.key(), file)]),
        staging,
    )?;
    session.commit_atomic(prepared)?;
    Ok(impacts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn media_health_missing_is_an_explicit_nonmutating_result() {
        let check = Check {
            revision: 3,
            entries: vec![(
                Entry {
                    asset_id: AssetId("synthetic_missing".into()),
                    asset_revision: 1,
                    status: String::new(),
                    code: String::new(),
                    impacts: vec![],
                },
                MediaKind::Video,
                Err(StorageError::new("resource_missing")),
            )],
        };
        let report = check.run(&AtomicBool::new(false)).unwrap();
        assert_eq!(report.project_revision, 3);
        assert_eq!(report.entries[0].status, "missing");
    }
}
