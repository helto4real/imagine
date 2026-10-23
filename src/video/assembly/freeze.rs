use super::*;
use crate::video::{contract::*, persistence::*, service::Service};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
};
pub struct Frozen {
    pub plan: Plan,
    pub files: BTreeMap<String, File>,
    /// Runtime-only identity of linked originals, whose bytes may change after freeze.
    pub external_sources: BTreeSet<String>,
}
impl Frozen {
    pub fn try_clone(&self) -> Result<Self, StorageError> {
        Ok(Self {
            plan: self.plan.clone(),
            external_sources: self.external_sources.clone(),
            files: self
                .files
                .iter()
                .map(|(id, file)| {
                    Ok((
                        id.clone(),
                        file.try_clone()
                            .map_err(|_| StorageError::new("storage_io_failed"))?,
                    ))
                })
                .collect::<Result<_, StorageError>>()?,
        })
    }
}
fn fail(code: &str) -> StorageError {
    StorageError::new(code)
}
fn source(
    session: &ProjectSession,
    resource: &Resource,
    staging: &SessionStaging,
    files: &mut BTreeMap<String, File>,
) -> Result<Source, StorageError> {
    let mut source = session.resource_file(&resource.id)?;
    let copy = if resource.source.is_some() {
        source
    } else {
        staging.copy(&mut source)?
    };
    let witness = source_witness(&copy)?;
    let info = crate::video::media::detect_streamed(&copy).map_err(fail)?;
    if witness != source_witness(&copy)? {
        return Err(fail("source_changed"));
    }
    files.insert(resource.id.clone(), copy);
    let duration = info
        .frames
        .zip(info.fps)
        .and_then(|(n, fps)| {
            fps.den
                .checked_mul(n)
                .map(|num| Time { num, den: fps.num })
                .filter(|t| t.valid())
                .map(Time::normalized)
        })
        .or(info.duration)
        .ok_or_else(|| fail("media_duration_missing"))?;
    Ok(Source {
        resource_id: resource.id.clone(),
        sha256: resource.sha256.clone(),
        bytes: resource.bytes,
        mime_type: info.mime_type,
        width: info.width,
        height: info.height,
        duration,
        fps: info.fps,
        frames: info.frames,
        audio: info.audio,
    })
}
fn reference<'a>(session: &'a ProjectSession, id: &AssetId) -> Result<&'a Resource, StorageError> {
    let asset = session
        .document()
        .project
        .assets
        .get(id)
        .ok_or_else(|| fail("asset_not_found"))?;
    session.document().resources.iter().find(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id==id&&*revision==asset.revision)).ok_or_else(||fail("reference_resource_missing"))
}

pub fn freeze(
    service: &Service,
    request: &Request,
    staging: &SessionStaging,
) -> Result<Frozen, StorageError> {
    if request.schema_version != 1 {
        return Err(fail("unsupported_schema"));
    }
    let session = service.projects.session(&request.project_id)?;
    let project = &session.document().project;
    if project.project_revision != request.expected_project_revision {
        return Err(fail("revision_conflict"));
    }
    let delivery = project.delivery.clone();
    let fps = seconds(delivery.fps);
    let mut normalized = request.options.clone();
    normalized.output_filename = output_filename(normalized.output_filename.as_deref())?;
    let options = &normalized;
    let frame_bytes = (delivery.width as u64)
        .checked_mul(delivery.height as u64)
        .and_then(|n| n.checked_mul(12));
    if frame_bytes.is_none_or(|n| n > crate::video::media::MAX_UPLOAD) {
        return Err(fail("assembly_decode_memory_limit"));
    }
    if !delivery.fps.positive()
        || delivery.width == 0
        || delivery.height == 0
        || !delivery.width.is_multiple_of(2)
        || !delivery.height.is_multiple_of(2)
        || ![8, 10].contains(&options.bit_depth)
    {
        return Err(fail("invalid_delivery_geometry"));
    }
    if ![
        options.generated_gain,
        options.timeline_gain,
        options.ducking_strength,
    ]
    .iter()
    .all(|v| v.is_finite())
        || options.generated_gain < 0.
        || options.timeline_gain < 0.
        || !(0. ..=1.).contains(&options.ducking_strength)
    {
        return Err(fail("invalid_mix_options"));
    }
    let mut plan = Plan {
        schema_version: 1,
        id: crate::storage::id("assembly"),
        project_id: project.id.clone(),
        project_revision: project.project_revision,
        fingerprint: String::new(),
        delivery,
        sample_rate: 44100,
        frames: 0,
        samples: 0,
        options: options.clone(),
        clips: vec![],
        audio: vec![],
        warnings: vec![],
    };
    let mut files = BTreeMap::new();
    for scene in &project.scenes {
        let (resource, take_id, asset_id, source_in, source_out, include_audio) = if project
            .scene_kind(&scene.id)
            == imagine_video_contract::creative::SceneKind::Imported
        {
            let clip = project
                .scene_clip(&scene.id)
                .ok_or_else(|| fail("assembly_imported_clip_missing"))?;
            (
                reference(session, &clip.asset)?,
                None,
                Some(clip.asset.clone()),
                clip.source_in,
                clip.source_out,
                clip.include_audio,
            )
        } else {
            let take = scene
                .accepted_take
                .as_ref()
                .ok_or_else(|| fail("assembly_scene_without_take"))?;
            let resource = session
                .document()
                .resources
                .iter()
                .find(|r| matches!(&r.kind,ResourceKind::TakeVideo{take_id} if take_id==&take.id))
                .ok_or_else(|| fail("take_resource_missing"))?;
            if crate::video::review::states(session.document())
                .get(&take.id)
                .is_some_and(|s| s.outdated)
            {
                plan.warnings.push(format!("outdated_take:{}", scene.id.0));
            }
            (resource, Some(take.id.clone()), None, time(0.), None, true)
        };
        let source = source(session, resource, staging, &mut files)?;
        if source.width == 0 || source.height == 0 {
            return Err(fail("video_stream_missing"));
        }
        let frames = round_samples(seconds(scene.duration), fps).max(1);
        let boundary = plan.clips.last().and_then(|left: &ClipPlan| {
            project
                .assembly
                .boundaries
                .iter()
                .find(|b| b.left == left.scene_id && b.right == scene.id)
        });
        let mode = boundary.map(|b| b.mode).unwrap_or_default();
        // A successful worker confirms the exact frozen incoming dependency.
        // Old/unrelated Takes remain usable with Director's explicit warning.
        let applied = take_id
            .as_ref()
            .and_then(|id| session.document().takes.iter().find(|t| &t.id == id))
            .and_then(|t| t.provenance.as_ref())
            .and_then(|p| p.boundary.as_ref())
            .is_some_and(|receipt| {
                receipt["applied"] == true
                    && imagine_video_contract::boundary::resolve(project, &scene.id)
                        .ok()
                        .flatten()
                        .is_some_and(|current| {
                            receipt["dependency"]
                                == imagine_video_contract::boundary::dependency(
                                    &current,
                                    receipt["dependency"]["reference_previous_clip"] == true,
                                )
                        })
            });
        if mode == BoundaryMode::Transition && !applied {
            plan.warnings
                .push("Transition was not generated; joining the existing Takes.".into());
        }
        let overlap = if mode == BoundaryMode::BlendSeam {
            boundary.unwrap().blend_frames as u64
        } else {
            0
        };
        if overlap > 0 && (overlap > frames || overlap > plan.frames) {
            return Err(fail("assembly_blend_exceeds_clip"));
        }
        if frame_bytes
            .and_then(|n| n.checked_mul(overlap.max(3)))
            .is_none_or(|n| n > crate::video::media::MAX_UPLOAD)
        {
            return Err(fail("assembly_decode_memory_limit"));
        }
        let output_start = plan
            .frames
            .checked_sub(overlap)
            .ok_or_else(|| fail("invalid_assembly_length"))?;
        plan.frames = output_start
            .checked_add(frames)
            .ok_or_else(|| fail("invalid_assembly_length"))?;
        plan.clips.push(ClipPlan {
            scene_id: scene.id.clone(),
            take_id,
            asset_id,
            source,
            source_in,
            source_out,
            timeline_start: scene.start,
            timeline_duration: scene.duration,
            frames,
            output_start,
            overlap,
            boundary: mode,
            include_audio,
        });
    }
    if plan.clips.is_empty() {
        return Err(fail("assembly_empty"));
    }
    plan.samples = (plan.frames as f64 / fps * 44100.).ceil() as u64;
    // The output mix is resident RAM. This bound prevents an allocation overflow/host OOM.
    if plan
        .samples
        .checked_mul(8)
        .is_none_or(|n| n > crate::video::media::MAX_UPLOAD)
    {
        return Err(fail("assembly_audio_memory_limit"));
    }
    let video_soundtracks = imagine_video_contract::audio::references::soundtrack_clips(project);
    for audio in project
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|a| a.enabled && a.usage.final_mix)
        .chain(video_soundtracks.iter())
    {
        let Some(id) = &audio.asset else { continue };
        let source = source(session, reference(session, id)?, staging, &mut files)?;
        if !source.audio {
            return Err(fail("audio_stream_missing"));
        }
        let a = seconds(audio.start);
        let b = a + seconds(audio.duration);
        for (index, clip) in plan.clips.iter().enumerate() {
            let c = seconds(clip.timeline_start);
            let d = c + seconds(clip.timeline_duration);
            let start = a.max(c);
            let mut end = b.min(d);
            let source_in = seconds(audio.source_in) + (start - a);
            if let Some(out) = audio.source_out {
                end = end.min(start + (seconds(out) - source_in).max(0.));
            }
            if end <= start {
                continue;
            }
            let scale = clip.frames as f64 / fps / (d - c);
            let out_start = clip.output_start as f64 / fps + (start - c) * scale;
            let out_end = out_start + (end - start) * scale;
            let previous_fade = clip.overlap as f64 / fps;
            let next_fade = plan
                .clips
                .get(index + 1)
                .map(|n| n.overlap as f64 / fps)
                .unwrap_or(0.);
            plan.audio.push(AudioPlan {
                clip_id: audio.id.clone(),
                source: source.clone(),
                start_sample: round_samples(out_start, 44100.),
                samples: round_samples(out_end, 44100.)
                    .saturating_sub(round_samples(out_start, 44100.)),
                source_in: time(source_in),
                source_out: time(seconds(audio.source_in) + (end - a)),
                source_start_sample: Some(round_samples(source_in, 44100.)),
                source_end_sample: Some(round_samples(
                    seconds(audio.source_in) + (end - a),
                    44100.,
                )),
                volume: audio.volume,
                fade_in: if start == a { audio.fade_in } else { 0. }.max(if start == c {
                    previous_fade
                } else {
                    0.
                }),
                fade_out: if end == b { audio.fade_out } else { 0. }.max(if end == d {
                    next_fade
                } else {
                    0.
                }),
                normalization: audio.normalization.clone(),
            });
        }
    }
    plan.fingerprint=crate::storage::hash(&serde_json::to_vec(&serde_json::json!({"project":project,"options":options,"clips":plan.clips,"audio":plan.audio})).map_err(|_|fail("invalid_assembly_plan"))?);
    let external_sources = session
        .document()
        .resources
        .iter()
        .filter(|resource| resource.source.is_some() && files.contains_key(&resource.id))
        .map(|resource| resource.id.clone())
        .collect();
    Ok(Frozen {
        plan,
        files,
        external_sources,
    })
}

#[cfg(test)]
mod external_source_tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;
    #[test]
    fn assembly_external_audio_source_is_the_original_file_not_a_staged_copy() {
        let root =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let path = root.path().join("own-sine-placeholder.wav");
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36_u32 + 64).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&32000_u32.to_le_bytes());
        wav.extend_from_slice(&64000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&64_u32.to_le_bytes());
        wav.extend_from_slice(&[0; 64]);
        std::fs::write(&path, &wav).unwrap();
        let asset = AssetId("own-audio".into());
        let mut file = File::open(&path).unwrap();
        let resource = Resource::external(
            "own-audio-resource".into(),
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
                "../../../design/video-contract/create-project.json"
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
                kind: MediaKind::Audio,
                name: String::new(),
                library_source: None,
            },
        );
        document.resources.push(resource.clone());
        let staging = SessionStaging::select(root.path()).unwrap();
        session
            .update(0, document, BTreeMap::new(), &staging)
            .unwrap();
        let mut files = BTreeMap::new();
        let description = source(&session, &resource, &staging, &mut files).unwrap();
        assert!(description.audio);
        assert_eq!(description.bytes, wav.len() as u64);
        let actual = files[&resource.id].metadata().unwrap();
        let original = std::fs::metadata(&path).unwrap();
        assert_eq!(
            (actual.dev(), actual.ino()),
            (original.dev(), original.ino())
        );
    }
}
