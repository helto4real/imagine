//! Opt-in CPU-only native Timeline QA service. No model loading or live coordinator.
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action},
    storage::Store,
    video::{contract::*, output::Destination, persistence::*},
    video_destination::{self as d, Config, System},
};
use serde_json::json;
use std::{
    io::{Cursor, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
    sync::{Semaphore, watch},
    task::JoinSet,
};
#[path = "support/authoring_fixture.rs"]
mod authoring_fixture;
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
struct Synthetic {
    preview_state: Option<watch::Sender<EngineState>>,
    cache: Option<imagine_server::video::reference_cache_owner::Owner>,
    evidence: PathBuf,
    delay: std::time::Duration,
    fail_next: bool,
    failure_code: &'static str,
}
impl Backend for Synthetic {
    fn begin_job(&mut self, id: &str) {
        if let Some(state) = &self.preview_state {
            state.borrow().sampling_preview.begin(id);
            state.send_modify(|s| {
                s.job_id = Some(id.into());
                s.phase = "sampling".into();
            });
        }
    }
    fn mark_idle(&mut self) {
        if let Some(state) = &self.preview_state {
            let id = state.borrow().job_id.clone();
            if let Some(id) = id {
                state.borrow().sampling_preview.end(&id);
            }
            state.send_modify(|s| s.job_id = None);
        }
    }
    fn reference_cache(&self) -> Option<Arc<dyn imagine_server::video::reference_cache::Control>> {
        self.cache
            .as_ref()
            .map(|c| c.clone() as Arc<dyn imagine_server::video::reference_cache::Control>)
    }
    fn generate<'a>(
        &'a mut self,
        request: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        let preview_state = self.preview_state.clone();
        let total = request.sampling.steps;
        Box::pin(async move {
            if let Some(state) = preview_state {
                let hub = state.borrow().sampling_preview.clone();
                let job = state.borrow().job_id.clone().unwrap();
                let settings = hub.settings();
                let start = std::time::Instant::now();
                let mut slot = 0;
                for step in 1..=total {
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    let elapsed = start.elapsed().as_secs_f64();
                    state.send_modify(|s| {
                        s.phase = "inference".into();
                        s.requested_steps = total;
                        s.progress = Some(imagine_server::progress::SamplingProgress {
                            completed: step,
                            total,
                            elapsed_seconds: elapsed,
                            remaining_seconds: elapsed / f64::from(step) * f64::from(total - step),
                            updated_at_ms: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64,
                        });
                    });
                    let due = (elapsed / settings.interval_seconds).floor() as u64;
                    if settings.enabled && due > slot {
                        slot = due;
                        hub.publish(
                            &job,
                            imagine_server::h3_protocol::preview::Metadata {
                                sequence: u64::from(step),
                                step,
                                width: 64,
                                height: 32,
                                frames: 1,
                                frame_duration_ms: 500,
                                source_duration_seconds: 0.5,
                            },
                            vec![(step % 200 + 30) as u8; 64 * 32 * 3],
                        );
                    }
                }
            }
            let mut out = Cursor::new(vec![]);
            image::RgbImage::from_pixel(64, 64, image::Rgb([80, 110, 150]))
                .write_to(&mut out, image::ImageFormat::Png)?;
            Ok(out.into_inner())
        })
    }
    fn generate_video<'a>(
        &'a mut self,
        input: &'a imagine_server::h3_protocol::Request,
    ) -> EngineFuture<'a, imagine_server::short_av::Av> {
        let fail = std::mem::take(&mut self.fail_next);
        let failure_code = self.failure_code;
        let delay = self.delay;
        let preview_state = self.preview_state.clone();
        let capture = self
            .evidence
            .join(format!("generated-input-{}.json", input.id));
        Box::pin(async move {
            std::fs::write(
                capture,
                serde_json::to_vec_pretty(
                    &json!({"seed":input.seed,"prompt":input.prompt,"shape":input.shape,"sampling":input.sampling,"options":input.options,"reference_quality":input.references.quality,
                        "boundary_receipt":input.references.boundary_receipt(),"reference_manifest":input.references.manifest,
                        "start_image":input.image,"reference_shapes":{
                            "images":input.references.images.iter().map(|i|[i.width,i.height]).collect::<Vec<_>>(),
                            "videos":input.references.videos.iter().map(|v|json!({"width":v.width,"height":v.height,"frames":v.frames,"fps":v.source_fps,"audio_samples":v.audio.as_ref().map(|a|a.samples)})).collect::<Vec<_>>()}}),
                )?,
            )?;
            tokio::time::sleep(delay).await;
            if let Some(state) = preview_state {
                // Synthetic UI fixture only: real lifecycle/API but no model.
                let hub = state.borrow().sampling_preview.clone();
                let job = state.borrow().job_id.clone().unwrap();
                let settings = hub.settings();
                let start = std::time::Instant::now();
                let mut slot = 0;
                let mut sequence = 0;
                for step in 1..=20 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    let elapsed = start.elapsed().as_secs_f64();
                    let due = (elapsed / settings.interval_seconds).floor() as u64;
                    if settings.enabled && due > slot {
                        slot = due;
                        sequence += 1;
                        let metadata = imagine_server::h3_protocol::preview::Metadata {
                            sequence,
                            step,
                            width: 64,
                            height: 32,
                            frames: 2,
                            frame_duration_ms: 812,
                            source_duration_seconds: 39. / 24.,
                        };
                        let mut rgb = vec![30; 64 * 32 * 3];
                        rgb.extend(vec![130; 64 * 32 * 3]);
                        hub.publish(&job, metadata, rgb);
                    }
                }
            }
            anyhow::ensure!(!fail, imagine_server::engine::EngineError(failure_code));
            Ok(pattern(input.shape))
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        if let Some(state) = &self.preview_state {
            state.borrow().sampling_preview.clear();
        }
        Box::pin(async { Ok(()) })
    }
}
fn pattern(shape: imagine_server::h3_protocol::Shape) -> imagine_server::short_av::Av {
    let frames = (0..shape.frames)
        .map(|frame| {
            let mut rgb = Vec::with_capacity(shape.frame_bytes());
            for y in 0..shape.height {
                for x in 0..shape.width {
                    let sea = y > shape.height * 3 / 5;
                    let beacon =
                        x > shape.width * 2 / 3 && x < shape.width * 3 / 4 && y > shape.height / 4;
                    let color = if beacon {
                        [220, 180, 100]
                    } else if sea {
                        [28 + (frame % 16) as u8, 54, 82]
                    } else {
                        [65, 85, 120]
                    };
                    rgb.extend_from_slice(&color);
                }
            }
            rgb
        })
        .collect();
    let pcm = (0..shape.audio_samples())
        .flat_map(|sample| {
            let v = ((sample as f32 * 220. * std::f32::consts::TAU / 32000.).sin() * 0.02)
                .to_le_bytes();
            v.into_iter().chain(v)
        })
        .collect();
    let continuation = if shape.frames >= 22 {
        let descriptor = imagine_server::h3_protocol::continuation::Descriptor {
            handoff_upscale: None,
            schema_version: 1,
            shape,
            sampled_frames: shape.frames,
            visible_end_frame: shape.frames,
            selected_end_frame: shape.frames,
        };
        let bytes = vec![0.125f32; descriptor.bytes() / 4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        Some(imagine_server::h3_protocol::continuation::Artifact { descriptor, bytes })
    } else {
        None
    };
    imagine_server::short_av::Av {
        continuation,
        shape,
        frames,
        pcm,
    }
}
fn source(root: &Path, bytes: &[u8]) -> anyhow::Result<std::fs::File> {
    let mut file = tempfile::tempfile_in(root.join("tmp"))?;
    file.write_all(bytes)?;
    Ok(file)
}
fn seed(
    root: &Path,
    with_audio: bool,
    assembly: bool,
    long_assembly: bool,
    with_cast: bool,
    review_takes: bool,
) -> anyhow::Result<()> {
    let model = ModelSelection {
        adapter: "h3".into(),
        model: "minimax-h3".into(),
        variant: "reference".into(),
        profile: "h3-av-noref-v2".into(),
        profile_revision: "2".into(),
    };
    let mut project = VideoProject::create(CreateProject {
        generation: None,
        schema_version: 1,
        default_model: model.clone(),
        delivery: DeliverySettings {
            width: 416,
            height: 224,
            fps: Time { num: 24, den: 1 },
        },
    })
    .map_err(StorageError::from)?;
    project.global_prompt = "Synthetic lighthouse study".into();
    let slot = SettingsSlot {
        model: model.clone(),
        schema_version: 1,
        values: json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}),
    };
    for (index, (start, duration, description)) in [
        (0, 3, "Across the rocks"),
        (3, 5, "The light turns"),
        (8, 4, "The horizon"),
    ]
    .into_iter()
    .enumerate()
    {
        let (start, duration) = if long_assembly {
            (index as u64 * 50, 50)
        } else {
            (start, duration)
        };
        project.scenes.push(Scene {
            id: SceneId(format!("scene_qa_{}", index + 1)),
            start: Time { num: start, den: 1 },
            duration: Time {
                num: duration,
                den: 1,
            },
            description: description.into(),
            model: ModelChoice::ProjectDefault,
            settings: vec![slot.clone()],
            sections: vec![Section {
                id: SectionId(format!("section_qa_{}", index + 1)),
                start: Time { num: 0, den: 1 },
                duration: Time {
                    num: duration,
                    den: 1,
                },
                prompt: description.into(),
            }],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
    }
    let mut session = ProjectSession::new(project)?;
    let mut document = session.document().clone();
    let staging = SessionStaging::select(&root.join("tmp"))?;
    let mut files = std::collections::BTreeMap::new();
    let mut image = Cursor::new(vec![]);
    image::RgbImage::from_fn(128, 80, |x, y| {
        image::Rgb([40 + (x / 3) as u8, 60 + (y / 3) as u8, 110])
    })
    .write_to(&mut image, image::ImageFormat::Png)?;
    let asset = AssetId("asset_qa_lighthouse".into());
    document.project.assets.insert(
        asset.clone(),
        Asset {
            library_source: None,
            name: String::new(),
            id: asset.clone(),
            revision: 1,
            kind: MediaKind::Image,
        },
    );
    let mut file = source(root, &image.into_inner())?;
    let resource = Resource::inspect(
        "reference_qa".into(),
        ResourceKind::Reference {
            asset_id: asset,
            revision: 1,
        },
        &mut file,
    )?;
    files.insert(resource.key(), file);
    document.resources.push(resource);
    if with_audio {
        let samples = 32000_u32 * 2;
        let length = samples * 8;
        let mut wav = Vec::with_capacity(length as usize + 44);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + length).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&3_u16.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&32000_u32.to_le_bytes());
        wav.extend_from_slice(&256000_u32.to_le_bytes());
        wav.extend_from_slice(&8_u16.to_le_bytes());
        wav.extend_from_slice(&32_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&length.to_le_bytes());
        for i in 0..samples {
            let v = (0.02 * (i as f32 * 220. * std::f32::consts::TAU / 32000.).sin()).to_le_bytes();
            wav.extend_from_slice(&v);
            wav.extend_from_slice(&v);
        }
        let id = AssetId("asset_qa_tone".into());
        document.project.assets.insert(
            id.clone(),
            Asset {
                library_source: None,
                name: String::new(),
                id: id.clone(),
                kind: MediaKind::Audio,
                revision: 1,
            },
        );
        let mut file = source(root, &wav)?;
        let resource = Resource::inspect(
            "reference_qa_tone".into(),
            ResourceKind::Reference {
                asset_id: id,
                revision: 1,
            },
            &mut file,
        )?;
        files.insert(resource.key(), file);
        document.resources.push(resource);
    }
    let shape = imagine_server::h3_protocol::Selection {
        seconds: 3.,
        ratio: "16:9".into(),
        ..Default::default()
    }
    .resolve()
    .map_err(anyhow::Error::msg)?;
    let mut av = pattern(shape);
    let endpoint = imagine_server::video::artifacts::Produced::endpoint(&av)?;
    let motion = av
        .continuation
        .take()
        .map(imagine_server::video::artifacts::Produced::H3Motion);
    let video = imagine_server::short_av::encode(av).map_err(anyhow::Error::msg)?;
    let mut file = source(root, &video)?;
    let take = TakeId("take_qa_first".into());
    let resource = Resource::inspect(
        "video_qa_first".into(),
        ResourceKind::TakeVideo {
            take_id: take.clone(),
        },
        &mut file,
    )?;
    files.insert(resource.key(), file);
    document.resources.push(resource);
    document.takes.push(StoredTake {
        id: take,
        scene_id: SceneId("scene_qa_1".into()),
        status: TakeStatus::Candidate,
        provenance: Some(TakeProvenance {
            boundary: None,
            resolved_loras: None,
            model,
            seed: 42,
            input_revision: imagine_server::video::planning::input_revision(
                &document.project,
                &SceneId("scene_qa_1".into()),
            )
            .map_err(StorageError::from)?,
            compiled_prompt: "Synthetic fixture; no trained model".into(),
            settings: slot,
            references: vec![],
            source_take: None,
            continuity: None,
            authored: None,
        }),
    });
    if let Some(motion) = motion {
        let (resource, file) = motion.prepare(&document.takes[0], &staging)?;
        files.insert(resource.key(), file);
        document.resources.push(resource);
    }
    let (resource, file) = endpoint.prepare(&document.takes[0], &staging)?;
    files.insert(resource.key(), file);
    document.resources.push(resource);
    if review_takes {
        for index in 1..=2 {
            let mut av = pattern(shape);
            av.continuation = None;
            av.pcm = (0..av.pcm.len() / 8)
                .flat_map(|sample| {
                    let v = (sample as f32 * (220. + index as f32 * 110.) * std::f32::consts::TAU
                        / 32000.)
                        .sin()
                        * 0.04;
                    let value = v.to_le_bytes();
                    value.into_iter().chain(value)
                })
                .collect();
            for frame in &mut av.frames {
                for rgb in frame.chunks_exact_mut(3) {
                    rgb[index] = rgb[index].saturating_add(45);
                }
            }
            let mut take = document.takes[0].clone();
            take.id = TakeId(format!("take_qa_review_{index}"));
            take.provenance.as_mut().unwrap().seed += index as u64;
            let video = imagine_server::short_av::encode(av).map_err(anyhow::Error::msg)?;
            let mut file = source(root, &video)?;
            let resource = Resource::inspect(
                format!("video_qa_review_{index}"),
                ResourceKind::TakeVideo {
                    take_id: take.id.clone(),
                },
                &mut file,
            )?;
            files.insert(resource.key(), file);
            document.resources.push(resource);
            document.takes.push(take);
        }
        let first = &mut document.takes[0];
        first.status = TakeStatus::Accepted;
        document.project.scenes[0].accepted_take =
            Some(first.accepted_summary(&document.resources)?);
    }
    if assembly {
        for index in 0..document.project.scenes.len() {
            let scene_id = document.project.scenes[index].id.clone();
            let mut take = document.takes[0].clone();
            if index > 0 {
                take.id = TakeId(format!("take_qa_assembly_{index}"));
                take.scene_id = scene_id.clone();
                take.provenance.as_mut().unwrap().input_revision =
                    imagine_server::video::planning::input_revision(&document.project, &scene_id)
                        .map_err(StorageError::from)?;
                let mut resource = document
                    .resources
                    .iter()
                    .find(|r| matches!(r.kind, ResourceKind::TakeVideo { .. }))
                    .unwrap()
                    .clone();
                resource.id = format!("video_qa_assembly_{index}");
                resource.kind = ResourceKind::TakeVideo {
                    take_id: take.id.clone(),
                };
                document.resources.push(resource);
                take.status = TakeStatus::Accepted;
                document.takes.push(take.clone());
            } else {
                document.takes[0].status = TakeStatus::Accepted;
            }
            document.project.scenes[index].accepted_take =
                Some(take.accepted_summary(&document.resources)?);
        }
        document.project.assembly.boundaries = vec![
            imagine_server::video::assembly::Boundary {
                left: document.project.scenes[0].id.clone(),
                right: document.project.scenes[1].id.clone(),
                mode: imagine_server::video::assembly::BoundaryMode::ContinuousShot,
                blend_frames: 0,
                ..imagine_server::video::assembly::Boundary::new(
                    document.project.scenes[0].id.clone(),
                    document.project.scenes[1].id.clone(),
                )
            },
            imagine_server::video::assembly::Boundary {
                left: document.project.scenes[1].id.clone(),
                right: document.project.scenes[2].id.clone(),
                mode: imagine_server::video::assembly::BoundaryMode::BlendSeam,
                blend_frames: 6,
                ..imagine_server::video::assembly::Boundary::new(
                    document.project.scenes[1].id.clone(),
                    document.project.scenes[2].id.clone(),
                )
            },
        ];
        if with_audio {
            let mut clip = AudioClip::new(
                AudioClipId("audio_assembly".into()),
                Time { num: 1, den: 1 },
                Time { num: 3, den: 1 },
                &document.project.audio,
            );
            clip.asset = Some(AssetId("asset_qa_tone".into()));
            clip.fade_in = 0.1;
            clip.fade_out = 0.2;
            document.project.audio_tracks.push(AudioTrack {
                id: AudioTrackId("audio_assembly_track".into()),
                clips: vec![clip],
            });
        }
    }
    if with_cast {
        use imagine_video_contract::creative::*;
        document.project.creative.references =
            [("keeper", "Lighthouse keeper"), ("visitor", "Visitor")]
                .into_iter()
                .map(|(alias, name)| VisualReference {
                    id: format!("cast_{alias}"),
                    name: name.into(),
                    alias: alias.into(),
                    kind: "Character".into(),
                    description: format!("Own fictional {name}; wears a blue coat."),
                    tags: vec!["own fixture".into()],
                    enabled: true,
                    strength: 1.,
                    retention: String::new(),
                    relationship: None,
                    images: vec![],
                    voice: Some(Voice {
                        reference_settings: Default::default(),
                        language: "Swedish".into(),
                        description: "calm clear speech".into(),
                        assets: vec![],
                    }),
                    origin: None,
                })
                .collect();
    }
    session.update(0, document, files, &staging)?;
    session.save_to(&root.join("saved-project"))?;
    Ok(())
}
async fn synthetic_ticket(stream: tokio::net::UnixStream, semaphore: Arc<Semaphore>) {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
        return;
    }
    if writer.write_all(b"{\"queued\":true}\n").await.is_err() {
        return;
    }
    let _permit = semaphore.acquire_owned().await.unwrap();
    if writer.write_all(b"{\"granted\":true}\n").await.is_err() {
        return;
    }
    line.clear();
    let _ = reader.read_line(&mut line).await;
}
fn copy_owned_project(source: &Path, destination: &Path, budget: &mut u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        source.canonicalize()? == source,
        "own project path must not be a symlink"
    );
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = destination.join(entry.file_name());
        if kind.is_dir() {
            std::fs::create_dir(&path)?;
            copy_owned_project(&entry.path(), &path, budget)?;
        } else {
            anyhow::ensure!(
                kind.is_file(),
                "only regular own fixture files can be copied"
            );
            *budget = budget
                .checked_sub(entry.metadata()?.len())
                .ok_or_else(|| anyhow::anyhow!("fixture copy too large"))?;
            std::fs::copy(entry.path(), path)?;
        }
    }
    Ok(())
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let arguments = std::env::args().skip(2).collect::<Vec<_>>();
    anyhow::ensure!(
        arguments.iter().all(|v| v == "--slow"
            || v == "--sampling-preview"
            || v == "--private-storage"
            || v == "--reference-cache"
            || v == "--h3-failure"
            || v.strip_prefix("--port=")
                .is_some_and(|p| p.parse::<u16>().is_ok())
            || v == "--authoring"
            || v == "--story-failure"
            || v == "--story-slow"
            || v == "--fail-first-comparison"
            || v == "--fail-first"
            || v == "--drop-first-status"
            || v == "--drop-first-admission"
            || v == "--drop-first-freeze"
            || v == "--publication-failure"
            || v == "--editor-save-failures"
            || v == "--lora-catalog"
            || v == "--with-audio"
            || v == "--review-takes"
            || v == "--assembly"
            || v == "--assembly-long"
            || v == "--cast"
            || v.strip_prefix("--reopen=").is_some_and(|c| c.len() > 1
                && c.len() < 9
                && c.starts_with('q')
                && c[1..].bytes().all(|b| b.is_ascii_digit()))),
        "unknown fixture option"
    );
    let delay = std::time::Duration::from_millis(if arguments.iter().any(|v| v == "--slow") {
        2000
    } else {
        150
    });
    let fail_next = arguments.iter().any(|v| v == "--fail-first");
    let drop_status = Arc::new(std::sync::atomic::AtomicBool::new(
        arguments.iter().any(|v| v == "--drop-first-status"),
    ));
    let drop_admission = Arc::new(std::sync::atomic::AtomicBool::new(
        arguments.iter().any(|v| v == "--drop-first-admission"),
    ));
    let drop_freeze = Arc::new(std::sync::atomic::AtomicBool::new(
        arguments.iter().any(|v| v == "--drop-first-freeze"),
    ));
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("owned case root required"))?,
    );
    anyhow::ensure!(
        task_fixtures::is_owned(&root),
        "only the approved synthetic task root is allowed"
    );
    let fsid =
        d::fsid(&d::directory(&root).map_err(anyhow::Error::msg)?).map_err(anyhow::Error::msg)?;
    let base = Config {
        root: root.clone(),
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
        fsid,
    };
    let (_, fd) = d::inspect(&base, &System).map_err(anyhow::Error::msg)?;
    d::writable(&fd, 0).map_err(anyhow::Error::msg)?;
    for name in [
        "state",
        "output",
        "tmp",
        "ui",
        "saved-project",
        "evidence",
        "cache",
    ] {
        std::fs::create_dir_all(root.join(name))?;
    }
    anyhow::ensure!(
        !root.join("saved-project/project.json").exists(),
        "QA root already has a project; use a new case directory"
    );
    if let Some(case) = arguments.iter().find_map(|a| a.strip_prefix("--reopen=")) {
        let source = root.parent().unwrap().join(case).join("saved-project");
        anyhow::ensure!(
            task_fixtures::is_owned(&source),
            "own synthetic reopen source required"
        );
        anyhow::ensure!(
            source != root.join("saved-project"),
            "use a new fixture root"
        );
        let mut budget = 2_u64 * 1024 * 1024 * 1024;
        copy_owned_project(&source, &root.join("saved-project"), &mut budget)?;
    } else {
        seed(
            &root,
            arguments.iter().any(|v| v == "--with-audio"),
            arguments.iter().any(|v| v == "--assembly"),
            arguments.iter().any(|v| v == "--assembly-long"),
            arguments.iter().any(|v| v == "--cast"),
            arguments.iter().any(|v| v == "--review-takes"),
        )?;
    }
    let config = Config {
        root: root.join("output"),
        ..base.clone()
    };
    let destination = Destination::new(config.clone());
    destination.proof().map_err(anyhow::Error::msg)?;
    std::fs::write(
        root.join("state/video-output.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let private = if arguments.iter().any(|v| v == "--private-storage") {
        let path = root.join("private");
        let runtime = root.join("runtime");
        std::fs::create_dir(&path)?;
        std::fs::create_dir(&runtime)?;
        Some(
            imagine_server::private_storage::Root::with_environment(
                Config {
                    root: path,
                    ..base.clone()
                },
                Arc::new(System),
            )
            .and_then(|private| private.with_runtime_directory(runtime))
            .map_err(anyhow::Error::msg)?,
        )
    } else {
        None
    };
    let store = if let Some(private) = &private {
        Store::open_private(private.clone(), "own-image-workspace".into()).await?
    } else {
        let mut store = Store::open(&root.join("state")).await?;
        store.videos.destination = Some(destination);
        store
    };
    let listener = UnixListener::bind(root.join("c.sock"))?;
    let semaphore = Arc::new(Semaphore::new(1));
    let broker = tokio::spawn(async move {
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                value=listener.accept()=>{let(stream,_)=value.unwrap();tasks.spawn(synthetic_ticket(stream,semaphore.clone()));},
                _=tasks.join_next(),if !tasks.is_empty()=>{}
            }
        }
    });
    let (state, engine) = watch::channel(EngineState {
        backend: "rust".into(),
        h3_available: true,
        ..Default::default()
    });
    let (control, scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Synthetic {
            preview_state: arguments
                .iter()
                .any(|a| a == "--sampling-preview")
                .then_some(state),
            failure_code: if arguments.iter().any(|a| a == "--h3-failure") {
                "h3_generation_failed"
            } else {
                "generation_failed"
            },
            cache: arguments
                .iter()
                .any(|a| a == "--reference-cache")
                .then(imagine_server::video::reference_cache_owner::new),
            delay,
            fail_next,
            evidence: root.join("evidence"),
        }),
        Some(root.join("c.sock")),
    )
    .await?;
    let enhancement = match &private {
        Some(private) => imagine_server::enhancement::Service::open_private(
            private.clone(),
            "library/enhancement-profiles.json".into(),
        )?,
        None => imagine_server::enhancement::Service::open(root.join("state/enhancement.json"))?,
    };
    let authoring_worker = if std::env::args().any(|a| a == "--authoring") {
        let (url, worker, state) = authoring_fixture::spawn().await;
        if arguments.iter().any(|argument| argument == "--story-slow") {
            state
                .delay_ms
                .store(1000, std::sync::atomic::Ordering::SeqCst);
        }
        if std::env::args().any(|a| a == "--story-failure") {
            state
                .fail_second_once
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        enhancement
            .save(imagine_server::enhancement_types::ProfileUpdate {
                profile: imagine_server::enhancement_types::Profile {
                    base_url: url,
                    name: "Synthetic local test".into(),
                    model: "own-story-fixture".into(),
                    ..Default::default()
                },
            })
            .map_err(|e| anyhow::anyhow!(e.code))?;
        Some(worker)
    } else {
        None
    };
    let app = App {
        video: control.video.clone(),
        enhancement,
        library: if arguments.iter().any(|a| a == "--lora-catalog") {
            Arc::new(std::sync::Mutex::new(
                imagine_server::model_library::Library::load(
                    root.join("models"),
                    root.join("state/model-library.json"),
                )?,
            ))
        } else {
            Default::default()
        },
        profiles: Arc::new(imagine_server::profiles::Catalog::load(Path::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/config/image/profiles"),
        ))?),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    };
    let port = arguments
        .iter()
        .find_map(|a| a.strip_prefix("--port="))
        .unwrap_or("0");
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    let url = format!("http://{}", listener.local_addr()?);
    std::fs::write(
        root.join("evidence/endpoint.json"),
        serde_json::to_vec_pretty(
            &json!({"api":url,"destination":base,"project":root.join("saved-project"),"backend":"synthetic CPU only; no live models/GPU/coordinator"}),
        )?,
    )?;
    println!("OWN SYNTHETIC TIMELINE FIXTURE {url}");
    let stop = tokio_util::sync::CancellationToken::new();
    let token = stop.clone();
    let fail_comparison = arguments.iter().any(|a| a == "--fail-first-comparison");
    let preview_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let editor_faults = arguments.iter().any(|a| a == "--editor-save-failures");
    let fault_root = root.clone();
    let library_fault_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let review_fault_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let router=api::router(app).layer(axum::middleware::from_fn(move |request: axum::extract::Request,next: axum::middleware::Next| {
        let drop_status=drop_status.clone();
        let drop_admission=drop_admission.clone();
        let drop_freeze=drop_freeze.clone();
        let preview_reads=preview_reads.clone();
        let fault_root=fault_root.clone();
        let library_fault_seen=library_fault_seen.clone();
        let review_fault_seen=review_fault_seen.clone();
        async move {
            use axum::response::IntoResponse;
            if editor_faults && request.method()==axum::http::Method::POST {
                let path=request.uri().path();
                let library=path.starts_with("/v1/video/libraries/")&&path.ends_with("/characters");
                let review=path.starts_with("/v1/video/authoring/")&&path.ends_with("/review");
                let fail=library&&fault_root.join("librarysavefault.signal").is_file()&&!library_fault_seen.swap(true,std::sync::atomic::Ordering::AcqRel)
                    ||review&&fault_root.join("storyreviewfault.signal").is_file()&&!review_fault_seen.swap(true,std::sync::atomic::Ordering::AcqRel);
                if fail {return (axum::http::StatusCode::INSUFFICIENT_STORAGE,axum::Json(json!({"error":{"code":"storage_io_failed"}}))).into_response();}
            }
            let preview=request.method()==axum::http::Method::GET&&request.uri().path().strip_prefix("/v1/video/assets/").is_some_and(|tail|!tail.is_empty()&&!tail.contains('/'));
            if fail_comparison&&preview&&preview_reads.fetch_add(1,std::sync::atomic::Ordering::AcqRel)==2 {
                return (axum::http::StatusCode::NOT_FOUND,axum::Json(json!({"error":{"code":"synthetic_media_unavailable"}}))).into_response();
            }
            let status=request.method()==axum::http::Method::GET && request.uri().path().strip_prefix("/v1/jobs/").is_some_and(|tail|!tail.is_empty()&&!tail.contains('/'));
            if status && drop_status.swap(false,std::sync::atomic::Ordering::AcqRel) {
                return (axum::http::StatusCode::SERVICE_UNAVAILABLE,axum::Json(json!({"error":{"code":"synthetic_status_lost","message":"Synthetic status response lost. Reconnect."}}))).into_response();
            }
            let admission=request.method()==axum::http::Method::POST && (request.uri().path()=="/v1/jobs" || request.uri().path().starts_with("/v1/video/assemblies/") && request.uri().path().ends_with("/jobs"));
            let freeze=request.method()==axum::http::Method::POST && request.uri().path().starts_with("/v1/video/projects/") && request.uri().path().ends_with("/assembly");
            let response=next.run(request).await;
            if freeze && response.status().is_success() && drop_freeze.swap(false,std::sync::atomic::Ordering::AcqRel) {
                return (axum::http::StatusCode::SERVICE_UNAVAILABLE,axum::Json(json!({"error":{"code":"synthetic_freeze_lost"}}))).into_response();
            }
            if admission && response.status().is_success() && drop_admission.swap(false,std::sync::atomic::Ordering::AcqRel) {
                return (axum::http::StatusCode::SERVICE_UNAVAILABLE,axum::Json(json!({"error":{"code":"synthetic_admission_lost","message":"Synthetic admission response lost. Retrieve submission."}}))).into_response();
            }
            response
        }
    }));
    // Opt-in fault injection is confined to this synthetic example, never the product router.
    let router = if arguments.iter().any(|a| a == "--publication-failure") {
        sqlx::query("CREATE TRIGGER own_fixture_save_failure BEFORE INSERT ON assets WHEN EXISTS(SELECT 1 FROM jobs WHERE status='running' AND json_extract(request,'$.kind')='video_assembly') BEGIN SELECT RAISE(FAIL,'own fixture publication error'); END").execute(&store.db).await?;
        let publication_store = store.clone();
        let publication_video = control.video.clone();
        router.route(
            "/__fixture/allow-publication",
            axum::routing::post(move || {
                let store = publication_store.clone();
                let video = publication_video.clone();
                async move {
                    let retained = video.lock().await.assemblies.values().filter_map(|entry|entry.rendered.get()).map(|bytes|json!({"bytes":bytes.len(),"sha256":imagine_server::storage::hash(&bytes)})).collect::<Vec<_>>();
                    sqlx::query("DROP TRIGGER IF EXISTS own_fixture_save_failure")
                        .execute(&store.db)
                        .await
                        .unwrap();
                    axum::Json(json!({"publication_restored":true,"retained_encoded_results":retained}))
                }
            }),
        )
    } else {
        router
    };
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(token.cancelled_owned())
            .await
    });
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=term.recv()=>{}}
    stop.cancel();
    server.await??;
    if let Some(worker) = authoring_worker {
        worker.abort();
        let _ = worker.await;
    }
    control
        .call(Action::Shutdown)
        .await
        .map_err(|e| anyhow::anyhow!(e.code))?;
    scheduler.await??;
    broker.abort();
    let _ = broker.await;
    store.close().await?;
    Ok(())
}
