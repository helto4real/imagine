//! Job orchestration above the single-pass engine. No recursive queue admission.
use crate::{
    character_sheet,
    contract::{JobRequest, ModelOptions, Operation, Sampling},
    engine::Backend,
    model_adapter,
    storage::Store,
};
use anyhow::{Context, Result, bail};
use std::sync::Arc;

/// Internal inference input: resolved settings and owned pixels, no asset IDs or store.
#[derive(Clone)]
pub struct ImagePass {
    pub resolved_models: Option<crate::model_library::ResolvedModels>,
    pub model: String,
    pub model_options: ModelOptions,
    pub sampling: Sampling,
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub references: Vec<Arc<[u8]>>,
    /// MOSS control maps by role (`color`, `depth`), encoded images.
    pub maps: Vec<(&'static str, Arc<[u8]>)>,
}
pub struct Artifact {
    pub role: &'static str,
    pub bytes: Vec<u8>,
}
pub enum Generated {
    Delivery {
        bytes: Arc<Vec<u8>>,
        plan: Box<crate::video::assembly::Plan>,
        pending: Box<crate::video::output::Pending>,
    },
    Images {
        artifacts: Vec<Artifact>,
        pending: Vec<crate::video::output::Pending>,
    },
    Video {
        bytes: Vec<u8>,
        take_artifacts: Vec<crate::video::artifacts::Produced>,
        pending: Box<crate::video::output::Pending>,
    },
}
impl From<Vec<Artifact>> for Generated {
    fn from(v: Vec<Artifact>) -> Self {
        Self::Images {
            artifacts: v,
            pending: Vec::new(),
        }
    }
}
enum Task<'a> {
    Image,
    ControlMaps,
    CharacterSheet(
        &'a character_sheet::CharacterSheet,
        model_adapter::SheetStrategy,
    ),
}
impl<'a> Task<'a> {
    fn resolve(request: &'a JobRequest) -> Result<Self> {
        request.validate().map_err(anyhow::Error::msg)?;
        Ok(if request.operation == Operation::ControlMaps {
            Self::ControlMaps
        } else if request.operation == Operation::CharacterSheet {
            Self::CharacterSheet(
                request.character_sheet.as_ref().context("sheet input")?,
                model_adapter::sheet_strategy(&request.model).map_err(anyhow::Error::msg)?,
            )
        } else {
            Self::Image
        })
    }
}
pub async fn generate(
    engine: &mut dyn Backend,
    request: &JobRequest,
    owner: &str,
    store: &Store,
) -> Result<Generated> {
    if request.model == "minimax-h3" {
        let mut input = crate::video::job::request(request).map_err(crate::engine::EngineError)?;
        let pending = store
            .destination_for_proof(&request.client_metadata["video_destination"])
            .and_then(|d| {
                d.reserve(
                    &request.client_metadata["video_destination"],
                    input.shape.raw_bytes() + 64 * 1024 * 1024,
                )
            })
            .map_err(crate::engine::EngineError)?;
        if let Some(id) = &request.inputs.start_image {
            let asset = store
                .asset(id, owner)
                .await?
                .ok_or(crate::engine::EngineError("asset_not_found"))?;
            crate::video::image::validate_asset(&asset).map_err(crate::engine::EngineError)?;
            let bytes = store.read_asset(id, owner).await?;
            anyhow::ensure!(
                crate::storage::hash(&bytes) == asset["sha256"].as_str().unwrap_or(""),
                crate::engine::EngineError("invalid_h3_start_image")
            );
            let (descriptor, rgb) =
                tokio::task::spawn_blocking(move || crate::video::image::decode(&bytes)).await??;
            input.image = Some(descriptor);
            input.image_rgb = rgb;
        }
        let av = engine.generate_video(&input).await?;
        anyhow::ensure!(
            av.shape == input.shape,
            crate::engine::EngineError("h3_invalid_output")
        );
        engine.stage(Some("encoding_video"));
        let bytes = encode_video(av).await?;
        Ok(Generated::Video {
            bytes,
            take_artifacts: vec![],
            pending: Box::new(pending),
        })
    } else {
        let pending = store.reserve_images(request, owner).await?;
        let artifacts = generate_images(engine, request, owner, store).await?;
        Ok(Generated::Images { artifacts, pending })
    }
}
pub(crate) async fn encode_video(av: crate::short_av::Av) -> Result<Vec<u8>> {
    struct EncodingGuard(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for EncodingGuard {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Release);
        }
    }
    let guard = EncodingGuard(Arc::default());
    let cancelled = guard.0.clone();
    tokio::task::spawn_blocking(move || crate::short_av::encode_cancellable(av, &cancelled))
        .await?
        .map_err(|code| crate::engine::EngineError(code).into())
}
async fn generate_images(
    engine: &mut dyn Backend,
    request: &JobRequest,
    owner: &str,
    store: &Store,
) -> Result<Vec<Artifact>> {
    let task = Task::resolve(request)?;
    let mut pass = ImagePass {
        resolved_models: request.resolved_models.clone(),
        model: request.model.clone(),
        model_options: request.model_options.clone(),
        sampling: request.sampling.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        width: request.width,
        height: request.height,
        seed: request.seed.context("resolved seed")?,
        references: Vec::new(),
        maps: Vec::new(),
    };
    for id in &request.inputs.reference_images {
        pass.references
            .push(store.read_image_asset(id, owner).await?.into());
    }
    for (role, id) in [
        ("color", &request.inputs.color_map),
        ("depth", &request.inputs.depth_map),
    ] {
        if let Some(id) = id {
            pass.maps
                .push((role, store.read_image_asset(id, owner).await?.into()));
        }
    }
    // Region prompts turn the prompt into MOSS's layout prompt.
    if let Some((_, map)) = pass.maps.iter().find(|(role, _)| *role == "color")
        && !request.inputs.regions.is_empty()
    {
        let (map, regions, prompt) = (
            map.clone(),
            request.inputs.regions.clone(),
            pass.prompt.clone(),
        );
        let (width, height) = (pass.width, pass.height);
        pass.prompt = tokio::task::spawn_blocking(move || {
            crate::moss_layout::layout_prompt(&prompt, &regions, &map, width, height)
        })
        .await??;
    }
    match task {
        Task::ControlMaps => {
            let id = request
                .inputs
                .source_image
                .as_ref()
                .context("source image")?;
            let photo = store.read_image_asset(id, owner).await?;
            let (width, height) = (request.width, request.height);
            let photo = tokio::task::spawn_blocking(move || {
                crate::workflow_image::cover(&photo, width, height)
            })
            .await??;
            engine.stage(Some("control_maps"));
            let [depth, regions] = engine.control_maps(&photo, width, height).await?;
            Ok(vec![
                Artifact {
                    role: "depth_map",
                    bytes: depth,
                },
                Artifact {
                    role: "region_map",
                    bytes: regions,
                },
            ])
        }
        Task::Image => Ok(vec![Artifact {
            role: "image",
            bytes: engine.generate(&pass).await?,
        }]),
        Task::CharacterSheet(sheet, model_adapter::SheetStrategy::QwenTwoPass) => {
            // Validate/read the guide before spending GPU time on the portrait.
            let sizes = sheet.layout.pass_sizes().map_err(anyhow::Error::msg)?;
            let guide = if let Some(id) = &request.inputs.pose_guide {
                store.read_image_asset(id, owner).await?
            } else {
                character_sheet::pose_guide(sizes[1].0, sizes[1].1)?
            };
            let count = pass.references.len();
            pass.prompt = model_adapter::sheet_prompt(sheet, count, false);
            (pass.width, pass.height) = sizes[0];
            engine.stage(Some("portrait"));
            let portrait = engine.generate(&pass).await?;
            check_canvas(&portrait, sizes[0])?;
            pass.references.push(portrait.clone().into());
            pass.references.push(guide.into());
            pass.prompt = model_adapter::sheet_prompt(sheet, count, true);
            pass.seed = model_adapter::next_seed(pass.seed);
            (pass.width, pass.height) = sizes[1];
            engine.stage(Some("turnaround"));
            let turnaround = engine.generate(&pass).await?;
            check_canvas(&turnaround, sizes[1])?;
            engine.stage(Some("compose"));
            let layout = sheet.layout.clone();
            let result = tokio::task::spawn_blocking(move || {
                let sheet = character_sheet::compose(&portrait, &turnaround, &layout)?;
                Ok::<_, anyhow::Error>(vec![
                    Artifact {
                        role: "character_sheet",
                        bytes: sheet,
                    },
                    Artifact {
                        role: "portrait",
                        bytes: portrait,
                    },
                    Artifact {
                        role: "turnaround",
                        bytes: turnaround,
                    },
                ])
            })
            .await??;
            Ok(result)
        }
        Task::CharacterSheet(sheet, model_adapter::SheetStrategy::MossSinglePass) => {
            pass.prompt = model_adapter::moss_sheet_prompt(sheet);
            (pass.width, pass.height) = sheet.layout.dimensions();
            let bytes = engine.generate(&pass).await?;
            check_canvas(&bytes, (pass.width, pass.height))?;
            Ok(vec![Artifact {
                role: "character_sheet",
                bytes,
            }])
        }
    }
}
fn check_canvas(bytes: &[u8], expected: (u32, u32)) -> Result<()> {
    let info = crate::storage::inspect_image(bytes.to_vec())?;
    if (info.width, info.height) != expected {
        bail!(crate::engine::EngineError("unexpected_output_dimensions"));
    }
    Ok(())
}

/// Shared admission/validation checks; includes the optional pose asset.
pub async fn reference_metadata(
    request: &JobRequest,
    owner: &str,
    store: &Store,
) -> Result<Vec<serde_json::Value>, crate::scheduler::Fault> {
    use crate::scheduler::Fault;
    let mut result = Vec::new();
    for id in request.asset_ids() {
        let asset = store
            .asset(id, owner)
            .await?
            .ok_or(Fault::new(404, "asset_not_found"))?;
        if request.model == "minimax-h3" {
            crate::video::image::validate_asset(&asset).map_err(|e| Fault::new(422, e))?;
            result.push(serde_json::json!({"asset_id":id,"sha256":asset["sha256"],"source_width":asset["width"],"source_height":asset["height"],"width":request.width,"height":request.height,"preprocessing":"h3_pillow_lanczos_stretch_v1"}));
            continue;
        }
        let w = asset["width"].as_u64().unwrap() as u32;
        let h = asset["height"].as_u64().unwrap() as u32;
        let (_, _, width, height) = crate::workflow_image::dimensions(
            w,
            h,
            request
                .model_options
                .qwen_image_2_1
                .reference_max_pixels_per_image,
        );
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(Fault::new(422, "unsupported_reference_geometry"));
        }
        result.push(serde_json::json!({"asset_id":id,"source_width":w,"source_height":h,"vae_width":width,"vae_height":height,
            "pixels":width*height,"source":"aio_shrink_center_crop_v1"}));
    }
    Ok(result)
}
