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
}
pub struct Artifact {
    pub role: &'static str,
    pub bytes: Vec<u8>,
}
enum Task<'a> {
    Image,
    CharacterSheet(
        &'a character_sheet::CharacterSheet,
        model_adapter::SheetStrategy,
    ),
}
impl<'a> Task<'a> {
    fn resolve(request: &'a JobRequest) -> Result<Self> {
        request.validate().map_err(anyhow::Error::msg)?;
        Ok(if request.operation == Operation::CharacterSheet {
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
    };
    for id in &request.inputs.reference_images {
        pass.references
            .push(store.read_asset(id, owner).await?.into());
    }
    match task {
        Task::Image => Ok(vec![Artifact {
            role: "image",
            bytes: engine.generate(&pass).await?,
        }]),
        Task::CharacterSheet(sheet, model_adapter::SheetStrategy::QwenTwoPass) => {
            // Validate/read the guide before spending GPU time on the portrait.
            let sizes = sheet.layout.pass_sizes().map_err(anyhow::Error::msg)?;
            let guide = if let Some(id) = &request.inputs.pose_guide {
                store.read_asset(id, owner).await?
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
