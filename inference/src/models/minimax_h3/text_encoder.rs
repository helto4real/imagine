//! H3's selected Qwen3-VL-32B language/vision configuration. Parameter inventories
//! and full H3 presentation binding remain separate from the shared forwards.
use crate::{
    checkpoint::PreparedParameters,
    models::qwen_image21::{QwenTextEncoder, TextEncoderConfig, VisionEncoderConfig},
    native::Error,
};

pub fn language_config() -> TextEncoderConfig {
    TextEncoderConfig {
        hidden_size: 5120,
        intermediate_size: 25600,
        layers: 50,
        heads: 64,
        ..TextEncoderConfig::default()
    }
}

pub fn vision_config() -> VisionEncoderConfig {
    VisionEncoderConfig {
        out_hidden_size: 5120,
        ..VisionEncoderConfig::default()
    }
}

pub fn language_from_parameters(state: &PreparedParameters) -> Result<QwenTextEncoder, Error> {
    QwenTextEncoder::from_parameters_with_final_norm(language_config(), state, false)
}

use super::{
    preprocessing,
    tokenizer::{Entry, Token},
};
use crate::{
    execution::Runtime,
    models::qwen_image21::{self as shared, QwenVisionEncoder, TextEncoderExecution},
};
use tch::{Device, Kind, Tensor};

pub struct Conditioning {
    pub context: Tensor,
    pub token_tags: Tensor,
}

pub struct TextEncoder {
    language: QwenTextEncoder,
    vision: Option<QwenVisionEncoder>,
}
impl TextEncoder {
    pub fn from_parameters(state: &PreparedParameters, vision: bool) -> Result<Self, Error> {
        Self::from_components(
            language_from_parameters(state)?,
            if vision {
                Some(QwenVisionEncoder::from_parameters(vision_config(), state)?)
            } else {
                None
            },
        )
    }
    pub fn from_components(
        language: QwenTextEncoder,
        vision: Option<QwenVisionEncoder>,
    ) -> Result<Self, Error> {
        Ok(Self { language, vision })
    }
    /// Raw H3 presentation from its tokenizer: exactly one section with unit
    /// weights. Vision blocks retain their request order and explicit frame-pair
    /// marker. This has no Qwen Image chat/filter or attention-mask policy.
    /// # Safety
    /// Active language/vision parameters and original kernels satisfy their
    /// selected device, layout and allocation contracts.
    pub unsafe fn encode<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        entries: &[Entry],
        device: Device,
        output_device: Device,
        exec: &mut E,
    ) -> Result<Conditioning, Error> {
        if entries.is_empty() || entries.iter().any(|e| e.weight != 1.) {
            return Err(Error::BindingGap {
                backend: "minimax_h3".into(),
                operation: "H3 tokenizer requires one nonempty unit-weight presentation".into(),
            });
        }
        let ids = entries
            .iter()
            .filter_map(|e| match e.token {
                Token::Id(id) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let ids = Tensor::f_from_slice(&ids)?
            .f_unsqueeze(0)?
            .f_to_device(device)?;
        let mut embeds = unsafe { self.language.embed(rt, &ids, exec)? };
        let mut offset = 0;
        let mut pad_extra = 0;
        let mut info = vec![];
        for (index, entry) in entries.iter().enumerate() {
            let Token::Vision { data, video_block } = &entry.token else {
                continue;
            };
            let vision = self.vision.as_ref().ok_or_else(|| Error::BindingGap {
                backend: "minimax_h3".into(),
                operation: "Vision owner required for image/video tokens".into(),
            })?;
            let image = data.as_tensor()?;
            let (patches, grid) = if *video_block {
                preprocessing::video_block(&image, &rt.native)?
            } else {
                shared::VisionPreprocess::default().patches(&image)?
            };
            let rows = grid
                .f_to_device(Device::Cpu)?
                .f_to_kind(Kind::Int64)?
                .f_reshape([-1])?;
            let mut grid = vec![0_i64; rows.numel()];
            let len = grid.len();
            rows.f_copy_data(&mut grid, len)?;
            let grid = grid
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>();
            let patches = patches.f_to_device_(device, Kind::Float, false, false)?;
            let (merged, deepstack) = unsafe { vision.forward(rt, &patches, &grid, exec)? };
            let width = merged.size()[merged.dim() - 1];
            let emb =
                merged
                    .f_view([1, -1, width])?
                    .f_to_device_(device, Kind::Float, false, false)?;
            let size = emb.size()[1];
            if width == embeds.size()[2] {
                let start = index as i64 + offset;
                embeds = Tensor::f_cat(
                    &[
                        embeds.f_slice(1, 0, start, 1)?,
                        emb,
                        embeds.f_slice(1, start, i64::MAX, 1)?,
                    ],
                    1,
                )?;
                offset += size - 1;
                info.push(shared::ImageEmbeddingInfo {
                    index: start,
                    size,
                    grid,
                    deepstack,
                });
            } else {
                offset -= 1;
                pad_extra += size;
            }
        }
        if pad_extra > 0 {
            let pad = Tensor::f_full([1, pad_extra], 151643, (Kind::Int64, device))?;
            embeds = Tensor::f_cat(
                &[embeds, unsafe { self.language.embed(rt, &pad, exec)? }],
                1,
            )?;
        }
        let image_inputs = shared::build_image_inputs(&info, embeds.size()[1], device)?;
        let tags = preprocessing::token_tags(
            embeds.size()[1],
            &info.iter().map(|e| (e.index, e.size)).collect::<Vec<_>>(),
        )?;
        let result = unsafe {
            self.language.forward(
                rt,
                &embeds,
                shared::TextForwardOptions {
                    position_ids: image_inputs.position_ids.as_ref(),
                    intermediate: shared::Intermediate::None,
                    final_layer_norm_intermediate: false,
                    deepstack: image_inputs.deepstack.as_deref().unwrap_or(&[]),
                    visual_mask: image_inputs.visual_mask.as_ref(),
                    attention_mask: None,
                },
                exec,
            )?
        };
        // Original ClipTokenWeightEncoder concatenates even the single section.
        let context = Tensor::f_cat(&[result.last.f_to_kind(Kind::Float)?], -2)?
            .f_to_device(output_device)?;
        Ok(Conditioning {
            context,
            token_tags: tags,
        })
    }
}
