//! Frozen Qwen-VL image patching, multimodal positions and Qwen21 output filtering.
//! This is not yet the vision transformer or the outer Lanczos editing owner.
use super::{PromptToken, TextConditioning};
use crate::native::Error;
use tch::{Device, Kind, Tensor};

pub struct VisionPreprocess {
    pub min_pixels: i64,
    pub max_pixels: i64,
    pub patch_size: i64,
    pub temporal_patch_size: i64,
    pub merge_size: i64,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub bicubic: bool,
}
impl Default for VisionPreprocess {
    fn default() -> Self {
        Self {
            min_pixels: 3136,
            max_pixels: 12845056,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            mean: [0.5; 3],
            std: [0.5; 3],
            bicubic: false,
        }
    }
}
impl VisionPreprocess {
    /// Qwen3VL.preprocess_embed calls process_qwen2vl_images with these defaults.
    /// Preserve its first-image-only behavior even if a larger batch is supplied.
    pub fn patches(&self, images: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let images = images.f_slice(-1, 0, 3, 1)?;
        let shape = images.size();
        let (height, width) = (shape[1], shape[2]);
        let factor = self.patch_size * self.merge_size;
        let mut h = ((height as f64 / factor as f64).round_ties_even() as i64) * factor;
        let mut w = ((width as f64 / factor as f64).round_ties_even() as i64) * factor;
        if h * w > self.max_pixels {
            let beta = ((height * width) as f64 / self.max_pixels as f64).sqrt();
            h = factor.max((height as f64 / beta / factor as f64).floor() as i64 * factor);
            w = factor.max((width as f64 / beta / factor as f64).floor() as i64 * factor);
        } else if h * w < self.min_pixels {
            let beta = (self.min_pixels as f64 / (height * width) as f64).sqrt();
            h = (height as f64 * beta / factor as f64).ceil() as i64 * factor;
            w = (width as f64 * beta / factor as f64).ceil() as i64 * factor;
        }
        let image = images.f_permute([0, 3, 1, 2])?.f_get(0)?.f_unsqueeze(0)?;
        let image = if self.bicubic {
            image.f_upsample_bicubic2d([h, w], false, None, None)?
        } else {
            image.f_upsample_bilinear2d([h, w], false, None, None)?
        }
        .f_squeeze_dim(0)?;
        let normalized = image.copy();
        for c in 0..3 {
            normalized.f_get(c)?.f_copy_(
                &image
                    .f_get(c)?
                    .f_sub_scalar(self.mean[c as usize])?
                    .f_div_scalar(self.std[c as usize])?,
            )?;
        }
        let (gh, gw) = (h / self.patch_size, w / self.patch_size);
        let grid = Tensor::f_from_slice(&[1_i64, gh, gw])?
            .f_to_device(images.device())?
            .f_unsqueeze(0)?;
        let patches = normalized
            .f_unsqueeze(0)?
            .f_repeat([2, 1, 1, 1])?
            .f_reshape([
                1,
                self.temporal_patch_size,
                3,
                gh / self.merge_size,
                self.merge_size,
                self.patch_size,
                gw / self.merge_size,
                self.merge_size,
                self.patch_size,
            ])?
            .f_permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?
            .f_reshape([
                gh * gw,
                3 * self.temporal_patch_size * self.patch_size * self.patch_size,
            ])?;
        Ok((patches, grid))
    }
}

pub struct ImageEmbeddingInfo {
    pub index: i64,
    pub size: i64,
    pub grid: Vec<[i64; 3]>,
    pub deepstack: Vec<Tensor>,
}
/// Original qwen2vl_mrope_position_ids uses incoming order, not sorted order.
pub fn image_position_ids(
    info: &[ImageEmbeddingInfo],
    seq: i64,
    device: Device,
) -> Result<Option<Tensor>, Error> {
    let mut positions: Option<Tensor> = None;
    let mut offset = 0;
    for e in info {
        let start = e.index;
        let positions = if let Some(p) = &positions {
            p
        } else {
            positions = Some(Tensor::f_zeros([3, seq], (Kind::Float, device))?);
            let p = positions.as_ref().unwrap();
            p.f_slice(1, 0, start, 1)?
                .f_copy_(&Tensor::f_arange(start, (Kind::Int64, device))?)?;
            p
        };
        let end = start + e.size;
        let len_max = e.grid.iter().flatten().copied().max().unwrap() / 2;
        let next = len_max + start;
        positions
            .f_slice(1, end, i64::MAX, 1)?
            .f_copy_(&Tensor::f_arange_start(
                next + offset,
                next + seq - end + offset,
                (Kind::Int64, device),
            )?)?;
        let _ = positions
            .f_get(0)?
            .f_slice(0, start, end, 1)?
            .f_fill_(start + offset)?;
        let h = e.grid[0][1] / 2;
        let w = e.grid[0][2] / 2;
        let rows =
            Tensor::f_arange_start(start + offset, start + h + offset, (Kind::Int64, device))?
                .f_unsqueeze(1)?
                .f_repeat([1, ((e.size + h - 1) / h)])?
                .f_flatten(0, -1)?
                .f_slice(0, 0, e.size, 1)?;
        let cols =
            Tensor::f_arange_start(start + offset, start + w + offset, (Kind::Int64, device))?
                .f_unsqueeze(0)?
                .f_repeat([((e.size + w - 1) / w), 1])?
                .f_flatten(0, -1)?
                .f_slice(0, 0, e.size, 1)?;
        positions
            .f_get(1)?
            .f_slice(0, start, end, 1)?
            .f_copy_(&rows)?;
        positions
            .f_get(2)?
            .f_slice(0, start, end, 1)?
            .f_copy_(&cols)?;
        offset += len_max - e.size;
    }
    Ok(positions)
}

pub struct VisionLanguageInputs {
    pub position_ids: Option<Tensor>,
    pub visual_mask: Option<Tensor>,
    pub deepstack: Option<Vec<Tensor>>,
}
/// Original Qwen3VL.build_image_inputs sorts only masks/DeepStack by token index.
pub fn build_image_inputs(
    info: &[ImageEmbeddingInfo],
    seq: i64,
    device: Device,
) -> Result<VisionLanguageInputs, Error> {
    let mut images = info.iter().collect::<Vec<_>>();
    images.sort_by_key(|e| e.index);
    if images.is_empty() {
        return Ok(VisionLanguageInputs {
            position_ids: None,
            visual_mask: None,
            deepstack: None,
        });
    }
    let visual = Tensor::f_zeros([1, seq], (Kind::Bool, device))?;
    let mut deepstack: Option<Vec<Tensor>> = None;
    for e in images {
        let _ = visual
            .f_slice(1, e.index, e.index + e.size, 1)?
            .f_fill_(1)?;
        deepstack = Some(if let Some(prior) = deepstack {
            prior
                .into_iter()
                .enumerate()
                .map(|(i, a)| {
                    let b = e
                        .deepstack
                        .get(i)
                        .ok_or_else(|| Error::ReferenceException {
                            class: "IndexError",
                            message: "list index out of range".into(),
                        })?;
                    Ok(Tensor::f_cat(&[a, b.shallow_clone()], 0)?)
                })
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            e.deepstack.iter().map(Tensor::shallow_clone).collect()
        });
    }
    Ok(VisionLanguageInputs {
        position_ids: image_position_ids(info, seq, device)?,
        visual_mask: Some(visual),
        deepstack,
    })
}

/// QwenImage21TEModel.encode_token_weights after the shared encoder returns.
/// Image spans are expanded token coordinates; the DiT receives their remaining
/// text insertion positions only when keep_vision is false.
pub fn filter_image_conditioning(
    context: &Tensor,
    attention_mask: &Tensor,
    tokens: &[PromptToken],
    spans: &[(i64, i64)],
    keep_vision: bool,
) -> Result<TextConditioning, Error> {
    let mut starts = vec![];
    let mut offset = 0;
    let mut spans_iter = spans.iter();
    for (i, t) in tokens.iter().enumerate() {
        match t {
            PromptToken::Id(151644) => starts.push(i as i64 + offset),
            PromptToken::Image(_) => {
                offset += spans_iter.next().map(|(_, size)| *size).unwrap_or(1) - 1
            }
            _ => (),
        }
    }
    let keep = Tensor::f_ones([context.size()[1]], (Kind::Bool, Device::Cpu))?;
    let _ = keep
        .f_slice(0, 0, *starts.get(1).unwrap_or(&0), 1)?
        .f_fill_(0)?;
    let mut slots = vec![];
    if !keep_vision {
        for &(start, size) in spans {
            let _ = keep.f_slice(0, start, start + size, 1)?.f_fill_(0)?;
            slots.push(
                keep.f_slice(0, 0, start, 1)?
                    .f_sum(Kind::Int64)?
                    .int64_value(&[]),
            );
        }
    }
    let context = context.f_index(&[None, Some(&keep.f_to_device(context.device())?)])?;
    let mask =
        attention_mask.f_index(&[None, Some(&keep.f_to_device(attention_mask.device())?)])?;
    let mask = if mask
        .f_sum(Kind::Int64)?
        .f_eq(mask.numel() as i64)?
        .f_is_nonzero()?
    {
        None
    } else {
        Some(mask)
    };
    Ok(TextConditioning {
        context,
        pooled: None,
        attention_mask: mask,
        image_slots: slots,
    })
}
