//! MOSS-Image color/depth control (ComfyUI 426f5eea): `MOSSImageControl` builds the
//! control latent, `conditioning_set_values_with_timestep_range` limits it to a part
//! of sampling and `MOSSImage.concat_cond` turns it into the widened embedder's extra
//! input channels; zeros mean no control.
use super::MossSampling;
use crate::native::Error;
use tch::{Device, Kind, Tensor};

/// ComfyUI `common_upscale` with bilinear interpolation and a center crop, 4-d input.
pub fn upscale_bilinear_center(samples: &Tensor, width: i64, height: i64) -> Result<Tensor, Error> {
    let size = samples.size();
    let (old_height, old_width) = (size[size.len() - 2], size[size.len() - 1]);
    let old_aspect = old_width as f64 / old_height as f64;
    let new_aspect = width as f64 / height as f64;
    let (mut x, mut y) = (0, 0);
    // Python's round() rounds half to even.
    if old_aspect > new_aspect {
        x = ((old_width as f64 - old_width as f64 * (new_aspect / old_aspect)) / 2.)
            .round_ties_even() as i64;
    } else if old_aspect < new_aspect {
        y = ((old_height as f64 - old_height as f64 * (old_aspect / new_aspect)) / 2.)
            .round_ties_even() as i64;
    }
    Ok(samples
        .f_narrow(-2, y, old_height - y * 2)?
        .f_narrow(-1, x, old_width - x * 2)?
        .f_upsample_bilinear2d([height, width], false, None, None)?)
}
/// ComfyUI `resize_to_batch_size`: spreads or picks samples to a batch size.
pub fn resize_to_batch_size(tensor: &Tensor, batch: i64) -> Result<Tensor, Error> {
    let size = tensor.size();
    let input = size[0];
    if input == batch {
        return Ok(tensor.shallow_clone());
    }
    if batch <= 1 {
        return Ok(tensor.f_narrow(0, 0, batch.max(0).min(input))?);
    }
    let mut shape = size.clone();
    shape[0] = batch;
    let output = Tensor::f_empty(&shape, (tensor.kind(), tensor.device()))?;
    for i in 0..batch {
        let source = if batch < input {
            let scale = (input - 1) as f64 / (batch - 1) as f64;
            ((i as f64 * scale).round_ties_even() as i64).min(input - 1)
        } else {
            let scale = input as f64 / batch as f64;
            (((i as f64 + 0.5) * scale).floor() as i64).min(input - 1)
        };
        output.f_get(i)?.f_copy_(&tensor.f_get(source)?)?;
    }
    Ok(output)
}
/// `MOSSImageControl`'s control latent from the encoded color and depth maps:
/// [color | depth] on the channel axis, a missing map a zero latent, the depth map
/// resized to the color map's latent, scaled by `strength`. `None` without maps.
pub fn moss_control_latent(
    color: Option<&Tensor>,
    depth: Option<&Tensor>,
    strength: f64,
) -> Result<Option<Tensor>, Error> {
    let latents = [color, depth];
    let Some(reference) = latents.iter().flatten().next() else {
        return Ok(None);
    };
    let batch = latents.iter().flatten().map(|l| l.size()[0]).max().unwrap();
    let rsize = reference.size();
    let (height, width) = (rsize[rsize.len() - 2], rsize[rsize.len() - 1]);
    let mut parts = Vec::with_capacity(2);
    for latent in latents {
        let latent = match latent {
            None => reference.f_zeros_like()?,
            Some(l) if l.size()[l.dim() - 2..] == rsize[rsize.len() - 2..] => l.shallow_clone(),
            Some(l) => upscale_bilinear_center(l, width, height)?,
        };
        parts.push(resize_to_batch_size(&latent, batch)?);
    }
    Ok(Some(Tensor::f_cat(&parts, 1)?.f_mul_scalar(strength)?))
}
/// `MOSSImage.concat_cond`: the extra embedder channels for a noise shape, from the
/// control latent or zeros; `None` when the embedder is not widened.
pub fn moss_concat_cond(
    noise: &[i64],
    embedder_channels: i64,
    image: Option<&Tensor>,
    device: Device,
) -> Result<Option<Tensor>, Error> {
    let channels = embedder_channels - noise[1];
    if channels <= 0 {
        return Ok(None);
    }
    let Some(image) = image else {
        let mut shape = noise.to_vec();
        shape[1] = channels;
        return Ok(Some(Tensor::f_zeros(&shape, (Kind::Float, device))?));
    };
    let image = upscale_bilinear_center(
        &image.f_to_device(device)?,
        noise[noise.len() - 1],
        noise[noise.len() - 2],
    )?;
    // Flux2's process_in is the identity.
    Ok(Some(resize_to_batch_size(&image, noise[0])?))
}
/// One conditioning entry after the control's timestep range split it: whether it
/// carries the control, and the sigma range the sampler gates it by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MossControlSpan {
    pub control: bool,
    pub sigma_start: Option<f64>,
    pub sigma_end: Option<f64>,
}
impl MossControlSpan {
    /// `get_area_and_mult`'s gate, compared in the sigma's float32.
    pub fn active(&self, sigma: f32) -> bool {
        !self.sigma_start.is_some_and(|s| sigma > s as f32)
            && !self.sigma_end.is_some_and(|e| sigma < e as f32)
    }
}
/// `conditioning_set_values_with_timestep_range` on an entry without its own range,
/// then `calculate_start_end_timesteps`.
pub fn moss_control_spans(start: f64, end: f64, sampling: &MossSampling) -> Vec<MossControlSpan> {
    const EPS: f64 = 1e-5;
    let span = |control, from: f64, to: f64| MossControlSpan {
        control,
        sigma_start: Some(sampling.percent_to_sigma(from)),
        sigma_end: Some(sampling.percent_to_sigma(to)),
    };
    // start > end is rejected with a warning; no overlap leaves the entry unchanged.
    let (from, to) = (start.max(0.), end.min(1.));
    if start > end || from >= to {
        return vec![MossControlSpan {
            control: false,
            sigma_start: None,
            sigma_end: None,
        }];
    }
    let mut spans = Vec::with_capacity(3);
    if from > 0. {
        spans.push(span(false, 0., from - EPS));
    }
    spans.push(span(true, from, to));
    if to < 1. {
        spans.push(span(false, to + EPS, 1.));
    }
    spans
}
/// A sampling run's control: each span's `c_concat`, built once from the noise
/// shape as `process_conds` does.
pub struct MossControl {
    pub spans: Vec<(MossControlSpan, Tensor)>,
}
impl MossControl {
    /// `None` when the embedder is not widened; zeros throughout without a latent.
    pub fn new(
        latent: Option<&Tensor>,
        embedder_channels: i64,
        noise: &[i64],
        device: Device,
        (start, end): (f64, f64),
        sampling: &MossSampling,
    ) -> Result<Option<Self>, Error> {
        let spans = match latent {
            Some(_) => moss_control_spans(start, end, sampling),
            None => moss_control_spans(0., 0., sampling),
        };
        let mut out = Vec::with_capacity(spans.len());
        for span in spans {
            let image = if span.control { latent } else { None };
            let Some(concat) = moss_concat_cond(noise, embedder_channels, image, device)? else {
                return Ok(None);
            };
            out.push((span, concat));
        }
        Ok(Some(Self { spans: out }))
    }
    /// The `c_concat` of the one span the sampler admits at `sigma`.
    pub fn concat(&self, sigma: f32) -> Result<&Tensor, Error> {
        let mut active = self.spans.iter().filter(|(s, _)| s.active(sigma));
        match (active.next(), active.next()) {
            (Some((_, concat)), None) => Ok(concat),
            _ => Err(Error::BindingGap {
                backend: "moss_control".into(),
                operation: format!("sigma {sigma} is admitted by no or several control spans"),
            }),
        }
    }
}
