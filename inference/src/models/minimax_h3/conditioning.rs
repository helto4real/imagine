//! Original contiguous H3 extra_conds payload assembly, with no dtype casting.
//! Learned projection, mask pooling and PackedLayout are separate caller steps.
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};
fn optional(kwargs: &Options, key: &str) -> Result<Option<Options>, Error> {
    Ok(kwargs
        .get(key)?
        .filter(|v| !matches!(v, Options::Atom(V::None))))
}
fn sequence(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok(v.as_ref().clone()),
        _ => Err(Error::BindingGap {
            backend: "minimax_h3".into(),
            operation: "typed H3 references require their original list/tuple owner".into(),
        }),
    }
}
fn list(v: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(v)))
}
pub fn audio_scale(shape_count: Option<usize>, sampling_audio_scale: f64) -> f64 {
    if shape_count.is_none_or(|n| n < 2) {
        1.0
    } else {
        sampling_audio_scale
    }
}
pub fn payload(kwargs: &Options, audio_scale: f64) -> Result<Options, Error> {
    let output = Options::dictionary(vec![]);
    if let Some(tags) = optional(kwargs, "minimax_token_tags")? {
        output.set("text_token_tags", tags)?;
    }
    if let Some(keyframes) = optional(kwargs, "minimax_keyframes")? {
        output.set("keyframes", keyframes.clone())?;
        let mut video = vec![];
        let mut audio = vec![];
        for frame in sequence(&keyframes)? {
            if let Some(v) = optional(&frame, "latent")? {
                video.push(v);
            }
        }
        for frame in sequence(&keyframes)? {
            if let Some(v) = optional(&frame, "audio_latent")? {
                audio.push(v);
            }
        }
        output.set("cond_video_latents", list(video))?;
        output.set("cond_audio_latents", list(audio))?;
    }
    if let Some(refs) = optional(kwargs, "minimax_refs")? {
        output.set("refs", refs.clone())?;
        let mut video = output
            .get("cond_video_latents")?
            .map(|v| sequence(&v))
            .transpose()?
            .unwrap_or_default();
        for reference in sequence(&refs)? {
            if let Some(v) = reference.get("latent")? {
                video.push(v);
            }
        }
        output.set("cond_video_latents", list(video))?;
        let mut audio = output
            .get("cond_audio_latents")?
            .map(|v| sequence(&v))
            .transpose()?
            .unwrap_or_default();
        for reference in sequence(&refs)? {
            if let Some(v) = optional(&reference, "audio_latent")? {
                audio.push(v);
            }
        }
        output.set("cond_audio_latents", list(audio))?;
    }
    for (source, target) in [
        ("minimax_visual_cond_noise_aug", "visual_cond_noise_aug"),
        ("minimax_audio_cond_noise_aug", "audio_cond_noise_aug"),
    ] {
        if let Some(v) = optional(kwargs, source)? {
            output.set(target, v)?;
        }
    }
    output.set(
        "seed",
        kwargs.get("seed")?.unwrap_or(Options::Atom(V::Int(0))),
    )?;
    output.set("audio_scale", Options::Atom(V::Float(audio_scale)))?;
    Ok(output)
}
