//! Original auxiliary autoregressive sampler used by SamplerARVideo. It is
//! intentionally separate from the source's 45 ordinary sampler registrations.
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use tch::{Device, Tensor};

/// Original model/object operations. Concrete Causal-WAN model implementations
/// provide these operations; the shared sampler never substitutes a denoiser.
pub trait Bindings {
    fn get(&mut self, object: &Options, key: &str, default: Options) -> Result<Options, Error>;
    fn set(&mut self, object: &Options, key: &str, value: Options) -> Result<(), Error>;
    fn pop(&mut self, object: &Options, key: &str) -> Result<(), Error>;
    fn attribute(&mut self, object: &Options, name: &str) -> Result<Options, Error>;
    fn has_attribute(&mut self, object: &Options, name: &str) -> Result<bool, Error>;
    fn tensor_device(&mut self, tensor: &Tensor) -> Result<Options, Error>;
    fn method(&mut self, object: &Options, name: &str, args: &[Options]) -> Result<Options, Error>;
    fn cast_latent(
        &mut self,
        value: &Options,
        device: Device,
        dtype: &Options,
    ) -> Result<Tensor, Error>;
    fn model(
        &mut self,
        model: &Options,
        x: &Tensor,
        sigma: &Tensor,
        args: &Options,
    ) -> Result<Tensor, Error>;
    fn callback(&mut self, callback: &Options, fields: Options) -> Result<Options, Error>;
    fn reseed(&mut self, seed: &Options, block: i64, step: i64) -> Result<(), Error>;
    fn cache_end(&mut self, cache: &Options) -> Result<Options, Error>;
    fn inplace_subtract(&mut self, value: &Options, amount: i64) -> Result<Options, Error>;
    fn set_cache_end(&mut self, cache: &Options, value: Options) -> Result<(), Error>;
}
pub struct Inputs<'a> {
    pub model: &'a Options,
    pub x: &'a Tensor,
    pub sigmas: &'a Tensor,
    pub extra_args: Options,
    pub callback: Option<&'a Options>,
    pub frames_per_block: i64,
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn dict() -> Options {
    Options::dictionary(vec![])
}
fn integer(value: i64) -> Options {
    Options::Atom(V::Int(value))
}
fn tensor(value: &Tensor) -> Options {
    Options::Atom(V::tensor(value.shallow_clone()))
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn ceil_div(a: i64, b: i64) -> Result<i64, Error> {
    if b == 0 {
        return Err(error(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let (a, b) = (a as i128, b as i128);
    let result = a / b + i128::from(a % b != 0 && (a % b > 0) == (b > 0));
    i64::try_from(result).map_err(|_| Error::BindingGap {
        backend: "ar_video".into(),
        operation: "arbitrary precision block count".into(),
    })
}
fn checked_product(a: i64, b: i64) -> Result<i64, Error> {
    a.checked_mul(b).ok_or_else(|| Error::BindingGap {
        backend: "ar_video".into(),
        operation: "arbitrary precision cache sequence arithmetic".into(),
    })
}
fn state(frame: i64, kv: &Options, cross: &Options) -> Options {
    Options::dictionary(vec![
        ("start_frame".into(), integer(frame)),
        ("kv_caches".into(), kv.clone()),
        ("crossattn_caches".into(), cross.clone()),
    ])
}
fn decrement_caches(
    caches: &Options,
    amount: i64,
    last: &mut Option<Options>,
    bindings: &mut dyn Bindings,
) -> Result<(), Error> {
    let mut iterator = crate::patching::options_iterator::SequenceIterator::for_operation(
        caches,
        "autoregressive KV caches",
    )?;
    while let Some(cache) = iterator.next()? {
        drop(last.replace(cache));
        let cache = last.as_ref().unwrap();
        let end = bindings.cache_end(cache)?;
        let value = bindings.inplace_subtract(&end, amount)?;
        bindings.set_cache_end(cache, value)?;
    }
    Ok(())
}
pub fn sample(
    native: &crate::native::Kitchen,
    input: Inputs<'_>,
    bindings: &mut dyn Bindings,
) -> Result<Tensor, Error> {
    let _grad = tch::no_grad_guard();
    let Inputs {
        model,
        x,
        sigmas,
        extra_args,
        callback,
        frames_per_block,
    } = input;
    let extra_args = if matches!(extra_args, Options::Atom(V::None)) {
        dict()
    } else {
        extra_args
    };
    let model_options = bindings.get(&extra_args, "model_options", dict())?;
    let transformer_options = bindings.get(&model_options, "transformer_options", dict())?;
    if x.dim() != 5 {
        let shape = x
            .size()
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(error(
            "ValueError",
            format!(
                "ar_video sampler requires 5-D video latents [B,C,T,H,W], got {}-D tensor with shape torch.Size([{}]). This sampler is only compatible with autoregressive video models (e.g. Causal-WAN).",
                x.dim(),
                shape
            ),
        ));
    }
    let outer = bindings.attribute(model, "inner_model")?;
    let inner = bindings.attribute(&outer, "inner_model");
    drop(outer);
    let inner = inner?;
    let causal = bindings.attribute(&inner, "diffusion_model")?;
    if !(bindings.has_attribute(&causal, "init_kv_caches")?
        && bindings.has_attribute(&causal, "init_crossattn_caches")?)
    {
        return Err(error(
            "TypeError",
            "ar_video sampler requires a Causal-WAN compatible model whose diffusion_model exposes init_kv_caches() and init_crossattn_caches(). The loaded checkpoint does not support this interface — choose a different sampler.",
        ));
    }
    let seed = bindings.get(&extra_args, "seed", integer(0))?;
    let shape = x.size();
    let (bs, lat_t, lat_h, lat_w) = (shape[0], shape[2], shape[3], shape[4]);
    let frame_sequence = checked_product(ceil_div(lat_h, 2)?, ceil_div(lat_w, 2)?)?;
    let mut blocks = ceil_div(lat_t, frames_per_block)?;
    let device = x.device();
    let source_device = bindings.tensor_device(x)?;
    let dtype = bindings.method(&inner, "get_dtype", &[])?;
    let kv = bindings.method(
        &causal,
        "init_kv_caches",
        &[
            integer(bs),
            integer(checked_product(lat_t, frame_sequence)?),
            source_device.clone(),
            dtype.clone(),
        ],
    )?;
    let cross = bindings.method(
        &causal,
        "init_crossattn_caches",
        &[integer(bs), source_device.clone(), dtype.clone()],
    )?;
    let mut output = x.f_zeros_like()?;
    let s_in = x.f_new_ones([shape[0]], (x.kind(), device))?;
    let mut current = 0;
    // Named tensor locals survive later loop assignments exactly as in source.
    let mut locals = super::locals::SourceLocals::default();
    let config = bindings.get(&transformer_options, "ar_config", dict())?;
    let initial = bindings.get(&config, "initial_latent", none());
    drop(config);
    let initial = initial?;
    if !matches!(initial, Options::Atom(V::None)) {
        let processed =
            bindings.method(&inner, "process_latent_in", std::slice::from_ref(&initial))?;
        let casted = bindings.cast_latent(&processed, device, &dtype)?;
        drop(processed);
        drop(initial);
        let initial = locals.keep("initial_latent", casted);
        let count = *initial
            .size()
            .get(2)
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
        native.set_video_frames(&mut output, 0, count, &initial)?;
        bindings.set(&transformer_options, "ar_state", state(0, &kv, &cross))?;
        let zero = locals.keep(
            "zero_sigma",
            sigmas.f_new_zeros([1], (sigmas.kind(), sigmas.device()))?,
        );
        let _ = locals.keep(
            "_",
            bindings.model(model, &initial, &zero.f_mul(&s_in)?, &extra_args)?,
        );
        current = count;
        blocks = ceil_div(lat_t - count, frames_per_block)?;
    }
    let steps = sigmas
        .size()
        .first()
        .copied()
        .ok_or_else(|| error("TypeError", "len() of a 0-d tensor"))?
        - 1;
    let total = checked_product(blocks, steps)?;
    let mut step_count = 0;
    let mut last_cache = None;
    let result = (|| {
        for block in 0..blocks {
            let frames = frames_per_block.min(lat_t - current);
            let end = current + frames;
            let mut noisy = locals.keep("noisy_input", x.f_slice(2, current, end, 1)?);
            bindings.set(
                &transformer_options,
                "ar_state",
                state(current, &kv, &cross),
            )?;
            for i in 0..steps {
                let denoised = locals.keep(
                    "denoised",
                    bindings.model(model, &noisy, &sigmas.f_get(i)?.f_mul(&s_in)?, &extra_args)?,
                );
                if let Some(callback) = callback {
                    if total == 0 {
                        return Err(error(
                            "ZeroDivisionError",
                            "integer division or modulo by zero",
                        ));
                    }
                    let index = checked_product(step_count, steps)?.div_euclid(total);
                    drop(bindings.callback(
                        callback,
                        Options::dictionary(vec![
                            ("x".into(), tensor(&noisy)),
                            ("i".into(), integer(index)),
                            ("sigma".into(), tensor(&sigmas.f_get(i)?)),
                            ("sigma_hat".into(), tensor(&sigmas.f_get(i)?)),
                            ("denoised".into(), tensor(&denoised)),
                        ]),
                    )?);
                }
                if sigmas.f_get(i + 1)?.f_eq(0.)?.f_is_nonzero()? {
                    noisy = locals.keep("noisy_input", denoised);
                } else {
                    let next = locals.keep("sigma_next", sigmas.f_get(i + 1)?);
                    bindings.reseed(&seed, block, i)?;
                    let fresh = locals.keep("fresh_noise", denoised.f_randn_like()?);
                    noisy = locals.keep(
                        "noisy_input",
                        next.f_rsub_scalar(1.)?
                            .f_mul(&denoised)?
                            .f_add(&next.f_mul(&fresh)?)?,
                    );
                    decrement_caches(
                        &kv,
                        checked_product(frames, frame_sequence)?,
                        &mut last_cache,
                        bindings,
                    )?;
                }
                step_count += 1;
            }
            native.set_video_frames(&mut output, current, end, &noisy)?;
            decrement_caches(
                &kv,
                checked_product(frames, frame_sequence)?,
                &mut last_cache,
                bindings,
            )?;
            let zero = locals.keep(
                "zero_sigma",
                sigmas.f_new_zeros([1], (sigmas.kind(), sigmas.device()))?,
            );
            let _ = locals.keep(
                "_",
                bindings.model(model, &noisy, &zero.f_mul(&s_in)?, &extra_args)?,
            );
            current += frames;
        }
        Ok(output)
    })();
    crate::source_exception::finally(result, || {
        bindings.pop(&transformer_options, "ar_state").map(|_| ())
    })
}
