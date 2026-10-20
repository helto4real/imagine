//! Stateful KSAMPLER.sample. Required model, tensor and callable protocols retain
//! raw Options rather than normalizing mutable fields into a second profile.
use super::factory::SamplerState;
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use std::{cell::Cell, rc::Rc};
pub type Callback = Rc<dyn Fn(&Options) -> Result<Options, Error>>;
pub trait Bindings {
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error>;
    fn attribute_or(
        &self,
        object: &Options,
        name: &str,
        default: Options,
    ) -> Result<Options, Error>;
    fn get(&self, object: &Options, key: &str, default: Options) -> Result<Options, Error>;
    fn item(&self, object: &Options, key: &str) -> Result<Options, Error>;
    fn set_item(&self, object: &Options, key: &str, value: Options) -> Result<(), Error>;
    fn set_attribute(&self, object: &Options, name: &str, value: Options) -> Result<(), Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn index(&self, object: &Options, index: i64) -> Result<Options, Error>;
    fn length(&self, object: &Options) -> Result<i64, Error>;
    fn tuple(&self, object: &Options) -> Result<Options, Error>;
    fn add_one(&self, value: &Options) -> Result<Options, Error>;
    fn manual_seed(&self, value: &Options) -> Result<Options, Error>;
    fn randn_cpu(&self, shape: &Options, generator: &Options) -> Result<Options, Error>;
    fn inpaint(&self, model: &Options, sigmas: &Options) -> Result<Options, Error>;
    fn maximum_denoise(
        &self,
        state: &SamplerState,
        model: &Options,
        sigmas: &Options,
    ) -> Result<Options, Error>;
    fn call(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error>;
    fn call_with_mapping(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
        mapping: &Options,
    ) -> Result<Options, Error>;
    /// Produces an owned native callable with the source k_callback signature.
    /// Foreign samplers may retain it and invoke it after sample returns.
    fn callback(&self, function: Callback) -> Result<Options, Error>;
    fn detail(&self, args: &[Options]) -> Result<Options, Error>;
}
pub struct Arguments {
    pub model: Options,
    pub sigmas: Options,
    pub extra: Options,
    pub callback: Options,
    pub noise: Options,
    pub latent: Options,
    pub mask: Options,
    pub disable: Options,
}
fn integer(value: i64) -> Options {
    Options::Atom(V::Int(value))
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn chain(object: &Options, names: &[&str], bindings: &dyn Bindings) -> Result<Options, Error> {
    let mut object = object.clone();
    for name in names {
        let value = bindings.attribute(&object, name);
        drop(object);
        object = value?;
    }
    Ok(object)
}
fn callback(
    state: Rc<SamplerState>,
    model: Options,
    extra: Options,
    external: Options,
    total: i64,
    bindings: Rc<dyn Bindings>,
) -> Callback {
    let first = Cell::new(true);
    Rc::new(move |value| {
        if first.get() {
            let mut args = vec![
                Options::Atom(V::Text("First sampler step: model=%s sampler=%s step=%s total_steps=%s cfg=%s seed=%s sigma=%s sigma_hat=%s latent_shape=%s denoised_shape=%s".into())),
                chain(&model,&["model_patcher","model","__class__","__name__"],bindings.as_ref())?,
                bindings.attribute_or(&state.fields.get("sampler_function")?,"__name__",Options::Atom(V::Text("unknown".into())))?,
                bindings.item(value,"i")?,integer(total),bindings.attribute(&model,"cfg")?,
                bindings.get(&extra,"seed",none())?,bindings.get(value,"sigma",none())?,bindings.get(value,"sigma_hat",none())?,
            ];
            let shape = bindings.attribute(&bindings.item(value, "x")?, "shape")?;
            let tuple = bindings.tuple(&shape);
            drop(shape);
            args.push(tuple?);
            let shape = bindings.attribute(&bindings.item(value, "denoised")?, "shape")?;
            let tuple = bindings.tuple(&shape);
            drop(shape);
            args.push(tuple?);
            let result = bindings.detail(&args);
            drop(args);
            drop(result?);
            first.set(false);
        }
        if !matches!(external, Options::Atom(V::None)) {
            drop(bindings.call(
                &external,
                &[
                    bindings.item(value, "i")?,
                    bindings.item(value, "denoised")?,
                    bindings.item(value, "x")?,
                    integer(total),
                ],
                &[],
            )?);
        }
        Ok(none())
    })
}
pub fn sample(
    state: Rc<SamplerState>,
    mut args: Arguments,
    bindings: Rc<dyn Bindings>,
) -> Result<Options, Error> {
    bindings.set_item(&args.extra, "denoise_mask", args.mask.clone())?;
    let model_k = bindings.inpaint(&args.model, &args.sigmas)?;
    bindings.set_attribute(&model_k, "latent_image", args.latent.clone())?;
    let mut generator = None;
    let random = bindings.get(
        &state.fields.get("inpaint_options")?,
        "random",
        Options::Atom(V::Bool(false)),
    )?;
    let is_random = bindings.truth(&random);
    drop(random);
    if is_random? {
        let seed = bindings.get(&args.extra, "seed", integer(41))?;
        let incremented = bindings.add_one(&seed);
        drop(seed);
        let seed = incremented?;
        let value = bindings.manual_seed(&seed);
        drop(seed);
        generator = Some(value?);
        let shape = bindings.attribute(&args.noise, "shape")?;
        let noise = bindings.randn_cpu(&shape, generator.as_ref().unwrap());
        drop(shape);
        let noise = noise?;
        let method = bindings.attribute(&noise, "to")?;
        let dtype = bindings.attribute(&args.noise, "dtype")?;
        let cast = bindings.call(&method, &[dtype], &[]);
        drop(method);
        drop(noise);
        let cast = cast?;
        let method = bindings.attribute(&cast, "to")?;
        let device = bindings.attribute(&args.noise, "device")?;
        let moved = bindings.call(&method, &[device], &[]);
        drop(method);
        drop(cast);
        bindings.set_attribute(&model_k, "noise", moved?)?;
    } else {
        bindings.set_attribute(&model_k, "noise", args.noise.clone())?;
    }
    let sampling = chain(
        &args.model,
        &["inner_model", "model_sampling"],
        bindings.as_ref(),
    )?;
    let function = bindings.attribute(&sampling, "noise_scaling");
    drop(sampling);
    let function = function?;
    let first = bindings.index(&args.sigmas, 0)?;
    let maximum = bindings.maximum_denoise(&state, &args.model, &args.sigmas)?;
    let noise = bindings.call(
        &function,
        &[first, args.noise.clone(), args.latent.clone(), maximum],
        &[],
    );
    drop(function);
    drop(std::mem::replace(&mut args.noise, noise?));
    let total = bindings.length(&args.sigmas)? - 1;
    let function = callback(
        state.clone(),
        args.model.clone(),
        args.extra.clone(),
        args.callback.clone(),
        total,
        bindings.clone(),
    );
    let callback = bindings.callback(function)?;
    let function = state.fields.get("sampler_function")?;
    let options = state.fields.get("extra_options")?;
    let samples = bindings.call_with_mapping(
        &function,
        &[model_k.clone(), args.noise.clone(), args.sigmas.clone()],
        &[
            ("extra_args".into(), args.extra.clone()),
            ("callback".into(), callback.clone()),
            ("disable".into(), args.disable.clone()),
        ],
        &options,
    );
    drop(options);
    drop(function);
    let samples = samples?;
    let sampling = chain(
        &args.model,
        &["inner_model", "model_sampling"],
        bindings.as_ref(),
    )?;
    let function = bindings.attribute(&sampling, "inverse_noise_scaling");
    drop(sampling);
    let function = function?;
    let last = bindings.index(&args.sigmas, -1)?;
    let result = bindings.call(&function, &[last, samples], &[]);
    drop(function);
    drop(model_k);
    drop(generator);
    drop(callback);
    result
}
