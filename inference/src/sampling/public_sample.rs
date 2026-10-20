//! The module-level comfy.samplers.sample entry point. Device/model_options
//! are source parameters retained by its frame; CFGGuider owns execution.
use crate::{native::Error, patching::options::Options, source_exception::Frame};
use std::rc::Rc;
pub struct Arguments {
    pub model: Options,
    pub noise: Options,
    pub positive: Options,
    pub negative: Options,
    pub cfg: Options,
    pub device: Options,
    pub sampler: Options,
    pub sigmas: Options,
    pub model_options: Options,
    pub latent: Options,
    pub mask: Options,
    pub callback: Options,
    pub disable: Options,
    pub seed: Options,
}
pub trait Bindings {
    fn construct_guider(&self, model: &Options) -> Result<Options, Error>;
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error>;
    fn call(&self, function: &Options, args: &[Options]) -> Result<Options, Error>;
}
pub fn sample(args: Arguments, bindings: &dyn Bindings) -> Result<Options, Error> {
    let mut owned_guider = None;
    let result = (|| {
        owned_guider = Some(bindings.construct_guider(&args.model)?);
        let guider = owned_guider.as_ref().unwrap();
        let method = bindings.attribute(guider, "set_conds")?;
        let result = bindings.call(&method, &[args.positive.clone(), args.negative.clone()]);
        drop(method);
        drop(result?);
        let method = bindings.attribute(guider, "set_cfg")?;
        // Keep the source call's owned argument until the method returns.
        let call_args = [args.cfg.clone()];
        let result = bindings.call(&method, &call_args);
        drop(call_args);
        drop(method);
        drop(result?);
        let method = bindings.attribute(guider, "sample")?;
        let result = bindings.call(
            &method,
            &[
                args.noise.clone(),
                args.latent.clone(),
                args.sampler.clone(),
                args.sigmas.clone(),
                args.mask.clone(),
                args.callback.clone(),
                args.disable.clone(),
                args.seed.clone(),
            ],
        );
        drop(method);
        result
    })();
    result.map_err(|error| {
        if matches!(error.root(), Error::BindingGap { .. }) {
            return error;
        }
        let frame = Frame::new("sample");
        for (name, value) in [
            ("model", &args.model),
            ("noise", &args.noise),
            ("positive", &args.positive),
            ("negative", &args.negative),
            ("cfg", &args.cfg),
            ("device", &args.device),
            ("sampler", &args.sampler),
            ("sigmas", &args.sigmas),
            ("model_options", &args.model_options),
            ("latent_image", &args.latent),
            ("denoise_mask", &args.mask),
            ("callback", &args.callback),
            ("disable_pbar", &args.disable),
            ("seed", &args.seed),
        ] {
            frame.set(name, Rc::new(value.clone()));
        }
        if let Some(guider) = &owned_guider {
            frame.set("cfg_guider", Rc::new(guider.clone()));
        }
        frame.finish();
        error.retain_frame(frame)
    })
}
