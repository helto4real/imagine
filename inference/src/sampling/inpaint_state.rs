//! Original mutable KSamplerX0Inpaint fields and call body. Arithmetic/model
//! dispatch remains an explicit source provider, including in-place side effects.
use crate::{
    execution_value::Value as V,
    native::Error,
    patching::{attributes::AttributeStore, options::Options},
};
pub struct InpaintState {
    pub fields: AttributeStore,
}
impl InpaintState {
    pub fn new(model: Options, sigmas: Options) -> Self {
        let state = Self {
            fields: AttributeStore::new("KSamplerX0Inpaint"),
        };
        state.initialize(model, sigmas);
        state
    }
    pub fn initialize(&self, model: Options, sigmas: Options) {
        self.fields.set("inner_model", model);
        self.fields.set("sigmas", sigmas);
    }
}
/// One source __call__ definition; its omitted model_options dictionary is
/// shared by every instance of that class/definition.
#[derive(Clone)]
pub struct InpaintClass {
    default_options: Options,
}
impl Default for InpaintClass {
    fn default() -> Self {
        Self {
            default_options: Options::dictionary(vec![]),
        }
    }
}
pub struct Arguments {
    pub x: Options,
    pub sigma: Options,
    pub mask: Options,
    pub model_options: Option<Options>,
    pub seed: Options,
}
pub trait Bindings {
    fn contains(&self, object: &Options, key: &str) -> Result<bool, Error>;
    fn item(&self, object: &Options, key: &str) -> Result<Options, Error>;
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error>;
    fn call(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error>;
    fn subtract_from_one(&self, value: &Options) -> Result<Options, Error>;
    fn multiply(&self, left: &Options, right: &Options) -> Result<Options, Error>;
    fn add(&self, left: &Options, right: &Options) -> Result<Options, Error>;
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
impl InpaintClass {
    pub fn default_options(&self) -> Options {
        self.default_options.clone()
    }
    pub fn call(
        &self,
        state: &InpaintState,
        mut args: Arguments,
        bindings: &dyn Bindings,
    ) -> Result<Options, Error> {
        let model_options = args
            .model_options
            .take()
            .unwrap_or_else(|| self.default_options.clone());
        let mut latent_mask = None;
        if !none(&args.mask) {
            if bindings.contains(&model_options, "denoise_mask_function")? {
                let function = bindings.item(&model_options, "denoise_mask_function")?;
                let extra = Options::dictionary(vec![
                    ("model".into(), state.fields.get("inner_model")?),
                    ("sigmas".into(), state.fields.get("sigmas")?),
                ]);
                let mask = bindings.call(
                    &function,
                    &[args.sigma.clone(), args.mask.clone()],
                    &[("extra_options".into(), extra)],
                );
                drop(function);
                drop(std::mem::replace(&mut args.mask, mask?));
            }
            latent_mask = Some(bindings.subtract_from_one(&args.mask)?);
            let masked = bindings.multiply(&args.x, &args.mask)?;
            let inner = state.fields.get("inner_model")?;
            let model = bindings.attribute(&inner, "inner_model");
            drop(inner);
            let model = model?;
            let function = bindings.attribute(&model, "scale_latent_inpaint");
            drop(model);
            let function = function?;
            let scaled = bindings.call(
                &function,
                &[],
                &[
                    ("x".into(), args.x.clone()),
                    ("sigma".into(), args.sigma.clone()),
                    ("noise".into(), state.fields.get("noise")?),
                    ("latent_image".into(), state.fields.get("latent_image")?),
                    ("denoise_mask".into(), args.mask.clone()),
                ],
            );
            drop(function);
            let scaled = scaled?;
            let value = bindings.multiply(&scaled, latent_mask.as_ref().unwrap());
            drop(scaled);
            let scaled = value?;
            let value = bindings.add(&masked, &scaled);
            drop(masked);
            drop(scaled);
            let value = value?;
            drop(std::mem::replace(&mut args.x, value));
        }
        let function = state.fields.get("inner_model")?;
        let out = bindings.call(
            &function,
            &[args.x.clone(), args.sigma.clone()],
            &[
                ("model_options".into(), model_options.clone()),
                ("seed".into(), args.seed.clone()),
            ],
        );
        drop(function);
        let mut out = out?;
        if !none(&args.mask) {
            let masked = bindings.multiply(&out, &args.mask)?;
            let latent = state.fields.get("latent_image")?;
            let value = bindings.multiply(&latent, latent_mask.as_ref().unwrap());
            drop(latent);
            let latent = value?;
            let value = bindings.add(&masked, &latent);
            drop(masked);
            drop(latent);
            out = value?;
        }
        Ok(out)
    }
}
