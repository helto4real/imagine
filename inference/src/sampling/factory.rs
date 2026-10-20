//! ksampler / sampler_object construction and the original KSAMPLER fields.
//! The module owner retains the four distinct mutable default dictionaries.
use crate::{
    native::Error,
    patching::{attributes::AttributeStore, options::Options},
};
use std::rc::Rc;

pub struct SamplerState {
    pub fields: AttributeStore,
}
impl SamplerState {
    pub fn new(function: Options, extra: Options, inpaint: Options) -> Self {
        let fields = AttributeStore::new("KSAMPLER");
        let state = Self { fields };
        state.initialize(function, extra, inpaint);
        state
    }
    pub fn initialize(&self, function: Options, extra: Options, inpaint: Options) {
        self.fields.set("sampler_function", function);
        self.fields.set("extra_options", extra);
        self.fields.set("inpaint_options", inpaint);
    }
}
pub trait FunctionBindings {
    /// Actual module attribute lookup, including non-callable replacements and
    /// missing attributes. Construction does not prevalidate the returned value.
    fn attribute(&mut self, module: &str, name: &str) -> Result<Options, Error>;
}
pub trait MaxDenoiseBindings {
    fn attribute(&mut self, object: &Options, name: &str) -> Result<Options, Error>;
    fn first_sigma(&mut self, sigmas: &Options) -> Result<Options, Error>;
    fn float(&mut self, value: &Options) -> Result<f64, Error>;
}
pub fn max_denoise(
    model: &Options,
    sigmas: &Options,
    bindings: &mut dyn MaxDenoiseBindings,
) -> Result<bool, Error> {
    let inner = bindings.attribute(model, "inner_model")?;
    let sampling = bindings.attribute(&inner, "model_sampling");
    drop(inner);
    let sampling = sampling?;
    let maximum = bindings.attribute(&sampling, "sigma_max");
    drop(sampling);
    let maximum = maximum?;
    let value = bindings.float(&maximum);
    drop(maximum);
    let maximum = value?;
    let first = bindings.first_sigma(sigmas)?;
    let value = bindings.float(&first);
    drop(first);
    let sigma = value?;
    Ok(super::ksampler::maximum_denoise(maximum, sigma))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DpmWrapper {
    Fast,
    Adaptive,
}
impl DpmWrapper {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fast => "dpm_fast_function",
            Self::Adaptive => "dpm_adaptive_function",
        }
    }
    /// Original wrapper with no solver-option overrides. Adaptive overrides
    /// retain the public dpm::adaptive contract and are not silently discarded.
    pub fn sample_defaults(
        self,
        exec: &mut dyn super::solvers::Execution,
        native: &crate::native::Kitchen,
        x: tch::Tensor,
        sigmas: &tch::Tensor,
    ) -> Result<tch::Tensor, Error> {
        let Some((minimum, maximum, steps)) = super::dpm::bounds(sigmas)? else {
            return Ok(x);
        };
        match self {
            Self::Fast => super::dpm::fast(
                exec,
                native,
                x,
                &minimum,
                &maximum,
                steps,
                super::dpm::Stochastic::default(),
            ),
            Self::Adaptive => super::dpm::adaptive(
                exec,
                native,
                x,
                &minimum,
                &maximum,
                super::dpm::Adaptive::default(),
            )
            .map(|(samples, _)| samples),
        }
    }
}
/// One source module initialization. Clones retain the same defaults; a fresh
/// instance is a separate module/execution context, not a per-call allocation.
#[derive(Clone)]
pub struct Factory {
    function_extra: Options,
    function_inpaint: Options,
    constructor_extra: Options,
    constructor_inpaint: Options,
}
impl Default for Factory {
    fn default() -> Self {
        Self {
            function_extra: Options::dictionary(vec![]),
            function_inpaint: Options::dictionary(vec![]),
            constructor_extra: Options::dictionary(vec![]),
            constructor_inpaint: Options::dictionary(vec![]),
        }
    }
}
impl Factory {
    /// Function defaults followed by constructor defaults, in source order.
    pub fn default_dictionaries(&self) -> [Options; 4] {
        [
            self.function_extra.clone(),
            self.function_inpaint.clone(),
            self.constructor_extra.clone(),
            self.constructor_inpaint.clone(),
        ]
    }
    /// None is an omitted argument; Some(source None) is an explicit None.
    pub fn construct(
        &self,
        function: Options,
        extra: Option<Options>,
        inpaint: Option<Options>,
    ) -> SamplerState {
        SamplerState::new(
            function,
            extra.unwrap_or_else(|| self.constructor_extra.clone()),
            inpaint.unwrap_or_else(|| self.constructor_inpaint.clone()),
        )
    }
    pub fn ksampler(
        &self,
        name: &str,
        extra: Option<Options>,
        inpaint: Option<Options>,
        bindings: &mut dyn FunctionBindings,
    ) -> Result<SamplerState, Error> {
        let function = match name {
            "dpm_fast" => Options::Object(Rc::new(DpmWrapper::Fast)),
            "dpm_adaptive" => Options::Object(Rc::new(DpmWrapper::Adaptive)),
            _ => bindings.attribute("k_diffusion_sampling", &format!("sample_{name}"))?,
        };
        Ok(self.construct(
            function,
            Some(extra.unwrap_or_else(|| self.function_extra.clone())),
            Some(inpaint.unwrap_or_else(|| self.function_inpaint.clone())),
        ))
    }
    pub fn sampler_object(
        &self,
        name: &str,
        bindings: &mut dyn FunctionBindings,
    ) -> Result<SamplerState, Error> {
        match name {
            "uni_pc" | "uni_pc_bh2" => Ok(self.construct(
                bindings.attribute(
                    "uni_pc",
                    if name == "uni_pc" {
                        "sample_unipc"
                    } else {
                        "sample_unipc_bh2"
                    },
                )?,
                None,
                None,
            )),
            "ddim" => self.ksampler(
                "euler",
                None,
                Some(Options::dictionary(vec![(
                    "random".into(),
                    Options::Atom(crate::execution_value::Value::Bool(true)),
                )])),
                bindings,
            ),
            _ => self.ksampler(name, None, None, bindings),
        }
    }
}
