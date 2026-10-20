//! CFGGuider prediction dispatch. Wrappers come from the live guider options;
//! sampling receives the explicit call's options, not an implicit replacement.
use super::guider::GuiderState;
use crate::{
    native::Error,
    patching::{
        extensions::{Original, WrapperExecutor},
        options::Options,
    },
};
use std::{any::Any, rc::Rc};
pub struct PredictionArguments {
    pub x: Options,
    pub timestep: Options,
    pub model_options: Options,
    pub seed: Options,
}
pub struct SamplingArguments<'a> {
    pub model: &'a Options,
    pub x: &'a Options,
    pub timestep: &'a Options,
    pub negative: &'a Options,
    pub positive: &'a Options,
    pub cfg: &'a Options,
    pub model_options: &'a Options,
    pub seed: &'a Options,
}
pub trait PredictionBindings {
    fn sampling_function(&self, arguments: SamplingArguments<'_>) -> Result<Options, Error>;
}
pub trait PredictionWrappers {
    /// Bound current predict_noise method, with its original argument binder.
    fn original(&self) -> Result<Original<Options>, Error>;
    fn canonical_guider(&self) -> Rc<dyn Any>;
}
pub fn predict_noise(
    state: &GuiderState,
    args: &PredictionArguments,
    bindings: &dyn PredictionBindings,
) -> Result<Options, Error> {
    let model = state.fields.get("inner_model")?;
    let negative = state
        .fields
        .get("conds")?
        .get("negative")?
        .unwrap_or(Options::Atom(crate::execution_value::Value::None));
    let positive = state
        .fields
        .get("conds")?
        .get("positive")?
        .unwrap_or(Options::Atom(crate::execution_value::Value::None));
    let cfg = state.fields.get("cfg")?;
    bindings.sampling_function(SamplingArguments {
        model: &model,
        x: &args.x,
        timestep: &args.timestep,
        negative: &negative,
        positive: &positive,
        cfg: &cfg,
        model_options: &args.model_options,
        seed: &args.seed,
    })
}
/// The caller provides the actual shared default objects when arguments were
/// omitted. This body does not manufacture a fresh default model_options dict.
pub fn outer_predict_noise(
    state: &GuiderState,
    args: PredictionArguments,
    bindings: &dyn PredictionWrappers,
) -> Result<Options, Error> {
    let original = bindings.original()?;
    let guider = bindings.canonical_guider();
    let wrappers = super::preparation_wrappers::all_wrappers_of(
        "predict_noise",
        &state.fields.get("model_options")?,
    )?;
    let executor = WrapperExecutor::new(original, Some(guider), &wrappers, 0);
    // The source collector's temporary list is released after construction,
    // before invocation; retaining this Vec would keep removed wrappers alive.
    drop(wrappers);
    executor.execute(
        &[args.x, args.timestep, args.model_options, args.seed],
        &vec![],
    )
}
