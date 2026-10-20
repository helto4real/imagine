//! Original CFGGuider construction and condition setters. Delegating a dynamic
//! patcher deliberately does not refresh the constructor's model_options alias.
use super::guider::GuiderState;
use crate::{
    execution_value::Value as V,
    native::Error,
    patching::{options::Options, options_iterator::SequenceIterator},
};
pub trait StateBindings {
    fn model_options(&self, patcher: &Options) -> Result<Options, Error>;
    fn empty_dictionary(&self) -> Result<Options, Error>;
    fn condition_pair(&self, positive: Options, negative: Options) -> Result<Options, Error>;
    fn is_dynamic(&self, patcher: &Options) -> Result<Options, Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn condition_has_hooks(&self, conditions: &Options) -> Result<Options, Error>;
    fn non_dynamic_delegate(&self, patcher: &Options) -> Result<Options, Error>;
    fn convert_condition(&self, conditions: &Options) -> Result<Options, Error>;
}
pub fn initialize(
    state: &GuiderState,
    patcher: Options,
    bindings: &dyn StateBindings,
) -> Result<(), Error> {
    state.fields.set("model_patcher", patcher.clone());
    state
        .fields
        .set("model_options", bindings.model_options(&patcher)?);
    state
        .fields
        .set("original_conds", bindings.empty_dictionary()?);
    state.fields.set("cfg", Options::Atom(V::Float(1.)));
    Ok(())
}
pub fn set_cfg(state: &GuiderState, cfg: Options) {
    state.fields.set("cfg", cfg);
}
pub fn set_conds(
    state: &GuiderState,
    positive: Options,
    negative: Options,
    bindings: &dyn StateBindings,
) -> Result<(), Error> {
    let conditions = bindings.condition_pair(positive, negative)?;
    inner_set_conds(state, &conditions, bindings)
}
pub fn inner_set_conds(
    state: &GuiderState,
    conditions: &Options,
    bindings: &dyn StateBindings,
) -> Result<(), Error> {
    let mut keys = SequenceIterator::for_operation(conditions, "guider condition keys")?;
    let mut last = None;
    while let Some(key) = keys.next()? {
        drop(last.replace(key));
        let key = last.as_ref().unwrap();
        if bindings.truth(&bindings.is_dynamic(&state.fields.get("model_patcher")?)?)?
            && bindings.truth(&bindings.condition_has_hooks(&conditions.item(key)?)?)?
        {
            let delegate = bindings.non_dynamic_delegate(&state.fields.get("model_patcher")?)?;
            state.fields.set("model_patcher", delegate);
        }
        let converted = bindings.convert_condition(&conditions.item(key)?)?;
        state
            .fields
            .get("original_conds")?
            .set_item(key.clone(), converted)?;
    }
    Ok(())
}
