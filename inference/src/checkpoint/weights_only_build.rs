//! Original BUILD body. Every dynamic operation keeps its source getter/caller
//! boundary; this policy does not substitute normal dicts for custom state.
use super::weights_only_reduce::{self, Bindings as ReduceBindings};
use crate::{
    native::Error,
    patching::options::{Options, SourceNone},
};

pub trait Bindings: ReduceBindings {
    fn exact_type(&mut self, instance: &Options) -> Result<Options, Error>;
    fn tensor_type(&mut self) -> Result<Options, Error>;
    fn parameter_type(&mut self) -> Result<Options, Error>;
    fn ordered_dictionary_type(&mut self) -> Result<Options, Error>;
    fn legacy_tensor_set(&mut self, instance: &Options, state: &Options) -> Result<(), Error>;
    fn get_setstate(&mut self, instance: &Options) -> Result<Options, Error>;
    fn call_setstate(&mut self, method: &Options, state: &Options) -> Result<(), Error>;
    fn dictionary_update(&mut self, instance: &Options, state: &Options) -> Result<(), Error>;
    /// Original isinstance(tuple), len==2 and unpack operations, in order.
    fn split_slot_state(&mut self, state: &Options) -> Result<Option<(Options, Options)>, Error>;
    fn truth(&mut self, value: &Options) -> Result<bool, Error>;
    fn slot_items(&mut self, slots: &Options) -> Result<Options, Error>;
    /// Includes iteration and the original pair-unpack operation.
    fn next_slot(&mut self, iterator: &Options) -> Result<Option<(Options, Options)>, Error>;
    fn set_attribute(
        &mut self,
        instance: &Options,
        key: &Options,
        value: &Options,
    ) -> Result<(), Error>;
}
pub fn build(
    instance: &Options,
    state: &Options,
    bindings: &mut dyn Bindings,
) -> Result<(), Error> {
    let class = bindings.exact_type(instance)?;
    let base = bindings.tensor_type()?;
    let tensor = class.same_object(&base);
    drop(class);
    drop(base);
    if tensor {
        return bindings.legacy_tensor_set(instance, state);
    }
    let class = bindings.exact_type(instance)?;
    let base = bindings.parameter_type()?;
    let parameter = class.same_object(&base);
    drop(class);
    drop(base);
    if parameter {
        let method = bindings.get_setstate(instance)?;
        return bindings.call_setstate(&method, state);
    }
    let class = bindings.exact_type(instance)?;
    let base = bindings.ordered_dictionary_type()?;
    let ordered = class.same_object(&base);
    drop(class);
    drop(base);
    if ordered {
        return bindings.dictionary_update(instance, state);
    }
    let class = bindings.exact_type(instance)?;
    let user = bindings.user()?;
    let user_admitted = weights_only_reduce::contains(user, &class, bindings)?;
    drop(class);
    let admitted = if user_admitted {
        true
    } else {
        let class = bindings.exact_type(instance)?;
        let allowed = bindings.allowed()?;
        weights_only_reduce::contains(allowed, &class, bindings)?
    };
    if !admitted {
        let class = bindings.exact_type(instance)?;
        return Err(Error::ReferenceException {
            class: "UnpicklingError",
            message: format!(
                "Can only build Tensor, Parameter, OrderedDict or types allowlisted via `add_safe_globals`, but got {}",
                bindings.format(&class)?
            ),
        });
    }
    let has_setstate = match bindings.get_setstate(instance) {
        Ok(_) => true,
        Err(error) if bindings.is_attribute_error(&error) => false,
        Err(error) => return Err(error),
    };
    if has_setstate {
        let method = bindings.get_setstate(instance)?;
        return bindings.call_setstate(&method, state);
    }
    let (state, slots) = bindings
        .split_slot_state(state)?
        .unwrap_or_else(|| (state.clone(), SourceNone::value()));
    if bindings.truth(&state)? {
        bindings.dictionary_update(instance, &state)?;
    }
    if bindings.truth(&slots)? {
        let iterator = bindings.slot_items(&slots)?;
        let (mut key, mut value) = (None, None);
        while let Some((next_key, next_value)) = bindings.next_slot(&iterator)? {
            drop(key.replace(next_key));
            drop(value.replace(next_value));
            bindings.set_attribute(instance, key.as_ref().unwrap(), value.as_ref().unwrap())?;
        }
    }
    Ok(())
}
