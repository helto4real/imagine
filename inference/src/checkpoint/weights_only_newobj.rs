//! Original NEWOBJ authorization: Parameter identity precedes user values,
//! which precede default values. Never replace __new__ with a normal call.
use super::weights_only_reduce::{self, Bindings as ReduceBindings};
use crate::{native::Error, patching::options::Options};

pub trait Bindings: ReduceBindings {
    fn parameter_class(&mut self) -> Result<Options, Error>;
    fn call_parameter(&mut self, args: &Options) -> Result<Options, Error>;
    /// Read the original __new__ attribute and invoke it with cls first,
    /// after the source caller has expanded args. This is not cls(*args).
    fn call_new(&mut self, class: &Options, args: &Options) -> Result<Options, Error>;
}
pub fn new_object(
    class: &Options,
    args: &Options,
    bindings: &mut dyn Bindings,
) -> Result<Options, Error> {
    let parameter = bindings.parameter_class()?;
    let is_parameter = class.same_object(&parameter);
    drop(parameter);
    if is_parameter {
        return bindings.call_parameter(args);
    }
    let user = bindings.user()?;
    let admitted = if weights_only_reduce::contains(user, class, bindings)? {
        true
    } else {
        let allowed = bindings.allowed()?;
        weights_only_reduce::contains(allowed, class, bindings)?
    };
    if !admitted {
        return Err(Error::ReferenceException {
            class: "UnpicklingError",
            message: format!(
                "Can only create new object for nn.Parameter or classes allowlisted via `add_safe_globals` but got {}",
                bindings.format(class)?
            ),
        });
    }
    let result = bindings.call_new(class, args)?;
    if bindings.tensor_class_contains(class)? && bindings.module_contains_sparse(class)? {
        bindings.append_sparse(&result)?;
    }
    Ok(result)
}
