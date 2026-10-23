//! REDUCE's live authorization and post-call sparse registration, in original
//! order. Argument expansion and object protocols belong to the actual caller.
use super::weights_only_globals;
use crate::{native::Error, patching::options::Options};

pub trait Bindings: weights_only_globals::Bindings {
    /// Rich comparison after PyObject_RichCompareBool's identity shortcut.
    /// The dictionary value is the left operand, including reflected dispatch.
    fn equal(&mut self, value: &Options, function: &Options) -> Result<bool, Error>;
    /// Original empty-format-spec protocol, rather than repr substitution.
    fn format(&mut self, value: &Options) -> Result<String, Error>;
    fn bound_self(&mut self, function: &Options) -> Result<Options, Error>;
    /// Original exception-class matching, including AttributeError subclasses.
    /// This is class metadata, not a comparison of diagnostic strings.
    fn is_attribute_error(&self, error: &Error) -> bool;
    fn call(&mut self, function: &Options, args: &Options) -> Result<Options, Error>;
    fn tensor_class_contains(&mut self, function: &Options) -> Result<bool, Error>;
    fn module_contains_sparse(&mut self, function: &Options) -> Result<bool, Error>;
    fn append_sparse(&mut self, result: &Options) -> Result<(), Error>;
}

pub(crate) fn contains(
    dictionary: Options,
    function: &Options,
    bindings: &mut dyn Bindings,
) -> Result<bool, Error> {
    let values = dictionary.values_iterator()?;
    drop(dictionary);
    while let Some(value) = values.next_value()? {
        if value.same_object(function) || bindings.equal(&value, function)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn reduce(
    function: &Options,
    args: &Options,
    bindings: &mut dyn Bindings,
) -> Result<Options, Error> {
    let allowed = bindings.allowed()?;
    if !contains(allowed, function, bindings)? {
        let user = bindings.user()?;
        if !contains(user, function, bindings)? {
            let mut message = format!(
                "Trying to call reduce for unrecognized function {}",
                bindings.format(function)?
            );
            let has_self = match bindings.bound_self(function) {
                Ok(_) => true,
                Err(error) if bindings.is_attribute_error(&error) => false,
                Err(error) => return Err(error),
            };
            if has_self {
                // hasattr and the later attribute read are separate lookups.
                let receiver = bindings.bound_self(function)?;
                message.push_str(&format!(
                    " which belongs to {}",
                    bindings.format(&receiver)?
                ));
            }
            return Err(Error::ReferenceException {
                class: "UnpicklingError",
                message,
            });
        }
    }
    let result = bindings.call(function, args)?;
    if bindings.tensor_class_contains(function)? && bindings.module_contains_sparse(function)? {
        bindings.append_sparse(&result)?;
    }
    Ok(result)
}
