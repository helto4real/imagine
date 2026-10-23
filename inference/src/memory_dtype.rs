//! Original model_management.dtype_size, including comparison short circuits
//! and the bare-except fallback for source attribute exceptions.
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use crate::{
    native::ShellDType,
    registry::{AllowedType, ScalarType},
};
use tch::Kind;
pub trait DtypeSizeBindings {
    /// Source equality followed by truth conversion; errors propagate.
    fn equals(&self, value: &Options, dtype: Kind) -> Result<bool, Error>;
    /// Actual itemsize attribute, preserving its value without conversion.
    fn itemsize(&self, value: &Options) -> Result<Options, Error>;
}
/// Actual torch.dtype values represented by the locked tch Kind enum. Other
/// objects still require their equality/attribute protocols through the trait.
pub struct KindDtypeSizeBindings;
impl DtypeSizeBindings for KindDtypeSizeBindings {
    fn equals(&self, value: &Options, dtype: Kind) -> Result<bool, Error> {
        if let Some(value) = crate::checkpoint::dtype_objects::SourceDtype::get(value) {
            return Ok(value.kind() == Some(dtype));
        }
        match value {
            Options::Atom(V::DType(value)) => Ok(*value == dtype),
            _ => Err(Error::BindingGap {
                backend: "dtype_size".into(),
                operation: "non-Kind equality protocol".into(),
            }),
        }
    }
    fn itemsize(&self, value: &Options) -> Result<Options, Error> {
        if let Some(value) = crate::checkpoint::dtype_objects::SourceDtype::get(value) {
            return value.property("itemsize");
        }
        match value {
            Options::Atom(V::DType(value)) => {
                Ok(Options::Atom(V::Int(value.elt_size_in_bytes() as i64)))
            }
            _ => Err(Error::BindingGap {
                backend: "dtype_size".into(),
                operation: "non-Kind itemsize attribute".into(),
            }),
        }
    }
}
/// Registry type metadata already used by the Kitchen argument boundary,
/// including the two shell dtypes outside tch's Kind enum. This adapter reads
/// that metadata; it does not introduce a new source-object identity scheme.
pub struct RegisteredDtypeSizeBindings;
impl DtypeSizeBindings for RegisteredDtypeSizeBindings {
    fn equals(&self, value: &Options, dtype: Kind) -> Result<bool, Error> {
        if let Options::Object(value) = value
            && let Some(value) = value.downcast_ref::<AllowedType>()
        {
            return Ok(matches!(value, AllowedType::DType(kind) if *kind == dtype));
        }
        KindDtypeSizeBindings.equals(value, dtype)
    }
    fn itemsize(&self, value: &Options) -> Result<Options, Error> {
        if let Options::Object(value) = value
            && let Some(value) = value.downcast_ref::<AllowedType>()
        {
            return match value {
                AllowedType::DType(kind) => {
                    Ok(Options::Atom(V::Int(kind.elt_size_in_bytes() as i64)))
                }
                AllowedType::ShellDType(ShellDType::E8M0 | ShellDType::PackedE2M1) => {
                    Ok(Options::Atom(V::Int(1)))
                }
                AllowedType::Scalar(kind) => {
                    let name = match kind {
                        ScalarType::Int => "int",
                        ScalarType::Float => "float",
                        ScalarType::Bool => "bool",
                        ScalarType::Str => "str",
                        ScalarType::NoneType => "NoneType",
                    };
                    Err(Error::ReferenceException {
                        class: "AttributeError",
                        message: format!("type object '{name}' has no attribute 'itemsize'"),
                    })
                }
            };
        }
        KindDtypeSizeBindings.itemsize(value)
    }
}
pub fn dtype_size(value: &Options, bindings: &dyn DtypeSizeBindings) -> Result<Options, Error> {
    if bindings.equals(value, Kind::Half)? || bindings.equals(value, Kind::BFloat16)? {
        return Ok(Options::Atom(V::Int(2)));
    }
    if bindings.equals(value, Kind::Float)? {
        return Ok(Options::Atom(V::Int(4)));
    }
    match bindings.itemsize(value) {
        Ok(size) => Ok(size),
        Err(error)
            if matches!(
                error.root(),
                Error::ReferenceException { .. }
                    | Error::AcceleratorException { .. }
                    | Error::Torch(_)
            ) =>
        {
            Ok(Options::Atom(V::Int(4)))
        }
        // A missing native binding/invariant is not a caught Python exception.
        Err(error) => Err(error),
    }
}
