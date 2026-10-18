//! Owned source values with a borrowed call context for deferred validation.
use crate::{
    execution::Runtime,
    execution_value::Value,
    native::{Error, Kitchen},
    registry::{Argument, Failure},
};
#[derive(Clone)]
pub struct SourceArgument<'rt> {
    pub(crate) value: Value,
    pub(crate) native: Kitchen,
    pub(crate) metadata: Option<Box<Argument<'static>>>,
    pub(crate) runtime: Option<&'rt Runtime>,
}
impl std::fmt::Debug for SourceArgument<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceArgument")
            .field("type", &self.typename())
            .finish_non_exhaustive()
    }
}
pub(crate) fn failure(e: Error) -> Failure {
    match e {
        Error::ReferenceException { class, message } => Failure::raised(class, message),
        Error::AcceleratorException { message, .. } => Failure::raised("AcceleratorError", message),
        e => Failure::raised("RuntimeError", e.to_string()),
    }
}
impl SourceArgument<'_> {
    pub(crate) fn integers(&self) -> Option<Vec<i64>> {
        match &self.value {
            Value::List(v) | Value::Tuple(v) => v
                .iter()
                .map(|v| match v {
                    Value::Int(n) => Some(*n),
                    Value::Bool(b) => Some(i64::from(*b)),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
    pub(crate) fn typename(&self) -> &'static str {
        crate::backend_triton_fp8::typename(&self.value)
    }
    pub(crate) fn hashable(&self) -> Result<(), Failure> {
        crate::python_value::ensure_hashable(&self.value).map_err(failure)
    }

    pub(crate) fn display(&self) -> Result<String, Failure> {
        if let Some(metadata) = &self.metadata {
            return metadata.dtype_failure_value();
        }
        match &self.value {
            Value::Text(s) => Ok(s.clone()),
            Value::Device(_) => crate::python_value::display(&self.value).map_err(failure),
            v => crate::python_value::repr_with_native(v, &self.native).map_err(failure),
        }
    }
    pub(crate) fn op(&self, op: &str, args: &[Value]) -> Result<Value, Failure> {
        unsafe {
            if let Some(rt) = self.runtime {
                return rt.op(op, args, &vec![]).map_err(failure);
            }
            let (name, overload) = op.rsplit_once('.').unwrap();
            crate::boxed_aten::call(
                &self.native,
                name,
                if overload == "default" { "" } else { overload },
                args,
                &vec![],
            )
            .map_err(failure)
        }
    }
    pub(crate) fn len(&self) -> Result<usize, Failure> {
        match &self.value {
            Value::Tuple(v) | Value::List(v) => Ok(v.len()),
            Value::Dict(v) => Ok(v.len()),
            Value::Text(v) => Ok(v.chars().count()),
            Value::Tensor(_) | Value::Quantized(_) => {
                let shape = self.metadata.as_ref().unwrap().tensor_parts().unwrap().2;
                shape
                    .first()
                    .map(|n| *n as usize)
                    .ok_or_else(|| Failure::raised("TypeError", "len() of a 0-d tensor"))
            }
            _ => Err(Failure::raised(
                "TypeError",
                format!("object of type '{}' has no len()", self.typename()),
            )),
        }
    }
    pub(crate) fn items(&self) -> Result<Vec<Value>, Failure> {
        match &self.value {
            Value::List(v) | Value::Tuple(v) => Ok(v.clone()),
            Value::Dict(v) => Ok(v.iter().map(|(k, _)| Value::Text(k.clone())).collect()),
            Value::Text(v) => Ok(v.chars().map(|c| Value::Text(c.to_string())).collect()),
            Value::Tensor(_) | Value::Quantized(_) => {
                if self
                    .metadata
                    .as_ref()
                    .unwrap()
                    .tensor_parts()
                    .unwrap()
                    .2
                    .is_empty()
                {
                    return Err(Failure::raised("TypeError", "iteration over a 0-d tensor"));
                }
                match self.op("aten::unbind.int", &[self.value.clone(), Value::Int(0)])? {
                    Value::Tuple(v) | Value::List(v) => Ok(v),
                    _ => unreachable!("Original unbind returns tensor views"),
                }
            }
            _ => Err(Failure::raised(
                "TypeError",
                format!("'{}' object is not iterable", self.typename()),
            )),
        }
    }
    pub(crate) fn any_less_than_one(&self) -> Result<bool, Failure> {
        for v in self.items()? {
            let negative = match &v {
                Value::Int(n) => *n < 1,
                Value::UInt(n) => *n < 1,
                Value::Bool(b) => !b,
                Value::Float(n) => *n < 1.,
                Value::Tensor(_) | Value::Quantized(_) => {
                    let compared = self.op("aten::lt.Scalar", &[v.clone(), Value::Int(1)])?;
                    match self.op("aten::is_nonzero.default", &[compared])? {
                        Value::Bool(b) => b,
                        _ => unreachable!(),
                    }
                }
                _ => {
                    return Err(Failure::raised(
                        "TypeError",
                        format!(
                            "'<' not supported between instances of '{}' and 'int'",
                            crate::backend_triton_fp8::typename(&v)
                        ),
                    ));
                }
            };
            if negative {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(crate) fn list_repr(&self) -> Result<String, Failure> {
        crate::python_value::repr_with_native(&Value::List(self.items()?), &self.native)
            .map_err(failure)
    }
}
