//! Source calculate_shape retains raw tuple metadata before allocation validates it.
use super::{Patch, PatchValue, exception, gap};
use crate::{
    execution_value::{TORCH_SIZE_CLASS, Value},
    native::Error,
};
use std::cmp::Ordering;

pub fn size_metadata(shape: &[i64]) -> Value {
    Value::named_tuple(
        &TORCH_SIZE_CLASS,
        shape.iter().map(|value| ("", Value::Int(*value))).collect(),
    )
}
fn tuple(value: &Value) -> Result<Value, Error> {
    Ok(Value::Tuple(match value {
        Value::List(values) | Value::Tuple(values) => values.clone(),
        Value::NamedTuple(values) => values
            .fields
            .iter()
            .map(|(_, value)| value.clone())
            .collect(),
        Value::Dict(values) => values
            .iter()
            .map(|(key, _)| Value::Text(key.clone()))
            .collect(),
        Value::Text(value) => value.chars().map(|c| Value::Text(c.to_string())).collect(),
        Value::Tensor(_) => {
            let value = value.as_tensor()?;
            if value.dim() == 0 {
                return Err(exception("TypeError", "iteration over a 0-d tensor"));
            }
            value.f_unbind(0)?.into_iter().map(Value::tensor).collect()
        }
        _ => {
            return Err(exception(
                "TypeError",
                format!(
                    "'{}' object is not iterable",
                    crate::backend_triton_fp8::typename(value)
                ),
            ));
        }
    }))
}
pub fn calculate_shape_metadata(patches: &[Patch], mut shape: Value) -> Result<Value, Error> {
    for patch in patches {
        if patch.offset.is_some() || matches!(patch.value, PatchValue::Nested { .. }) {
            continue;
        }
        match &patch.value {
            PatchValue::Adapter(adapter) if adapter.name == "lora" => {
                let reshape = adapter
                    .weights
                    .get(5)
                    .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
                if !matches!(reshape, Value::None) {
                    shape = tuple(reshape)?;
                }
            }
            PatchValue::Diff { weight, pad: true } | PatchValue::Set(weight) => {
                shape = size_metadata(&weight.as_tensor()?.size())
            }
            _ => {}
        }
    }
    Ok(shape)
}
fn elements(shape: &Value) -> Result<Vec<&Value>, Error> {
    match shape {
        Value::Tuple(values) => Ok(values.iter().collect()),
        Value::NamedTuple(values) => Ok(values.fields.iter().map(|(_, v)| v).collect()),
        _ => Err(gap(
            "shape metadata must retain its source tuple/Size representation",
        )),
    }
}
pub fn shape_metadata_matches(shape: &Value, expected: &[i64]) -> Result<bool, Error> {
    let shape = elements(shape)?;
    // CPython tuple equality compares the shared prefix before lengths. A
    // tensor truth error can therefore win over unequal tuple lengths.
    for (value, expected) in shape.iter().zip(expected) {
        let equal = match value {
            Value::Int(value) => value == expected,
            Value::UInt(value) => i128::from(*value) == i128::from(*expected),
            Value::Bool(value) => i64::from(*value) == *expected,
            Value::Float(value) => {
                crate::python_numbers::int_float_cmp(i128::from(*expected), *value)
                    == Some(Ordering::Equal)
            }
            Value::Tensor(_) => super::tensor_scalar::compare_integer(
                &value.as_tensor()?,
                *expected,
                super::tensor_scalar::Comparison::Equal,
            )?
            .f_is_nonzero()?,
            _ => false,
        };
        if !equal {
            return Ok(false);
        }
    }
    Ok(shape.len() == expected.len())
}
pub(super) fn integer_dimensions(shape: &Value) -> Result<Vec<i64>, Error> {
    elements(shape)?.into_iter().map(|dimension|match dimension {
        Value::Int(value)=>Ok(*value),
        Value::UInt(value)=>i64::try_from(*value).map_err(|_|gap("unsigned shape metadata exceeds int64; retain the raw metadata until source allocation parsing")),
        _=>Err(gap("noninteger shape metadata requires its raw consumer")),
    }).collect()
}
