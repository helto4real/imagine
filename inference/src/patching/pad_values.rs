//! Both original padding bodies: eager comparison list, then zeros size parsing.
use super::exception;
use super::tensor_scalar::{Comparison as TensorComparison, checked, compare_integer};
use crate::{execution_value::Value, native::Error};
use std::cmp::Ordering;
use tch::{Kind, Tensor};
unsafe extern "C" {
    fn at_imagine_shape_unsigned_extent(value: u64, position: i64) -> i64;
    fn at_imagine_shape_tensor_extent(value: *const std::ffi::c_void) -> i64;
}
fn typename(value: &Value) -> &'static str {
    crate::backend_triton_fp8::typename(value)
}
fn len(value: &Value) -> Result<usize, Error> {
    Ok(match value {
        Value::List(v) | Value::Tuple(v) => v.len(),
        Value::NamedTuple(v) => v.fields.len(),
        Value::Dict(v) => v.len(),
        Value::Text(v) => v.chars().count(),
        Value::Tensor(_) => *value
            .as_tensor()?
            .size()
            .first()
            .ok_or_else(|| exception("TypeError", "len() of a 0-d tensor"))?
            as usize,
        _ => {
            return Err(exception(
                "TypeError",
                format!("object of type '{}' has no len()", typename(value)),
            ));
        }
    })
}
fn get(value: &Value, index: usize) -> Result<Value, Error> {
    Ok(match value {
        Value::List(v) | Value::Tuple(v) => v[index].clone(),
        Value::NamedTuple(v) => v.fields[index].1.clone(),
        Value::Text(v) => Value::Text(v.chars().nth(index).unwrap().to_string()),
        // Execution Value's dictionary admits text keys, which cannot equal int.
        Value::Dict(_) => return Err(exception("KeyError", index.to_string())),
        Value::Tensor(_) => Value::tensor(value.as_tensor()?.f_select(0, index as i64)?),
        _ => unreachable!("source len has already rejected non-subscriptable values"),
    })
}
enum Comparison {
    Bool(bool),
    Tensor(Tensor),
}
fn compare(value: &Value, old: i64) -> Result<Comparison, Error> {
    Ok(Comparison::Bool(match value {
        Value::Int(v) => *v < old,
        Value::UInt(v) => i128::from(*v) < i128::from(old),
        Value::Bool(v) => i64::from(*v) < old,
        Value::Float(v) => {
            crate::python_numbers::int_float_cmp(i128::from(old), *v) == Some(Ordering::Greater)
        }
        Value::Tensor(_) => {
            let tensor = value.as_tensor()?;
            return Ok(Comparison::Tensor(compare_integer(
                &tensor,
                old,
                TensorComparison::Less,
            )?));
        }
        _ => {
            return Err(exception(
                "TypeError",
                format!(
                    "'<' not supported between instances of '{}' and 'int'",
                    typename(value)
                ),
            ));
        }
    }))
}
fn integer_tensor(value: &Value) -> Result<Option<Tensor>, Error> {
    if !matches!(value, Value::Tensor(_)) {
        return Ok(None);
    }
    let tensor = value.as_tensor()?;
    // Original is_int_or_symint: one integral tensor element, including bool.
    Ok((tensor.numel() == 1
        && matches!(
            tensor.kind(),
            Kind::Uint8 | Kind::Int8 | Kind::Int16 | Kind::Int | Kind::Int64 | Kind::Bool
        ))
    .then_some(tensor))
}
fn parses_as_int(value: &Value) -> Result<bool, Error> {
    Ok(matches!(value, Value::Int(_) | Value::UInt(_)) || integer_tensor(value)?.is_some())
}
fn extent(value: &Value, index: usize) -> Result<i64, Error> {
    match value {
        Value::Int(v) => Ok(*v),
        Value::UInt(v) => {
            checked(|| unsafe { at_imagine_shape_unsigned_extent(*v, index as i64 + 1) })
        }
        _ => {
            if let Some(tensor) = integer_tensor(value)? {
                return checked(|| unsafe {
                    at_imagine_shape_tensor_extent(tensor.as_ptr().cast())
                });
            }
            Err(exception(
                "TypeError",
                format!(
                    "zeros(): argument 'size' failed to unpack the object at pos {} with error \"type must be tuple of ints,but got {}\"",
                    index + 1,
                    typename(value)
                ),
            ))
        }
    }
}
fn zeros_shape(value: &Value) -> Result<Vec<i64>, Error> {
    let values: Vec<&Value> = match value {
        Value::List(v) | Value::Tuple(v) => v.iter().collect(),
        Value::NamedTuple(v) => v.fields.iter().map(|(_, v)| v).collect(),
        _ => {
            if parses_as_int(value)? {
                return Ok(vec![extent(value, 0)?]);
            }
            return Err(exception(
                "TypeError",
                format!(
                    "zeros(): argument 'size' (position 1) must be tuple of ints, not {}",
                    typename(value)
                ),
            ));
        }
    };
    if let Some(first) = values.first()
        && !parses_as_int(first)?
    {
        return Err(exception(
            "TypeError",
            format!(
                "zeros(): argument 'size' (position 1) must be tuple of ints, but found element of type {} at pos 0",
                typename(first)
            ),
        ));
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| extent(value, index))
        .collect()
}
/// Source raw-shape handling; no inferred integer coercion or skipped reshape.
pub fn pad_to_value_shape(tensor: &Tensor, shape: &Value) -> Result<Tensor, Error> {
    let mut comparisons = Vec::new();
    for index in 0..len(shape)? {
        let dimension = get(shape, index)?;
        let old = *tensor
            .size()
            .get(index)
            .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
        comparisons.push(compare(&dimension, old)?);
    }
    for comparison in comparisons {
        let smaller = match comparison {
            Comparison::Bool(value) => value,
            Comparison::Tensor(value) => value.f_is_nonzero()?,
        };
        if smaller {
            return Err(exception(
                "ValueError",
                "The new shape must be larger than the original tensor in all dimensions",
            ));
        }
    }
    if len(shape)? != tensor.dim() {
        return Err(exception(
            "ValueError",
            "The new shape must have the same number of dimensions as the original tensor",
        ));
    }
    let output = Tensor::f_zeros(zeros_shape(shape)?, (tensor.kind(), tensor.device()))?;
    let mut target = output.shallow_clone();
    for (index, size) in tensor.size().into_iter().enumerate() {
        target = target.f_narrow(index as i64, 0, size)?;
    }
    target.f_copy_(tensor)?;
    Ok(output)
}
