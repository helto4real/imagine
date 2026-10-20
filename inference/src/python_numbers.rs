//! Original Python numeric conversion at the frozen host argument boundary.
use crate::{execution::Runtime, execution_value::Value, native::Error};
use std::cmp::Ordering;
#[path = "python_decimal_digits.rs"]
mod digits;
pub(crate) fn decimal(c: char) -> Option<u8> {
    let n = u32::from(c);
    digits::ZEROS
        .iter()
        .find_map(|zero| n.checked_sub(*zero).filter(|d| *d < 10).map(|d| d as u8))
}
pub(crate) fn int_float_cmp(a: i128, b: f64) -> Option<Ordering> {
    if b.is_nan() {
        return None;
    }
    if b >= i128::MAX as f64 {
        return Some(Ordering::Less);
    }
    if b < i128::MIN as f64 {
        return Some(Ordering::Greater);
    }
    let truncated = b as i128;
    Some(a.cmp(&truncated).then_with(|| {
        if b.fract() > 0. {
            Ordering::Less
        } else if b.fract() < 0. {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }))
}

pub(crate) fn integer_ratio(a: i128, b: i128) -> Result<f64, Error> {
    if b == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "division by zero".into(),
        });
    }
    let negative = (a < 0) != (b < 0);
    let a = a.unsigned_abs();
    let b = b.unsigned_abs();
    if a > u128::from(u64::MAX) || b > u128::from(u64::MAX) {
        return Err(Error::BindingGap {
            backend: "integer arithmetic".into(),
            operation: "integer true division exceeds admitted 64-bit operands".into(),
        });
    }
    if a == 0 {
        return Ok(if negative { -0. } else { 0. });
    }
    let mut exponent = (128 - a.leading_zeros()) as i32 - (128 - b.leading_zeros()) as i32;
    if (exponent >= 0 && a < (b << exponent)) || (exponent < 0 && (a << -exponent) < b) {
        exponent -= 1;
    }
    let shift = 52 - exponent;
    let (numerator, denominator) = if shift >= 0 {
        (a << shift, b)
    } else {
        (a, b << -shift)
    };
    let mut significand = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder * 2 > denominator || (remainder * 2 == denominator && significand % 2 != 0) {
        significand += 1;
    }
    let value = significand as f64 * 2_f64.powi(exponent - 52);
    Ok(if negative { -value } else { value })
}

fn space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c') || (!c.is_ascii() && c.is_whitespace())
}
fn text_number(text: &str) -> Option<String> {
    let text = text.trim_matches(space);
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' {
            if i == 0
                || i + 1 == chars.len()
                || decimal(chars[i - 1]).is_none()
                || decimal(chars[i + 1]).is_none()
            {
                return None;
            }
        } else if let Some(d) = decimal(*c) {
            out.push(char::from(b'0' + d));
        } else if c.is_ascii() {
            out.push(*c);
        } else {
            return None;
        }
    }
    Some(out)
}
pub(crate) fn float_value(rt: &Runtime, value: &Value) -> Result<f64, Error> {
    let invalid = || {
        crate::execution_value::type_error(format!(
            "float() argument must be a string or a real number, not '{}'",
            crate::backend_triton_fp8::typename(value)
        ))
    };
    match value {
        Value::Float(n) => Ok(*n),
        Value::Int(n) => Ok(*n as f64),
        Value::UInt(n) => Ok(*n as f64),
        Value::Bool(b) => Ok(f64::from(*b)),
        Value::Text(text) => {
            let parsed = text_number(text).and_then(|text| {
                let lower = text.to_ascii_lowercase();
                let unsigned = lower.strip_prefix(['+', '-']).unwrap_or(&lower);
                if unsigned == "nan" {
                    return Some(f64::NAN.copysign(if lower.starts_with('-') { -1. } else { 1. }));
                }
                if unsigned == "inf" || unsigned == "infinity" {
                    return Some(if lower.starts_with('-') {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    });
                }
                text.parse::<f64>().ok()
            });
            parsed.ok_or_else(|| Error::ReferenceException {
                class: "ValueError",
                message: format!(
                    "could not convert string to float: {}",
                    crate::triton_specialization::python_str_repr(text)
                ),
            })
        }
        Value::Tensor(_) | Value::Quantized(_) => {
            let value = unsafe { rt.dequantize_args(value)? };
            let tensor = value.as_tensor()?;
            if tensor.numel() != 1 {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "only one element tensors can be converted to Python scalars".into(),
                });
            }
            rt.native.item_double(&tensor)
        }
        _ => Err(invalid()),
    }
}

fn surrogate_decode(mut bytes: &[u8]) -> Vec<u32> {
    let mut points = vec![];
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                points.extend(text.chars().map(u32::from));
                break;
            }
            Err(error) => {
                points.extend(
                    std::str::from_utf8(&bytes[..error.valid_up_to()])
                        .unwrap()
                        .chars()
                        .map(u32::from),
                );
                bytes = &bytes[error.valid_up_to()..];
                let invalid = error.error_len().unwrap_or(bytes.len());
                points.extend(
                    bytes[..invalid]
                        .iter()
                        .map(|byte| 0xdc00 + u32::from(*byte)),
                );
                bytes = &bytes[invalid..];
            }
        }
    }
    points
}
/// The original unbounded host value is consumed only by min(tensor_rows, n).
/// Tensor dimensions are signed int64; saturating that positive bound preserves
/// every possible consumed chunk without rejecting an original valid setting.
pub(crate) fn chunk_bound(bytes: &[u8]) -> Result<i64, Error> {
    let points = surrogate_decode(bytes);
    let invalid = || Error::ReferenceException {
        class: "ValueError",
        message: format!(
            "invalid literal for int() with base 10: {}",
            crate::triton_specialization::python_codepoint_str_repr(&points)
                .chars()
                .take(200)
                .collect::<String>()
        ),
    };
    let text = points
        .iter()
        .copied()
        .map(char::from_u32)
        .collect::<Option<String>>()
        .ok_or_else(invalid)?;
    let text = text.trim_matches(space);
    let (negative, text) = if let Some(t) = text.strip_prefix('-') {
        (true, t)
    } else {
        (false, text.strip_prefix('+').unwrap_or(text))
    };
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return Err(invalid());
    }
    let mut digits_count = 0;
    let mut number = 0_i64;
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' {
            if i == 0
                || i + 1 == chars.len()
                || decimal(chars[i - 1]).is_none()
                || decimal(chars[i + 1]).is_none()
            {
                return Err(invalid());
            }
        } else {
            let digit = decimal(*c).ok_or_else(invalid)?;
            digits_count += 1;
            number = number.saturating_mul(10).saturating_add(i64::from(digit));
        }
    }
    if digits_count > 4300 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!(
                "Exceeds the limit (4300 digits) for integer string conversion: value has {digits_count} digits; use sys.set_int_max_str_digits() to increase the limit"
            ),
        });
    }
    Ok(if negative { 1 } else { number.max(1) })
}
pub(crate) fn chunk_bound_from_process() -> Result<i64, Error> {
    use std::os::unix::ffi::OsStrExt;
    match std::env::var_os("COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N") {
        None => Ok(4096),
        Some(value) => chunk_bound(value.as_bytes()),
    }
}

/// Python int(value) at a source-native integer boundary. Wider values are
/// retained beyond every accepted uint64/int64/int32 range before rejection.
pub(crate) fn integer_value(rt: &Runtime, value: &Value) -> Result<i128, Error> {
    let float = |n: f64| -> Result<i128, Error> {
        if n.is_nan() {
            Err(Error::ReferenceException {
                class: "ValueError",
                message: "cannot convert float NaN to integer".into(),
            })
        } else if n.is_infinite() {
            Err(Error::ReferenceException {
                class: "OverflowError",
                message: "cannot convert float infinity to integer".into(),
            })
        } else {
            Ok(n.trunc() as i128)
        }
    };
    match value {
        Value::Int(n) => Ok(i128::from(*n)),
        Value::UInt(n) => Ok(i128::from(*n)),
        Value::Bool(n) => Ok(i128::from(*n)),
        Value::Float(n) => float(*n),
        Value::Text(text) => {
            let invalid = || Error::ReferenceException {
                class: "ValueError",
                message: format!(
                    "invalid literal for int() with base 10: {}",
                    crate::triton_specialization::python_str_repr(text)
                        .chars()
                        .take(200)
                        .collect::<String>()
                ),
            };
            let normalized = text_number(text).ok_or_else(invalid)?;
            let (negative, digits) = if let Some(v) = normalized.strip_prefix('-') {
                (true, v)
            } else {
                (false, normalized.strip_prefix('+').unwrap_or(&normalized))
            };
            if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
                return Err(invalid());
            }
            if digits.len() > 4300 {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: format!(
                        "Exceeds the limit (4300 digits) for integer string conversion: value has {} digits; use sys.set_int_max_str_digits() to increase the limit",
                        digits.len()
                    ),
                });
            }
            let number = digits.bytes().fold(0_i128, |n, c| {
                n.saturating_mul(10).saturating_add(i128::from(c - b'0'))
            });
            Ok(if negative { -number } else { number })
        }
        Value::Tensor(_) | Value::Quantized(_) => {
            let tensor = unsafe { rt.dequantize_args(value)? }.as_tensor()?;
            if tensor.numel() != 1 {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "only one element tensors can be converted to Python scalars".into(),
                });
            }
            if tensor.f_is_floating_point()? {
                float(rt.native.item_double(&tensor)?)
            } else {
                Ok(i128::from(tensor.f_reshape([])?.f_int64_value(&[])?))
            }
        }
        _ => Err(crate::execution_value::type_error(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            crate::backend_triton_fp8::typename(value)
        ))),
    }
}
