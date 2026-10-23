//! Locked numeric repr mechanics. The caller supplies its interpreter's active
//! decimal digit limit; zero disables it, as in CPython.
use super::Number;
use crate::{native::Error, registry::python_float};
pub(super) fn repr(value: &Number, limit: usize) -> Result<String, Error> {
    Ok(match value {
        Number::Bool(v) => if *v { "True" } else { "False" }.into(),
        Number::Float(v) => python_float(*v),
        Number::Integer(bytes) => decimal(bytes, limit)?,
        Number::Complex(real, imaginary) => {
            let component = |v: f64| {
                let out = python_float(v);
                out.strip_suffix(".0").unwrap_or(&out).to_string()
            };
            let imaginary = component(*imaginary);
            if *real == 0.0 && !real.is_sign_negative() {
                format!("{imaginary}j")
            } else {
                let sign = if imaginary.starts_with('-') { "" } else { "+" };
                format!("({}{sign}{imaginary}j)", component(*real))
            }
        }
    })
}
fn decimal(bytes: &[u8], limit: usize) -> Result<String, Error> {
    let negative = bytes.last().is_some_and(|b| b & 128 != 0);
    let mut magnitude = bytes.to_vec();
    if negative {
        let mut carry = 1_u16;
        for b in &mut magnitude {
            let v = u16::from(!*b) + carry;
            *b = v as u8;
            carry = v >> 8;
        }
    }
    let mut limbs = vec![0_u32];
    for byte in magnitude.iter().rev() {
        let mut carry = u64::from(*byte);
        for limb in &mut limbs {
            let value = u64::from(*limb) * 256 + carry;
            *limb = (value % 1_000_000_000) as u32;
            carry = value / 1_000_000_000;
        }
        if carry != 0 {
            limbs.push(carry as u32);
        }
    }
    let mut text = limbs.pop().unwrap().to_string();
    for limb in limbs.iter().rev() {
        use std::fmt::Write;
        write!(text, "{limb:09}").unwrap();
    }
    if limit != 0 && text.len() > limit {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!(
                "Exceeds the limit ({limit} digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit"
            ),
        });
    }
    if negative {
        text.insert(0, '-');
    }
    Ok(text)
}
