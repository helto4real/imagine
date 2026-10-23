//! Source-mapped builtin complex constructor after the caller's positional
//! sequence expansion. Generic user/subclass conversion hooks remain explicit.
use crate::execution_value::Value;
use crate::patching::options::{MetadataObject, MetadataValue};
use crate::{
    native::Error,
    patching::options::{
        GeneralMapping, Options, SourceByteArray, SourceBytes, SourceNone, SourceNumber as N,
        SourceText,
    },
};
use std::any::Any;

/// The actual process-owned builtin class, distinct from any registry key or
/// alias referring to it. Registry authorization always precedes invocation.
pub struct ComplexClass {
    _marker: u8,
}
thread_local! { static COMPLEX_CLASS: Options = MetadataObject::wrap(ComplexClass { _marker: 0 }); }
impl ComplexClass {
    pub fn value() -> Options {
        COMPLEX_CLASS.with(Clone::clone)
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn representation(&self) -> &'static str {
        "<class 'complex'>"
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    /// Invocation after the actual source caller has expanded positional args.
    pub fn call_positional(&self, args: &[Options]) -> Result<Options, Error> {
        complex(args)
    }
}
impl MetadataValue for ComplexClass {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(Error::BindingGap {
            backend: "builtin_complex_class".into(),
            operation: "class object requires its typed caller, not a scalar snapshot".into(),
        })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|v| std::ptr::eq(self, v))
    }
    fn type_name(&self) -> &str {
        "type"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(true)
    }
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap() -> Error {
    Error::BindingGap {
        backend: "builtin_complex".into(),
        operation: "non-builtin class requires its __complex__/__float__/__index__ protocols"
            .into(),
    }
}
fn invalid(value: &Options, first: bool) -> Error {
    let name = if SourceNone::get(value).is_some() {
        "NoneType"
    } else if SourceBytes::get(value).is_some() {
        "bytes"
    } else if SourceByteArray::get(value).is_some() {
        "bytearray"
    } else if super::dtype_objects::SourceDtype::get(value).is_some() {
        "torch.dtype"
    } else if crate::patching::options::GeneralSet::get(value).is_some() {
        "set"
    } else if crate::patching::options::FrozenSet::get(value).is_some() {
        "frozenset"
    } else if ComplexClass::get(value).is_some()
        || super::builtin_set_ops::SetClass::get(value).is_some()
    {
        "type"
    } else if matches!(value, Options::List(_)) {
        "list"
    } else if matches!(value, Options::Tuple(_)) {
        "tuple"
    } else if GeneralMapping::get(value).is_some() {
        "dict"
    } else {
        return gap();
    };
    error(
        "TypeError",
        format!(
            "complex() {} argument must be {}number, not '{name}'",
            if first { "first" } else { "second" },
            if first { "a string or a " } else { "a " }
        ),
    )
}
pub fn complex(args: &[Options]) -> Result<Options, Error> {
    if args.len() > 2 {
        return Err(error(
            "TypeError",
            format!("complex() takes at most 2 arguments ({} given)", args.len()),
        ));
    }
    let zero = N::from_i64(0);
    let real = args.first().unwrap_or(&zero);
    let imag = args.get(1);
    if imag.is_none() && N::get(real).is_some_and(|v| v.complex_bits().is_some()) {
        return Ok(real.clone());
    }
    if let Some(text) = SourceText::get(real) {
        if imag.is_some() {
            return Err(error(
                "TypeError",
                "complex() can't take second arg if first is a string",
            ));
        }
        let (r, i) = parse(text.codepoints())?;
        return Ok(N::complex_parts(r, i));
    }
    if imag.is_some_and(|v| SourceText::get(v).is_some()) {
        return Err(error("TypeError", "complex() second arg can't be a string"));
    }
    let real_number = N::get(real).ok_or_else(|| invalid(real, true))?;
    let imag_number = imag
        .map(|v| N::get(v).ok_or_else(|| invalid(v, false)))
        .transpose()?;
    let real_complex = real_number.complex_bits();
    let (mut r, real_i) = match real_complex {
        Some((r, i)) => (f64::from_bits(r), f64::from_bits(i)),
        None => (real_number.to_float()?, 0.0),
    };
    let mut i = match imag_number {
        Some(n) => match n.complex_bits() {
            Some((r, _)) => f64::from_bits(r),
            None => n.to_float()?,
        },
        None => real_i,
    };
    if let Some((_, other_i)) = imag_number.and_then(N::complex_bits) {
        r = crate::python_float_ops::sub(r, f64::from_bits(other_i))?;
    }
    if real_complex.is_some() && imag_number.is_some() {
        i = crate::python_float_ops::add(i, real_i)?;
    }
    Ok(N::complex_parts(r, i))
}
fn malformed() -> Error {
    error("ValueError", "complex() arg is a malformed string")
}
fn space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c')
}
fn parse(points: &[u32]) -> Result<(f64, f64), Error> {
    let mut ascii = Vec::with_capacity(points.len());
    if points.iter().all(|p| *p < 128) {
        ascii.extend(points.iter().map(|p| *p as u8));
    } else {
        for &point in points {
            if point < 127 {
                ascii.push(point as u8);
            } else if let Some(c) = char::from_u32(point) {
                if c.is_whitespace() {
                    ascii.push(b' ');
                } else if let Some(d) = crate::python_numbers::decimal(c) {
                    ascii.push(b'0' + d);
                } else {
                    ascii.push(b'?');
                    break;
                }
            } else {
                ascii.push(b'?');
                break;
            }
        }
    }
    let first_nul = ascii.iter().position(|b| *b == 0).unwrap_or(ascii.len());
    let underscores = ascii[..first_nul].contains(&b'_');
    let mut data = vec![];
    if underscores {
        let underscore_error = || {
            error(
                "ValueError",
                format!(
                    "could not convert string to complex: {}",
                    crate::triton_specialization::python_codepoint_str_repr(points)
                ),
            )
        };
        let mut previous = 0_u8;
        for &byte in &ascii[..first_nul] {
            if byte == b'_' {
                if !previous.is_ascii_digit() {
                    return Err(underscore_error());
                }
            } else {
                data.push(byte);
                if previous == b'_' && !byte.is_ascii_digit() {
                    return Err(underscore_error());
                }
            }
            previous = byte;
        }
        if previous == b'_' || first_nul != ascii.len() {
            return Err(underscore_error());
        }
    } else {
        data = ascii;
    }
    if data.contains(&0) {
        return Err(malformed());
    }
    let (mut pos, mut r, mut i) = (0, 0.0, 0.0);
    while pos < data.len() && space(data[pos]) {
        pos += 1;
    }
    let bracket = data.get(pos) == Some(&b'(');
    if bracket {
        pos += 1;
        while pos < data.len() && space(data[pos]) {
            pos += 1;
        }
    }
    if let Some((number, end)) = prefix(&data, pos) {
        pos = end;
        match data.get(pos) {
            Some(b'+' | b'-') => {
                r = number;
                if let Some((number, end)) = prefix(&data, pos) {
                    i = number;
                    pos = end;
                } else {
                    i = if data[pos] == b'+' { 1.0 } else { -1.0 };
                    pos += 1;
                }
                if !matches!(data.get(pos), Some(b'j' | b'J')) {
                    return Err(malformed());
                }
                pos += 1;
            }
            Some(b'j' | b'J') => {
                i = number;
                pos += 1;
            }
            _ => r = number,
        }
    } else {
        i = match data.get(pos) {
            Some(b'+') => {
                pos += 1;
                1.0
            }
            Some(b'-') => {
                pos += 1;
                -1.0
            }
            _ => 1.0,
        };
        if !matches!(data.get(pos), Some(b'j' | b'J')) {
            return Err(malformed());
        }
        pos += 1;
    }
    while pos < data.len() && space(data[pos]) {
        pos += 1;
    }
    if bracket {
        if data.get(pos) != Some(&b')') {
            return Err(malformed());
        }
        pos += 1;
        while pos < data.len() && space(data[pos]) {
            pos += 1;
        }
    }
    if pos != data.len() {
        return Err(malformed());
    }
    Ok((r, i))
}
fn prefix(data: &[u8], start: usize) -> Option<(f64, usize)> {
    let mut pos = start;
    if matches!(data.get(pos), Some(b'+' | b'-')) {
        pos += 1;
    }
    let negative = data.get(start) == Some(&b'-');
    let tail = &data[pos..];
    for name in [b"infinity".as_slice(), b"inf".as_slice(), b"nan".as_slice()] {
        if tail.len() >= name.len() && tail[..name.len()].eq_ignore_ascii_case(name) {
            let v = if name == b"nan" {
                f64::NAN
            } else {
                f64::INFINITY
            };
            return Some((
                v.copysign(if negative { -1.0 } else { 1.0 }),
                pos + name.len(),
            ));
        }
    }
    let first = pos;
    while data.get(pos).is_some_and(u8::is_ascii_digit) {
        pos += 1;
    }
    let mut count = pos - first;
    if data.get(pos) == Some(&b'.') {
        pos += 1;
        let first = pos;
        while data.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        count += pos - first;
    }
    if count == 0 {
        return None;
    }
    if matches!(data.get(pos), Some(b'e' | b'E')) {
        let exponent = pos;
        pos += 1;
        if matches!(data.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let first = pos;
        while data.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        if pos == first {
            pos = exponent;
        }
    }
    let text = std::str::from_utf8(&data[start..pos]).ok()?;
    Some((text.parse().ok()?, pos))
}
