//! Native immutable numeric key objects; payload and source object identity are
//! separate. Arbitrary-width integers retain canonical signed little-endian bytes.
use super::{MetadataObject, MetadataValue, Options};
use crate::{execution_value::Value, native::Error, python_hash};
use std::any::Any;
#[path = "options_number_format.rs"]
mod formatting;
#[derive(Clone)]
enum Number {
    Bool(bool),
    Integer(Vec<u8>),
    Float(f64),
    Complex(f64, f64),
}
pub struct SourceNumber {
    number: Number,
}
thread_local! {
 static SMALL:Vec<Options>=(-5_i64..=256).map(|v|SourceNumber::wrap(Number::Integer(canonical(v.to_le_bytes().to_vec())))).collect();
 static FALSE:Options=SourceNumber::wrap(Number::Bool(false));
 static TRUE:Options=SourceNumber::wrap(Number::Bool(true));
}
fn canonical(mut bytes: Vec<u8>) -> Vec<u8> {
    while bytes.len() > 1 {
        let n = bytes.len();
        if (bytes[n - 1] == 0 && bytes[n - 2] & 128 == 0)
            || (bytes[n - 1] == 255 && bytes[n - 2] & 128 != 0)
        {
            bytes.pop();
        } else {
            break;
        }
    }
    if bytes == [0] {
        bytes.clear();
    }
    bytes
}
fn small(bytes: &[u8]) -> Option<i64> {
    if bytes.len() > 8 {
        return None;
    }
    let mut out = [if bytes.last().is_some_and(|b| b & 128 != 0) {
        255
    } else {
        0
    }; 8];
    out[..bytes.len()].copy_from_slice(bytes);
    Some(i64::from_le_bytes(out))
}
impl SourceNumber {
    fn wrap(number: Number) -> Options {
        MetadataObject::wrap(Self { number })
    }
    pub fn boolean(value: bool) -> Options {
        if value {
            TRUE.with(Clone::clone)
        } else {
            FALSE.with(Clone::clone)
        }
    }
    /// The reference PyLong_FromLong/_PyLong_FromByteArray cache is shared by
    /// opcode and LONG1 constructors; aliases clone the returned Options owner.
    pub fn integer(bytes: &[u8]) -> Options {
        let bytes = canonical(bytes.to_vec());
        if let Some(v) = small(&bytes).filter(|v| (-5..=256).contains(v)) {
            return SMALL.with(|a| a[(v + 5) as usize].clone());
        }
        Self::wrap(Number::Integer(bytes))
    }
    pub fn from_i64(value: i64) -> Options {
        Self::integer(&value.to_le_bytes())
    }
    pub fn float(value: f64) -> Options {
        Self::wrap(Number::Float(value))
    }
    /// Already materialized Py_complex payload; this does not implement complex()
    /// argument conversion or complex arithmetic.
    pub fn complex_parts(real: f64, imaginary: f64) -> Options {
        Self::wrap(Number::Complex(real, imaginary))
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(v) = value else {
            return None;
        };
        MetadataObject::get(v.as_ref())
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn hash(&self) -> i64 {
        let owner = python_hash::identity_hash_address(self.identity_token() as usize);
        match &self.number {
            Number::Bool(b) => i64::from(*b),
            Number::Integer(v) => python_hash::hash_signed_integer_bytes(v),
            Number::Float(v) => python_hash::hash_float(*v, owner),
            Number::Complex(r, i) => python_hash::hash_complex(*r, *i, owner),
        }
    }
    pub fn is_exact_integer(&self) -> bool {
        matches!(self.number, Number::Integer(_))
    }
    pub fn integer_bytes(&self) -> Option<&[u8]> {
        if let Number::Integer(v) = &self.number {
            Some(v)
        } else {
            None
        }
    }
    pub fn float_bits(&self) -> Option<u64> {
        if let Number::Float(v) = self.number {
            Some(v.to_bits())
        } else {
            None
        }
    }
    pub fn complex_bits(&self) -> Option<(u64, u64)> {
        if let Number::Complex(r, i) = self.number {
            Some((r.to_bits(), i.to_bits()))
        } else {
            None
        }
    }
    pub fn equal(&self, other: &Self) -> bool {
        equal(&self.number, &other.number)
    }
    /// Exact builtin PyNumber_Float conversion. Integer conversion is
    /// independent of the decimal-repr digit limit and rejects overflow.
    pub fn to_float(&self) -> Result<f64, Error> {
        match &self.number {
            Number::Bool(v) => Ok(f64::from(*v)),
            Number::Float(v) => Ok(*v),
            Number::Integer(_) => {
                let value = self
                    .repr_with_limit(0)?
                    .parse::<f64>()
                    .map_err(|_| Error::Native("invalid generated integer decimal".into()))?;
                if value.is_infinite() {
                    Err(Error::ReferenceException {
                        class: "OverflowError",
                        message: "int too large to convert to float".into(),
                    })
                } else {
                    Ok(value)
                }
            }
            Number::Complex(..) => Err(Error::ReferenceException {
                class: "TypeError",
                message: "float() argument must be a string or a real number, not 'complex'".into(),
            }),
        }
    }
    pub fn repr_with_limit(&self, limit: usize) -> Result<String, Error> {
        formatting::repr(&self.number, limit)
    }
    pub fn truth(&self) -> bool {
        match &self.number {
            Number::Bool(v) => *v,
            Number::Integer(v) => !v.is_empty(),
            Number::Float(v) => *v != 0.0,
            Number::Complex(r, i) => *r != 0.0 || *i != 0.0,
        }
    }
}
fn integer(number: &Number) -> Option<Vec<u8>> {
    match number {
        Number::Bool(b) => Some(if *b { vec![1] } else { vec![] }),
        Number::Integer(v) => Some(v.clone()),
        _ => None,
    }
}
fn real(number: &Number) -> Option<f64> {
    match number {
        Number::Float(v) => Some(*v),
        Number::Complex(r, i) if *i == 0.0 => Some(*r),
        _ => None,
    }
}
/// Finite integral binary64 -> exact integer bytes, without truncating to i128
/// or rounding a large integer to float for comparison.
fn integral_float(value: f64) -> Option<Vec<u8>> {
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(vec![]);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32 - 1023 - 52;
    let mut significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let shift = if exponent < 0 {
        significand >>= -exponent;
        0
    } else {
        exponent as usize
    };
    let mut bytes = vec![0_u8; shift / 8];
    let mut carry = 0_u16;
    for byte in significand.to_le_bytes() {
        let v = (u16::from(byte) << (shift % 8)) | carry;
        bytes.push(v as u8);
        carry = v >> 8;
    }
    if carry != 0 {
        bytes.push(carry as u8);
    }
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    if bytes.last().is_some_and(|v| v & 128 != 0) {
        bytes.push(0);
    }
    if value.is_sign_negative() {
        let mut carry = 1_u16;
        for b in &mut bytes {
            let v = u16::from(!*b) + carry;
            *b = v as u8;
            carry = v >> 8;
        }
    }
    Some(canonical(bytes))
}
fn equal(a: &Number, b: &Number) -> bool {
    if let (Some(a), Some(b)) = (integer(a), integer(b)) {
        return a == b;
    }
    if let (Some(a), Some(b)) = (integer(a), real(b)) {
        return integral_float(b).is_some_and(|b| a == b);
    }
    if let (Some(a), Some(b)) = (real(a), integer(b)) {
        return integral_float(a).is_some_and(|a| a == b);
    }
    match (a, b) {
        (Number::Float(a), Number::Float(b)) => a == b,
        (Number::Complex(ar, ai), Number::Complex(br, bi)) => ar == br && ai == bi,
        (Number::Float(a), Number::Complex(br, bi)) => a == br && *bi == 0.0,
        (Number::Complex(ar, ai), Number::Float(b)) => ar == b && *ai == 0.0,
        _ => false,
    }
}
impl MetadataValue for SourceNumber {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        match &self.number {
            Number::Bool(v) => Ok(Value::Bool(*v)),
            Number::Float(v) => Ok(Value::Float(*v)),
            Number::Integer(v) => small(v).map(Value::Int).ok_or_else(|| Error::BindingGap {
                backend: "numeric_metadata".into(),
                operation: "arbitrary-width integer requires its typed native consumer".into(),
            }),
            Number::Complex(..) => Err(Error::BindingGap {
                backend: "numeric_metadata".into(),
                operation: "complex payload requires its typed native consumer".into(),
            }),
        }
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|v| std::ptr::eq(self, v))
    }
    fn type_name(&self) -> &str {
        match self.number {
            Number::Bool(_) => "bool",
            Number::Integer(_) => "int",
            Number::Float(_) => "float",
            Number::Complex(..) => "complex",
        }
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(SourceNumber::truth(self))
    }
}

/// Concrete numeric key provider for the existing native CPython dictionary.
/// The complete checkpoint provider composes this with Unicode/tuple/object keys.
pub struct NumericKeyBindings;
fn numeric(value: &Options) -> Result<&SourceNumber, Error> {
    SourceNumber::get(value).ok_or_else(|| Error::BindingGap {
        backend: "numeric_metadata".into(),
        operation: "non-numeric key requires its original key provider".into(),
    })
}
impl super::DictionaryKeyBindings for NumericKeyBindings {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        Ok(numeric(key)?.hash())
    }
    fn identity_token(&self, key: &Options) -> u64 {
        SourceNumber::get(key).map_or(0, SourceNumber::identity_token)
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        Ok(numeric(a)?.equal(numeric(b)?))
    }
    fn exact_unicode(&self, _key: &Options) -> bool {
        false
    }
}

/// Source None singleton. Its locked CPython hash is a fixed constant, not
/// PyObject_GenericHash; keep a real native owner for alias-aware consumers.
pub struct SourceNone {
    _marker: u8,
}
thread_local! {static NONE:Options=MetadataObject::wrap(SourceNone{_marker:0});}
impl SourceNone {
    pub fn value() -> Options {
        NONE.with(Clone::clone)
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(v) = value else {
            return None;
        };
        MetadataObject::get(v.as_ref())
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn hash(&self) -> i64 {
        0xFCA86420
    }
}
impl MetadataValue for SourceNone {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Ok(Value::None)
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|v| std::ptr::eq(self, v))
    }
    fn type_name(&self) -> &str {
        "NoneType"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(false)
    }
}
