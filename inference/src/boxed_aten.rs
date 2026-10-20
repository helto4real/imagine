//! The pinned torch-sys IValue ABI, with ownership on every success/error path.
use crate::{
    execution_value::{Kwargs, Value},
    native::{Error, Kitchen},
};
use std::{
    ffi::{CStr, CString},
    ptr::NonNull,
};
use tch::{Device, Kind, Tensor};
use torch_sys::*;

pub(crate) struct Boxed(NonNull<CIValue>);
impl Drop for Boxed {
    fn drop(&mut self) {
        unsafe { ati_free(self.0.as_ptr()) }
    }
}
fn checked<T>(v: T) -> Result<T, Error> {
    let p = unsafe { get_and_reset_last_err() };
    if p.is_null() {
        Ok(v)
    } else {
        let msg = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        unsafe { libc::free(p.cast()) };
        Err(tch::TchError::Torch(msg).into())
    }
}
fn string(s: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|_| Error::BindingGap {
        backend: "aten".into(),
        operation: "embedded NUL string".into(),
    })
}
fn scalar_type(k: Kind) -> i64 {
    match k {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::ComplexHalf => 8,
        Kind::ComplexFloat => 9,
        Kind::ComplexDouble => 10,
        Kind::Bool => 11,
        Kind::QInt8 => 12,
        Kind::QUInt8 => 13,
        Kind::QInt32 => 14,
        Kind::BFloat16 => 15,
        Kind::Float8e5m2 => 23,
        Kind::Float8e4m3fn => 24,
        Kind::Float8e5m2fnuz => 25,
        Kind::Float8e4m3fnuz => 26,
    }
}
/// Original Torch list-constructor dtype promotion without tensor arithmetic.
pub(crate) fn promote_types(native: &Kitchen, left: Kind, right: Kind) -> Result<Kind, Error> {
    let result = unsafe {
        call(
            native,
            "aten::promote_types",
            "",
            &[Value::DType(left), Value::DType(right)],
            &vec![],
        )?
    };
    let code = match result {
        Value::Int(n) => n,
        Value::UInt(n) => i64::try_from(n)
            .map_err(|_| Error::Native("Invalid original ScalarType value".into()))?,
        _ => {
            return Err(Error::Native(
                "Original promote_types must return ScalarType".into(),
            ));
        }
    };
    [
        Kind::Uint8,
        Kind::Int8,
        Kind::Int16,
        Kind::Int,
        Kind::Int64,
        Kind::Half,
        Kind::Float,
        Kind::Double,
        Kind::ComplexHalf,
        Kind::ComplexFloat,
        Kind::ComplexDouble,
        Kind::Bool,
        Kind::QInt8,
        Kind::QUInt8,
        Kind::QInt32,
        Kind::BFloat16,
        Kind::Float8e5m2,
        Kind::Float8e4m3fn,
        Kind::Float8e5m2fnuz,
        Kind::Float8e4m3fnuz,
    ]
    .into_iter()
    .find(|kind| scalar_type(*kind) == code)
    .ok_or_else(|| Error::Native("Unknown frozen promote_types dtype".into()))
}
impl Boxed {
    fn own(p: *mut CIValue) -> Result<Self, Error> {
        Ok(Self(NonNull::new(checked(p)?).ok_or_else(|| {
            Error::Native("Null IValue without exception".into())
        })?))
    }
    fn pack(native: &Kitchen, v: &Value) -> Result<Self, Error> {
        unsafe {
            Self::own(match v {
                Value::None => ati_none(),
                Value::Bool(b) => ati_bool(i32::from(*b)),
                Value::Int(n) => ati_int(*n),
                Value::UInt(n) if *n <= i64::MAX as u64 => ati_int(*n as i64),
                Value::UInt(n) => native.ivalue_uint(*n)?.cast(),
                Value::Float(n) => ati_double(*n),
                Value::DType(k) => ati_int(scalar_type(*k)),
                Value::Device(d) => ati_device(match d {
                    Device::Cpu => -1,
                    Device::Cuda(i) => i32::try_from(*i)
                        .map_err(|_| Error::Native("CUDA index exceeds ABI".into()))?,
                    Device::Mps => -2,
                    Device::Vulkan => -3,
                }),
                Value::Text(s) => ati_string(string(s)?.as_ptr()),
                Value::Tensor(t) => ati_tensor(t.lock().unwrap().as_ptr().cast_mut()),
                Value::Tuple(items) | Value::List(items) => {
                    let owned = items
                        .iter()
                        .map(|v| Self::pack(native, v))
                        .collect::<Result<Vec<_>, _>>()?;
                    let pointers: Vec<_> = owned.iter().map(|x| x.0.as_ptr()).collect();
                    let len = i32::try_from(items.len())
                        .map_err(|_| Error::Native("IValue sequence too long".into()))?;
                    if matches!(v, Value::Tuple(_)) {
                        ati_tuple(pointers.as_ptr(), len)
                    } else {
                        ati_generic_list(pointers.as_ptr(), len)
                    }
                }
                Value::NamedTuple(object) => {
                    return Self::pack(
                        native,
                        &Value::Tuple(object.fields.iter().map(|(_, v)| v.clone()).collect()),
                    );
                }
                Value::Dict(v) => {
                    let mut items = Vec::new();
                    for (k, v) in v {
                        items.push(Self::pack(native, &Value::Text(k.clone()))?);
                        items.push(Self::pack(native, v)?);
                    }
                    let p: Vec<_> = items.iter().map(|v| v.0.as_ptr()).collect();
                    ati_generic_dict(
                        p.as_ptr(),
                        i32::try_from(v.len())
                            .map_err(|_| Error::Native("IValue dict too long".into()))?,
                    )
                }
                Value::Quantized(_) => {
                    return Err(Error::Native(
                        "Quantized value reached ordinary ATen binding".into(),
                    ));
                }
            })
        }
    }
    fn unpack(&self, native: &Kitchen) -> Result<Value, Error> {
        unsafe {
            let p = self.0.as_ptr();
            if let Some(value) = native.ivalue_get_uint(p.cast()) {
                return Ok(Value::UInt(value));
            }
            let tag = checked(ati_tag(p))?;
            Ok(match tag {
                0 => Value::None,
                1 => Value::tensor(Tensor::from_ptr(checked(ati_to_tensor(p))?)),
                2 => Value::Float(checked(ati_to_double(p))?),
                3 => Value::Int(checked(ati_to_int(p))?),
                4 => Value::Bool(checked(ati_to_bool(p))? != 0),
                9 => {
                    let p = checked(ati_to_string(p))?;
                    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
                    libc::free(p.cast());
                    Value::Text(s)
                }
                5 | 12 | 13 => {
                    let n = if tag == 5 {
                        checked(ati_tuple_length(p))?
                    } else {
                        checked(ati_length(p))?
                    };
                    let mut raw =
                        vec![std::ptr::null_mut(); n as usize * if tag == 13 { 2 } else { 1 }];
                    match tag {
                        5 => checked(ati_to_tuple(p, raw.as_mut_ptr(), n))?,
                        12 => checked(ati_to_generic_list(p, raw.as_mut_ptr(), n))?,
                        _ => checked(ati_to_generic_dict(p, raw.as_mut_ptr(), n))?,
                    };
                    let owned = raw
                        .into_iter()
                        .map(Self::own)
                        .collect::<Result<Vec<_>, _>>()?;
                    let values = owned
                        .iter()
                        .map(|v| v.unpack(native))
                        .collect::<Result<Vec<_>, _>>()?;
                    if tag == 5 {
                        Value::Tuple(values)
                    } else if tag == 12 {
                        Value::List(values)
                    } else {
                        let mut pairs = Vec::new();
                        let mut it = values.into_iter();
                        while let Some(key) = it.next() {
                            let Value::Text(key) = key else {
                                return Err(Error::BindingGap {
                                    backend: "aten".into(),
                                    operation: "non-string dictionary key".into(),
                                });
                            };
                            pairs.push((key, it.next().unwrap()));
                        }
                        Value::Dict(pairs)
                    }
                }
                6 => {
                    let n = checked(ati_length(p))?;
                    let mut a = vec![0; n as usize];
                    checked(ati_to_int_list(p, a.as_mut_ptr(), n))?;
                    Value::List(a.into_iter().map(Value::Int).collect())
                }
                7 => {
                    let n = checked(ati_length(p))?;
                    let mut a = vec![0.; n as usize];
                    checked(ati_to_double_list(p, a.as_mut_ptr(), n))?;
                    Value::List(a.into_iter().map(Value::Float).collect())
                }
                8 => {
                    let n = checked(ati_length(p))?;
                    let mut a = vec![0_i8; n as usize];
                    checked(ati_to_bool_list(p, a.as_mut_ptr(), n))?;
                    Value::List(a.into_iter().map(|x| Value::Bool(x != 0)).collect())
                }
                10 => {
                    let n = checked(ati_length(p))?;
                    let mut a = vec![std::ptr::null_mut(); n as usize];
                    checked(ati_to_tensor_list(p, a.as_mut_ptr(), n))?;
                    Value::List(
                        a.into_iter()
                            .map(|t| Value::tensor(Tensor::from_ptr(t)))
                            .collect(),
                    )
                }
                _ => {
                    return Err(Error::BindingGap {
                        backend: "aten".into(),
                        operation: format!("IValue return tag {tag}"),
                    });
                }
            })
        }
    }
}
/// # Safety
/// Inputs must satisfy the selected ATen operator contract, including any
/// explicitly unsafe native operator. All values have been dequantized.
pub unsafe fn call(
    native: &Kitchen,
    name: &str,
    overload: &str,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<Value, Error> {
    let a = args
        .iter()
        .map(|v| Boxed::pack(native, v))
        .collect::<Result<Vec<_>, _>>()?;
    let v = kwargs
        .iter()
        .map(|(_, v)| Boxed::pack(native, v))
        .collect::<Result<Vec<_>, _>>()?;
    let keys = kwargs
        .iter()
        .map(|(k, _)| string(k))
        .collect::<Result<Vec<_>, _>>()?;
    let ap: Vec<_> = a
        .iter()
        .map(|v| v.0.as_ptr().cast::<std::ffi::c_void>().cast_const())
        .collect();
    let vp: Vec<_> = v
        .iter()
        .map(|v| v.0.as_ptr().cast::<std::ffi::c_void>().cast_const())
        .collect();
    let kp: Vec<_> = keys.iter().map(|k| k.as_ptr()).collect();
    let raw = unsafe { native.aten_boxed_raw(&string(name)?, &string(overload)?, &ap, &kp, &vp)? };
    let value = Boxed::own(raw.cast())?.unpack(native)?;
    // THPVariable_Wrap reuses the existing Python object for a TensorImpl. Keep
    // that identity (and its object attributes), not merely shared storage.
    let mut identities = std::collections::BTreeMap::new();
    fn remember(
        native: &Kitchen,
        value: &Value,
        identities: &mut std::collections::BTreeMap<u64, Value>,
    ) {
        match value {
            Value::Tensor(t) => {
                identities
                    .entry(native.tensor_identity(&t.lock().unwrap()))
                    .or_insert_with(|| value.clone());
            }
            Value::List(v) | Value::Tuple(v) => {
                for v in v {
                    remember(native, v, identities);
                }
            }
            Value::Dict(v) => {
                for (_, v) in v {
                    remember(native, v, identities);
                }
            }
            _ => {}
        }
    }
    fn intern(
        native: &Kitchen,
        value: Value,
        identities: &mut std::collections::BTreeMap<u64, Value>,
    ) -> Value {
        match value {
            Value::Tensor(ref t) => {
                let key = native.tensor_identity(&t.lock().unwrap());
                identities.entry(key).or_insert(value).clone()
            }
            Value::List(v) => Value::List(
                v.into_iter()
                    .map(|v| intern(native, v, identities))
                    .collect(),
            ),
            Value::Tuple(v) => Value::Tuple(
                v.into_iter()
                    .map(|v| intern(native, v, identities))
                    .collect(),
            ),
            Value::Dict(v) => Value::Dict(
                v.into_iter()
                    .map(|(k, v)| (k, intern(native, v, identities)))
                    .collect(),
            ),
            other => other,
        }
    }
    for value in args.iter().chain(kwargs.iter().map(|(_, v)| v)) {
        remember(native, value, &mut identities);
    }
    Ok(intern(native, value, &mut identities))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsigned_ivalue_ownership_and_tag_cross_the_pinned_c10_abi_without_narrowing() {
        let native = unsafe {
            Kitchen::load(std::path::Path::new(
                &std::env::var("IMAGINE_KITCHEN_LIBRARY")
                    .expect("Explicit isolated native library required"),
            ))
            .unwrap()
        };
        for number in [
            0,
            1,
            i64::MAX as u64,
            1_u64 << 63,
            (1_u64 << 63) + 1,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let boxed = Boxed::pack(&native, &Value::UInt(number)).unwrap();
            match boxed.unpack(&native).unwrap() {
                Value::UInt(n) => assert_eq!(n, number),
                Value::Int(n) => assert_eq!(u64::try_from(n).unwrap(), number),
                _ => panic!("Original c10 integer tag required"),
            }
        }
    }
}
