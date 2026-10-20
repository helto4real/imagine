//! Source Tensor.item host scalar categories, without a Python runtime.
use crate::{execution_value::Value, native::Error};
use std::{
    ffi::{CStr, c_void},
    os::raw::c_int,
};
use tch::Tensor;
unsafe extern "C" {
    fn at_imagine_tensor_item(
        value: *const c_void,
        kind: *mut c_int,
        integer: *mut i64,
        unsigned_integer: *mut u64,
        real: *mut f64,
        imaginary: *mut f64,
    );
    fn at_imagine_tolist_data(value: *const c_void) -> *mut c_void;
    fn at_imagine_tolist_scalar(
        value: *const c_void,
        offset: i64,
        kind: *mut c_int,
        integer: *mut i64,
        unsigned_integer: *mut u64,
        real: *mut f64,
        imaginary: *mut f64,
    );
}
#[derive(Debug)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex { real: f64, imaginary: f64 },
}
fn read_scalar(
    call: impl FnOnce(*mut c_int, *mut i64, *mut u64, *mut f64, *mut f64),
) -> Result<Scalar, Error> {
    let (mut kind, mut integer, mut unsigned_integer, mut real, mut imaginary) = (-1, 0, 0, 0., 0.);
    call(
        &mut kind,
        &mut integer,
        &mut unsigned_integer,
        &mut real,
        &mut imaginary,
    );
    check_error()?;
    Ok(match kind {
        0 => Scalar::Bool(integer != 0),
        1 => Scalar::Int(integer),
        2 => {
            if unsigned_integer <= i64::MAX as u64 {
                Scalar::Int(unsigned_integer as i64)
            } else {
                Scalar::UInt(unsigned_integer)
            }
        }
        3 => Scalar::Float(real),
        4 => Scalar::Complex { real, imaginary },
        _ => return Err(Error::Native("Missing native item scalar category".into())),
    })
}
fn check_error() -> Result<(), Error> {
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    if !error.is_null() {
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe {
            libc::free(error.cast());
        }
        if message.starts_with("[imagine-binding-gap]") {
            return Err(Error::BindingGap {
                backend: "host scalar".into(),
                operation: "tolist source tensor subclass".into(),
            });
        }
        let error: Error = tch::TchError::Torch(message).into();
        return Err(match error {
            Error::ReferenceException {
                class: "RuntimeError",
                ref message,
            } if message.starts_with("[imagine-binding-gap]") => Error::BindingGap {
                backend: "host scalar".into(),
                operation: "tolist source tensor subclass".into(),
            },
            Error::Torch(tch::TchError::Torch(message)) => Error::ReferenceException {
                class: "RuntimeError",
                message: message
                    .split("\nException raised from")
                    .next()
                    .unwrap()
                    .trim_end()
                    .to_owned(),
            },
            error => error,
        });
    }
    Ok(())
}
pub fn item(value: &Tensor) -> Result<Scalar, Error> {
    read_scalar(|kind, integer, unsigned_integer, real, imaginary| unsafe {
        at_imagine_tensor_item(
            value.as_ptr().cast(),
            kind,
            integer,
            unsigned_integer,
            real,
            imaginary,
        )
    })
}
pub fn scalar_value(value: &Tensor) -> Result<Value, Error> {
    Ok(match item(value)? {
        Scalar::Bool(value) => Value::Bool(value),
        Scalar::Int(value) => Value::Int(value),
        Scalar::UInt(value) => Value::UInt(value),
        Scalar::Float(value) => Value::Float(value),
        Scalar::Complex { .. } => {
            return Err(Error::BindingGap {
                backend: "host scalar".into(),
                operation: "complex scalar value binding".into(),
            });
        }
    })
}
#[derive(Debug)]
pub enum HostValue {
    Scalar(Scalar),
    List(Vec<HostValue>),
}
impl HostValue {
    pub fn into_value(self) -> Result<Value, Error> {
        Ok(match self {
            Self::List(values) => Value::List(
                values
                    .into_iter()
                    .map(Self::into_value)
                    .collect::<Result<_, _>>()?,
            ),
            Self::Scalar(Scalar::Bool(value)) => Value::Bool(value),
            Self::Scalar(Scalar::Int(value)) => Value::Int(value),
            Self::Scalar(Scalar::UInt(value)) => Value::UInt(value),
            Self::Scalar(Scalar::Float(value)) => Value::Float(value),
            Self::Scalar(Scalar::Complex { .. }) => {
                return Err(Error::BindingGap {
                    backend: "host scalar".into(),
                    operation: "complex scalar value binding".into(),
                });
            }
        })
    }
}
pub fn tolist(value: &Tensor) -> Result<HostValue, Error> {
    let data = unsafe { at_imagine_tolist_data(value.as_ptr().cast()) };
    check_error()?;
    if data.is_null() {
        return Err(Error::Native("Missing native tolist data".into()));
    }
    let data = unsafe { Tensor::from_ptr(data.cast()) };
    fn walk(
        data: &Tensor,
        sizes: &[i64],
        strides: &[i64],
        dim: usize,
        offset: i64,
    ) -> Result<HostValue, Error> {
        if dim == sizes.len() {
            return Ok(HostValue::Scalar(read_scalar(
                |kind, integer, unsigned_integer, real, imaginary| unsafe {
                    at_imagine_tolist_scalar(
                        data.as_ptr().cast(),
                        offset,
                        kind,
                        integer,
                        unsigned_integer,
                        real,
                        imaginary,
                    )
                },
            )?));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(sizes[dim] as usize)
            .map_err(|_| Error::ReferenceException {
                class: "MemoryError",
                message: String::new(),
            })?;
        for index in 0..sizes[dim] {
            output.push(walk(
                data,
                sizes,
                strides,
                dim + 1,
                offset + index * strides[dim],
            )?);
        }
        Ok(HostValue::List(output))
    }
    walk(&data, &data.size(), &data.stride(), 0, 0)
}
