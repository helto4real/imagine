//! Drain an operator error before destroying its temporary ATen Scalar.
use crate::native::Error;
use std::ffi::CStr;
use tch::Tensor;
pub(super) fn checked<T>(call: impl FnOnce() -> T) -> Result<T, Error> {
    let result = call();
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    if !error.is_null() {
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe {
            libc::free(error.cast());
        }
        return Err(tch::TchError::Torch(message).into());
    }
    Ok(result)
}
pub(super) enum Comparison {
    Less,
    Equal,
}
pub(super) fn compare_integer(
    tensor: &Tensor,
    value: i64,
    operation: Comparison,
) -> Result<Tensor, Error> {
    // tch's generated scalar temporary drops before it reads a failed
    // operator's error, which can panic in Drop instead of returning the error.
    let scalar = checked(|| unsafe { torch_sys::ats_int(value) })?;
    let result = checked(|| unsafe {
        let mut output = std::ptr::null_mut();
        match operation {
            Comparison::Less => {
                torch_sys::c_generated::atg_lt(&mut output, tensor.as_ptr().cast_mut(), scalar)
            }
            Comparison::Equal => {
                torch_sys::c_generated::atg_eq(&mut output, tensor.as_ptr().cast_mut(), scalar)
            }
        }
        output
    });
    unsafe {
        torch_sys::ats_free(scalar);
    }
    Ok(unsafe { Tensor::from_ptr(result?) })
}
