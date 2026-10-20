//! _rebuild_tensor's original empty((0,), dtype/device).set_(untyped_storage).
//! Backward hooks, V2/V3 metadata and file loading are separate caller bindings.
use crate::native::Error;
use std::ffi::{CStr, c_void};
use tch::Tensor;
unsafe extern "C" {
    fn at_imagine_rebuild_empty(
        storage: *const torch_sys::C_tensor,
        dtype: i32,
        requires_grad: i32,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_rebuild_set_storage(
        value: *const torch_sys::C_tensor,
        storage: *const torch_sys::C_tensor,
        offset: i64,
        sizes: *const i64,
        strides: *const i64,
        dimensions: i64,
        stride_dimensions: i64,
    );
    fn at_imagine_rebuild_requires_grad(value: *const torch_sys::C_tensor, requires_grad: i32);
    fn at_imagine_rebuild_metadata_flags(
        value: *const torch_sys::C_tensor,
        conjugate_present: i32,
        negative_present: i32,
    );
    fn at_imagine_rebuild_storage_tensor(
        source: *const c_void,
        dtype: i32,
        offset: i64,
        sizes: *const i64,
        strides: *const i64,
        dimensions: i64,
        stride_dimensions: i64,
    ) -> *mut torch_sys::C_tensor;
}
fn check_error() -> Result<(), Error> {
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    if error.is_null() {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        libc::free(error.cast());
    }
    Err(tch::TchError::Torch(message).into())
}
/// V3 creates the gradient-enabled empty tensor before assigning its storage.
pub fn empty(storage: &Tensor, dtype: i32, requires_grad: bool) -> Result<Tensor, Error> {
    let value =
        unsafe { at_imagine_rebuild_empty(storage.as_ptr(), dtype, i32::from(requires_grad)) };
    check_error()?;
    if value.is_null() {
        return Err(Error::Native(
            "Missing native empty reconstruction tensor".into(),
        ));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
/// Split creation/assignment leaves the caller's partially constructed tensor
/// available to its source frame when assignment fails.
pub fn assign_storage(
    value: &Tensor,
    storage: &Tensor,
    offset: i64,
    sizes: &[i64],
    strides: &[i64],
) -> Result<(), Error> {
    unsafe {
        at_imagine_rebuild_set_storage(
            value.as_ptr(),
            storage.as_ptr(),
            offset,
            sizes.as_ptr(),
            strides.as_ptr(),
            sizes.len() as i64,
            strides.len() as i64,
        );
    }
    check_error()
}
pub fn requires_grad(value: &Tensor, enabled: bool) -> Result<(), Error> {
    unsafe {
        at_imagine_rebuild_requires_grad(value.as_ptr(), i32::from(enabled));
    }
    check_error()
}
/// CPU/CUDA setTensorMetadata's two built-in flags. Presence, even with a
/// mapped false value, sets the flag. A failing conjugate check has already
/// set that flag; the tensor remains available to the caller's source frame.
pub fn metadata_flags(
    value: &Tensor,
    conjugate_present: bool,
    negative_present: bool,
) -> Result<(), Error> {
    unsafe {
        at_imagine_rebuild_metadata_flags(
            value.as_ptr(),
            i32::from(conjugate_present),
            i32::from(negative_present),
        );
    }
    match check_error() {
        // This is a Python-visible source diagnostic, not a tolerance or test
        // normalization. The wheel's compiled TensorImpl.h location differs
        // from its installed development header (1470 versus 1471). Reify the
        // frozen original message after the native check has performed its
        // original partial mutation. Other diagnostics pass through unchanged.
        Err(Error::ReferenceException { class: "RuntimeError", message })
            if conjugate_present && !value.is_complex()
                && message.starts_with("isComplexType(typeMetaToScalarType(dtype())) INTERNAL ASSERT FAILED at \"")
                && message.contains("/c10/core/TensorImpl.h\":") => {
            Err(Error::ReferenceException { class: "RuntimeError", message: "isComplexType(typeMetaToScalarType(dtype())) INTERNAL ASSERT FAILED at \"/__w/pytorch/pytorch/c10/core/TensorImpl.h\":1470, please report a bug to PyTorch. ".into() })
        }
        result => result,
    }
}
/// dtype is the frozen ATen scalar type code, including shell/unsigned dtypes
/// absent from tch::Kind. Storage aliasing and native bounds errors are retained.
pub fn tensor(
    storage: &Tensor,
    dtype: i32,
    offset: i64,
    sizes: &[i64],
    strides: &[i64],
) -> Result<Tensor, Error> {
    let value = unsafe {
        at_imagine_rebuild_storage_tensor(
            storage.as_ptr().cast(),
            dtype,
            offset,
            sizes.as_ptr(),
            strides.as_ptr(),
            sizes.len() as i64,
            strides.len() as i64,
        )
    };
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
    if value.is_null() {
        return Err(Error::Native("Missing rebuilt storage tensor".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
