//! Source async-error discard. Exception class hierarchy remains an explicit
//! provider, while the native C10 adapter recognizes its builtin classes.
use crate::native::Error;
pub trait OomBindings<E> {
    fn oom_exception(&mut self, error: &E) -> Result<bool, Error>;
    fn accelerator_exception(&mut self, error: &E) -> Result<bool, Error>;
    fn oom_code(&mut self, error: &E) -> Result<bool, Error>;
    fn oom_text(&mut self, error: &E) -> Result<bool, Error>;
    fn discard_async_error(&mut self) -> Result<(), Error>;
}
pub fn is_oom<E>(error: &E, bindings: &mut dyn OomBindings<E>) -> Result<bool, Error> {
    if bindings.oom_exception(error)? {
        return Ok(true);
    }
    if bindings.accelerator_exception(error)?
        && (bindings.oom_code(error)? || bindings.oom_text(error)?)
    {
        bindings.discard_async_error()?;
        return Ok(true);
    }
    Ok(false)
}
pub fn raise_non_oom(error: Error, bindings: &mut dyn OomBindings<Error>) -> Result<(), Error> {
    if !is_oom(&error, bindings)? {
        return Err(error);
    }
    Ok(())
}
/// Typed C10 exceptions produced by the native bridges. General source class
/// hierarchies/properties use OomBindings directly instead of guessing by text.
pub struct NativeOomBindings<F>(pub F);
impl<F: FnMut() -> Result<(), Error>> OomBindings<Error> for NativeOomBindings<F> {
    fn oom_exception(&mut self, error: &Error) -> Result<bool, Error> {
        Ok(matches!(
            error.root(),
            Error::ReferenceException {
                class: "OutOfMemoryError",
                ..
            }
        ))
    }
    fn accelerator_exception(&mut self, error: &Error) -> Result<bool, Error> {
        Ok(matches!(
            error.root(),
            Error::AcceleratorException { .. }
                | Error::ReferenceException {
                    class: "AcceleratorError",
                    ..
                }
        ))
    }
    fn oom_code(&mut self, error: &Error) -> Result<bool, Error> {
        Ok(matches!(
            error.root(),
            Error::AcceleratorException { code: 2, .. }
        ))
    }
    fn oom_text(&mut self, error: &Error) -> Result<bool, Error> {
        Ok(match error.root() {
            Error::AcceleratorException { message, .. }
            | Error::ReferenceException {
                class: "AcceleratorError",
                message,
            } => message.to_lowercase().contains("out of memory"),
            _ => false,
        })
    }
    fn discard_async_error(&mut self) -> Result<(), Error> {
        (self.0)()
    }
}
pub trait AsyncErrorDiscard {
    type Tensor;
    type Device;
    fn torch_device(&mut self) -> Result<Self::Device, Error>;
    fn byte_tensor(&mut self, device: Self::Device) -> Result<Self::Tensor, Error>;
    fn add(&mut self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Error>;
    fn synchronize(&mut self) -> Result<(), Error>;
    fn runtime_error(&self, error: &Error) -> bool;
}
pub fn discard_cuda_async_error<B: AsyncErrorDiscard>(bindings: &mut B) -> Result<(), Error> {
    // Retain a,b,sum through the handler; source frame clears them in this order.
    struct Frame<T> {
        a: Option<T>,
        b: Option<T>,
        sum: Option<T>,
    }
    let mut frame = Frame {
        a: None,
        b: None,
        sum: None,
    };
    let result = (|| {
        let device = bindings.torch_device()?;
        frame.a = Some(bindings.byte_tensor(device)?);
        let device = bindings.torch_device()?;
        frame.b = Some(bindings.byte_tensor(device)?);
        frame.sum = Some(bindings.add(frame.a.as_ref().unwrap(), frame.b.as_ref().unwrap())?);
        bindings.synchronize()
    })();
    match result {
        Err(error) if bindings.runtime_error(&error) => Ok(()),
        result => result,
    }
}
pub fn native_runtime_error(error: &Error) -> bool {
    matches!(
        error.root(),
        Error::Native(_)
            | Error::Torch(tch::TchError::Torch(_))
            | Error::AcceleratorException { .. }
            | Error::ReferenceException {
                class: "RuntimeError"
                    | "_LinAlgError"
                    | "OutOfMemoryError"
                    | "AcceleratorError"
                    | "NotImplementedError"
                    | "RecursionError",
                ..
            }
    )
}

/// Fixed device adapter for the admitted synchronous Linux CUDA profiles.
/// Ambient/custom source device selection uses AsyncErrorDiscard directly.
pub fn discard_native_cuda_async_error(
    native: &crate::native::Kitchen,
    device: tch::Device,
) -> Result<(), Error> {
    use tch::{Device, Tensor};
    struct Discard<'a> {
        native: &'a crate::native::Kitchen,
        device: Device,
    }
    impl AsyncErrorDiscard for Discard<'_> {
        type Tensor = Tensor;
        type Device = Device;
        fn torch_device(&mut self) -> Result<Device, Error> {
            Ok(self.device)
        }
        fn byte_tensor(&mut self, device: Device) -> Result<Tensor, Error> {
            Ok(Tensor::f_from_slice(&[1_u8])?.f_to_device(device)?)
        }
        fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            Ok(a.f_add(b)?)
        }
        fn synchronize(&mut self) -> Result<(), Error> {
            if let Device::Cuda(index) = self.device {
                self.native.synchronize_device(index)?;
            }
            Ok(())
        }
        fn runtime_error(&self, error: &Error) -> bool {
            native_runtime_error(error)
        }
    }
    discard_cuda_async_error(&mut Discard { native, device })
}
