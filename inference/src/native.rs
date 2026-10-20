//! Audited ownership boundary to the standalone Kitchen/ATen C ABI.
use std::any::Any;
use std::os::unix::ffi::OsStrExt;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
    sync::Arc,
};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub enum Error {
    Selection(crate::registry::SelectionError),
    /// Incomplete development binding, never a reference-compatible fallback.
    BindingGap {
        backend: String,
        operation: String,
    },
    Native(String),
    ReferenceException {
        class: &'static str,
        message: String,
    },
    AcceleratorException {
        code: i32,
        message: String,
    },
    Torch(tch::TchError),
    Retained(std::rc::Rc<crate::source_exception::Exception>),
}
/// LibTorch's exact thread-local InferenceMode context. Guards must be dropped
/// on their creating thread and nested in the same order as Python contexts.
#[must_use]
pub struct InferenceMode {
    handle: NonNull<c_void>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
unsafe extern "C" {
    fn at_imagine_inference_mode_new(enabled: i32) -> *mut c_void;
    fn at_imagine_inference_mode_free(value: *mut c_void);
    fn at_imagine_storage_info(value: *const c_void, identity: *mut usize, bytes: *mut u64);
    fn at_imagine_weak_storage_new(value: *const c_void) -> *mut c_void;
    fn at_imagine_weak_storage_expired(value: *mut c_void) -> i32;
    fn at_imagine_weak_storage_free(value: *mut c_void);
    fn at_imagine_source_normal(
        source: *const torch_sys::C_tensor,
        handle: *mut c_void,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_index_copy_to(
        target: *const torch_sys::C_tensor,
        source: *const torch_sys::C_tensor,
    );
    fn at_imagine_context_index(
        full: *const torch_sys::C_tensor,
        values: *const i64,
        count: i64,
        dim: i64,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_context_index_put(
        full: *const torch_sys::C_tensor,
        values: *const i64,
        count: i64,
        dim: i64,
        source: *const torch_sys::C_tensor,
    );
    fn at_imagine_context_size(full: *const torch_sys::C_tensor, dim: i64) -> i64;
    fn at_imagine_context_scalar_at(
        full: *const torch_sys::C_tensor,
        index: i64,
        dim: i64,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_context_scalar_at_put(
        full: *const torch_sys::C_tensor,
        index: i64,
        dim: i64,
        source: *const torch_sys::C_tensor,
    );
    fn at_imagine_context_slice(
        full: *const torch_sys::C_tensor,
        dim: i64,
        start: i64,
        end: i64,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_context_scalar_index(
        full: *const torch_sys::C_tensor,
        values: *const i64,
        count: i64,
    ) -> *mut torch_sys::C_tensor;
    fn at_imagine_global_rng_state(device: i32) -> *mut torch_sys::C_tensor;
    fn at_imagine_clone(value: *const torch_sys::C_tensor) -> *mut torch_sys::C_tensor;
    fn at_imagine_current_stream_capturing() -> i32;
    fn at_imagine_generator_randperm(handle: *mut c_void, n: i64) -> *mut torch_sys::C_tensor;
}
/// The original Tensor indexing assignment copy/broadcast boundary. Calling
/// plain copy_ omits its prefix-unit view and expand step (and changes errors).
pub fn index_copy_to(target: &Tensor, source: &Tensor) -> Result<()> {
    unsafe {
        at_imagine_index_copy_to(target.as_ptr(), source.as_ptr());
    }
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
    Ok(())
}
pub fn context_index(full: &Tensor, indices: &[i64], dim: i64) -> Result<Tensor> {
    let value = unsafe {
        at_imagine_context_index(full.as_ptr(), indices.as_ptr(), indices.len() as i64, dim)
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
        return Err(Error::Native("Null context index result".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
pub fn context_index_put(full: &Tensor, indices: &[i64], dim: i64, source: &Tensor) -> Result<()> {
    unsafe {
        at_imagine_context_index_put(
            full.as_ptr(),
            indices.as_ptr(),
            indices.len() as i64,
            dim,
            source.as_ptr(),
        );
    }
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
    Ok(())
}
pub fn context_slice(full: &Tensor, dim: i64, start: i64, end: i64) -> Result<Tensor> {
    let value = unsafe { at_imagine_context_slice(full.as_ptr(), dim, start, end) };
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
        return Err(Error::Native("Null context slice result".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
pub fn context_size(full: &Tensor, dim: i64) -> Result<i64> {
    let value = unsafe { at_imagine_context_size(full.as_ptr(), dim) };
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
    Ok(value)
}
pub fn context_scalar_at(full: &Tensor, index: i64, dim: i64) -> Result<Tensor> {
    let value = unsafe { at_imagine_context_scalar_at(full.as_ptr(), index, dim) };
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
        return Err(Error::Native("Null scalar-at context index result".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
pub fn context_scalar_at_put(full: &Tensor, index: i64, dim: i64, source: &Tensor) -> Result<()> {
    unsafe {
        at_imagine_context_scalar_at_put(full.as_ptr(), index, dim, source.as_ptr());
    }
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
    Ok(())
}
pub fn context_scalar_index(full: &Tensor, indices: &[i64]) -> Result<Tensor> {
    let value = unsafe {
        at_imagine_context_scalar_index(full.as_ptr(), indices.as_ptr(), indices.len() as i64)
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
        return Err(Error::Native("Null scalar context index result".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
/// torch.cuda.is_current_stream_capturing, including its no-context fast path.
pub fn current_stream_capturing() -> Result<bool> {
    let value = unsafe { at_imagine_current_stream_capturing() };
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
    Ok(value != 0)
}
/// The original out-of-place aten::clone, including its autograd operation.
pub fn clone_tensor(value: &Tensor) -> Result<Tensor> {
    let value = unsafe { at_imagine_clone(value.as_ptr()) };
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
        return Err(Error::Native("Null cloned tensor".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
/// State of the ordinary global stream; used to validate source reseeding/draws.
pub fn global_rng_state(device: Device) -> Result<Tensor> {
    let index = match device {
        Device::Cpu => -1,
        Device::Cuda(index) => index as i32,
        _ => {
            return Err(Error::Native(
                "Initial-platform RNG device must be CPU or CUDA".into(),
            ));
        }
    };
    let value = unsafe { at_imagine_global_rng_state(index) };
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
        return Err(Error::Native("Null global RNG state".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
/// Storage identity is process-local and only meaningful while its tensor lives.
/// Views retain their complete allocation, including storage outside the view.
pub fn storage_info(value: &Tensor) -> Result<(usize, u64)> {
    let (mut identity, mut bytes) = (0, 0);
    unsafe {
        at_imagine_storage_info(value.as_ptr().cast(), &mut identity, &mut bytes);
    }
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
    Ok((identity, bytes))
}
/// ATen's original weak StorageImpl owner. This does not retain an allocation's
/// payload; it permits exact allocation-lifetime checks without reading data.
pub struct WeakStorage(NonNull<c_void>);
impl WeakStorage {
    pub fn new(value: &Tensor) -> Result<Self> {
        let handle = unsafe { at_imagine_weak_storage_new(value.as_ptr().cast()) };
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
        Ok(Self(NonNull::new(handle).ok_or_else(|| {
            Error::Native("Null ATen weak storage".into())
        })?))
    }
    pub fn expired(&self) -> bool {
        unsafe { at_imagine_weak_storage_expired(self.0.as_ptr()) != 0 }
    }
}
impl Drop for WeakStorage {
    fn drop(&mut self) {
        unsafe {
            at_imagine_weak_storage_free(self.0.as_ptr());
        }
    }
}
pub fn inference_mode(enabled: bool) -> Result<InferenceMode> {
    let handle = unsafe { at_imagine_inference_mode_new(i32::from(enabled)) };
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    if !error.is_null() {
        let text = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { libc::free(error.cast()) };
        if !handle.is_null() {
            unsafe { at_imagine_inference_mode_free(handle) };
        }
        return Err(tch::TchError::Torch(text).into());
    }
    Ok(InferenceMode {
        handle: NonNull::new(handle)
            .ok_or_else(|| Error::Native("Null InferenceMode guard".into()))?,
        _thread: std::marker::PhantomData,
    })
}
impl Drop for InferenceMode {
    fn drop(&mut self) {
        unsafe { at_imagine_inference_mode_free(self.handle.as_ptr()) };
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Selection(error) => error.fmt(f),
            Self::BindingGap { backend, operation } => {
                write!(f, "Unbound development operation {backend}:{operation}")
            }
            Self::Native(message) => f.write_str(message),
            Self::ReferenceException { message, .. } => f.write_str(message),
            Self::AcceleratorException { message, .. } => f.write_str(message),
            Self::Torch(error) => error.fmt(f),
            Self::Retained(error) => error.source().fmt(f),
        }
    }
}
impl std::error::Error for Error {}
impl From<crate::registry::SelectionError> for Error {
    fn from(error: crate::registry::SelectionError) -> Self {
        Self::Selection(error)
    }
}
impl From<tch::TchError> for Error {
    fn from(error: tch::TchError) -> Self {
        let error = match error {
            tch::TchError::Torch(message) => {
                tch::TchError::Torch(crate::python_error_messages::translate(message))
            }
            error => error,
        };
        if let tch::TchError::Torch(message) = &error {
            for (tag, class) in [
                ("value", "ValueError"),
                ("type", "TypeError"),
                ("not-implemented", "NotImplementedError"),
                ("buffer", "BufferError"),
                ("syntax", "SyntaxError"),
                ("runtime", "RuntimeError"),
            ] {
                if let Some(message) = message.strip_prefix(&format!("[imagine-c10:{tag}]\n")) {
                    return Self::ReferenceException {
                        class,
                        message: message.into(),
                    };
                }
            }
            if let Some(message) = message.strip_prefix("[imagine-c10:linalg]\n") {
                return Self::ReferenceException {
                    class: "_LinAlgError",
                    message: message.into(),
                };
            }
            if let Some(message) = message.strip_prefix("[imagine-c10:index]\n") {
                return Self::ReferenceException {
                    class: "IndexError",
                    message: message.into(),
                };
            }
            if let Some(message) = message.strip_prefix("[imagine-c10:oom]\n") {
                return Self::ReferenceException {
                    class: "OutOfMemoryError",
                    message: message.into(),
                };
            }
            if let Some((code, message)) = message
                .strip_prefix("[imagine-c10:accelerator:")
                .and_then(|m| m.split_once("]\n"))
                .and_then(|(c, m)| c.parse().ok().map(|c| (c, m)))
            {
                return Self::AcceleratorException {
                    code,
                    message: message.into(),
                };
            }
        }
        Self::Torch(error)
    }
}
type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
struct RawArg {
    kind: i64,
    tensor: *const c_void,
    integer: i64,
    real: f64,
}
pub enum KernelArg<'a> {
    None,
    Tensor(&'a Tensor),
    Int(i64),
    Float(f64),
    Bool(bool),
}
impl KernelArg<'_> {
    fn raw(&self) -> RawArg {
        let mut out = RawArg {
            kind: 0,
            tensor: std::ptr::null(),
            integer: 0,
            real: 0.0,
        };
        match self {
            Self::None => {}
            Self::Tensor(t) => {
                out.kind = 1;
                out.tensor = t.as_ptr().cast();
            }
            Self::Int(i) => {
                out.kind = 2;
                out.integer = *i;
            }
            Self::Float(x) => {
                out.kind = 3;
                out.real = *x;
            }
            Self::Bool(b) => {
                out.kind = 4;
                out.integer = i64::from(*b);
            }
        }
        out
    }
}
type LastText = unsafe extern "C" fn() -> *const c_char;
type Call = unsafe extern "C" fn(*const c_char, *const RawArg, usize, *mut i64) -> i32;
type AtenBoxed = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const *const c_void,
    usize,
    *const *const c_char,
    *const *const c_void,
    usize,
    *mut *mut c_void,
) -> i32;
type ScalarOut = unsafe extern "C" fn(*mut i64) -> i32;
type DevicePropertiesOut = unsafe extern "C" fn(i64, *mut CudaDeviceProperties) -> i32;
type ParseDevice = unsafe extern "C" fn(*const c_char, *mut i64, *mut i64) -> i32;
type TensorToDevice = unsafe extern "C" fn(*const c_void, i64, i64, *mut *mut c_void) -> i32;
type CapabilityOut = unsafe extern "C" fn(*mut i64, *mut i64) -> i32;
type Cat = unsafe extern "C" fn(*const *const c_void, usize, i64, *mut *mut c_void) -> i32;
type Embedding = unsafe extern "C" fn(*const c_void, *const c_void, *mut *mut c_void) -> i32;
type Fp8ScaledMm = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    i64,
    *mut *mut c_void,
) -> i32;
type Stream = unsafe extern "C" fn(i64, *mut u64) -> i32;
type HandleNew = unsafe extern "C" fn(i64, *mut *mut c_void) -> i32;
type HandleFree = unsafe extern "C" fn(*mut c_void);
type EventWait = unsafe extern "C" fn(*mut c_void, i64) -> i32;
type EventSync = unsafe extern "C" fn(*mut c_void) -> i32;
type TimingEventNew = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type EventElapsed = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut f64) -> i32;
type DeviceSync = unsafe extern "C" fn(i64) -> i32;
type GenNew = unsafe extern "C" fn(i64, u64, *mut *mut c_void) -> i32;
type GenFree = unsafe extern "C" fn(*mut c_void);
type GenState = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
type GenSetState = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;
type GenNormal = unsafe extern "C" fn(*mut c_void, *const i64, usize, i64, *mut *mut c_void) -> i32;
type GenBytes = unsafe extern "C" fn(*mut c_void, *const i64, usize, *mut *mut c_void) -> i32;
type GenInt = unsafe extern "C" fn(*mut c_void, i64, *const i64, usize, *mut *mut c_void) -> i32;
type TensorClone = unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> i32;
unsafe extern "C" fn release_buffer_owner(ptr: *mut c_void) {
    drop(unsafe { Box::from_raw(ptr.cast::<Arc<dyn Any + Send + Sync>>()) });
}
type TensorEmpty = unsafe extern "C" fn(*const c_void, i64, *mut *mut c_void) -> i32;
type TensorCopy = unsafe extern "C" fn(*mut c_void, *const c_void, i64) -> i32;
type TensorType = unsafe extern "C" fn(*const c_void, *mut TensorTypeInfo) -> i32;
type TensorIdentity = unsafe extern "C" fn(*const c_void) -> u64;
type TensorStorage = unsafe extern "C" fn(*const c_void, *mut TensorStorageInfo) -> i32;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorStorageInfo {
    pub base: u64,
    pub offset: i64,
    pub element_bytes: i64,
}
type TensorShellView = unsafe extern "C" fn(*const c_void, i64, *mut *mut c_void) -> i32;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorTypeInfo {
    pub scalar_type: i64,
    pub dlpack_code: i64,
    pub bits: i64,
    pub lanes: i64,
    pub element_bytes: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellDType {
    E8M0,
    PackedE2M1,
}

/// The same ATen-cached CUDA properties read by torch.cuda.get_device_properties.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaDeviceProperties {
    pub major: i64,
    pub minor: i64,
    pub shared_memory_per_block: i64,
    pub shared_memory_per_block_optin: i64,
    pub multiprocessor_count: i64,
    pub l2_cache_size: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RmsTemporaryMetadata {
    pub start: u64,
    pub bytes: u64,
    pub count: i64,
}
type RmsBegin = unsafe extern "C" fn() -> i32;
type RmsEnd = unsafe extern "C" fn(*mut RmsTemporaryMetadata) -> i32;

type TensorGradName = unsafe extern "C" fn(*const c_void, *mut *const c_char) -> i32;
type TensorItemDouble = unsafe extern "C" fn(*const c_void, *mut f64) -> i32;
type IValueUInt = unsafe extern "C" fn(u64, *mut *mut c_void) -> i32;
type IValueGetUInt = unsafe extern "C" fn(*const c_void, *mut u64) -> i32;
struct Library {
    beta_ppf: unsafe extern "C" fn(
        f64,
        f64,
        f64,
        *mut f64,
        *mut *const c_char,
        *mut *const c_char,
    ) -> i32,
    item_double: TensorItemDouble,
    grad_name: TensorGradName,
    ivalue_uint: IValueUInt,
    ivalue_get_uint: IValueGetUInt,
    aten_boxed: AtenBoxed,
    handle: NonNull<c_void>,
    rms_begin: RmsBegin,
    rms_end: RmsEnd,
    error: LastText,
    metadata: LastText,
    call: Call,
    cublaslt_available: ScalarOut,
    cuda_device_count: ScalarOut,
    cuda_current_device: ScalarOut,
    cuda_current_capability: CapabilityOut,
    cuda_device_properties: DevicePropertiesOut,
    parse_device: ParseDevice,
    to_device_spec: TensorToDevice,
    empty_like_device_spec: TensorToDevice,
    default_dtype: ScalarOut,
    set_default_dtype: DeviceSync,
    fp8_scaled_mm: Fp8ScaledMm,
    embedding: Embedding,
    amax_last: TensorClone,
    cat: Cat,
    stream: Stream,
    stream_guard_new: HandleNew,
    stream_guard_free: HandleFree,
    event_record: HandleNew,
    event_free: HandleFree,
    event_wait: EventWait,
    event_sync: EventSync,
    timing_event_new: TimingEventNew,
    event_record_existing: EventWait,
    event_elapsed: EventElapsed,
    device_sync: DeviceSync,
    gen_new: GenNew,
    gen_free: GenFree,
    gen_state: GenState,
    gen_set_state: GenSetState,
    gen_normal: GenNormal,
    gen_uniform: GenNormal,
    gen_bytes: GenBytes,
    gen_int: GenInt,
    tensor_clone: TensorClone,
    tensor_empty: TensorEmpty,
    tensor_empty_dtype: TensorEmpty,
    tensor_copy: TensorCopy,
    tensor_type: TensorType,
    tensor_storage: TensorStorage,
    tensor_identity: TensorIdentity,
    tensor_shell_view: TensorShellView,
}
// dlopen handles are shared; bridge diagnostics and CUDA guards are thread-local.
// Generator ownership itself is kept separate and not marked Send/Sync.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle.as_ptr());
        }
    }
}
impl Library {
    fn exception(&self, class: &'static str, message: String) -> Error {
        if class == "AcceleratorError" {
            let symbol = unsafe {
                libc::dlsym(
                    self.handle.as_ptr(),
                    c"imagine_last_accelerator_code".as_ptr(),
                )
            };
            if symbol.is_null() {
                return Error::BindingGap {
                    backend: "native_exception".into(),
                    operation: "Missing accelerator error code ABI".into(),
                };
            }
            let code: unsafe extern "C" fn() -> i32 = unsafe { std::mem::transmute(symbol) };
            Error::AcceleratorException {
                code: unsafe { code() },
                message,
            }
        } else {
            Error::ReferenceException { class, message }
        }
    }
    fn check(&self, code: i32) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // Error storage is thread-local and is copied before another bridge call.
        let text = unsafe { CStr::from_ptr((self.error)()) }
            .to_string_lossy()
            .into_owned();
        match code {
            -9 => Err(self.exception("OutOfMemoryError", text)),
            -10 => Err(self.exception("AcceleratorError", text)),
            -20 | -21 => Err(Error::ReferenceException {
                class: if code == -20 {
                    "ValueError"
                } else {
                    "IndexError"
                },
                message: text,
            }),
            _ => Err(Error::Native(text)),
        }
    }
}

#[derive(Clone)]
pub struct Kitchen {
    library: Arc<Library>,
    launch_guard: Option<LaunchGuard>,
}
pub type LaunchGuard = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;
pub struct KernelResult {
    pub value: i64,
    pub metadata: Option<String>,
}
pub struct BetaQuantile {
    pub value: f64,
    /// Original SciPy RuntimeWarning text, retained for the caller's warning sink.
    pub warning: Option<String>,
}
pub struct ScaledMm<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub scale_a: &'a Tensor,
    pub scale_b: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub output_kind: Option<Kind>,
}
pub struct SdpaOptions<'a> {
    pub mask: Option<&'a Tensor>,
    pub scale: Option<f64>,
    pub enable_gqa: bool,
    /// The frozen ops.py CUDA/priority wrapper is global even for CPU inputs.
    pub priority_wrapper: bool,
}
#[derive(Debug)]
pub enum ScaledMmFailure {
    Runtime(Error),
    Uncaught(Error),
}
/// A parsed c10 device retains an absent index (cuda != cuda:0, cpu != cpu:0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSpec {
    type_code: i64,
    index: i64,
}
impl DeviceSpec {
    pub fn equals_tensor_device(self, device: Device) -> bool {
        match device {
            Device::Cpu => self.type_code == 0 && self.index == -1,
            Device::Cuda(i) => self.type_code == 1 && self.index == i as i64,
            Device::Mps => self.type_code == 13 && self.index == -1,
            Device::Vulkan => self.type_code == 10 && self.index == -1,
        }
    }
}
impl Kitchen {
    /// Unchanged SciPy 1.17.1 beta inverse body and its exact Boost dependency.
    pub fn beta_quantile(&self, p: f64, a: f64, b: f64) -> Result<BetaQuantile> {
        let mut value = 0.;
        let mut warning = std::ptr::null();
        let mut error = std::ptr::null();
        let status =
            unsafe { (self.library.beta_ppf)(p, a, b, &mut value, &mut warning, &mut error) };
        let text = |p: *const c_char| {
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        };
        let warning = text(warning);
        if status != 0 {
            let message = text(error).unwrap_or_default();
            return Err(if status == 1 {
                Error::ReferenceException {
                    class: "OverflowError",
                    message,
                }
            } else {
                Error::Native(message)
            });
        }
        Ok(BetaQuantile { value, warning })
    }
    pub(crate) fn buffer_owner(
        &self,
        tensor: &Tensor,
    ) -> Result<Option<Arc<dyn Any + Send + Sync>>> {
        type Context = unsafe extern "C" fn(
            *const c_void,
            unsafe extern "C" fn(*mut c_void),
            *mut *mut c_void,
        ) -> i32;
        let symbol = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_tensor_buffer_context".as_ptr(),
            )
        };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_tensor_buffer_context".into(),
            });
        }
        let call: Context = unsafe { std::mem::transmute(symbol) };
        let mut owner = std::ptr::null_mut();
        self.library
            .check(unsafe { call(tensor.as_ptr().cast(), release_buffer_owner, &mut owner) })?;
        Ok(if owner.is_null() {
            None
        } else {
            Some(unsafe { (&*owner.cast::<Arc<dyn Any + Send + Sync>>()).clone() })
        })
    }
    /// Frozen NVIDIA `comfy.ops.scaled_dot_product_attention`, including the
    /// source's small-input branch, capability probes and context restoration.
    pub fn comfy_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        options: SdpaOptions<'_>,
    ) -> Result<Tensor> {
        type Sdpa = unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            i64,
            f64,
            i64,
            i64,
            *mut *mut c_void,
        ) -> i32;
        let symbol =
            unsafe { libc::dlsym(self.library.handle.as_ptr(), c"imagine_comfy_sdpa".as_ptr()) };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_comfy_sdpa".into(),
            });
        }
        let call: Sdpa = unsafe { std::mem::transmute(symbol) };
        let mut out = std::ptr::null_mut();
        let code = unsafe {
            call(
                q.as_ptr().cast(),
                k.as_ptr().cast(),
                v.as_ptr().cast(),
                options.mask.map_or(std::ptr::null(), |m| m.as_ptr().cast()),
                i64::from(options.scale.is_some()),
                options.scale.unwrap_or(0.),
                i64::from(options.enable_gqa),
                i64::from(options.priority_wrapper),
                &mut out,
            )
        };
        if code != 0 {
            return Err(self.library.exception(
                match code {
                    -2 => "ValueError",
                    -3 => "IndexError",
                    -4 => "BufferError",
                    -5 => "SyntaxError",
                    -7 => "TypeError",
                    -8 => "NotImplementedError",
                    -9 => "OutOfMemoryError",
                    -10 => "AcceleratorError",
                    _ => "RuntimeError",
                },
                unsafe { CStr::from_ptr((self.library.error)()) }
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
        if out.is_null() {
            return Err(Error::Native("Null SDPA result".into()));
        }
        Ok(unsafe { Tensor::from_ptr(out.cast()) })
    }
    /// The five source-visible flags and complete priority order. Used to
    /// preserve caller state and verify finally behavior, not to tune kernels.
    pub fn sdpa_state(&self) -> Result<[i64; 10]> {
        let mut state = [0; 10];
        self.sdpa_state_raw(&mut state, false)?;
        Ok(state)
    }
    /// Original cuda.mem_get_info plus active/reserved allocator counters.
    /// No process inventory or production data is queried.
    pub fn cuda_memory(&self, index: usize) -> Result<[u64; 4]> {
        type Query = unsafe extern "C" fn(i64, *mut u64) -> i32;
        let sym = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_cuda_memory".as_ptr(),
            )
        };
        if sym.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_cuda_memory".into(),
            });
        }
        let call: Query = unsafe { std::mem::transmute(sym) };
        let mut values = [0; 4];
        self.library
            .check(unsafe { call(index as i64, values.as_mut_ptr()) })?;
        Ok(values)
    }
    /// LibTorch aggregate allocated/reserved current and true peak byte counts.
    /// This optional bridge extension leaves frozen earlier bridges loadable.
    pub fn cuda_allocator_stats(&self, index: usize) -> Result<[u64; 4]> {
        type Query = unsafe extern "C" fn(i64, *mut u64) -> i32;
        let symbol = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_cuda_allocator_stats".as_ptr(),
            )
        };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_cuda_allocator_stats".into(),
            });
        }
        let call: Query = unsafe { std::mem::transmute(symbol) };
        let mut values = [0; 4];
        self.library
            .check(unsafe { call(index as i64, values.as_mut_ptr()) })?;
        Ok(values)
    }
    /// Calls the CUDA registration operation without tensor guards or recovery.
    ///
    /// # Safety
    /// The caller must retain the live CPU allocation covering pointer/bytes
    /// throughout registration, its use and the corresponding unregistration.
    pub unsafe fn host_registration_raw(
        &self,
        pointer: usize,
        bytes: usize,
        pin: bool,
    ) -> Result<bool> {
        type Call = unsafe extern "C" fn(u64, u64, i64, *mut i64) -> i32;
        let sym = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_host_registration_raw".as_ptr(),
            )
        };
        if sym.is_null() {
            return Err(Error::BindingGap {
                backend: "CUDA".into(),
                operation: "imagine_host_registration_raw".into(),
            });
        }
        let call: Call = unsafe { std::mem::transmute(sym) };
        let mut success = 0;
        self.library
            .check(unsafe { call(pointer as u64, bytes as u64, i64::from(pin), &mut success) })?;
        Ok(success != 0)
    }
    /// Original output[:, :, start:end] = value, including setitem's leading
    /// unit-dimension stripping, broadcasting and native error dispatch.
    pub fn set_video_frames(
        &self,
        output: &mut Tensor,
        start: i64,
        end: i64,
        value: &Tensor,
    ) -> Result<()> {
        type Call = unsafe extern "C" fn(*mut c_void, i64, i64, *const c_void) -> i32;
        let symbol = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_tensor_video_slice_assign".as_ptr(),
            )
        };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_tensor_video_slice_assign".into(),
            });
        }
        let call: Call = unsafe { std::mem::transmute(symbol) };
        self.check_aten_result(unsafe {
            call(
                output.as_mut_ptr().cast(),
                start,
                end,
                value.as_ptr().cast(),
            )
        })
    }
    pub fn host_registration(&self, tensor: &Tensor, pin: bool) -> Result<bool> {
        if tensor.device() != tch::Device::Cpu {
            return Ok(false);
        }
        if pin && (tensor.f_is_pinned(tch::Device::Cuda(0))? || !tensor.is_contiguous()) {
            return Ok(false);
        }
        let pointer = tensor.data_ptr() as usize;
        if pin && pointer == 0 {
            return Ok(false);
        }
        let size = tensor.numel() * tensor.kind().elt_size_in_bytes();
        // The caller retains tensor storage through registration and unpin.
        let success = unsafe { self.host_registration_raw(pointer, size, pin) }?;
        if !success {
            eprintln!(
                "WARNING:root:{}",
                if pin { "Pin error." } else { "Unpin error." }
            );
            crate::memory::exceptions::discard_native_cuda_async_error(self, tch::Device::Cuda(0))?;
        }
        Ok(success)
    }

    pub fn comfy_math_state(&self) -> Result<[i64; 3]> {
        let mut state = [0; 3];
        self.comfy_math_state_raw(&mut state, false)?;
        Ok(state)
    }
    pub fn set_comfy_math_state(&self, mut state: [i64; 3]) -> Result<()> {
        self.comfy_math_state_raw(&mut state, true)
    }
    fn comfy_math_state_raw(&self, state: &mut [i64; 3], write: bool) -> Result<()> {
        type State = unsafe extern "C" fn(*mut i64, i64) -> i32;
        let sym = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_comfy_math_state".as_ptr(),
            )
        };
        if sym.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_comfy_math_state".into(),
            });
        }
        let call: State = unsafe { std::mem::transmute(sym) };
        self.library
            .check(unsafe { call(state.as_mut_ptr(), i64::from(write)) })
    }
    pub fn cuda_soft_empty_cache(&self, index: usize) -> Result<()> {
        type Empty = unsafe extern "C" fn(i64) -> i32;
        let sym = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_cuda_soft_empty_cache".as_ptr(),
            )
        };
        if sym.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_cuda_soft_empty_cache".into(),
            });
        }
        let call: Empty = unsafe { std::mem::transmute(sym) };
        self.library.check(unsafe { call(index as i64) })
    }
    pub fn set_sdpa_state(&self, mut state: [i64; 10]) -> Result<()> {
        self.sdpa_state_raw(&mut state, true)
    }
    fn sdpa_state_raw(&self, state: &mut [i64; 10], write: bool) -> Result<()> {
        type State = unsafe extern "C" fn(*mut i64, i64) -> i32;
        let symbol =
            unsafe { libc::dlsym(self.library.handle.as_ptr(), c"imagine_sdpa_state".as_ptr()) };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_sdpa_state".into(),
            });
        }
        let call: State = unsafe { std::mem::transmute(symbol) };
        self.library
            .check(unsafe { call(state.as_mut_ptr(), i64::from(write)) })
    }
    /// ATen empty including the wider unsigned dtypes not exposed by tch::Kind.
    pub(crate) fn empty_scalar(&self, shape: &[i64], scalar: i64) -> Result<Tensor> {
        type EmptyScalar = unsafe extern "C" fn(*const i64, usize, i64, *mut *mut c_void) -> i32;
        let symbol = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_tensor_empty_scalar".as_ptr(),
            )
        };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_tensor_empty_scalar".into(),
            });
        }
        let call: EmptyScalar = unsafe { std::mem::transmute(symbol) };
        let mut output = std::ptr::null_mut();
        self.library
            .check(unsafe { call(shape.as_ptr(), shape.len(), scalar, &mut output) })?;
        if output.is_null() {
            return Err(Error::Native("Null scalar empty tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    /// Attach a source-owned read-only model buffer to an ATen Storage.
    /// Views keep `owner` alive, including after the loader and Kitchen drop.
    /// # Safety
    /// The buffer must contain the initialized, validated contiguous tensor
    /// described by `shape`/`scalar`, and remain valid until `owner` drops.
    pub(crate) unsafe fn tensor_from_owned_buffer<T: Send + Sync + 'static>(
        &self,
        data: *const u8,
        shape: &[i64],
        scalar: i64,
        owner: Arc<T>,
    ) -> Result<Tensor> {
        type OwnedBuffer = unsafe extern "C" fn(
            *const u8,
            *const i64,
            usize,
            i64,
            *mut c_void,
            unsafe extern "C" fn(*mut c_void),
            *mut *mut c_void,
        ) -> i32;
        let symbol = unsafe {
            libc::dlsym(
                self.library.handle.as_ptr(),
                c"imagine_tensor_owned_buffer".as_ptr(),
            )
        };
        if symbol.is_null() {
            return Err(Error::BindingGap {
                backend: "ATen".into(),
                operation: "imagine_tensor_owned_buffer".into(),
            });
        }
        let call: OwnedBuffer = unsafe { std::mem::transmute(symbol) };
        let mut output = std::ptr::null_mut();
        let owner: Arc<dyn Any + Send + Sync> = owner;
        let owner = Box::into_raw(Box::new(owner)).cast();
        // The bridge consumes the owner even when ATen rejects the geometry.
        let code = unsafe {
            call(
                data,
                shape.as_ptr(),
                shape.len(),
                scalar,
                owner,
                release_buffer_owner,
                &mut output,
            )
        };
        self.library.check(code)?;
        if output.is_null() {
            return Err(Error::Native("Null owned-buffer tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    /// Original Tensor.grad_fn accessor, including the repr Invalid fallback.
    pub(crate) fn grad_fn_name(&self, tensor: &Tensor) -> Result<Option<String>> {
        let mut name = std::ptr::null();
        self.library
            .check(unsafe { (self.library.grad_name)(tensor.as_ptr().cast(), &mut name) })?;
        let name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        Ok(if name.is_empty() {
            None
        } else {
            Some(name.rsplit("::").next().unwrap().into())
        })
    }
    /// Original Tensor float conversion with its typed C++ exception.
    pub fn item_double(&self, value: &Tensor) -> Result<f64> {
        let mut output = 0.;
        let code = unsafe { (self.library.item_double)(value.as_ptr().cast(), &mut output) };
        if code == 0 {
            Ok(output)
        } else {
            Err(self.library.exception(
                match code {
                    -2 => "ValueError",
                    -3 => "IndexError",
                    -4 => "TypeError",
                    -5 => "NotImplementedError",
                    -9 => "OutOfMemoryError",
                    -10 => "AcceleratorError",
                    _ => "RuntimeError",
                },
                unsafe { CStr::from_ptr((self.library.error)()) }
                    .to_string_lossy()
                    .into_owned(),
            ))
        }
    }
    /// # Safety
    /// Returned c10::IValue ownership uses the pinned torch-sys allocator ABI.
    pub(crate) unsafe fn ivalue_uint(&self, value: u64) -> Result<*mut c_void> {
        let mut output = std::ptr::null_mut();
        let code = unsafe { (self.library.ivalue_uint)(value, &mut output) };
        if code == 0 {
            Ok(output)
        } else {
            Err(Error::Native(
                unsafe { CStr::from_ptr((self.library.error)()) }
                    .to_string_lossy()
                    .into_owned(),
            ))
        }
    }
    /// # Safety
    /// Input is a live c10::IValue from the pinned ABI, borrowed for this call.
    pub(crate) unsafe fn ivalue_get_uint(&self, value: *const c_void) -> Option<u64> {
        let mut output = 0;
        (unsafe { (self.library.ivalue_get_uint)(value, &mut output) } == 1).then_some(output)
    }
    /// # Safety
    /// Every value pointer is a live borrowed c10::IValue from the pinned
    /// torch-sys ABI. The returned value is owned and must be freed with ati_free.
    pub(crate) unsafe fn aten_boxed_raw(
        &self,
        name: &CStr,
        overload: &CStr,
        args: &[*const c_void],
        keys: &[*const c_char],
        values: &[*const c_void],
    ) -> Result<*mut c_void> {
        assert_eq!(keys.len(), values.len());
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.aten_boxed)(
                name.as_ptr(),
                overload.as_ptr(),
                args.as_ptr(),
                args.len(),
                keys.as_ptr(),
                values.as_ptr(),
                keys.len(),
                &mut output,
            )
        };
        if code == 0 {
            return Ok(output);
        }
        let class = match code {
            -2 => "ValueError",
            -3 => "IndexError",
            -4 => "TypeError",
            -5 => "NotImplementedError",
            -9 => "OutOfMemoryError",
            -10 => "AcceleratorError",
            _ => "RuntimeError",
        };
        Err(self.library.exception(
            class,
            unsafe { CStr::from_ptr((self.library.error)()) }
                .to_string_lossy()
                .into_owned(),
        ))
    }
    /// Load a specifically selected standalone bridge. No alternate backend is selected.
    ///
    /// # Safety
    /// The caller must verify the selected library is the trusted bridge build
    /// with this exact C ABI. Loading foreign native code can execute initializers.
    pub unsafe fn load(path: &Path) -> Result<Self> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| Error::Native("NUL in library path".into()))?;
        let handle =
            NonNull::new(unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) })
                .ok_or_else(|| {
                    Error::Native(
                        unsafe { CStr::from_ptr(libc::dlerror()) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                })?;
        unsafe fn symbol<T: Copy>(handle: NonNull<c_void>, name: &CStr) -> Result<T> {
            let ptr = unsafe { libc::dlsym(handle.as_ptr(), name.as_ptr()) };
            if ptr.is_null() {
                return Err(Error::Native(format!(
                    "Missing native symbol: {}",
                    name.to_string_lossy()
                )));
            }
            assert_eq!(std::mem::size_of::<T>(), std::mem::size_of_val(&ptr));
            Ok(unsafe { std::mem::transmute_copy(&ptr) })
        }
        let loaded = (|| unsafe {
            Ok::<_, Error>(Library {
                beta_ppf: symbol(handle, c"imagine_beta_ppf")?,
                aten_boxed: symbol(handle, c"imagine_aten_boxed")?,
                item_double: symbol(handle, c"imagine_tensor_item_double")?,
                grad_name: symbol(handle, c"imagine_tensor_grad_name")?,
                ivalue_uint: symbol(handle, c"imagine_ivalue_uint")?,
                ivalue_get_uint: symbol(handle, c"imagine_ivalue_get_uint")?,
                handle,
                rms_begin: symbol(handle, c"imagine_rms_metadata_begin")?,
                rms_end: symbol(handle, c"imagine_rms_metadata_end")?,
                error: symbol(handle, c"imagine_last_error")?,
                metadata: symbol(handle, c"imagine_last_metadata")?,
                call: symbol(handle, c"imagine_kitchen_call")?,
                cublaslt_available: symbol(handle, c"imagine_cublaslt_available")?,
                cuda_current_device: symbol(handle, c"imagine_cuda_current_device")?,
                cuda_device_count: symbol(handle, c"imagine_cuda_device_count")?,
                cuda_current_capability: symbol(handle, c"imagine_cuda_current_capability")?,
                cuda_device_properties: symbol(handle, c"imagine_cuda_device_properties")?,
                parse_device: symbol(handle, c"imagine_parse_device")?,
                empty_like_device_spec: symbol(handle, c"imagine_tensor_empty_like_device_spec")?,
                to_device_spec: symbol(handle, c"imagine_tensor_to_device_spec")?,
                default_dtype: symbol(handle, c"imagine_default_dtype")?,
                set_default_dtype: symbol(handle, c"imagine_set_default_dtype")?,
                fp8_scaled_mm: symbol(handle, c"imagine_fp8_scaled_mm")?,
                embedding: symbol(handle, c"imagine_embedding")?,
                amax_last: symbol(handle, c"imagine_amax_last")?,
                cat: symbol(handle, c"imagine_cat")?,
                stream: symbol(handle, c"imagine_current_stream")?,
                stream_guard_new: symbol(handle, c"imagine_stream_guard_new")?,
                stream_guard_free: symbol(handle, c"imagine_stream_guard_free")?,
                event_record: symbol(handle, c"imagine_event_record")?,
                event_free: symbol(handle, c"imagine_event_free")?,
                event_wait: symbol(handle, c"imagine_event_wait")?,
                event_sync: symbol(handle, c"imagine_event_synchronize")?,
                timing_event_new: symbol(handle, c"imagine_timing_event_new")?,
                event_record_existing: symbol(handle, c"imagine_event_record_existing")?,
                event_elapsed: symbol(handle, c"imagine_event_elapsed")?,
                device_sync: symbol(handle, c"imagine_device_synchronize")?,
                gen_new: symbol(handle, c"imagine_generator_new")?,
                gen_free: symbol(handle, c"imagine_generator_free")?,
                gen_state: symbol(handle, c"imagine_generator_state")?,
                gen_set_state: symbol(handle, c"imagine_generator_set_state")?,
                gen_normal: symbol(handle, c"imagine_generator_randn")?,
                gen_uniform: symbol(handle, c"imagine_generator_rand")?,
                gen_bytes: symbol(handle, c"imagine_generator_randbytes")?,
                gen_int: symbol(handle, c"imagine_generator_randint")?,
                tensor_clone: symbol(handle, c"imagine_tensor_clone")?,
                tensor_empty: symbol(handle, c"imagine_tensor_empty_like")?,
                tensor_empty_dtype: symbol(handle, c"imagine_tensor_empty_like_dtype")?,
                tensor_copy: symbol(handle, c"imagine_tensor_copy")?,
                tensor_type: symbol(handle, c"imagine_tensor_type")?,
                tensor_identity: symbol(handle, c"imagine_tensor_identity")?,
                tensor_storage: symbol(handle, c"imagine_tensor_storage")?,
                tensor_shell_view: symbol(handle, c"imagine_tensor_shell_view")?,
            })
        })();
        match loaded {
            Ok(library) => Ok(Self {
                library: Arc::new(library),
                launch_guard: None,
            }),
            Err(error) => {
                unsafe {
                    libc::dlclose(handle.as_ptr());
                }
                Err(error)
            }
        }
    }
    /// Same original runtime singleton used by the Python extension's HAS_CUBLASLT.
    pub fn cublaslt_available(&self) -> Result<bool> {
        let mut available = 0;
        self.library
            .check(unsafe { (self.library.cublaslt_available)(&mut available) })?;
        Ok(available != 0)
    }
    pub fn cuda_available(&self) -> Result<bool> {
        let mut count = 0;
        self.library
            .check(unsafe { (self.library.cuda_device_count)(&mut count) })?;
        Ok(count > 0)
    }
    pub fn current_device(&self) -> Result<usize> {
        let mut device = 0;
        self.library
            .check(unsafe { (self.library.cuda_current_device)(&mut device) })?;
        usize::try_from(device).map_err(|_| Error::Native("Negative current CUDA device".into()))
    }
    pub fn current_compute_capability(&self) -> Result<(u32, u32)> {
        let (mut major, mut minor) = (0, 0);
        self.library
            .check(unsafe { (self.library.cuda_current_capability)(&mut major, &mut minor) })?;
        Ok((major as u32, minor as u32))
    }
    pub fn device_properties(&self, device: usize) -> Result<CudaDeviceProperties> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        let mut output = CudaDeviceProperties::default();
        self.library
            .check(unsafe { (self.library.cuda_device_properties)(index.into(), &mut output) })?;
        Ok(output)
    }
    fn check_aten_result(&self, code: i32) -> Result<()> {
        if code != 0 {
            let message = unsafe { CStr::from_ptr((self.library.error)()) }
                .to_string_lossy()
                .into_owned();
            let class = match code {
                -2 => "ValueError",
                -3 => "IndexError",
                -4 => "BufferError",
                -5 => "SyntaxError",
                -7 => "TypeError",
                -8 => "NotImplementedError",
                -9 => "OutOfMemoryError",
                -10 => "AcceleratorError",
                _ => "RuntimeError",
            };
            return Err(self.library.exception(class, message));
        }
        Ok(())
    }
    fn aten_tensor_result(&self, code: i32, output: *mut c_void) -> Result<Tensor> {
        self.check_aten_result(code)?;
        if output.is_null() {
            return Err(Error::Native("Null ATen result".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    /// Exact ATen embedding defaults, retaining Python's error translation.
    pub fn embedding(&self, weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.embedding)(weight.as_ptr().cast(), indices.as_ptr().cast(), &mut output)
        };
        self.aten_tensor_result(code, output)
    }
    /// Original ATen cat, retaining ValueError for an empty input sequence.
    pub fn cat(&self, inputs: &[Tensor], dim: i64) -> Result<Tensor> {
        let tensors: Vec<*const c_void> = inputs.iter().map(|t| t.as_ptr().cast()).collect();
        let mut output = std::ptr::null_mut();
        let code = unsafe { (self.library.cat)(tensors.as_ptr(), tensors.len(), dim, &mut output) };
        self.aten_tensor_result(code, output)
    }
    /// amax(dim=-1,keepdim=True), retaining the original IndexError category.
    pub fn amax_last(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe { (self.library.amax_last)(input.as_ptr().cast(), &mut output) };
        self.aten_tensor_result(code, output)
    }
    /// Kitchen's contiguous-input helper and Torch 2.13 scaled_mm contract.
    pub fn scaled_mm_fp8(
        &self,
        request: ScaledMm<'_>,
    ) -> std::result::Result<Tensor, ScaledMmFailure> {
        let kind = match request.output_kind {
            None => -1,
            Some(kind) => {
                let prototype = Tensor::f_empty([0], (kind, Device::Cpu))
                    .map_err(|e| ScaledMmFailure::Uncaught(e.into()))?;
                unsafe { torch_sys::at_scalar_type(prototype.as_ptr().cast_mut()) as i64 }
            }
        };
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.fp8_scaled_mm)(
                request.input.as_ptr().cast(),
                request.weight.as_ptr().cast(),
                request.scale_a.as_ptr().cast(),
                request.scale_b.as_ptr().cast(),
                request.bias.map_or(std::ptr::null(), |b| b.as_ptr().cast()),
                kind,
                &mut output,
            )
        };
        self.aten_tensor_result(code, output).map_err(|e| {
            if matches!(
                &e,
                Error::ReferenceException {
                    class: "RuntimeError"
                        | "TypeError"
                        | "NotImplementedError"
                        | "OutOfMemoryError"
                        | "AcceleratorError",
                    ..
                } | Error::AcceleratorException { .. }
            ) {
                ScaledMmFailure::Runtime(e)
            } else {
                ScaledMmFailure::Uncaught(e)
            }
        })
    }
    pub fn parse_device(&self, text: &str) -> Result<DeviceSpec> {
        let text = CString::new(text).map_err(|_| Error::ReferenceException {
            class: "RuntimeError",
            message: "Invalid device string".into(),
        })?;
        let mut type_code = 0;
        let mut index = 0;
        self.library.check(unsafe {
            (self.library.parse_device)(text.as_ptr(), &mut type_code, &mut index)
        })?;
        Ok(DeviceSpec { type_code, index })
    }
    pub fn device_spec(&self, device: Device) -> Result<DeviceSpec> {
        let text = match device {
            Device::Cpu => "cpu".into(),
            Device::Cuda(i) => format!("cuda:{i}"),
            Device::Mps => "mps".into(),
            Device::Vulkan => "vulkan".into(),
        };
        self.parse_device(&text)
    }
    pub fn empty_like_device_spec(&self, t: &Tensor, device: DeviceSpec) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.empty_like_device_spec)(
                t.as_ptr().cast(),
                device.type_code,
                device.index,
                &mut output,
            )
        };
        self.aten_tensor_result(code, output)
    }
    pub fn to_device_spec(&self, t: &Tensor, device: DeviceSpec) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.to_device_spec)(
                t.as_ptr().cast(),
                device.type_code,
                device.index,
                &mut output,
            )
        };
        self.aten_tensor_result(code, output)
    }
    pub fn default_dtype(&self) -> Result<Kind> {
        let mut kind = 0;
        self.library
            .check(unsafe { (self.library.default_dtype)(&mut kind) })?;
        Ok(match kind {
            5 => Kind::Half,
            6 => Kind::Float,
            7 => Kind::Double,
            15 => Kind::BFloat16,
            _ => return Err(Error::Native(format!("Unbound ATen default dtype {kind}"))),
        })
    }
    /// Set ATen's process-wide floating default, as used by Python scalar inference.
    /// Callers must serialize changes against other work using this global state.
    pub fn set_default_dtype(&self, kind: Kind) -> Result<()> {
        let code = match kind {
            Kind::Half => 5,
            Kind::Float => 6,
            Kind::Double => 7,
            Kind::BFloat16 => 15,
            Kind::Float8e4m3fn | Kind::Float8e5m2 | Kind::Float8e4m3fnuz | Kind::Float8e5m2fnuz => {
                let name = match kind {
                    Kind::Float8e4m3fn => "Float8_e4m3fn",
                    Kind::Float8e5m2 => "Float8_e5m2",
                    Kind::Float8e4m3fnuz => "Float8_e4m3fnuz",
                    _ => "Float8_e5m2fnuz",
                };
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: format!("couldn't find storage object {name}Storage"),
                });
            }
            _ => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: "only floating-point types are supported as the default type".into(),
                });
            }
        };
        self.library
            .check(unsafe { (self.library.set_default_dtype)(code) })
    }
    /// Invoke the original low-level launch wrapper.
    ///
    /// # Safety
    /// All arguments, output allocations, scalar ranges, pointer alignment and
    /// stream lifetimes must satisfy the pinned wrapper/kernel's full contract.
    pub unsafe fn call_unchecked(
        &self,
        name: &str,
        args: &[KernelArg<'_>],
    ) -> Result<KernelResult> {
        if let Some(guard) = &self.launch_guard {
            guard(name)?;
        }
        let name = CString::new(name).map_err(|_| Error::Native("NUL in kernel name".into()))?;
        let args: Vec<_> = args.iter().map(KernelArg::raw).collect();
        let mut value = 0;
        self.library.check(unsafe {
            (self.library.call)(name.as_ptr(), args.as_ptr(), args.len(), &mut value)
        })?;
        let meta = unsafe { CStr::from_ptr((self.library.metadata)()) }.to_string_lossy();
        Ok(KernelResult {
            value,
            metadata: (!meta.is_empty()).then(|| meta.into_owned()),
        })
    }
    /// Development metadata guard before any original launch wrapper. Ordinary
    /// execution installs none; guards must not inspect payloads or alter streams.
    #[doc(hidden)]
    pub fn with_launch_guard(mut self, guard: Option<LaunchGuard>) -> Self {
        self.launch_guard = guard;
        self
    }
    /// Thread-local metadata only; no storage ownership, read or stream change.
    pub fn rms_metadata_observer(&self) -> Result<RmsMetadataGuard> {
        self.library.check(unsafe { (self.library.rms_begin)() })?;
        Ok(RmsMetadataGuard {
            library: self.library.clone(),
            active: true,
            _same_thread: std::marker::PhantomData,
        })
    }
    pub fn current_stream(&self, device: usize) -> Result<u64> {
        let device = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        let mut stream = 0;
        self.library
            .check(unsafe { (self.library.stream)(device.into(), &mut stream) })?;
        Ok(stream)
    }
    /// The guard changes thread-local ATen state and must be dropped in LIFO
    /// order on the same thread. No implicit synchronization occurs.
    pub fn stream_guard(&self, device: usize) -> Result<StreamGuard> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        let mut handle = std::ptr::null_mut();
        self.library
            .check(unsafe { (self.library.stream_guard_new)(index.into(), &mut handle) })?;
        Ok(StreamGuard {
            library: self.library.clone(),
            handle: NonNull::new(handle)
                .ok_or_else(|| Error::Native("Null stream guard".into()))?,
        })
    }
    pub fn record_event(&self, device: usize) -> Result<CudaEvent> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        let mut handle = std::ptr::null_mut();
        self.library
            .check(unsafe { (self.library.event_record)(index.into(), &mut handle) })?;
        Ok(CudaEvent {
            library: self.library.clone(),
            handle: NonNull::new(handle).ok_or_else(|| Error::Native("Null CUDA event".into()))?,
        })
    }
    pub fn timing_event(&self) -> Result<CudaEvent> {
        let mut handle = std::ptr::null_mut();
        self.library
            .check(unsafe { (self.library.timing_event_new)(&mut handle) })?;
        Ok(CudaEvent {
            library: self.library.clone(),
            handle: NonNull::new(handle)
                .ok_or_else(|| Error::Native("Null timing event".into()))?,
        })
    }
    /// Explicit device-wide synchronization, matching torch.cuda.synchronize.
    pub fn synchronize_device(&self, device: usize) -> Result<()> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        self.library
            .check(unsafe { (self.library.device_sync)(index.into()) })
    }
    pub fn clone_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        self.library
            .check(unsafe { (self.library.tensor_clone)(tensor.as_ptr().cast(), &mut output) })?;
        if output.is_null() {
            return Err(Error::Native("Null cloned tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    pub fn empty_like(&self, tensor: &Tensor, device: Device) -> Result<Tensor> {
        let index = match device {
            Device::Cpu => -1,
            Device::Cuda(index) => i8::try_from(index)
                .map(i64::from)
                .map_err(|_| Error::Native("CUDA device index out of range".into()))?,
            _ => return Err(Error::Native("Initial target supports CPU/CUDA".into())),
        };
        let mut output = std::ptr::null_mut();
        self.library.check(unsafe {
            (self.library.tensor_empty)(tensor.as_ptr().cast(), index, &mut output)
        })?;
        if output.is_null() {
            return Err(Error::Native("Null empty tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    /// torch.empty_like(input, dtype=kind), retaining Preserve memory format.
    pub fn empty_like_dtype(&self, tensor: &Tensor, kind: tch::Kind) -> Result<Tensor> {
        let prototype = Tensor::f_empty([0], (kind, Device::Cpu))?;
        let code = unsafe { torch_sys::at_scalar_type(prototype.as_ptr().cast_mut()) as i64 };
        let mut output = std::ptr::null_mut();
        self.library.check(unsafe {
            (self.library.tensor_empty_dtype)(tensor.as_ptr().cast(), code, &mut output)
        })?;
        if output.is_null() {
            return Err(Error::Native("Null empty tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    pub fn copy_tensor(
        &self,
        destination: &mut Tensor,
        source: &Tensor,
        non_blocking: bool,
    ) -> Result<()> {
        self.library.check(unsafe {
            (self.library.tensor_copy)(
                destination.as_mut_ptr().cast(),
                source.as_ptr().cast(),
                i64::from(non_blocking),
            )
        })
    }
    pub fn tensor_type(&self, tensor: &Tensor) -> Result<TensorTypeInfo> {
        let mut info = TensorTypeInfo::default();
        self.library
            .check(unsafe { (self.library.tensor_type)(tensor.as_ptr().cast(), &mut info) })?;
        Ok(info)
    }
    /// Metadata only; no tensor payload is read or synchronized.
    /// Metadata identity only; no storage dereference or device synchronization.
    pub fn tensor_identity(&self, tensor: &Tensor) -> u64 {
        unsafe { (self.library.tensor_identity)(tensor.as_ptr().cast()) }
    }
    pub fn tensor_storage(&self, tensor: &Tensor) -> Result<TensorStorageInfo> {
        let mut info = TensorStorageInfo::default();
        self.library
            .check(unsafe { (self.library.tensor_storage)(tensor.as_ptr().cast(), &mut info) })?;
        Ok(info)
    }
    pub fn view_shell_dtype(&self, tensor: &Tensor, kind: ShellDType) -> Result<Tensor> {
        let scalar = match kind {
            ShellDType::E8M0 => 44,
            ShellDType::PackedE2M1 => 45,
        };
        let mut output = std::ptr::null_mut();
        self.library.check(unsafe {
            (self.library.tensor_shell_view)(tensor.as_ptr().cast(), scalar, &mut output)
        })?;
        if output.is_null() {
            return Err(Error::Native("Null shell-dtype tensor".into()));
        }
        Ok(unsafe { Tensor::from_ptr(output.cast()) })
    }
    pub fn generator(&self, device: Device, seed: u64) -> Result<Generator> {
        let index = match device {
            Device::Cpu => -1,
            Device::Cuda(index) => i8::try_from(index)
                .map(i64::from)
                .map_err(|_| Error::Native("CUDA device index out of range".into()))?,
            _ => {
                return Err(Error::Native(
                    "Initial target supports CPU/CUDA generators".into(),
                ));
            }
        };
        let mut handle = std::ptr::null_mut();
        self.library
            .check(unsafe { (self.library.gen_new)(index, seed, &mut handle) })?;
        Ok(Generator {
            library: self.library.clone(),
            device,
            handle: NonNull::new(handle)
                .ok_or_else(|| Error::Native("Null generator handle".into()))?,
        })
    }
}

pub struct RmsMetadataGuard {
    library: Arc<Library>,
    active: bool,
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl RmsMetadataGuard {
    pub fn finish(mut self) -> Result<RmsTemporaryMetadata> {
        let mut output = RmsTemporaryMetadata::default();
        let result = self
            .library
            .check(unsafe { (self.library.rms_end)(&mut output) });
        self.active = false;
        result?;
        if output.count != 1 {
            return Err(Error::Native(
                "Expected exactly one original fused RMSNorm metadata event".into(),
            ));
        }
        Ok(output)
    }
}
impl Drop for RmsMetadataGuard {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                (self.library.rms_end)(std::ptr::null_mut());
            }
        }
    }
}

/// !Send/!Sync: the ATen guard restores state on its creating thread.
pub struct StreamGuard {
    library: Arc<Library>,
    handle: NonNull<c_void>,
}
impl Drop for StreamGuard {
    fn drop(&mut self) {
        unsafe {
            (self.library.stream_guard_free)(self.handle.as_ptr());
        }
    }
}
pub struct CudaEvent {
    library: Arc<Library>,
    handle: NonNull<c_void>,
}
impl CudaEvent {
    pub fn record_current(&mut self, device: usize) -> Result<()> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        self.library.check(unsafe {
            (self.library.event_record_existing)(self.handle.as_ptr(), index.into())
        })
    }
    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f64> {
        let mut elapsed = 0.;
        self.library.check(unsafe {
            (self.library.event_elapsed)(self.handle.as_ptr(), end.handle.as_ptr(), &mut elapsed)
        })?;
        Ok(elapsed)
    }
    /// Enqueue a wait on the selected device's current stream. No host wait.
    pub fn wait_current(&self, device: usize) -> Result<()> {
        let index = i8::try_from(device)
            .map_err(|_| Error::Native("CUDA device index out of range".into()))?;
        self.library
            .check(unsafe { (self.library.event_wait)(self.handle.as_ptr(), index.into()) })
    }
    /// Explicit host synchronization, for lifecycle/validation boundaries.
    pub fn synchronize(&self) -> Result<()> {
        self.library
            .check(unsafe { (self.library.event_sync)(self.handle.as_ptr()) })
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            (self.library.event_free)(self.handle.as_ptr());
        }
    }
}

pub struct Generator {
    library: Arc<Library>,
    device: Device,
    handle: NonNull<c_void>,
}
/// Original torch.randn(size, dtype, layout, device, generator) using the live
/// captured tensor metadata, including native generator/device mismatch errors.
pub fn normal_from_source(source: &Tensor, generator: Option<&mut Generator>) -> Result<Tensor> {
    let value = unsafe {
        at_imagine_source_normal(
            source.as_ptr(),
            generator.map_or(std::ptr::null_mut(), |g| g.handle.as_ptr()),
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
        return Err(Error::Native("Null source normal result".into()));
    }
    Ok(unsafe { Tensor::from_ptr(value) })
}
impl Drop for Generator {
    fn drop(&mut self) {
        unsafe {
            (self.library.gen_free)(self.handle.as_ptr());
        }
    }
}
impl Generator {
    pub fn randperm(&mut self, n: i64) -> Result<Tensor> {
        let value = unsafe { at_imagine_generator_randperm(self.handle.as_ptr(), n) };
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
            return Err(Error::Native("Null generator randperm result".into()));
        }
        Ok(unsafe { Tensor::from_ptr(value) })
    }
    fn tensor_result(&self, code: i32, ptr: *mut c_void) -> Result<Tensor> {
        self.library.check(code)?;
        if ptr.is_null() {
            return Err(Error::Native("Null native tensor".into()));
        }
        // Native bridge allocates one at::Tensor handle; tch owns and frees it.
        Ok(unsafe { Tensor::from_ptr(ptr.cast()) })
    }
    pub fn state(&self) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe { (self.library.gen_state)(self.handle.as_ptr(), &mut output) };
        self.tensor_result(code, output)
    }
    pub fn set_state(&mut self, state: &Tensor) -> Result<()> {
        self.library.check(unsafe {
            (self.library.gen_set_state)(self.handle.as_ptr(), state.as_ptr().cast())
        })
    }
    pub fn randn(&mut self, shape: &[i64], kind: Kind) -> Result<Tensor> {
        let prototype = Tensor::f_empty([0], (kind, self.device))?;
        let scalar_type = unsafe { torch_sys::at_scalar_type(prototype.as_ptr().cast_mut()) };
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.gen_normal)(
                self.handle.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                scalar_type.into(),
                &mut output,
            )
        };
        self.tensor_result(code, output)
    }
    pub fn random_bytes(&mut self, shape: &[i64]) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.gen_bytes)(
                self.handle.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                &mut output,
            )
        };
        self.tensor_result(code, output)
    }
    pub fn randint(&mut self, high: i64, shape: &[i64]) -> Result<Tensor> {
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.gen_int)(
                self.handle.as_ptr(),
                high,
                shape.as_ptr(),
                shape.len(),
                &mut output,
            )
        };
        self.tensor_result(code, output)
    }
    pub fn rand(&mut self, shape: &[i64], kind: Kind) -> Result<Tensor> {
        let prototype = Tensor::f_empty([0], (kind, self.device))?;
        let scalar_type = unsafe { torch_sys::at_scalar_type(prototype.as_ptr().cast_mut()) };
        let mut output = std::ptr::null_mut();
        let code = unsafe {
            (self.library.gen_uniform)(
                self.handle.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                scalar_type.into(),
                &mut output,
            )
        };
        self.tensor_result(code, output)
    }
}
