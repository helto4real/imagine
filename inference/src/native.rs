//! Audited ownership boundary to the standalone Kitchen/ATen C ABI.
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
    Torch(tch::TchError),
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
            Self::Torch(error) => error.fmt(f),
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
    fn check(&self, code: i32) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // Error storage is thread-local and is copied before another bridge call.
        let text = unsafe { CStr::from_ptr((self.error)()) }
            .to_string_lossy()
            .into_owned();
        match code {
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
pub struct ScaledMm<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub scale_a: &'a Tensor,
    pub scale_b: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub output_kind: Option<Kind>,
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
            Err(Error::ReferenceException {
                class: match code {
                    -2 => "ValueError",
                    -3 => "IndexError",
                    -4 => "TypeError",
                    -5 => "NotImplementedError",
                    -9 => "OutOfMemoryError",
                    -10 => "AcceleratorError",
                    _ => "RuntimeError",
                },
                message: unsafe { CStr::from_ptr((self.library.error)()) }
                    .to_string_lossy()
                    .into_owned(),
            })
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
            _ => "RuntimeError",
        };
        Err(Error::ReferenceException {
            class,
            message: unsafe { CStr::from_ptr((self.library.error)()) }
                .to_string_lossy()
                .into_owned(),
        })
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
    fn aten_tensor_result(&self, code: i32, output: *mut c_void) -> Result<Tensor> {
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
            return Err(Error::ReferenceException { class, message });
        }
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
                }
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
impl Drop for Generator {
    fn drop(&mut self) {
        unsafe {
            (self.library.gen_free)(self.handle.as_ptr());
        }
    }
}
impl Generator {
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
