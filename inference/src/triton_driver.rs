//! Owned C ABI to the frozen NVIDIA driver load/launch bodies.
use crate::native::Error;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
    sync::Arc,
};
type Load = unsafe extern "C" fn(
    *const c_char,
    *const u8,
    i32,
    i32,
    *mut *mut c_void,
    *mut *mut c_void,
    *mut i32,
) -> i32;
type Pointer = unsafe extern "C" fn(*mut u64) -> i32;
type Launch = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
) -> i32;
type LastError = unsafe extern "C" fn() -> *const c_char;
struct Library {
    handle: NonNull<c_void>,
    load: Load,
    pointer: Pointer,
    launch: Launch,
    error: LastError,
}
// Like CUDA's original driver, these functions may be invoked across host
// threads. Current-context and stream selection remain the caller's contract.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle.as_ptr());
        }
    }
}
#[derive(Clone)]
pub struct Driver {
    library: Arc<Library>,
}
pub struct Kernel {
    _driver: Driver,
    pub module: NonNull<c_void>,
    pub function: NonNull<c_void>,
    pub registers: i32,
    pub spills: i32,
    pub max_threads: i32,
}
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}
#[derive(Clone)]
pub enum Scalar {
    /// Keep the original TensorImpl alive and extract its current data_ptr at
    /// CompiledKernel's original post-enter-hook argument boundary.
    TensorObject(crate::execution_value::Value),
    Pointer(u64),
    TensorPointer(u64),
    I32(i32),
    I64(i64),
    U64(u64),
    F32(f32),
    Bool(bool),
}
impl Driver {
    /// # Safety
    /// Load only the verified development driver built from the frozen source.
    pub unsafe fn open(path: &Path) -> Result<Self, Error> {
        let path =
            CString::new(path.as_os_str().as_bytes()).map_err(|e| Error::Native(e.to_string()))?;
        let handle =
            NonNull::new(unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) })
                .ok_or_else(|| {
                    Error::Native(
                        unsafe { CStr::from_ptr(libc::dlerror()) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                })?;
        unsafe fn symbol<T: Copy>(h: NonNull<c_void>, name: &CStr) -> Result<T, Error> {
            let p = unsafe { libc::dlsym(h.as_ptr(), name.as_ptr()) };
            if p.is_null() {
                return Err(Error::Native(format!(
                    "Missing Triton driver symbol {name:?}"
                )));
            }
            Ok(unsafe { std::mem::transmute_copy(&p) })
        }
        let library = (|| unsafe {
            Ok::<_, Error>(Library {
                handle,
                load: symbol(handle, c"imagine_triton_load")?,
                pointer: symbol(handle, c"imagine_triton_pointer")?,
                launch: symbol(handle, c"imagine_triton_launch")?,
                error: symbol(handle, c"imagine_triton_error")?,
            })
        })();
        match library {
            Ok(library) => Ok(Self {
                library: Arc::new(library),
            }),
            Err(e) => {
                unsafe {
                    libc::dlclose(handle.as_ptr());
                }
                Err(e)
            }
        }
    }
    fn status(&self, status: i32) -> Result<(), Error> {
        if status == 0 {
            return Ok(());
        }
        let message = unsafe { CStr::from_ptr((self.library.error)()) }
            .to_string_lossy()
            .into_owned();
        Err(Error::ReferenceException {
            class: if status == -2 {
                "ValueError"
            } else {
                "RuntimeError"
            },
            message,
        })
    }
    /// # Safety
    /// The current CUDA context/device must match the original launch contract;
    /// data must be an original-compatible cubin and shared its exact metadata.
    pub unsafe fn load(
        &self,
        name: &str,
        data: &[u8],
        shared: i32,
        device: i32,
    ) -> Result<Kernel, Error> {
        let name = CString::new(name).map_err(|_| Error::ReferenceException {
            class: "ValueError",
            message: "embedded null character".into(),
        })?;
        // Original Python bytes always provide one trailing NUL, including
        // an empty cache image. Do not pass an empty Rust slice sentinel.
        let mut data = data.to_vec();
        data.push(0);
        let (mut module, mut function) = (std::ptr::null_mut(), std::ptr::null_mut());
        let mut info = [0; 3];
        self.status(unsafe {
            (self.library.load)(
                name.as_ptr(),
                data.as_ptr(),
                shared,
                device,
                &mut module,
                &mut function,
                info.as_mut_ptr(),
            )
        })?;
        Ok(Kernel {
            _driver: self.clone(),
            module: NonNull::new(module)
                .ok_or_else(|| Error::Native("Null loaded CUDA module".into()))?,
            function: NonNull::new(function)
                .ok_or_else(|| Error::Native("Null loaded CUDA function".into()))?,
            registers: info[0],
            spills: info[1],
            max_threads: info[2],
        })
    }
    /// Original typed tensor-pointer validation. Zero remains a valid nullptr;
    /// no dequantization, host copying or pointer reinterpretation is performed.
    /// # Safety
    /// The pointer must come from an owned argument with the original lifetime.
    pub unsafe fn tensor_pointer(&self, mut pointer: u64) -> Result<u64, Error> {
        self.status(unsafe { (self.library.pointer)(&mut pointer) })?;
        Ok(pointer)
    }
    /// # Safety
    /// Scalar widths/order, appended scratch pointers, capacities, current CUDA
    /// context, stream lifetime and launch metadata must match the cubin ABI.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: &Kernel,
        grid: [i32; 3],
        warps: i32,
        ctas: i32,
        cooperative: bool,
        pdl: bool,
        shared: i32,
        stream: u64,
        args: &mut [Scalar],
    ) -> Result<(), Error> {
        if args
            .iter()
            .any(|value| matches!(value, Scalar::TensorObject(_)))
        {
            return Err(Error::Native(
                "Tensor objects require CompiledKernel argument extraction".into(),
            ));
        }
        let mut ptrs = args
            .iter_mut()
            .map(|v| match v {
                Scalar::Pointer(v) | Scalar::TensorPointer(v) => {
                    std::ptr::from_mut(v).cast::<c_void>()
                }
                Scalar::I32(v) => std::ptr::from_mut(v).cast(),
                Scalar::I64(v) => std::ptr::from_mut(v).cast(),
                Scalar::U64(v) => std::ptr::from_mut(v).cast(),
                Scalar::F32(v) => std::ptr::from_mut(v).cast(),
                Scalar::Bool(v) => std::ptr::from_mut(v).cast(),
                Scalar::TensorObject(_) => {
                    unreachable!("Object arguments were checked before raw driver launch")
                }
            })
            .collect::<Vec<_>>();
        self.status(unsafe {
            (self.library.launch)(
                grid[0],
                grid[1],
                grid[2],
                warps,
                ctas,
                cooperative.into(),
                pdl.into(),
                shared,
                stream as *mut c_void,
                kernel.function.as_ptr(),
                ptrs.as_mut_ptr(),
            )
        })
    }
}
