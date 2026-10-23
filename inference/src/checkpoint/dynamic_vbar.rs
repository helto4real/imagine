//! Process-owned AIMDO dynamic VRAM and operation-scoped page pins.
//!
//! Mirrors comfy_aimdo.control/model_vbar. Virtual ranges do not own resident
//! contents: callers must compare fault signatures before reusing tensor views.
use crate::native::Error;
use std::{
    ffi::{CStr, CString, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex, MutexGuard},
};

const PAGE_SIZE: u64 = 32 * 1024 * 1024;
type Allocate = unsafe extern "C" fn(*mut c_void, u64, i32) -> *mut c_void;
type Get = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
type Free = unsafe extern "C" fn(*mut c_void, *mut c_void);
type FaultFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, *mut u32) -> i32;
type RangeFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64);
type Prioritize = unsafe extern "C" fn(*mut c_void, *mut c_void, u64);
type LoadedSize = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;

struct Api {
    handle: NonNull<c_void>,
    allocate: Allocate,
    get: Get,
    free: Free,
    fault: FaultFn,
    unpin: RangeFn,
    prioritize: Prioritize,
    deprioritize: Free,
    loaded_size: LoadedSize,
}

impl Drop for Api {
    fn drop(&mut self) {
        unsafe { libc::dlclose(self.handle.as_ptr()) };
    }
}

struct Runtime {
    api: Api,
    devices: Vec<NonNull<c_void>>,
    calls: Mutex<()>,
}

// Raw handles are process-owned. Native control calls are serialized; CUDA
// operation/view lifetimes are the caller's separate responsibility.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

// Keep the allocator hooks, contexts and shared object alive for the process.
// Deinitializing after a component drops could invalidate later ATen allocations
// or another component's live views. Failed initialization is not cached.
static RUNTIME: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

fn native(message: impl Into<String>) -> Error {
    Error::Native(message.into())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Native calls cannot unwind through the FFI boundary. Recover poisoning
    // from a caller panic so guards can still unpin/free their owned ranges.
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

unsafe fn symbol<T: Copy>(handle: NonNull<c_void>, name: &CStr) -> Result<T, Error> {
    let pointer = unsafe { libc::dlsym(handle.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        return Err(native(format!(
            "Missing AIMDO dynamic VRAM symbol: {}",
            name.to_string_lossy()
        )));
    }
    Ok(unsafe { std::mem::transmute_copy(&pointer) })
}

unsafe fn initialize(path: &Path) -> Result<Arc<Runtime>, Error> {
    let mut shared = lock(&RUNTIME);
    if let Some(runtime) = shared.as_ref() {
        return Ok(Arc::clone(runtime));
    }
    let filename = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| native("NUL in AIMDO library path"))?;
    let handle = NonNull::new(unsafe {
        libc::dlopen(filename.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL)
    })
    .ok_or_else(|| {
        let error = unsafe { libc::dlerror() };
        native(if error.is_null() {
            format!("Could not load AIMDO library {}", path.display())
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        })
    })?;
    let bindings = (|| unsafe {
        Ok(Api {
            handle,
            allocate: symbol(handle, c"vbar_allocate")?,
            get: symbol(handle, c"vbar_get")?,
            free: symbol(handle, c"vbar_free")?,
            fault: symbol(handle, c"vbar_fault")?,
            unpin: symbol(handle, c"vbar_unpin")?,
            prioritize: symbol(handle, c"vbar_prioritize")?,
            deprioritize: symbol(handle, c"vbar_deprioritize")?,
            loaded_size: symbol(handle, c"vbar_loaded_size")?,
        })
    })();
    let api = match bindings {
        Ok(api) => api,
        Err(error) => {
            unsafe { libc::dlclose(handle.as_ptr()) };
            return Err(error);
        }
    };
    type PlatInit = unsafe extern "C" fn() -> bool;
    type Init = unsafe extern "C" fn(*const i32, *const u64, usize) -> bool;
    type GetDevice = unsafe extern "C" fn(i32) -> *mut c_void;
    type Cleanup = unsafe extern "C" fn();
    type SetPressure = unsafe extern "C" fn(bool);
    let (plat_init, init, get_device, cleanup, plat_cleanup, set_pressure): (
        PlatInit,
        Init,
        GetDevice,
        Cleanup,
        Cleanup,
        SetPressure,
    ) = unsafe {
        (
            symbol(handle, c"plat_init")?,
            symbol(handle, c"init")?,
            symbol(handle, c"get_devctx")?,
            symbol(handle, c"cleanup")?,
            symbol(handle, c"plat_cleanup")?,
            symbol(handle, c"set_nvml_pressure")?,
        )
    };
    // ComfyUI initializes every visible torch CUDA device once. Zero means no
    // user-added headroom; leave the library's default simple budget unchanged.
    let count = tch::Cuda::device_count();
    let count = i32::try_from(count).map_err(|_| native("Invalid CUDA device count for AIMDO"))?;
    if count <= 0 {
        return Err(native("AIMDO dynamic VRAM requires a visible CUDA device"));
    }
    let devices: Vec<i32> = (0..count).collect();
    let headrooms = vec![0_u64; devices.len()];
    unsafe { set_pressure(true) };
    if !unsafe { plat_init() } {
        return Err(native("AIMDO plat_init failed"));
    }
    if !unsafe { init(devices.as_ptr(), headrooms.as_ptr(), devices.len()) } {
        unsafe { plat_cleanup() };
        return Err(native("AIMDO device initialization failed"));
    }
    let contexts = devices
        .into_iter()
        .map(|device| {
            NonNull::new(unsafe { get_device(device) })
                .ok_or_else(|| native(format!("AIMDO has no context for CUDA device {device}")))
        })
        .collect::<Result<Vec<_>, _>>();
    let contexts = match contexts {
        Ok(contexts) => contexts,
        Err(error) => {
            unsafe {
                cleanup();
                plat_cleanup();
            }
            return Err(error);
        }
    };
    let runtime = Arc::new(Runtime {
        api,
        devices: contexts,
        calls: Mutex::new(()),
    });
    *shared = Some(Arc::clone(&runtime));
    Ok(runtime)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub offset: u64,
    pub size: u64,
}

pub(crate) struct Vbar {
    runtime: Arc<Runtime>,
    context: NonNull<c_void>,
    pointer: NonNull<c_void>,
    base: u64,
    capacity: u64,
    offset: Mutex<u64>,
}

// The owner is immutable apart from its protected suballocation cursor. Native
// mutations are serialized by Runtime::calls; each PinGuard retains its owner.
unsafe impl Send for Vbar {}
unsafe impl Sync for Vbar {}

impl Vbar {
    /// Reserve virtual address space; this does not populate all model weights.
    /// The first call selects the process-wide AIMDO implementation.
    ///
    /// # Safety
    /// `path` must identify the trusted AIMDO library implementing this ABI.
    /// The caller must use the matching CUDA runtime/device and synchronize all
    /// GPU use before dropping the last owner/view. Per-operation unpin uses
    /// AIMDO's internal synchronization, as in Comfy's uncast_bias_weight.
    pub(crate) unsafe fn load(path: &Path, device: usize, size: u64) -> Result<Arc<Self>, Error> {
        let runtime = unsafe { initialize(path) }?;
        let context = *runtime
            .devices
            .get(device)
            .ok_or_else(|| native(format!("AIMDO CUDA device {device} is not initialized")))?;
        let device =
            i32::try_from(device).map_err(|_| native("CUDA device index exceeds AIMDO ABI"))?;
        let (pointer, base) = {
            let _calls = lock(&runtime.calls);
            let pointer =
                NonNull::new(unsafe { (runtime.api.allocate)(context.as_ptr(), size, device) })
                    .ok_or_else(|| {
                        native(format!(
                            "AIMDO VBAR allocation failed ({size} virtual bytes)"
                        ))
                    })?;
            let base = unsafe { (runtime.api.get)(context.as_ptr(), pointer.as_ptr()) };
            if base == 0
                || base
                    .checked_add(size)
                    .and_then(|end| usize::try_from(end).ok())
                    .is_none()
            {
                unsafe { (runtime.api.free)(context.as_ptr(), pointer.as_ptr()) };
                return Err(native("AIMDO returned an invalid VBAR address range"));
            }
            (pointer, base)
        };
        Ok(Arc::new(Self {
            runtime,
            context,
            pointer,
            base,
            capacity: size,
            offset: Mutex::new(0),
        }))
    }

    pub(crate) fn allocate(&self, size: u64) -> Result<Range, Error> {
        let mut cursor = lock(&self.offset);
        let (offset, end) = allocation(*cursor, size, self.capacity)?;
        *cursor = end;
        Ok(Range { offset, size })
    }

    pub(crate) fn address(&self, offset: u64) -> *const u8 {
        assert!(
            offset <= self.capacity,
            "VBAR offset outside reserved range"
        );
        (self.base + offset) as usize as *const u8
    }

    /// A successful signature owns an active pin until the guard is dropped.
    /// OOM returns no signature/guard, matching Python's temporary-buffer path.
    /// Keep the guard through the last GPU use, including failure cleanup.
    pub(crate) fn acquire(self: &Arc<Self>, offset: u64, size: u64) -> Result<Fault, Error> {
        if offset
            .checked_add(size)
            .is_none_or(|end| end > self.capacity)
        {
            return Err(native("AIMDO fault range exceeds the reserved VBAR"));
        }
        let entries = usize::try_from(size / PAGE_SIZE + 2)
            .map_err(|_| native("AIMDO fault signature exceeds addressable memory"))?;
        let mut signature = vec![0_u32; entries];
        let result = {
            let _calls = lock(&self.runtime.calls);
            unsafe {
                (self.runtime.api.fault)(
                    self.context.as_ptr(),
                    self.pointer.as_ptr(),
                    offset,
                    size,
                    signature.as_mut_ptr(),
                )
            }
        };
        match result {
            0 => Ok(Fault {
                signature: Some(signature),
                _guard: Some(PinGuard {
                    owner: Arc::clone(self),
                    offset,
                    size,
                }),
            }),
            1 => Ok(Fault {
                signature: None,
                _guard: None,
            }),
            result => Err(native(format!(
                "AIMDO VBAR fault failed: {result} (offset {offset}, size {size})"
            ))),
        }
    }

    pub(crate) fn prioritize(&self) {
        let _calls = lock(&self.runtime.calls);
        unsafe {
            (self.runtime.api.prioritize)(self.context.as_ptr(), self.pointer.as_ptr(), u64::MAX)
        };
    }

    pub(crate) fn deprioritize(&self) {
        let _calls = lock(&self.runtime.calls);
        unsafe { (self.runtime.api.deprioritize)(self.context.as_ptr(), self.pointer.as_ptr()) };
    }

    pub(crate) fn loaded_size(&self) -> usize {
        let _calls = lock(&self.runtime.calls);
        unsafe { (self.runtime.api.loaded_size)(self.context.as_ptr(), self.pointer.as_ptr()) }
    }
}

impl Drop for Vbar {
    fn drop(&mut self) {
        self.deprioritize();
        let _calls = lock(&self.runtime.calls);
        unsafe { (self.runtime.api.free)(self.context.as_ptr(), self.pointer.as_ptr()) };
    }
}

#[must_use = "retain the fault guard until GPU use of these pages has completed"]
pub(crate) struct Fault {
    pub signature: Option<Vec<u32>>,
    _guard: Option<PinGuard>,
}

#[must_use = "dropping this guard unpins the GPU pages"]
pub(crate) struct PinGuard {
    owner: Arc<Vbar>,
    offset: u64,
    size: u64,
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        let _calls = lock(&self.owner.runtime.calls);
        unsafe {
            (self.owner.runtime.api.unpin)(
                self.owner.context.as_ptr(),
                self.owner.pointer.as_ptr(),
                self.offset,
                self.size,
            );
        }
    }
}

fn allocation(cursor: u64, size: u64, capacity: u64) -> Result<(u64, u64), Error> {
    let offset = cursor
        .checked_add(511)
        .map(|value| value & !511)
        .ok_or_else(|| native("AIMDO VBAR suballocation offset overflow"))?;
    let end = offset
        .checked_add(size)
        .filter(|end| *end <= capacity)
        .ok_or_else(|| native("AIMDO VBAR virtual address range exhausted"))?;
    Ok((offset, end))
}

#[cfg(test)]
mod tests {
    use super::allocation;

    #[test]
    fn suballocation_matches_python_alignment() {
        assert_eq!(allocation(0, 17, 4096).unwrap(), (0, 17));
        assert_eq!(allocation(17, 1024, 4096).unwrap(), (512, 1536));
        assert_eq!(allocation(1536, 1, 4096).unwrap(), (1536, 1537));
    }

    #[test]
    fn suballocation_rejects_only_overflow_and_exhaustion() {
        assert_eq!(allocation(512, 512, 1024).unwrap(), (512, 1024));
        assert!(allocation(513, 1, 1024).is_err());
        assert!(allocation(u64::MAX, 0, u64::MAX).is_err());
        assert!(allocation(0, u64::MAX, 1024).is_err());
    }
}
