//! Frozen ComfyUI checkpoint loading. ModelMMAP is the original AIMDO C ABI;
//! ATen Storage, rather than a loader scope, owns each nonempty mapped view.
use crate::native::{Error, Kitchen};
use serde::de::{MapAccess, Visitor};
use serde_json::Value as Json;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{Arc, Mutex},
};
use tch::{Kind, Tensor};
pub mod component;
pub mod detection;
mod detector_program;
mod detector_values;
mod diffusers_program;
mod embedding;
mod metadata_json;
pub mod model_registry;
mod prepared;
pub mod rebuild_versions;
mod registry_data;
pub mod state_dict;
pub mod storage_rebuild;
pub mod torch_archive;
pub mod unet_names;
mod unet_names_program;
pub mod weights_only;
pub mod weights_only_globals;
pub use embedding::load_quantized_embedding;
mod quantized;
pub use prepared::{LinearPolicy, PreparedParameters};
pub use quantized::{LoadOptions, QuantizedModule, bind_comfy_layouts, load_quantized_module};

type Allocate = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type Get = unsafe extern "C" fn(*mut c_void) -> *mut u8;
type FileHandle = unsafe extern "C" fn(*mut c_void) -> u64;
type Free = unsafe extern "C" fn(*mut c_void);
struct AimdoLibrary {
    handle: NonNull<c_void>,
    allocate: Allocate,
    get: Get,
    file: FileHandle,
    free: Free,
}
// Only the immutable symbol table is shared. Every mapping owns a separate
// native ModelMMAP, and file-slice reads share that mapping's source lock.
unsafe impl Send for AimdoLibrary {}
unsafe impl Sync for AimdoLibrary {}
impl Drop for AimdoLibrary {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle.as_ptr());
        }
    }
}
#[derive(Clone)]
pub struct Aimdo(Arc<AimdoLibrary>);
impl Aimdo {
    /// # Safety
    /// `path` must select the trusted, frozen AIMDO library with this ABI.
    /// Loading a shared object can execute native initializers.
    pub unsafe fn load(path: &Path) -> Result<Self, Error> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| Error::Native("NUL in AIMDO library path".into()))?;
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
                    "Missing AIMDO symbol: {}",
                    name.to_string_lossy()
                )));
            }
            Ok(unsafe { std::mem::transmute_copy(&p) })
        }
        let result = (|| unsafe {
            Ok(AimdoLibrary {
                handle,
                allocate: symbol(handle, c"model_mmap_allocate")?,
                get: symbol(handle, c"model_mmap_get")?,
                file: symbol(handle, c"model_mmap_get_file_handle")?,
                free: symbol(handle, c"model_mmap_deallocate")?,
            })
        })();
        match result {
            Ok(library) => Ok(Self(Arc::new(library))),
            Err(e) => {
                unsafe {
                    libc::dlclose(handle.as_ptr());
                }
                Err(e)
            }
        }
    }
    fn map(&self, path: &Path, size: usize) -> Result<Arc<Mapping>, Error> {
        let filename = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| exception("ValueError", "embedded null byte"))?;
        let state =
            NonNull::new(unsafe { (self.0.allocate)(filename.as_ptr()) }).ok_or_else(|| {
                exception(
                    "RuntimeError",
                    format!("ModelMMAP allocation failed for {}", path.display()),
                )
            })?;
        let data = unsafe { (self.0.get)(state.as_ptr()) };
        Ok(Arc::new(Mapping {
            library: self.clone(),
            state,
            data,
            size,
            lock: Mutex::new(()),
        }))
    }
}
struct Mapping {
    library: Aimdo,
    state: NonNull<c_void>,
    data: *mut u8,
    size: usize,
    lock: Mutex<()>,
}
// AIMDO's mapping is immutable; native mapping disposal happens after the last
// Storage/file-slice owner drops. Callers must not mutate read-only model data.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            (self.library.0.free)(self.state.as_ptr());
        }
    }
}
impl Mapping {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }
}
#[derive(Clone)]
pub struct TensorFileSlice {
    mapping: Arc<Mapping>,
    pub offset: u64,
    pub size: u64,
}
impl TensorFileSlice {
    pub fn file_handle(&self) -> u64 {
        unsafe { (self.mapping.library.0.file)(self.mapping.state.as_ptr()) }
    }
    /// Original serialized access to the shared file handle. No tensor copy or
    /// device transfer is selected implicitly here.
    pub fn with_file_lock<T>(&self, f: impl FnOnce(u64, u64, u64) -> T) -> T {
        let _guard = self.mapping.lock.lock().unwrap();
        f(self.file_handle(), self.offset, self.size)
    }
}
/// Storage annotations follow views and shallow copies, while a real clone or
/// dtype/device conversion creates a new Storage without a file-slice tag.
pub fn tensor_file_slice(
    tensor: &Tensor,
    native: &Kitchen,
) -> Result<Option<TensorFileSlice>, Error> {
    Ok(native
        .buffer_owner(tensor)?
        .and_then(|owner| owner.downcast::<TensorFileSlice>().ok())
        .map(|slice| (*slice).clone()))
}
pub struct CheckpointTensor {
    pub tensor: Tensor,
    pub file_slice: Option<TensorFileSlice>,
    pub source_path: PathBuf,
}
pub struct Safetensors {
    /// Preserve Python JSON/dict insertion order, including duplicate-key updates.
    pub tensors: Vec<(String, CheckpointTensor)>,
    pub metadata: Json,
    // Keep source JSON order for native component consumers. The existing
    // serde view remains available to the accepted product loader.
    source_metadata: metadata_json::Json,
}
impl Safetensors {
    pub fn get(&self, name: &str) -> Option<&CheckpointTensor> {
        self.tensors
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }
    /// Source `utils.load_safetensors`, when AIMDO is enabled. This entry point
    /// does not guess the global policy or silently choose the safe_open branch.
    pub fn load_aimdo(path: &Path, aimdo: &Aimdo, native: &Kitchen) -> Result<Self, Error> {
        let size = std::fs::metadata(path)
            .map_err(|e| Error::Native(e.to_string()))?
            .len();
        if size < 8 {
            return Err(incomplete("The safetensors header is incomplete.", path));
        }
        let size = usize::try_from(size).map_err(|_| Error::BindingGap {
            backend: "checkpoint".into(),
            operation: "File exceeds addressable memory".into(),
        })?;
        let mapping = aimdo.map(path, size)?;
        let header_size = u64::from_le_bytes(mapping.bytes()[..8].try_into().unwrap());
        if header_size > 100_000_000 {
            return Err(invalid("The safetensors header is too large.", path));
        }
        let base = 8 + header_size as usize;
        if base > size {
            return Err(incomplete("The safetensors header is incomplete.", path));
        }
        let (header, source_metadata) =
            parse_header_with_metadata(&mapping.bytes()[8..base], path)?;
        let source_path = std::fs::canonicalize(path).map_err(|e| Error::Native(e.to_string()))?;
        let mut tensors = vec![];
        let mut metadata = Json::Object(Default::default());
        for (name, info) in header {
            if name == "__metadata__" {
                metadata = info;
                continue;
            }
            let offsets = member(&info, "data_offsets")?
                .as_array()
                .ok_or_else(|| exception("TypeError", "cannot unpack non-iterable int object"))?;
            if offsets.len() != 2 {
                return Err(exception(
                    "ValueError",
                    if offsets.len() < 2 {
                        format!(
                            "not enough values to unpack (expected 2, got {})",
                            offsets.len()
                        )
                    } else {
                        "too many values to unpack (expected 2)".into()
                    },
                ));
            }
            let dtype_name = member(&info, "dtype")?
                .as_str()
                .ok_or_else(|| exception("TypeError", "unhashable type: 'dict'"))?;
            let (scalar, bytes, _kind) = dtype(dtype_name)?;
            let start = offsets[0]
                .as_i64()
                .ok_or_else(|| gap("Non-integer safetensors offsets"))?;
            let end = offsets[1]
                .as_i64()
                .ok_or_else(|| gap("Non-integer safetensors offsets"))?;
            if start < 0 || end < start {
                return Err(invalid(
                    &format!("Tensor '{name}' has invalid data offsets."),
                    path,
                ));
            }
            if end as u64 > (size - base) as u64 {
                return Err(incomplete(
                    &format!("Tensor '{name}' extends past the end of the file."),
                    path,
                ));
            }
            let shape = member(&info, "shape")?
                .as_array()
                .ok_or_else(|| gap("Non-sequence safetensors shape"))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| gap("Non-integer safetensors shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let numel = shape
                .iter()
                .try_fold(1_i128, |acc, d| acc.checked_mul(i128::from(*d)))
                .ok_or_else(|| gap("Unbounded Python shape product"))?;
            if numel.checked_mul(bytes.into()) != Some(i128::from(end - start)) {
                return Err(invalid(
                    &format!("Tensor '{name}' does not match its declared shape and dtype."),
                    path,
                ));
            }
            let (tensor, file_slice) = if start == end {
                // `empty` owns its own Storage and has no file-slice annotation.
                let tensor = native.empty_scalar(&shape, scalar)?;
                (tensor, None)
            } else {
                let slice = TensorFileSlice {
                    mapping: mapping.clone(),
                    offset: (base as i64 + start) as u64,
                    size: (end - start) as u64,
                };
                let tensor = unsafe {
                    native.tensor_from_owned_buffer(
                        mapping.data.add(base + start as usize),
                        &[(end - start) / bytes],
                        scalar,
                        Arc::new(slice.clone()),
                    )?
                }
                .f_view(shape.as_slice())?;
                (tensor, Some(slice))
            };
            tensors.push((
                name,
                CheckpointTensor {
                    tensor,
                    file_slice,
                    source_path: source_path.clone(),
                },
            ));
        }
        Ok(Self {
            tensors,
            metadata,
            source_metadata,
        })
    }
}
fn gap(operation: &str) -> Error {
    Error::BindingGap {
        backend: "checkpoint".into(),
        operation: operation.into(),
    }
}
fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn invalid(message: &str, path: &Path) -> Error {
    exception(
        "ValueError",
        format!(
            "{message}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.",
            path.display()
        ),
    )
}
fn incomplete(message: &str, path: &Path) -> Error {
    exception(
        "ValueError",
        format!(
            "{message}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.",
            path.display()
        ),
    )
}
fn member<'a>(v: &'a Json, name: &str) -> Result<&'a Json, Error> {
    v.get(name)
        .ok_or_else(|| exception("KeyError", format!("'{name}'")))
}
fn dtype(name: &str) -> Result<(i64, i64, Option<Kind>), Error> {
    Ok(match name {
        "F64" => (7, 8, Some(Kind::Double)),
        "F32" => (6, 4, Some(Kind::Float)),
        "F16" => (5, 2, Some(Kind::Half)),
        "BF16" => (15, 2, Some(Kind::BFloat16)),
        "I64" => (4, 8, Some(Kind::Int64)),
        "I32" => (3, 4, Some(Kind::Int)),
        "I16" => (2, 2, Some(Kind::Int16)),
        "I8" => (1, 1, Some(Kind::Int8)),
        "U8" => (0, 1, Some(Kind::Uint8)),
        "BOOL" => (11, 1, Some(Kind::Bool)),
        "F8_E4M3" => (24, 1, Some(Kind::Float8e4m3fn)),
        "F8_E5M2" => (23, 1, Some(Kind::Float8e5m2)),
        "C64" => (9, 8, Some(Kind::ComplexFloat)),
        "U16" => (27, 2, None),
        "U32" => (28, 4, None),
        "U64" => (29, 8, None),
        _ => return Err(exception("KeyError", format!("'{name}'"))),
    })
}
struct OrderedHeader;
impl<'de> Visitor<'de> for OrderedHeader {
    type Value = Vec<(String, Json)>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON object")
    }
    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut items: Vec<(String, Json)> = vec![];
        while let Some((key, value)) = map.next_entry::<String, Json>()? {
            if let Some((_, old)) = items.iter_mut().find(|(name, _)| *name == key) {
                *old = value;
            } else {
                items.push((key, value));
            }
        }
        Ok(items)
    }
}
/// Header-only parsing preserves Python dictionary order and duplicate updates.
pub fn parse_header(bytes: &[u8], path: &Path) -> Result<Vec<(String, Json)>, Error> {
    parse_header_with_metadata(bytes, path).map(|(header, _)| header)
}
type ParsedHeader = (Vec<(String, Json)>, metadata_json::Json);
fn parse_header_with_metadata(bytes: &[u8], path: &Path) -> Result<ParsedHeader, Error> {
    use serde::de::Deserializer;
    let text = crate::python_unicode_codec::utf8(bytes, false)
        .map_err(|error| header_decode_error(error, path))?;
    let text = crate::patching::options::SourceText::get(&text)
        .unwrap()
        .utf8()
        .unwrap();
    let source = metadata_json::loads(&crate::patching::options::Options::Atom(
        crate::execution_value::Value::Text(text.clone()),
    ))
    .map_err(|error| header_decode_error(error, path))?;
    if !matches!(source, metadata_json::Json::Dict(_)) {
        return Err(invalid("The safetensors header is invalid.", path));
    }
    let source_metadata = match &source {
        metadata_json::Json::Dict(values) => values
            .iter()
            .find(|(name, _)| name == &metadata_json::points("__metadata__"))
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| metadata_json::Json::Dict(vec![])),
        _ => unreachable!(),
    };
    // The typed legacy header API admits ordinary finite JSON. Raw surrogate,
    // arbitrary-integer and nonfinite metadata need their source-value binding;
    // never coerce them or replace source syntax errors with Serde diagnostics.
    let mut decoder = serde_json::Deserializer::from_str(&text);
    let header = (&mut decoder)
        .deserialize_map(OrderedHeader)
        .map_err(|_| gap("Extended safetensors JSON requires the raw source-value binding"))?;
    decoder
        .end()
        .map_err(|_| gap("Source-validated safetensors JSON representation"))?;
    Ok((header, source_metadata))
}
fn header_decode_error(error: Error, path: &Path) -> Error {
    if !matches!(
        error.root(),
        Error::ReferenceException {
            class: "UnicodeDecodeError" | "JSONDecodeError",
            ..
        }
    ) {
        return error;
    }
    let cause = error.into_exception();
    let raised = invalid(&cause.source().to_string(), path)
        .during_handling(Error::Retained(cause.clone()))
        .into_exception();
    raised
        .set_cause(crate::source_exception::ExceptionLink::Exception(cause))
        .expect("an exception is a valid explicit cause");
    Error::Retained(raised)
}
