//! The original PyTorchFileReader's native ZIP reader, without its pickle VM.
use crate::{native::Error, python_unicode_codec};
use std::{
    ffi::{CStr, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
};
use tch::Tensor;
unsafe extern "C" {
    fn at_imagine_storage_resizable(value: *const torch_sys::C_tensor) -> i32;
    fn at_imagine_archive_open(path: *const u8, size: usize) -> *mut c_void;
    fn at_imagine_archive_free(reader: *mut c_void);
    fn at_imagine_archive_has(reader: *mut c_void, key: *const u8, size: usize) -> i32;
    fn at_imagine_archive_record(reader: *mut c_void, key: *const u8, size: usize) -> *mut c_void;
    fn at_imagine_archive_data(record: *mut c_void, size: *mut usize) -> *const u8;
    fn at_imagine_archive_record_free(record: *mut c_void);
    fn at_imagine_archive_records(reader: *mut c_void, size: *mut usize) -> *mut c_void;
    fn at_imagine_archive_record_name(
        names: *mut c_void,
        index: usize,
        size: *mut usize,
    ) -> *const u8;
    fn at_imagine_archive_records_free(names: *mut c_void);
    fn at_imagine_archive_offset(reader: *mut c_void, key: *const u8, size: usize) -> u64;
    fn at_imagine_archive_storage(
        reader: *mut c_void,
        key: *const u8,
        size: usize,
        numel: usize,
        dtype: i32,
    ) -> *mut torch_sys::C_tensor;
}
pub fn storage_resizable(value: &Tensor) -> Result<bool, Error> {
    let value = unsafe { at_imagine_storage_resizable(value.as_ptr()) };
    check_error()?;
    Ok(value != 0)
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
struct Record(NonNull<c_void>);
impl Drop for Record {
    fn drop(&mut self) {
        unsafe {
            at_imagine_archive_record_free(self.0.as_ptr());
        }
    }
}
struct Names(NonNull<c_void>);
impl Drop for Names {
    fn drop(&mut self) {
        unsafe {
            at_imagine_archive_records_free(self.0.as_ptr());
        }
    }
}
/// An independently owned reader. Returned storages outlive this file handle.
pub struct Reader(NonNull<c_void>);
impl Drop for Reader {
    fn drop(&mut self) {
        unsafe {
            at_imagine_archive_free(self.0.as_ptr());
        }
    }
}
impl Reader {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let path = path.as_os_str().as_bytes();
        let value = unsafe { at_imagine_archive_open(path.as_ptr(), path.len()) };
        check_error()?;
        Ok(Self(NonNull::new(value).ok_or_else(|| {
            Error::Native("Missing native archive reader".into())
        })?))
    }
    pub fn has_record(&self, key: &str) -> Result<bool, Error> {
        let found = unsafe { at_imagine_archive_has(self.0.as_ptr(), key.as_ptr(), key.len()) };
        check_error()?;
        Ok(found != 0)
    }
    pub fn record(&self, key: &str) -> Result<Vec<u8>, Error> {
        let value = unsafe { at_imagine_archive_record(self.0.as_ptr(), key.as_ptr(), key.len()) };
        check_error()?;
        let record = Record(
            NonNull::new(value)
                .ok_or_else(|| Error::Native("Missing native archive record".into()))?,
        );
        let mut size = 0;
        let data = unsafe { at_imagine_archive_data(record.0.as_ptr(), &mut size) };
        Ok(if size == 0 {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(data, size) }.to_vec()
        })
    }
    pub fn records(&self) -> Result<Vec<String>, Error> {
        let mut size = 0;
        let value = unsafe { at_imagine_archive_records(self.0.as_ptr(), &mut size) };
        check_error()?;
        let names = Names(
            NonNull::new(value)
                .ok_or_else(|| Error::Native("Missing native archive names".into()))?,
        );
        (0..size)
            .map(|index| {
                let mut size = 0;
                let data =
                    unsafe { at_imagine_archive_record_name(names.0.as_ptr(), index, &mut size) };
                let bytes = if size == 0 {
                    &[][..]
                } else {
                    unsafe { std::slice::from_raw_parts(data, size) }
                };
                crate::patching::options::SourceText::get(&python_unicode_codec::utf8(
                    bytes, false,
                )?)
                .unwrap()
                .utf8()
                .ok_or_else(|| Error::Native("Invalid strict UTF-8 result".into()))
            })
            .collect()
    }
    pub fn record_offset(&self, key: &str) -> Result<u64, Error> {
        let value = unsafe { at_imagine_archive_offset(self.0.as_ptr(), key.as_ptr(), key.len()) };
        check_error()?;
        Ok(value)
    }
    /// Uses PyTorchFileReader's exact size check and nonresizable StorageImpl.
    pub fn storage(&self, key: &str, numel: usize, dtype: i32) -> Result<Tensor, Error> {
        let value = unsafe {
            at_imagine_archive_storage(self.0.as_ptr(), key.as_ptr(), key.len(), numel, dtype)
        };
        check_error()?;
        if value.is_null() {
            return Err(Error::Native("Missing native archive storage".into()));
        }
        Ok(unsafe { Tensor::from_ptr(value) })
    }
}
