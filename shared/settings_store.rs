//! Shared non-private settings and atomic namespace merge. Locks live only in runtime.
use serde_json::{Map, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, OpenOptionsExt},
    },
    path::Path,
};

pub fn read_value(path: &Path) -> io::Result<Option<Value>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(io::Error::other),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}
pub fn read_namespace(path: &Path, namespace: &str) -> io::Result<Option<Value>> {
    let Some(value) = read_value(path)? else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| io::Error::other("Settings must be a JSON object"))?;
    Ok(object.get(namespace).cloned())
}
struct Lock(File);
impl Drop for Lock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}
fn lock(runtime: &Path) -> io::Result<Lock> {
    if !runtime.is_absolute() {
        return Err(io::Error::other("Explicit runtime directory required"));
    }
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(runtime)?;
    if fs::symlink_metadata(runtime)?.file_type().is_symlink() {
        return Err(io::Error::other("Runtime directory must not be a symlink"));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(runtime.join("settings.lock"))?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Lock(file))
}
fn atomic(path: &Path, value: &Value) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("Settings parent required"))?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(parent)?;
    let temp = parent.join(format!(
        ".settings-{}.partial",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(value).map_err(io::Error::other)?)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
pub fn write_value(path: &Path, value: Value, runtime: &Path) -> io::Result<()> {
    let _lock = lock(runtime)?;
    atomic(path, &value)
}
pub fn write_namespace(
    path: &Path,
    namespace: &str,
    value: Value,
    runtime: &Path,
) -> io::Result<()> {
    let _lock = lock(runtime)?;
    let mut object = match read_value(path)? {
        Some(Value::Object(object)) => object,
        Some(_) => return Err(io::Error::other("Settings must be a JSON object")),
        None => Map::new(),
    };
    object.insert(namespace.into(), value);
    atomic(path, &Value::Object(object))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn namespace_writers_preserve_each_other_and_keep_locks_out_of_config() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("home/.config/imagine/models.json");
        let runtime = root.path().join("runtime/imagine");
        std::thread::scope(|scope| {
            for namespace in ["catalog", "selection"] {
                let path = &path;
                let runtime = &runtime;
                scope.spawn(move || {
                    for n in 0..12 {
                        write_namespace(path, namespace, json!({"own_iteration":n}), runtime)
                            .unwrap();
                    }
                });
            }
        });
        for ns in ["catalog", "selection"] {
            assert_eq!(
                read_namespace(&path, ns).unwrap(),
                Some(json!({"own_iteration":11}))
            );
        }
        assert!(runtime.join("settings.lock").is_file());
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        let before = fs::read(&path).unwrap();
        let blocked = root.path().join("blocked-runtime");
        fs::write(&blocked, b"own runtime failure fixture").unwrap();
        assert!(write_namespace(&path, "selection", Value::Null, &blocked).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        fs::write(&path, b"[]").unwrap();
        assert!(write_namespace(&path, "selection", Value::Null, &runtime).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"[]");
    }
}
