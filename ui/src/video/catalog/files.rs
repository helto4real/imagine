//! Explicitly selected catalog documents; atomic writes keep editor drafts recoverable.
use super::Loaded;
#[cfg(test)]
use crate::test_destination::task_fixtures;
use imagine_video_contract::prompt_catalog;
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::{self, OpenOptions},
    io::{Read, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};
fn read(path: &Path) -> Result<String, String> {
    let mut f = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    if !f.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("Catalog source is not a regular file.".into());
    }
    let mut text = String::new();
    f.read_to_string(&mut text).map_err(|e| e.to_string())?;
    Ok(text)
}
pub(super) fn load(folder: &Path, selected: Option<String>) -> Result<Loaded, String> {
    let mut sources = BTreeMap::new();
    for entry in fs::read_dir(folder).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("yaml") || s.eq_ignore_ascii_case("yml"))
        {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("Catalog filename is not UTF-8.")?
                .to_owned();
            sources.insert(name, read(&path)?);
        }
    }
    Ok(Loaded {
        folder: folder.into(),
        sources,
        selected,
        warning: None,
        written: None,
        private_copy: false,
    })
}

fn filename(name: &str) -> Result<(), String> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(
            path.components().next(),
            Some(std::path::Component::Normal(_))
        )
        || !path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("yaml") || s.eq_ignore_ascii_case("yml"))
    {
        return Err("Choose a YAML filename within the private catalog.".into());
    }
    Ok(())
}

/// First edits copy the already loaded YAML map, never mutate/rescan an external
/// source. Each file is atomic; the multi-file copy is published only on success.
pub(super) fn save_private(
    root: &crate::private_storage::Root,
    folder: Option<&Path>,
    sources: &BTreeMap<String, String>,
    name: &str,
    text: &str,
    expected: Option<&str>,
    delete: bool,
) -> Result<Loaded, String> {
    filename(name)?;
    let layout = root.layout();
    let parent = Path::new("library/prompt-catalog");
    let existing = folder
        .and_then(|p| p.strip_prefix(layout.root()).ok())
        .filter(|p| p.starts_with(parent) && *p != parent);
    let mut next = sources.clone();
    if let Some(relative) = existing {
        let current = root.read(&relative.join(name)).map_err(str::to_owned)?;
        if expected.is_some_and(|expected| current.as_deref() != Some(expected.as_bytes())) {
            return Err(
                "The source file changed on disk. Reload before saving; your draft is retained."
                    .into(),
            );
        }
    }
    if delete {
        next.remove(name);
    } else {
        next.insert(name.into(), text.into());
    }
    prompt_catalog::load(&next).map_err(|e| e.to_string())?;
    for name in next.keys() {
        filename(name)?;
    }
    let relative = existing
        .map(Path::to_path_buf)
        .unwrap_or_else(|| parent.join(format!("catalog-{}", uuid::Uuid::new_v4().simple())));
    let mut warning = None;
    if existing.is_none() {
        root.directory(&relative).map_err(str::to_owned)?;
        for (name, source) in &next {
            root.write(&relative.join(name), source.as_bytes())
                .map_err(str::to_owned)?;
        }
    } else if delete {
        let directory = root.directory(&relative).map_err(str::to_owned)?;
        let name =
            CString::new(name).map_err(|_| "Choose a YAML filename within the private catalog.")?;
        root.proof().map_err(str::to_owned)?;
        if unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err("private_catalog_delete_failed".into());
        }
        warning = directory.sync_all().err().map(|_| {
            "Catalog committed; directory durability is uncertain. Reload or save again.".into()
        });
    } else {
        match root.write(&relative.join(name), text.as_bytes()) {
            Ok(()) => {}
            Err("private_storage_write_unconfirmed") => {
                warning = Some(
                    "Catalog committed; directory durability is uncertain. Reload or save again."
                        .into(),
                )
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(Loaded {
        folder: layout.root().join(relative),
        sources: next,
        selected: (!delete).then(|| name.into()),
        written: Some(name.into()),
        warning,
        private_copy: true,
    })
}
pub(super) fn save(
    path: &Path,
    text: &str,
    expected: Option<&str>,
    delete: bool,
) -> Result<Loaded, String> {
    let folder = path.parent().ok_or("Choose a catalog folder.")?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("Choose a YAML filename.")?
        .to_owned();
    if !path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("yaml") || s.eq_ignore_ascii_case("yml"))
    {
        return Err("Catalog files use .yaml or .yml.".into());
    }
    let mut loaded = load(folder, None)?;
    if let Some(expected) = expected
        && loaded.sources.get(&name).map(String::as_str) != Some(expected)
    {
        return Err(
            "The source file changed on disk. Reload before saving; your draft is retained.".into(),
        );
    }
    if delete {
        loaded.sources.remove(&name);
    } else {
        loaded.sources.insert(name.clone(), text.into());
    }
    prompt_catalog::load(&loaded.sources).map_err(|e| e.to_string())?;
    if delete {
        fs::remove_file(path).map_err(|e| e.to_string())?;
    } else {
        let temporary = folder.join(format!(
            ".catalog-{}.partial",
            uuid::Uuid::new_v4().simple()
        ));
        let result = (|| {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)
                .map_err(|e| e.to_string())?;
            f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
            fs::rename(&temporary, path).map_err(|e| e.to_string())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
    }
    loaded.written = Some(name.clone());
    loaded.selected = (!delete).then_some(name);
    loaded.warning = fs::File::open(folder)
        .and_then(|f| f.sync_all())
        .err()
        .map(|_| {
            "Catalog committed; directory durability is uncertain. Reload or save again.".into()
        });
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{private_storage::Root, video_destination as d};
    use std::sync::{Arc, Mutex};
    struct Environment(Mutex<String>);
    impl d::Environment for Environment {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-private-catalog".into())
        }
    }
    fn private_fixture() -> (
        tempfile::TempDir,
        Root,
        Arc<Environment>,
        BTreeMap<String, String>,
    ) {
        let own = tempfile::tempdir_in(task_fixtures::env_root("TMPDIR")).unwrap();
        let mount = own.path().join("own-volume");
        let path = mount.join("imagine");
        fs::create_dir_all(&path).unwrap();
        let environment = Arc::new(Environment(Mutex::new(format!(
            "1 0 0:1 / {} rw - fixturefs own-catalog rw\n",
            mount.display()
        ))));
        let root = Root::with_environment(
            d::Config {
                root: path,
                mount: mount.clone(),
                source: "own-catalog".into(),
                filesystem: "fixturefs".into(),
                fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
            },
            environment.clone(),
        )
        .unwrap();
        let extra = "schema_version: '1.1'\nmode: merge\nnamespaces:\n  - key: new\n    choices:\n      - key: continuation\n        label: Own linked phrase\n        append: none\n        placeholders:\n          subject: {label: Person, initial_text: Own, hint: Own fixture}\n        segments:\n          - placeholder_ref: subject\n          - {text: ' follows '}\n          - placeholder_ref: subject\n";
        let sources = BTreeMap::from([
            ("base.yaml".into(), super::super::form::starter()),
            ("linked.yaml".into(), extra.into()),
        ]);
        (own, root, environment, sources)
    }
    #[test]
    fn private_first_save_copies_ram_parts_and_preserves_original_and_linked_grammar() {
        let (own, root, _, sources) = private_fixture();
        let external = own.path().join("own-external");
        fs::create_dir(&external).unwrap();
        for (name, text) in &sources {
            fs::write(external.join(name), text).unwrap();
        }
        fs::write(
            external.join("not-loaded.yaml"),
            "not loaded, must not be scanned",
        )
        .unwrap();
        assert!(
            prompt_catalog::load(&BTreeMap::from([(
                "linked.yaml".into(),
                sources["linked.yaml"].clone()
            )]))
            .is_err()
        );
        let changed = format!("{}\n# OWN PRIVATE EDIT\n", sources["base.yaml"]);
        let saved = save_private(
            &root,
            Some(&external),
            &sources,
            "base.yaml",
            &changed,
            Some(&sources["base.yaml"]),
            false,
        )
        .unwrap();
        assert!(saved.private_copy);
        assert!(
            saved
                .folder
                .starts_with(root.layout().library().join("prompt-catalog"))
        );
        assert_eq!(saved.sources.len(), 2);
        assert!(!saved.folder.join("not-loaded.yaml").exists());
        assert_eq!(
            read(&external.join("base.yaml")).unwrap(),
            sources["base.yaml"]
        );
        let reopened = load(&saved.folder, None).unwrap();
        assert_eq!(reopened.sources, saved.sources);
        let parsed = prompt_catalog::load(&reopened.sources).unwrap();
        let choice = parsed
            .namespaces
            .iter()
            .find(|n| n.key == "new")
            .unwrap()
            .choices
            .iter()
            .find(|c| c.key == "continuation")
            .unwrap();
        assert_eq!(choice.expand(&BTreeMap::new()).text, "Own follows Own");
        assert!(
            save_private(
                &root,
                Some(&saved.folder),
                &saved.sources,
                "../outside.yaml",
                &changed,
                None,
                false
            )
            .is_err()
        );
        assert!(
            save_private(
                &root,
                Some(&saved.folder),
                &saved.sources,
                external.join("outside.yaml").to_str().unwrap(),
                &changed,
                None,
                false
            )
            .is_err()
        );
        assert!(!external.join("outside.yaml").exists());
    }
    #[test]
    fn private_first_delete_copies_remaining_parts_and_guarded_conflict_failure_recovers() {
        let (own, root, environment, sources) = private_fixture();
        let external = own.path().join("own-external");
        fs::create_dir(&external).unwrap();
        for (name, text) in &sources {
            fs::write(external.join(name), text).unwrap();
        }
        let deleted = save_private(
            &root,
            Some(&external),
            &sources,
            "linked.yaml",
            "",
            Some(&sources["linked.yaml"]),
            true,
        )
        .unwrap();
        assert_eq!(deleted.sources.len(), 1);
        assert!(!deleted.folder.join("linked.yaml").exists());
        assert_eq!(
            read(&external.join("linked.yaml")).unwrap(),
            sources["linked.yaml"]
        );
        let relative = deleted
            .folder
            .strip_prefix(root.layout().root())
            .unwrap()
            .to_path_buf();
        let current = format!("{}\n# OWN DISK CHANGE\n", sources["base.yaml"]);
        root.write(&relative.join("base.yaml"), current.as_bytes())
            .unwrap();
        assert!(
            save_private(
                &root,
                Some(&deleted.folder),
                &deleted.sources,
                "base.yaml",
                &sources["base.yaml"],
                Some(&sources["base.yaml"]),
                false
            )
            .err()
            .unwrap()
            .contains("changed on disk")
        );
        let mounted = environment.0.lock().unwrap().clone();
        *environment.0.lock().unwrap() = mounted.replace(" rw", " ro");
        let before = deleted.sources.clone();
        assert!(
            save_private(
                &root,
                Some(&deleted.folder),
                &deleted.sources,
                "base.yaml",
                &current,
                Some(&current),
                false
            )
            .is_err()
        );
        assert!(
            save_private(
                &root,
                Some(&external),
                &sources,
                "linked.yaml",
                "",
                Some(&sources["linked.yaml"]),
                true
            )
            .is_err()
        );
        assert_eq!(deleted.sources, before);
        assert_eq!(read(&deleted.folder.join("base.yaml")).unwrap(), current);
        *environment.0.lock().unwrap() = String::new();
        assert!(
            save_private(
                &root,
                Some(&deleted.folder),
                &deleted.sources,
                "base.yaml",
                &current,
                Some(&current),
                false
            )
            .is_err()
        );
        *environment.0.lock().unwrap() = mounted;
        let saved = save_private(
            &root,
            Some(&deleted.folder),
            &deleted.sources,
            "base.yaml",
            &current,
            Some(&current),
            false,
        )
        .unwrap();
        // Deletion within an existing private copy is dirfd-relative as well.
        let with_linked = save_private(
            &root,
            Some(&saved.folder),
            &saved.sources,
            "linked.yaml",
            &sources["linked.yaml"],
            None,
            false,
        )
        .unwrap();
        let again = save_private(
            &root,
            Some(&with_linked.folder),
            &with_linked.sources,
            "linked.yaml",
            "",
            Some(&sources["linked.yaml"]),
            true,
        )
        .unwrap();
        assert!(!again.folder.join("linked.yaml").exists());
        assert_eq!(
            read(&external.join("linked.yaml")).unwrap(),
            sources["linked.yaml"]
        );
    }
    #[test]
    fn invalid_and_conflicting_writes_preserve_sources_then_reload_saved_edit() {
        let root = std::env::temp_dir();
        assert!(
            task_fixtures::is_owned(&root),
            "Use the approved relocated synthetic test directory"
        );
        let folder = root.join(format!("catalog-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir(&folder).unwrap();
        let path = folder.join("own.yaml");
        let original = super::super::form::starter();
        save(&path, &original, None, false).unwrap();
        assert!(save(&path, "not valid", Some(&original), false).is_err());
        assert_eq!(read(&path).unwrap(), original);
        assert!(save(&path, &original, Some("stale text"), false).is_err());
        assert_eq!(read(&path).unwrap(), original);
        let updated = original.replace("New phrase", "Own reviewed phrase");
        let loaded = save(&path, &updated, Some(&original), false).unwrap();
        assert_eq!(loaded.sources["own.yaml"], updated);
        assert_eq!(load(&folder, None).unwrap().sources["own.yaml"], updated);
        // Only this newly created synthetic fixture directory is removed.
        fs::remove_file(path).unwrap();
        fs::remove_dir(folder).unwrap();
    }
}
