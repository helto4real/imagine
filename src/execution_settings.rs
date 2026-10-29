//! The operator's execution choices: kept in private storage, applied by the
//! engine to every job it starts from then on, and recorded with its output.
pub use crate::h3_protocol::execution::{Attention, Settings};
use std::{path::PathBuf, sync::Mutex};

pub struct Store {
    settings: Mutex<Settings>,
    private: Option<(crate::private_storage::Root, PathBuf)>,
}
impl Default for Store {
    /// Defaults held in memory only, for tests and engines without storage.
    fn default() -> Self {
        Self {
            settings: Mutex::new(Settings::default()),
            private: None,
        }
    }
}
impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionSettings")
            .field("settings", &self.get())
            .finish()
    }
}
impl Store {
    /// The saved choices, or the defaults when none were saved.
    pub fn open_private(
        root: crate::private_storage::Root,
        relative: PathBuf,
    ) -> anyhow::Result<Self> {
        let settings = match root.read(&relative).map_err(anyhow::Error::msg)? {
            Some(bytes) => serde_json::from_slice(&bytes)
                .map_err(|_| anyhow::anyhow!("Invalid saved execution settings"))?,
            None => Settings::default(),
        };
        Ok(Self {
            settings: Mutex::new(settings),
            private: Some((root, relative)),
        })
    }
    pub fn get(&self) -> Settings {
        *self.settings.lock().unwrap()
    }
    /// Saves first, so a failed save leaves the previous choices in force.
    pub fn set(&self, settings: Settings) -> Result<(), &'static str> {
        let mut current = self.settings.lock().unwrap();
        if let Some((root, relative)) = &self.private {
            let bytes = serde_json::to_vec_pretty(&settings).map_err(|_| "storage_error")?;
            root.write(relative, &bytes)?;
        }
        *current = settings;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{private_storage::Root, video_destination as destination};
    use std::sync::Arc;
    struct Volume(String);
    impl destination::Environment for Volume {
        fn mounts(&self) -> destination::Result<String> {
            Ok(self.0.clone())
        }
        fn host(&self) -> destination::Result<String> {
            Ok("own-execution-settings".into())
        }
    }
    #[test]
    fn choices_default_to_auto_and_survive_a_reopen() {
        let directory =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let mount = directory.path().join("own-volume");
        let path = mount.join("imagine");
        std::fs::create_dir_all(path.join("library")).unwrap();
        let config = destination::Config {
            root: path,
            mount: mount.clone(),
            source: "own-volume".into(),
            filesystem: "fixturefs".into(),
            fsid: destination::fsid(&destination::directory(&mount).unwrap()).unwrap(),
        };
        let volume = Arc::new(Volume(format!(
            "1 0 0:1 / {} rw - fixturefs own-volume rw\n",
            mount.display()
        )));
        let root = Root::with_environment(config, volume).unwrap();
        let relative = PathBuf::from("library/runtime-execution.json");
        let store = Store::open_private(root.clone(), relative.clone()).unwrap();
        assert_eq!(store.get().attention, Attention::Auto);
        store
            .set(Settings {
                attention: Attention::PytorchSdpa,
            })
            .unwrap();
        let reopened = Store::open_private(root, relative).unwrap();
        assert_eq!(reopened.get().attention, Attention::PytorchSdpa);
    }
}
