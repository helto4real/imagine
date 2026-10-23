use imagine_server::{
    video::output::Destination,
    video_destination::{self as d, Config, Environment},
};
use std::{path::Path, sync::Arc};
struct Fixture {
    mount: String,
}
impl Environment for Fixture {
    fn mounts(&self) -> d::Result<String> {
        Ok(format!(
            "100 99 0:1 / {} rw - fixture fixture rw",
            self.mount
        ))
    }
    fn host(&self) -> d::Result<String> {
        Ok("own-synthetic-host".into())
    }
}
pub fn destination(mount: &Path) -> Destination {
    let root = mount.join("persistent-videos");
    std::fs::create_dir_all(&root).unwrap();
    let config = Config {
        root: root.clone(),
        mount: mount.into(),
        source: "fixture".into(),
        filesystem: "fixture".into(),
        fsid: d::fsid(&d::directory(&root).unwrap()).unwrap(),
    };
    Destination::with_environment(
        config,
        Arc::new(Fixture {
            mount: mount.display().to_string(),
        }),
    )
}
