//! Synthetic fixtures use only the explicitly approved task root on encrypted /home.
#[path = "../../../shared/task_fixtures.rs"]
pub mod task_fixtures;
pub fn destination(root: &std::path::Path) -> imagine_server::video::output::Destination {
    use imagine_server::video_destination as d;
    assert!(
        task_fixtures::is_owned(root),
        "own synthetic task destination required"
    );
    std::fs::create_dir_all(root).unwrap();
    let config = d::Config {
        root: root.into(),
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
        fsid: d::fsid(&d::directory(root).unwrap()).unwrap(),
    };
    let dest = imagine_server::video::output::Destination::new(config);
    dest.proof().unwrap();
    dest
}
pub fn proof(root: &std::path::Path) -> serde_json::Value {
    serde_json::to_value(destination(root).proof().unwrap()).unwrap()
}
