#[path = "../../../shared/task_fixtures.rs"]
mod task_fixtures;
pub fn proof(root: &std::path::Path) -> serde_json::Value {
    use st_proxy::media_destination as d;
    assert!(task_fixtures::is_owned(root));
    let config = d::Config {
        root: root.into(),
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
        fsid: d::fsid(&d::directory(root).unwrap()).unwrap(),
    };
    serde_json::to_value(d::inspect(&config, &d::System).unwrap().0).unwrap()
}
