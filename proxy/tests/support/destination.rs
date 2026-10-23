pub fn proof(root: &std::path::Path) -> serde_json::Value {
    use st_proxy::media_destination as d;
    assert!(
        root.to_string_lossy()
            .starts_with("/run/media/veracrypt1/imagine/.codex-")
    );
    let config = d::Config {
        root: root.into(),
        mount: "/run/media/veracrypt1".into(),
        source: "/dev/mapper/veracrypt1".into(),
        filesystem: "ext3".into(),
        fsid: d::fsid(&d::directory(root).unwrap()).unwrap(),
    };
    serde_json::to_value(d::inspect(&config, &d::System).unwrap().0).unwrap()
}
