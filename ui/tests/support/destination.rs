//! Synthetic fixture roots must be placed on the explicitly authorized encrypted volume.
pub fn destination(root: &std::path::Path) -> imagine_server::video::output::Destination {
    use imagine_server::video_destination as d;
    assert!(
        root.to_string_lossy()
            .starts_with("/run/media/veracrypt1/imagine/.codex-")
    );
    std::fs::create_dir_all(root).unwrap();
    let config = d::Config {
        root: root.into(),
        mount: "/run/media/veracrypt1".into(),
        source: "/dev/mapper/veracrypt1".into(),
        filesystem: "ext3".into(),
        fsid: d::fsid(&d::directory(root).unwrap()).unwrap(),
    };
    let dest = imagine_server::video::output::Destination::new(config);
    dest.proof().unwrap();
    dest
}
pub fn proof(root: &std::path::Path) -> serde_json::Value {
    serde_json::to_value(destination(root).proof().unwrap()).unwrap()
}
