//! Prepare an owned test instance, never the selected product or live settings.
use imagine_server::video_destination::{self as d, Config, System};
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args_os().nth(1).expect("owned instance root"));
    anyhow::ensure!(
        root.is_absolute()
            && root
                .to_string_lossy()
                .starts_with("/run/media/veracrypt1/imagine/.codex-video-night-")
    );
    for name in ["instance", "state", "cache", "output", "tmp", "evidence"] {
        std::fs::create_dir_all(root.join(name))?;
    }
    let config = Config {
        root: root.join("output"),
        mount: "/run/media/veracrypt1".into(),
        source: "/dev/mapper/veracrypt1".into(),
        filesystem: "ext3".into(),
        fsid: d::fsid(&d::directory(&root).map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?,
    };
    let (proof, fd) = d::inspect(&config, &System).map_err(anyhow::Error::msg)?;
    d::writable(&fd, 0).map_err(anyhow::Error::msg)?;
    std::fs::write(
        root.join("state/video-output.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    std::fs::write(
        root.join("evidence/instance-destination-proof.json"),
        serde_json::to_vec_pretty(&proof)?,
    )?;
    println!(
        "Owned encrypted destination verified: {}",
        config.root.display()
    );
    Ok(())
}
