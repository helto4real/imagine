//! CPU-only own-fixture decoder boundary; no services, models or GPU.
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
fn own(path: &Path) -> PathBuf {
    assert!(path.starts_with("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e"));
    path.into()
}
fn main() -> anyhow::Result<()> {
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(own(Path::new(
        &std::env::args().nth(1).unwrap(),
    )))?)?;
    let file = std::fs::File::open(own(Path::new(data["source"].as_str().unwrap())))?;
    let cancel = AtomicBool::new(false);
    let mut video = imagine_server::video::reference_video::decode(&file, &data["entry"], &cancel)?;
    std::fs::write(own(Path::new(data["rgb"].as_str().unwrap())), &video.rgb)?;
    if data["with_audio"] == true {
        let audio = imagine_server::video::reference_video::audio(&file, &data["entry"], &cancel)?;
        std::fs::write(own(Path::new(data["pcm"].as_str().unwrap())), &audio.pcm)?;
        video.audio = Some(audio);
    }
    println!("{}", serde_json::to_string(&video)?);
    Ok(())
}
