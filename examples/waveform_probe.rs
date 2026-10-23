//! Opt-in own-fixture waveform component probe. No model or server startup.
use std::{fs::File, io::Read, path::PathBuf};
fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(
        "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01",
    );
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let cases: Vec<(PathBuf, String, usize)> = serde_json::from_str(&input)?;
    let mut results = vec![];
    for (path, mime, count) in cases {
        anyhow::ensure!(
            path.starts_with(&root) && path.canonicalize()? == path,
            "Only direct own waveform fixtures"
        );
        let file = File::open(&path)?;
        let result = imagine_server::video::waveform::decode(&file, &mime, count);
        results.push(match result {
            Ok(value) => serde_json::json!({"path":path,"count":count,"waveform":value}),
            Err(code) => serde_json::json!({"path":path,"count":count,"error":code}),
        });
    }
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}
