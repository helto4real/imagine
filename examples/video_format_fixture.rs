//! Procedural CPU fixture; the caller supplies its owned encrypted evidence directory.
use imagine_server::{
    h3_protocol::Shape,
    short_av::{self, Av},
};
fn main() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(std::env::args_os().nth(1).expect("evidence directory"));
    for frames in [22, 39, 124] {
        let shape = Shape {
            width: 64,
            height: 96,
            frames,
            fps: 24,
        };
        let av = Av {
            continuation: None,
            shape,
            frames: vec![vec![91; shape.frame_bytes()]; frames as usize],
            pcm: (0..shape.audio_samples() * 2)
                .flat_map(|i| {
                    (0.15 * (std::f32::consts::TAU * 440. * (i / 2) as f32 / 32000.).sin())
                        .to_le_bytes()
                })
                .collect(),
        };
        let bytes = short_av::encode(av).map_err(anyhow::Error::msg)?;
        std::fs::write(root.join(format!("own-rust-{frames}.mp4")), &bytes)?;
        println!(
            "{frames} {:?}",
            short_av::decode(&bytes).map(|v| (v.frames.len(), v.pcm.len()))
        );
    }
    Ok(())
}
