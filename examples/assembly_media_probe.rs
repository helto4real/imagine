//! Own synthetic project oracle through the actual freeze/raw/encode path, CPU only.
use imagine_server::video::{assembly, persistence::*, service::Service};
use std::{io::Write, path::PathBuf};
fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("own root required"))?,
    );
    anyhow::ensure!(
        root.is_absolute()
            && root
                .starts_with("/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01"),
        "own moved synthetic probe root required"
    );
    let session = ProjectSession::open(&root.join("project"))?;
    let project = session.document().project.clone();
    let options = serde_json::from_slice(&std::fs::read(root.join("options.json"))?)?;
    let mut service = Service::default();
    service
        .projects
        .insert(session)
        .map_err(StorageError::from)?;
    let frozen = assembly::freeze(
        &service,
        &assembly::Request {
            schema_version: 1,
            project_id: project.id,
            expected_project_revision: project.project_revision,
            options,
        },
        &SessionStaging::select(&root)?,
    )?;
    std::fs::write(
        root.join("rust-plan.json"),
        serde_json::to_vec_pretty(&frozen.plan)?,
    )?;
    let assembly::PreparedMedia {
        spec,
        mut frames,
        pcm,
        ..
    } = assembly::prepare_media(frozen.try_clone()?, Default::default(), Default::default())
        .map_err(anyhow::Error::msg)?;
    let mut output = std::fs::File::create_new(root.join("rust-rgb.raw"))?;
    let mut floats = std::fs::File::create_new(root.join("rust-rgb-f32.raw"))?;
    for frame in &mut frames {
        let frame = frame?;
        for value in &frame {
            floats.write_all(&value.to_le_bytes())?;
        }
        output.write_all(&assembly::quantize_rgb_frame(&frame, spec.bit_depth))?;
    }
    std::fs::write(root.join("rust-pcm.raw"), pcm)?;
    let bytes = assembly::render(frozen, Default::default(), Default::default())
        .map_err(anyhow::Error::msg)?;
    std::fs::write(root.join("delivery.mp4"), bytes)?;
    Ok(())
}
