//! Descriptor-based RGB decode and Director source-frame selection; no media paths.
use super::{assembly::media, media as inspect};
use crate::h3_protocol::references::{Audio, Video};
use serde_json::Value;
use std::{fs::File, sync::atomic::AtomicBool};
pub fn audio(file: &File, entry: &Value, cancel: &AtomicBool) -> anyhow::Result<Audio> {
    let samples = media::audio(file, None, cancel).map_err(anyhow::Error::msg)?;
    let start =
        (entry["source_in"].as_f64().unwrap_or(0.).max(0.) * 44100.).round_ties_even() as usize;
    let end = entry["source_out"]
        .as_f64()
        .map(|v| (v.max(0.) * 44100.).round_ties_even() as usize)
        .unwrap_or(samples.len())
        .min(samples.len());
    let values = &samples[start.min(end)..end];
    Ok(Audio {
        sample_rate: 44100,
        samples: values.len().try_into()?,
        pcm: values
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    })
}
pub fn decode(file: &File, entry: &Value, cancel: &AtomicBool) -> anyhow::Result<Video> {
    let info = inspect::detect(file).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        info.kind == super::contract::MediaKind::Video,
        "reference_asset_type_mismatch"
    );
    let source_fps = info
        .fps
        .map_or(24., |t| t.num as f64 / t.den as f64)
        .max(0.001);
    let rgb = media::decode_with_input(
        file,
        &["-noautorotate"],
        &[
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
        ],
        inspect::MAX_UPLOAD,
        cancel,
    )
    .map_err(anyhow::Error::msg)?;
    let frame_bytes = info.width as usize * info.height as usize * 3;
    anyhow::ensure!(
        rgb.len().is_multiple_of(frame_bytes) && !rgb.is_empty(),
        "invalid_reference_video_frames"
    );
    let count = rgb.len() / frame_bytes;
    let duration = count as f64 / source_fps;
    let start = entry["source_in"]
        .as_f64()
        .unwrap_or(0.)
        .clamp(0., duration);
    let end = entry["source_out"]
        .as_f64()
        .unwrap_or(duration)
        .clamp(0., duration);
    let first = ((start * source_fps).floor() as usize).min(count);
    let last = ((end * source_fps).ceil() as usize).max(first).min(count);
    anyhow::ensure!(last > first, "empty_reference_video_range");
    Ok(Video {
        width: info.width,
        height: info.height,
        frames: (last - first).try_into()?,
        source_fps,
        tail_frames: entry["requested_tail_frames"]
            .as_u64()
            .map(u32::try_from)
            .transpose()?,
        audio: None,
        rgb: rgb[first * frame_bytes..last * frame_bytes].to_vec(),
    })
}
