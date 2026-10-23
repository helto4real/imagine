//! Descriptor-based interval-bounded CPU RGB/PCM decode. No media paths.
use super::{assembly::media, media as inspect};
use crate::h3_protocol::references::{Audio, Image, MAX_BYTES, Video};
use serde_json::Value;
use std::{
    fs::File,
    io::Read,
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};
fn seconds(entry: &Value, key: &str) -> anyhow::Result<Option<f64>> {
    if entry[key].is_null() {
        return Ok(None);
    }
    let value = entry[key]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.)
        .ok_or_else(|| anyhow::anyhow!("invalid_reference_source_range"))?;
    Ok(Some(value))
}
fn range(entry: &Value, duration: Option<f64>, rate: f64) -> anyhow::Result<(u64, Option<u64>)> {
    let start = seconds(entry, "source_in")?.unwrap_or(0.);
    let end = seconds(entry, "source_out")?.or(duration);
    anyhow::ensure!(
        end.is_none_or(|v| v > start),
        "empty_reference_source_range"
    );
    let start = duration.map_or(start, |d| start.min(d));
    let end = end.map(|end| duration.map_or(end, |d| end.min(d)));
    anyhow::ensure!(
        start * rate < u64::MAX as f64 && end.is_none_or(|v| v * rate < u64::MAX as f64),
        "invalid_reference_source_range"
    );
    let first = (start * rate).floor() as u64;
    let last = end.map(|v| (v * rate).ceil() as u64);
    anyhow::ensure!(
        last.is_none_or(|v| v > first),
        "empty_reference_source_range"
    );
    Ok((first, last))
}
pub fn audio(file: &File, entry: &Value, cancel: &AtomicBool) -> anyhow::Result<Audio> {
    audio_with_budget(file, entry, cancel, MAX_BYTES as u64)
}
pub(crate) fn audio_with_budget(
    file: &File,
    entry: &Value,
    cancel: &AtomicBool,
    budget: u64,
) -> anyhow::Result<Audio> {
    // Sample-accurate atrim after resampling preserves the previous 44.1k source semantics.
    let start = seconds(entry, "source_in")?.unwrap_or(0.);
    let end = seconds(entry, "source_out")?;
    anyhow::ensure!(
        end.is_none_or(|v| v > start),
        "empty_reference_source_range"
    );
    anyhow::ensure!(
        start * 44100. < u64::MAX as f64 && end.is_none_or(|v| v * 44100. < u64::MAX as f64),
        "invalid_reference_source_range"
    );
    let first = (start * 44100.).round_ties_even() as u64;
    let last = end.map(|v| (v * 44100.).round_ties_even() as u64);
    anyhow::ensure!(
        last.is_none_or(|last| last > first),
        "empty_reference_source_range"
    );
    let filter = format!(
        "aresample=44100,atrim=start_sample={first}{}",
        last.map(|v| format!(":end_sample={v}")).unwrap_or_default()
    ) + ",asetpts=PTS-STARTPTS";
    let duration = last.map(|last| format!("{:.12}", (last - first) as f64 / 44100.));
    let mut args = vec![
        "-map", "0:a:0", "-vn", "-ar", "44100", "-ac", "2", "-af", &filter, "-f", "f32le",
    ];
    if let Some(duration) = &duration {
        args.extend(["-t", duration.as_str()]);
    }
    let bytes = media::decode(file, &args, budget, cancel).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !bytes.is_empty()
            && bytes.len().is_multiple_of(8)
            && bytes
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "invalid_audio_samples"
    );
    Ok(Audio {
        sample_rate: 44100,
        samples: (bytes.len() / 8).try_into()?,
        pcm: bytes,
    })
}
pub fn decode(file: &File, entry: &Value, cancel: &AtomicBool) -> anyhow::Result<Video> {
    decode_with_budget(file, entry, cancel, MAX_BYTES as u64)
}
pub(crate) fn decode_with_budget(
    file: &File,
    entry: &Value,
    cancel: &AtomicBool,
    budget: u64,
) -> anyhow::Result<Video> {
    let info = inspect::detect_streamed(file).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        info.kind == super::contract::MediaKind::Video,
        "reference_asset_type_mismatch"
    );
    let source_fps = info
        .fps
        .map_or(24., |t| t.num as f64 / t.den as f64)
        .max(0.001);
    let (first, last) = range(
        entry,
        info.duration.map(|t| t.num as f64 / t.den as f64),
        source_fps,
    )?;
    let frame_bytes = u64::from(info.width)
        .checked_mul(u64::from(info.height))
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
    anyhow::ensure!(
        frame_bytes > 0 && frame_bytes <= budget,
        "reference_decode_memory_limit"
    );
    let limit = if let Some(last) = last {
        let bytes = (last - first)
            .checked_mul(frame_bytes)
            .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
        anyhow::ensure!(bytes <= budget, "reference_decode_memory_limit");
        bytes
    } else {
        budget
    };
    // trim works in decoded frame indices (also VFR), retaining floor/ceil Director selection.
    // It avoids materializing discarded frames and stops decoding after the selected interval.
    let filter = format!(
        "trim=start_frame={first}{},setpts=PTS-STARTPTS",
        last.map(|v| format!(":end_frame={v}")).unwrap_or_default()
    );
    let frame_count = last.map(|v| (v - first).to_string());
    let mut args = vec![
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        &filter,
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
    ];
    if let Some(count) = &frame_count {
        args.extend(["-frames:v", count.as_str()]);
    }
    let rgb = media::decode_with_input(file, &["-noautorotate"], &args, limit, cancel)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !rgb.is_empty() && (rgb.len() as u64).is_multiple_of(frame_bytes),
        "invalid_reference_video_frames"
    );
    Ok(Video {
        width: info.width,
        height: info.height,
        frames: (rgb.len() as u64 / frame_bytes).try_into()?,
        source_fps,
        tail_frames: entry["requested_tail_frames"]
            .as_u64()
            .map(u32::try_from)
            .transpose()?,
        audio: None,
        rgb,
    })
}

/// Already selected at 24 fps. The image is the clip endpoint fallback, before
/// any boundary-Picture sizing or frame-0-guide preprocessing.
pub(crate) struct Boundary {
    pub video: Video,
    pub last_frame: Image,
    pub effective_tail_frames: u32,
    /// Continuous Director source interval for the PAIRED AUDIO, not frame edges.
    pub source_in: f64,
    pub source_out: f64,
}

struct BoundaryRange {
    first: u64,
    end: u64,
    start_seconds: f64,
    end_seconds: f64,
}
fn boundary_float(value: &Value, fallback: f64) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.trim().parse().ok()))
        .or_else(|| value.as_bool().map(|value| f64::from(u8::from(value))))
        .unwrap_or(fallback)
}
fn boundary_range(entry: &Value, count: u64, fps: f64) -> anyhow::Result<BoundaryRange> {
    let duration = count as f64 / fps;
    let start = boundary_float(&entry["source_in"], 0.)
        .max(0.)
        .min(duration);
    let end = boundary_float(&entry["source_out"], duration)
        .max(0.)
        .min(duration);
    anyhow::ensure!(
        start.is_finite() && end.is_finite(),
        "invalid_reference_source_range"
    );
    let first = ((start * fps).floor() as u64).min(count);
    let last = ((end * fps).ceil() as u64).max(first).min(count);
    anyhow::ensure!(last > first, "empty_reference_source_range");
    Ok(BoundaryRange {
        first,
        end: last,
        start_seconds: start,
        end_seconds: end,
    })
}

/// Torch CPU linspace uses a float32 step and computes the second half from the
/// ending point. A double-precision linspace changes exact halfway selections.
fn linspace_index(source_frames: u64, output_frames: u64, index: u64) -> u64 {
    if output_frames <= 1 || source_frames <= 1 {
        return 0;
    }
    let end = (source_frames - 1) as f32;
    let step = end / (output_frames - 1) as f32;
    let value = if index < output_frames / 2 {
        step.mul_add(index as f32, 0.)
    } else {
        (-step).mul_add((output_frames - index - 1) as f32, end)
    };
    (value.round_ties_even() as u64).min(source_frames - 1)
}
fn boundary_counts(
    frames: u64,
    fps: f64,
    requested: u64,
    target_frames: u32,
) -> anyhow::Result<(u64, u64)> {
    let resampled = if frames <= 1 || (fps - 24.).abs() <= 1e-6 {
        frames
    } else {
        let count = (frames as f64 / fps * 24.).round_ties_even().max(1.);
        anyhow::ensure!(count < u64::MAX as f64, "reference_decode_memory_limit");
        count as u64
    };
    let mut selected = resampled.min(requested).min(u64::from(target_frames));
    anyhow::ensure!(selected >= 5, "continuity_tail_too_short");
    selected -= (selected - 5) % 17;
    Ok((resampled, selected))
}

fn boundary_frame_count(file: &File, cancel: &AtomicBool) -> anyhow::Result<u64> {
    // Count a decoded frame stream without buffering the full source or imposing
    // a source-length quota on a small selected tail. Only one 4 KiB block lives
    // in the reader; the decoder still sees the original complete frame sequence.
    anyhow::ensure!(!cancel.load(Ordering::Acquire), "media_process_cancelled");
    let mut command = Command::new("/usr/bin/ffmpeg");
    command
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-noautorotate",
            "-i",
            "/proc/self/fd/3",
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "scale=1:1:flags=neighbor",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let fd = file.as_raw_fd();
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|_| anyhow::anyhow!("media_processor_unavailable"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("media_process_failed"))?;
    let reader = thread::spawn(move || -> std::io::Result<u64> {
        let mut buffer = [0; 4096];
        let mut count = 0u64;
        loop {
            let read = stdout.read(&mut buffer)?;
            if read == 0 {
                return Ok(count);
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| std::io::Error::other("decoded frame count overflow"))?;
        }
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if !cancel.load(Ordering::Acquire) => thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                break None;
            }
        }
    };
    let _ = child.wait();
    let bytes = reader
        .join()
        .map_err(|_| anyhow::anyhow!("media_process_failed"))??;
    anyhow::ensure!(!cancel.load(Ordering::Acquire), "media_process_cancelled");
    anyhow::ensure!(
        status.is_some_and(|status| status.success()),
        "media_decode_failed"
    );
    anyhow::ensure!(
        bytes > 0 && bytes.is_multiple_of(3),
        "invalid_reference_video_frames"
    );
    Ok(bytes / 3)
}

/// Exact Director boundary video + optional paired audio from one frozen source.
///
/// The first pass counts decoded frames at 1x1, so source duration follows actual
/// decoded_count/fps rather than container duration. The RGB pass then decodes
/// only the contiguous source span needed by the WHOLE trimmed span's linspace.
/// No ffmpeg fps filter or frame interpolation participates in temporal selection.
/// `budget` bounds retained video + endpoint + PCM; the temporary RGB decode
/// buffer is bounded by the same existing raw-media budget. Frame counting is
/// streamed, so an otherwise usable tail has no additional source-length quota.
/// With `boundary_source_only`, only the final trimmed source frame is selected;
/// no tail/grid/audio condition applies and `effective_tail_frames` is zero.
pub(crate) fn boundary(
    file: &File,
    entry: &Value,
    target_frames: u32,
    cancel: &AtomicBool,
    budget: u64,
) -> anyhow::Result<Boundary> {
    anyhow::ensure!(!cancel.load(Ordering::Acquire), "media_process_cancelled");
    let budget = budget.min(MAX_BYTES as u64);
    let info = inspect::detect_streamed(file).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        info.kind == super::contract::MediaKind::Video,
        "reference_asset_type_mismatch"
    );
    let fps = info
        .fps
        .map_or(24., |fps| fps.num as f64 / fps.den as f64)
        .max(0.001);
    let frame_bytes = u64::from(info.width)
        .checked_mul(u64::from(info.height))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
    anyhow::ensure!(
        frame_bytes > 0 && frame_bytes <= budget,
        "reference_decode_memory_limit"
    );
    let total_frames = boundary_frame_count(file, cancel)?;
    let selected_range = boundary_range(entry, total_frames, fps)?;
    let trimmed_frames = selected_range.end - selected_range.first;
    let source_only = entry["boundary_source_only"] == true;
    let (resampled, selected) = if source_only {
        (trimmed_frames, 1)
    } else {
        let requested = entry["requested_tail_frames"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("invalid_boundary_tail_frames"))?;
        boundary_counts(trimmed_frames, fps, requested, target_frames)?
    };
    let effective_tail_frames = if source_only { 0 } else { selected as u32 };
    let output_bytes = selected
        .checked_mul(frame_bytes)
        .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
    let retained_bytes = output_bytes
        .checked_add(frame_bytes)
        .filter(|bytes| *bytes <= budget)
        .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
    let source_index = |index| {
        if source_only || (fps - 24.).abs() <= 1e-6 {
            index
        } else {
            linspace_index(trimmed_frames, resampled, index)
        }
    };
    let first = selected_range.first + source_index(resampled - selected);
    let last = selected_range.first + source_index(resampled - 1) + 1;
    let span_bytes = (last - first)
        .checked_mul(frame_bytes)
        .filter(|bytes| *bytes <= budget)
        .ok_or_else(|| anyhow::anyhow!("reference_decode_memory_limit"))?;
    let filter = format!("trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS");
    let decoded = media::decode_with_input(
        file,
        &["-noautorotate"],
        &[
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            &filter,
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
        ],
        span_bytes,
        cancel,
    )
    .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        decoded.len() as u64 == span_bytes,
        "invalid_reference_video_frames"
    );
    let mut rgb = Vec::with_capacity(usize::try_from(output_bytes)?);
    for index in resampled - selected..resampled {
        anyhow::ensure!(!cancel.load(Ordering::Acquire), "media_process_cancelled");
        let offset =
            usize::try_from((selected_range.first + source_index(index) - first) * frame_bytes)?;
        rgb.extend_from_slice(&decoded[offset..offset + frame_bytes as usize]);
    }
    drop(decoded);
    let last_frame = Image {
        width: info.width,
        height: info.height,
        rgb: rgb[rgb.len() - frame_bytes as usize..].to_vec(),
    };
    // Director keeps authored continuous trim endpoints, not floor/ceil edges.
    let source_out = selected_range.end_seconds;
    let source_in = if source_only {
        selected_range.start_seconds
    } else {
        selected_range
            .start_seconds
            .max(source_out - f64::from(effective_tail_frames) / 24.)
    };
    let audio = if !source_only && entry["include_paired_audio"] == true {
        Some(audio_with_budget(
            file,
            &serde_json::json!({
                "source_in": source_in, "source_out": source_out,
            }),
            cancel,
            budget - retained_bytes,
        )?)
    } else {
        None
    };
    let video = Video {
        width: info.width,
        height: info.height,
        frames: selected as u32,
        source_fps: 24.,
        tail_frames: None,
        audio,
        rgb,
    };
    anyhow::ensure!(
        video.bytes() as u64 + frame_bytes <= budget,
        "reference_decode_memory_limit"
    );
    Ok(Boundary {
        video,
        last_frame,
        effective_tail_frames,
        source_in,
        source_out,
    })
}

#[cfg(test)]
use crate::task_fixtures;
#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, path::PathBuf, process::Command};
    fn fixtures() -> PathBuf {
        let root = PathBuf::from(
            std::env::var_os("IMAGINE_WORKER_A_FIXTURES").expect("set own encrypted fixture root"),
        );
        assert!(task_fixtures::is_owned(&root));
        assert!(
            !std::fs::symlink_metadata(&root)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        root
    }
    #[test]
    fn boundary_torch_float32_linspace_halfway_regression() {
        // Source-executed torch.linspace(0,3,43).round(): its center is 1,
        // whereas a double-precision 1.5 rounded ties-to-even produces 2.
        assert_eq!(linspace_index(4, 43, 21), 1);
        assert_eq!(linspace_index(4, 85, 42), 1);
        let expected = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6];
        assert_eq!(
            (0..18)
                .map(|index| linspace_index(7, 18, index))
                .collect::<Vec<_>>(),
            expected
        );
    }
    #[test]
    fn boundary_counts_align_only_after_whole_trim_span_resampling() {
        assert_eq!(boundary_counts(80, 24., 72, 124).unwrap(), (80, 56));
        assert_eq!(boundary_counts(52, 10., 39, 56).unwrap(), (125, 39));
        assert_eq!(boundary_counts(52, 10., 72, 22).unwrap(), (125, 22));
        assert_eq!(boundary_counts(80, 24., u64::MAX, 124).unwrap(), (80, 73));
        assert!(boundary_counts(52, 10., 4, 124).is_err());
        assert!(boundary_counts(1, 10., 72, 124).is_err());
    }
    #[test]
    fn boundary_range_preserves_continuous_audio_endpoints() {
        let selected = boundary_range(
            &serde_json::json!({"source_in":1.01,"source_out":6.19}),
            80,
            10.,
        )
        .unwrap();
        assert_eq!((selected.first, selected.end), (10, 62));
        assert_eq!((selected.start_seconds, selected.end_seconds), (1.01, 6.19));
        let selected = boundary_range(
            &serde_json::json!({"source_in":-1.,"source_out":""}),
            80,
            10.,
        )
        .unwrap();
        assert_eq!((selected.first, selected.end), (0, 80));
        let selected = boundary_range(
            &serde_json::json!({"source_in":6.01,"source_out":1000000.}),
            80,
            10.,
        )
        .unwrap();
        assert_eq!((selected.first, selected.end), (60, 80));
        assert_eq!((selected.start_seconds, selected.end_seconds), (6.01, 8.));
    }
    #[test]
    fn boundary_video_matches_director_indices_and_paired_pcm_interval() {
        let root = fixtures();
        let path = root.join(format!("boundary-{}.mkv", uuid::Uuid::new_v4().simple()));
        assert!(
            Command::new("/usr/bin/ffmpeg")
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=10:duration=8",
                    "-f",
                    "lavfi",
                    "-i",
                    "aevalsrc=0.1*sin(2*PI*713*t)|0.1*cos(2*PI*997*t):s=44100:d=8",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-threads",
                    "1",
                    "-c:v",
                    "ffv1",
                    "-c:a",
                    "pcm_f32le",
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let file = File::open(path).unwrap();
        let cancel = AtomicBool::new(false);
        let entry = serde_json::json!({"source_in":1.01,"source_out":6.19,
            "requested_tail_frames":39,"include_paired_audio":true});
        let selected = boundary(&file, &entry, 56, &cancel, 2_000_000).unwrap();
        assert_eq!(selected.effective_tail_frames, 39);
        assert_eq!(selected.video.frames, 39);
        assert_eq!(selected.video.source_fps, 24.);
        assert_eq!(selected.video.tail_frames, None);
        assert_eq!(selected.source_in, 4.565);
        assert_eq!(selected.source_out, 6.19);
        let whole = decode(&file, &serde_json::json!({}), &cancel).unwrap();
        // CPU source oracle, torch.linspace(0,51,125).round()[-39:] +10.
        let indices = [
            45, 46, 46, 47, 47, 47, 48, 48, 49, 49, 49, 50, 50, 51, 51, 52, 52, 52, 53, 53, 54, 54,
            54, 55, 55, 56, 56, 56, 57, 57, 58, 58, 59, 59, 59, 60, 60, 61, 61,
        ];
        let frame_bytes = 64 * 48 * 3;
        let expected = indices
            .into_iter()
            .flat_map(|index| {
                whole.rgb[index * frame_bytes..(index + 1) * frame_bytes]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(selected.video.rgb, expected);
        assert_eq!(
            selected.last_frame.rgb,
            whole.rgb[61 * frame_bytes..62 * frame_bytes]
        );
        let whole_audio = audio(&file, &serde_json::json!({}), &cancel).unwrap();
        let paired = selected.video.audio.as_ref().unwrap();
        assert_eq!(paired.samples, 272979 - 201317);
        assert_eq!(paired.pcm, whole_audio.pcm[201317 * 8..272979 * 8]);
        let mut silent_entry = entry.clone();
        silent_entry["include_paired_audio"] = serde_json::json!(false);
        let silent = boundary(&file, &silent_entry, 56, &cancel, 500_000).unwrap();
        assert!(silent.video.audio.is_none());
        assert_eq!(silent.video.rgb, selected.video.rgb);
        assert!(boundary(&file, &entry, 56, &cancel, 500_000).is_err());
        assert!(boundary(&file, &entry, 56, &AtomicBool::new(true), 2_000_000).is_err());
        // EOF clamping follows decoded count / fps, not the authored overshoot.
        let overshoot = boundary(
            &file,
            &serde_json::json!({"source_in":6.01,
            "source_out":1000000.,"requested_tail_frames":39}),
            39,
            &cancel,
            500_000,
        )
        .unwrap();
        assert_eq!(overshoot.source_out, 8.);
        assert_eq!(overshoot.source_in, 6.375);
        assert_eq!(
            overshoot.last_frame.rgb,
            whole.rgb[79 * frame_bytes..80 * frame_bytes]
        );
    }
    #[test]
    fn boundary_i2v_source_only_accepts_one_frame_without_tail_or_target_quota() {
        let root = fixtures();
        let path = root.join(format!(
            "boundary-image-{}.mkv",
            uuid::Uuid::new_v4().simple()
        ));
        assert!(
            Command::new("/usr/bin/ffmpeg")
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:size=16x16:rate=1",
                    "-frames:v",
                    "1",
                    "-threads",
                    "1",
                    "-c:v",
                    "ffv1",
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let file = File::open(path).unwrap();
        let cancel = AtomicBool::new(false);
        let selected = boundary(
            &file,
            &serde_json::json!({"boundary_source_only":true}),
            0,
            &cancel,
            4096,
        )
        .unwrap();
        let ordinary = decode(&file, &serde_json::json!({}), &cancel).unwrap();
        assert_eq!(selected.effective_tail_frames, 0);
        assert_eq!(selected.video.frames, 1);
        assert!(selected.video.audio.is_none());
        assert_eq!(selected.last_frame.rgb, ordinary.rgb);
        assert!(
            boundary(
                &file,
                &serde_json::json!({"requested_tail_frames":u64::MAX}),
                124,
                &cancel,
                4096
            )
            .is_err()
        );
    }
    #[test]
    fn reference_interval_validation() {
        assert_eq!(
            range(
                &serde_json::json!({"source_in":1.01,"source_out":1.19}),
                Some(8.),
                10.
            )
            .unwrap(),
            (10, Some(12))
        );
        assert!(
            range(
                &serde_json::json!({"source_in":3.,"source_out":2.}),
                None,
                10.
            )
            .is_err()
        );
        assert!(range(&serde_json::json!({"source_in":-1}), None, 10.).is_err());
    }
    #[test]
    fn reference_video_long_source_small_interval_and_budget() {
        let root = fixtures();
        let path = root.join(format!("video-{}.mkv", uuid::Uuid::new_v4().simple()));
        assert!(
            Command::new("/usr/bin/ffmpeg")
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=10:duration=8",
                    "-threads",
                    "1",
                    "-c:v",
                    "ffv1"
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let file = File::open(path).unwrap();
        let flag = AtomicBool::new(false);
        let clip = decode_with_budget(
            &file,
            &serde_json::json!({"source_in":6.01,"source_out":6.19}),
            &flag,
            20000,
        )
        .unwrap();
        assert_eq!(clip.frames, 2);
        assert_eq!(clip.rgb.len(), 2 * 64 * 48 * 3);
        let full = decode(&file, &serde_json::json!({}), &flag).unwrap();
        assert_eq!(clip.rgb, full.rgb[60 * 64 * 48 * 3..62 * 64 * 48 * 3]);
        assert!(decode_with_budget(&file, &serde_json::json!({}), &flag, 20000).is_err());
        assert!(
            decode_with_budget(
                &file,
                &serde_json::json!({"source_in":6.01,"source_out":6.19}),
                &AtomicBool::new(true),
                20000
            )
            .is_err()
        );
    }
    #[test]
    fn reference_audio_interval_is_sample_exact() {
        let root = fixtures();
        let path = root.join(format!("audio-{}.wav", uuid::Uuid::new_v4().simple()));
        let mut f = File::create(&path).unwrap();
        let samples = 44100u32 * 8;
        let bytes = samples * 4;
        f.write_all(b"RIFF").unwrap();
        f.write_all(&(36 + bytes).to_le_bytes()).unwrap();
        f.write_all(b"WAVEfmt ").unwrap();
        for v in [16u32, 0x00020001, 44100, 176400, 0x00100004] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(b"data").unwrap();
        f.write_all(&bytes.to_le_bytes()).unwrap();
        for i in 0..samples {
            let v = ((i % 200) as i16 - 100) * 100;
            f.write_all(&v.to_le_bytes()).unwrap();
            f.write_all(&(-v).to_le_bytes()).unwrap();
        }
        drop(f);
        let f = File::open(path).unwrap();
        let clip = audio_with_budget(
            &f,
            &serde_json::json!({"source_in":6.,"source_out":6.1}),
            &AtomicBool::new(false),
            40000,
        )
        .unwrap();
        assert_eq!(clip.samples, 4410);
        assert_eq!(clip.pcm.len(), 4410 * 8);
        let full = audio(&f, &serde_json::json!({}), &AtomicBool::new(false)).unwrap();
        assert_eq!(clip.pcm, full.pcm[264600 * 8..269010 * 8]);
        let overshoot = audio_with_budget(
            &f,
            &serde_json::json!({"source_in":7.9,"source_out":1000000.}),
            &AtomicBool::new(false),
            40000,
        )
        .unwrap();
        assert_eq!(overshoot.samples, 4410);
        assert!(
            audio_with_budget(&f, &serde_json::json!({}), &AtomicBool::new(false), 40000).is_err()
        );
    }
}
