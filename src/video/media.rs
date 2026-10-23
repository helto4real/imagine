//! CPU inspection of an owned media descriptor; no paths, tags or prompts in metadata.
use super::contract::{MediaKind, Time};
use serde::Serialize;
use std::{
    fs::File,
    io::{BufRead, Read},
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
pub const MAX_UPLOAD: u64 = 2 * 1024 * 1024 * 1024;
pub const MIME_TYPES: [&str; 11] = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "video/mp4",
    "video/webm",
    "video/x-matroska",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/mpeg",
    "audio/ogg",
];
#[derive(Clone, Serialize)]
pub struct Info {
    pub kind: MediaKind,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub duration: Option<Time>,
    pub fps: Option<Time>,
    pub frames: Option<u64>,
    pub audio: bool,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
}
pub fn detect(file: &File) -> Result<Info, &'static str> {
    detect_with_policy(file, false)
}
/// Descriptor-only AV callers may inspect a large compressed source without buffering it.
/// Image callers retain the existing encoded-buffer bound.
pub fn detect_streamed(file: &File) -> Result<Info, &'static str> {
    detect_with_policy(file, true)
}
fn detect_with_policy(file: &File, streamed: bool) -> Result<Info, &'static str> {
    use std::os::unix::fs::FileExt;
    let mut header = [0u8; 64];
    let n = file
        .read_at(&mut header, 0)
        .map_err(|_| "media_io_failed")?;
    let h = &header[..n];
    let mime = if h.starts_with(b"\x89PNG") {
        "image/png"
    } else if h.starts_with(b"\xff\xd8") {
        "image/jpeg"
    } else if h.starts_with(b"RIFF") && h.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else if h.starts_with(b"RIFF") && h.get(8..12) == Some(b"WAVE") {
        "audio/wav"
    } else if h.starts_with(b"fLaC") {
        "audio/flac"
    } else if h.starts_with(b"OggS") {
        "audio/ogg"
    } else if h.starts_with(b"ID3")
        || (h.first() == Some(&255) && h.get(1).is_some_and(|v| v & 0xe0 == 0xe0))
    {
        "audio/mpeg"
    } else if h.get(4..8) == Some(b"ftyp") {
        "video/mp4"
    } else if h.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        "video/x-matroska"
    } else {
        return Err("unsupported_media_type");
    };
    inspect_with_policy(file, mime, streamed)
}
pub fn inspect(file: &File, mime: &str) -> Result<Info, &'static str> {
    inspect_with_policy(file, mime, false)
}
pub fn inspect_streamed(file: &File, mime: &str) -> Result<Info, &'static str> {
    inspect_with_policy(file, mime, true)
}
fn inspect_with_policy(file: &File, mime: &str, streamed: bool) -> Result<Info, &'static str> {
    let length = file.metadata().map_err(|_| "media_io_failed")?.len();
    if length == 0 || ((!streamed || mime.starts_with("image/")) && length > MAX_UPLOAD) {
        return Err("invalid_media_size");
    }
    if mime.starts_with("image/") {
        let source = File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .map_err(|_| "media_io_failed")?;
        let reader = image::ImageReader::new(std::io::BufReader::new(source))
            .with_guessed_format()
            .map_err(|_| "invalid_media")?;
        let actual = match reader.format() {
            Some(image::ImageFormat::Png) => "image/png",
            Some(image::ImageFormat::Jpeg) => "image/jpeg",
            Some(image::ImageFormat::WebP) => "image/webp",
            _ => return Err("unsupported_media_type"),
        };
        if actual != mime {
            return Err("media_type_mismatch");
        }
        let (width, height) = reader.into_dimensions().map_err(|_| "invalid_media")?;
        if width == 0 || height == 0 || u64::from(width) * u64::from(height) * 4 > MAX_UPLOAD {
            return Err("invalid_media_dimensions");
        }
        return Ok(Info {
            kind: MediaKind::Image,
            mime_type: actual.into(),
            width,
            height,
            duration: None,
            fps: None,
            frames: None,
            audio: false,
            sample_rate: None,
            channels: None,
        });
    }
    let format = match mime {
        "video/mp4" => "mov",
        "video/webm" | "video/x-matroska" => "matroska",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        _ => return Err("unsupported_media_type"),
    };
    let value = probe(file, format)?;
    let streams = value["streams"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 16)
        .ok_or("invalid_media")?;
    let video = streams.iter().find(|s| s["codec_type"] == "video");
    let audio = streams.iter().find(|s| s["codec_type"] == "audio");
    if (mime.starts_with("video/") && video.is_none())
        || (mime.starts_with("audio/") && audio.is_none())
    {
        return Err("media_type_mismatch");
    }
    let width = video.and_then(|v| v["width"].as_u64()).unwrap_or(0);
    let height = video.and_then(|v| v["height"].as_u64()).unwrap_or(0);
    if width > u32::MAX as u64
        || height > u32::MAX as u64
        || (video.is_some()
            && (width == 0
                || height == 0
                || width.saturating_mul(height).saturating_mul(4) > MAX_UPLOAD))
    {
        return Err("invalid_media_dimensions");
    }
    let fps = video
        .and_then(|v| v["avg_frame_rate"].as_str())
        .and_then(rational)
        .filter(|t| t.num > 0)
        .or_else(|| {
            video
                .and_then(|v| v["r_frame_rate"].as_str())
                .and_then(rational)
                .filter(|t| t.num > 0)
        });
    // MP4 stream duration accounts for edit lists and codec padding. Matroska
    // and MP3 may copy a bitrate guess into *both* stream and format duration.
    let known_duration = match format {
        "mov" | "wav" | "flac" | "ogg" => stream_duration(streams),
        _ => None,
    };
    let duration = match known_duration {
        Some(duration) => Some(duration),
        None => packet_duration(file, format)?,
    };
    let sample_rate = audio
        .and_then(|v| v["sample_rate"].as_str())
        .and_then(|v| v.parse().ok());
    let channels = audio
        .and_then(|v| v["channels"].as_u64())
        .and_then(|v| u32::try_from(v).ok());
    Ok(Info {
        kind: if video.is_some() {
            MediaKind::Video
        } else {
            MediaKind::Audio
        },
        mime_type: mime.into(),
        width: width as u32,
        height: height as u32,
        duration,
        fps,
        frames: video
            .and_then(|v| v["nb_frames"].as_str())
            .and_then(|v| v.parse().ok()),
        audio: audio.is_some(),
        sample_rate,
        channels,
    })
}
fn rational(value: &str) -> Option<Time> {
    let (n, d) = value.split_once('/')?;
    let t = Time {
        num: n.parse().ok()?,
        den: d.parse().ok()?,
    };
    t.positive().then(|| t.normalized())
}
fn decimal(value: &str) -> Option<Time> {
    let (n, fraction) = value.split_once('.').unwrap_or((value, ""));
    let den = 10u64.checked_pow(fraction.len().try_into().ok()?)?;
    let num = n
        .parse::<u64>()
        .ok()?
        .checked_mul(den)?
        .checked_add(if fraction.is_empty() {
            0
        } else {
            fraction.parse().ok()?
        })?;
    let t = Time { num, den };
    t.valid().then(|| t.normalized())
}
pub(super) fn probe_command(file: &File, format: &str, entries: &str, output: &str) -> Command {
    let mut command = Command::new("/usr/bin/ffprobe");
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-f",
            format,
            "-i",
            "/proc/self/fd/3",
            "-show_entries",
            entries,
            "-of",
            output,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let source = file.as_raw_fd();
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(source, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command
}
fn wait(child: &mut std::process::Child) -> bool {
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let _ = child.wait();
    status.is_some_and(|s| s.success())
}
fn micros(value: &str) -> Option<i128> {
    let negative = value.starts_with('-');
    let time = decimal(value.strip_prefix('-').unwrap_or(value))?;
    let value = i128::from(time.num) * 1_000_000 / i128::from(time.den);
    Some(if negative { -value } else { value })
}
fn span(start: i128, end: i128) -> Option<Time> {
    let num = u64::try_from(end - start.max(0)).ok().filter(|n| *n > 0)?;
    let time = Time {
        num,
        den: 1_000_000,
    };
    time.valid().then(|| time.normalized())
}
fn stream_duration(streams: &[serde_json::Value]) -> Option<Time> {
    let mut first = i128::MAX;
    let mut last = 0;
    for stream in streams
        .iter()
        .filter(|s| s["codec_type"] == "video" || s["codec_type"] == "audio")
    {
        let duration = micros(stream["duration"].as_str()?).filter(|v| *v > 0)?;
        let start = stream["start_time"].as_str().and_then(micros).unwrap_or(0);
        first = first.min(start);
        last = last.max(start + duration);
    }
    span(first, last)
}
fn packet_duration(file: &File, format: &str) -> Result<Option<Time>, &'static str> {
    let mut child = probe_command(
        file,
        format,
        "packet=pts_time,dts_time,duration_time",
        "csv=p=0",
    )
    .spawn()
    .map_err(|_| "media_processor_unavailable")?;
    let stdout = child.stdout.take().ok_or("media_probe_failed")?;
    let read = thread::spawn(move || -> std::io::Result<(i128, i128)> {
        let mut reader = std::io::BufReader::new(stdout);
        let mut first = i128::MAX;
        let mut end = 0;
        loop {
            let mut line = String::new();
            if reader.by_ref().take(1025).read_line(&mut line)? == 0 {
                break;
            }
            if line.len() > 1024 {
                return Err(std::io::Error::other("oversized packet record"));
            }
            let mut fields = line.trim().split(',');
            let pts = fields.next().and_then(micros);
            let dts = fields.next().and_then(micros);
            let duration = fields.next().and_then(micros);
            if let (Some(start), Some(duration)) = (pts.or(dts), duration) {
                first = first.min(start);
                end = end.max(start + duration);
            }
        }
        Ok((first, end))
    });
    let success = wait(&mut child);
    let (first, end) = read
        .join()
        .map_err(|_| "media_probe_failed")?
        .map_err(|_| "media_probe_failed")?;
    if !success {
        return Err("media_probe_failed");
    }
    Ok(span(first, end))
}
pub(super) fn probe(file: &File, format: &str) -> Result<serde_json::Value, &'static str> {
    let mut child = probe_command(
        file,
        format,
        "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,sample_rate,channels,sample_fmt,start_time,duration:format=duration",
        "json",
    )
    .spawn()
    .map_err(|_| "media_processor_unavailable")?;
    let stdout = child.stdout.take().ok_or("media_probe_failed")?;
    let read = thread::spawn(move || {
        let mut bytes = vec![];
        stdout.take(16385).read_to_end(&mut bytes).map(|_| bytes)
    });
    let success = wait(&mut child);
    let bytes = read
        .join()
        .map_err(|_| "media_probe_failed")?
        .map_err(|_| "media_probe_failed")?;
    if !success || bytes.len() > 16384 {
        return Err("invalid_media");
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid_media")
}
