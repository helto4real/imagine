//! Director waveform reduction, streamed twice so full PCM never becomes resident.
use serde::Serialize;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Child, Command, Stdio},
};

#[derive(Debug, Serialize)]
pub struct Waveform {
    pub duration_seconds: f64,
    pub sample_rate: u64,
    pub channels: u64,
    pub peaks: Vec<f64>,
}
struct Decoder(Child);
impl Drop for Decoder {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}
// Preserve the decoder's numeric type, rather than normalizing through float:
// PyAV abs(integer minimum) wraps before conversion to Python float.
fn output(format: &str) -> Result<(&'static str, &'static str, usize), &'static str> {
    match format.trim_end_matches('p') {
        "u8" => Ok(("pcm_u8", "u8", 1)),
        "s16" => Ok(("pcm_s16le", "s16le", 2)),
        "s32" => Ok(("pcm_s32le", "s32le", 4)),
        "s64" => Ok(("pcm_s64le", "s64le", 8)),
        "flt" => Ok(("pcm_f32le", "f32le", 4)),
        "dbl" => Ok(("pcm_f64le", "f64le", 8)),
        _ => Err("unsupported_audio_sample_format"),
    }
}
fn magnitude(format: &str, bytes: &[u8]) -> f64 {
    match format.trim_end_matches('p') {
        "u8" => f64::from(bytes[0]),
        "s16" => f64::from(i16::from_le_bytes(bytes.try_into().unwrap()).wrapping_abs()),
        "s32" => f64::from(i32::from_le_bytes(bytes.try_into().unwrap()).wrapping_abs()),
        "s64" => i64::from_le_bytes(bytes.try_into().unwrap()).wrapping_abs() as f64,
        "flt" => f64::from(f32::from_le_bytes(bytes.try_into().unwrap()).abs()),
        "dbl" => f64::from_le_bytes(bytes.try_into().unwrap()).abs(),
        _ => unreachable!("output format is resolved before decoding"),
    }
}
fn samples(
    file: &File,
    format: &str,
    channels: u64,
    start: u64,
    end: Option<u64>,
    mut visit: impl FnMut(f64),
) -> Result<u64, &'static str> {
    let (codec, raw, width) = output(format)?;
    // Each pass gets its own offset while retaining the already owned inode.
    let input =
        File::open(format!("/proc/self/fd/{}", file.as_raw_fd())).map_err(|_| "media_io_failed")?;
    let mut command = Command::new("/usr/bin/ffmpeg");
    command
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-flags2",
            "+skip_manual",
            "-i",
            "/proc/self/fd/3",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            codec,
            "-f",
            raw,
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let fd = input.as_raw_fd();
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = Decoder(command.spawn().map_err(|_| "media_processor_unavailable")?);
    let mut reader = BufReader::new(child.0.stdout.take().ok_or("waveform_decode_failed")?);
    let planar = format.ends_with('p');
    let mut channel = 0;
    let mut maximum = f64::NEG_INFINITY;
    let mut raw_index = 0u64;
    loop {
        let mut bytes = [0; 8];
        let mut length = 0;
        while length < width {
            let n = reader
                .read(&mut bytes[length..width])
                .map_err(|_| "waveform_decode_failed")?;
            if n == 0 {
                break;
            }
            length += n;
        }
        if length == 0 {
            break;
        }
        if length != width {
            return Err("invalid_audio_samples");
        }
        let value = magnitude(format, &bytes[..width]);
        if !value.is_finite() {
            return Err("invalid_audio_samples");
        }
        let index = raw_index;
        raw_index += 1;
        if index < start || end.is_some_and(|end| index >= end) {
            continue;
        }
        if planar {
            maximum = maximum.max(value);
            channel += 1;
            if channel == channels {
                visit(maximum);
                channel = 0;
                maximum = f64::NEG_INFINITY;
            }
        } else {
            visit(value)
        }
    }
    if channel != 0
        || !child
            .0
            .wait()
            .map_err(|_| "waveform_decode_failed")?
            .success()
    {
        return Err("waveform_decode_failed");
    }
    Ok(raw_index)
}
// Decoder defaults skip priming and explicitly declared discard padding. The
// CLI additionally trims to container duration; PyAV's ndarray does not. Read
// only these packet fields, then keep the decoder samples rather than that CLI trim.
fn padding(file: &File, format: &str) -> Result<(u64, u64), &'static str> {
    let input =
        File::open(format!("/proc/self/fd/{}", file.as_raw_fd())).map_err(|_| "media_io_failed")?;
    let mut command = super::media::probe_command(
        &input,
        format,
        "packet_side_data=skip_samples,discard_padding",
        "compact=p=0:nk=0",
    );
    command.args(["-select_streams", "a:0"]);
    let mut child = Decoder(command.spawn().map_err(|_| "media_processor_unavailable")?);
    let stdout = child.0.stdout.take().ok_or("waveform_probe_failed")?;
    let mut skip = 0;
    let mut discard = 0;
    for line in BufReader::new(stdout).lines() {
        let line = line.map_err(|_| "waveform_probe_failed")?;
        for field in line.split('|') {
            let field = field.rsplit(':').next().unwrap_or(field);
            if let Some(value) = field.strip_prefix("skip_samples=")
                && skip == 0
            {
                skip = value.parse().map_err(|_| "invalid_audio_padding")?;
            }
            if let Some(value) = field.strip_prefix("discard_padding=") {
                discard = value.parse().map_err(|_| "invalid_audio_padding")?;
            }
        }
    }
    if !child
        .0
        .wait()
        .map_err(|_| "waveform_probe_failed")?
        .success()
    {
        return Err("waveform_probe_failed");
    }
    Ok((skip, discard))
}
pub fn decode(file: &File, mime: &str, peaks: usize) -> Result<Waveform, &'static str> {
    let input = match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        "video/mp4" => "mov",
        "video/webm" | "video/x-matroska" => "matroska",
        _ => return Err("audio_stream_missing"),
    };
    let info = super::media::probe(file, input)?;
    let audio = info["streams"]
        .as_array()
        .and_then(|v| v.iter().find(|v| v["codec_type"] == "audio"))
        .ok_or("audio_stream_missing")?;
    let format = audio["sample_fmt"]
        .as_str()
        .ok_or("audio_sample_format_missing")?;
    let channels = audio["channels"]
        .as_u64()
        .filter(|v| *v > 0)
        .ok_or("audio_channels_missing")?;
    let rate = audio["sample_rate"]
        .as_str()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .ok_or("audio_sample_rate_missing")?;
    let peaks = peaks.clamp(16, 512);
    let (skip, mut discard) = padding(file, input)?;
    // FFmpeg9 MOV synthesizes final AAC discard padding from track duration;
    // Director's installed PyAV/libavformat62 emits only the leading skip.
    // Keep its real final decoder frame, rather than fabricate a padded signal.
    if input == "mov" && audio["codec_name"] == "aac" {
        discard = 0;
    }
    let start = skip.checked_mul(channels).ok_or("invalid_audio_padding")?;
    // Declared end padding needs only a counting pass, never a PCM tail buffer.
    let end = if discard > 0 {
        let raw = samples(file, format, channels, 0, None, |_| {})?;
        Some(
            raw.checked_sub(
                discard
                    .checked_mul(channels)
                    .ok_or("invalid_audio_padding")?,
            )
            .ok_or("invalid_audio_padding")?,
        )
    } else {
        None
    };
    let mut count = 0u64;
    let mut maximum = 1f64;
    samples(file, format, channels, start, end, |v| {
        count += 1;
        maximum = maximum.max(v);
    })?;
    let chunk = count.div_ceil(peaks as u64).max(1);
    let mut bins = vec![f64::NEG_INFINITY; peaks];
    let mut index = 0u64;
    samples(file, format, channels, start, end, |v| {
        let bin = (index / chunk) as usize;
        if bin < peaks {
            bins[bin] = bins[bin].max(v);
        }
        index += 1;
    })?;
    if index != count {
        return Err("waveform_source_changed");
    }
    let duration = info["format"]["duration"]
        .as_str()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.)
        .unwrap_or(
            count as f64
                / rate as f64
                / if format.ends_with('p') {
                    1.
                } else {
                    channels as f64
                },
        );
    Ok(Waveform {
        duration_seconds: duration,
        sample_rate: rate,
        channels,
        peaks: bins
            .into_iter()
            .map(|v| {
                if v.is_finite() {
                    (v / maximum * 10000.).round_ties_even() / 10000.
                } else {
                    0.
                }
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_conversion_keeps_pyav_integer_abs_and_float_amplitudes() {
        assert_eq!(magnitude("s16", &i16::MIN.to_le_bytes()), -32768.);
        assert_eq!(magnitude("s16p", &(-8192i16).to_le_bytes()), 8192.);
        assert_eq!(magnitude("s32", &1073741824i32.to_le_bytes()), 1073741824.);
        assert_eq!(magnitude("s64", &i64::MIN.to_le_bytes()), i64::MIN as f64);
        assert_eq!(
            magnitude("s64", &(i64::MIN + 1).to_le_bytes()),
            i64::MAX as f64
        );
        assert_eq!(magnitude("u8", &[0]), 0.);
        assert_eq!(magnitude("u8", &[128]), 128.);
        assert_eq!(
            magnitude("fltp", &(-0.02f32).to_le_bytes()),
            f64::from(0.02f32)
        );
        assert_eq!(magnitude("dbl", &(-0.25f64).to_le_bytes()), 0.25);
    }
}
