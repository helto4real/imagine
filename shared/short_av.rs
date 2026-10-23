//! CPU-only AV boundary. Fixed codec arguments, anonymous RAM files and bounded pipes.
//! No private media is written to a temporary filesystem path.
use crate::h3_protocol as h3;
use std::{
    fs::File,
    io::{Read, Seek, Write},
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::{net::UnixStream, process::CommandExt},
    },
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};
pub const MIME: &str = "video/mp4";
pub const MAX_CONTAINER: usize = h3::geometry::MAX_RAW_BYTES as usize + 64 * 1024 * 1024;
pub struct Av {
    pub shape: h3::Shape,
    pub frames: Vec<Vec<u8>>,
    /// Interleaved stereo f32 LE, no normalization or resampling.
    pub pcm: Vec<u8>,
    pub continuation: Option<h3::continuation::Artifact>,
}
impl Av {
    pub fn valid(&self) -> bool {
        self.shape.valid()
            && self
                .continuation
                .as_ref()
                .is_none_or(|a| a.descriptor.shape == self.shape && a.valid())
            && self.frames.len() == self.shape.frames as usize
            && self
                .frames
                .iter()
                .all(|f| f.len() == self.shape.frame_bytes())
            && self.pcm.len() == self.shape.audio_samples() as usize * h3::CHANNELS as usize * 4
            && self
                .pcm
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    }
}
fn command() -> Command {
    let mut c = Command::new("/usr/bin/ffmpeg");
    c.args([
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
    ])
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}
fn memory_file() -> Result<File, &'static str> {
    let fd = unsafe { libc::memfd_create(c"imagine-av".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err("media_process_failed");
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn inherit(c: &mut Command, sources: &[RawFd]) {
    let sources = sources.to_vec();
    unsafe {
        c.pre_exec(move || {
            // Duplicate all sources before assigning 3/4: an original may already use either fd.
            let mut copies = [-1; 2];
            for (i, source) in sources.iter().enumerate() {
                copies[i] = libc::fcntl(*source, libc::F_DUPFD_CLOEXEC, 5);
                if copies[i] < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            for (i, copy) in copies[..sources.len()].iter().enumerate() {
                if libc::dup2(*copy, 3 + i as i32) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                libc::close(*copy);
            }
            Ok(())
        });
    }
}
fn collect(
    mut child: std::process::Child,
    writes: Vec<thread::JoinHandle<std::io::Result<()>>>,
    max: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<u8>, &'static str> {
    let stdout = child.stdout.take().ok_or("media_process_failed")?;
    let reader = thread::spawn(move || {
        let mut b = Vec::new();
        stdout.take(max as u64 + 1).read_to_end(&mut b)?;
        Ok::<_, std::io::Error>(b)
    });
    let deadline = Instant::now() + Duration::from_secs(180);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None)
                if Instant::now() < deadline
                    && !cancelled.is_some_and(|c| c.load(Ordering::Acquire)) =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(if cancelled.is_some_and(|c| c.load(Ordering::Acquire)) {
                    "media_process_cancelled"
                } else {
                    "media_process_failed"
                });
            }
        }
    };
    // Reap before joining writers: an early process failure closes both input pipes.
    let _ = child.wait();
    let mut wrote = true;
    for writer in writes {
        wrote &= writer.join().is_ok_and(|v| v.is_ok());
    }
    let bytes = reader
        .join()
        .map_err(|_| "media_process_failed")?
        .map_err(|_| "media_process_failed")?;
    if !status?.success() || !wrote || bytes.len() > max {
        return Err("media_process_failed");
    }
    Ok(bytes)
}
/// Helto Save Video Advanced's H.264 MP4 profile, directly from RGB24 and stereo PCM.
pub fn encode(av: Av) -> Result<Vec<u8>, &'static str> {
    encode_cancellable(av, &AtomicBool::new(false))
}
pub fn encode_cancellable(av: Av, cancelled: &AtomicBool) -> Result<Vec<u8>, &'static str> {
    if !av.valid() {
        return Err("invalid_av_result");
    }
    let spec = EncodeSpec {
        width: av.shape.width,
        height: av.shape.height,
        frames: av.shape.frames as u64,
        fps_num: 24,
        fps_den: 1,
        sample_rate: 32000,
        bit_depth: 8,
    };
    encode_stream(spec, av.frames.into_iter().map(Ok), av.pcm, cancelled)
}
/// Direct CPU delivery encoding. Raw frames/PCM are streamed, never losslessly muxed first.
#[derive(Clone, Copy)]
pub struct EncodeSpec {
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    pub fps_num: u64,
    pub fps_den: u64,
    pub sample_rate: u32,
    pub bit_depth: u8,
}
impl EncodeSpec {
    pub fn seconds(self) -> f64 {
        self.frames as f64 * self.fps_den as f64 / self.fps_num as f64
    }
    pub fn frame_bytes(self) -> usize {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(if self.bit_depth == 10 { 6 } else { 3 }))
            .unwrap_or(usize::MAX)
    }
    fn valid(self) -> bool {
        self.width > 0
            && self.height > 0
            && self.width.is_multiple_of(2)
            && self.height.is_multiple_of(2)
            && self.fps_num > 0
            && self.fps_den > 0
            && self.frames > 0
            && [8, 10].contains(&self.bit_depth)
            && self.sample_rate > 0
            && self.frame_bytes() <= 64 * 1024 * 1024
    }
}
pub fn encode_stream<I>(
    spec: EncodeSpec,
    frames: I,
    pcm: Vec<u8>,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>, &'static str>
where
    I: Iterator<Item = std::io::Result<Vec<u8>>> + Send + 'static,
{
    if !spec.valid() || !pcm.len().is_multiple_of(8) {
        return Err("invalid_delivery_input");
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("media_process_cancelled");
    }
    // Ordinary MP4 needs seeking to finish its index/duration. memfd is anonymous RAM,
    // not /tmp, a pathname, or a lossless intermediate container.
    let mut output = memory_file()?;
    let (mut audio, read_audio) = UnixStream::pair().map_err(|_| "media_process_failed")?;
    let mut c = command();
    c.args([
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        if spec.bit_depth == 10 {
            "rgb48le"
        } else {
            "rgb24"
        },
        "-color_range",
        "pc",
        "-colorspace",
        "rgb",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-video_size",
        &format!("{}x{}", spec.width, spec.height),
        "-framerate",
        &format!("{}/{}", spec.fps_num, spec.fps_den),
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        &spec.sample_rate.to_string(),
        "-ac",
        "2",
        "-i",
        "pipe:3",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        if spec.bit_depth == 10 {
            "yuv420p10le"
        } else {
            "yuv420p"
        },
        "-crf",
        "19",
        "-vf",
        "scale=out_color_matrix=bt709",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-c:a",
        "aac",
        "-af",
        &format!("apad=whole_dur={}", spec.seconds() + 1.),
        // Helto's second pass copies video and stops at its end. In a direct encode,
        // -shortest can stop AAC one packet early while H.264 is still buffered.
        "-t",
        &spec.seconds().to_string(),
        "-fs",
        &MAX_CONTAINER.to_string(),
        "-f",
        "mp4",
        "/proc/self/fd/4",
    ]);
    inherit(&mut c, &[read_audio.as_raw_fd(), output.as_raw_fd()]);
    let mut child = c.spawn().map_err(|_| "media_processor_unavailable")?;
    drop(read_audio);
    let mut input = child.stdin.take().ok_or("media_process_failed")?;
    let video = thread::spawn(move || {
        let mut count = 0u64;
        for frame in frames {
            let frame = frame?;
            if frame.len() != spec.frame_bytes() || count >= spec.frames {
                return Err(std::io::Error::other("invalid_delivery_frames"));
            }
            input.write_all(&frame)?;
            count += 1;
        }
        if count != spec.frames {
            return Err(std::io::Error::other("incomplete_delivery_frames"));
        }
        Ok(())
    });
    let audio = thread::spawn(move || audio.write_all(&pcm));
    collect(child, vec![video, audio], 0, Some(cancelled))?;
    if output.metadata().map_err(|_| "media_process_failed")?.len() > MAX_CONTAINER as u64 {
        return Err("media_process_failed");
    }
    output.rewind().map_err(|_| "media_process_failed")?;
    let mut bytes = Vec::new();
    output
        .read_to_end(&mut bytes)
        .map_err(|_| "media_process_failed")?;
    Ok(bytes)
}
fn source_file(bytes: &[u8]) -> Result<File, &'static str> {
    if bytes.len() > MAX_CONTAINER || bytes.get(4..8) != Some(b"ftyp") {
        return Err("invalid_av_container");
    }
    let mut source = memory_file()?;
    source
        .write_all(bytes)
        .map_err(|_| "media_process_failed")?;
    source.rewind().map_err(|_| "media_process_failed")?;
    Ok(source)
}
fn decode_stream(bytes: &[u8], args: &[&str], max: usize) -> Result<Vec<u8>, &'static str> {
    let source = source_file(bytes)?;
    let mut c = command();
    c.args([
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-i",
        "/proc/self/fd/3",
    ])
    .args(args)
    .arg("pipe:1");
    inherit(&mut c, &[source.as_raw_fd()]);
    let child = c.spawn().map_err(|_| "media_processor_unavailable")?;
    collect(child, vec![], max, None)
}
fn verify_streams(bytes: &[u8]) -> Result<h3::Shape, &'static str> {
    let source = source_file(bytes)?;
    let mut command = Command::new("/usr/bin/ffprobe");
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-f",
            "mov",
            "-i",
            "/proc/self/fd/3",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,nb_read_frames,pix_fmt,color_range,color_space,color_transfer,color_primaries",
            "-of",
            "json",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    inherit(&mut command, &[source.as_raw_fd()]);
    let child = command.spawn().map_err(|_| "media_processor_unavailable")?;
    let metadata = collect(child, vec![], 16 * 1024, None)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&metadata).map_err(|_| "invalid_av_container")?;
    let streams = metadata["streams"]
        .as_array()
        .ok_or("invalid_av_container")?;
    if streams.len() != 2 {
        return Err("invalid_av_container");
    }
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or("invalid_av_container")?;
    let audio = streams
        .iter()
        .find(|s| s["codec_type"] == "audio")
        .ok_or("invalid_av_container")?;
    let number = |key: &str| {
        video[key]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("invalid_av_container")
    };
    let shape = h3::Shape {
        width: number("width")?,
        height: number("height")?,
        frames: video["nb_read_frames"]
            .as_str()
            .and_then(|v| v.parse().ok())
            .ok_or("invalid_av_container")?,
        fps: 24,
    };
    if !shape.valid()
        || video["codec_name"] != "h264"
        || video["pix_fmt"] != "yuv420p"
        || video["color_range"] != "tv"
        || video["color_space"] != "bt709"
        || video["color_transfer"] != "bt709"
        || video["color_primaries"] != "bt709"
        || video["r_frame_rate"] != "24/1"
        || audio["codec_name"] != "aac"
        || audio["sample_rate"] != "32000"
        || audio["channels"] != h3::CHANNELS
    {
        return Err("invalid_av_container");
    }
    Ok(shape)
}
/// Decode a validated product geometry. Bound the output before allocation.
pub fn decode(bytes: &[u8]) -> Result<Av, &'static str> {
    let shape = verify_streams(bytes)?;
    let pixels = decode_stream(
        bytes,
        &[
            "-map", "0:v:0", "-an", "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-f",
            "rawvideo",
        ],
        shape.frames as usize * shape.frame_bytes(),
    )?;
    let pcm = decode_stream(
        bytes,
        &[
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            &format!("atrim=end_sample={}", shape.audio_samples()),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
        ],
        shape.audio_samples() as usize * 8,
    )?;
    if !pixels.len().is_multiple_of(shape.frame_bytes()) {
        return Err("invalid_av_result");
    }
    let av = Av {
        continuation: None,
        shape,
        frames: pixels
            .chunks_exact(shape.frame_bytes())
            .map(<[u8]>::to_vec)
            .collect(),
        pcm,
    };
    if !av.valid() {
        return Err("invalid_av_result");
    }
    Ok(av)
}
