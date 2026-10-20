//! CPU-only short AV container boundary. No model/runtime/cache/file IO.
//! ffmpeg is invoked with fixed arguments; private bytes travel only in pipes.
use crate::h3_protocol as h3;
use std::{
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
pub const MIME: &str = "video/x-matroska";
pub const MAX_CONTAINER: usize = h3::geometry::MAX_RAW_BYTES as usize + 64 * 1024 * 1024;
pub struct Av {
    pub shape: h3::Shape,
    pub frames: Vec<Vec<u8>>,
    /// Interleaved stereo f32 LE, no normalization or resampling.
    pub pcm: Vec<u8>,
}
impl Av {
    pub fn valid(&self) -> bool {
        self.shape.valid()
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
fn collect(
    mut child: std::process::Child,
    writes: Vec<thread::JoinHandle<std::io::Result<()>>>,
    max: usize,
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
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break Err("media_process_failed");
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
/// Lossless FFV1/RGB + PCM-f32 Matroska. All transient data stays in bounded pipes.
pub fn encode(av: Av) -> Result<Vec<u8>, &'static str> {
    if !av.valid() {
        return Err("invalid_av_result");
    }
    let (mut audio, read_audio) = UnixStream::pair().map_err(|_| "media_process_failed")?;
    let audio_fd = read_audio.as_raw_fd();
    let mut c = command();
    c.args([
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        &format!("{}x{}", av.shape.width, av.shape.height),
        "-framerate",
        "24",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-i",
        "pipe:3",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-c:a",
        "pcm_f32le",
        "-threads",
        "1",
        "-f",
        "matroska",
        "pipe:1",
    ]);
    // Only an owned pipe is inherited at fd 3. No shell, private paths or argv.
    unsafe {
        c.pre_exec(move || {
            if libc::dup2(audio_fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = c.spawn().map_err(|_| "media_processor_unavailable")?;
    drop(read_audio);
    let mut input = child.stdin.take().ok_or("media_process_failed")?;
    let video = thread::spawn(move || {
        for frame in av.frames {
            input.write_all(&frame)?;
        }
        Ok(())
    });
    let audio = thread::spawn(move || audio.write_all(&av.pcm));
    collect(child, vec![video, audio], MAX_CONTAINER)
}
fn decode_stream(bytes: &[u8], args: &[&str], max: usize) -> Result<Vec<u8>, &'static str> {
    if bytes.len() > MAX_CONTAINER || !bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Err("invalid_av_container");
    }
    let mut c = command();
    c.args([
        "-protocol_whitelist",
        "pipe",
        "-f",
        "matroska",
        "-i",
        "pipe:0",
    ])
    .args(args)
    .arg("pipe:1");
    let mut child = c.spawn().map_err(|_| "media_processor_unavailable")?;
    let mut input = child.stdin.take().ok_or("media_process_failed")?;
    let bytes = bytes.to_vec();
    collect(
        child,
        vec![thread::spawn(move || input.write_all(&bytes))],
        max,
    )
}
fn verify_streams(bytes: &[u8]) -> Result<h3::Shape, &'static str> {
    if bytes.len() > MAX_CONTAINER || !bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Err("invalid_av_container");
    }
    let mut command = Command::new("/usr/bin/ffprobe");
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-f",
            "matroska",
            "-i",
            "pipe:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,nb_read_frames",
            "-of",
            "json",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().map_err(|_| "media_processor_unavailable")?;
    let mut input = child.stdin.take().ok_or("media_process_failed")?;
    let source = bytes.to_vec();
    let metadata = collect(
        child,
        vec![thread::spawn(move || match input.write_all(&source) {
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
            result => result,
        })],
        16 * 1024,
    )?;
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
        || video["codec_name"] != "ffv1"
        || video["r_frame_rate"] != "24/1"
        || audio["codec_name"] != "pcm_f32le"
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
        &["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "f32le"],
        shape.audio_samples() as usize * 8,
    )?;
    if !pixels.len().is_multiple_of(shape.frame_bytes()) {
        return Err("invalid_av_result");
    }
    let av = Av {
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
