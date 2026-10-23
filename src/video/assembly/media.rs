use super::cancelled;
use std::{
    fs::File,
    io::Read,
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Command, Stdio},
    sync::atomic::AtomicBool,
    thread,
    time::Duration,
};
pub fn decode(
    file: &File,
    arguments: &[&str],
    limit: u64,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, &'static str> {
    decode_with_input(file, &[], arguments, limit, cancel)
}
pub fn decode_with_input(
    file: &File,
    input_arguments: &[&str],
    arguments: &[&str],
    limit: u64,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, &'static str> {
    if cancelled(cancel) {
        return Err("media_process_cancelled");
    }
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
        "-protocol_whitelist",
        "file,pipe",
    ])
    .args(input_arguments)
    .args(["-i", "/proc/self/fd/3"])
    .args(arguments)
    .arg("pipe:1")
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    let fd = file.as_raw_fd();
    unsafe {
        c.pre_exec(move || {
            if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = c.spawn().map_err(|_| "media_processor_unavailable")?;
    let stdout = child.stdout.take().ok_or("media_process_failed")?;
    let reader = thread::spawn(move || {
        let mut bytes = vec![];
        stdout
            .take(limit + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if !cancelled(cancel) => thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                break Err(if cancelled(cancel) {
                    "media_process_cancelled"
                } else {
                    "media_process_failed"
                });
            }
        }
    };
    let _ = child.wait();
    let bytes = reader
        .join()
        .map_err(|_| "media_process_failed")?
        .map_err(|_| "media_process_failed")?;
    if bytes.len() as u64 > limit {
        return Err("assembly_decode_memory_limit");
    }
    if !status?.success() {
        return Err("media_decode_failed");
    }
    Ok(bytes)
}
pub fn audio(
    file: &File,
    filters: Option<&str>,
    cancel: &AtomicBool,
) -> Result<Vec<super::dsp::Stereo>, &'static str> {
    let mut args = vec!["-map", "0:a:0", "-vn", "-ar", "44100", "-ac", "2"];
    if let Some(filter) = filters {
        args.extend(["-af", filter]);
    }
    args.extend(["-f", "f32le"]);
    let bytes = decode(file, &args, crate::video::media::MAX_UPLOAD, cancel)?;
    let result = bytes
        .chunks_exact(8)
        .map(|s| {
            [
                f32::from_le_bytes(s[..4].try_into().unwrap()),
                f32::from_le_bytes(s[4..].try_into().unwrap()),
            ]
        })
        .collect::<Vec<_>>();
    if !bytes.len().is_multiple_of(8) || result.iter().flatten().any(|v| !v.is_finite()) {
        return Err("invalid_audio_samples");
    }
    Ok(result)
}

/// Count decoded frames when the container has no declared frame count. A
/// one-pixel output keeps this pass bounded without buffering full RGB frames.
pub fn decoded_video_frame_count(file: &File, cancel: &AtomicBool) -> Result<usize, &'static str> {
    let bytes = decode(
        file,
        &[
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
        ],
        crate::video::media::MAX_UPLOAD,
        cancel,
    )?;
    if bytes.is_empty() {
        return Err("invalid_video_frames");
    }
    Ok(bytes.len())
}
