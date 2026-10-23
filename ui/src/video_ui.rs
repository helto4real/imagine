//! H3 availability for Timeline generation and the audio preview helper.
use super::*;
use std::{
    io::Write,
    process::{Child, Command, Stdio},
};
#[derive(Default)]
pub(super) struct VideoDesk {
    pub available: bool,
    pub checked: bool,
    pub error: Option<String>,
}
pub(super) struct Audio {
    child: Child,
    writer: Option<std::thread::JoinHandle<()>>,
}
impl Audio {
    #[cfg(feature = "ui-preview")]
    pub(super) fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
    pub(super) fn start(bytes: Vec<u8>) -> Result<Self, &'static str> {
        Self::start_command(Self::command(), bytes)
    }
    fn command() -> Command {
        use std::os::unix::process::CommandExt;
        let mut command = Command::new("/usr/bin/ffplay");
        command
            .args([
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "f32le",
                "-sample_rate",
                "32000",
                "-ch_layout",
                "stereo",
                "-i",
                "pipe:0",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                    || libc::getppid() != parent
                {
                    return Err(std::io::Error::other("parent exited"));
                }
                Ok(())
            });
        }
        command
    }
    fn start_command(mut command: Command, bytes: Vec<u8>) -> Result<Self, &'static str> {
        let mut child = command
            .spawn()
            .map_err(|_| "Could not start audio preview. Check ffplay and the audio device.")?;
        let Some(mut input) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Could not start audio preview.");
        };
        let writer = std::thread::spawn(move || {
            let _ = input.write_all(&bytes);
        });
        Ok(Self {
            child,
            writer: Some(writer),
        })
    }
}
impl Drop for Audio {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}
pub(super) enum Event {
    Available(Result<bool, Failure>),
}
impl ImagineApp {
    pub(super) fn refresh_video(&mut self, ctx: &egui::Context) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.video_available());
            send(&tx, &ctx, Message::Video(Event::Available(result)));
        });
    }
    pub(super) fn video_message(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::Available(result) => {
                self.video.checked = true;
                match result {
                    Ok(v) => {
                        self.video.available = v;
                        self.video.error = None;
                    }
                    Err(e) => self.video.error = Some(e.message),
                }
            }
        }
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3_protocol as h3;
    use std::time::Instant;
    #[test]
    fn audio_helper_uses_valid_stereo_flags_and_reaps_with_dummy_devices() {
        let mut command = Audio::command();
        command
            .env("SDL_VIDEODRIVER", "dummy")
            .env("SDL_AUDIODRIVER", "dummy");
        let mut audio =
            Audio::start_command(command, vec![0; h3::AUDIO_SAMPLES as usize * 8]).unwrap();
        let pid = audio.child.id();
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            if let Some(status) = audio.child.try_wait().unwrap() {
                assert!(status.success(), "own dummy audio helper failed");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "own dummy audio helper exceeded bound"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(audio);
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
}
