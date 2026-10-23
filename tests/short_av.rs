//! Own procedural pixels/PCM only. CPU ffmpeg, no playback/model/GPU.
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
use imagine_server::{
    h3_protocol as h3,
    short_av::{self, Av},
};
#[test]
fn mp4_roundtrip_retains_timing_and_audible_stereo_with_lossy_codecs() {
    let av = Av {
        continuation: None,
        shape: Default::default(),
        frames: (0..h3::FRAMES)
            .map(|n| vec![n as u8; h3::FRAME_BYTES])
            .collect(),
        pcm: (0..h3::AUDIO_SAMPLES * 2)
            .flat_map(|i| {
                let frequency = if i % 2 == 0 { 440. } else { 660. };
                (0.15 * (std::f32::consts::TAU * frequency * (i / 2) as f32 / 32000.).sin())
                    .to_le_bytes()
            })
            .collect(),
    };
    let expected_frames = av.frames.clone();
    let expected_pcm = av.pcm.clone();
    let bytes = short_av::encode(av).unwrap();
    let result = short_av::decode(&bytes).unwrap();
    assert_eq!(result.frames.len(), expected_frames.len());
    let rgb_error: f64 = result
        .frames
        .iter()
        .flatten()
        .zip(expected_frames.iter().flatten())
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / (h3::FRAMES as usize * h3::FRAME_BYTES) as f64;
    assert!(rgb_error < 2.5, "RGB quantization error {rgb_error}");
    assert_eq!(result.pcm.len(), expected_pcm.len());
    let audio_error = result
        .pcm
        .chunks_exact(4)
        .zip(expected_pcm.chunks_exact(4))
        .map(|(a, b)| {
            (f64::from(f32::from_le_bytes(a.try_into().unwrap()))
                - f64::from(f32::from_le_bytes(b.try_into().unwrap())))
            .powi(2)
        })
        .sum::<f64>();
    let audio_rmse = (audio_error / (expected_pcm.len() / 4) as f64).sqrt();
    assert!(audio_rmse < 0.01, "AAC stereo waveform RMSE {audio_rmse}");
    assert_eq!(&bytes[4..8], b"ftyp");
    assert!(bytes.len() < short_av::MAX_CONTAINER);
    assert!(short_av::decode(b"not media").is_err());
    assert!(short_av::decode(&bytes[..bytes.len() / 2]).is_err());
}
#[test]
fn cancelled_encode_never_starts_a_codec_or_publishes_a_container() {
    let shape = h3::Shape {
        width: 64,
        height: 64,
        frames: 5,
        fps: 24,
    };
    let av = Av {
        continuation: None,
        shape,
        frames: vec![vec![80; shape.frame_bytes()]; 5],
        pcm: vec![0; shape.audio_samples() as usize * 8],
    };
    assert_eq!(
        short_av::encode_cancellable(av, &std::sync::atomic::AtomicBool::new(true)),
        Err("media_process_cancelled")
    );
}
#[test]
fn mp4_duration_color_and_aac_padding_match_the_frame_clock_for_short_and_five_second_clips() {
    use std::{io::Write, process::Command};
    let temp = task_fixtures::env_root("TMPDIR");
    for frames in [22, 39, 124] {
        let shape = h3::Shape {
            width: 64,
            height: 64,
            frames,
            fps: 24,
        };
        let bytes = short_av::encode(Av {
            continuation: None,
            shape,
            frames: vec![vec![91; shape.frame_bytes()]; frames as usize],
            pcm: (0..shape.audio_samples() * 2)
                .flat_map(|i| {
                    (0.15 * (std::f32::consts::TAU * 440. * (i / 2) as f32 / 32000.).sin())
                        .to_le_bytes()
                })
                .collect(),
        })
        .unwrap();
        let mut file = tempfile::NamedTempFile::new_in(&temp).unwrap();
        file.write_all(&bytes).unwrap();
        let output = Command::new("/usr/bin/ffprobe").args([
            "-v", "error", "-count_frames", "-show_entries",
            "format=duration:stream=codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,duration,nb_read_frames,sample_rate,channels",
            "-of", "json",
        ]).arg(file.path()).output().unwrap();
        assert!(output.status.success());
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let duration =
            |v: &serde_json::Value| v["duration"].as_str().unwrap().parse::<f64>().unwrap();
        assert!((duration(&value["format"]) - shape.seconds()).abs() < 0.001);
        let streams = value["streams"].as_array().unwrap();
        let video = &streams[0];
        let audio = &streams[1];
        assert_eq!(video["codec_name"], "h264");
        assert_eq!(video["nb_read_frames"], frames.to_string());
        assert_eq!(video["pix_fmt"], "yuv420p");
        assert_eq!(video["color_range"], "tv");
        for key in ["color_space", "color_transfer", "color_primaries"] {
            assert_eq!(video[key], "bt709");
        }
        assert_eq!(audio["codec_name"], "aac");
        assert_eq!(audio["sample_rate"], "32000");
        assert_eq!(audio["channels"], 2);
        assert!(
            (duration(audio) - shape.seconds()).abs() < 0.001,
            "AAC ended early: {audio}"
        );
        let decoded = short_av::decode(&bytes).unwrap();
        assert_eq!(decoded.pcm.len(), shape.audio_samples() as usize * 8);
    }
}
#[test]
fn malformed_av_is_rejected_before_any_codec_process() {
    assert!(
        short_av::encode(Av {
            continuation: None,
            shape: Default::default(),
            frames: vec![],
            pcm: vec![]
        })
        .is_err()
    );
    let av = Av {
        continuation: None,
        shape: Default::default(),
        frames: vec![vec![0; h3::FRAME_BYTES]; h3::FRAMES as usize],
        pcm: vec![0; h3::AUDIO_SAMPLES as usize * 8],
    };
    assert!(av.valid());
    let mut av = av;
    av.pcm[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(!av.valid());
}
