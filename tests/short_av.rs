//! Own procedural pixels/PCM only. CPU ffmpeg, no playback/model/GPU.
use imagine_server::{
    h3_protocol as h3,
    short_av::{self, Av},
};
#[test]
fn lossless_av_roundtrip_retains_every_frame_and_stereo_sample() {
    let av = Av {
        shape: Default::default(),
        frames: (0..h3::FRAMES)
            .map(|n| vec![n as u8; h3::FRAME_BYTES])
            .collect(),
        pcm: (0..h3::AUDIO_SAMPLES * 2)
            .flat_map(|i| ((i % 17) as f32 / 32.).to_le_bytes())
            .collect(),
    };
    let expected_frames = av.frames.clone();
    let expected_pcm = av.pcm.clone();
    let bytes = short_av::encode(av).unwrap();
    let result = short_av::decode(&bytes).unwrap();
    assert_eq!(result.frames, expected_frames);
    assert_eq!(result.pcm, expected_pcm);
    assert!(bytes.len() < short_av::MAX_CONTAINER);
    assert!(short_av::decode(b"not media").is_err());
    assert!(short_av::decode(&bytes[..bytes.len() / 2]).is_err());
}
#[test]
fn malformed_av_is_rejected_before_any_codec_process() {
    assert!(
        short_av::encode(Av {
            shape: Default::default(),
            frames: vec![],
            pcm: vec![]
        })
        .is_err()
    );
    let av = Av {
        shape: Default::default(),
        frames: vec![vec![0; h3::FRAME_BYTES]; h3::FRAMES as usize],
        pcm: vec![0; h3::AUDIO_SAMPLES as usize * 8],
    };
    assert!(av.valid());
    let mut av = av;
    av.pcm[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(!av.valid());
}
