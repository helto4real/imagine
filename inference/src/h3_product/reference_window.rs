//! Director boundary video/paired-audio attention window; independent of source capture.
use super::{motion::Window, protocol::Shape};
use crate::{
    models::minimax_h3::layout::{ReferenceBlock, ReferenceKind},
    native::Error,
};
/// Attention rows of the packed reference blocks, in the order `prepare` walks them.
pub fn rows(blocks: &[ReferenceBlock]) -> i64 {
    blocks
        .iter()
        .map(|block| {
            let frame = (block.latent_h / 2) * (block.latent_w / 2);
            match block.kind {
                ReferenceKind::Image => frame,
                ReferenceKind::Audio => block.ref_audio_t * 2,
                ReferenceKind::Video | ReferenceKind::VideoAudio => {
                    block.ref_audio_t.max(0) * 2 + block.latent_t * frame
                }
            }
        })
        .sum()
}
/// Reference blocks are packed image/audio/video in the model's conditioning order.
/// Only the first video and its paired audio are restricted; pictures stay available.
pub fn prepare(
    shape: Shape,
    text: i64,
    keyframe_rows: i64,
    blocks: &[ReferenceBlock],
) -> Result<Window, Error> {
    let target_frame_rows = i64::from(shape.height / 32) * i64::from(shape.width / 32);
    let video_rows = shape.video_latent_t() * target_frame_rows;
    let audio_rows = shape.audio_latent_t() * 2;
    let mut cursor = text + keyframe_rows;
    let mut limited = vec![];
    let mut found = false;
    for block in blocks {
        let frame = (block.latent_h / 2) * (block.latent_w / 2);
        match block.kind {
            ReferenceKind::Image => cursor += frame,
            ReferenceKind::Audio => cursor += block.ref_audio_t * 2,
            ReferenceKind::Video | ReferenceKind::VideoAudio => {
                if block.ref_audio_t > 0 {
                    let stop = cursor + block.ref_audio_t * 2;
                    if !found {
                        limited.push((cursor, stop));
                    }
                    cursor = stop;
                }
                let stop = cursor + block.latent_t * frame;
                if !found {
                    limited.push((cursor, stop));
                    found = true;
                }
                cursor = stop;
            }
        }
    }
    if !found {
        return Err(Error::Native(
            "Boundary video reference missing from conditioning".into(),
        ));
    }
    let weights = super::protocol::continuation::step_weights(shape.video_latent_t() as u32, 0)
        .into_iter()
        .flat_map(|w| std::iter::repeat_n(w, target_frame_rows as usize))
        .collect();
    Ok(Window::from_layout(
        cursor + audio_rows + video_rows,
        video_rows,
        limited,
        weights,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    fn block(kind: ReferenceKind, t: i64, h: i64, w: i64, audio: i64) -> ReferenceBlock {
        ReferenceBlock {
            kind,
            latent_t: t,
            latent_h: h,
            latent_w: w,
            ref_audio_t: audio,
        }
    }

    /// Q/K are zero, so genuine softmax attention is uniform over the supplied
    /// keys. Return the native backend layout [batch, queries, heads * dim].
    fn cpu_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        let dimensions = q.size();
        Ok(q.f_matmul(&k.f_transpose(2, 3)?)?
            .f_softmax(-1, Kind::Double)?
            .f_matmul(v)?
            .f_transpose(1, 2)?
            .f_contiguous()?
            .f_view([dimensions[0], dimensions[2], dimensions[1] * dimensions[3]])?)
    }

    fn check_reference_visibility(paired_boundary_audio: bool) -> Result<(), Error> {
        let _guard = tch::no_grad_guard();
        // Two spatial rows per video latent cell checks broadcasting as well as
        // the temporal horizon. 56 frames include full, fading and blocked cells.
        let shape = Shape {
            width: 64,
            height: 32,
            frames: 56,
            fps: 24,
        };
        assert!(shape.valid());
        let blocks = [
            block(ReferenceKind::Image, 1, 4, 4, 0),
            block(ReferenceKind::Audio, 0, 0, 0, 3),
            block(
                if paired_boundary_audio {
                    ReferenceKind::VideoAudio
                } else {
                    ReferenceKind::Video
                },
                2,
                2,
                4,
                if paired_boundary_audio { 2 } else { 0 },
            ),
            block(ReferenceKind::VideoAudio, 3, 4, 2, 3),
            block(ReferenceKind::Audio, 0, 0, 0, 1),
        ];
        let window = prepare(shape, 2, 3, &blocks)?;
        // Explicit independent packed spans: text+frame0, image, standalone
        // audio, boundary audio/video, later paired audio/video, standalone audio.
        // Each class has a distinct value channel so removing any wrong span is
        // observable, even if that span happens to have the same number of keys.
        let mut spans = vec![(0, 5, 6usize), (5, 4, 0), (9, 6, 1)];
        let mut cursor = 15;
        if paired_boundary_audio {
            spans.push((cursor, 4, 2));
            cursor += 4;
        }
        spans.push((cursor, 4, 3));
        cursor += 4;
        spans.push((cursor, 6, 4));
        cursor += 6;
        spans.push((cursor, 6, 5));
        cursor += 6;
        spans.push((cursor, 2, 1));
        cursor += 2;
        assert_eq!(cursor, if paired_boundary_audio { 37 } else { 33 });
        let frame_rows = 2i64;
        let video_rows = shape.video_latent_t() * frame_rows;
        let video_start = cursor + shape.audio_latent_t() * 2;
        let tokens = video_start + video_rows;
        let (batches, heads, dim) = (2i64, 2i64, 7i64);
        let mut values = vec![0f64; (batches * heads * tokens * dim) as usize];
        let mut counts = [0i64; 7];
        for &(start, count, channel) in &spans {
            counts[channel] += count;
            for batch in 0..batches {
                for head in 0..heads {
                    for token in start..start + count {
                        let offset =
                            (((batch * heads + head) * tokens + token) * dim) as usize + channel;
                        values[offset] = ((batch + 1) * (head + 1)) as f64;
                    }
                }
            }
        }
        let q = Tensor::f_zeros([batches, heads, tokens, dim], (Kind::Double, Device::Cpu))?;
        let k = q.f_zeros_like()?;
        let v = Tensor::f_from_slice(&values)?.f_view([batches, heads, tokens, dim])?;
        let output = window.run(&q, &k, &v, cpu_attention)?;
        assert_eq!(output.size(), [batches, tokens, heads * dim]);
        let excluded = if paired_boundary_audio { 8 } else { 4 };
        // Centers of the native [1,4,4,4,4] frame spans are 0,2.5,6.5,...
        // Full through frame16, fade through frame32, zero thereafter. These are
        // expected constants, not computed by continuation::step_weights.
        let expected_weights = [
            1.,
            1.,
            1.,
            1.,
            1.,
            15. / 16.,
            12.5 / 16.,
            8.5 / 16.,
            4.5 / 16.,
            0.5 / 16.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        ];
        assert_eq!(expected_weights.len() as i64, shape.video_latent_t());
        for batch in 0..batches {
            for head in 0..heads {
                for query in 0..tokens {
                    // Text, keyframe, reference and target-audio queries are
                    // unchanged: only target-video queries receive this window.
                    let weight = if query < video_start {
                        1.
                    } else {
                        expected_weights[((query - video_start) / frame_rows) as usize]
                    };
                    for (channel, count) in counts.iter().enumerate().take(dim as usize) {
                        let amplitude = ((batch + 1) * (head + 1)) as f64;
                        let full = amplitude * *count as f64 / tokens as f64;
                        let limited = matches!(channel, 2 | 3);
                        let restricted = if limited {
                            0.
                        } else {
                            amplitude * *count as f64 / (tokens - excluded) as f64
                        };
                        let expected = restricted + weight * (full - restricted);
                        let actual =
                            output.double_value(&[batch, query, head * dim + channel as i64]);
                        assert!(
                            (actual - expected).abs() < 1e-12,
                            "paired={paired_boundary_audio}, batch={batch}, head={head}, query={query}, channel={channel}: {actual} != {expected}"
                        );
                        if query >= video_start + 10 * frame_rows {
                            if limited {
                                assert_eq!(actual, 0.);
                            } else {
                                assert!(actual > 0.);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn first_boundary_video_and_paired_audio_fade_while_other_references_remain()
    -> Result<(), Error> {
        check_reference_visibility(true)
    }
    #[test]
    fn unpaired_first_video_does_not_restrict_later_videos_paired_audio() -> Result<(), Error> {
        check_reference_visibility(false)
    }
    #[test]
    fn missing_boundary_video_is_an_explicit_error() {
        let shape = Shape {
            width: 64,
            height: 32,
            frames: 56,
            fps: 24,
        };
        for blocks in [
            vec![],
            vec![
                block(ReferenceKind::Image, 1, 4, 4, 0),
                block(ReferenceKind::Audio, 0, 0, 0, 3),
            ],
        ] {
            match prepare(shape, 2, 3, &blocks) {
                Err(Error::Native(message)) => assert_eq!(
                    message,
                    "Boundary video reference missing from conditioning"
                ),
                _ => panic!("Missing boundary video must be reported"),
            }
        }
    }
}
