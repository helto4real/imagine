//! Director boundary video/paired-audio attention window; independent of source capture.
use super::{motion::Window, protocol::Shape};
use crate::{
    models::minimax_h3::layout::{ReferenceBlock, ReferenceKind},
    native::Error,
};
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
