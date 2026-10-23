use super::protocol::Shape;

/// ComfyUI estimate_memory for packed H3 AV, BF16 and flash SDPA: retain the
/// two-batch request but permit static placement against the one-batch minimum.
pub fn dit(shape: Shape) -> (f64, f64) {
    let elements = 24.
        * shape.video_latent_t() as f64
        * (shape.height / 16) as f64
        * (shape.width / 16) as f64
        + 32. * 2. * shape.audio_latent_t() as f64;
    let requested = 2. * elements * 2. * 0.01 * 0.114 * 1048576.;
    (requested, requested / 2.)
}
