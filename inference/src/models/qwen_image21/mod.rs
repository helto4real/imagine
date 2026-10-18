//! Frozen `comfy.ldm.qwen_image21.model` components shared by all checkpoint variants.
//! Complete resident DiT computation; shared runtime integration, checkpoint
//! loading, encoder/VAE execution and sampling have separate acceptance gates.
mod cache;
mod checkpoint;
pub use checkpoint::*;
mod detection;
mod dit;
pub use cache::*;
mod geometry;
mod layers;
mod numeric;
mod offload;
pub use offload::*;
mod resident;
pub use resident::*;
mod sampling;
pub use sampling::*;
mod text_encoder;
mod tokenizer;
pub use text_encoder::*;
mod unicode;
mod vae;
pub use detection::{DetectedConfig, detect};
pub use dit::*;
pub use geometry::{
    PositionEncoding, Segment, Sequence, SequenceProjections, TextProjection, build_sequence,
};
pub use layers::{Linear, Modulation};
pub use numeric::{
    LatentNormalization, gated_residual, modulated_norm, prefix_cache_key, round_timesteps,
    split_rows, timestep_embedding, zero_centered_rms_norm,
};
pub use tokenizer::*;
pub use unicode::Unicode;
pub use vae::*;
