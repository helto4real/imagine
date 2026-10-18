//! Frozen `comfy.ldm.qwen_image21.model` components shared by all checkpoint variants.
//! Complete resident DiT computation; shared runtime integration, checkpoint
//! loading, encoder/VAE execution and sampling have separate acceptance gates.
mod cache;
mod host_cache;
pub use host_cache::*;
mod checkpoint;
pub use checkpoint::*;
mod detection;
mod dit;
pub use cache::*;
mod geometry;
mod layers;
mod numeric;
mod offload;
mod ownership;
pub use offload::*;
pub use ownership::*;
mod resident;
pub use resident::*;
mod sampling;
pub use sampling::*;
mod text_encoder;
mod tokenizer;
pub use text_encoder::*;
mod unicode;
mod vae;
mod vision;
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
pub use vision::*;

mod vision_encoder;
pub use vision_encoder::*;

mod image_preprocess;
pub use image_preprocess::*;

mod patching;
pub use patching::component_patch_owner;
