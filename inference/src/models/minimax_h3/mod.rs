//! Locked H3 AV primitives. Trained model execution and dynamic Python-object
//! bindings are separate acceptance gates.
pub mod audio_vae;
pub mod checkpoint;
pub mod conditioning;
pub mod dit;
pub mod flow;
pub mod layout;
pub mod preprocessing;
pub mod sampling;
pub mod streams;
pub mod text_encoder;
pub mod tokenizer;
pub mod video_vae;
