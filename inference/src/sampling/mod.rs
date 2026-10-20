//! Shared inference algorithms from the locked ComfyUI source.
//! Registry inventory is separate from executable, differentially tested support.
pub mod adams;
pub mod advanced_guidance;
pub mod ar_video;
pub mod brownian;
pub mod control_hooks;
pub mod default_call;
pub mod deis;
pub mod dpm;
pub mod dpm_wrapper;
pub mod exponential;
pub mod factory;
pub mod inpaint_state;
mod ipndm_v_formulas;
pub mod noise;
pub mod numpy_scalar;
pub mod parameterization;
pub mod plan;
pub mod preparation;
pub mod preparation_wrappers;
pub mod public_sample;
pub mod quadrature;
pub mod sampler_state;
pub mod schedules;
pub mod sde;
pub mod solvers;
pub mod unipc;

pub mod locals;
pub mod model_sampling;
pub mod sigma_ops;

pub mod ksampler;

pub mod cast_options;
pub mod guidance;
pub mod guider;
pub mod guider_prediction;
pub mod guider_sample;
pub mod guider_state;

pub mod table_schedules;
