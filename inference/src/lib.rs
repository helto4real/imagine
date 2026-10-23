//! ComfyUI-compatible inference foundation. See the pinned reference contract.
//!
//! This crate is separate from Imagine's production engine selection.

pub mod attention;
pub mod autotuning;
pub mod awq;
pub mod awq_layout;
pub mod backend_contract;
mod backend_triton_adaln;
mod backend_triton_fp8;
mod backend_triton_mxfp8;
mod backend_triton_rms_rope;
mod backend_triton_rope;
mod backend_triton_rowwise;
pub mod block_scales;
pub mod boxed_aten;
pub mod checkpoint;
pub mod conditioning;
pub mod conditioning_preprocess;
pub mod context_windows;
pub mod convrot;
pub mod convrot_w4a4;
pub mod cuda_linear;
pub mod eager;
pub mod execution;
mod execution_bindings;
pub mod execution_value;
pub mod fp8_ops;
pub mod h3_product;
pub mod int4_linear;
pub mod int8_eager_ops;
pub mod int8_linear;
pub mod int8_utils;
pub mod kitchen_direct;
pub mod kitchen_sage;
pub mod kitchen_sol;
pub mod layouts;
pub mod memory;
pub mod memory_finalizers;
pub mod models;
pub mod native;
pub mod packing;
pub mod patching;
pub mod quantization;
pub mod registry;
pub mod registry_source;
pub mod rng_seed;
pub mod sampling;
pub mod source_exception;
pub mod storage;
pub mod svdquant;
pub mod tensor_host;
pub mod triton_arguments;
pub mod triton_compilation_modes;
pub mod triton_compiler;
pub mod triton_driver;
pub mod triton_environment;
pub mod triton_file_cache;
pub mod triton_frontend;
pub mod triton_hooks;
pub mod triton_specialization;

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    #[test]
    fn libtorch_cpu_tensor_views_and_dtypes() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let view = input.transpose(0, 1);
        assert_eq!(view.stride(), [1, 2]);
        assert!(!view.is_contiguous());
        assert_eq!(view.device(), Device::Cpu);
        assert_eq!(view.kind(), Kind::Float);
        assert_eq!(view.double_value(&[0, 1]), 3.0);
        assert_eq!(view.to_kind(Kind::Double).double_value(&[1, 0]), 2.0);
    }
}

mod backend_triton_nvfp4;

mod backend_triton_int8;

mod backend_triton_w4a8;

mod backend_triton_na;

mod backend_signatures;

mod backend_triton_scalar;

mod triton_scalar_sources;

mod python_value;

mod python_error_messages;
mod python_numbers;
pub mod python_unicode_codec;

mod registry_value_ops;

mod python_tensor_repr;

pub mod server;

pub mod conditioning_batch;

pub mod python_hash;
mod source_set;

pub mod source_key_error;

mod python_float_repr;

mod python_float_ops;
