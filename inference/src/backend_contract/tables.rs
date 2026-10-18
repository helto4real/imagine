//! Generated from frozen Kitchen constraint builders; see tools/reference_backend_contract.py.
use super::{f, p};
use crate::registry::{FunctionConstraints, ShapeRule};
use std::collections::BTreeMap;
pub(super) fn eager() -> BTreeMap<String, FunctionConstraints> {
    BTreeMap::from([
        (
            "adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "fp16_conv3d".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "residual",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "group_norm_silu_pad3d".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "stochastic_rounding_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("rng", Some(&["torch.uint8"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "qx",
                        Some(&["torch.uint8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                    p("block_scales", Some(&["torch.float8_e4m3fn"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "scaled_mm_nvfp4".into(),
            f(
                vec![
                    p(
                        "a",
                        Some(&["torch.uint8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "b",
                        Some(&["torch.uint8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("tensor_scale_a", Some(&["torch.float32"]), None, vec![]),
                    p("tensor_scale_b", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "block_scale_a",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p(
                        "block_scale_b",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_svdquant_w4a4".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "smooth",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "lora_down",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "scaled_mm_svdquant_w4a4".into(),
            f(
                vec![
                    p(
                        "act",
                        Some(&["torch.int8", "torch.uint8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("wgt", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "ascales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "lora_act_in",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "lora_up",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_convrot_w4a4_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_convrot_w4a4_weight".into(),
            f(
                vec![
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p(
                        "output_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "convrot_w4a4_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qweight",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p("linear_dtype", Some(&["str"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_w4a8_int8_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("symmetric", Some(&["bool"]), None, vec![]),
                    p(
                        "scale_dtype",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p("codebook", Some(&["bool"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_w4a8_int8_weight".into(),
            f(
                vec![
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_rel",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_channel",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "codebook",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "correction",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p(
                        "output_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "w4a8_int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_rel",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_channel",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "codebook",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "correction",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "prepare_int4_weight_for_int8_linear".into(),
            f(
                vec![p(
                    "weight",
                    Some(&["torch.int8"]),
                    None,
                    vec![ShapeRule::ExactDims(2)],
                )],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "gemv_awq_w4a16".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qweight",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wzeros",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_int8_tensorwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&[
                            "float",
                            "str",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_int8_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_and_rotate_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "H",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rotate_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_convrot_weight_dtype".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("output_dtype_code", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_embedding".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "indices",
                        Some(&["torch.int32", "torch.int64"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("output_dtype_code", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_simple".into(),
            f(
                vec![
                    p("q", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_simple_dtype".into(),
            f(
                vec![
                    p("q", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("output_dtype_code", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "fp16_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "residual",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "residual_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("weight", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "weight_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("convrot", Some(&["bool"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_mxfp8".into(),
            f(
                vec![p(
                    "x",
                    Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                    None,
                    vec![ShapeRule::ExactDims(2)],
                )],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_mxfp8".into(),
            f(
                vec![
                    p(
                        "qx",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "block_scales",
                        Some(&["torch.float8_e8m0fnu"]),
                        None,
                        vec![],
                    ),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "scaled_mm_mxfp8".into(),
            f(
                vec![
                    p(
                        "a",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "b",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "block_scale_a",
                        Some(&["torch.float8_e8m0fnu"]),
                        None,
                        vec![],
                    ),
                    p(
                        "block_scale_b",
                        Some(&["torch.float8_e8m0fnu"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "sol_attn".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "v",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &["sol_attn_common_call_rule"],
            ),
        ),
        (
            "na3d".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "v",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &["na3d_common_call_rule"],
            ),
        ),
        (
            "fp16_conv3d_out".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "residual",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "out",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "group_norm_silu_pad3d_out".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "out",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                ],
                &["*", "cpu", "cuda", "hpu", "meta", "mps", "xpu"],
                None,
                &[],
            ),
        ),
    ])
}
pub(super) fn triton() -> BTreeMap<String, FunctionConstraints> {
    BTreeMap::from([
        (
            "adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                    p("scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_mxfp8".into(),
            f(
                vec![p(
                    "x",
                    Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                    None,
                    vec![ShapeRule::ExactDims(2)],
                )],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "qx",
                        Some(&["torch.uint8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                    p("block_scales", Some(&["torch.float8_e4m3fn"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((10, 0)),
                &[],
            ),
        ),
        (
            "apply_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("weight", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "weight_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("convrot", Some(&["bool"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("input_act", Some(&["NoneType", "str"]), None, vec![]),
                ],
                &["cuda", "xpu"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "w4a8_int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_rel",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_channel",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "codebook",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "correction",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "quantize_int8_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "quantize_and_rotate_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "H",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda", "xpu"],
                None,
                &[],
            ),
        ),
        (
            "na3d".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "v",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &["triton_na3d"],
            ),
        ),
    ])
}
pub(super) fn cuda() -> BTreeMap<String, FunctionConstraints> {
    BTreeMap::from([
        (
            "sol_attn".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "v",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &["sol_attn_common_call_rule"],
            ),
        ),
        (
            "na3d".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "v",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &["cuda_na3d"],
            ),
        ),
        (
            "adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "rms_adaln".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "shift",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "fp16_conv3d".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p("bias", Some(&["NoneType", "torch.float16"]), None, vec![]),
                    p(
                        "residual",
                        Some(&["NoneType", "torch.float16"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "group_norm_silu_pad3d".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_per_tensor_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                    p("scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "stochastic_rounding_fp8".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("rng", Some(&["torch.uint8"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.float8_e4m3fn", "torch.float8_e5m2"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_mxfp8".into(),
            f(
                vec![p(
                    "x",
                    Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                    None,
                    vec![ShapeRule::ExactDims(2)],
                )],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_nvfp4".into(),
            f(
                vec![
                    p(
                        "qx",
                        Some(&["torch.uint8"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 0, factor: 16 },
                        ],
                    ),
                    p("per_tensor_scale", Some(&["torch.float32"]), None, vec![]),
                    p("block_scales", Some(&["torch.float8_e4m3fn"]), None, vec![]),
                    p(
                        "output_type",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope"],
            ),
        ),
        (
            "rms_rope1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope1"],
            ),
        ),
        (
            "rms_rope_split_half".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope"],
            ),
        ),
        (
            "rms_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope1"],
            ),
        ),
        (
            "quantize_int8_tensorwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "scale",
                        Some(&[
                            "float",
                            "str",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_int8_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_and_rotate_rowwise".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "H",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "rotate_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_w4a8_int8_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("symmetric", Some(&["bool"]), None, vec![]),
                    p(
                        "scale_dtype",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p("codebook", Some(&["bool"]), None, vec![]),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "dequantize_w4a8_int8_weight".into(),
            f(
                vec![
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_rel",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_channel",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "codebook",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "correction",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p(
                        "output_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "w4a8_int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_rel",
                        Some(&["torch.float32", "torch.float8_e4m3fn"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "s_channel",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "codebook",
                        Some(&["torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "correction",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "dequantize_int8_convrot_weight".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_convrot_weight_dtype".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("group_size", Some(&["int"]), None, vec![]),
                    p("output_dtype_code", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_simple".into(),
            f(
                vec![
                    p("q", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "dequantize_int8_simple_dtype".into(),
            f(
                vec![
                    p("q", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("output_dtype_code", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "quantize_svdquant_w4a4".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 64 },
                        ],
                    ),
                    p(
                        "smooth",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                    p(
                        "lora_down",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "scaled_mm_svdquant_w4a4".into(),
            f(
                vec![
                    p(
                        "act",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("wgt", Some(&["torch.int8"]), None, vec![]),
                    p(
                        "ascales",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                    p(
                        "lora_act_in",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "lora_up",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "quantize_convrot_w4a4_weight".into(),
            f(
                vec![
                    p(
                        "weight",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 64 },
                        ],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p("stochastic_rounding", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                Some((7, 5)),
                &[],
            ),
        ),
        (
            "dequantize_convrot_w4a4_weight".into(),
            f(
                vec![
                    p(
                        "qdata",
                        Some(&["torch.int8"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 32 },
                        ],
                    ),
                    p(
                        "scales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p(
                        "output_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((7, 5)),
                &[],
            ),
        ),
        (
            "convrot_w4a4_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::MinDims(2)],
                    ),
                    p(
                        "qweight",
                        Some(&["torch.int8"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 32 },
                        ],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                    p("quant_group_size", Some(&["int"]), None, vec![]),
                    p("linear_dtype", Some(&["str"]), None, vec![]),
                ],
                &["cuda"],
                Some((7, 5)),
                &[],
            ),
        ),
        (
            "prepare_int4_weight_for_int8_linear".into(),
            f(
                vec![p(
                    "weight",
                    Some(&["torch.int8"]),
                    None,
                    vec![
                        ShapeRule::ExactDims(2),
                        ShapeRule::DivisibleBy { dim: 1, factor: 32 },
                    ],
                )],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "gemv_awq_w4a16".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                    p(
                        "qweight",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wscales",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p(
                        "wzeros",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "fp16_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::MinDims(2)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("bias", Some(&["torch.float16"]), None, vec![]),
                    p("residual", Some(&["torch.float16"]), None, vec![]),
                    p("residual_scale", Some(&["torch.float16"]), None, vec![]),
                ],
                &["cuda"],
                Some((8, 0)),
                &[],
            ),
        ),
        (
            "apply_rope_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "xq",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "xk",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "apply_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "rms_rope_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope"],
            ),
        ),
        (
            "rms_rope1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope1"],
            ),
        ),
        (
            "rms_rope_split_half_".into(),
            f(
                vec![
                    p(
                        "q",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "k",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "q_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                    p(
                        "k_scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope"],
            ),
        ),
        (
            "rms_rope_split_half1_".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(4)],
                    ),
                    p(
                        "freqs_cis",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(6)],
                    ),
                    p(
                        "scale",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::ExactDims(1)],
                    ),
                ],
                &["cuda"],
                None,
                &["_validate_cuda_rms_rope1"],
            ),
        ),
        (
            "fp16_conv3d_out".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p("bias", Some(&["NoneType", "torch.float16"]), None, vec![]),
                    p(
                        "residual",
                        Some(&["NoneType", "torch.float16"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out",
                        Some(&["torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
        (
            "group_norm_silu_pad3d_out".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                    p(
                        "weight",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "bias",
                        Some(&[
                            "NoneType",
                            "torch.bfloat16",
                            "torch.float16",
                            "torch.float32",
                        ]),
                        None,
                        vec![],
                    ),
                    p(
                        "out",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![ShapeRule::ExactDims(5)],
                    ),
                ],
                &["cuda"],
                None,
                &[],
            ),
        ),
    ])
}
pub(super) fn cuda_lt_additions() -> BTreeMap<String, FunctionConstraints> {
    BTreeMap::from([
        (
            "int8_linear".into(),
            f(
                vec![
                    p(
                        "x",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![ShapeRule::MinDims(2)],
                    ),
                    p(
                        "weight",
                        Some(&["torch.int8"]),
                        None,
                        vec![ShapeRule::ExactDims(2)],
                    ),
                    p("weight_scale", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "bias",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16", "torch.float32"]),
                        None,
                        vec![],
                    ),
                    p("convrot", Some(&["bool"]), None, vec![]),
                    p("convrot_groupsize", Some(&["int"]), None, vec![]),
                ],
                &["cuda"],
                Some((7, 5)),
                &[],
            ),
        ),
        (
            "scaled_mm_nvfp4".into(),
            f(
                vec![
                    p(
                        "a",
                        Some(&["torch.uint8"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 16 },
                        ],
                    ),
                    p(
                        "b",
                        Some(&["torch.uint8"]),
                        None,
                        vec![
                            ShapeRule::ExactDims(2),
                            ShapeRule::DivisibleBy { dim: 1, factor: 16 },
                        ],
                    ),
                    p("tensor_scale_a", Some(&["torch.float32"]), None, vec![]),
                    p("tensor_scale_b", Some(&["torch.float32"]), None, vec![]),
                    p(
                        "block_scale_a",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p(
                        "block_scale_b",
                        Some(&["torch.float8_e4m3fn"]),
                        None,
                        vec![],
                    ),
                    p(
                        "out_dtype",
                        Some(&["torch.bfloat16", "torch.float16"]),
                        None,
                        vec![],
                    ),
                ],
                &["cuda"],
                Some((10, 0)),
                &[],
            ),
        ),
    ])
}
