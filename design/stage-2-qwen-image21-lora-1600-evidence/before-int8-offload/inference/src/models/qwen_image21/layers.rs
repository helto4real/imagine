//! Resident source operations; prepared weights keep their Kitchen layout.
use crate::checkpoint::PreparedParameters;
use crate::{execution::Runtime, execution_value::Value, native::Error, storage::ScalarField};
use tch::{Kind, Tensor};

// Selected static Comfy cast_bias_weight path: synchronous stream selection
// (NUM_STREAMS=0), Preserve empty_like, nonblocking copy, then dtype cast.
fn offload_cast(rt: &Runtime, value: &Value, x: &Tensor) -> Result<Option<Value>, Error> {
    if let Ok(t) = value.as_tensor()
        && t.device() == tch::Device::Cpu
        && matches!(x.device(), tch::Device::Cuda(_))
    {
        let mut out = rt.native.empty_like(&t, x.device())?;
        rt.native.copy_tensor(&mut out, &t, true)?;
        return Ok(Some(Value::tensor(out.f_to_kind(x.kind())?)));
    }
    Ok(None)
}

pub struct Linear {
    pub weight_patches: Vec<crate::patching::Patch>,
    pub bias_patches: Vec<crate::patching::Patch>,
    pub weight: Value,
    pub bias: Option<Value>,
    pub full_precision: bool,
    pub force_forward: bool,
    pub pre_quant_scale: Option<Value>,
    pub input_scale: Option<Value>,
}
impl Linear {
    pub fn from_parameters(
        state: &PreparedParameters,
        name: &str,
        bias: bool,
    ) -> Result<Self, Error> {
        let get = |suffix: &str| {
            state
                .get(&format!("{name}.{suffix}"))
                .cloned()
                .ok_or_else(|| Error::BindingGap {
                    backend: "comfy_ops".into(),
                    operation: format!("Missing prepared {name}.{suffix}"),
                })
        };
        let policy = state.linear_policies.get(name).cloned().unwrap_or_default();
        Ok(Self {
            weight: get("weight")?,
            weight_patches: state
                .lowvram_patches
                .get(&format!("{name}.weight"))
                .cloned()
                .unwrap_or_default(),
            bias_patches: state
                .lowvram_patches
                .get(&format!("{name}.bias"))
                .cloned()
                .unwrap_or_default(),
            bias: if bias { Some(get("bias")?) } else { None },
            full_precision: policy.full_precision,
            force_forward: policy.force_forward,
            pre_quant_scale: policy.pre_quant_scale,
            input_scale: policy.input_scale,
        })
    }
    /// # Safety
    /// Prepared parameters and inputs must satisfy their original backend contracts.
    pub unsafe fn forward(&self, rt: &Runtime, x: &Tensor) -> Result<Tensor, Error> {
        let smoothed = match &self.pre_quant_scale {
            Some(scale) => Some(x.f_mul(&scale.as_tensor()?.f_to_device_(
                x.device(),
                x.kind(),
                false,
                false,
            )?)?),
            None => None,
        };
        let x = smoothed.as_ref().unwrap_or(x);
        let input_shape = x.size();
        let layout = self.weight.as_quantized().ok().map(|q| q.layout);
        let quantize_input = layout.as_deref().is_some_and(|l| {
            !matches!(
                l,
                "TensorWiseINT8Layout" | "TensorCoreConvRotW4A4Layout" | "AsymW4A8Int8Layout"
            )
        });
        let use_quantized = layout.is_some() && !self.full_precision && !self.force_forward;
        let weight_only = use_quantized && !quantize_input;
        let mut input = Value::tensor(x.shallow_clone());
        let mut reshaped = false;
        if use_quantized && quantize_input {
            let flat = if x.dim() >= 3 {
                x.f_reshape([-1, *input_shape.last().unwrap()])?
            } else {
                x.shallow_clone()
            };
            if flat.dim() == 2 {
                let scale = self
                    .input_scale
                    .as_ref()
                    .map(|v| v.as_tensor()?.f_to_device(x.device()).map_err(Error::from))
                    .transpose()?;
                let layout = layout.as_deref().unwrap();
                let q = if matches!(
                    layout,
                    "TensorCoreFP8E4M3Layout" | "TensorCoreFP8E5M2Layout"
                ) {
                    let scale = match scale {
                        Some(scale) => scale,
                        None => Tensor::f_ones([], (Kind::Float, x.device()))?,
                    };
                    let dtype = if layout == "TensorCoreFP8E4M3Layout" {
                        Kind::Float8e4m3fn
                    } else {
                        Kind::Float8e5m2
                    };
                    let payload = unsafe {
                        rt.call_kitchen(
                            "quantize_per_tensor_fp8",
                            &vec![
                                ("x".into(), Value::tensor(flat.shallow_clone())),
                                ("scale".into(), Value::tensor(scale.shallow_clone())),
                                ("output_type".into(), Value::DType(dtype)),
                            ],
                        )?
                    }
                    .as_tensor()?;
                    let mut q =
                        crate::layouts::fp8_parts(payload, scale, flat.kind(), flat.size())?;
                    q.layout = layout.into();
                    q
                } else {
                    let kwargs = vec![(
                        "scale".into(),
                        scale.map(Value::tensor).unwrap_or(Value::None),
                    )];
                    unsafe { rt.from_float(&flat, layout, &kwargs)? }
                };
                input = Value::quantized(q);
                reshaped = x.dim() >= 3;
            }
        }
        let cast_dequantize = self.full_precision
            || (!weight_only
                && self
                    .weight
                    .as_quantized()
                    .is_ok_and(|q| q.params.original_kind != x.kind()));
        let mut w = if let Some(w) = offload_cast(rt, &self.weight, x)? {
            w
        } else {
            let mut options = vec![
                ("dtype".into(), Value::DType(x.kind())),
                ("device".into(), Value::Device(x.device())),
            ];
            if !self.weight_patches.is_empty() {
                // cast_bias_weight copies before applying mutating functions.
                options.push(("copy".into(), Value::Bool(true)));
            }
            unsafe {
                rt.op(
                    "aten::to.dtype_layout",
                    std::slice::from_ref(&self.weight),
                    &options,
                )?
            }
        };
        if cast_dequantize {
            w = unsafe { rt.dequantize_args(&w)? };
        }
        let mut bias = match &self.bias {
            Some(b) => {
                if let Some(b) = offload_cast(rt, b, x)? {
                    b
                } else {
                    let mut options = vec![
                        ("dtype".into(), Value::DType(x.kind())),
                        ("device".into(), Value::Device(x.device())),
                    ];
                    if !self.bias_patches.is_empty() {
                        options.push(("copy".into(), Value::Bool(true)));
                    }
                    unsafe { rt.op("aten::to.dtype_layout", std::slice::from_ref(b), &options)? }
                }
            }
            None => Value::None,
        };
        // Original LowVramPatch runs after cast_bias_weight, in the execution
        // dtype; it is not the resident FP16 patch-and-requantize route.
        for (value, patches) in [
            (&mut w, &self.weight_patches),
            (&mut bias, &self.bias_patches),
        ] {
            if !patches.is_empty() {
                let t = value.as_tensor()?;
                let mut diagnostics = crate::patching::Diagnostics::default();
                *value = Value::tensor(crate::patching::calculate_weight(
                    &rt.native,
                    patches,
                    t,
                    "lowvram",
                    x.kind(),
                    None,
                    &mut diagnostics,
                )?);
                for message in diagnostics.errors {
                    eprintln!("{message}");
                }
            }
        }
        // MixedPrecisionOps' weight-only quantization preserves the logical
        // layout; F.linear still decomposes the non-contiguous image path.
        if matches!(w, Value::Quantized(_))
            && matches!(input, Value::Tensor(_))
            && x.dim() == 3
            && !x.is_contiguous()
        {
            let out = unsafe { super::geometry::linear(rt, x.shallow_clone(), &w)? };
            return Ok(if matches!(bias, Value::None) {
                out
            } else {
                out.f_add(&bias.as_tensor()?)?
            });
        }
        let out = unsafe {
            rt.op("aten::linear.default", &[input, w, bias], &vec![])?
                .as_tensor()?
        };
        if reshaped {
            let mut shape = input_shape[..input_shape.len() - 1].to_vec();
            shape.push(self.weight.as_quantized()?.params.original_shape[0]);
            Ok(out.f_reshape(shape.as_slice())?)
        } else {
            Ok(out)
        }
    }
    /// Original `linear_input_act` INT8 path or its eager activation fallback.
    /// # Safety
    /// Same resident parameter and original fused-kernel contracts as forward.
    pub unsafe fn swiglu(
        &self,
        rt: &Runtime,
        x: &Tensor,
        training: bool,
        check: &mut dyn FnMut() -> Result<(), Error>,
    ) -> Result<Tensor, Error> {
        check()?;
        if !training
            && !self.full_precision
            && !self.force_forward
            && let Ok(q) = self.weight.as_quantized()
            && q.layout == "TensorWiseINT8Layout"
            && q.params.fields.get("transposed") != Some(&ScalarField::Bool(true))
        {
            let w = unsafe {
                rt.op(
                    "aten::to.dtype_layout",
                    std::slice::from_ref(&self.weight),
                    &vec![
                        ("dtype".into(), Value::DType(x.kind())),
                        ("device".into(), Value::Device(x.device())),
                    ],
                )?
            }
            .as_quantized()?;
            let bias = match &self.bias {
                Some(b) => Value::tensor(b.as_tensor()?.f_to_device_(
                    x.device(),
                    x.kind(),
                    false,
                    false,
                )?),
                None => Value::None,
            };
            let group = match w.params.fields.get("convrot_groupsize") {
                Some(ScalarField::Integer(g)) => *g,
                _ => 256,
            };
            return unsafe {
                rt.call_kitchen(
                    "int8_linear",
                    &vec![
                        ("x".into(), Value::tensor(x.shallow_clone())),
                        ("weight".into(), Value::tensor(w.payload.shallow_clone())),
                        (
                            "weight_scale".into(),
                            Value::tensor(w.params.tensors["scale"].shallow_clone()),
                        ),
                        ("bias".into(), bias),
                        ("out_dtype".into(), Value::DType(x.kind())),
                        (
                            "convrot".into(),
                            Value::Bool(
                                w.params.fields.get("convrot") == Some(&ScalarField::Bool(true)),
                            ),
                        ),
                        ("convrot_groupsize".into(), Value::Int(group)),
                        ("input_act".into(), Value::Text("swiglu".into())),
                    ],
                )?
                .as_tensor()
            };
        }
        let pieces = x.f_chunk(2, -1)?;
        let y = pieces[0].f_silu()?.f_mul_(&pieces[1])?;
        check()?;
        unsafe { self.forward(rt, &y) }
    }
}

pub struct Modulation {
    pub scale1: (Tensor, Tensor),
    pub gate1: (Tensor, Tensor),
    pub scale2: (Tensor, Tensor),
    pub gate2: (Tensor, Tensor),
    pub zero: Tensor,
}
impl Modulation {
    pub fn from_projection(projected: Tensor) -> Result<Self, Error> {
        let p = projected.f_chunk(4, -1)?;
        Ok(Self {
            scale1: super::split_rows(&p[0])?,
            gate1: super::split_rows(&p[1].f_tanh()?)?,
            scale2: super::split_rows(&p[2])?,
            gate2: super::split_rows(&p[3].f_tanh()?)?,
            zero: p[0].f_slice(0, 0, 1, 1)?.f_unsqueeze(1)?.f_zeros_like()?,
        })
    }
}

/// Source training branch of apply_rope1 (inference uses Kitchen).
pub fn training_rope(x: &Tensor, pe: &Tensor) -> Result<Tensor, Error> {
    let shape = x.size();
    let view = x
        .f_to_kind(pe.kind())?
        .f_reshape([shape[0], shape[1], shape[2], -1, 1, 2])?;
    let mut pe = pe.shallow_clone();
    if shape[2] != 1 && pe.size()[2] != 1 && shape[2] != pe.size()[2] {
        pe = pe.f_slice(2, 0, shape[2], 1)?;
    }
    let mut out = pe.f_select(-1, 0)?.f_mul(&view.f_select(-1, 0)?)?;
    let _ = out.f_addcmul_(&pe.f_select(-1, 1)?, &view.f_select(-1, 1)?)?;
    Ok(out.f_reshape(shape.as_slice())?.f_to_kind(x.kind())?)
}
pub fn clamp_fp16(x: Tensor) -> Result<Tensor, Error> {
    let mut x = x;
    if x.kind() == Kind::Half {
        let _ = x.f_clamp_(-65504., 65504.)?;
    }
    Ok(x)
}

/// Original MixedPrecisionOps.Embedding row lookup; text requests Float32 while
/// vision position lookup retains the original weight dtype when unspecified.
/// # Safety
/// Indices, layout, storage and stream/device capacities satisfy original kernels.
pub(super) unsafe fn embedding_rows(
    rt: &Runtime,
    weight: &Value,
    ids: &Tensor,
    out_dtype: Option<Kind>,
) -> Result<Tensor, Error> {
    match weight {
        Value::Quantized(_) => {
            let q = weight.as_quantized()?;
            if q.layout == "TensorWiseINT8Layout" {
                let group = if q.params.fields.get("convrot") == Some(&ScalarField::Bool(true)) {
                    match q.params.fields.get("convrot_groupsize") {
                        Some(ScalarField::Integer(n)) => *n,
                        _ => 256,
                    }
                } else {
                    0
                };
                let code = match q.params.original_kind {
                    Kind::Half => 1,
                    Kind::BFloat16 => 2,
                    _ => 0,
                };
                let out = unsafe {
                    rt.call_kitchen(
                        "dequantize_int8_embedding",
                        &vec![
                            (
                                "q".into(),
                                Value::tensor(q.payload.f_to_device(ids.device())?),
                            ),
                            (
                                "scale".into(),
                                Value::tensor(q.params.tensors["scale"].f_to_device(ids.device())?),
                            ),
                            ("indices".into(), Value::tensor(ids.shallow_clone())),
                            ("group_size".into(), Value::Int(group)),
                            ("output_dtype_code".into(), Value::Int(code)),
                        ],
                    )?
                }
                .as_tensor()?;
                Ok(out.f_to_kind(out_dtype.unwrap_or(q.params.original_kind))?)
            } else if matches!(
                q.layout.as_str(),
                "TensorCoreFP8E4M3Layout" | "TensorCoreFP8E5M2Layout"
            ) {
                let x = rt
                    .native
                    .embedding(&q.payload.f_to_device(ids.device())?, ids)?
                    .f_to_kind(out_dtype.unwrap_or(q.params.original_kind))?;
                Ok(x.f_mul(&q.params.tensors["scale"].f_to_device_(
                    ids.device(),
                    out_dtype.unwrap_or(q.params.original_kind),
                    false,
                    false,
                )?)?)
            } else {
                Err(Error::BindingGap {
                    backend: "qwen3vl".into(),
                    operation: format!("Embedding module fallback for {}", q.layout),
                })
            }
        }
        _ => Ok(rt
            .native
            .embedding(&weight.as_tensor()?.f_to_device(ids.device())?, ids)?
            .f_to_kind(out_dtype.unwrap_or(weight.as_tensor()?.kind()))?),
    }
}
