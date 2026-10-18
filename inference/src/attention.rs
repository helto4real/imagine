//! Shared source attention wrappers. The caller supplies the original policy;
//! no backend is selected from a model filename or a performance heuristic.
use crate::{
    execution::Runtime,
    execution_value::Value,
    native::{Error, SdpaOptions},
};
use serde_json::Value as Json;
use tch::{Device, Kind, Tensor};

#[derive(Clone)]
pub enum AttentionMethod {
    Pytorch,
    Basic,
    KitchenInt8,
    KitchenSol { tau: Value },
}
#[derive(Clone)]
pub struct Attention {
    pub method: AttentionMethod,
    pub priority_wrapper: bool,
    pub attention_precision: Option<Kind>,
    pub low_precision_attention: bool,
    /// Source get_torch_device used when loading ComfyAttention preferences.
    pub model_device: Device,
}
#[derive(Clone, Copy)]
pub struct AttentionInput<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub heads: i64,
    pub mask: Option<&'a Tensor>,
    pub skip_reshape: bool,
    pub skip_output_reshape: bool,
    pub enable_gqa: bool,
    pub scale: Option<f64>,
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn scalar(v: &Json) -> Result<Value, Error> {
    Ok(match v {
        Json::Null => Value::None,
        Json::Bool(v) => Value::Bool(*v),
        Json::String(v) => Value::Text(v.clone()),
        Json::Number(v) => {
            if let Some(n) = v.as_i64() {
                Value::Int(n)
            } else {
                Value::Float(v.as_f64().ok_or_else(|| Error::BindingGap {
                    backend: "attention".into(),
                    operation: "Unbounded JSON numeric metadata".into(),
                })?)
            }
        }
        _ => {
            return Err(Error::BindingGap {
                backend: "attention".into(),
                operation: "Structured Sol tau metadata".into(),
            });
        }
    })
}
pub fn repeat_kv(
    q_heads: i64,
    k: &Tensor,
    v: &Tensor,
    dim: i64,
) -> Result<(Tensor, Tensor), Error> {
    let index = if dim < 0 {
        (k.dim() as i64 + dim) as usize
    } else {
        dim as usize
    };
    let kh = k.size()[index];
    let vh = v.size()[index];
    if kh != vh {
        return Err(error(
            "ValueError",
            format!("Key/value head count mismatch for GQA: {kh} != {vh}"),
        ));
    }
    if q_heads == kh {
        return Ok((k.shallow_clone(), v.shallow_clone()));
    }
    if kh == 0 {
        return Err(error("ZeroDivisionError", "integer modulo by zero"));
    }
    if q_heads % kh != 0 {
        return Err(error(
            "ValueError",
            format!("Query heads must be divisible by key/value heads for GQA: {q_heads} vs {kh}"),
        ));
    }
    let n = q_heads / kh;
    Ok(if n > 1 {
        (
            k.f_repeat_interleave_self_int(n, dim, None)?,
            v.f_repeat_interleave_self_int(n, dim, None)?,
        )
    } else {
        (k.shallow_clone(), v.shallow_clone())
    })
}
fn shape_qkv(
    input: &AttentionInput<'_>,
    expand: bool,
) -> Result<(Tensor, Tensor, Tensor, i64, i64), Error> {
    let b = input.q.size()[0];
    let heads = input.heads;
    if input.skip_reshape {
        let d = input.q.size()[3];
        return Ok((
            input.q.shallow_clone(),
            input.k.shallow_clone(),
            input.v.shallow_clone(),
            b,
            d,
        ));
    }
    if heads == 0 {
        return Err(error(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let d = input.q.size()[2] / heads;
    let head_count = |t: &Tensor, name: &str| -> Result<i64, Error> {
        if d == 0 {
            return Err(error("ZeroDivisionError", "integer modulo by zero"));
        }
        let inner = *t.size().last().unwrap();
        if inner % d != 0 {
            return Err(error(
                "ValueError",
                format!("{name} inner dimension {inner} is not divisible by head dimension {d}"),
            ));
        }
        Ok(inner / d)
    };
    let kh = if input.enable_gqa {
        head_count(input.k, "Key")?
    } else {
        heads
    };
    let vh = if input.enable_gqa {
        head_count(input.v, "Value")?
    } else {
        heads
    };
    let q = input.q.f_unsqueeze(3)?.f_reshape([b, -1, heads, d])?;
    let k = input.k.f_unsqueeze(3)?.f_reshape([b, -1, kh, d])?;
    let v = input.v.f_unsqueeze(3)?.f_reshape([b, -1, vh, d])?;
    let (k, v) = if input.enable_gqa && expand {
        repeat_kv(heads, &k, &v, -2)?
    } else {
        (k, v)
    };
    Ok((
        q.f_transpose(1, 2)?,
        k.f_transpose(1, 2)?,
        v.f_transpose(1, 2)?,
        b,
        d,
    ))
}
fn mask_heads(mask: Option<&Tensor>) -> Result<Option<Tensor>, Error> {
    mask.map(|m| {
        let mut m = m.shallow_clone();
        if m.dim() == 2 {
            m = m.f_unsqueeze(0)?;
        }
        if m.dim() == 3 {
            m = m.f_unsqueeze(1)?;
        }
        Ok(m)
    })
    .transpose()
}
impl Attention {
    pub fn preferred(&self, rt: &Runtime, config: Option<&Json>) -> Result<AttentionMethod, Error> {
        let configs = match config {
            None => return Ok(self.method.clone()),
            Some(Json::Array(v)) => v.iter().collect::<Vec<_>>(),
            Some(v) => vec![v],
        };
        for config in configs {
            if !config.is_object() {
                let name = match config {
                    Json::Null => "NoneType",
                    Json::Bool(_) => "bool",
                    Json::Number(n) => {
                        if n.is_i64() || n.is_u64() {
                            "int"
                        } else {
                            "float"
                        }
                    }
                    Json::String(_) => "str",
                    Json::Array(_) => "list",
                    Json::Object(_) => unreachable!(),
                };
                return Err(error(
                    "AttributeError",
                    format!("'{name}' object has no attribute 'get'"),
                ));
            }
            match config.get("attention").and_then(Json::as_str) {
                Some("comfy_kitchen_int8")
                    if rt.int8_attention_available(Some(self.model_device))? =>
                {
                    return Ok(AttentionMethod::KitchenInt8);
                }
                Some("comfy_kitchen_sol")
                    if matches!(self.model_device, Device::Cuda(_))
                        && rt.native.cuda_available()?
                        && rt.backends.policy.is_available("cuda") =>
                {
                    let device = match self.model_device {
                        Device::Cuda(i) => i,
                        _ => unreachable!(),
                    };
                    let p = rt.native.device_properties(device)?;
                    if (p.major, p.minor) >= (8, 0)
                        && rt
                            .backends
                            .policy
                            .get_constraints("cuda", "sol_attn")
                            .is_some()
                    {
                        return Ok(AttentionMethod::KitchenSol {
                            tau: match config.get("tau") {
                                Some(v) => scalar(v)?,
                                None => Value::Float(1.),
                            },
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(self.method.clone())
    }
    /// # Safety
    /// Source-selected Kitchen methods must receive their original valid native
    /// storage/capacity/stream contracts. No substitute backend is forced here.
    pub unsafe fn run(&self, rt: &Runtime, input: AttentionInput<'_>) -> Result<Tensor, Error> {
        match self.method.clone() {
            AttentionMethod::Pytorch => self.pytorch(rt, &input),
            AttentionMethod::Basic => self.basic(&input),
            AttentionMethod::KitchenInt8 => {
                if !self.low_precision_attention && input.q.kind() == Kind::Float {
                    return self.pytorch(rt, &input);
                }
                let (q, k, v, b, d) = shape_qkv(&input, false)?;
                let mask = mask_heads(input.mask)?;
                let out = unsafe { rt.int8_attention(&q, &k, &v, input.scale, mask.as_ref())? };
                if input.skip_output_reshape {
                    Ok(out)
                } else {
                    Ok(out.f_transpose(1, 2)?.f_reshape([b, -1, input.heads * d])?)
                }
            }
            AttentionMethod::KitchenSol { tau } => {
                if input.mask.is_some() {
                    return self.pytorch(rt, &input);
                }
                let (q, k, v, b, d) = shape_qkv(
                    &AttentionInput {
                        enable_gqa: false,
                        ..input
                    },
                    true,
                )?;
                let out = unsafe {
                    rt.call_kitchen(
                        "sol_attn",
                        &vec![
                            ("q".into(), Value::tensor(q.f_transpose(1, 2)?)),
                            ("k".into(), Value::tensor(k.f_transpose(1, 2)?)),
                            ("v".into(), Value::tensor(v.f_transpose(1, 2)?)),
                            ("tau".into(), tau),
                            (
                                "scale".into(),
                                input.scale.map(Value::Float).unwrap_or(Value::None),
                            ),
                        ],
                    )?
                }
                .as_tensor()?;
                if input.skip_output_reshape {
                    Ok(out.f_transpose(1, 2)?)
                } else {
                    Ok(out.f_reshape([b, -1, input.heads * d])?)
                }
            }
        }
    }
    fn pytorch(&self, rt: &Runtime, input: &AttentionInput<'_>) -> Result<Tensor, Error> {
        let (q, k, v, b, d) = shape_qkv(input, false)?;
        let mask = mask_heads(input.mask)?;
        if b <= 32768 {
            let out = rt.native.comfy_sdpa(
                &q,
                &k,
                &v,
                SdpaOptions {
                    mask: mask.as_ref(),
                    scale: input.scale,
                    enable_gqa: input.enable_gqa,
                    priority_wrapper: self.priority_wrapper,
                },
            )?;
            if input.skip_output_reshape {
                Ok(out)
            } else {
                Ok(out.f_transpose(1, 2)?.f_reshape([b, -1, input.heads * d])?)
            }
        } else {
            let out = Tensor::f_empty([b, q.size()[2], input.heads * d], (q.kind(), q.device()))?;
            for start in (0..b).step_by(32768) {
                let end = (start + 32768).min(b);
                let m = mask
                    .as_ref()
                    .map(|m| {
                        if m.size()[0] > 1 {
                            m.f_slice(0, start, end, 1)
                        } else {
                            Ok(m.shallow_clone())
                        }
                    })
                    .transpose()?;
                let y = rt.native.comfy_sdpa(
                    &q.f_slice(0, start, end, 1)?,
                    &k.f_slice(0, start, end, 1)?,
                    &v.f_slice(0, start, end, 1)?,
                    SdpaOptions {
                        mask: m.as_ref(),
                        scale: input.scale,
                        enable_gqa: input.enable_gqa,
                        priority_wrapper: self.priority_wrapper,
                    },
                )?;
                out.f_slice(0, start, end, 1)?
                    .f_copy_(&y.f_transpose(1, 2)?.f_reshape([
                        -1,
                        q.size()[2],
                        input.heads * d,
                    ])?)?;
            }
            Ok(out)
        }
    }
    fn basic(&self, input: &AttentionInput<'_>) -> Result<Tensor, Error> {
        let (q, k, v, b, d) = shape_qkv(input, true)?;
        let h = input.heads;
        let (k, v) = if input.skip_reshape && input.enable_gqa {
            repeat_kv(q.size()[1], &k, &v, -3)?
        } else {
            (k, v)
        };
        let mut q = q.f_reshape([b * h, -1, d])?;
        let mut k = k.f_reshape([b * h, -1, d])?;
        let mut v = v.f_reshape([b * h, -1, d])?;
        if !input.skip_reshape {
            q = q.f_contiguous()?;
            k = k.f_contiguous()?;
            v = v.f_contiguous()?;
        }
        let (q, k) = if self.attention_precision == Some(Kind::Float) {
            (q.f_to_kind(Kind::Float)?, k.f_to_kind(Kind::Float)?)
        } else {
            (q, k)
        };
        let mut sim = Tensor::f_einsum("b i d, b j d -> b i j", &[&q, &k], None::<&[i64]>)?
            .f_mul_scalar(input.scale.unwrap_or((d as f64).powf(-0.5)))?;
        if let Some(mask) = input.mask {
            if mask.kind() == Kind::Bool {
                let m = mask
                    .f_reshape([mask.size()[0], -1])?
                    .f_repeat_interleave_self_int(h, 0, None)?
                    .f_unsqueeze(1)?;
                let min = match sim.kind() {
                    Kind::Half => -65504.,
                    Kind::BFloat16 => -(f32::from_bits(0x7f7f0000) as f64),
                    Kind::Float => -(f32::MAX as f64),
                    _ => -f64::MAX,
                };
                sim = sim.f_masked_fill_(&m.f_logical_not()?, min)?;
            } else {
                let s = mask.size();
                let bs = if s.len() == 2 { 1 } else { s[0] };
                let n = s.len();
                let mask = mask
                    .f_reshape([bs, -1, s[n - 2], s[n - 1]])?
                    .f_expand([b, h, -1, -1], false)?
                    .f_reshape([-1, s[n - 2], s[n - 1]])?;
                sim = sim.f_add_(&mask)?;
            }
        }
        let probs = sim.f_softmax(-1, None)?.f_to_kind(v.kind())?;
        let out = Tensor::f_einsum("b i j, b j d -> b i d", &[&probs, &v], None::<&[i64]>)?
            .f_unsqueeze(0)?
            .f_reshape([b, h, -1, d])?;
        if input.skip_output_reshape {
            Ok(out)
        } else {
            Ok(out.f_permute([0, 2, 1, 3])?.f_reshape([b, -1, h * d])?)
        }
    }
}
