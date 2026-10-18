//! Original constraints.py call rules and CUDA RMS-RoPE registration predicates.
use crate::registry::{AllowedType, Argument, CallArgs, Failure};
use tch::Kind;
type Result = std::result::Result<(), Failure>;
type TensorParts<'a> = (AllowedType, &'a str, &'a [i64]);

fn tensor<'a>(args: &'a CallArgs, name: &str) -> Option<(AllowedType, &'a str, &'a [i64])> {
    args.get(name).and_then(Argument::tensor_parts)
}
fn required<'a>(
    args: &'a CallArgs,
    name: &str,
) -> std::result::Result<(AllowedType, &'a str, &'a [i64]), Failure> {
    if !args.contains_key(name) {
        return Err(Failure::raised("KeyError", format!("'{name}'")));
    }
    tensor(args, name).ok_or_else(|| Failure::raised("AssertionError", ""))
}
fn present(value: Option<&Argument>) -> bool {
    value.is_some_and(|v| !matches!(v, Argument::None))
}
fn attribute_error(value: &Argument, attr: &str) -> Failure {
    let name = match value {
        Argument::Source(v) => v.typename(),
        Argument::DType(_) | Argument::ShellDType(_) => "torch.dtype",
        Argument::Integer(_) | Argument::Unsigned(_) => "int",
        Argument::FloatValue(_) => "float",
        Argument::BoolValue(_) => "bool",
        Argument::Text(_) => "str",
        Argument::Sequence(_) | Argument::BoolSequence(_) => "list",
        Argument::None => "NoneType",
        _ => "object",
    };
    Failure::raised(
        "AttributeError",
        format!("'{name}' object has no attribute '{attr}'"),
    )
}
fn optional_tensor<'a>(
    args: &'a CallArgs,
    name: &str,
) -> std::result::Result<Option<TensorParts<'a>>, Failure> {
    if !present(args.get(name)) {
        return Ok(None);
    }
    let value = &args[name];
    value
        .tensor_parts()
        .map(Some)
        .ok_or_else(|| attribute_error(value, "shape"))
}
fn tuple(shape: &[i64]) -> String {
    format!(
        "({}{})",
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if shape.len() == 1 { "," } else { "" }
    )
}
fn dtype(kind: &AllowedType) -> &'static str {
    match kind {
        AllowedType::DType(Kind::Float) => "torch.float32",
        AllowedType::DType(Kind::Double) => "torch.float64",
        AllowedType::DType(Kind::Half) => "torch.float16",
        AllowedType::DType(Kind::BFloat16) => "torch.bfloat16",
        AllowedType::DType(Kind::Int) => "torch.int32",
        AllowedType::DType(Kind::Int8) => "torch.int8",
        AllowedType::DType(Kind::Uint8) => "torch.uint8",
        AllowedType::DType(Kind::Int64) => "torch.int64",
        AllowedType::DType(Kind::Float8e4m3fn) => "torch.float8_e4m3fn",
        AllowedType::DType(Kind::Float8e5m2) => "torch.float8_e5m2",
        AllowedType::ShellDType(_) => "torch.float8_e8m0fnu",
        _ => "unknown dtype",
    }
}
fn scalar(args: &CallArgs, name: &str) -> f64 {
    match args.get(name) {
        Some(Argument::Integer(x)) => *x as f64,
        Some(Argument::Unsigned(x)) => *x as f64,
        Some(Argument::FloatValue(x)) => *x,
        Some(Argument::BoolValue(x)) => i32::from(*x) as f64,
        _ => 0.,
    }
}
fn same_q(args: &CallArgs) -> Result {
    if present(args.get("q")) {
        for name in ["k", "v"] {
            if let Some((other_kind, other_device, other_shape)) = optional_tensor(args, name)? {
                let (kind, device, shape) = optional_tensor(args, "q")?.unwrap();
                if shape != other_shape {
                    return Err(Failure::new(
                        name,
                        format!(
                            "must have the same shape as q {}, got {}",
                            tuple(shape),
                            tuple(other_shape)
                        ),
                    ));
                }
                if kind != other_kind {
                    return Err(Failure::new(
                        name,
                        format!(
                            "must have the same dtype as q ({}), got {}",
                            dtype(&kind),
                            dtype(&other_kind)
                        ),
                    ));
                }
                if device != other_device {
                    return Err(Failure::new(
                        name,
                        format!("must be on the same device as q ({device}), got {other_device}"),
                    ));
                }
            }
        }
    }
    Ok(())
}
fn na3d(args: &CallArgs) -> Result {
    same_q(args)?;
    for name in ["kernel_size", "is_causal"] {
        let Some(value) = args.get(name).filter(|v| !matches!(v, Argument::None)) else {
            continue;
        };
        let length = match value {
            Argument::Source(v) => v.len()?,
            _ => match value.integer_sequence() {
                Some(v) => v.len(),
                None => {
                    return Err(Failure::raised(
                        "TypeError",
                        format!(
                            "object of type '{}' has no len()",
                            match value {
                                Argument::Integer(_) | Argument::Unsigned(_) => "int",
                                Argument::FloatValue(_) => "float",
                                Argument::BoolValue(_) => "bool",
                                Argument::DType(_) | Argument::ShellDType(_) => "torch.dtype",
                                Argument::Text(v) => {
                                    let length = v.chars().count();
                                    if length != 3 {
                                        return Err(Failure::new(
                                            name,
                                            format!("must have 3 elements, got {length}"),
                                        ));
                                    }
                                    continue;
                                }
                                _ => "object",
                            }
                        ),
                    ));
                }
            },
        };
        if length != 3 {
            return Err(Failure::new(
                name,
                format!("must have 3 elements, got {length}"),
            ));
        }
    }
    if let Some(value) = args
        .get("kernel_size")
        .filter(|v| !matches!(v, Argument::None))
    {
        match value {
            Argument::Source(v) if v.any_less_than_one()? => {
                return Err(Failure::new(
                    "kernel_size",
                    format!("entries must be positive, got {}", v.list_repr()?),
                ));
            }
            Argument::Text(_) => {
                return Err(Failure::raised(
                    "TypeError",
                    "'<' not supported between instances of 'str' and 'int'",
                ));
            }
            _ if value
                .integer_sequence()
                .is_some_and(|v| v.iter().any(|n| *n < 1)) =>
            {
                return Err(Failure::new(
                    "kernel_size",
                    format!(
                        "entries must be positive, got {}",
                        value.sequence_repr().unwrap()
                    ),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}
fn sol(args: &CallArgs) -> Result {
    same_q(args)?;
    if let Some((_, device, shape)) = optional_tensor(args, "q")? {
        let last = shape
            .last()
            .ok_or_else(|| Failure::raised("IndexError", "tuple index out of range"))?;
        if *last != 128 {
            return Err(Failure::new("q", "head_dim must be 128"));
        }
        if let Some((kind, other_device, other_shape)) = tensor(args, "block_len") {
            let n = (shape[1] + 63) / 64;
            if kind != AllowedType::DType(Kind::Int)
                || other_shape.len() != 1
                || other_shape.iter().product::<i64>() != n
            {
                return Err(Failure::new(
                    "block_len",
                    format!(
                        "must be int32 of shape ({n},), got {} {}",
                        dtype(&kind),
                        tuple(other_shape)
                    ),
                ));
            }
            if other_device != device {
                return Err(Failure::new(
                    "block_len",
                    format!("must be on {device}, got {other_device}"),
                ));
            }
        }
        if let Some((_, other_device, other_shape)) = tensor(args, "coarse_gate") {
            if other_shape != shape {
                return Err(Failure::new(
                    "coarse_gate",
                    format!(
                        "must have q's shape {}, got {}",
                        tuple(shape),
                        tuple(other_shape)
                    ),
                ));
            }
            if other_device != device {
                return Err(Failure::new(
                    "coarse_gate",
                    format!("must be on {device}, got {other_device}"),
                ));
            }
        }
    }
    use crate::registry_value_ops::Operand;
    let zero = Operand::integer(0);
    for name in ["sink_blocks", "sink_q"] {
        let Some(value) = args.get(name).filter(|v| !matches!(v, Argument::None)) else {
            continue;
        };
        let value = Operand::argument(value)?;
        let length = value.len()?;
        if length != 2 {
            return Err(Failure::new(
                name,
                format!("must have 2 elements [start, end), got {length}"),
            ));
        }
        let values = value.items()?;
        if values[0].lt(&zero)? || values[1].lt(&zero)? {
            return Err(Failure::new(
                name,
                format!("entries must be non-negative, got {}", value.list_repr()?),
            ));
        }
        if values[1].lt(&values[0])? {
            return Err(Failure::new(
                name,
                format!(
                    "must be [start, end) with end >= start, got {}",
                    value.list_repr()?
                ),
            ));
        }
    }
    if let Some(topk) = args.get("topk_ratio") {
        let topk = Operand::argument(topk)?;
        if topk.truthy()? && !(Operand::float(0.).lt(&topk)? && topk.lt(&Operand::float(1.))?) {
            return Err(Failure::new(
                "topk_ratio",
                format!("must be 0 or in (0, 1), got {}", topk.display()?),
            ));
        }
    }
    if let Some(aug) = args.get("token_aug") {
        let aug = Operand::argument(aug)?;
        if aug.truthy()?
            && (aug.lt(&zero)? || aug.gt(&Operand::integer(256))? || aug.remainder64()?)
        {
            return Err(Failure::new(
                "token_aug",
                format!(
                    "must be 0 or a multiple of 64 up to 256, got {}",
                    aug.display()?
                ),
            ));
        }
    }
    Ok(())
}
fn native_layout(kind: &AllowedType, shape: &[i64], freq: &[i64], rot_dim: i64) -> bool {
    if shape.len() != 4 || freq.len() != 6 {
        return false;
    }
    let head = shape[3];
    let rot = if rot_dim == 0 { head } else { rot_dim };
    if head == 0
        || head % 2 != 0
        || rot % 2 != 0
        || rot > head
        || freq[3..] != [rot / 2, 2, 2]
        || ![1, shape[0]].contains(&freq[0])
    {
        return false;
    }
    let bnhd = (freq[1] == 1 && [1, shape[2]].contains(&freq[2]))
        || ([1, shape[1]].contains(&freq[1]) && freq[2] == 1);
    bnhd && head >= 32
        && head % 32 == 0
        && [
            AllowedType::DType(Kind::Half),
            AllowedType::DType(Kind::BFloat16),
        ]
        .contains(kind)
}
fn rms_single(args: &CallArgs) -> Result {
    if !args.contains_key("x") {
        return Err(Failure::raised("KeyError", "'x'"));
    }
    if !args.contains_key("freqs_cis") {
        return Err(Failure::raised("KeyError", "'freqs_cis'"));
    }
    let (kind, _, shape) = required(args, "x")?;
    let (_, _, freq) = required(args, "freqs_cis")?;
    if !native_layout(&kind, shape, freq, 0) {
        return Err(Failure::new(
            "x",
            "CUDA fused RMS-RoPE requires a supported native layout",
        ));
    }
    Ok(())
}
fn rms_pair(args: &CallArgs) -> Result {
    let q = args
        .get("q")
        .ok_or_else(|| Failure::raised("KeyError", "'q'"))?;
    let k = args
        .get("k")
        .ok_or_else(|| Failure::raised("KeyError", "'k'"))?;
    let Some(q_scale) = args.get("q_scale").filter(|v| !matches!(v, Argument::None)) else {
        return Err(Failure::new(
            "q_scale",
            "is required for fused CUDA RMS-RoPE",
        ));
    };
    let k_scale = args
        .get("k_scale")
        .filter(|v| !matches!(v, Argument::None))
        .unwrap_or(q_scale);
    let ((kind, _, shape), (k_kind, _, k_shape), (s_kind, _, s_shape), (ks_kind, _, ks_shape)) =
        match (
            q.tensor_parts(),
            k.tensor_parts(),
            q_scale.tensor_parts(),
            k_scale.tensor_parts(),
        ) {
            (Some(q), Some(k), Some(s), Some(ks)) => (q, k, s, ks),
            _ => return Err(Failure::raised("AssertionError", "")),
        };
    let (_, _, freq) = required(args, "freqs_cis")?;
    let rot = scalar(args, "rot_dim") as i64;
    if !native_layout(&kind, shape, freq, rot) {
        return Err(Failure::new(
            "q",
            "CUDA fused RMS-RoPE requires a supported native layout",
        ));
    }
    if k_kind != kind {
        return Err(Failure::new(
            "k",
            "must have the same dtype as q for fused CUDA RMS-RoPE",
        ));
    }
    if k_shape != shape && !native_layout(&k_kind, k_shape, freq, rot) {
        return Err(Failure::new(
            "k",
            "CUDA fused RMS-RoPE requires a supported native layout",
        ));
    }
    if s_shape.len() != 1 || s_shape[0] != shape[shape.len() - 1] {
        return Err(Failure::new(
            "q_scale",
            "must be a 1D tensor matching q head_dim",
        ));
    }
    if ks_shape.len() != 1 || ks_shape[0] != k_shape[k_shape.len() - 1] {
        return Err(Failure::new(
            "k_scale",
            "must be a 1D tensor matching k head_dim",
        ));
    }
    if k_shape == shape && s_kind != ks_kind {
        return Err(Failure::new(
            "k_scale",
            "must match q_scale dtype for fused CUDA RMS-RoPE",
        ));
    }
    Ok(())
}
pub(super) fn validate(name: &str, args: &CallArgs) -> Result {
    match name {
        "sol_attn_common_call_rule" => sol(args),
        "na3d_common_call_rule" => na3d(args),
        "cuda_na3d" | "triton_na3d" => {
            na3d(args)?;
            if let Some((_, _, shape)) = optional_tensor(args, "q")? {
                let hd = *shape
                    .last()
                    .ok_or_else(|| Failure::raised("IndexError", "tuple index out of range"))?;
                if name == "triton_na3d" {
                    if hd > 128 {
                        return Err(Failure::new(
                            "q",
                            "head_dim > 128 not supported by triton na3d",
                        ));
                    }
                } else {
                    if hd % 16 != 0 || hd > 64 {
                        return Err(Failure::new(
                            "q",
                            "head_dim must be a multiple of 16 and <= 64",
                        ));
                    }
                    if shape[1] * shape[2] > 65535 || shape[0] * shape[4] > 65535 {
                        return Err(Failure::new("q", "grid dims exceed CUDA limits"));
                    }
                }
            }
            Ok(())
        }
        "_validate_cuda_rms_rope1" => rms_single(args),
        "_validate_cuda_rms_rope" => rms_pair(args),
        _ => panic!("Unknown frozen call rule {name}"),
    }
}
