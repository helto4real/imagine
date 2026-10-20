//! Ordered, alias-preserving state-dict helpers from the frozen comfy.utils.
//! Maps remain shared on errors; QKV splitting creates views of original storage.
use crate::{
    execution::Runtime, execution_value::Value as V, native::Error, patching::options::Options,
};
use tch::Kind;

fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn atom(value: &Options) -> Result<&V, Error> {
    if let Options::Atom(value) = value {
        return Ok(value);
    }
    Err(exception(
        "AttributeError",
        format!("'{}' object has no attribute 'shape'", value.typename()),
    ))
}
fn shape(value: &Options) -> Result<Vec<i64>, Error> {
    match atom(value)? {
        V::Tensor(value) => Ok(value.lock().unwrap().size()),
        V::Quantized(value) => Ok(value.lock().unwrap().params.original_shape.clone()),
        _ => Err(exception(
            "AttributeError",
            format!("'{}' object has no attribute 'shape'", value.typename()),
        )),
    }
}
fn dtype(value: &Options) -> Result<Kind, Error> {
    match value {
        Options::Atom(V::Tensor(value)) => Ok(value.lock().unwrap().kind()),
        Options::Atom(V::Quantized(value)) => Ok(value.lock().unwrap().params.original_kind),
        _ => Err(exception(
            "AttributeError",
            format!("'{}' object has no attribute 'dtype'", value.typename()),
        )),
    }
}
fn numel(value: &Options, method: &str) -> Result<u64, Error> {
    if !matches!(value, Options::Atom(V::Tensor(_) | V::Quantized(_))) {
        return Err(exception(
            "AttributeError",
            format!("'{}' object has no attribute '{method}'", value.typename()),
        ));
    }
    shape(value)?.into_iter().try_fold(1_u64, |a, b| {
        a.checked_mul(b as u64).ok_or_else(|| Error::BindingGap {
            backend: "state_dict".into(),
            operation: "tensor element count exceeds u64".into(),
        })
    })
}
pub fn calculate_parameters(sd: &Options, prefix: &str) -> Result<u64, Error> {
    let mut result = 0_u64;
    for (key, value) in sd.dict_items()? {
        if key.starts_with(prefix) {
            result = result
                .checked_add(numel(&value, "nelement")?)
                .ok_or_else(|| Error::BindingGap {
                    backend: "state_dict".into(),
                    operation: "parameter total exceeds u64".into(),
                })?;
        }
    }
    Ok(result)
}
pub fn weight_dtype(sd: &Options, prefix: &str) -> Result<Option<Kind>, Error> {
    let mut counts = Vec::<(Kind, u64)>::new();
    for (key, value) in sd.dict_items()? {
        if !key.starts_with(prefix) {
            continue;
        }
        // dtype is read before numel, including the original failure boundary.
        let kind = dtype(&value)?;
        let count = numel(&value, "numel")?;
        if let Some((_, total)) = counts.iter_mut().find(|(k, _)| *k == kind) {
            *total = total.checked_add(count).ok_or_else(|| Error::BindingGap {
                backend: "state_dict".into(),
                operation: "dtype count exceeds u64".into(),
            })?;
        } else {
            counts.push((kind, count));
        }
    }
    // Python max keeps the first inserted dtype on ties, even for empty tensors.
    let mut best = None;
    for (kind, count) in counts {
        if best.is_none_or(|(_, n)| count > n) {
            best = Some((kind, count));
        }
    }
    Ok(best.map(|(kind, _)| kind))
}
pub fn replace_keys(sd: &Options, replacements: &[(String, String)]) -> Result<Options, Error> {
    for (from, to) in replacements {
        if let Some(value) = sd.pop(from)? {
            sd.set(to, value)?;
        }
    }
    Ok(sd.clone())
}
pub fn replace_prefixes(
    sd: &Options,
    replacements: &[(String, String)],
    filter: bool,
) -> Result<Options, Error> {
    let out = if filter {
        Options::dictionary(vec![])
    } else {
        sd.clone()
    };
    for (prefix, to) in replacements {
        // Snapshot keys per prefix, not once before the loop. Earlier replacements
        // can become inputs to later prefixes and can overwrite pending keys.
        let keys = sd
            .dict_items()?
            .into_iter()
            .filter_map(|(key, _)| {
                key.strip_prefix(prefix)
                    .map(|tail| (key.clone(), format!("{to}{tail}")))
            })
            .collect::<Vec<_>>();
        for (key, target) in keys {
            let value = sd.pop(&key)?.ok_or_else(|| {
                exception(
                    "KeyError",
                    crate::triton_specialization::python_str_repr(&key),
                )
            })?;
            out.set(&target, value)?;
        }
    }
    Ok(out)
}
pub fn detect_layer_quantization(sd: &Options, prefix: &str) -> Result<bool, Error> {
    Ok(sd
        .dict_items()?
        .iter()
        .any(|(key, _)| key.starts_with(prefix) && key.ends_with(".comfy_quant")))
}
/// Original legacy-FP8 and explicit JSON quantization metadata normalization.
/// The input map keeps the original partial pop mutations even when conversion
/// fails; explicit metadata instead writes directly into the same shared map.
///
/// # Safety
/// Tensor operands and runtime must satisfy the frozen ATen/Kitchen ABI.
pub unsafe fn convert_old_quants(
    rt: &Runtime,
    state: &Options,
    prefix: &str,
    metadata: &Options,
) -> Result<(Options, Options), Error> {
    use super::metadata_json::{self as j, Json};
    let metadata = if matches!(metadata, Options::Atom(V::None)) {
        Options::dictionary(vec![])
    } else {
        metadata.clone()
    };
    let mut output = state.clone();
    let quant = if let Some(encoded) = metadata.get("_quantization_metadata")? {
        Some(j::loads(&encoded)?)
    } else {
        let marker = format!("{prefix}scaled_fp8");
        if let Some(value) = state.get(&marker)? {
            let _dtype = dtype(&value)?;
            let full = numel(&value, "nelement")? == 2;
            output = Options::dictionary(vec![]);
            let mut layers = vec![];
            for (key, _) in state.dict_items()? {
                if key == marker {
                    continue;
                }
                if !key.starts_with(prefix) {
                    output.set(&key, state.get(&key)?.unwrap())?;
                    continue;
                }
                let value = state.pop(&key)?.unwrap();
                let mut target = key.clone();
                if let Some(layer) = key.strip_suffix(".scale_weight") {
                    target = format!("{layer}.weight_scale");
                    let mut fields =
                        vec![(j::points("format"), Json::Text(j::points("float8_e4m3fn")))];
                    if full {
                        fields.push((j::points("full_precision_matrix_mult"), Json::Bool(true)));
                    }
                    let layer = j::points(layer);
                    if let Some((_, old)) = layers.iter_mut().find(|(name, _)| *name == layer) {
                        *old = Json::Dict(fields);
                    } else {
                        layers.push((layer, Json::Dict(fields)));
                    }
                }
                if let Some(layer) = target.strip_suffix(".scale_input") {
                    let new_target = format!("{layer}.input_scale");
                    if !matches!(&value, Options::Atom(V::Tensor(_) | V::Quantized(_))) {
                        return Err(exception(
                            "AttributeError",
                            format!("'{}' object has no attribute 'item'", value.typename()),
                        ));
                    }
                    let scalar =
                        unsafe { rt.op("aten::item.default", &[value.snapshot()?], &vec![]) }?;
                    if crate::conditioning::is_equal(&scalar, &V::Float(1.))? {
                        continue;
                    }
                    target = new_target;
                }
                output.set(&target, value)?;
            }
            Some(Json::Dict(vec![(j::points("layers"), Json::Dict(layers))]))
        } else {
            None
        }
    };
    if let Some(quant) = quant
        && !matches!(quant, Json::Null)
    {
        for (key, value) in quant.get("layers")?.items()? {
            let key = format!("{}.comfy_quant", j::string(key)?);
            let bytes = value.dumps();
            let tensor = tch::Tensor::f_from_slice(bytes.as_bytes())?;
            output.set(&key, Options::Atom(V::tensor(tensor)))?;
        }
    }
    Ok((output, metadata))
}
pub fn unet_prefix(sd: &Options) -> Result<&'static str, Error> {
    let entries = sd.dict_items()?;
    let contains = |key: &str| entries.iter().any(|(k, _)| k == key);
    if entries.iter().any(|(k, _)| k.starts_with("detector."))
        && entries.iter().any(|(k, _)| k.starts_with("tracker."))
    {
        return Ok("");
    }
    if contains("fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight")
        && contains("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight")
    {
        return Ok("");
    }
    let candidates = ["model.diffusion_model.", "model.model.", "net."];
    let mut counts = [0; 3];
    for (key, _) in entries {
        if let Some(index) = candidates.iter().position(|prefix| key.starts_with(prefix)) {
            counts[index] += 1;
        }
    }
    let mut top = 0;
    for index in 1..3 {
        if counts[index] > counts[top] {
            top = index;
        }
    }
    Ok(if counts[top] > 5 {
        candidates[top]
    } else {
        "model."
    })
}
fn move_key(sd: &Options, from: &str, to: &str) -> Result<(), Error> {
    if let Some(value) = sd.pop(from)? {
        sd.set(to, value)?;
    }
    Ok(())
}
/// # Safety
/// Tensor operands and the runtime must satisfy the frozen ATen/Kitchen ABI.
pub unsafe fn transformers_convert(
    rt: &Runtime,
    sd: &Options,
    from: &str,
    to: &str,
    number: i64,
) -> Result<Options, Error> {
    for (old, new) in [
        (
            "positional_embedding",
            "embeddings.position_embedding.weight",
        ),
        (
            "token_embedding.weight",
            "embeddings.token_embedding.weight",
        ),
        ("ln_final.weight", "final_layer_norm.weight"),
        ("ln_final.bias", "final_layer_norm.bias"),
    ] {
        move_key(sd, &format!("{from}{old}"), &format!("{to}{new}"))?;
    }
    for block in 0..number {
        for (old, new) in [
            ("ln_1", "layer_norm1"),
            ("ln_2", "layer_norm2"),
            ("mlp.c_fc", "mlp.fc1"),
            ("mlp.c_proj", "mlp.fc2"),
            ("attn.out_proj", "self_attn.out_proj"),
        ] {
            for kind in ["weight", "bias"] {
                move_key(
                    sd,
                    &format!("{from}transformer.resblocks.{block}.{old}.{kind}"),
                    &format!("{to}encoder.layers.{block}.{new}.{kind}"),
                )?;
            }
        }
        for kind in ["weight", "bias"] {
            if let Some(weights) = sd.pop(&format!(
                "{from}transformer.resblocks.{block}.attn.in_proj_{kind}"
            ))? {
                let shape = shape(&weights)?;
                let width = shape
                    .first()
                    .ok_or_else(|| exception("IndexError", "tuple index out of range"))?
                    / 3;
                for (index, name) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
                    let index = index as i64;
                    let slice = unsafe {
                        rt.op(
                            "aten::slice.Tensor",
                            &[
                                atom(&weights)?.clone(),
                                V::Int(0),
                                V::Int(width * index),
                                V::Int(width * (index + 1)),
                                V::Int(1),
                            ],
                            &vec![],
                        )
                    }?;
                    sd.set(
                        &format!("{to}encoder.layers.{block}.self_attn.{name}.{kind}"),
                        Options::Atom(slice),
                    )?;
                }
            }
        }
    }
    Ok(sd.clone())
}
/// # Safety
/// Same operand and runtime requirements as `transformers_convert`.
pub unsafe fn clip_text_transformers_convert(
    rt: &Runtime,
    sd: &Options,
    from: &str,
    to: &str,
) -> Result<Options, Error> {
    unsafe { transformers_convert(rt, sd, from, &format!("{to}text_model."), 32) }?;
    move_key(
        sd,
        &format!("{from}text_projection.weight"),
        &format!("{to}text_projection.weight"),
    )?;
    if let Some(weight) = sd.pop(&format!("{from}text_projection"))? {
        let transposed = unsafe {
            rt.op(
                "aten::transpose.int",
                &[atom(&weight)?.clone(), V::Int(0), V::Int(1)],
                &vec![],
            )
        }?;
        let output = unsafe { rt.op("aten::contiguous", &[transposed], &vec![]) }?;
        sd.set(
            &format!("{to}text_projection.weight"),
            Options::Atom(output),
        )?;
    }
    Ok(sd.clone())
}
