//! Concrete metadata/shape/export handlers from Kitchen tensor layout modules.
//!
//! These helpers do not implement full mixed-input layout dispatch or matmul.
use crate::{
    native::Error,
    storage::{LayoutParams, QuantizedStorage, ScalarField},
};
use std::collections::BTreeMap;
use tch::{Kind, Tensor};

fn scale(q: &QuantizedStorage) -> Result<&Tensor, Error> {
    q.params
        .tensors
        .get("scale")
        .ok_or_else(|| Error::Native("Missing layout tensor field scale".into()))
}
pub fn fp8_parts(
    payload: Tensor,
    scale: Tensor,
    original_kind: Kind,
    original_shape: Vec<i64>,
) -> Result<QuantizedStorage, Error> {
    // TensorCoreFP8Layout.Params._validate_tensor_fields uses non_blocking=True.
    let scale = scale.f_to_dtype(Kind::Float, true, false)?;
    Ok(QuantizedStorage::from_validated_parts(
        payload,
        "TensorCoreFP8Layout".into(),
        LayoutParams {
            original_kind,
            original_shape,
            tensors: BTreeMap::from([("scale".into(), scale)]),
            fields: BTreeMap::new(),
            field_order: ["scale", "orig_dtype", "orig_shape"]
                .map(str::to_owned)
                .to_vec(),
        },
    ))
}
pub fn int8_parts(
    payload: Tensor,
    scale: Tensor,
    original_kind: Kind,
    original_shape: Vec<i64>,
    is_weight: bool,
    convrot: bool,
    group_size: i64,
) -> QuantizedStorage {
    QuantizedStorage::from_validated_parts(
        payload,
        "TensorWiseINT8Layout".into(),
        LayoutParams {
            original_kind,
            original_shape,
            tensors: BTreeMap::from([("scale".into(), scale)]),
            fields: BTreeMap::from([
                ("is_weight".into(), ScalarField::Bool(is_weight)),
                ("convrot".into(), ScalarField::Bool(convrot)),
                ("convrot_groupsize".into(), ScalarField::Integer(group_size)),
                ("transposed".into(), ScalarField::Bool(false)),
            ]),
            field_order: [
                "scale",
                "orig_dtype",
                "orig_shape",
                "is_weight",
                "convrot",
                "convrot_groupsize",
                "transposed",
            ]
            .map(str::to_owned)
            .to_vec(),
        },
    )
}
pub fn nvfp4_parts(
    payload: Tensor,
    scale: Tensor,
    block_scale: Tensor,
    original_kind: Kind,
    original_shape: Vec<i64>,
) -> Result<QuantizedStorage, Error> {
    // NVFP4 validates only the global scale's dtype, not block_scale.
    let scale = scale.f_to_dtype(Kind::Float, true, false)?;
    Ok(QuantizedStorage::from_validated_parts(
        payload,
        "TensorCoreNVFP4Layout".into(),
        LayoutParams {
            original_kind,
            original_shape,
            tensors: BTreeMap::from([("scale".into(), scale), ("block_scale".into(), block_scale)]),
            fields: BTreeMap::from([("transposed".into(), ScalarField::Bool(false))]),
            field_order: [
                "scale",
                "orig_dtype",
                "orig_shape",
                "block_scale",
                "transposed",
            ]
            .map(str::to_owned)
            .to_vec(),
        },
    ))
}
pub fn mxfp8_parts(
    payload: Tensor,
    scale: Tensor,
    original_kind: Kind,
    original_shape: Vec<i64>,
) -> QuantizedStorage {
    // MXFP8 scale is E8M0 and its validator is a no-op. Never cast it to FP32.
    QuantizedStorage::from_validated_parts(
        payload,
        "TensorCoreMXFP8Layout".into(),
        LayoutParams {
            original_kind,
            original_shape,
            tensors: BTreeMap::from([("scale".into(), scale)]),
            fields: BTreeMap::from([("transposed".into(), ScalarField::Bool(false))]),
            field_order: ["scale", "orig_dtype", "orig_shape", "transposed"]
                .map(str::to_owned)
                .to_vec(),
        },
    )
}
/// Original NVFP4/MXFP8 t handlers leave packed data and block scales untouched.
/// View/reshape are deliberately absent here: their fallback must dequantize.
pub fn packed_transpose(q: &QuantizedStorage) -> Result<QuantizedStorage, Error> {
    let shape = &q.params.original_shape;
    if shape.len() < 2 {
        return Err(Error::Native("IndexError: tuple index out of range".into()));
    }
    let shape = vec![shape[1], shape[0]];
    let mut result = match q.layout.as_str() {
        "TensorCoreNVFP4Layout" => nvfp4_parts(
            q.payload.shallow_clone(),
            scale(q)?.shallow_clone(),
            q.params
                .tensors
                .get("block_scale")
                .ok_or_else(|| Error::Native("Missing layout tensor field block_scale".into()))?
                .shallow_clone(),
            q.params.original_kind,
            shape,
        )?,
        "TensorCoreMXFP8Layout" => mxfp8_parts(
            q.payload.shallow_clone(),
            scale(q)?.shallow_clone(),
            q.params.original_kind,
            shape,
        ),
        _ => {
            return Err(Error::Native(
                "Expected NVFP4 or MXFP8 layout handler".into(),
            ));
        }
    };
    let transposed = matches!(
        q.params.fields.get("transposed"),
        Some(ScalarField::Bool(true))
    );
    result
        .params
        .fields
        .insert("transposed".into(), ScalarField::Bool(!transposed));
    Ok(result)
}
pub fn packed_state_dict<'a>(
    q: &'a QuantizedStorage,
    prefix: &str,
) -> Result<Vec<(String, &'a Tensor)>, Error> {
    match q.layout.as_str() {
        "TensorCoreNVFP4Layout" => Ok(vec![
            (prefix.to_owned(), &q.payload),
            (
                format!("{prefix}_scale"),
                q.params.tensors.get("block_scale").ok_or_else(|| {
                    Error::Native("Missing layout tensor field block_scale".into())
                })?,
            ),
            (format!("{prefix}_scale_2"), scale(q)?),
        ]),
        "TensorCoreMXFP8Layout" => Ok(vec![
            (prefix.to_owned(), &q.payload),
            (format!("{prefix}_scale"), scale(q)?),
        ]),
        _ => Err(Error::Native(
            "Expected NVFP4 or MXFP8 state-dictionary handler".into(),
        )),
    }
}
pub enum Fp8ShapeOp<'a> {
    View(&'a [i64]),
    Reshape(&'a [i64]),
    Transpose,
}
pub fn fp8_shape(q: &QuantizedStorage, op: Fp8ShapeOp<'_>) -> Result<QuantizedStorage, Error> {
    if q.layout != "TensorCoreFP8Layout" {
        return Err(Error::Native("Expected FP8 layout handler".into()));
    }
    let payload = match op {
        Fp8ShapeOp::View(shape) => q.payload.f_view(shape)?,
        Fp8ShapeOp::Reshape(shape) => q.payload.f_reshape(shape)?,
        Fp8ShapeOp::Transpose => q.payload.f_tr()?,
    };
    let shape = payload.size();
    fp8_parts(
        payload,
        scale(q)?.shallow_clone(),
        q.params.original_kind,
        shape,
    )
}
pub fn int8_transpose(q: &QuantizedStorage) -> Result<QuantizedStorage, Error> {
    if q.layout != "TensorWiseINT8Layout" {
        return Err(Error::Native(
            "Expected tensorwise INT8 layout handler".into(),
        ));
    }
    let shape = &q.params.original_shape;
    if shape.len() < 2 {
        return Err(Error::Native("IndexError: tuple index out of range".into()));
    }
    let mut params = q.params.shallow_copy();
    params.original_shape = vec![shape[1], shape[0]];
    let transposed = matches!(
        params.fields.get("transposed"),
        Some(ScalarField::Bool(true))
    );
    params
        .fields
        .insert("transposed".into(), ScalarField::Bool(!transposed));
    Ok(QuantizedStorage::from_validated_parts(
        q.payload.shallow_clone(),
        q.layout.clone(),
        params,
    ))
}
/// State-dictionary mapping for the two tensorwise layouts, retaining tensors
/// rather than casting, cloning or writing checkpoint/runtime data to disk.
pub fn tensorwise_state_dict<'a>(
    q: &'a QuantizedStorage,
    prefix: &str,
) -> Result<Vec<(String, &'a Tensor)>, Error> {
    if q.layout != "TensorCoreFP8Layout" && q.layout != "TensorWiseINT8Layout" {
        return Err(Error::Native(
            "Expected tensorwise state-dictionary handler".into(),
        ));
    }
    Ok(vec![
        (prefix.to_owned(), &q.payload),
        (format!("{prefix}_scale"), scale(q)?),
    ])
}
pub fn int8_requantize_kwargs(
    q: &QuantizedStorage,
) -> Result<BTreeMap<String, ScalarField>, Error> {
    if q.layout != "TensorWiseINT8Layout" {
        return Err(Error::Native(
            "Expected tensorwise INT8 layout handler".into(),
        ));
    }
    let is_weight = !matches!(
        q.params.fields.get("is_weight"),
        Some(ScalarField::Bool(false))
    );
    let convrot = matches!(
        q.params.fields.get("convrot"),
        Some(ScalarField::Bool(true))
    );
    let group = q
        .params
        .fields
        .get("convrot_groupsize")
        .cloned()
        .unwrap_or(ScalarField::Integer(256));
    Ok(BTreeMap::from([
        ("is_weight".into(), ScalarField::Bool(is_weight)),
        (
            "per_channel".into(),
            ScalarField::Bool(is_weight && (convrot || scale(q)?.dim() > 0)),
        ),
        ("convrot".into(), ScalarField::Bool(convrot)),
        ("convrot_groupsize".into(), group),
    ]))
}
/// QuantizedTensor.dequantize's common contiguous/crop/logical-transpose policy.
/// The callback supplies the chosen layout/backend's actual dequantizer.
pub fn dequantize_with(
    q: &QuantizedStorage,
    dequantizer: impl FnOnce(&Tensor, &LayoutParams) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let payload = if q.payload.is_contiguous() {
        q.payload.shallow_clone()
    } else {
        q.payload.f_contiguous()?
    };
    let transposed = matches!(
        q.params.fields.get("transposed"),
        Some(ScalarField::Bool(true))
    );
    let orig = &q.params.original_shape;
    if transposed && orig.len() < 2 {
        return Err(Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        });
    }
    let mut full = dequantizer(&payload, &q.params)?;
    let shape = if transposed {
        vec![orig[1], orig[0]]
    } else {
        orig.clone()
    };
    let needs_crop = if transposed {
        full.size().get(..2) != Some(shape.as_slice())
    } else {
        full.size() != shape
    };
    if needs_crop {
        for (dim, size) in shape.iter().enumerate() {
            full = full.f_slice(dim as i64, 0, *size, 1)?;
        }
    }
    Ok(if transposed { full.f_tr()? } else { full })
}
