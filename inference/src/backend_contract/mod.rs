//! Frozen Kitchen registration contract, separate from executable capabilities.
//!
//! A table entry is not an implementation: callers must bind actual operations
//! before advertising a runnable backend. No table registers itself globally.
mod rules;
mod tables;
use crate::{native::ShellDType, registry::*};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use tch::Kind;

pub(crate) fn validate_sol(args: &CallArgs) -> Result<(), Failure> {
    rules::validate("sol_attn_common_call_rule", args)
}

pub fn constraints(
    backend: &str,
    cublas_lt_available: bool,
) -> Option<BTreeMap<String, FunctionConstraints>> {
    Some(match backend {
        "eager" => tables::eager(),
        "triton" => tables::triton(),
        "cuda" => {
            let mut table = tables::cuda();
            if cublas_lt_available {
                table.extend(tables::cuda_lt_additions());
            }
            table
        }
        _ => return None,
    })
}

fn dtype(name: &str) -> AllowedType {
    use AllowedType::{DType as D, Scalar as S, ShellDType as H};
    match name {
        "torch.float32" => D(Kind::Float),
        "torch.float16" => D(Kind::Half),
        "torch.bfloat16" => D(Kind::BFloat16),
        "torch.float8_e4m3fn" => D(Kind::Float8e4m3fn),
        "torch.float8_e5m2" => D(Kind::Float8e5m2),
        "torch.float8_e8m0fnu" => H(ShellDType::E8M0),
        "torch.int8" => D(Kind::Int8),
        "torch.uint8" => D(Kind::Uint8),
        "torch.int32" => D(Kind::Int),
        "torch.int64" => D(Kind::Int64),
        "int" => S(ScalarType::Int),
        "float" => S(ScalarType::Float),
        "bool" => S(ScalarType::Bool),
        "str" => S(ScalarType::Str),
        "NoneType" => S(ScalarType::NoneType),
        _ => panic!("Unknown frozen contract dtype {name}"),
    }
}
fn devices(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).into()).collect()
}
fn p(
    name: &str,
    types: Option<&[&str]>,
    allowed_devices: Option<&[&str]>,
    shape_rules: Vec<ShapeRule>,
) -> (String, ParamConstraint) {
    (
        name.into(),
        ParamConstraint {
            types: types.map(|names| names.iter().map(|name| dtype(name)).collect()),
            devices: allowed_devices.map(devices),
            shape_rules,
        },
    )
}
fn f(
    params: Vec<(String, ParamConstraint)>,
    allowed_devices: &[&str],
    min_capability: Option<(u32, u32)>,
    call_rules: &[&str],
) -> FunctionConstraints {
    FunctionConstraints {
        params,
        default_devices: devices(allowed_devices),
        min_capability,
        call_rules: call_rules
            .iter()
            .map(|name| {
                let name = (*name).to_owned();
                Arc::new(move |args: &CallArgs| rules::validate(&name, args)) as CallRule
            })
            .collect(),
    }
}
