//! Numeric scalar specializations shared by original registered JIT functions.
use super::{
    Element,
    expressions::{Ty, Value},
};
#[derive(Clone, Copy)]
pub enum ScalarType {
    F32,
    F64,
    I32 { divisible_by_16: bool },
    I64 { divisible_by_16: bool },
    U64 { divisible_by_16: bool },
    One,
    Bool,
}
impl ScalarType {
    pub(super) fn ty(self) -> Ty {
        match self.signature() {
            "fp32" => Ty::Float(Element::F32),
            "fp64" => Ty::F64,
            "i32" | "constexpr" => Ty::I32,
            "i64" => Ty::I64,
            "u64" => Ty::U64,
            "u1" => Ty::I1,
            _ => unreachable!("Original numeric scalar signature"),
        }
    }
    pub fn signature(self) -> &'static str {
        match self {
            Self::F32 => "fp32",
            Self::F64 => "fp64",
            Self::I32 { .. } => "i32",
            Self::I64 { .. } => "i64",
            Self::U64 { .. } => "u64",
            Self::One => "constexpr",
            Self::Bool => "u1",
        }
    }
    pub(crate) fn divisible(self) -> bool {
        matches!(
            self,
            Self::I32 {
                divisible_by_16: true
            } | Self::I64 {
                divisible_by_16: true
            } | Self::U64 {
                divisible_by_16: true
            }
        )
    }
    pub(super) fn value(self, name: &str) -> Value {
        if matches!(self, Self::One) {
            Value::integer(1)
        } else {
            Value {
                id: format!("%{name}"),
                ty: self.ty(),
                shape: vec![],
                literal: None,
                float_literal: None,
                constexpr: false,
            }
        }
    }
}
