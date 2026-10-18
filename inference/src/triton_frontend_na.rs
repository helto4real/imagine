//! Source-positioned online softmax and three nested neighborhood loops.
pub use super::scalar::ScalarType as Scale;
use super::{
    Count, Element,
    expressions::{Builder, Ty, Value},
    nvfp4::build_module,
};
use std::fmt::Write;
#[derive(Clone, Copy)]
pub struct Specialization<'a> {
    pub input: Element,
    pub key: Element,
    pub value: Element,
    pub scale: Scale,
    pub kt: i64,
    pub kh: i64,
    pub kw: i64,
    pub causal_t: bool,
    pub causal_h: bool,
    pub causal_w: bool,
    pub hd: u32,
    pub hd_pad: u32,
    pub block_q: u32,
    pub block_k: u32,
    pub scalars: [Count; 9],
    pub pointer_divisibility_16: [bool; 4],
    pub source_path: &'a str,
    pub standard_source_path: &'a str,
}
include!("triton_frontend_na_program.rs");
include!("triton_frontend_na_math.rs");
#[path = "triton_frontend_na_integers.rs"]
mod integers;
#[derive(Clone, Copy)]
pub enum KernelValue {
    Int(i64),
    UInt(u64),
    Bool(bool),
    Float(f64),
}
impl KernelValue {
    fn value(self) -> Value {
        match self {
            Self::Int(n) => Value::integer(n),
            Self::UInt(n) => Value::integer(n as i64),
            Self::Bool(b) => {
                let mut value = Value::integer(i64::from(b));
                value.ty = Ty::I1;
                value
            }
            Self::Float(n) => Value::number(n),
        }
    }
}
pub fn source_ttir(s: &Specialization<'_>) -> String {
    source_ttir_kernels(
        s,
        [
            KernelValue::Int(s.kt),
            KernelValue::Int(s.kh),
            KernelValue::Int(s.kw),
        ],
    )
}
pub fn source_ttir_kernels(s: &Specialization<'_>, kernels: [KernelValue; 3]) -> String {
    let mut arguments = vec![
        ("q_ptr", Ty::Pointer(s.input)),
        ("k_ptr", Ty::Pointer(s.key)),
        ("v_ptr", Ty::Pointer(s.value)),
        ("out_ptr", Ty::Pointer(s.input)),
    ];
    let mut aligned = s.pointer_divisibility_16.to_vec();
    let mut indices = vec![];
    for (name, c) in SCALAR_NAMES.iter().zip(s.scalars) {
        match c {
            Count::One => indices.push(None),
            Count::I32 { divisible_by_16 } | Count::I64 { divisible_by_16 } => {
                indices.push(Some(arguments.len()));
                arguments.push((
                    name,
                    if matches!(c, Count::I64 { .. }) {
                        Ty::I64
                    } else {
                        Ty::I32
                    },
                ));
                aligned.push(divisible_by_16);
            }
        }
    }
    if !matches!(s.scale, Scale::One) {
        arguments.push(("scale", s.scale.ty()));
        aligned.push(matches!(
            s.scale,
            Scale::I32 {
                divisible_by_16: true
            } | Scale::I64 {
                divisible_by_16: true
            } | Scale::U64 {
                divisible_by_16: true
            }
        ));
    }
    build_module(
        s.source_path,
        s.block_k,
        &arguments,
        &aligned,
        &[],
        &[],
        21,
        "_na3d_kernel",
        |b, p, _| {
            let scalars = indices
                .iter()
                .map(|i| i.map_or_else(|| Value::integer(1), |i| p[i].clone()))
                .collect::<Vec<_>>();
            lower_na(
                b,
                s,
                kernels.map(KernelValue::value),
                &p[..4],
                &scalars,
                if matches!(s.scale, Scale::One) {
                    Value::integer(1)
                } else {
                    p.last().unwrap().clone()
                },
            );
        },
    )
}

/// The frozen Python frontend accepts these constexpr values in the signature,
/// then rejects float loop bounds while visiting the corresponding for node.
pub fn kernel_failure(
    s: &Specialization<'_>,
    kernels: [KernelValue; 3],
) -> Option<super::errors::CompilationFailure> {
    if let Some(failure) = integers::failure(s, kernels) {
        return Some(failure);
    }
    let i = kernels
        .iter()
        .position(|v| matches!(v, KernelValue::Float(_)))?;
    let causal = [s.causal_t, s.causal_h, s.causal_w][i];
    let KernelValue::Float(value) = kernels[i] else {
        unreachable!()
    };
    let dtype = if Value::number(value).ty == Ty::F64 {
        "fp64"
    } else {
        "fp32"
    };
    Some(super::errors::CompilationFailure::at(
        FUNCTION_SOURCE,
        FUNCTION_DEFINITION,
        [(83, 4), (84, 8), (86, 12)][i],
        &format!(
            "TypeError('For loop bounds and step must all be ints, are ({dtype}, {}, int32)')",
            if causal { "int32" } else { dtype }
        ),
    ))
}
pub fn checked_source_kernels(
    s: &Specialization<'_>,
    kernels: [KernelValue; 3],
) -> Result<String, super::errors::CompilationFailure> {
    if let Some(failure) = kernel_failure(s, kernels) {
        return Err(failure);
    }
    checked_source(s)?;
    Ok(source_ttir_kernels(s, kernels))
}

/// Semantic type checks before IR generation, evaluated only on a source miss.
pub fn checked_source(s: &Specialization<'_>) -> Result<String, super::errors::CompilationFailure> {
    if let Some(failure) = integers::failure(
        s,
        [
            KernelValue::Int(s.kt),
            KernelValue::Int(s.kh),
            KernelValue::Int(s.kw),
        ],
    ) {
        return Err(failure);
    }
    // The score's multiplication with scale promotes FP32 to FP64 only when
    // the explicitly requested AST signature is FP64. Python float JIT values
    // always specialize as FP32, including large/non-finite values.
    let signature = |e: Element| match e {
        Element::F64 => "fp64",
        Element::F32 => "fp32",
        Element::F16 => "fp16",
        Element::BF16 => "bf16",
        Element::E4M3 => "fp8e4nv",
        Element::E5M2 => "fp8e5",
    };
    let fp64 = matches!(s.scale, Scale::F64);
    let failure = if s.input != s.key {
        Some((
            if s.input == Element::F32 {
                QUERY_DOT_IEEE_POSITION
            } else {
                QUERY_DOT_DEFAULT_POSITION
            },
            format!(
                "Both operands must be same dtype. Got {} and {}",
                signature(s.input),
                signature(s.key)
            ),
        ))
    } else if s.input == Element::F32 && (fp64 || s.value != Element::F32) {
        Some((
            ACCUMULATOR_DOT_POSITION,
            format!(
                "Both operands must be same dtype. Got {} and {}",
                if fp64 { "fp64" } else { "fp32" },
                signature(s.value)
            ),
        ))
    } else if fp64 {
        let message = format!(
            "Loop-carried variable m_i has initial type <['{}'], fp32> but is re-assigned to <['{}'], fp64> in loop! Please make sure that the type stays consistent.",
            s.block_q, s.block_q
        );
        Some((
            INNER_LOOP_POSITION,
            format!(
                "AssertionError({})",
                serde_json::to_string(&message).unwrap()
            ),
        ))
    } else {
        None
    };
    if let Some((at, cause)) = failure {
        return Err(super::errors::CompilationFailure::at(
            FUNCTION_SOURCE,
            FUNCTION_DEFINITION,
            at,
            &cause,
        ));
    }
    Ok(source_ttir(s))
}
