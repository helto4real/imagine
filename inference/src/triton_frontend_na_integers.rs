//! Frozen scalar-to-Tensor range checks in source window-evaluation order.
use super::{KernelValue, Specialization};
use crate::triton_frontend::{Count, errors::CompilationFailure};
#[derive(Clone, Copy)]
enum Integer {
    I32,
    U32,
    I64,
    U64,
}
#[derive(Clone, Copy)]
enum Operand {
    Constant(i128),
    Tensor(Integer),
}
fn promote(a: Integer, b: Integer) -> Integer {
    use Integer::*;
    match (a, b) {
        (U64, _) | (_, U64) => U64,
        (I64, _) | (_, I64) => I64,
        (U32, _) | (_, U32) => U32,
        _ => I32,
    }
}
fn materialize(v: Operand) -> Result<Integer, String> {
    Ok(match v {
        Operand::Tensor(t) => t,
        Operand::Constant(n) if i32::try_from(n).is_ok() => Integer::I32,
        Operand::Constant(n) if u32::try_from(n).is_ok() => Integer::U32,
        Operand::Constant(n) if i64::try_from(n).is_ok() => Integer::I64,
        Operand::Constant(n) if u64::try_from(n).is_ok() => Integer::U64,
        Operand::Constant(n) => return Err(format!("ValueError('Nonrepresentable integer {n}.')")),
    })
}
fn binary(a: Operand, b: Operand, subtract: bool) -> Result<Operand, String> {
    use Operand::*;
    if let (Constant(a), Constant(b)) = (a, b) {
        return Ok(Constant(if subtract { a - b } else { a + b }));
    }
    let t = match (a, b) {
        (Tensor(a), Tensor(b)) => promote(a, b),
        (Tensor(t), _) | (_, Tensor(t)) => t,
        _ => unreachable!(),
    };
    for scalar in [a, b] {
        if let Constant(n) = scalar {
            let (lo, hi, name) = match t {
                Integer::I32 => (i128::from(i32::MIN), i128::from(i32::MAX), "int32"),
                Integer::U32 => (0, i128::from(u32::MAX), "uint32"),
                Integer::I64 => (i128::from(i64::MIN), i128::from(i64::MAX), "int64"),
                Integer::U64 => (0, i128::from(u64::MAX), "uint64"),
            };
            if lo == 0 && n < 0 {
                return Err("ValueError('Cannot perform a binary operation between an unsigned tensor and a negative scalar. Perform a explicit cast on one of them.')".into());
            }
            if n < lo || n > hi {
                return Err(format!(
                    "ValueError('Scalar {n} is out of range for type {name}')"
                ));
            }
        }
    }
    Ok(Tensor(t))
}
fn extreme(a: Operand, b: Operand) -> Result<Operand, String> {
    Ok(Operand::Tensor(promote(materialize(a)?, materialize(b)?)))
}
fn size(c: Count) -> Operand {
    match c {
        Count::One => Operand::Constant(1),
        Count::I32 { .. } => Operand::Tensor(Integer::I32),
        Count::I64 { .. } => Operand::Tensor(Integer::I64),
    }
}
pub(super) fn failure(
    s: &Specialization<'_>,
    kernels: [KernelValue; 3],
) -> Option<CompilationFailure> {
    use Operand::{Constant, Tensor};
    for (axis, kernel) in kernels.into_iter().enumerate() {
        let n = match kernel {
            KernelValue::Int(n) => i128::from(n),
            KernelValue::UInt(n) => i128::from(n),
            KernelValue::Bool(n) => i128::from(n),
            KernelValue::Float(_) => continue,
        };
        // pid_th / h_size and pid_th % h_size have the same promoted type.
        // The W query is tl.where(w_valid, w_off, w_size - 1).
        let query = Tensor(
            if matches!(s.scalars[if axis == 2 { 2 } else { 1 }], Count::I64 { .. }) {
                Integer::I64
            } else {
                Integer::I32
            },
        );
        let causal = [s.causal_t, s.causal_h, s.causal_w][axis];
        let mut position = (0, 0);
        let mut run = || -> Result<(), String> {
            if causal {
                position = [(50, 26), (56, 26), (65, 29)][axis];
                binary(query, Constant(n), true)?;
            } else {
                position = [(53, 37), (59, 37), (72, 40)][axis];
                let low = binary(query, Constant(n.div_euclid(2)), true)?;
                let low = extreme(low, Constant(0))?;
                position = [(53, 56), (59, 56), (72, 59)][axis];
                let bound = binary(size(s.scalars[axis]), Constant(n), true)?;
                position = [(53, 15), (59, 15), (72, 18)][axis];
                let low = extreme(low, bound)?;
                position = [(54, 15), (60, 15), (73, 16)][axis];
                binary(low, Constant(n), false)?;
            }
            Ok(())
        };
        if let Err(cause) = run() {
            return Some(CompilationFailure::at(
                super::FUNCTION_SOURCE,
                super::FUNCTION_DEFINITION,
                position,
                &cause,
            ));
        }
    }
    None
}
